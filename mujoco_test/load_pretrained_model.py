"""Validate loading a downloaded LeWM pretrained checkpoint.

This script only constructs the local JEPA model and loads the checkpoint
state_dict. It does not run MuJoCo, train, or evaluate rollouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WRAPPER_KEYS = ("state_dict", "model", "module", "model_state_dict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that pretrained LeWM weights load into the local JEPA model."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("trained_model"),
        help="Directory containing config.json and weights.pt.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for loading the checkpoint and model.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict state_dict loading. Use --no-strict to inspect mismatches.",
    )
    return parser.parse_args()


def resolve_model_dir(model_dir: Path) -> Path:
    if model_dir.exists():
        return model_dir

    if model_dir == Path("trained_model"):
        fallback = Path("Trained model")
        if fallback.exists():
            print(
                "warning: trained_model/ was not found; using legacy folder "
                "'Trained model/' instead"
            )
            return fallback

    raise FileNotFoundError(f"model directory not found: {model_dir}")


def public_kwargs(config_block: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config_block.items() if not key.startswith("_")}


def import_or_report(module_name: str):
    try:
        return __import__(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or exc.name.startswith(f"{module_name}."):
            raise RuntimeError(
                f"Missing dependency: {exc.name}. Install the package that provides "
                f"{module_name!r} before running this validator."
            ) from exc
        raise


def build_model(config: Mapping[str, Any], device: str, torch: Any):
    stable_pretraining = import_or_report("stable_pretraining")

    try:
        from jepa import JEPA
        from module import ARPredictor, Embedder, MLP
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing local import dependency: {exc.name}. Run this script from the "
            "repository root with the repo dependencies installed."
        ) from exc

    encoder_cfg = public_kwargs(config["encoder"])
    predictor_cfg = public_kwargs(config["predictor"])
    action_encoder_cfg = public_kwargs(config["action_encoder"])

    encoder = stable_pretraining.backbone.utils.vit_hf(
        encoder_cfg["size"],
        patch_size=encoder_cfg["patch_size"],
        image_size=encoder_cfg["image_size"],
        pretrained=encoder_cfg.get("pretrained", False),
        use_mask_token=encoder_cfg.get("use_mask_token", False),
    )

    def make_mlp(name: str):
        mlp_cfg = public_kwargs(config[name])
        return MLP(
            input_dim=mlp_cfg["input_dim"],
            output_dim=mlp_cfg["output_dim"],
            hidden_dim=mlp_cfg["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**predictor_cfg),
        action_encoder=Embedder(**action_encoder_cfg),
        projector=make_mlp("projector"),
        pred_proj=make_mlp("pred_proj"),
    )
    return model.to(device)


def unwrap_state_dict(checkpoint: Any) -> tuple[Mapping[str, Any], str]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {type(checkpoint)!r}")

    for key in WRAPPER_KEYS:
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            return nested, f"nested checkpoint key: {key}"

    return checkpoint, "plain state_dict"


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> int:
    args = parse_args()

    try:
        model_dir = resolve_model_dir(args.model_dir)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "weights.pt"

        if not config_path.exists():
            raise FileNotFoundError(f"config file not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"weights file not found: {weights_path}")

        config = json.loads(config_path.read_text())
        torch = import_or_report("torch")
        checkpoint = torch.load(weights_path, map_location=args.device)
        state_dict, checkpoint_format = unwrap_state_dict(checkpoint)

        print(f"model directory: {model_dir}")
        print(f"config target: {config.get('_target_', '<missing>')}")
        print(f"checkpoint format: {checkpoint_format}")
        print(f"checkpoint keys: {len(state_dict)}")
        sys.stdout.flush()

        model = build_model(config, args.device, torch)
        print(f"model parameters: {count_parameters(model):,}")

        load_result = model.load_state_dict(state_dict, strict=args.strict)

        if args.strict:
            print("strict loading passed: yes")
        else:
            missing = list(load_result.missing_keys)
            unexpected = list(load_result.unexpected_keys)
            print("strict loading passed: not requested")
            print(f"missing keys count: {len(missing)}")
            if missing:
                print(f"missing keys first 20: {missing[:20]}")
            print(f"unexpected keys count: {len(unexpected)}")
            if unexpected:
                print(f"unexpected keys first 20: {unexpected[:20]}")

    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
