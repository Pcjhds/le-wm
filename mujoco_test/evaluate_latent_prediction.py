"""Evaluate a pretrained LeWM one-step latent prediction on MuJoCo rollouts.

This is a smoke test for latent prediction compatibility only. It does not
train, control MuJoCo, or compare image-space predictions.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from load_pretrained_model import (
    build_model,
    import_or_report,
    resolve_model_dir,
    unwrap_state_dict,
)
from prepare_model_inputs import (
    build_window_indices,
    load_rollout,
    make_action_windows,
    make_image_windows,
    make_target_images,
    validate_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test LeWM latent prediction on adapted MuJoCo rollouts."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("trained_model"),
        help="Directory containing config.json and weights.pt.",
    )
    parser.add_argument(
        "--rollout",
        type=Path,
        default=Path("mujoco_test/rollouts/simple_push_rollouts.npz"),
        help="Input rollout .npz file.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for model and input batch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prepared windows to evaluate.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=3,
        help="Number of consecutive history steps per window.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Output square image size.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=25,
        help="Output action dimension expected by the pretrained model.",
    )
    return parser.parse_args()


def load_pretrained_model(model_dir: Path, device: str):
    torch = import_or_report("torch")
    model_dir = resolve_model_dir(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "weights.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    config = json.loads(config_path.read_text())
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict, checkpoint_format = unwrap_state_dict(checkpoint)

    model = build_model(config, device, torch)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, torch, model_dir, checkpoint_format, len(state_dict)


def prepare_tensors(args: argparse.Namespace, torch: Any):
    rollout = load_rollout(args.rollout)
    validate_rollout(rollout)
    windows = build_window_indices(
        rollout["episode_index"],
        rollout["step_index"],
        args.history_size,
    )
    if windows.shape[0] == 0:
        raise ValueError("no valid consecutive windows found in rollout")

    pixels = make_image_windows(rollout["image_t"], windows, args.image_size)
    actions = make_action_windows(rollout["action_t"], windows, args.action_dim)
    target_pixels = make_target_images(
        rollout["image_t_plus_1"],
        windows,
        args.image_size,
    )

    batch_size = min(args.batch_size, windows.shape[0])
    pixels_tensor = torch.from_numpy(pixels[:batch_size]).to(args.device)
    actions_tensor = torch.from_numpy(actions[:batch_size]).to(args.device)
    target_tensor = torch.from_numpy(target_pixels[:batch_size]).to(args.device)
    return pixels_tensor, actions_tensor, target_tensor, windows, batch_size


def safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def print_predictor_debug(model: Any, emb: Any | None, act_emb: Any | None) -> None:
    public_attrs = [name for name in dir(model) if not name.startswith("_")]
    predictor = getattr(model, "predictor", None)
    predictor_forward = getattr(predictor, "forward", None)

    print(f"available model attributes: {public_attrs[:80]}")
    print(f"predictor type: {type(predictor).__name__}")
    print(f"model.predict signature: {safe_signature(getattr(model, 'predict', None))}")
    print(f"predictor.forward signature: {safe_signature(predictor_forward)}")
    if emb is not None:
        print(f"emb shape: {tuple(emb.shape)}")
    if act_emb is not None:
        print(f"act_emb shape: {tuple(act_emb.shape)}")


def compute_metrics(torch: Any, predicted: Any, target: Any) -> tuple[float, float]:
    mse = torch.nn.functional.mse_loss(predicted, target).item()
    cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean().item()
    return mse, cosine


def main() -> int:
    args = parse_args()

    try:
        model, torch, model_dir, checkpoint_format, num_keys = load_pretrained_model(
            args.model_dir,
            args.device,
        )
        pixels, actions, target_pixels, windows, batch_size = prepare_tensors(args, torch)

        print(f"model directory: {model_dir}")
        print(f"checkpoint format: {checkpoint_format}")
        print(f"checkpoint keys: {num_keys}")
        print(f"rollout: {args.rollout}")
        print(f"valid windows: {windows.shape[0]}")
        print(f"batch size: {batch_size}")
        print(f"pixels batch shape: {tuple(pixels.shape)}")
        print(f"action batch shape: {tuple(actions.shape)}")
        print(f"target image batch shape: {tuple(target_pixels.shape)}")
        sys.stdout.flush()

        with torch.no_grad():
            encoded = model.encode({"pixels": pixels, "action": actions})
            context_emb = encoded["emb"]
            action_emb = encoded["act_emb"]

            if not hasattr(model, "predict"):
                print("predictor interface unclear: model has no predict method")
                print_predictor_debug(model, context_emb, action_emb)
                return 1

            # Local jepa.py defines predict(emb, act_emb) and training compares
            # the returned sequence against next-image latents. With a T-step
            # context window and image_t_plus_1 from the final step as target,
            # the final predicted element is the one-step-ahead latent.
            predicted_sequence = model.predict(context_emb, action_emb)
            predicted_next = predicted_sequence[:, -1]

            if not hasattr(model, "encode"):
                print("target encoder unavailable: model has no encode method")
                print_predictor_debug(model, context_emb, action_emb)
                return 1

            target_encoded = model.encode({"pixels": target_pixels.unsqueeze(1)})
            target_next = target_encoded["emb"][:, 0]

            mse, cosine = compute_metrics(torch, predicted_next, target_next)

        print(f"context emb shape: {tuple(context_emb.shape)}")
        print(f"action emb shape: {tuple(action_emb.shape)}")
        print(f"predicted latent shape: {tuple(predicted_next.shape)}")
        print(f"target latent shape: {tuple(target_next.shape)}")
        print(f"latent prediction MSE: {mse:.8f}")
        print(f"latent cosine similarity mean: {cosine:.8f}")
        print(
            "note: this is a tensor-compatibility smoke test; MuJoCo actions are "
            "zero-padded placeholders for the pretrained 25D action interface"
        )
        print("latent prediction smoke test passed: yes")

    except Exception as exc:
        print("latent prediction smoke test passed: no", file=sys.stderr)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            print_predictor_debug(locals().get("model"), locals().get("context_emb"), locals().get("action_emb"))
        except Exception:
            pass
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
