"""Run a pretrained LeWM encode smoke test on adapted MuJoCo rollout tensors.

This script verifies tensor compatibility only. It does not train, control
MuJoCo, compute losses, or compare predictions.
"""

from __future__ import annotations

import argparse
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
    validate_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test pretrained LeWM encoding on MuJoCo rollout tensors."
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
        help="Number of prepared windows to encode.",
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
    pixels = make_image_windows(rollout["image_t"], windows, args.image_size)
    actions = make_action_windows(rollout["action_t"], windows, args.action_dim)

    if windows.shape[0] == 0:
        raise ValueError("no valid consecutive windows found in rollout")

    batch_size = min(args.batch_size, windows.shape[0])
    pixels_tensor = torch.from_numpy(pixels[:batch_size]).to(args.device)
    actions_tensor = torch.from_numpy(actions[:batch_size]).to(args.device)
    return pixels_tensor, actions_tensor, windows, batch_size


def tensor_min_max(tensor: Any) -> str:
    if tensor.numel() == 0:
        return "n/a"
    if not (tensor.is_floating_point() or tensor.is_complex()):
        values = tensor.detach()
    elif tensor.is_complex():
        return "n/a (complex tensor)"
    else:
        values = tensor.detach()

    values = values.float().cpu()
    return f"{values.min().item():.6f} / {values.max().item():.6f}"


def print_output_summary(output: Any, torch: Any) -> None:
    if not isinstance(output, dict):
        print(f"output type: {type(output).__name__}")
        return

    print(f"output keys: {list(output.keys())}")
    for key, value in output.items():
        if torch.is_tensor(value):
            print(
                f"{key}: shape={tuple(value.shape)} "
                f"dtype={value.dtype} min/max={tensor_min_max(value)}"
            )
        else:
            print(f"{key}: type={type(value).__name__}")


def main() -> int:
    args = parse_args()

    try:
        model, torch, model_dir, checkpoint_format, num_keys = load_pretrained_model(
            args.model_dir,
            args.device,
        )
        pixels, actions, windows, batch_size = prepare_tensors(args, torch)

        print(f"model directory: {model_dir}")
        print(f"checkpoint format: {checkpoint_format}")
        print(f"checkpoint keys: {num_keys}")
        print(f"rollout: {args.rollout}")
        print(f"valid windows: {windows.shape[0]}")
        print(f"batch size: {batch_size}")
        print(f"pixels batch shape: {tuple(pixels.shape)}")
        print(f"action batch shape: {tuple(actions.shape)}")
        print("attempted call: model.encode({'pixels': pixels, 'action': actions})")
        sys.stdout.flush()

        with torch.no_grad():
            output = model.encode({"pixels": pixels, "action": actions})

        print_output_summary(output, torch)
        print("encode smoke test passed: yes")
    except Exception as exc:
        print("encode smoke test passed: no", file=sys.stderr)
        print(
            "attempted call: model.encode({'pixels': pixels, 'action': actions})",
            file=sys.stderr,
        )
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "suggested expected keys from jepa.py: 'pixels' with shape "
            "(B, T, C, H, W) and optional 'action' with shape (B, T, action_dim)",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
