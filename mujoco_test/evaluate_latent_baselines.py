"""Compare LeWM latent prediction against simple MuJoCo rollout baselines.

This script evaluates latent-space metrics only. It does not train, control
MuJoCo, or compare image-space predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluate_latent_prediction import (
    compute_metrics,
    load_pretrained_model,
    prepare_tensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pretrained LeWM latent predictions against simple baselines."
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


def format_metrics(name: str, metrics: tuple[float, float]) -> str:
    mse, cosine = metrics
    return f"{name:<34} mse={mse:.8f}  cosine_mean={cosine:.8f}"


def evaluate_methods(model: Any, torch: Any, pixels: Any, actions: Any, target_pixels: Any):
    with torch.no_grad():
        encoded = model.encode({"pixels": pixels, "action": actions})
        context_emb = encoded["emb"]
        action_emb = encoded["act_emb"]

        target_encoded = model.encode({"pixels": target_pixels.unsqueeze(1)})
        target_next = target_encoded["emb"][:, 0]

        model_pred = model.predict(context_emb, action_emb)[:, -1]

        copy_last = context_emb[:, -1]

        zero_actions = torch.zeros_like(actions)
        zero_action_emb = model.action_encoder(zero_actions)
        zero_action_pred = model.predict(context_emb, zero_action_emb)[:, -1]

        torch.manual_seed(0)
        random_pred = torch.randn_like(target_next)

        metrics = {
            "model padded-action prediction": compute_metrics(torch, model_pred, target_next),
            "copy-last-latent baseline": compute_metrics(torch, copy_last, target_next),
            "zero-action model prediction": compute_metrics(torch, zero_action_pred, target_next),
            "random latent baseline": compute_metrics(torch, random_pred, target_next),
        }

    return metrics, context_emb, target_next


def main() -> int:
    args = parse_args()

    try:
        model, torch, model_dir, checkpoint_format, num_keys = load_pretrained_model(
            args.model_dir,
            args.device,
        )
        pixels, actions, target_pixels, windows, batch_size = prepare_tensors(args, torch)
        metrics, context_emb, target_next = evaluate_methods(
            model,
            torch,
            pixels,
            actions,
            target_pixels,
        )

        print(f"model directory: {model_dir}")
        print(f"checkpoint format: {checkpoint_format}")
        print(f"checkpoint keys: {num_keys}")
        print(f"rollout: {args.rollout}")
        print(f"valid windows: {windows.shape[0]}")
        print(f"batch size: {batch_size}")
        print(f"pixels batch shape: {tuple(pixels.shape)}")
        print(f"action batch shape: {tuple(actions.shape)}")
        print(f"context emb shape: {tuple(context_emb.shape)}")
        print(f"target latent shape: {tuple(target_next.shape)}")
        print("baseline comparison:")
        for name, method_metrics in metrics.items():
            print(format_metrics(name, method_metrics))
        print(
            "note: MuJoCo actions are zero-padded placeholders for the pretrained "
            "25D action interface, so these metrics are a compatibility baseline, "
            "not a task-performance claim"
        )
        print("latent baseline comparison passed: yes")

    except Exception as exc:
        print("latent baseline comparison passed: no")
        print(f"error: {type(exc).__name__}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
