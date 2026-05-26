"""Evaluate multistep LeWM latent prediction on MuJoCo rollouts.

This script increases the temporal gap between the context window and target
image so copy-last baselines are less trivial. It does not train or control
MuJoCo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_latent_prediction import compute_metrics, load_pretrained_model
from prepare_model_inputs import (
    load_rollout,
    make_image_windows,
    resize_to_chw_float,
    validate_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multistep latent predictions against simple baselines."
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
        help="Number of consecutive context steps per window.",
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
    parser.add_argument(
        "--target-offset",
        type=int,
        default=5,
        help="Target frame offset after the final context step.",
    )
    return parser.parse_args()


def build_multistep_indices(
    episode_index: np.ndarray,
    step_index: np.ndarray,
    history_size: int,
    target_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    if history_size <= 0:
        raise ValueError("history_size must be positive")
    if target_offset <= 0:
        raise ValueError("target_offset must be positive")

    n_steps = episode_index.shape[0]
    context_windows: list[np.ndarray] = []
    target_indices: list[int] = []
    span = history_size + target_offset

    for start in range(0, n_steps - span + 1):
        context = np.arange(start, start + history_size)
        final_context = context[-1]
        target_idx = final_context + target_offset
        full_span = np.arange(start, target_idx + 1)

        episodes = episode_index[full_span]
        steps = step_index[full_span]
        same_episode = np.all(episodes == episodes[0])
        consecutive_steps = np.all(np.diff(steps) == 1)
        if same_episode and consecutive_steps:
            context_windows.append(context)
            target_indices.append(target_idx)

    if not context_windows:
        return (
            np.empty((0, history_size), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    return np.stack(context_windows).astype(np.int64), np.asarray(target_indices, dtype=np.int64)


def make_action_sequences(
    actions: np.ndarray,
    windows: np.ndarray,
    target_indices: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    if action_dim < 2:
        raise ValueError("action_dim must be at least 2 for MuJoCo x/y actions")

    n_windows, history_size = windows.shape
    sequence_len = int(target_indices[0] - windows[0, 0])
    output = np.zeros((n_windows, sequence_len, action_dim), dtype=np.float32)

    for out_idx, (window, target_idx) in enumerate(zip(windows, target_indices)):
        start = window[0]
        action_slice = actions[start:target_idx, :2].astype(np.float32)
        if action_slice.shape[0] != sequence_len:
            raise ValueError("inconsistent multistep action sequence length")
        output[out_idx, :, :2] = action_slice

    if sequence_len < history_size:
        raise ValueError("action sequence shorter than history window")
    return output


def make_multistep_targets(
    images: np.ndarray,
    target_indices: np.ndarray,
    image_size: int,
) -> np.ndarray:
    output = np.empty((target_indices.shape[0], 3, image_size, image_size), dtype=np.float32)
    for out_idx, target_idx in enumerate(target_indices):
        output[out_idx] = resize_to_chw_float(images[target_idx], image_size)
    return output


def prepare_tensors(args: argparse.Namespace, torch: Any):
    rollout = load_rollout(args.rollout)
    validate_rollout(rollout)

    windows, target_indices = build_multistep_indices(
        rollout["episode_index"],
        rollout["step_index"],
        args.history_size,
        args.target_offset,
    )
    if windows.shape[0] == 0:
        print(
            "no valid multistep windows found; collect longer rollouts or use a "
            "smaller --target-offset"
        )
        print(
            "example: python mujoco_test/collect_rollouts.py --episodes 3 --steps 50"
        )
        return None

    pixels = make_image_windows(rollout["image_t"], windows, args.image_size)
    actions = make_action_sequences(
        rollout["action_t"],
        windows,
        target_indices,
        args.action_dim,
    )
    target_pixels = make_multistep_targets(
        rollout["image_t"],
        target_indices,
        args.image_size,
    )

    batch_size = min(args.batch_size, windows.shape[0])
    return (
        torch.from_numpy(pixels[:batch_size]).to(args.device),
        torch.from_numpy(actions[:batch_size]).to(args.device),
        torch.from_numpy(target_pixels[:batch_size]).to(args.device),
        windows,
        target_indices,
        batch_size,
    )


def rollout_latent(
    model: Any,
    torch: Any,
    raw_actions: Any,
    context_emb: Any,
    history_size: int,
    target_offset: int,
) -> Any:
    emb_seq = context_emb
    act_seq = raw_actions[:, :history_size]

    for step in range(target_offset):
        act_emb = model.action_encoder(act_seq)
        pred_next = model.predict(
            emb_seq[:, -history_size:],
            act_emb[:, -history_size:],
        )[:, -1]
        if step == target_offset - 1:
            return pred_next

        emb_seq = torch.cat([emb_seq, pred_next.unsqueeze(1)], dim=1)
        next_action_idx = history_size + step
        next_action = raw_actions[:, next_action_idx : next_action_idx + 1]
        act_seq = torch.cat([act_seq, next_action], dim=1)

    raise RuntimeError("target_offset loop did not produce a prediction")


def format_metrics(name: str, metrics: tuple[float, float]) -> str:
    mse, cosine = metrics
    return f"{name:<34} mse={mse:.8f}  cosine_mean={cosine:.8f}"


def evaluate_methods(
    model: Any,
    torch: Any,
    pixels: Any,
    actions: Any,
    target_pixels: Any,
    history_size: int,
    target_offset: int,
):
    with torch.no_grad():
        context_actions = actions[:, :history_size]
        encoded = model.encode({"pixels": pixels, "action": context_actions})
        context_emb = encoded["emb"]

        target_encoded = model.encode({"pixels": target_pixels.unsqueeze(1)})
        target_next = target_encoded["emb"][:, 0]

        model_pred = rollout_latent(
            model,
            torch,
            actions,
            context_emb,
            history_size,
            target_offset,
        )

        copy_last = context_emb[:, -1]

        zero_actions = torch.zeros_like(actions)
        zero_action_pred = rollout_latent(
            model,
            torch,
            zero_actions,
            context_emb,
            history_size,
            target_offset,
        )

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
        prepared = prepare_tensors(args, torch)
        if prepared is None:
            return 0

        pixels, actions, target_pixels, windows, target_indices, batch_size = prepared
        metrics, context_emb, target_next = evaluate_methods(
            model,
            torch,
            pixels,
            actions,
            target_pixels,
            args.history_size,
            args.target_offset,
        )

        print(f"model directory: {model_dir}")
        print(f"checkpoint format: {checkpoint_format}")
        print(f"checkpoint keys: {num_keys}")
        print(f"rollout: {args.rollout}")
        print(f"number of valid windows: {windows.shape[0]}")
        print(f"target offset: {args.target_offset}")
        print(f"batch size: {batch_size}")
        print(f"pixels batch shape: {tuple(pixels.shape)}")
        print(f"action sequence shape: {tuple(actions.shape)}")
        print(f"context emb shape: {tuple(context_emb.shape)}")
        print(f"target latent shape: {tuple(target_next.shape)}")
        print("multistep latent comparison:")
        for name, method_metrics in metrics.items():
            print(format_metrics(name, method_metrics))
        print(
            "note: MuJoCo actions are zero-padded placeholders for the pretrained "
            "25D action interface, so these metrics are a compatibility baseline, "
            "not a task-performance claim"
        )
        print("multistep latent evaluation passed: yes")

    except Exception as exc:
        print("multistep latent evaluation passed: no")
        print(f"error: {type(exc).__name__}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
