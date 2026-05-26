"""Prepare MuJoCo rollout arrays as LeWorldModel-style model inputs.

This adapter is intentionally inspection-only. It builds image/action history
windows and pads the 2D MuJoCo action into the 25D action shape expected by the
pretrained cube checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


REQUIRED_KEYS = (
    "image_t",
    "action_t",
    "image_t_plus_1",
    "state_t",
    "state_t_plus_1",
    "episode_index",
    "step_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SimplePush MuJoCo rollouts into model-ready input tensors."
    )
    parser.add_argument(
        "--rollout",
        type=Path,
        default=Path("mujoco_test/rollouts/simple_push_rollouts.npz"),
        help="Input rollout .npz file.",
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
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path. Uses .pt with torch.save, otherwise .npz.",
    )
    return parser.parse_args()


def load_rollout(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"rollout file not found: {path}")

    with np.load(path) as loaded:
        missing = [key for key in REQUIRED_KEYS if key not in loaded]
        if missing:
            raise KeyError(f"rollout missing required keys: {missing}")
        return {key: loaded[key] for key in REQUIRED_KEYS}


def validate_rollout(rollout: dict[str, np.ndarray]) -> None:
    n_steps = rollout["image_t"].shape[0]
    for key in REQUIRED_KEYS:
        if rollout[key].shape[0] != n_steps:
            raise ValueError(
                f"{key} has {rollout[key].shape[0]} rows, expected {n_steps}"
            )

    if rollout["image_t"].ndim != 4 or rollout["image_t"].shape[-1] != 3:
        raise ValueError("image_t must have shape (steps, height, width, 3)")
    if rollout["image_t_plus_1"].ndim != 4 or rollout["image_t_plus_1"].shape[-1] != 3:
        raise ValueError("image_t_plus_1 must have shape (steps, height, width, 3)")
    if rollout["action_t"].ndim != 2 or rollout["action_t"].shape[1] < 2:
        raise ValueError("action_t must have shape (steps, action_dim >= 2)")


def build_window_indices(
    episode_index: np.ndarray,
    step_index: np.ndarray,
    history_size: int,
) -> np.ndarray:
    if history_size <= 0:
        raise ValueError("history_size must be positive")

    n_steps = episode_index.shape[0]
    windows: list[np.ndarray] = []
    for start in range(0, n_steps - history_size + 1):
        window = np.arange(start, start + history_size)
        episodes = episode_index[window]
        steps = step_index[window]
        same_episode = np.all(episodes == episodes[0])
        consecutive_steps = np.all(np.diff(steps) == 1)
        if same_episode and consecutive_steps:
            windows.append(window)

    if not windows:
        return np.empty((0, history_size), dtype=np.int64)
    return np.stack(windows).astype(np.int64)


def resize_to_chw_float(image: np.ndarray, image_size: int) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = Image.fromarray(image).resize((image_size, image_size), resampling)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def make_image_windows(
    images: np.ndarray,
    windows: np.ndarray,
    image_size: int,
) -> np.ndarray:
    n_windows, history_size = windows.shape
    output = np.empty((n_windows, history_size, 3, image_size, image_size), dtype=np.float32)

    for out_idx, window in enumerate(windows):
        for t_idx, source_idx in enumerate(window):
            output[out_idx, t_idx] = resize_to_chw_float(images[source_idx], image_size)

    return output


def make_target_images(
    image_t_plus_1: np.ndarray,
    windows: np.ndarray,
    image_size: int,
) -> np.ndarray:
    n_windows = windows.shape[0]
    output = np.empty((n_windows, 3, image_size, image_size), dtype=np.float32)

    for out_idx, window in enumerate(windows):
        final_step = window[-1]
        output[out_idx] = resize_to_chw_float(image_t_plus_1[final_step], image_size)

    return output


def make_action_windows(
    actions: np.ndarray,
    windows: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    if action_dim < 2:
        raise ValueError("action_dim must be at least 2 for MuJoCo x/y actions")

    n_windows, history_size = windows.shape
    output = np.zeros((n_windows, history_size, action_dim), dtype=np.float32)
    output[..., :2] = actions[windows, :2].astype(np.float32)
    return output


def save_outputs(
    out: Path,
    pixels: np.ndarray,
    actions: np.ndarray,
    target_pixels: np.ndarray,
    windows: np.ndarray,
    rollout: dict[str, np.ndarray],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pixels": pixels,
        "action": actions,
        "target_pixels": target_pixels,
        "window_indices": windows,
        "window_episode_index": rollout["episode_index"][windows[:, 0]],
        "window_start_step_index": rollout["step_index"][windows[:, 0]],
    }

    if out.suffix in {".pt", ".pth"}:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("saving .pt outputs requires torch") from exc
        torch.save(payload, out)
    else:
        np.savez_compressed(out, **payload)


def main() -> int:
    args = parse_args()
    rollout = load_rollout(args.rollout)
    validate_rollout(rollout)

    windows = build_window_indices(
        rollout["episode_index"],
        rollout["step_index"],
        args.history_size,
    )
    pixels = make_image_windows(rollout["image_t"], windows, args.image_size)
    actions = make_action_windows(rollout["action_t"], windows, args.action_dim)
    target_pixels = make_target_images(
        rollout["image_t_plus_1"],
        windows,
        args.image_size,
    )

    print(f"rollout: {args.rollout}")
    print(f"number of windows: {windows.shape[0]}")
    print(f"image tensor shape: {pixels.shape}")
    print(f"action tensor shape: {actions.shape}")
    print(f"target image tensor shape: {target_pixels.shape}")
    if pixels.size:
        print(f"min/max image values: {pixels.min():.6f} / {pixels.max():.6f}")
    else:
        print("min/max image values: n/a (no valid windows)")
    print(
        "action adapter: copied MuJoCo x/y actions into the first 2 "
        f"dimensions of {args.action_dim}D action vectors; remaining "
        f"{args.action_dim - 2} dimensions are zero placeholders"
    )

    if args.out is not None:
        save_outputs(args.out, pixels, actions, target_pixels, windows, rollout)
        print(f"saved prepared inputs: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
