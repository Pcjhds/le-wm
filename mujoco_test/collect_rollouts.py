"""Collect random-action SimplePushEnv rollouts for world-model evaluation.

Example:
    python mujoco_test/collect_rollouts.py --episodes 3 --steps 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from simple_push_env import SimplePushEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect random-action SimplePushEnv rollouts as compressed NumPy arrays."
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to collect.")
    parser.add_argument("--steps", type=int, default=100, help="Maximum transitions per episode.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mujoco_test/rollouts/simple_push_rollouts.npz"),
        help="Output .npz path.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def state_from_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([obs["ee_pos"], obs["object_pos"]]).astype(np.float32)


def collect_rollouts(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    env = SimplePushEnv(render_mode="rgb_array", max_episode_steps=args.steps)

    image_t: list[np.ndarray] = []
    action_t: list[np.ndarray] = []
    image_t_plus_1: list[np.ndarray] = []
    state_t: list[np.ndarray] = []
    state_t_plus_1: list[np.ndarray] = []
    reward_t: list[float] = []
    terminated_t: list[bool] = []
    truncated_t: list[bool] = []
    episode_index: list[int] = []
    step_index: list[int] = []

    try:
        for ep_idx in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep_idx)

            for step_idx in range(args.steps):
                action = rng.uniform(
                    env.action_space.low,
                    env.action_space.high,
                    size=env.action_space.shape,
                ).astype(np.float32)

                next_obs, reward, terminated, truncated, _ = env.step(action)

                image_t.append(obs["image"])
                action_t.append(action)
                image_t_plus_1.append(next_obs["image"])
                state_t.append(state_from_obs(obs))
                state_t_plus_1.append(state_from_obs(next_obs))
                reward_t.append(reward)
                terminated_t.append(terminated)
                truncated_t.append(truncated)
                episode_index.append(ep_idx)
                step_index.append(step_idx)

                obs = next_obs
                if terminated or truncated:
                    break
    finally:
        env.close()

    return {
        "image_t": np.asarray(image_t, dtype=np.uint8),
        "action_t": np.asarray(action_t, dtype=np.float32),
        "image_t_plus_1": np.asarray(image_t_plus_1, dtype=np.uint8),
        "state_t": np.asarray(state_t, dtype=np.float32),
        "state_t_plus_1": np.asarray(state_t_plus_1, dtype=np.float32),
        "reward_t": np.asarray(reward_t, dtype=np.float32),
        "terminated_t": np.asarray(terminated_t, dtype=bool),
        "truncated_t": np.asarray(truncated_t, dtype=bool),
        "episode_index": np.asarray(episode_index, dtype=np.int64),
        "step_index": np.asarray(step_index, dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    arrays = collect_rollouts(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    print(f"episodes: {args.episodes}")
    print(f"transitions: {arrays['action_t'].shape[0]}")
    print(f"output: {args.out}")
    print(f"image_t shape: {arrays['image_t'].shape}")
    print(f"action_t shape: {arrays['action_t'].shape}")


if __name__ == "__main__":
    main()
