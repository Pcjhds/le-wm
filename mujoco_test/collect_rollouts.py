"""Collect SimplePushEnv rollouts for world-model evaluation.

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
        description="Collect SimplePushEnv rollouts as compressed NumPy arrays."
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to collect.")
    parser.add_argument("--steps", type=int, default=100, help="Maximum transitions per episode.")
    parser.add_argument(
        "--policy",
        choices=["random", "random_walk", "aggressive_random"],
        default="random",
        help="Action policy used to collect rollouts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mujoco_test/rollouts/simple_push_rollouts.npz"),
        help="Output .npz path.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.04,
        help="Per-step end-effector movement scale inside SimplePushEnv.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=128,
        help="Square RGB render size written to the rollout.",
    )
    parser.add_argument("--goal-x", type=float, default=0.35, help="Target x position.")
    parser.add_argument("--goal-y", type=float, default=0.0, help="Target y position.")
    return parser.parse_args()


def state_from_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([obs["ee_pos"], obs["object_pos"]]).astype(np.float32)


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def sample_random_action(env: SimplePushEnv, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(
        env.action_space.low,
        env.action_space.high,
        size=env.action_space.shape,
    ).astype(np.float32)


def sample_random_walk_action(
    env: SimplePushEnv,
    rng: np.random.Generator,
    policy_state: dict[str, np.ndarray | int],
) -> np.ndarray:
    remaining = int(policy_state.get("remaining", 0))
    if remaining <= 0:
        action = sample_random_action(env, rng)
        policy_state["action"] = action
        policy_state["remaining"] = int(rng.integers(5, 16))
    else:
        action = np.asarray(policy_state["action"], dtype=np.float32)
        policy_state["remaining"] = remaining - 1
    return action


def sample_aggressive_action(
    env: SimplePushEnv,
    obs: dict[str, np.ndarray],
    rng: np.random.Generator,
    policy_state: dict[str, np.ndarray | int],
) -> np.ndarray:
    remaining = int(policy_state.get("remaining", 0))
    if remaining > 0:
        policy_state["remaining"] = remaining - 1
        return np.asarray(policy_state["action"], dtype=np.float32)

    object_pos = obs["object_pos"]
    ee_pos = obs["ee_pos"]
    target_pos = env.target_pos.astype(np.float32)

    push_direction = normalized(target_pos - object_pos)
    if np.linalg.norm(push_direction) < 1e-6:
        push_direction = np.array([1.0, 0.0], dtype=np.float32)

    behind_object = object_pos - 0.09 * push_direction
    ee_to_behind = behind_object - ee_pos
    ee_to_object = object_pos - ee_pos

    if np.linalg.norm(ee_to_behind) > 0.045:
        desired = ee_to_behind
    elif np.linalg.norm(ee_to_object) > 0.035:
        desired = ee_to_object
    else:
        desired = push_direction + 0.25 * normalized(target_pos - ee_pos)

    action = normalized(desired)
    action += rng.normal(0.0, 0.18, size=2).astype(np.float32)
    action = normalized(action)
    action *= rng.uniform(0.85, 1.25)
    action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)

    policy_state["action"] = action
    policy_state["remaining"] = int(rng.integers(4, 12))
    return action


def choose_action(
    env: SimplePushEnv,
    obs: dict[str, np.ndarray],
    rng: np.random.Generator,
    policy: str,
    policy_state: dict[str, np.ndarray | int],
) -> np.ndarray:
    if policy == "random":
        return sample_random_action(env, rng)
    if policy == "random_walk":
        return sample_random_walk_action(env, rng, policy_state)
    if policy == "aggressive_random":
        return sample_aggressive_action(env, obs, rng, policy_state)
    raise ValueError(f"Unsupported policy: {policy}")


def collect_rollouts(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    env = SimplePushEnv(
        render_mode="rgb_array",
        image_size=(args.render_size, args.render_size),
        max_episode_steps=args.steps,
        action_scale=args.action_scale,
        target_pos=(args.goal_x, args.goal_y),
    )

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
            policy_state: dict[str, np.ndarray | int] = {}

            for step_idx in range(args.steps):
                action = choose_action(env, obs, rng, args.policy, policy_state)

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


def print_motion_diagnostics(arrays: dict[str, np.ndarray]) -> None:
    episode_ids = np.unique(arrays["episode_index"])
    object_displacements = []
    object_max_displacements = []
    ee_displacements = []

    for ep_idx in episode_ids:
        mask = arrays["episode_index"] == ep_idx
        states = arrays["state_t"][mask]
        if states.size == 0:
            continue

        ee_pos = states[:, :2]
        object_pos = states[:, 2:4]
        first_object = object_pos[0]
        last_object = object_pos[-1]
        first_ee = ee_pos[0]
        last_ee = ee_pos[-1]

        object_displacement = float(np.linalg.norm(last_object - first_object))
        object_max_displacement = float(
            np.linalg.norm(object_pos - first_object, axis=1).max()
        )
        ee_displacement = float(np.linalg.norm(last_ee - first_ee))
        object_displacements.append(object_displacement)
        object_max_displacements.append(object_max_displacement)
        ee_displacements.append(ee_displacement)

        print(
            f"episode {int(ep_idx)} object first={first_object} "
            f"last={last_object} displacement={object_displacement:.4f} "
            f"max_displacement={object_max_displacement:.4f}"
        )

    if not object_displacements:
        print("motion diagnostics: no episodes found")
        return

    print(f"mean object displacement per episode: {np.mean(object_displacements):.4f}")
    print(f"max object displacement: {np.max(object_max_displacements):.4f}")
    print(f"mean end-effector displacement: {np.mean(ee_displacements):.4f}")


def main() -> None:
    args = parse_args()
    arrays = collect_rollouts(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    print(f"episodes: {args.episodes}")
    print(f"policy: {args.policy}")
    print(f"action_scale: {args.action_scale}")
    print(f"render_size: {args.render_size}")
    print(f"goal: ({args.goal_x}, {args.goal_y})")
    print(f"transitions: {arrays['action_t'].shape[0]}")
    print(f"output: {args.out}")
    print(f"image_t shape: {arrays['image_t'].shape}")
    print(f"action_t shape: {arrays['action_t'].shape}")
    print_motion_diagnostics(arrays)


if __name__ == "__main__":
    main()
