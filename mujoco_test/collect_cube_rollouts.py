"""Collect CubePushEnv rollouts with 25D actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cube_push_env import CubePushEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect CubePushEnv rollouts as compressed NumPy arrays."
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to collect.")
    parser.add_argument("--steps", type=int, default=200, help="Maximum transitions per episode.")
    parser.add_argument(
        "--policy",
        choices=["random_25d", "goal_push_25d"],
        default="goal_push_25d",
        help="Action policy used to collect rollouts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mujoco_test/rollouts/cube_push_rollouts.npz"),
        help="Output .npz path.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.035,
        help="Per-step pusher movement scale inside CubePushEnv.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=224,
        help="Square RGB render size written to the rollout.",
    )
    return parser.parse_args()


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def state_from_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [obs["pusher_pos"], obs["cube_pos"], obs["goal_pos"]],
    ).astype(np.float32)


def random_25d_action(env: CubePushEnv, rng: np.random.Generator) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[:2] = rng.uniform(-1.0, 1.0, size=2)
    action[2:4] = rng.uniform(-0.25, 0.25, size=2)
    return action


def goal_push_25d_action(
    env: CubePushEnv,
    obs: dict[str, np.ndarray],
    rng: np.random.Generator,
    policy_state: dict[str, np.ndarray | int],
) -> np.ndarray:
    remaining = int(policy_state.get("remaining", 0))
    if remaining > 0:
        policy_state["remaining"] = remaining - 1
        return np.asarray(policy_state["action"], dtype=np.float32)

    cube_pos = obs["cube_pos"]
    pusher_pos = obs["pusher_pos"]
    goal_pos = obs["goal_pos"]

    push_dir = normalized(goal_pos - cube_pos)
    if np.linalg.norm(push_dir) < 1e-6:
        push_dir = np.array([1.0, 0.0], dtype=np.float32)

    behind_cube = cube_pos - 0.12 * push_dir
    pusher_to_behind = behind_cube - pusher_pos
    pusher_to_cube = cube_pos - pusher_pos

    if np.linalg.norm(pusher_to_behind) > 0.045:
        desired = pusher_to_behind
    elif np.linalg.norm(pusher_to_cube) > 0.035:
        desired = pusher_to_cube
    else:
        desired = push_dir + 0.2 * normalized(goal_pos - pusher_pos)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[:2] = normalized(desired)
    action[:2] += rng.normal(0.0, 0.10, size=2).astype(np.float32)
    action[:2] = np.clip(normalized(action[:2]) * rng.uniform(0.85, 1.1), -1.0, 1.0)
    policy_state["action"] = action
    policy_state["remaining"] = int(rng.integers(3, 9))
    return action


def choose_action(
    env: CubePushEnv,
    obs: dict[str, np.ndarray],
    rng: np.random.Generator,
    policy: str,
    policy_state: dict[str, np.ndarray | int],
) -> np.ndarray:
    if policy == "random_25d":
        return random_25d_action(env, rng)
    if policy == "goal_push_25d":
        return goal_push_25d_action(env, obs, rng, policy_state)
    raise ValueError(f"Unsupported policy: {policy}")


def collect_rollouts(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    env = CubePushEnv(
        render_mode="rgb_array",
        image_size=(args.render_size, args.render_size),
        max_episode_steps=args.steps,
        action_scale=args.action_scale,
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


def print_diagnostics(arrays: dict[str, np.ndarray]) -> None:
    episode_ids = np.unique(arrays["episode_index"])
    cube_displacements = []
    cube_max_displacements = []
    distance_improvements = []

    for ep_idx in episode_ids:
        mask = arrays["episode_index"] == ep_idx
        states = arrays["state_t"][mask]
        if states.size == 0:
            continue

        cube_pos = states[:, 2:4]
        goal_pos = states[:, 4:6]
        first_cube = cube_pos[0]
        last_cube = cube_pos[-1]
        displacement = float(np.linalg.norm(last_cube - first_cube))
        max_displacement = float(np.linalg.norm(cube_pos - first_cube, axis=1).max())
        initial_distance = float(np.linalg.norm(first_cube - goal_pos[0]))
        final_distance = float(np.linalg.norm(last_cube - goal_pos[-1]))
        improvement = initial_distance - final_distance

        cube_displacements.append(displacement)
        cube_max_displacements.append(max_displacement)
        distance_improvements.append(improvement)

        print(
            f"episode {int(ep_idx)} cube first={first_cube} last={last_cube} "
            f"goal={goal_pos[-1]} displacement={displacement:.4f} "
            f"distance_improvement={improvement:.4f}"
        )

    if not cube_displacements:
        print("diagnostics: no episodes found")
        return

    print(f"mean cube displacement: {np.mean(cube_displacements):.4f}")
    print(f"max cube displacement: {np.max(cube_max_displacements):.4f}")
    print(f"mean cube-goal distance improvement: {np.mean(distance_improvements):.4f}")
    print(f"action_t shape: {arrays['action_t'].shape}")


def main() -> None:
    args = parse_args()
    arrays = collect_rollouts(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    print(f"episodes: {args.episodes}")
    print(f"policy: {args.policy}")
    print(f"action_scale: {args.action_scale}")
    print(f"render_size: {args.render_size}")
    print(f"transitions: {arrays['action_t'].shape[0]}")
    print(f"output: {args.out}")
    print(f"image_t shape: {arrays['image_t'].shape}")
    print(f"image_t_plus_1 shape: {arrays['image_t_plus_1'].shape}")
    print_diagnostics(arrays)


if __name__ == "__main__":
    main()
