"""MuJoCo cube-pushing environment with a 25D action interface."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


CUBE_PUSH_XML = """
<mujoco model="cube_push">
  <compiler angle="degree"/>
  <option timestep="0.005" gravity="0 0 -9.81" iterations="60"/>

  <default>
    <geom condim="3" friction="1.2 0.01 0.0001" solref="0.015 1" solimp="0.9 0.95 0.001"/>
  </default>

  <asset>
    <texture name="table_grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.74 0.75 0.73" rgb2="0.92 0.92 0.89"/>
    <material name="table_mat" texture="table_grid" texrepeat="6 5" reflectance="0.12"/>
  </asset>

  <worldbody>
    <light name="key_light" pos="-0.2 -0.4 2.0" dir="0.1 0.2 -1" directional="true"/>
    <light name="fill_light" pos="0.5 0.4 1.2" dir="-0.4 -0.3 -1"/>
    <geom name="table" type="plane" size="0.65 0.48 0.02" material="table_mat"/>

    <site name="goal" type="cylinder" pos="0.32 0.0 0.006" size="0.08 0.006"
          rgba="0.1 0.7 0.25 0.45"/>

    <body name="pusher_mocap" mocap="true" pos="-0.25 0 0.055">
      <geom name="pusher_geom" type="cylinder" size="0.04 0.055"
            rgba="0.95 0.22 0.16 1"/>
    </body>

    <body name="cube" pos="0 0 0.055">
      <joint name="cube_x" type="slide" axis="1 0 0" damping="0.9"/>
      <joint name="cube_y" type="slide" axis="0 1 0" damping="0.9"/>
      <joint name="cube_yaw" type="hinge" axis="0 0 1" damping="0.08"/>
      <geom name="cube_geom" type="box" size="0.055 0.055 0.055"
            mass="0.25" rgba="0.12 0.32 0.95 1"/>
    </body>

    <camera name="top" pos="0 0 1.15" quat="1 0 0 0" fovy="44"/>
    <camera name="angled" pos="0.0 -0.55 0.78" euler="52 0 0" fovy="46"/>
  </worldbody>
</mujoco>
"""


class CubePushEnv(gym.Env):
    """Planar cube pushing task with MuJoCo physics and 25D actions."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        image_size: tuple[int, int] = (224, 224),
        frame_skip: int = 10,
        max_episode_steps: int = 200,
        action_scale: float = 0.035,
        camera: str = "angled",
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode={render_mode!r}")

        self.render_mode = render_mode
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.camera = camera

        self.model = mujoco.MjModel.from_xml_string(CUBE_PUSH_XML)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(25,), dtype=np.float32)
        obs_spaces: dict[str, spaces.Space[Any]] = {
            "cube_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "pusher_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "goal_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        }
        if self.render_mode == "rgb_array":
            height, width = self.image_size
            obs_spaces["image"] = spaces.Box(
                low=0, high=255, shape=(height, width, 3), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_spaces)

        self.workspace_low = np.array([-0.45, -0.30], dtype=np.float32)
        self.workspace_high = np.array([0.45, 0.30], dtype=np.float32)
        self.cube_spawn_low = np.array([-0.10, -0.14], dtype=np.float32)
        self.cube_spawn_high = np.array([0.12, 0.14], dtype=np.float32)
        self.goal_spawn_low = np.array([0.22, -0.22], dtype=np.float32)
        self.goal_spawn_high = np.array([0.42, 0.22], dtype=np.float32)
        self.pusher_z = 0.055
        self._pusher_pos = np.array([-0.28, 0.0, self.pusher_z], dtype=np.float64)
        self.goal_pos = np.array([0.32, 0.0], dtype=np.float32)
        self._step_count = 0

        self._renderer: mujoco.Renderer | None = None
        self._renderer_failed = False
        self._viewer = None
        self._viewer_failed = False

        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self._cube_x_adr = self._joint_qpos_addr("cube_x")
        self._cube_y_adr = self._joint_qpos_addr("cube_y")
        self._cube_yaw_adr = self._joint_qpos_addr("cube_yaw")

    def _joint_qpos_addr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[joint_id])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options

        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        cube_xy = self.np_random.uniform(self.cube_spawn_low, self.cube_spawn_high)
        goal_xy = self.np_random.uniform(self.goal_spawn_low, self.goal_spawn_high)
        approach_offset = np.array([-0.16, 0.0], dtype=np.float64)
        pusher_xy = cube_xy + approach_offset + self.np_random.uniform(
            [-0.04, -0.08],
            [0.02, 0.08],
        )
        pusher_xy = np.clip(pusher_xy, self.workspace_low, self.workspace_high)

        self.goal_pos = goal_xy.astype(np.float32)
        self.model.site_pos[self._goal_site_id, :2] = self.goal_pos

        self.data.qpos[self._cube_x_adr] = cube_xy[0]
        self.data.qpos[self._cube_y_adr] = cube_xy[1]
        self.data.qpos[self._cube_yaw_adr] = self.np_random.uniform(-0.4, 0.4)
        self.data.qvel[:] = 0.0

        self._pusher_pos[:] = [pusher_xy[0], pusher_xy[1], self.pusher_z]
        self.data.mocap_pos[0] = self._pusher_pos
        self.data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}, got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        xy_command = action[:2] + 0.25 * action[2:4]
        xy_command = np.clip(xy_command, -1.0, 1.0)
        self._pusher_pos[:2] = np.clip(
            self._pusher_pos[:2] + xy_command * self.action_scale,
            self.workspace_low,
            self.workspace_high,
        )

        for _ in range(self.frame_skip):
            self.data.mocap_pos[0] = self._pusher_pos
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        info = self._get_info()
        reward = -info["cube_goal_distance"]
        terminated = info["cube_goal_distance"] < 0.045
        truncated = self._step_count >= self.max_episode_steps

        if self.render_mode == "human":
            self._sync_viewer()

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray:
        height, width = self.image_size
        if self._renderer is None and not self._renderer_failed:
            try:
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            except Exception as exc:
                self._renderer_failed = True
                print(
                    f"MuJoCo RGB renderer unavailable ({exc}). "
                    "Using cube top-down fallback renderer."
                )

        if self._renderer is not None:
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render()

        return self._fallback_render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {
            "cube_pos": self.data.xpos[self._cube_body_id, :2].astype(np.float32).copy(),
            "pusher_pos": self._pusher_pos[:2].astype(np.float32).copy(),
            "goal_pos": self.goal_pos.astype(np.float32).copy(),
        }
        if self.render_mode == "rgb_array":
            obs["image"] = self.render()
        return obs

    def _get_info(self) -> dict[str, Any]:
        cube_pos = self.data.xpos[self._cube_body_id, :2]
        return {
            "cube_goal_distance": float(np.linalg.norm(cube_pos - self.goal_pos)),
            "step_count": self._step_count,
        }

    def _fallback_render(self) -> np.ndarray:
        height, width = self.image_size
        image = np.full((height, width, 3), 226, dtype=np.uint8)
        grid_step = max(12, min(height, width) // 8)
        image[::grid_step, :, :] = 202
        image[:, ::grid_step, :] = 202

        goal_px = self._world_to_pixel(self.goal_pos)
        cube_px = self._world_to_pixel(self.data.xpos[self._cube_body_id, :2])
        pusher_px = self._world_to_pixel(self._pusher_pos[:2])

        self._draw_line(image, pusher_px, cube_px, color=(245, 135, 55), thickness=max(2, width // 100))
        self._draw_line(image, cube_px, goal_px, color=(65, 155, 75), thickness=max(2, width // 110))
        self._draw_circle(image, goal_px, radius=max(9, width // 12), color=(70, 190, 90))
        self._draw_box(image, cube_px, half_size=max(8, width // 13), color=(35, 85, 230))
        self._draw_circle(image, pusher_px, radius=max(7, width // 17), color=(235, 70, 50))
        return image

    def _world_to_pixel(self, xy: np.ndarray) -> tuple[int, int]:
        x = float(np.clip(xy[0], self.workspace_low[0], self.workspace_high[0]))
        y = float(np.clip(xy[1], self.workspace_low[1], self.workspace_high[1]))
        width = self.image_size[1]
        height = self.image_size[0]
        px = int(round((x - self.workspace_low[0]) / (self.workspace_high[0] - self.workspace_low[0]) * (width - 1)))
        py = int(round((self.workspace_high[1] - y) / (self.workspace_high[1] - self.workspace_low[1]) * (height - 1)))
        return px, py

    @staticmethod
    def _draw_box(image: np.ndarray, center: tuple[int, int], half_size: int, color: tuple[int, int, int]) -> None:
        height, width = image.shape[:2]
        cx, cy = center
        x0 = max(0, cx - half_size)
        x1 = min(width, cx + half_size + 1)
        y0 = max(0, cy - half_size)
        y1 = min(height, cy + half_size + 1)
        image[y0:y1, x0:x1] = color

    @staticmethod
    def _draw_circle(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
        height, width = image.shape[:2]
        cx, cy = center
        y, x = np.ogrid[:height, :width]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        image[mask] = color

    @classmethod
    def _draw_line(
        cls,
        image: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        length = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, length + 1).round().astype(np.int64)
        ys = np.linspace(y0, y1, length + 1).round().astype(np.int64)
        for x, y in zip(xs, ys):
            cls._draw_box(image, (int(x), int(y)), thickness, color)

    def _sync_viewer(self) -> None:
        if self._viewer_failed:
            return
        if self._viewer is None:
            try:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as exc:
                self._viewer_failed = True
                print(f"Human viewer unavailable ({exc}). Continuing without it.")
                return
        self._viewer.sync()


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _goal_biased_action(env: CubePushEnv, obs: dict[str, np.ndarray]) -> np.ndarray:
    cube_pos = obs["cube_pos"]
    pusher_pos = obs["pusher_pos"]
    goal_pos = obs["goal_pos"]
    push_dir = _normalized(goal_pos - cube_pos)
    if np.linalg.norm(push_dir) < 1e-6:
        push_dir = np.array([1.0, 0.0], dtype=np.float32)

    behind_cube = cube_pos - 0.12 * push_dir
    if np.linalg.norm(pusher_pos - behind_cube) > 0.045:
        desired = behind_cube - pusher_pos
    elif np.linalg.norm(pusher_pos - cube_pos) > 0.035:
        desired = cube_pos - pusher_pos
    else:
        desired = push_dir

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[:2] = _normalized(desired)
    return action


def main() -> None:
    env = CubePushEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    print(
        f"Reset: cube={obs['cube_pos']} pusher={obs['pusher_pos']} "
        f"goal={obs['goal_pos']} image={obs['image'].shape}"
    )

    for step_idx in range(100):
        action = _goal_biased_action(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if step_idx % 20 == 0 or terminated or truncated:
            print(
                f"step={step_idx:03d} reward={reward:.3f} "
                f"cube={obs['cube_pos']} pusher={obs['pusher_pos']} "
                f"goal={obs['goal_pos']} dist={info['cube_goal_distance']:.3f}"
            )

        if terminated or truncated:
            break

    frame = env.render()
    print(f"Rendered RGB frame shape: {frame.shape}, dtype={frame.dtype}")
    env.close()


if __name__ == "__main__":
    main()
