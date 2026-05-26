"""A tiny MuJoCo 2D pushing environment for image/action rollout tests."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PUSH_XML = """
<mujoco model="simple_push">
  <compiler angle="degree"/>
  <option timestep="0.005" gravity="0 0 -9.81" iterations="50"/>

  <default>
    <geom condim="3" friction="1.0 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.76 0.78 0.79" rgb2="0.90 0.91 0.91"/>
    <material name="table_mat" texture="grid" texrepeat="6 5" reflectance="0.15"/>
  </asset>

  <worldbody>
    <light name="top_light" pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <geom name="table" type="plane" size="0.7 0.5 0.02" material="table_mat"/>
    <site name="target" type="cylinder" pos="0.35 0 0.004" size="0.07 0.004"
          rgba="0.1 0.7 0.25 0.35"/>

    <body name="ee_mocap" mocap="true" pos="-0.25 0 0.045">
      <geom name="ee_geom" type="cylinder" size="0.035 0.045"
            rgba="0.95 0.25 0.18 1"/>
    </body>

    <body name="puck" pos="0 0 0.04">
      <joint name="puck_x" type="slide" axis="1 0 0" damping="0.7"/>
      <joint name="puck_y" type="slide" axis="0 1 0" damping="0.7"/>
      <joint name="puck_yaw" type="hinge" axis="0 0 1" damping="0.05"/>
      <geom name="puck_geom" type="box" size="0.045 0.045 0.035"
            mass="0.2" rgba="0.15 0.38 0.95 1"/>
    </body>

    <camera name="top" pos="0 0 1.12" quat="1 0 0 0" fovy="45"/>
  </worldbody>
</mujoco>
"""


class SimplePushEnv(gym.Env):
    """Simple planar pushing task with MuJoCo RGB rendering."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        image_size: tuple[int, int] = (128, 128),
        frame_skip: int = 10,
        max_episode_steps: int = 200,
        action_scale: float = 0.04,
        target_pos: tuple[float, float] | np.ndarray | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode={render_mode!r}")

        self.render_mode = render_mode
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale

        self.model = mujoco.MjModel.from_xml_string(PUSH_XML)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        obs_spaces: dict[str, spaces.Space[Any]] = {
            "object_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        }
        if self.render_mode == "rgb_array":
            height, width = self.image_size
            obs_spaces["image"] = spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)

        self.target_pos = np.array(
            target_pos if target_pos is not None else [0.35, 0.0],
            dtype=np.float32,
        )
        self.workspace_low = np.array([-0.45, -0.30], dtype=np.float32)
        self.workspace_high = np.array([0.45, 0.30], dtype=np.float32)
        self.target_pos = np.clip(self.target_pos, self.workspace_low, self.workspace_high)
        self.ee_z = 0.045
        self._ee_pos = np.array([-0.25, 0.0, self.ee_z], dtype=np.float64)
        self._step_count = 0

        self._renderer: mujoco.Renderer | None = None
        self._renderer_failed = False
        self._viewer = None
        self._viewer_failed = False

        self._puck_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "puck")
        self._target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.model.site_pos[self._target_site_id, :2] = self.target_pos
        self._puck_x_adr = self._joint_qpos_addr("puck_x")
        self._puck_y_adr = self._joint_qpos_addr("puck_y")
        self._puck_yaw_adr = self._joint_qpos_addr("puck_yaw")

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

        puck_xy = np.array(
            [
                self.np_random.uniform(-0.03, 0.10),
                self.np_random.uniform(-0.12, 0.12),
            ],
            dtype=np.float64,
        )
        ee_xy = np.array(
            [
                self.np_random.uniform(-0.35, -0.25),
                self.np_random.uniform(-0.12, 0.12),
            ],
            dtype=np.float64,
        )

        self.data.qpos[self._puck_x_adr] = puck_xy[0]
        self.data.qpos[self._puck_y_adr] = puck_xy[1]
        self.data.qpos[self._puck_yaw_adr] = self.np_random.uniform(-0.2, 0.2)
        self.data.qvel[:] = 0.0

        self._ee_pos[:] = [ee_xy[0], ee_xy[1], self.ee_z]
        self.data.mocap_pos[0] = self._ee_pos
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
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._ee_pos[:2] = np.clip(
            self._ee_pos[:2] + action * self.action_scale,
            self.workspace_low,
            self.workspace_high,
        )

        for _ in range(self.frame_skip):
            self.data.mocap_pos[0] = self._ee_pos
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        info = self._get_info()
        reward = -info["distance_to_target"]
        terminated = info["distance_to_target"] < 0.04
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
                print(f"MuJoCo RGB renderer unavailable ({exc}). Using top-down fallback renderer.")

        if self._renderer is not None:
            self._renderer.update_scene(self.data, camera="top")
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
            "object_pos": self.data.xpos[self._puck_body_id, :2].astype(np.float32).copy(),
            "ee_pos": self._ee_pos[:2].astype(np.float32).copy(),
        }
        if self.render_mode == "rgb_array":
            obs["image"] = self.render()
        return obs

    def _get_info(self) -> dict[str, Any]:
        object_pos = self.data.xpos[self._puck_body_id, :2]
        return {
            "target_pos": self.target_pos.copy(),
            "distance_to_target": float(np.linalg.norm(object_pos - self.target_pos)),
            "step_count": self._step_count,
        }

    def _fallback_render(self) -> np.ndarray:
        height, width = self.image_size
        image = np.full((height, width, 3), 230, dtype=np.uint8)

        grid_step = max(8, min(height, width) // 8)
        image[::grid_step, :, :] = 205
        image[:, ::grid_step, :] = 205

        target_px = self._world_to_pixel(self.target_pos)
        object_px = self._world_to_pixel(self.data.xpos[self._puck_body_id, :2])
        ee_px = self._world_to_pixel(self._ee_pos[:2])

        self._draw_line(image, ee_px, object_px, color=(245, 135, 55), thickness=max(2, width // 80))
        self._draw_line(image, object_px, target_px, color=(65, 155, 75), thickness=max(2, width // 90))
        self._draw_circle(image, target_px, radius=max(7, width // 12), color=(90, 190, 95))
        self._draw_box(image, object_px, half_size=max(7, width // 14), color=(40, 95, 230))
        self._draw_circle(image, ee_px, radius=max(6, width // 16), color=(235, 70, 50))
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


def _heuristic_action(env: SimplePushEnv, obs: dict[str, np.ndarray]) -> np.ndarray:
    object_pos = obs["object_pos"]
    ee_pos = obs["ee_pos"]

    approach_offset = np.array([-0.08, 0.0], dtype=np.float32)
    approach_pos = object_pos + approach_offset
    if np.linalg.norm(ee_pos - approach_pos) > 0.04:
        desired = approach_pos - ee_pos
    else:
        desired = env.target_pos - object_pos

    norm = np.linalg.norm(desired)
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (desired / norm).astype(np.float32)


def main() -> None:
    env = SimplePushEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    print(f"Reset: object={obs['object_pos']}, ee={obs['ee_pos']}, image={obs['image'].shape}")

    for step_idx in range(120):
        action = _heuristic_action(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if step_idx % 20 == 0 or terminated or truncated:
            print(
                f"step={step_idx:03d} reward={reward:.3f} "
                f"object={obs['object_pos']} ee={obs['ee_pos']} "
                f"dist={info['distance_to_target']:.3f}"
            )

        if terminated or truncated:
            break

    frame = env.render()
    print(f"Rendered RGB frame shape: {frame.shape}, dtype={frame.dtype}")
    env.close()


if __name__ == "__main__":
    main()
