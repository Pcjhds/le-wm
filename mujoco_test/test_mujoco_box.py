"""Minimal MuJoCo sanity check with a falling free box."""

from __future__ import annotations

import argparse
import time

import mujoco


BOX_XML = """
<mujoco model="box_sanity_check">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.82 0.82 0.82" rgb2="0.92 0.92 0.92"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light name="key" pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="3 3 0.1" material="grid_mat"/>

    <body name="box" pos="0 0 0.8">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.12 0.12 0.12"
            mass="0.5" rgba="0.2 0.45 0.95 1"/>
    </body>

    <camera name="view" pos="1.4 -1.6 1.1" xyaxes="0.75 0.66 0 -0.33 0.38 0.86"/>
  </worldbody>
</mujoco>
"""


def maybe_launch_passive_viewer(model: mujoco.MjModel, data: mujoco.MjData):
    """Try the official passive viewer and fall back to headless simulation."""
    try:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)
        print("Passive viewer launched. Close the window or wait for the demo to finish.")
        return viewer
    except Exception as exc:  # Viewer availability depends on the local display setup.
        print(f"Passive viewer unavailable ({exc}). Running headless.")
        return None


def run(seconds: float, use_viewer: bool) -> None:
    model = mujoco.MjModel.from_xml_string(BOX_XML)
    data = mujoco.MjData(model)

    viewer = maybe_launch_passive_viewer(model, data) if use_viewer else None
    wall_start = time.time()

    try:
        while data.time < seconds:
            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                elapsed_sim = data.time
                elapsed_wall = time.time() - wall_start
                if elapsed_sim > elapsed_wall:
                    time.sleep(elapsed_sim - elapsed_wall)
    finally:
        if viewer is not None:
            viewer.close()

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    print(f"Finished {data.time:.2f}s sim. Final box position: {data.xpos[box_id]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--no-viewer", action="store_true", help="Run without the passive viewer.")
    args = parser.parse_args()

    run(seconds=args.seconds, use_viewer=not args.no_viewer)


if __name__ == "__main__":
    main()
