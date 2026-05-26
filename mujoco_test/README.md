# MuJoCo Test Scaffold

Small MuJoCo sandbox for evaluating LeWorldModel-style image/action/world-model rollouts. This is intentionally minimal: no Isaac Sim, no robot arm, no training loop, and no LeWorldModel checkpoint integration yet.

## Setup

From the parent project directory:

```bash
conda create -n mujoco_env python=3.11 -y
conda activate mujoco_env
pip install -r mujoco_test/requirements.txt
```

If you are already inside this `mujoco_test` directory, use:

```bash
pip install -r requirements.txt
```

## Run the MuJoCo box test

```bash
python mujoco_test/test_mujoco_box.py
```

This creates a tiny XML scene with a ground plane, light, free box, and optional passive viewer. If the passive viewer is unavailable in your local display setup, the script falls back to a headless simulation.

On macOS, if the passive viewer complains about GLFW/Cocoa threading, try:

```bash
mjpython mujoco_test/test_mujoco_box.py
```

## Test the simple pushing environment

```bash
python mujoco_test/simple_push_env.py
```

`SimplePushEnv` follows a Gymnasium-style API:

- action: 2D end-effector delta movement, clipped to `[-1, 1]`
- observation: object position, end-effector position, and RGB camera image when `render_mode="rgb_array"`
- step: applies action, advances MuJoCo, returns `obs, reward, terminated, truncated, info`
- reset: randomizes object and end-effector positions
- render: returns a top-down RGB image

The physics are deliberately simple. A kinematic MuJoCo mocap body acts as the point end-effector and pushes a small planar block across a table.

## Future LeWorldModel Integration

Next steps:

1. collect rollouts: `image_t`, `action_t`, `image_t+1`
2. feed image/action sequences to LeWorldModel
3. compare predicted next latent/image with real MuJoCo next observation
4. only after that, connect a policy loop
