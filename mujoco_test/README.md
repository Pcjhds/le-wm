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

## Using pretrained files

Put the downloaded pretrained files under `trained_model/`:

```text
trained_model/
  config.json
  weights.pt
```

Validate that the checkpoint loads into the local JEPA model:

```bash
python mujoco_test/load_pretrained_model.py --model-dir trained_model
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

## Collect random rollouts

```bash
python mujoco_test/collect_rollouts.py --episodes 3 --steps 50
```

For a more visually dynamic rollout, use the persistent goal-biased random policy:

```bash
python mujoco_test/collect_rollouts.py --episodes 5 --steps 200 --policy aggressive_random --out mujoco_test/rollouts/simple_push_dynamic.npz
```

## Preparing MuJoCo rollouts for model input

```bash
python mujoco_test/prepare_model_inputs.py --rollout mujoco_test/rollouts/simple_push_rollouts.npz
```

## Run pretrained model on MuJoCo rollout

```bash
python mujoco_test/run_model_on_rollout.py --model-dir trained_model --rollout mujoco_test/rollouts/simple_push_rollouts.npz
```

## Evaluate latent prediction on MuJoCo rollout

```bash
python mujoco_test/evaluate_latent_prediction.py --model-dir trained_model --rollout mujoco_test/rollouts/simple_push_rollouts.npz
```

## Compare latent prediction baselines

```bash
python mujoco_test/evaluate_latent_baselines.py --model-dir trained_model --rollout mujoco_test/rollouts/simple_push_rollouts.npz
```

## Evaluate multistep latent prediction

```bash
python mujoco_test/evaluate_multistep_latent.py --target-offset 5
```

## Future LeWorldModel Integration

Next steps:

1. collect rollouts: `image_t`, `action_t`, `image_t+1`
2. feed image/action sequences to LeWorldModel
3. compare predicted next latent/image with real MuJoCo next observation
4. only after that, connect a policy loop
