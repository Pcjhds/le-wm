# LeWM OpenShift Training Setup

This setup builds the LeWorldModel repository as a container image, then runs training as an OpenShift Job. It is not intended to run as a web app Deployment.

## Branch And GitHub

Work from this branch:

```bash
git switch openshift-training-setup
git status
```

Push the branch to your GitHub fork:

```bash
git push -u myfork openshift-training-setup
```

If you prefer `origin` to point at your fork, update it explicitly first:

```bash
git remote set-url origin https://github.com/Pcjhds/le-wm.git
git push -u origin openshift-training-setup
```

Do not force push.

## Build Image With Import From Git

In the OpenShift console, use Import from Git with:

```text
Git repo URL: https://github.com/Pcjhds/le-wm.git
Git branch: openshift-training-setup
Build strategy: Dockerfile
Dockerfile path: Dockerfile
Image/Application name: le-wm
```

If the console creates a Deployment or Route, do not use that for training. The training workload is the Job YAML in this directory.

The Job manifests reference this image placeholder:

```text
image-registry.openshift-image-registry.svc:5000/YOUR_PROJECT/le-wm:latest
```

Replace `YOUR_PROJECT` with your OpenShift project name, or apply with `sed` as shown below.

## Persistent Storage

Create the PVC:

```bash
oc apply -f openshift/lewm-pvc.yaml
```

The PVC is mounted at:

```text
/workspace/stablewm
```

The container sets:

```text
STABLEWM_HOME=/workspace/stablewm
PYTHONPATH=/workspace/le-wm
WANDB_MODE=offline
```

Put the DMC training dataset expected by `config/train/data/dmc.yaml` on the PVC. For the default smoke test, the expected dataset name is:

```text
$STABLEWM_HOME/reacher.h5
```

## Debug The PVC

Create a debug pod:

```bash
sed "s/YOUR_PROJECT/$(oc project -q)/g" openshift/lewm-debug-pod.yaml | oc apply -f -
```

Open a shell:

```bash
oc rsh lewm-debug
ls -lah /workspace/stablewm
```

Remove it when done:

```bash
oc delete pod lewm-debug
```

## Run CPU Smoke Training Job

Apply the CPU Job:

```bash
sed "s/YOUR_PROJECT/$(oc project -q)/g" openshift/lewm-train-job.yaml | oc apply -f -
```

It runs:

```bash
python train.py data=dmc trainer.max_epochs=1 loader.batch_size=2 num_workers=0
```

Watch logs:

```bash
oc logs -f job/lewm-train-smoke
```

Inspect pods:

```bash
oc get pods
```

Note: the repository default training config uses `trainer.accelerator=gpu`. If your CPU-only cluster has no GPU and Lightning exits early, rerun a CPU-specific test by adding Hydra overrides such as `trainer.accelerator=cpu trainer.devices=1 trainer.precision=32`.

## Run GPU Smoke Training Job

Apply the GPU Job:

```bash
sed "s/YOUR_PROJECT/$(oc project -q)/g" openshift/lewm-train-gpu-job.yaml | oc apply -f -
```

Watch logs:

```bash
oc logs -f job/lewm-train-gpu-smoke
```

The GPU Job requests:

```text
cpu: "8"
memory: "32Gi"
nvidia.com/gpu: "1"
```

## Outputs

Training artifacts should appear under:

```text
$STABLEWM_HOME/checkpoints
```

From the debug pod:

```bash
oc rsh lewm-debug
find /workspace/stablewm/checkpoints -maxdepth 3 -type f
```
