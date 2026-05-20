# LeWM 在 NERC OpenShift AI 上自动下载 Hugging Face 数据并训练的操作流程

本文档说明如何把当前 LeWM 项目部署到 NERC Red Hat OpenShift AI 中，让训练任务在服务器端自动下载 Hugging Face 数据集、解压、训练、保存 checkpoint，并在训练后调用模型做测试。

官方 Hugging Face collection：

```text
https://huggingface.co/collections/quentinll/lewm
```

注意：collection 中同时包含 dataset repo 和同名 model repo。下载训练数据时必须使用 `--repo-type dataset`，例如：

```bash
hf download quentinll/lewm-pusht --repo-type dataset --local-dir /mnt/lewm/raw/lewm-pusht
```

## 1. 目标流程

整体流程如下：

```text
NERC OpenShift AI Data Science Project
  -> PVC 持久化存储
  -> 自定义训练镜像
  -> PyTorchJob 或 Data Science Pipeline
  -> 训练 Pod 启动
  -> 自动从 Hugging Face 下载 .h5.zst 或 .tar.zst 数据
  -> 解压为 stable_worldmodel 可读取的 .h5 数据
  -> python train.py data=hf_pusht launcher=nerc
  -> 保存 weights_epoch_*.pt 和 config.json
  -> python eval.py 调用训练好的模型测试
```

推荐先使用 `PyTorchJob` 跑通单次训练；确认无误后，再把相同逻辑封装成 OpenShift AI Data Science Pipeline，并设置 recurring / cron run 实现周期性自主训练。

## 1.1 NERC OpenShift 和 OpenShift AI 的区别

简单说：

```text
NERC OpenShift = 底层 Kubernetes / container 平台
NERC OpenShift AI = 运行在 OpenShift 之上的 AI/ML 平台层
```

二者关系不是二选一，而是上下层关系。

### NERC OpenShift

NERC OpenShift 是 NERC 提供的 Red Hat OpenShift / OpenShift Container Platform 环境。它本质上是一个企业版 Kubernetes 平台，用来运行 container workload。

在 OpenShift 中你直接管理这些资源：

```text
Project / Namespace
Pod
Job
Deployment
PVC
Secret
ConfigMap
Route
Service
ResourceQuota
GPU resource requests
```

如果只使用 OpenShift，你仍然可以运行 LeWM 训练。方式是创建普通 Kubernetes `Job`，让 Job 启动一个 GPU container，自动下载 Hugging Face 数据，然后执行 `python train.py`。

本仓库对应文件是：

```text
deploy/openshift/lewm-train-job.yaml
deploy/openshift/lewm-eval-job.yaml
deploy/openshift/pvc.yaml
deploy/openshift/hf-secret.example.yaml
```

### NERC OpenShift AI

NERC OpenShift AI 是在 OpenShift 之上提供的 AI/ML 工作流平台。它通常提供更适合机器学习项目的 UI 和组件，例如：

```text
Data Science Project
Workbench / Notebook
Data Science Pipelines
Model Serving
Connections
Storage attachment
GPU accelerator profile
```

NERC 文档中说明，OpenShift AI 的 Data Science Project 会对应 NERC OpenShift resource allocation。也就是说，OpenShift AI 是更高层的机器学习入口，但底层创建出来的仍然是 OpenShift / Kubernetes 资源。

### 对本项目的影响

对 LeWM 来说，两种方式都可以：

```text
只用 OpenShift:
  用 PVC + Secret + 普通 Job 训练/评估
  适合最小化部署、命令行操作、没有 RHOAI pipeline 需求的情况

使用 OpenShift AI:
  用 Data Science Project + Workbench + Pipeline/PyTorchJob + Model Serving
  适合交互调试、周期训练、多人协作和后续模型服务化
```

如果目标只是“在服务器上自动下载 Hugging Face 数据并训练”，普通 OpenShift 已经足够。

如果目标是“像 MLOps 一样管理训练、评估、周期运行、模型保存、模型部署”，建议使用 OpenShift AI。

## 2. 前置条件

需要具备以下条件：

1. 已经有 NERC OpenShift / OpenShift AI 账号和对应 resource allocation。
2. 已经创建或拥有一个 Data Science Project。
3. 项目 namespace 中有 GPU quota，例如 A100、H100 或 V100。
4. 训练 Pod 可以访问 Hugging Face。
5. 有足够大的 PVC 保存数据和模型。
6. 能够构建并推送自定义 container image。

建议 PVC 容量：

```text
PushT 数据压缩包约 13GB，解压后更大。
建议最少 100Gi；如果还要保留多个 epoch checkpoint，建议 200Gi 或更高。
```

## 3. 目录约定

在训练 Pod 中建议把 PVC 挂载到：

```text
/mnt/lewm
```

并设置：

```bash
export STABLEWM_HOME=/mnt/lewm
export LOCAL_DATASET_DIR=/mnt/lewm
export HF_HOME=/mnt/lewm/hf-cache
```

训练后会产生类似目录：

```text
/mnt/lewm/
  pusht_expert_train.h5
  raw/
    lewm-pusht/
      pusht_expert_train.h5.zst
  hf-cache/
  checkpoints/
    pusht/
      lewm/
        config.json
        weights_epoch_1.pt
        weights_epoch_2.pt
        ...
```

## 4. 需要新增或修改的 YAML 文件

本仓库已经新增这些文件：

```text
Dockerfile
.dockerignore
config/train/data/hf_pusht.yaml
config/train/data/hf_tworoom.yaml
config/train/data/hf_cube.yaml
config/train/data/hf_reacher.yaml
config/train/launcher/nerc.yaml
deploy/openshift/pvc.yaml
deploy/openshift/hf-secret.example.yaml
deploy/openshift/github-secret.example.yaml
deploy/openshift/lewm-buildconfig.yaml
deploy/openshift/lewm-train-pytorchjob.yaml
deploy/openshift/lewm-train-job.yaml
deploy/openshift/lewm-train-from-github-job.yaml
deploy/openshift/lewm-eval-job.yaml
```

如果需要自动周期性训练，再新增：

```text
pipelines/lewm_train_pipeline.py
pipelines/lewm_train_pipeline.yaml
```

其中 `pipeline.yaml` 不建议手写，应该由 Kubeflow Pipelines SDK 从 `pipeline.py` 编译生成。

## 5. Hugging Face 数据集对应关系

官方 collection 中目前有 4 个 dataset repo：

```text
quentinll/lewm-pusht
quentinll/lewm-tworooms
quentinll/lewm-cube
quentinll/lewm-reacher
```

对应关系：

```text
HF dataset repo              下载文件                         解压后训练配置
quentinll/lewm-pusht         pusht_expert_train.h5.zst        data=hf_pusht
quentinll/lewm-tworooms      tworoom.tar.zst                  data=hf_tworoom
quentinll/lewm-cube          cube_single_expert.tar.zst       data=hf_cube
quentinll/lewm-reacher       reacher.tar.zst                  data=hf_reacher
```

PushT 是单个 `.h5.zst` 文件，直接用 `zstd -d` 解压。TwoRoom、Cube、Reacher 是 `.tar.zst`，应使用：

```bash
tar --zstd -xvf archive.tar.zst -C /mnt/lewm
```

## 6. 新增训练数据配置

PushT 配置文件：

```text
config/train/data/hf_pusht.yaml
```

内容：

```yaml
dataset:
  num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}
  frameskip: 5
  name: pusht_expert_train.h5
  keys_to_load:
    - pixels
    - action
    - proprio
    - state
  keys_to_cache:
    - action
    - proprio
    - state
```

说明：

- `name: pusht_expert_train.h5` 必须与最终解压到 `/mnt/lewm` 下的数据文件名一致。
- `train.py` 会优先从 `LOCAL_DATASET_DIR` 读取数据；这里把 `LOCAL_DATASET_DIR` 和 `STABLEWM_HOME` 都设为 `/mnt/lewm`。
- 训练至少需要 `pixels` 和 `action`。
- `proprio`、`state` 常用于环境评估和 reset，不一定直接参与 LeWM loss，但建议保留。

如果换成其他 Hugging Face 数据集，不能只改 dataset 名称。当前代码需要 `.h5` 或 Lance 数据，且字段需要能被 `stable_worldmodel` 读取。普通 Hugging Face parquet/json/video 数据集需要先转换成 LeWM 需要的格式。

其他官方数据集也已经生成了对应配置：

```text
config/train/data/hf_tworoom.yaml
config/train/data/hf_cube.yaml
config/train/data/hf_reacher.yaml
```

训练命令示例：

```bash
python train.py data=hf_pusht launcher=nerc output_model_name=pusht/lewm
python train.py data=hf_tworoom launcher=nerc output_model_name=tworoom/lewm
python train.py data=hf_cube launcher=nerc output_model_name=cube/lewm
python train.py data=hf_reacher launcher=nerc output_model_name=reacher/lewm
```

## 7. 新增 NERC 训练配置

新增文件：

```text
config/train/launcher/nerc.yaml
```

内容：

```yaml
# @package _global_

defaults:
  - override /hydra/launcher: basic

trainer:
  accelerator: gpu
  devices: 1
  precision: bf16
  max_epochs: 100
  gradient_clip_val: 1.0

loader:
  batch_size: 64
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 2
  pin_memory: true

wandb:
  enabled: false
  config:
    entity: lewm
    project: lewm
    name: ${output_model_name}
    id: ${subdir}
    resume: allow
    log_model: false
```

说明：

- 如果 GPU 显存足够，可以把 `loader.batch_size` 调回默认的 `128`。
- 如果 Pod 内 DataLoader worker 不稳定，可以先把 `num_workers` 降到 `2` 或 `0`。
- 如果使用 V100 且 `bf16` 不稳定，可以改为：

```yaml
trainer:
  precision: 16-mixed
```

## 8. PVC YAML

新增文件：

```text
deploy/openshift/pvc.yaml
```

内容：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: lewm-storage
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 200Gi
```

应用：

```bash
oc apply -f deploy/openshift/pvc.yaml
```

## 9. Hugging Face Token Secret

如果数据集是公开的，可以不创建 token secret。

如果数据集是 private 或 gated，新增文件：

```text
deploy/openshift/hf-secret.yaml
```

内容：

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hf-token
type: Opaque
stringData:
  HF_TOKEN: "hf_xxx"
```

应用：

```bash
oc apply -f deploy/openshift/hf-secret.yaml
```

注意：不要把真实 token 提交到 Git 仓库。实际使用时建议用 `oc create secret`：

```bash
oc create secret generic hf-token --from-literal=HF_TOKEN=hf_xxx
```

## 10. 自定义训练镜像

训练镜像至少需要包含：

```text
Python 3.10
PyTorch / torchvision
Lightning
Hydra
stable-worldmodel[train,env]
stable-pretraining
huggingface_hub
zstandard 或系统 zstd
当前 LeWM repo 代码
```

示例 Dockerfile：

```dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/le-wm-main

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    hydra-core \
    lightning \
    einops \
    huggingface_hub \
    zstandard \
    "stable-worldmodel[train,env]"

COPY . /workspace/le-wm-main

ENV PYTHONUNBUFFERED=1
```

构建并推送：

```bash
podman build -t quay.io/YOUR_ORG/lewm-train:latest .
podman push quay.io/YOUR_ORG/lewm-train:latest
```

如果 NERC 集群不能直接拉取你的镜像，需要配置 image pull secret。

本仓库现在已经提供了一个用于 NERC/OpenShift 构建训练镜像的 `Dockerfile`。它会：

```text
使用 CUDA runtime base image
安装 Python 3.10
安装 PyTorch CUDA wheel
安装 stable-worldmodel[train,env]
安装 hydra/lightning/einops/huggingface_hub/zstandard
复制当前 LeWM 项目代码
设置 OpenShift arbitrary UID 兼容权限
```

因此你可以不在本地构建镜像，而是在 OpenShift 中用 BuildConfig 从 GitHub 构建。

对应 YAML：

```text
deploy/openshift/lewm-buildconfig.yaml
```

这个 BuildConfig 会从：

```text
https://github.com/Pcjhds/le-wm.git
```

拉取分支：

```text
yaalbert-yamlud-0519
```

然后使用仓库根目录的：

```text
Dockerfile
```

构建内部镜像：

```text
image-registry.openshift-image-registry.svc:5000/digital-twins-for-automated-process-f532cb/lewm-train:latest
```

## 11. PyTorchJob YAML

新增文件：

```text
deploy/openshift/lewm-train-pytorchjob.yaml
```

内容：

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: lewm-pusht-train
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        spec:
          containers:
            - name: pytorch
              image: quay.io/YOUR_ORG/lewm-train:latest
              imagePullPolicy: IfNotPresent
              command: ["/bin/bash", "-lc"]
              args:
                - |
                  set -euo pipefail

                  export STABLEWM_HOME=/mnt/lewm
                  export LOCAL_DATASET_DIR=/mnt/lewm
                  export HF_HOME=/mnt/lewm/hf-cache

                  mkdir -p /mnt/lewm/raw/lewm-pusht
                  mkdir -p /mnt/lewm/checkpoints
                  mkdir -p /mnt/lewm/hydra_runs

                  if [ ! -f /mnt/lewm/pusht_expert_train.h5 ]; then
                    echo "Downloading LeWM PushT dataset from Hugging Face..."
                    HF_CLI=hf
                    if ! command -v hf >/dev/null 2>&1; then
                      HF_CLI=huggingface-cli
                    fi
                    $HF_CLI download quentinll/lewm-pusht \
                      --repo-type dataset \
                      --local-dir /mnt/lewm/raw/lewm-pusht

                    echo "Decompressing dataset..."
                    zstd -d -f \
                      /mnt/lewm/raw/lewm-pusht/pusht_expert_train.h5.zst \
                      -o /mnt/lewm/pusht_expert_train.h5
                  else
                    echo "Dataset already exists. Skipping download."
                  fi

                  echo "Starting LeWM training..."
                  python train.py \
                    data=hf_pusht \
                    launcher=nerc \
                    output_model_name=pusht/lewm \
                    hydra.run.dir=/mnt/lewm/hydra_runs/${HOSTNAME}
              envFrom:
                - secretRef:
                    name: hf-token
                    optional: true
              resources:
                requests:
                  cpu: "4"
                  memory: "32Gi"
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "8"
                  memory: "64Gi"
                  nvidia.com/gpu: "1"
              volumeMounts:
                - name: lewm-storage
                  mountPath: /mnt/lewm
          volumes:
            - name: lewm-storage
              persistentVolumeClaim:
                claimName: lewm-storage
  runPolicy:
    suspend: false
```

应用：

```bash
oc apply -f deploy/openshift/lewm-train-pytorchjob.yaml
```

查看 Pod：

```bash
oc get pods
```

查看日志：

```bash
oc logs -f pytorchjob/lewm-pusht-train
```

如果 `oc logs -f pytorchjob/...` 不可用，则先找到 master pod：

```bash
oc get pods | grep lewm-pusht-train
oc logs -f <pod-name>
```

## 12. 训练完成后会得到什么

训练完成后，PVC 中应有：

```text
/mnt/lewm/checkpoints/pusht/lewm/
  config.json
  weights_epoch_1.pt
  weights_epoch_2.pt
  ...
  weights_epoch_100.pt
```

还可能有 Hydra / Lightning 运行目录：

```text
/mnt/lewm/checkpoints/<hydra-job-id>/
  config.yaml
```

最重要的文件是：

```text
config.json
weights_epoch_*.pt
```

后续测试通常指定最后一个 epoch：

```text
pusht/lewm/weights_epoch_100.pt
```

## 13. 测试训练好的模型

可以新建一个测试 PyTorchJob，或者直接在同一个 Workbench / Pod 中执行：

```bash
export STABLEWM_HOME=/mnt/lewm
export LOCAL_DATASET_DIR=/mnt/lewm

python eval.py \
  --config-name=pusht.yaml \
  policy=pusht/lewm/weights_epoch_100.pt \
  hydra.run.dir=/mnt/lewm/hydra_eval_runs/${HOSTNAME}
```

评估会：

1. 加载训练好的 LeWM 模型。
2. 使用 `stable_worldmodel` 的 world environment。
3. 使用 CEM 或 Adam solver 规划动作。
4. 打印 `metrics`。
5. 写入结果文件，例如：

```text
pusht_results.txt
```

## 14. 可选：用 Data Science Pipeline 实现周期性自主训练

如果你希望它自动定期训练，而不是手动 `oc apply`，建议使用 OpenShift AI Data Science Pipelines。

流程：

```text
pipeline.py
  -> kfp compiler
  -> lewm_train_pipeline.yaml
  -> OpenShift AI Dashboard 导入 pipeline
  -> 创建 recurring run / cron run
```

Pipeline 的任务可以拆成：

```text
download_dataset
  -> train_lewm
  -> eval_lewm
```

也可以先把三步都放在一个 container task 中，逻辑与 PyTorchJob 的 shell 脚本一致。

注意：

- `pipelines/lewm_train_pipeline.yaml` 应由 KFP SDK 编译生成。
- OpenShift AI Dashboard 中可以导入 pipeline YAML。
- 导入后可以创建 scheduled / recurring run。
- 训练产物仍建议写入 `/mnt/lewm/checkpoints/...` 或对象存储。

## 15. 常见问题

### 15.1 找不到 stable_worldmodel

错误：

```text
ModuleNotFoundError: No module named 'stable_worldmodel'
```

解决：

```bash
pip install "stable-worldmodel[train,env]"
```

最好把它写入自定义镜像，而不是 Pod 启动后临时安装。

### 15.2 找不到数据集

错误可能类似：

```text
FileNotFoundError: pusht_expert_train.h5
```

检查：

```bash
echo $STABLEWM_HOME
ls -lh /mnt/lewm/pusht_expert_train.h5
```

确保：

```text
config/train/data/hf_pusht.yaml 中的 name
```

与实际文件名一致。

### 15.3 下载重复执行

PyTorchJob 中已经有判断：

```bash
if [ ! -f /mnt/lewm/pusht_expert_train.h5 ]; then
  download...
fi
```

只要 PVC 不删除，第二次运行会跳过下载。

### 15.4 GPU 不可用

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果不可用，检查：

- namespace 是否有 GPU quota。
- PyTorchJob 是否请求了 `nvidia.com/gpu: "1"`。
- 镜像是否为 CUDA 镜像。
- OpenShift AI 中是否选择了 GPU accelerator。

### 15.5 bf16 不支持

如果使用旧 GPU 或 bf16 报错，把 `config/train/launcher/nerc.yaml` 中：

```yaml
precision: bf16
```

改成：

```yaml
precision: 16-mixed
```

或者调试时先用：

```yaml
precision: 32
```

## 16. 最小落地清单

第一次落地建议按这个顺序：

1. 创建 PVC：`deploy/openshift/pvc.yaml`
2. 构建并推送训练镜像：`quay.io/YOUR_ORG/lewm-train:latest`
3. 新增 `config/train/data/hf_pusht.yaml`
4. 新增 `config/train/launcher/nerc.yaml`
5. 创建 PyTorchJob：`deploy/openshift/lewm-train-pytorchjob.yaml`
6. `oc apply -f deploy/openshift/lewm-train-pytorchjob.yaml`
7. 查看日志确认下载、解压、训练是否成功
8. 在 PVC 中确认 checkpoint
9. 运行 `eval.py` 测试模型
10. 成功后再封装为 Data Science Pipeline 做周期训练

## 17. 在 NERC OpenShift AI 中的实际操作步骤

本仓库已经生成了以下 YAML 文件：

```text
config/train/data/hf_pusht.yaml
config/train/data/hf_tworoom.yaml
config/train/data/hf_cube.yaml
config/train/data/hf_reacher.yaml
config/train/launcher/nerc.yaml
deploy/openshift/pvc.yaml
deploy/openshift/hf-secret.example.yaml
deploy/openshift/lewm-train-pytorchjob.yaml
deploy/openshift/lewm-train-job.yaml
deploy/openshift/lewm-eval-job.yaml
```

你需要手动替换的内容主要有：

```text
quay.io/YOUR_ORG/lewm-train:latest
REPLACE_WITH_YOUR_HUGGINGFACE_TOKEN
```

如果 Hugging Face 数据集是公开的，可以不创建 `hf-token` secret。训练 YAML 中的 secret 是 `optional: true`，没有 secret 时也能启动。

### 17.1 通过 NERC OpenShift AI Dashboard 操作

1. 登录 NERC OpenShift AI Dashboard。
2. 进入或创建你的 Data Science Project。
3. 确认项目关联了正确的 NERC OpenShift allocation。
4. 在项目中创建 Workbench，用于首次调试。
5. Workbench 创建时选择合适的 notebook image、CPU、memory、cluster storage 和 GPU accelerator。
6. 如果页面提供 A100、H100 或 V100 等 accelerator profile，选择与你 quota 匹配的 GPU 类型和数量。
7. 启动 Workbench，进入 JupyterLab 或终端。
8. 在 Workbench 中 clone 当前 LeWM repo，或者使用已经构建好的训练镜像。
9. 在 OpenShift Web Console 中进入同一个 project / namespace。
10. 点击顶部 `+` 或 `Import YAML`。
11. 先导入 `deploy/openshift/pvc.yaml`。
12. 如果需要 HF token，导入修改后的 `hf-secret.example.yaml`，或者用 `oc create secret` 创建。
13. 优先导入 `deploy/openshift/lewm-train-pytorchjob.yaml`。
14. 如果集群提示没有 `PyTorchJob` 这个 kind，改用 `deploy/openshift/lewm-train-job.yaml`。
15. 在 Pods 页面查看训练 Pod 是否启动。
16. 打开训练 Pod logs，确认出现下载、解压和 training log。
17. 训练完成后，确认 PVC 中存在 checkpoint。
18. 导入 `deploy/openshift/lewm-eval-job.yaml` 或在 Workbench 终端中运行 `eval.py` 测试。

NERC Dashboard 的关键点是：Data Science Project 管理项目资源，Workbench 用来交互调试，OpenShift Web Console 的 `Import YAML` 用来创建 PVC、Secret、PyTorchJob 和 Job。

### 17.2 通过 oc CLI 操作

先登录 OpenShift：

```bash
oc login <YOUR_OPENSHIFT_API_URL> --token=<YOUR_TOKEN>
```

切换到你的 NERC project / namespace：

```bash
oc project <YOUR_PROJECT_NAMESPACE>
```

创建 PVC：

```bash
oc apply -f deploy/openshift/pvc.yaml
```

如果需要 Hugging Face token，推荐用命令创建 secret：

```bash
oc create secret generic hf-token --from-literal=HF_TOKEN=hf_xxx
```

如果你选择使用 YAML，则复制示例并替换 token：

```bash
cp deploy/openshift/hf-secret.example.yaml deploy/openshift/hf-secret.yaml
# 编辑 deploy/openshift/hf-secret.yaml，把 REPLACE_WITH_YOUR_HUGGINGFACE_TOKEN 换成真实 token
oc apply -f deploy/openshift/hf-secret.yaml
```

提交训练任务：

```bash
oc apply -f deploy/openshift/lewm-train-pytorchjob.yaml
```

如果你的 NERC namespace 没有 `PyTorchJob` CRD，使用普通 Kubernetes Job：

```bash
oc apply -f deploy/openshift/lewm-train-job.yaml
```

查看资源：

```bash
oc get pytorchjob
oc get job
oc get pods
oc get pvc
```

查看训练日志：

```bash
oc get pods | grep lewm-pusht-train
oc logs -f <TRAIN_POD_NAME>
```

训练完成后运行评估：

```bash
oc apply -f deploy/openshift/lewm-eval-job.yaml
```

查看评估日志：

```bash
oc get pods | grep lewm-pusht-eval
oc logs -f <EVAL_POD_NAME>
```

如果要重新跑训练任务，先删除旧的 PyTorchJob，再重新 apply：

```bash
oc delete pytorchjob lewm-pusht-train
oc apply -f deploy/openshift/lewm-train-pytorchjob.yaml
```

如果使用的是普通 Job：

```bash
oc delete job lewm-pusht-train
oc apply -f deploy/openshift/lewm-train-job.yaml
```

如果要重新跑评估任务：

```bash
oc delete job lewm-pusht-eval
oc apply -f deploy/openshift/lewm-eval-job.yaml
```

### 17.3 在 Workbench 中确认 PVC 内容

如果 Workbench 也挂载了同一个 `lewm-storage` PVC，可以在终端里检查：

```bash
ls -lh /mnt/lewm
ls -lh /mnt/lewm/checkpoints/pusht/lewm
```

应该看到：

```text
config.json
weights_epoch_1.pt
weights_epoch_2.pt
...
weights_epoch_100.pt
```

如果 Workbench 的挂载路径不是 `/mnt/lewm`，需要根据 Workbench 中实际挂载路径检查。

### 17.4 修改训练资源

如需调整 GPU、CPU、内存，修改：

```text
deploy/openshift/lewm-train-pytorchjob.yaml
```

重点字段：

```yaml
resources:
  requests:
    cpu: "4"
    memory: "32Gi"
    nvidia.com/gpu: "1"
  limits:
    cpu: "8"
    memory: "64Gi"
    nvidia.com/gpu: "1"
```

如需调整 epoch、batch size、precision，修改：

```text
config/train/launcher/nerc.yaml
```

重点字段：

```yaml
trainer:
  precision: bf16
  max_epochs: 100

loader:
  batch_size: 64
  num_workers: 4
```

### 17.5 修改 Hugging Face 数据集

当前训练 YAML 下载的是：

```text
quentinll/lewm-pusht
```

如果换成另一个 LeWM 官方数据集，需要同时改：

```text
deploy/openshift/lewm-train-pytorchjob.yaml
config/train/data/hf_pusht.yaml
```

例如 TwoRoom：

```bash
hf download quentinll/lewm-tworooms --repo-type dataset --local-dir /mnt/lewm/raw/lewm-tworooms
tar --zstd -xvf /mnt/lewm/raw/lewm-tworooms/tworoom.tar.zst -C /mnt/lewm
python train.py data=hf_tworoom launcher=nerc output_model_name=tworoom/lewm
```

例如 Cube：

```bash
hf download quentinll/lewm-cube --repo-type dataset --local-dir /mnt/lewm/raw/lewm-cube
tar --zstd -xvf /mnt/lewm/raw/lewm-cube/cube_single_expert.tar.zst -C /mnt/lewm
python train.py data=hf_cube launcher=nerc output_model_name=cube/lewm
```

例如 Reacher：

```bash
hf download quentinll/lewm-reacher --repo-type dataset --local-dir /mnt/lewm/raw/lewm-reacher
tar --zstd -xvf /mnt/lewm/raw/lewm-reacher/reacher.tar.zst -C /mnt/lewm
python train.py data=hf_reacher launcher=nerc output_model_name=reacher/lewm
```

通用替换下载 repo：

```bash
hf download <HF_DATASET_REPO> --repo-type dataset --local-dir /mnt/lewm/raw/<NAME>
```

同时保证解压后的文件名和 data config 中的 `dataset.name` 一致：

```yaml
dataset:
  name: your_dataset.h5
```

如果 Hugging Face 数据不是 LeWM 兼容的 `.h5.zst` 或 `.h5`，需要先添加数据转换步骤，把数据转换成包含 `pixels`、`action`、`episode_idx` / `step_idx` 等字段的 HDF5 或 Lance 数据。

### 17.6 用 Pipeline 做自动周期训练

单次 PyTorchJob 跑通后，可以把下载、训练、评估逻辑封装成 OpenShift AI Data Science Pipeline。

Dashboard 路径通常是：

```text
OpenShift AI Dashboard
  -> Data Science Project
  -> Data Science Pipelines
  -> Import pipeline
  -> Runs
  -> Schedules
```

建议 pipeline 分三步：

```text
download_dataset
  -> train_lewm
  -> eval_lewm
```

第一版也可以把三步放进一个 container task 中，直接复用 `deploy/openshift/lewm-train-pytorchjob.yaml` 里的 shell 逻辑。等 PyTorchJob 方式稳定后，再做 pipeline 化，排错成本会低很多。

## 18. 只使用 NERC OpenShift 是否可以

可以。上面的要求不强依赖 OpenShift AI。只要 NERC OpenShift project / namespace 具备以下能力，就能直接运行：

```text
可以拉取你的训练镜像
可以访问 Hugging Face
有可用 PVC
有 GPU node 和 GPU quota
namespace 支持请求 nvidia.com/gpu
```

只用 OpenShift 时，推荐使用普通 Kubernetes Job，而不是 PyTorchJob 或 OpenShift AI Pipeline。

需要用到的文件：

```text
deploy/openshift/pvc.yaml
deploy/openshift/hf-secret.example.yaml
deploy/openshift/lewm-train-job.yaml
deploy/openshift/lewm-eval-job.yaml
config/train/data/hf_pusht.yaml
config/train/launcher/nerc.yaml
```

### 18.1 只用 OpenShift 的最小步骤

1. 登录 OpenShift：

```bash
oc login <YOUR_OPENSHIFT_API_URL> --token=<YOUR_TOKEN>
```

2. 切换 namespace：

```bash
oc project <YOUR_PROJECT_NAMESPACE>
```

3. 创建 PVC：

```bash
oc apply -f deploy/openshift/pvc.yaml
```

4. 如果需要 Hugging Face token，创建 secret：

```bash
oc create secret generic hf-token --from-literal=HF_TOKEN=hf_xxx
```

5. 提交训练 Job：

如果你已经有可用训练镜像，直接运行：

```bash
oc apply -f deploy/openshift/lewm-train-job.yaml
```

如果还没有训练镜像，先让 OpenShift 从 GitHub + Dockerfile 构建镜像：

```bash
oc apply -f deploy/openshift/lewm-buildconfig.yaml
oc start-build lewm-train --follow
oc apply -f deploy/openshift/lewm-train-job.yaml
```

6. 查看日志：

```bash
oc get pods | grep lewm-pusht-train
oc logs -f <TRAIN_POD_NAME>
```

7. 训练完成后提交评估 Job：

```bash
oc apply -f deploy/openshift/lewm-eval-job.yaml
```

### 18.2 只用 OpenShift 时需要改动什么

通常需要改这些内容：

1. 修改训练镜像地址。

文件：

```text
deploy/openshift/lewm-train-job.yaml
deploy/openshift/lewm-eval-job.yaml
```

把：

```yaml
image: quay.io/YOUR_ORG/lewm-train:latest
```

改成你的真实镜像，例如：

```yaml
image: quay.io/my-team/lewm-train:latest
```

2. 根据 quota 调整 CPU、内存、GPU。

如果 namespace 只有 1 张 GPU，保持：

```yaml
nvidia.com/gpu: "1"
```

如果没有 GPU，训练仍可在 CPU 上跑通流程，但速度会非常慢。此时需要：

```yaml
resources:
  requests:
    cpu: "4"
    memory: "32Gi"
  limits:
    cpu: "8"
    memory: "64Gi"
```

并把 `config/train/launcher/nerc.yaml` 中：

```yaml
trainer:
  accelerator: gpu
  devices: 1
```

改成：

```yaml
trainer:
  accelerator: cpu
  devices: 1
```

3. 如果集群 GPU resource name 不是 `nvidia.com/gpu`，需要按 NERC 项目提供的 accelerator resource name 修改。

默认是：

```yaml
nvidia.com/gpu: "1"
```

4. 如果镜像是私有镜像，需要添加 image pull secret。

示例：

```yaml
imagePullSecrets:
  - name: my-registry-secret
```

放在 Pod spec 下：

```yaml
spec:
  imagePullSecrets:
    - name: my-registry-secret
  containers:
    ...
```

5. 如果 OpenShift 的 restricted security context 导致容器没有权限写工作目录，需要让镜像支持 arbitrary UID。

建议在镜像里避免写 `/workspace/le-wm-main`，训练输出全部写入：

```text
/mnt/lewm
```

当前 Job 已经设置：

```bash
export STABLEWM_HOME=/mnt/lewm
export LOCAL_DATASET_DIR=/mnt/lewm
export HF_HOME=/mnt/lewm/hf-cache
```

如果 Hydra 仍尝试在工作目录创建输出，可以在启动命令中追加：

```bash
hydra.run.dir=/mnt/lewm/hydra_runs/${HOSTNAME}
hydra.output_subdir=null
```

例如：

```bash
python train.py \
  data=hf_pusht \
  launcher=nerc \
  output_model_name=pusht/lewm \
  hydra.run.dir=/mnt/lewm/hydra_runs/${HOSTNAME}
```

6. 如果不使用 OpenShift AI，则不需要这些内容：

```text
Data Science Project 页面操作
Workbench
Data Science Pipeline
Model Serving
PyTorchJob
```

### 18.3 OpenShift 与 OpenShift AI 对应使用建议

建议选择：

```text
只想训练一次/手动触发:
  用 OpenShift Job

想要调试 notebook、查看数据、交互运行:
  用 OpenShift AI Workbench

想要周期性自动训练:
  用 OpenShift AI Data Science Pipeline

想要把训练好的模型作为服务暴露:
  用 OpenShift AI Model Serving，或者纯 OpenShift Deployment + Route
```

### 18.4 通过 NERC 网页设置，并从 GitHub 拉取当前项目运行

如果你不想先在本地构建 `quay.io/.../lewm-train:latest` 这种自定义镜像，可以让 OpenShift Job 启动后从 GitHub 拉取当前项目代码，再安装依赖并训练。

本仓库已经提供模板：

```text
deploy/openshift/lewm-train-from-github-job.yaml
deploy/openshift/github-secret.example.yaml
```

这种方式的流程是：

```text
你把当前 project 上传到 GitHub
  -> OpenShift Job 使用一个 PyTorch CUDA 基础镜像启动
  -> Job 在 Pod 内 git clone GitHub repo
  -> pip install 训练依赖
  -> 从 Hugging Face 下载 LeWM 数据
  -> 解压数据
  -> python train.py
```

注意：这里仍然需要一个基础 container image，但它不需要包含你的项目代码。它只需要提供 Python、PyTorch、CUDA、pip，最好也包含 `git`。

#### 18.4.1 先把当前项目上传到 GitHub

在本地创建 GitHub repo 后，把当前项目推上去。仓库可以是 public，也可以是 private。

最终你需要得到一个 repo URL，例如：

```text
https://github.com/Pcjhds/le-wm.git
```

如果是 private repo，需要创建 GitHub token，并在 OpenShift 里创建 secret。

#### 18.4.2 在 NERC 网页中创建 PVC

进入 OpenShift Web Console：

```text
Administrator 或 Developer 视图
  -> 选择你的 Project / Namespace
  -> Storage
  -> PersistentVolumeClaims
  -> Create PersistentVolumeClaim
```

或者使用 `Import YAML` 导入：

```text
deploy/openshift/pvc.yaml
```

PVC 名称应保持：

```text
lewm-storage
```

因为 Job YAML 中挂载的是这个名字。

#### 18.4.3 如果 GitHub repo 是 private，创建 GitHub Secret

复制：

```text
deploy/openshift/github-secret.example.yaml
```

改成：

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: github-token
type: Opaque
stringData:
  GITHUB_TOKEN: "ghp_xxx"
```

然后在 NERC OpenShift 网页中：

```text
Project
  -> Secrets
  -> Create
  -> From YAML
```

导入这个 Secret。

如果 repo 是 public，可以跳过这一步。

#### 18.4.4 如果 Hugging Face 数据需要 token，创建 HF Secret

公开 LeWM 数据通常可以直接下载。如果你使用 private / gated 数据集，创建：

```text
hf-token
```

可以用：

```text
deploy/openshift/hf-secret.example.yaml
```

Secret 名称保持：

```text
hf-token
```

#### 18.4.5 修改 from-github Job YAML

打开：

```text
deploy/openshift/lewm-train-from-github-job.yaml
```

至少修改 3 个地方。

第一，确认基础镜像：

```yaml
image: image-registry.openshift-image-registry.svc:5000/digital-twins-for-automated-process-f532cb/world-model-rob-git:latest
```

这个值来自你在 OpenShift ImageStreams 页面看到的内部镜像：

```text
image-registry.openshift-image-registry.svc:5000/digital-twins-for-automated-process-f532cb/world-model-rob-git
```

实际在 Pod YAML 中通常需要带 tag，所以这里使用：

```text
:latest
```

如果 ImageStream 详情页显示 tag 不是 `latest`，请把 `:latest` 改成实际 tag。

如果你在 OpenShift AI 里创建过 Workbench，可以在 OpenShift Web Console 中找到 Workbench 对应 Pod，进入 Pod YAML，复制其中的 `containers.image` 值。

这个镜像最好满足：

```text
Python
PyTorch
CUDA
pip
git
```

如果镜像没有 `git`，这个 Job 会在 `git clone` 步骤失败。此时需要换一个包含 git 的镜像，或者回到自定义镜像方案。

你当前 ImageStream 标签里曾显示类似：

```text
runtime=python
runtime-version=3.12-minimal-ubi
```

这说明它可能只是普通 Python 3.12 minimal 镜像。LeWM README 推荐 Python 3.10，并且训练需要 PyTorch/CUDA。第一次运行时请重点检查 Pod 日志中的这些步骤：

```text
git clone 是否成功
pip install 是否成功
torch 是否能安装并识别 GPU
stable_worldmodel 是否能 import
python train.py 是否进入 training loop
```

如果失败，优先换成 NERC 提供的 PyTorch CUDA Workbench 镜像，或者构建自定义训练镜像。

第二，修改 GitHub repo：

```yaml
env:
  - name: GIT_REPO_URL
    value: "https://github.com/Pcjhds/le-wm.git"
  - name: GIT_REF
    value: "yaalbert-yamlud-0519"
```

例如：

```yaml
env:
  - name: GIT_REPO_URL
    value: "https://github.com/Pcjhds/le-wm.git"
  - name: GIT_REF
    value: "yaalbert-yamlud-0519"
```

第三，根据 NERC quota 修改资源：

```yaml
resources:
  requests:
    cpu: "4"
    memory: "32Gi"
    nvidia.com/gpu: "1"
  limits:
    cpu: "8"
    memory: "64Gi"
    nvidia.com/gpu: "1"
```

如果你的 namespace 没有 GPU，删除 `nvidia.com/gpu`，并把 `config/train/launcher/nerc.yaml` 中 `accelerator` 改成 `cpu`。

#### 18.4.6 在 NERC 网页中启动训练

进入 OpenShift Web Console：

```text
Project / Namespace
  -> +Add 或 Import YAML
  -> 粘贴 deploy/openshift/lewm-train-from-github-job.yaml
  -> Create
```

然后查看：

```text
Workloads
  -> Pods
  -> lewm-pusht-train-from-github-...
  -> Logs
```

你应该能看到类似日志：

```text
Cloning public GitHub repository...
Installing Python dependencies...
Downloading LeWM PushT dataset from Hugging Face collection...
Decompressing dataset...
Starting LeWM training...
```

训练完成后，checkpoint 会在 PVC 中：

```text
/mnt/lewm/checkpoints/pusht/lewm/
  config.json
  weights_epoch_1.pt
  ...
```

#### 18.4.7 这种方式的优缺点

优点：

```text
不需要你先构建自定义项目镜像
可以直接通过 NERC 网页创建 Job
每次 Job 都能拉取 GitHub 上最新代码
适合前期调试
```

缺点：

```text
每次启动都要 pip install，比较慢
依赖外网稳定性
依赖基础镜像里有 git / CUDA / PyTorch
长期稳定训练不如自定义镜像可靠
```

建议：

```text
前期调试:
  用 lewm-train-from-github-job.yaml

后续稳定训练:
  构建自定义镜像，再用 lewm-train-job.yaml
```

### 18.4.8 更推荐的 GitHub + Dockerfile 方式

NERC 网页上的 `Import from Git` / `BuildConfig` 推荐方式，本质是：

```text
GitHub repo
  -> Dockerfile
  -> OpenShift BuildConfig
  -> ImageStream
  -> Job 使用这个 ImageStream 镜像训练
```

这和 `lewm-train-from-github-job.yaml` 不同：

```text
lewm-train-from-github-job.yaml:
  每次 Job 启动后 git clone + pip install

Dockerfile + BuildConfig:
  先构建一次镜像，依赖都打进镜像
  后续 Job 直接启动训练
```

更推荐后者，因为训练 Job 启动更快，也更稳定。

本仓库已经提供：

```text
Dockerfile
deploy/openshift/lewm-buildconfig.yaml
deploy/openshift/lewm-train-job.yaml
```

网页操作路线：

```text
OpenShift Web Console
  -> 你的 Project / Namespace
  -> +Add / Import YAML
  -> 导入 deploy/openshift/lewm-buildconfig.yaml
  -> Builds
  -> BuildConfigs
  -> lewm-train
  -> Start build
  -> 等待 build 成功
  -> +Add / Import YAML
  -> 导入 deploy/openshift/lewm-train-job.yaml
```

构建成功后会出现 ImageStream：

```text
lewm-train:latest
```

训练 Job 中已经使用：

```yaml
image: image-registry.openshift-image-registry.svc:5000/digital-twins-for-automated-process-f532cb/lewm-train:latest
```

如果你的 namespace 名称变化，需要把这段中的 namespace 改成当前 project 名：

```text
digital-twins-for-automated-process-f532cb
```

### 18.5 不要用 Import from Git 创建 Deployment 来跑训练

OpenShift 网页里的 `Import from Git` 默认会创建：

```text
BuildConfig
Deployment
Service
Route
```

这套流程适合 Web app，例如 Flask/FastAPI/Node 服务。它会假设你的程序是一个长期运行的服务，并且监听某个端口，例如 `8080`。

LeWM 训练不是 Web 服务，而是一次性 batch job：

```text
启动容器
下载数据
训练
保存 checkpoint
容器退出
```

因此不要在 `Import from Git` 页面里保持：

```text
Resource type = Deployment
Target port = 8080
Create a route
```

这些设置不会帮你训练模型，反而可能导致创建失败、Pod 反复重启，或者 Deployment 因为没有监听 8080 而被认为不健康。

如果你想“网页操作 + GitHub 拉代码运行”，正确做法是：

```text
OpenShift Web Console
  -> 你的 Project / Namespace
  -> +Add
  -> Import YAML
  -> 先导入 pvc.yaml
  -> 再导入 lewm-train-from-github-job.yaml
```

也就是说，使用：

```text
deploy/openshift/lewm-train-from-github-job.yaml
```

而不是 `Import from Git` 的 `Deployment` 创建页面。

`Import from Git` 可以以后用来构建自定义镜像，但构建完成后仍然建议用 `Job` 来启动训练。

## 19. 参考资料

- LeWM Hugging Face collection：https://huggingface.co/collections/quentinll/lewm
- NERC OpenShift AI Data Science Project 文档：https://nerc-project.github.io/nerc-docs/openshift-ai/data-science-project/using-projects-the-rhoai/
- NERC OpenShift AI 文档入口：https://nerc-project.github.io/nerc-docs/openshift-ai/
- Red Hat OpenShift AI Data Science Pipelines 文档：https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.16/html-single/working_with_data_science_pipelines/index
