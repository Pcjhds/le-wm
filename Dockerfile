FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    STABLEWM_HOME=/workspace/stablewm \
    PYTHONPATH=/workspace/le-wm \
    WANDB_MODE=offline \
    MUJOCO_GL=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libegl1 \
    libglew2.2 \
    libglfw3 \
    libglib2.0-0 \
    libgl1 \
    libhdf5-dev \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    pkg-config \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
    "stable-worldmodel[train,env]" \
    hydra-core \
    lightning \
    wandb \
    torchvision \
    scikit-learn \
    einops

WORKDIR /workspace/le-wm

COPY . /workspace/le-wm

RUN mkdir -p /workspace/stablewm \
    && chgrp -R 0 /workspace \
    && chmod -R g=u /workspace

CMD ["python", "-c", "import torch, hydra, lightning, stable_worldmodel, stable_pretraining; print('LeWM container smoke test OK')"]
