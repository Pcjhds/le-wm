FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    HOME=/opt/app-root/src \
    STABLEWM_HOME=/mnt/lewm \
    LOCAL_DATASET_DIR=/mnt/lewm \
    HF_HOME=/mnt/lewm/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install \
      hydra-core \
      lightning \
      einops \
      huggingface_hub \
      zstandard \
      "stable-worldmodel[train,env]"

WORKDIR /opt/app-root/src
COPY . .

RUN mkdir -p /opt/app-root/src /mnt/lewm && \
    chgrp -R 0 /opt/app-root/src /opt/venv /mnt/lewm && \
    chmod -R g=u /opt/app-root/src /opt/venv /mnt/lewm

USER 1001

CMD ["python", "train.py", "data=hf_pusht", "launcher=nerc", "output_model_name=pusht/lewm"]
