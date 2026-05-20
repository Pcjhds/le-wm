FROM registry.access.redhat.com/ubi9/python-311:latest

USER 0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    HOME=/opt/app-root/src \
    STABLEWM_HOME=/mnt/lewm \
    LOCAL_DATASET_DIR=/mnt/lewm \
    HF_HOME=/mnt/lewm/hf-cache \
    SETUPTOOLS_USE_DISTUTILS=stdlib

RUN dnf install -y \
    gcc \
    gcc-c++ \
    git \
    glib2 \
    libSM \
    libXext \
    libXrender \
    make \
    mesa-libEGL \
    mesa-libGL \
    zstd \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN python -m pip install --upgrade "pip<24" "setuptools==65.5.0" "wheel==0.38.4" && \
    python -m pip install "swig==4.1.1.post0" && \
    python -m pip install "gym==0.21.0" --no-build-isolation && \
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
    chgrp -R 0 /opt/app-root/src /mnt/lewm && \
    chmod -R g=u /opt/app-root/src /mnt/lewm

USER 1001

CMD ["python", "train.py", "data=hf_pusht", "launcher=nerc", "output_model_name=pusht/lewm"]
