FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /workspace/LongCat-Video
COPY . /workspace/LongCat-Video

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --dry-run --no-deps -r requirements.runpod.txt && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 && \
    python3 -m pip install ninja packaging psutil==6.0.0 && \
    python3 -m pip install flash-attn==2.7.4.post1 --no-build-isolation && \
    python3 -m pip install -r requirements.runpod.txt

CMD ["python3", "-u", "handler.py"]
