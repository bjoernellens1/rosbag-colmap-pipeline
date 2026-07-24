# ROCm 7.2 + PyTorch base (Ubuntu 24.04, Python 3.12) -- reused so this
# image inherits a working ROCm/HIP + PyTorch stack for future GPU-accelerated
# experimentation (e.g. a PyTorch/Kornia depth-BA backend), even though the
# default depth-ba stage (kornia-rs) is CPU-only today.
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# The base image is optimized for PyTorch/ROCm, not general C++ builds --
# COLMAP's build/runtime deps are not guaranteed present, so install them
# explicitly regardless of what the base ships.
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    cmake \
    build-essential \
    libeigen3-dev \
    libopencv-dev \
    libsqlite3-dev \
    libboost-all-dev \
    libceres-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --system-site-packages so this project's venv can still see the base
# image's ROCm-enabled PyTorch install, while keeping this project's own
# dependencies isolated from the base image's site-packages.
RUN python3.12 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --upgrade pip setuptools wheel

RUN apt-get update && apt-get install -y colmap && rm -rf /var/lib/apt/lists/*

# Pretrained vocab tree for COLMAP's sequential-matcher loop detection --
# reconnects frames whose overlap with their immediate temporal neighbors
# has dropped (fast motion, low-texture stretches) by re-matching against
# visually-similar past frames, not just a linear window. Baked in at
# build time so no runtime network access is needed. 32K-word tree: sized
# for hundreds-to-low-thousands of images, matching this pipeline's scale.
RUN mkdir -p /opt/colmap_vocab_tree && \
    curl -fsSL -o /opt/colmap_vocab_tree/vocab_tree_flickr100K_words32K.bin \
    https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words32K.bin

WORKDIR /app

COPY requirements/base.txt /app/requirements/
RUN uv pip install -r requirements/base.txt

COPY pyproject.toml /app/
COPY configs/ /app/configs/
COPY src/ /app/src/
RUN uv pip install -e ".[depth-ba]"

ENTRYPOINT ["gttool"]
CMD ["--help"]
