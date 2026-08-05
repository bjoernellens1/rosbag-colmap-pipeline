# HIP-enabled COLMAP build for local AMD GPU dispatch (gfx1151, Radeon 8060S)
# -- alongside docker/Dockerfile.cuda (A100 cluster dispatch, untouched) and
# docker/Dockerfile (plain apt-get colmap, CPU-only). Same ROCm/PyTorch base
# as docker/Dockerfile, but builds COLMAP from source with HIP_ENABLED +
# CASPAR_ENABLED instead of `apt-get install colmap` (which is CPU-only and,
# per docker/Dockerfile's own history, predates GLOMAP/global_mapper anyway).
#
# COLMAP source comes from docker/colmap-rocm-hip-src/ -- a plain copy (not a
# git submodule) of bjoernellens1/colmap's `hip-integration` branch at commit
# 0b079c55 ("docs: verify Caspar-HIP OpenCV distortion model against real TUM
# fr1 data"), the exact tested state this session's colmap-rocm verification
# work landed on: HIP-accelerated SiftGPU (feature extraction/matching),
# patch_match_stereo, and Caspar bundle adjustment (native OpenCV
# camera-model support included), all verified end-to-end on real data on
# this host's gfx1151 GPU. Copied rather than `git clone`d at build time
# because this exact worktree state is the one that was actually tested --
# cloning the branch at build time would pick up whatever HEAD happens to be
# if anyone pushes more commits later. Re-sync with:
#   rsync -a --exclude='.git' --exclude=build <worktree>/ docker/colmap-rocm-hip-src/
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ARG ROCM_ARCH=gfx1151

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Same dependency list as docker/Dockerfile (the base image is optimized for
# PyTorch/ROCm, not general C++ builds -- COLMAP's build/runtime deps are not
# guaranteed present). Keeping libceres-dev: Caspar only accelerates the BA
# hot path (per docker/Dockerfile.cuda's comment on the identical CUDA case),
# COLMAP's other Ceres-backed code paths (feature-matching geometry, non-BA
# estimators) still need it.
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    cmake \
    ninja-build \
    build-essential \
    libeigen3-dev \
    libopencv-dev \
    libsqlite3-dev \
    libboost-all-dev \
    libceres-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgtest-dev \
    libgmock-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libcurl4-openssl-dev \
    libopenimageio-dev \
    openimageio-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --system-site-packages so this project's venv can still see the base
# image's ROCm-enabled PyTorch install (same rationale as docker/Dockerfile).
RUN python3.12 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --upgrade pip setuptools wheel

# COLMAP built from source with HIP + Caspar (GPU bundle adjustment, native
# OpenCV camera-model support, no image preprocessing needed -- same reasoning
# as docker/Dockerfile.cuda's Caspar section, just on colmap-rocm's HIP port
# instead of upstream CUDA). CMAKE_HIP_ARCHITECTURES pinned to gfx1151
# (ROCM_ARCH build arg) -- this host's actual GPU, not autodetected (mirrors
# colmap-rocm's own top-level Dockerfile).
ARG COLMAP_BUILD_JOBS=
COPY docker/colmap-rocm-hip-src /opt/colmap_src
RUN mkdir -p /opt/colmap_src/build \
    && cd /opt/colmap_src/build \
    && cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DHIP_ENABLED=ON \
        -DCUDA_ENABLED=OFF \
        -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH} \
        -DCASPAR_ENABLED=ON \
    && ninja -j "${COLMAP_BUILD_JOBS:-$(nproc)}" \
    && ninja install \
    && rm -rf /opt/colmap_src

# Same vocab tree as docker/Dockerfile / docker/Dockerfile.cuda.
RUN mkdir -p /opt/colmap_vocab_tree && \
    curl -fsSL -o /opt/colmap_vocab_tree/vocab_tree_flickr100K_words32K.bin \
    https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words32K.bin

WORKDIR /app

COPY requirements/base.txt /app/requirements/
RUN uv pip install -r requirements/base.txt

COPY pyproject.toml /app/
COPY configs/ /app/configs/
COPY src/ /app/src/
RUN uv pip install -e ".[depth-ba]"

LABEL org.opencontainers.image.source="https://github.com/bjoernellens1/rosbag-colmap-pipeline" \
      org.opencontainers.image.description="HIP-enabled COLMAP build for local AMD GPU dispatch (gfx1151)" \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["gttool"]
CMD ["--help"]
