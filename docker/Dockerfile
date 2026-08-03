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
# image's ROCm-enabled PyTorch install, while keeping this project's own
# dependencies isolated from the base image's site-packages.
RUN python3.12 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --upgrade pip setuptools wheel

# COLMAP built from source, PINNED to release tag 4.1.1 (not `main` --
# reproducible builds, not "whatever HEAD happened to be").
#
# 2026-08-02: was `apt-get install -y colmap`, which on Ubuntu 24.04 (noble)
# resolves to 3.9.1 -- predates COLMAP's merge of the GLOMAP global-SfM
# pipeline (`global_mapper` / `automatic_reconstructor --mapper GLOBAL`,
# landed in 4.0.0). Rebuilt from source after a real production incident
# this session: `colmap mapper` (incremental SfM) on a 2056-frame kitchen
# scene ran for 10+ CPU-hours without finishing, showing genuine numerical
# ill-conditioning (a bundle-adjustment round with cost frozen across 13+
# iterations after a "Matrix not positive definite" CHOLMOD warning, not
# just "big job, needs more time"). `global_mapper` on the IDENTICAL
# database.db reconstructed the SAME scene -- 100% of frames registered
# into ONE connected model (no fragmentation, unlike this session's earlier
# hallway-scene incremental-mapper run which split into 5 disconnected
# models with only ~10% of frames in the largest one) -- in nine 2894.87s
# (~48 minutes) end-to-end, reusing the exact same feature_extractor +
# sequential_matcher stages/database format (GLOMAP's global positioning +
# bundle adjustment reads/writes the same COLMAP database and sparse-model
# format, confirmed empirically, not just per upstream docs).
#
# Requires OpenImageIO (new hard dependency in 4.x, absent in 3.9.1's build
# graph) -- libopenimageio-dev alone is insufficient, its CMake config
# references /usr/bin/iconvert which only openimageio-tools installs; found
# by a real configure-time failure, not assumed. CUDA disabled (no GPU
# passthrough assumed for this pipeline's default CPU-first design; ROCm/HIP
# is a separate, unrelated toolchain COLMAP itself doesn't use). GUI/Qt5
# deps kept even though this pipeline never launches the GUI, because
# COLMAP's CMake unconditionally links colmap_ui into the `colmap` binary
# unless GUI_ENABLED=OFF is passed; not itself disabled here to keep this
# build path closer to a stock invocation (revisit if image size becomes a
# real problem later).
# COLMAP_BUILD_JOBS: defaults to nproc, but override at build time
# (`--build-arg COLMAP_BUILD_JOBS=12`) to avoid starving other CPU-heavy
# jobs sharing the same host during the build -- this compile is
# genuinely heavy (~300 translation units including PoseLib/faiss/
# onnxruntime fetched via CMake FetchContent).
ARG COLMAP_BUILD_JOBS=
RUN git clone --depth 1 --branch 4.1.1 https://github.com/colmap/colmap.git /opt/colmap_src \
    && mkdir -p /opt/colmap_src/build \
    && cd /opt/colmap_src/build \
    && cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=OFF \
    && ninja -j "${COLMAP_BUILD_JOBS:-$(nproc)}" \
    && ninja install \
    && rm -rf /opt/colmap_src

# Pretrained vocab tree for COLMAP's sequential-matcher loop detection --
# reconnects frames whose overlap with their immediate temporal neighbors
# has dropped (fast motion, low-texture stretches) by re-matching against
# visually-similar past frames, not just a linear window. Baked in at
# build time so no runtime network access is needed. 32K-word tree: sized
# for hundreds-to-low-thousands of images, matching this pipeline's scale.
#
# FIXED 2026-08-02: was vocab_tree_flickr100K_words32K.bin (no "faiss" in
# the name) -- COLMAP switched its vocab-tree index format from flann to
# faiss in May 2025; the pre-May-2025-named asset is the OLD flann-format
# file (still hosted for backward compat), which now crashes 4.1.1's
# reader outright: "Check failed: file_version == 1 || file_version == 2
# ... This may be caused by reading a legacy flann-based index." Found via
# a real `run-colmap` crash, not assumed from the changelog. Same release
# (3.11.1) also hosts the faiss-format file under a distinctly-prefixed
# name -- use that one.
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

ENTRYPOINT ["gttool"]
CMD ["--help"]
