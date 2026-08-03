# Installation

## Docker (Recommended)

```bash
docker build -t colmap-rgbd-gt -f docker/Dockerfile .
```

!!! note
    `docker/Dockerfile` builds COLMAP **4.1.1 from source** (pinned tag,
    not `main`) rather than installing it via `apt`. This adds real build time to
    the image build — roughly 15-25 minutes for the COLMAP compile alone,
    depending on core count and the `COLMAP_BUILD_JOBS` build-arg (defaults to
    `nproc`; pass e.g. `--build-arg COLMAP_BUILD_JOBS=12` to cap parallelism on a
    host running other CPU-heavy work concurrently). This is required for
    `global_mapper` support — see [Mapper Selection](mapper-selection.md) for why that matters.
    The build also fetches PoseLib, faiss, and onnxruntime automatically via
    CMake `FetchContent`; no separate steps needed.

A prebuilt image is also published to `ghcr.io/bjoernellens1/colmap-rgbd-gt:latest`
on every push to `main` (see the repo's `.github/workflows/build-image.yml`) —
`docker compose -f docker/docker-compose.yml run gttool ...` pulls it automatically
rather than rebuilding locally.

## From Source

```bash
pip install uv  # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .
```

Or with pip:
```bash
pip install -e .
```

Requires: Python 3.12+, COLMAP CLI, OpenCV

!!! warning "COLMAP version"
    The `global_mapper` command (this pipeline's default
    mapper as of 2026-08-02, see [Mapper Selection](mapper-selection.md)) requires **COLMAP
    >=4.0** — build from source (see `docker/Dockerfile` for the exact cmake
    invocation and dependency list) or use a distro package new enough to include
    it. Ubuntu 24.04's `apt-get install colmap` resolves to 3.9.1, which predates
    the merge and does not have `global_mapper` at all (`colmap help` won't list
    it) — `run-colmap` will still work with `mapper_type: incremental` on such an
    install, but the default `mapper_type: global` will fail outright.

## Docker Usage

The image built from `docker/Dockerfile` includes a from-source COLMAP 4.1.1
build with `global_mapper` support — see the note above for build-time
notes and [Mapper Selection](mapper-selection.md) for why that's the default reconstruction
mode.

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  colmap-rgbd-gt \
  full /app/data/raw/session01.bag --config /app/configs/default.yaml
```

Development image with ROCm 7.2 compatibility:

```bash
docker compose -f docker/docker-compose.yml run dev
```
