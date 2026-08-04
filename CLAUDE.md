# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`colmap-rgbd-gt` (CLI: `gttool`) converts RGBD rosbag recordings into metric pseudo-ground-truth
trajectories: extract RGB/depth/calibration from a rosbag → run COLMAP SfM on RGB → estimate
metric scale from depth → export TUM/CSV trajectories for SLAM/VO evaluation. Output is
*pseudo*-GT (depth-scaled COLMAP reconstruction), not true ground truth.

Full docs: https://bjoernellens1.github.io/rosbag-colmap-pipeline (mkdocs, source in `docs/`).

## Commands

```bash
# Setup
uv pip install -e ".[dev]"

# Lint / format / typecheck (CI equivalent — no Makefile in this repo, run directly)
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run mypy src/

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_scale_estimation.py::test_umeyama_3D -v
uv run pytest tests/ -k "scale" -v
uv run pytest tests/ --cov=colmap_rgbd_gt --cov-report=term-missing

# Tests requiring COLMAP binary (run in Docker)
cd docker && podman compose run --rm dev pytest tests/ -v

# CLI
uv run gttool inspect-bag /path/to/bag.bag
uv run gttool full /path/to/bag.bag --config configs/default.yaml --workspace /path/to/workspace
uv run gttool extract /path/to/bag.bag --workspace /path/to/workspace
uv run gttool run-colmap /path/to/workspace
uv run gttool scale-depth /path/to/workspace
```

Docker is the primary supported way to run the full pipeline (COLMAP is a native binary
dependency, not pip-installable): `docker build -t colmap-rgbd-gt -f docker/Dockerfile .`
A prebuilt image is published to `ghcr.io/bjoernellens1/colmap-rgbd-gt:latest` on every push
to `main`; `docker compose -f docker/docker-compose.yml run gttool ...` pulls it automatically.

## Architecture

All source under `src/colmap_rgbd_gt/`, organized as a pipeline of stages, each stage owning
one directory:

- `ingest/` — rosbag reading (`bag_reader.py`), topic discovery, image/depth decode, camera
  info extraction, keyframe selection. Uses `rosbags>=0.11` (see API notes below).
- `preprocessing/` — decompression of compressed image/depth topics.
- `rectify/` — camera undistortion.
- `dataset/` — the canonical on-disk **Workspace** format that stages communicate through
  (`schema.py` defines `Workspace`/`WorkspaceLayout`; `manifest.py` for the frame manifest;
  `synchronization.py` for RGB/depth timestamp alignment).
- `colmap/` — COLMAP invocation and result handling: `runner.py` drives the COLMAP binary,
  `database.py`/`colmap_io.py` read/write COLMAP's DB and text formats, `reconstruction.py`,
  `pose_extract.py`, `pose_outliers.py`, `scale_regime_correction.py`.
- `scaling/` — metric scale estimation from depth vs. COLMAP's up-to-scale poses: backprojects
  depth into 3D, finds correspondences, Umeyama alignment (`scale_estimation.py`), robust
  statistics to reject outliers, `diagnostics.py` for QA output.
- `optimization/` — optional depth-aware bundle adjustment (`depth_ba.py`, needs the
  `depth-ba` extra / `kornia-rs`).
- `export/` — output writers: `tum.py`, `csv.py`, `evo.py` (evo-format trajectories),
  `report.py`, `scene_metadata.py`, `rosbag_writer.py`.
- `pipelines/` — orchestrates the above into CLI-facing stages: `extract_only.py`,
  `colmap_only.py`, `scale_only.py`, `depth_ba_pipeline.py`, `export_bag_pipeline.py`, and
  `full_pipeline.py` which chains extract → colmap → scale → (optional) depth-BA in sequence,
  short-circuiting on stage failure (each stage function returns `bool`).
- `cli.py` — Typer app (`gttool`) exposing each pipeline stage as a subcommand plus `full`.
  Workspace path defaults to `<repo_root>/data/workspaces/<bag-stem>` (`_default_workspace` /
  `_repo_root` in `cli.py`), resolved by walking up from `cli.py`'s own location to find
  `pyproject.toml` — this makes the default workspace location independent of invocation cwd,
  both in a dev checkout and inside the container (where it lands on `/app`).

### The Workspace

Everything downstream of ingestion communicates through a `Workspace` directory
(`dataset/schema.py`): `manifest.json`, `rgb/`, `depth/`, `camera/`, `timestamps/`, `colmap/`,
`outputs/`. Real workspaces live under `data/workspaces/<scene>/`. Stages read/write this
layout rather than passing data in-process, so a stage can be re-run standalone against an
existing workspace (`run-colmap`, `scale-depth`, etc.) without repeating extraction.

Configuration flows through the pipeline as `dict[str, Any]` loaded from YAML (`configs/`);
access nested keys with `.get(...)` defaults rather than assuming presence.

## ROS Bags API (rosbags>=0.11)

```python
from rosbags.typesys.stores import Stores, get_typestore
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__Image as ImageV1

typestore = get_typestore(Stores.ROS1_NOETIC)  # ROS1
typestore = get_typestore(Stores.ROS2_HUMBLE)  # ROS2

msg = typestore.deserialize_ros1(rawdata, msgtype)  # ROS1
msg = typestore.deserialize_cdr(rawdata, msgtype)   # ROS2
```

## Code Style

- Python 3.12+, line length 100, 4-space indent, double-quoted strings.
- All function params/returns require type annotations (`mypy: disallow_untyped_defs = true`).
  Use `X | None` / `str | Path` union syntax, not `Optional`/`Union`.
- Import order: stdlib → third-party → `colmap_rgbd_gt.*` local, blank line between groups.
- Google-style docstrings with Args/Returns/Raises.
- Structured logging via `get_logger(__name__)` from `colmap_rgbd_gt.logging`, never `print`.
  Let exceptions propagate from internal functions; catch only at boundary layers (never bare
  `except:` or silent `except Exception: pass`).
- Naming: modules/functions/variables `snake_case`, classes/type-aliases `PascalCase`,
  constants `UPPER_SNAKE`, private members `_leading_underscore`.
- Always use `pathlib.Path`, joined with `/`, not string paths.
- Tests in `tests/test_*.py`; mock rosbag files and the COLMAP binary; focus assertions on
  transformation logic rather than I/O; tests needing the real COLMAP binary run in Docker
  (see Commands above).
