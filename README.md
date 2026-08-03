# COLMAP RGBD GT

A Docker-based pipeline for converting rosbag recordings to metric pseudo-ground-truth trajectories using COLMAP and depth-based scaling.

## Overview

This tool creates **metric pseudo-GT trajectories** from RGBD rosbag data by:

1. Extracting RGB, depth, and camera calibration data from rosbag files
2. Running COLMAP on RGB images to estimate camera poses (up to scale)
3. Estimating metric scale from depth data
4. Exporting trajectories in TUM and CSV formats for evaluation

**Important**: This produces *pseudo-ground-truth*, not true ground truth. The trajectories are depth-scaled COLMAP reconstructions suitable as reference trajectories for SLAM/VO evaluation.

## Features

- **ROS1 and ROS2 support**: Works with both `.bag` (ROS1) and `.db3`/`.mcap` (ROS2) files
- **Pure Python bag access**: No ROS runtime dependency using `rosbags` library
- **CPU-first COLMAP**: Runs COLMAP CPU-only by default for portability
- **ROCm 7.2 compatible**: Docker setup works on AMD ROCm environments
- **Robust scale estimation**: Multiple methods (median, Umeyama, RANSAC)
- **Comprehensive diagnostics**: Scale reports, histograms, and confidence scores

## Installation

### Docker (Recommended)

```bash
docker build -t colmap-rgbd-gt -f docker/Dockerfile .
```

**Note:** `docker/Dockerfile` builds COLMAP **4.1.1 from source** (pinned tag,
not `main`) rather than installing it via `apt`. This adds real build time to
the image build — roughly 15-25 minutes for the COLMAP compile alone,
depending on core count and the `COLMAP_BUILD_JOBS` build-arg (defaults to
`nproc`; pass e.g. `--build-arg COLMAP_BUILD_JOBS=12` to cap parallelism on a
host running other CPU-heavy work concurrently). This is required for
`global_mapper` support — see "Mapper Selection" below for why that matters.
The build also fetches PoseLib, faiss, and onnxruntime automatically via
CMake `FetchContent`; no separate steps needed.

### From Source

```bash
pip install uv  # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .
```

Or with pip:
```bash
pip install -e .
```

Requires: Python 3.12+, COLMAP CLI, OpenCV

**COLMAP version:** the `global_mapper` command (this pipeline's default
mapper as of 2026-08-02, see "Mapper Selection" below) requires **COLMAP
>=4.0** — build from source (see `docker/Dockerfile` for the exact cmake
invocation and dependency list) or use a distro package new enough to include
it. Ubuntu 24.04's `apt-get install colmap` resolves to 3.9.1, which predates
the merge and does not have `global_mapper` at all (`colmap help` won't list
it) — `run-colmap` will still work with `mapper_type: incremental` on such an
install, but the default `mapper_type: global` will fail outright.

## Quick Start

```bash
gttool full data/raw/session01.bag --config configs/default.yaml
```

Output: `data/workspaces/session01/outputs/trajectory_metric_tum.txt`

## CLI Commands

| Command | Description |
|---------|-------------|
| `gttool inspect-bag <bag>` | Inspect bag and list topics |
| `gttool extract <bag>` | Extract RGB/depth/camera info |
| `gttool run-colmap <workspace>` | Run COLMAP reconstruction |
| `gttool scale-depth <workspace>` | Estimate metric scale |
| `gttool depth-ba <workspace>` | Depth-aware bundle adjustment (optional, requires `[depth-ba]` extra) |
| `gttool export-tum <workspace>` | Export TUM trajectory |
| `gttool full <bag>` | Run complete pipeline (`--depth-ba` to include bundle adjustment) |

### Depth-Aware Bundle Adjustment (optional)

Beyond post-hoc scale correction, `depth-ba` jointly refines camera poses and
sparse structure against both reprojection error and metric depth
measurements, using [`kornia-rs`](https://github.com/kornia/kornia-rs)'s
Schur-complement bundle adjuster. It runs after `scale-depth` (which supplies
a roughly-metric initialization) and is CPU-only.

```bash
pip install -e ".[depth-ba]"
gttool depth-ba data/workspaces/session01
# or as part of the full pipeline:
gttool full data/raw/session01.bag --config configs/default.yaml --depth-ba
```

Outputs: a refined sparse model under `colmap/sparse/0_refined/`,
`outputs/trajectory_depth_ba_tum.txt`, and `outputs/depth_ba_report.json`
(convergence status, observation counts).

This is an experimental, opt-in stage — `kornia-rs`'s bundle-adjustment
module is young, so it is not part of the default pipeline.

### Topic Overrides

```bash
gttool full data/raw/session01.db3 \
  --rgb /camera/color/image_raw \
  --depth /camera/aligned_depth_to_color/image_raw \
  --camera-info /camera/color/camera_info
```

## Configuration

Configuration files in `configs/`:

| File | Use Case |
|------|----------|
| `default.yaml` | General purpose |
| `realsense.yaml` | Intel RealSense cameras |
| `oakd.yaml` | OAK-D cameras |
| `tum_like.yaml` | TUM RGBD-like datasets |

### Key Settings

```yaml
sync:
  max_rgb_depth_dt_sec: 0.03  # RGB-depth sync tolerance

colmap:
  camera_model: OPENCV
  matcher: sequential
  use_gpu: false
  mapper_type: global  # global (default) or incremental -- see "Mapper Selection" below

scaling:
  method: umeyama  # median, umeyama, ransac
  max_frames_for_scale: 100
```

## Mapper Selection: global vs. incremental

`colmap:mapper_type` selects which COLMAP reconstruction algorithm
`run-colmap` (and the `run-colmap` stage inside `full`) uses. **`global` is
the default** (`ColmapRunner.run_full_pipeline`'s code-level default, not
just this repo's `configs/default.yaml`) as of 2026-08-02.

- **`global`** → `colmap global_mapper`. This is COLMAP's built-in global
  Structure-from-Motion pipeline — the functionality that used to live in the
  separate GLOMAP project (`colmap/glomap` on GitHub) before GLOMAP was
  merged into COLMAP itself and archived. It solves rotation averaging +
  global positioning + iterative bundle adjustment over the *entire* image
  set in one shot, rather than growing the reconstruction incrementally.
  Requires COLMAP ≥4.0 (see Installation above).
- **`incremental`** → `colmap mapper`, COLMAP's classic image-by-image
  incremental SfM. Available as an explicit opt-in for a scene that
  genuinely needs it (see "When incremental might still be right" below).

### Why global is the default

Real, controlled comparison from this project's own use (2026-08-02), same
already-extracted-and-matched `database.db` fed to both mappers on two
production scenes:

| Scene | Frames | Incremental mapper | Global mapper |
|---|---|---|---|
| kitchen | 2056 | 598.8 CPU-minutes (~10 hours), then gave up ("No good initial image pair found"), 3 disconnected sub-models | 48 minutes, **100% connected** (2056/2056 images, 1 model), 98564 points, 0.878px mean reprojection error |
| hallway (long corridor, real content difficulty — Open3D odometry fitness degraded 0.86→0.37 across the trajectory) | 5139 | Fragmented into 5 disconnected sub-models even after retuning matching (wider `sequential_overlap`, `loop_detection` enabled) — largest held only ~518 frames (~10%) | Reconstructed as a single connected model |

No scene tested during this comparison did better with incremental than
global.

**Why global tends to win on difficult/long sequences:** incremental SfM
grows the model image-by-image, via repeated partial re-triangulation and
*local* bundle adjustment. A single weakly-connected segment (motion blur,
low parallax, a textureless stretch) can break the growing chain — COLMAP
then either re-seeds a new, disconnected sub-model from a different starting
pair, or, in a worse case, fails to find any good re-seed pair at all and
just stops (kitchen scene above). Global SfM instead sees the whole problem
at once and solves it as one joint optimization, so a locally weak segment
gets compensated by consistency constraints from the rest of the image set
rather than causing a hard break in the reconstruction.

### When incremental might still be right

Global reconstruction methods can, in some cases, be more sensitive to a bad
match or outlier propagating *globally* through the joint optimization,
rather than staying locally contained the way it would in an incremental
pipeline. Incremental mapper is kept as a non-default option
(`mapper_type: incremental`) for exactly this reason — if a future scene
shows the opposite pattern (global mapper produces a worse or fragmented
result, incremental succeeds cleanly), that is a legitimate, scene-specific
reason to override the default, not a sign that something is broken or that
the default choice was wrong in general.

## Workspace Structure

```
workspace/
├── manifest.json
├── rgb/
│   ├── 000000.png
│   └── ...
├── depth/
│   ├── 000000.png
│   └── ...
├── camera/
│   ├── intrinsics.json
│   └── distortion.json
├── timestamps/
│   ├── rgb.csv
│   ├── depth.csv
│   └── associations.csv
├── colmap/
│   ├── database.db
│   └── sparse/
└── outputs/
    ├── trajectory_colmap_unscaled.txt
    ├── trajectory_metric_tum.txt
    └── scale_report.json
```

## Scale Estimation Methods

1. **median**: Simple ratio of COLMAP to depth point norms
2. **umeyama**: Umeyama alignment with scale (default)
3. **ransac**: RANSAC-based robust estimation

The pipeline computes per-frame scale estimates and aggregates them robustly.

## Supported ROS Topics

- `sensor_msgs/Image` (raw RGB/depth)
- `sensor_msgs/CompressedImage`
- `sensor_msgs/CameraInfo`
- `sensor_msgs/PointCloud2` (fallback)

## Docker Usage

The image built from `docker/Dockerfile` includes a from-source COLMAP 4.1.1
build with `global_mapper` support — see "Installation" above for build-time
notes and "Mapper Selection" above for why that's the default reconstruction
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

## Output Formats

### TUM RGBD Format

```
timestamp tx ty tz qx qy qz qw
```

### CSV Format

Additional columns: frame_id, rotation matrix, camera center

## Evaluation with evo

`trajectory_metric_tum.txt` is already metric (a scale factor was applied during the
`scale-depth` step). Evaluate it with rigid SE(3) alignment only — do **not** pass
`--correct_scale`, since that would silently re-optimize scale during evaluation and
mask a broken scale estimate:

```bash
evo_ape tum groundtruth.txt trajectory_metric_tum.txt -va --align
```

To separately quantify how far off the recovered scale was, run the same comparison
a second time with `--correct_scale` and compare the reported scale correction factor
against 1.0 (`|reported_scale_correction - 1.0|` is the scale error):

```bash
evo_ape tum groundtruth.txt trajectory_metric_tum.txt -va --align --correct_scale
```

## Limitations

- COLMAP runs CPU-only (not GPU-accelerated)
- Requires sufficient texture for COLMAP
- Scale accuracy depends on depth quality
- Not suitable for real-time operation

## License

MIT License

## Acknowledgments

- [COLMAP](https://colmap.github.io/) - Structure-from-Motion pipeline
- [rosbags](https://pypi.org/project/rosbags/) - Pure Python rosbag access
- [evo](https://github.com/MichaelGrupp/evo) - Trajectory evaluation tools
