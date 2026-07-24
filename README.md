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

scaling:
  method: umeyama  # median, umeyama, ransac
  max_frames_for_scale: 100
```

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
