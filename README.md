# COLMAP RGBD GT

A Docker-based pipeline for converting rosbag recordings to metric pseudo-ground-truth trajectories using COLMAP and depth-based scaling.

**Full documentation:** https://bjoernellens1.github.io/rosbag-colmap-pipeline

## Overview

This tool creates **metric pseudo-GT trajectories** from RGBD rosbag data by:

1. Extracting RGB, depth, and camera calibration data from rosbag files
2. Running COLMAP on RGB images to estimate camera poses (up to scale)
3. Estimating metric scale from depth data
4. Exporting trajectories in TUM and CSV formats for evaluation

**Important**: This produces *pseudo-ground-truth*, not true ground truth. The trajectories are depth-scaled COLMAP reconstructions suitable as reference trajectories for SLAM/VO evaluation. See [Limitations](https://bjoernellens1.github.io/rosbag-colmap-pipeline/limitations/).

## Quick Start

```bash
docker build -t colmap-rgbd-gt -f docker/Dockerfile .
gttool full data/raw/session01.bag --config configs/default.yaml
```

Output: `data/workspaces/session01/outputs/trajectory_metric_tum.txt`

A prebuilt image is also published to `ghcr.io/bjoernellens1/colmap-rgbd-gt:latest`
on every push to `main` — `docker compose -f docker/docker-compose.yml run gttool ...`
pulls it automatically rather than rebuilding locally.

See the docs for full details:

- [Installation](https://bjoernellens1.github.io/rosbag-colmap-pipeline/installation/) — Docker (recommended) or from-source setup
- [Usage / Quick Start](https://bjoernellens1.github.io/rosbag-colmap-pipeline/usage/)
- [CLI Reference](https://bjoernellens1.github.io/rosbag-colmap-pipeline/cli-reference/)
- [Configuration](https://bjoernellens1.github.io/rosbag-colmap-pipeline/configuration/)
- [Mapper Selection (global vs. incremental)](https://bjoernellens1.github.io/rosbag-colmap-pipeline/mapper-selection/) — why `global_mapper` is the default, with real numbers
- [Depth-Aware Bundle Adjustment](https://bjoernellens1.github.io/rosbag-colmap-pipeline/depth-ba/)
- [Workspace Structure](https://bjoernellens1.github.io/rosbag-colmap-pipeline/workspace-structure/)
- [Output Formats & Evaluation](https://bjoernellens1.github.io/rosbag-colmap-pipeline/output-formats/)
- [Limitations](https://bjoernellens1.github.io/rosbag-colmap-pipeline/limitations/)

## License

MIT License

## Acknowledgments

- [COLMAP](https://colmap.github.io/) - Structure-from-Motion pipeline
- [rosbags](https://pypi.org/project/rosbags/) - Pure Python rosbag access
- [evo](https://github.com/MichaelGrupp/evo) - Trajectory evaluation tools
