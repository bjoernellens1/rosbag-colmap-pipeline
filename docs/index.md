# COLMAP RGBD GT

A Docker-based pipeline for converting rosbag recordings to metric pseudo-ground-truth trajectories using COLMAP and depth-based scaling.

## Overview

This tool creates **metric pseudo-GT trajectories** from RGBD rosbag data by:

1. Extracting RGB, depth, and camera calibration data from rosbag files
2. Running COLMAP on RGB images to estimate camera poses (up to scale)
3. Estimating metric scale from depth data
4. Exporting trajectories in TUM and CSV formats for evaluation

!!! important
    This produces *pseudo-ground-truth*, not true ground truth. The trajectories are depth-scaled COLMAP reconstructions suitable as reference trajectories for SLAM/VO evaluation.

## Features

- **ROS1 and ROS2 support**: Works with both `.bag` (ROS1) and `.db3`/`.mcap` (ROS2) files
- **Pure Python bag access**: No ROS runtime dependency using `rosbags` library
- **CPU-first COLMAP**: Runs COLMAP CPU-only by default for portability
- **ROCm 7.2 compatible**: Docker setup works on AMD ROCm environments
- **Robust scale estimation**: Multiple methods (median, Umeyama, RANSAC)
- **Comprehensive diagnostics**: Scale reports, histograms, and confidence scores
- **Global SfM by default**: reconstructs via COLMAP's `global_mapper`, dramatically more robust than incremental SfM on long/difficult sequences — see [Mapper Selection](mapper-selection.md)

## Where to go next

- [Installation](installation.md) — Docker (recommended) or from-source setup
- [Quick Start](usage.md) — first run in under a minute
- [CLI Reference](cli-reference.md) — every `gttool` command
- [Mapper Selection](mapper-selection.md) — why `global_mapper` is the default, with real numbers
- [Depth-Aware Bundle Adjustment](depth-ba.md) — optional refinement stage
- [Workspace Structure](workspace-structure.md) — what gets written where
- [Output Formats & Evaluation](output-formats.md) — TUM/CSV formats, `evo` usage
- [Limitations](limitations.md)

## Acknowledgments

- [COLMAP](https://colmap.github.io/) — Structure-from-Motion pipeline
- [rosbags](https://pypi.org/project/rosbags/) — Pure Python rosbag access
- [evo](https://github.com/MichaelGrupp/evo) — Trajectory evaluation tools

## License

MIT License
