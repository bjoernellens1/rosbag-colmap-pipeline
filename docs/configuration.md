# Configuration

Configuration files live in `configs/`:

| File | Use Case |
|------|----------|
| `default.yaml` | General purpose |
| `realsense.yaml` | Intel RealSense cameras |
| `oakd.yaml` | OAK-D cameras |
| `orbbec.yaml` | Orbbec Femto Mega/Bolt cameras |
| `tum_like.yaml` | TUM RGBD-like datasets |

## Key settings

```yaml
sync:
  max_rgb_depth_dt_sec: 0.03  # RGB-depth sync tolerance

colmap:
  camera_model: OPENCV
  matcher: sequential
  use_gpu: false
  mapper_type: global  # global (default) or incremental -- see mapper-selection.md

scaling:
  method: umeyama  # median, umeyama, ransac
  max_frames_for_scale: 100
  camera_fallback_profile: null  # e.g. orbbec_femto_mega -- see "Camera intrinsics fallback" below

depth_ba:
  enabled: true  # on by default; pass --no-depth-ba to gttool full to skip
  camera_fallback_profile: null
```

## Camera intrinsics fallback

COLMAP feature extraction, scale estimation, and depth-BA all prefer the
bag's own `camera_info` topic when present. If it's missing, or fails a
plausibility check (e.g. an unpopulated identity `K` matrix — a common
driver-stub artifact before real calibration loads), each of those stages
falls back to a named, hardcoded factory-calibration profile instead of
silently reprojecting with garbage intrinsics (which otherwise manifests
as scale estimation finding zero correspondences).

Set `camera_fallback_profile` under `colmap:`, `scaling:`, or `depth_ba:`
to enable this (it's opt-in per stage, `null`/unset by default — no
fallback is applied unless configured):

```yaml
colmap:
  camera_fallback_profile: orbbec_femto_mega
scaling:
  camera_fallback_profile: orbbec_femto_mega
depth_ba:
  camera_fallback_profile: orbbec_femto_mega
```

Available profiles (`KNOWN_CAMERA_PROFILES` in
`src/colmap_rgbd_gt/ingest/camera_info.py`), all sourced from this
project's own real per-unit calibration configs, not invented:

| Profile | Source |
|---------|--------|
| `orbbec_femto_bolt` | splatograph's ORB-SLAM3 `OrbbecFemtoBolt_RGBD.yaml` |
| `orbbec_femto_mega` | same sensor/lens as the Bolt profile above |
| `realsense_d435i` | `ros2-jazzy-realsense-fedora`'s `pyslam_realsense_d435i.yaml` |

The bag's own valid `camera_info` always takes priority — these are a
documented last resort for a specific known camera, not a general-purpose
default.

## Scale estimation methods

1. **median**: Simple ratio of COLMAP to depth point norms
2. **umeyama**: Umeyama alignment with scale (default)
3. **ransac**: RANSAC-based robust estimation

The pipeline computes per-frame scale estimates and aggregates them robustly.

## Supported ROS topics

- `sensor_msgs/Image` (raw RGB/depth)
- `sensor_msgs/CompressedImage`
- `sensor_msgs/CameraInfo`
- `sensor_msgs/PointCloud2` (fallback)
