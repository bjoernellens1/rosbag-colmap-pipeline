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
```

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
