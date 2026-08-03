# Quick Start

```bash
gttool full data/raw/session01.bag --config configs/default.yaml
```

Output: `data/workspaces/session01/outputs/trajectory_metric_tum.txt`

## Topic Overrides

```bash
gttool full data/raw/session01.db3 \
  --rgb /camera/color/image_raw \
  --depth /camera/aligned_depth_to_color/image_raw \
  --camera-info /camera/color/camera_info
```

See the [CLI Reference](cli-reference.md) for every command and the
[Configuration](configuration.md) page for what goes in `configs/*.yaml`.
