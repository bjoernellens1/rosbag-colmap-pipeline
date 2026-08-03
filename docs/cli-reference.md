# CLI Reference

| Command | Description |
|---------|-------------|
| `gttool inspect-bag <bag>` | Inspect bag and list topics |
| `gttool extract <bag>` | Extract RGB/depth/camera info |
| `gttool run-colmap <workspace>` | Run COLMAP reconstruction |
| `gttool scale-depth <workspace>` | Estimate metric scale |
| `gttool depth-ba <workspace>` | Depth-aware bundle adjustment (optional, requires `[depth-ba]` extra) |
| `gttool export-tum <workspace>` | Export TUM trajectory |
| `gttool full <bag>` | Run complete pipeline (`--depth-ba` to include bundle adjustment) |

`extract` and `full` default `--workspace` to `<repo_root>/data/workspaces/<bag-stem>`
when `-w`/`--workspace` is not given, regardless of the invocation cwd or where the
bag file itself lives — see [Workspace Structure](workspace-structure.md).

## Topic overrides

```bash
gttool full data/raw/session01.db3 \
  --rgb /camera/color/image_raw \
  --depth /camera/aligned_depth_to_color/image_raw \
  --camera-info /camera/color/camera_info
```

## Depth-Aware Bundle Adjustment (optional)

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

This is an experimental, opt-in stage — see [Depth-Aware Bundle Adjustment](depth-ba.md)
for current status (as of this writing it has not yet been validated on real data;
`depth_ba.enabled: false` everywhere).
