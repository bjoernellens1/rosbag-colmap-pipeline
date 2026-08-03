# CLI Reference

| Command | Description |
|---------|-------------|
| `gttool inspect-bag <bag>` | Inspect bag and list topics |
| `gttool extract <bag>` | Extract RGB/depth/camera info |
| `gttool run-colmap <workspace>` | Run COLMAP reconstruction |
| `gttool scale-depth <workspace>` | Estimate metric scale |
| `gttool depth-ba <workspace>` | Depth-aware bundle adjustment (requires `[depth-ba]` extra; on by default in `configs/default.yaml`) |
| `gttool export-tum <workspace>` | Export TUM trajectory |
| `gttool export-bag <bag> -w <workspace>` | Write a `_processed` copy of the bag with the GT trajectory added as new topics |
| `gttool full <bag>` | Run complete pipeline (`--no-depth-ba` to skip bundle adjustment) |

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

## Export a processed bag with GT trajectory

`export-bag` writes a copy of the original bag — every original message
copied verbatim (raw bytes, same QoS), plus the estimated metric GT
trajectory added as two new topics — so the trajectory can be consumed
directly by anything that already plays back the original bag (rviz2,
foxglove, splatograph, etc.) instead of requiring a separate TUM file kept
in sync by hand.

```bash
gttool export-bag data/raw/session01.bag --workspace data/workspaces/session01
```

Requires a workspace with a completed `scale-depth` run (it reads
`outputs/scale_report.json` for the metric scale and re-derives the
trajectory from the COLMAP sparse model). By default the output is written
alongside the source bag as `<bag-stem>_processed` (a sibling ROS2 bag
directory, not a single file — override with `-o/--output`).

| Flag | Default | Meaning |
|------|---------|---------|
| `-w/--workspace` | *(required)* | Workspace with a completed `scale-depth` run |
| `-o/--output` | `<bag>_processed` next to the source bag | Output bag path |
| `--pose-topic` | `/gt/colmap_pose` | Per-frame `geometry_msgs/PoseStamped`, one per registered COLMAP frame, stamped at that frame's *original* capture time |
| `--path-topic` | `/gt/path` | One summary `nav_msgs/Path` message containing every pose, for one-shot visualization |
| `--frame-id` | `map` | `frame_id` written into the new topics' headers |

`/gt/colmap_pose` is named to be unambiguous against a live SLAM stack's
own `/camera_pose`-style topic when both are present in the same bag —
this is the offline, COLMAP-derived pseudo-GT, not a live estimate.

Source connections that carry a custom/vendor message type, or that lack
a `rihs01` type-description digest (older recordings), are still copied
correctly: `export-bag` passes each connection's own message definition
through rather than asking a generic typestore to look it up.

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
