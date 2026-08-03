# Workspace Structure

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
    ├── scale_report.json
    ├── export_report.json
    ├── scene_metadata.json
    ├── trajectory_depth_ba_tum.txt   # only if depth_ba.enabled
    └── depth_ba_report.json          # only if depth_ba.enabled
```

`scene_metadata.json` is a quick-glance QC summary of the exported GT
trajectory -- trajectory length, pose count/registration ratio, average/
max/std camera speed, total/average/max rotation between consecutive
poses, trajectory bounding-box extent and a coarse AABB volume proxy, and
a revisit/loop-closure coverage signal -- written automatically by the
`scale-depth` stage (part of every `gttool full` run) alongside
`export_report.json`, so it's available for every scene without an extra
flag or command. See `compute_scene_metadata()` in
`src/colmap_rgbd_gt/export/scene_metadata.py` for the exact field
definitions and the bbox-volume proxy's stated limitation (axis-aligned
bounding box, not a real occupied-space estimate).

## Default workspace location

`gttool extract` and `gttool full` default `--workspace`/`-w` to
`<repo_root>/data/workspaces/<bag-stem>` when not explicitly given
(`_default_workspace()` in `src/colmap_rgbd_gt/cli.py`). The repo root is
resolved by walking up from the installed package location to the nearest
`pyproject.toml` — inside the container this is `/app`; on a dev checkout
it's wherever the repo was cloned — so the same bag always produces the
same workspace path regardless of the shell's current working directory
when `gttool` is invoked, or where the source bag file happens to live.

Always pass an explicit `-w/--workspace` if you deliberately want a
workspace outside `data/workspaces/` (e.g. a scratch/throwaway run).
