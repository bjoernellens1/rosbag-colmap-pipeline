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
    └── scale_report.json
```

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
