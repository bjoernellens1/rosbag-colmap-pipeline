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

**Reading `speed.max_mps`/`rotation.max_deg_per_frame`:** both come with
a `*_step_index`/`*_frame_ids`/`*_is_leading_frames` triple pinpointing
which pose transition produced the max. This exists because a spike in
the trajectory's first couple of transitions is a qualitatively different
(much less concerning) signal than a mid-trajectory one: COLMAP's
earliest registered poses haven't accumulated much multi-view constraint
yet and are characteristically less stable, not evidence of a genuine
tracking glitch or teleport. Confirmed on a real scene (table1,
2026-08-03): `max_mps` was driven entirely by frames 0->1 (9.499 m/s) and
1->2 (9.027 m/s) -- everything else in the top-8 speed ranking was
<=1.16 m/s, consistent with a normal handheld pace. `max_mps_is_leading_
frames=True`/`max_deg_is_leading_frames=True` flags exactly this pattern
so a reader doesn't have to manually re-derive it every time; treat
`is_leading_frames=False` (a spike well past the first couple of frames)
as the one actually worth investigating. If this leading-frames pattern
turns out to show up consistently across scenes once re-run with the
current pipeline, that would confirm it's a general COLMAP-early-
registration characteristic rather than something scene-specific -- worth
checking as the other rerun-queue scenes (floor3, tableware1, hallway,
kitchen1, workshop1) complete.

**floor2 reprocessed 2026-08-04** with `configs/navigation.yaml` (was
`configs/default.yaml`) -- fixed a real bad reconstruction: `default.yaml`'s
untuned matcher (`sequential_overlap: 10`, no `quadratic_overlap`, no
`loop_detection`) left a weak match-graph link around frames 327-332/
567-600 (both genuine, non-leading motion anomalies -- an 11.8°/frame
rotation spike and a 1.27 m/s speed spike), which the `global_mapper`
joint optimization then settled into as a stable-but-wrong two-scale-
regime local minimum (`scale_regime_correction.json` showed a 0.83 vs 2.65
COLMAP-to-depth ratio jump, ~3.2x, right at the frame-600 boundary), with
repeated Ceres "Matrix not positive definite" (CHOLMOD) warnings during
bundle adjustment. Re-running `run-colmap` + `scale-depth` with
`navigation.yaml` (wider `sequential_overlap: 20`, `quadratic_overlap`,
`loop_detection` via the baked-in vocab tree, a longer final global BA
pass) fixed it directly: 267/267 poses registered (was 256/267), zero
scale-regime discontinuity (`action_taken: false`), and a physically
plausible 4.8m x 1.2m x 11.9m trajectory extent (was a collapsed
5.2m x 0.6m x 8.1m). No re-extraction was needed -- `navigation.yaml`'s
`sync:`/`depth:` blocks are identical to `default.yaml`'s and its
`keyframe_selection` was separately validated at 267/267 coverage on this
same scene, so only the COLMAP and scale stages needed rerunning.

**Which config for which scene:** if `scene_metadata.json` from a
`default.yaml` run flags a genuine (non-leading, i.e. `is_leading_frames:
false`) rotation or speed spike, treat `default.yaml` as insufficient and
re-run with `configs/navigation.yaml` instead -- it exists specifically to
bridge the weak-overlap regions such spikes tend to produce. Do not merge
`navigation.yaml`'s settings into `default.yaml`: the wider matching +
loop-detection + longer final BA pass roughly doubles (or worse) matching/
mapping runtime, which is wasted cost on short/easy scenes with naturally
strong overlap. Keep `default.yaml` lean and pick `navigation.yaml`
deliberately per this rule.

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
