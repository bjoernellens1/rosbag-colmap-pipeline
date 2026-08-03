# Output Formats & Evaluation

## TUM RGBD format

```
timestamp tx ty tz qx qy qz qw
```

## CSV format

Additional columns: frame_id, rotation matrix, camera center

## Processed ROS2 bag

`gttool export-bag` writes the GT trajectory back into a copy of the
original bag as ROS messages (`/gt/colmap_pose` per-frame, `/gt/path`
summary), for playback tools that don't want a standalone TUM file — see
[CLI Reference](cli-reference.md#export-a-processed-bag-with-gt-trajectory).

## QC summary (`scene_metadata.json`)

Written automatically alongside `export_report.json`: trajectory length,
pose count/registration ratio, speed and rotation statistics (with the
specific pose transition that produced the max flagged, and whether it
falls in the trajectory's first couple of frames — see
[Workspace Structure](workspace-structure.md) for why that distinction
matters), and a bounding-box extent/volume proxy.

## Evaluation with evo

`trajectory_metric_tum.txt` is already metric (a scale factor was applied during the
`scale-depth` step). Evaluate it with rigid SE(3) alignment only — do **not** pass
`--correct_scale`, since that would silently re-optimize scale during evaluation and
mask a broken scale estimate:

```bash
evo_ape tum groundtruth.txt trajectory_metric_tum.txt -va --align
```

To separately quantify how far off the recovered scale was, run the same comparison
a second time with `--correct_scale` and compare the reported scale correction factor
against 1.0 (`|reported_scale_correction - 1.0|` is the scale error):

```bash
evo_ape tum groundtruth.txt trajectory_metric_tum.txt -va --align --correct_scale
```
