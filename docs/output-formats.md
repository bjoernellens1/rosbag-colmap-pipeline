# Output Formats & Evaluation

## TUM RGBD format

```
timestamp tx ty tz qx qy qz qw
```

## CSV format

Additional columns: frame_id, rotation matrix, camera center

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
