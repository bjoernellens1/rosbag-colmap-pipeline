# Depth-Aware Bundle Adjustment

`depth-ba` jointly refines camera poses and sparse structure against both
reprojection error and metric depth measurements, using
[`kornia-rs`](https://github.com/kornia/kornia-rs)'s Schur-complement bundle
adjuster. It runs after `scale-depth` (which supplies a roughly-metric
initialization) and is CPU-only.

```bash
pip install -e ".[depth-ba]"
gttool depth-ba data/workspaces/session01
# or as part of the full pipeline:
gttool full data/raw/session01.bag --config configs/default.yaml --depth-ba
```

Outputs: a refined sparse model under `colmap/sparse/0_refined/`,
`outputs/trajectory_depth_ba_tum.txt`, and `outputs/depth_ba_report.json`
(convergence status, observation counts).

## Status

`depth_ba.enabled: true` is now the default in `configs/default.yaml`,
backed by a real measured improvement in trajectory accuracy (ATE) on the
`fr3` TUM-RGBD sequence, following the same real-numbers-before-flipping-
a-default rigor as the `mapper_type` default (see
[Mapper Selection](mapper-selection.md)). It is covered by unit and
integration tests (`tests/test_depth_ba_wrapper.py`,
`tests/test_depth_ba_integration.py`).

`gttool full` fails soft on this stage: if `depth-ba` raises (e.g. the
`kornia-rs` extra isn't installed, or the solver doesn't converge for a
particular scene), the exception is caught and logged rather than crashing
the whole pipeline run — the non-depth-ba scale-only output
(`trajectory_metric_tum.txt`) remains valid either way. Pass
`--no-depth-ba` to `gttool full` to skip the stage outright (e.g. on a
machine without the `[depth-ba]` extra installed), or `--depth-ba` to force
it on regardless of what the config file says.

Gating depth observations correctly matters here: `depth-ba` compares each
observation's measured depth against a *bias-corrected* COLMAP depth
estimate (not the raw per-point COLMAP depth), and depth-correspondence
matching for scale estimation itself is now gated per-frame rather than by
one pooled global ratio — both fixes were needed for scenes where COLMAP's
own scale drifts meaningfully across the sequence (see
`scaling/scale_estimation.py` and `optimization/depth_ba.py`).
