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

This is an experimental, opt-in stage — `kornia-rs`'s bundle-adjustment
module is young, so it is not part of the default pipeline
(`depth_ba.enabled: false` in every shipped config). It is covered by unit
and integration tests (`tests/test_depth_ba_wrapper.py`,
`tests/test_depth_ba_integration.py`) but as of this writing has not yet
been exercised end-to-end on real reconstruction data — only mocked
scenarios. Real-data validation (a real `global_mapper` output run through
`depth-ba`, compared against the non-depth-ba result on scale confidence,
reprojection error, and registration ratio) is expected to happen once a
suitable production workspace (e.g. `trolley_femto`) has a completed
`global_mapper` reconstruction available. If/when that comparison shows a
real, measured improvement, `depth_ba.enabled` will flip to `true` as the
default in `configs/default.yaml` and `configs/navigation.yaml`, following
the same rigor as the `mapper_type` default flip (see
[Mapper Selection](mapper-selection.md)) — real numbers, not assumption.
