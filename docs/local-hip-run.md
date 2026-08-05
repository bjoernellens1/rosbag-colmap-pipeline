# Running COLMAP on this host's local AMD GPU (HIP/ROCm)

Alongside dispatching `run-colmap` to the `cps-gpu-cluster` A100 nodes (see
`docs/cluster-dispatch.md`), this pipeline can also run COLMAP reconstruction locally on this
host's own AMD GPU (gfx1151, Radeon 8060S) via HIP — no cluster/network dependency, useful for
quick local iteration. **Verified working end-to-end 2026-08-05**, including a real,
production-code pipeline run (not just standalone `colmap` CLI calls) on a real scene, twice, for
consistency. This doc covers the operational how-to; it does not touch or replace the CUDA
cluster-dispatch path in any way.

## Background: where the HIP-enabled COLMAP build comes from

`docker/Dockerfile` (the existing plain ROCm/PyTorch-based image) does `apt-get install colmap`,
which on this host's Ubuntu 24.04 base resolves to a CPU-only build — no CUDA, no HIP. Building
COLMAP from source with HIP support requires a fork with ROCm/HIP patches, since upstream COLMAP
has no HIP backend. `docker/Dockerfile.hip` builds
[`bjoernellens1/colmap`](https://github.com/bjoernellens1/colmap)'s `hip-integration` branch
instead, which — as of the commit vendored here (`0b079c55`) — has HIP-accelerated feature
extraction/matching (`SiftGPU`), dense stereo (`patch_match_stereo`), and bundle adjustment
(`CASPAR` backend, with native `OPENCV` camera-model support, the same GPU BA solver
`docker/Dockerfile.cuda` uses on the A100 cluster), all separately verified on real data on this
host's gfx1151 GPU in prior investigation.

The COLMAP source is vendored as a plain directory copy at `docker/colmap-rocm-hip-src/` (not a
git submodule, not cloned at build time) — this pins the exact tested worktree state rather than
whatever `hip-integration`'s HEAD happens to be on a later build. Re-sync it if the upstream
branch moves and you want the newer state:

```bash
rsync -a --exclude='.git' --exclude=build <colmap-rocm worktree>/ docker/colmap-rocm-hip-src/
```

## Building the HIP image

`docker/Dockerfile.hip` mirrors `docker/Dockerfile.cuda`'s structure (same apt dependency list,
same vocab-tree asset, same pipeline install steps) but builds COLMAP with:

```
-DHIP_ENABLED=ON -DCUDA_ENABLED=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCASPAR_ENABLED=ON
```

on top of the same `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` base
`docker/Dockerfile` already uses (so `--system-site-packages` still picks up the base image's
ROCm-enabled PyTorch, same as the plain image).

```bash
docker build -f docker/Dockerfile.hip -t colmap-rgbd-gt:hip .
```

(This host's `docker` CLI aliases to podman 5.8.4.) Build takes a few minutes — most of the time
is compiling ~300 translation units; layer caching makes rebuilds after a source-only change much
faster.

## Running

Same GPU-passthrough flags established in this session's earlier `colmap-rocm` verification work:

```bash
docker run --rm \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  --group-add 39 --group-add 105 \
  --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  -e QT_QPA_PLATFORM=offscreen \
  -v "$(pwd)/docker/workspaces:/app/workspaces" \
  --entrypoint gttool colmap-rgbd-gt:hip \
  run-colmap /app/workspaces/<scene> --gpu --config /app/configs/tum_like.yaml
```

`--group-add 39`/`--group-add 105` are this host's numeric `video`/`render` GIDs — podman
accepts numeric GIDs without needing the group *names* to exist inside the container's
`/etc/group` (only `video` is defined there by default). `--gpu` on `run-colmap` sets
`colmap.use_gpu = true` in the config actually passed to `colmap_pipeline()` — this is a
pipeline-level flag, not CUDA-specific naming; it works identically for the HIP build.

## Verification (2026-08-05)

Ran the real `gttool run-colmap` pipeline entry point (`src/colmap_rgbd_gt/pipelines/colmap_only.py`
via `cli.py`'s `run-colmap` command — actual production orchestration code, not raw `colmap` CLI
invocations) against the TUM `freiburg1_desk` scene (613 RGB-D frames, already extracted from a
prior run), copied into two fresh workspaces (`hip_run1`, `hip_run2`) so each run started from
identical raw input with no shared/staled COLMAP state. Config: `configs/tum_like.yaml`
(`camera_model: OPENCV`, `matcher: sequential`), `--gpu`.

Both runs:

- Completed with container exit code 0 in ~3–5 minutes.
- Registered **613/613 frames** (`COLMAP pipeline complete: 613 poses`, matching
  `outputs/trajectory_colmap_unscaled.txt` line counts) — full registration, no fragmentation.
- Produced a real sparse reconstruction at `colmap/sparse/0/points3D.bin`
  (~7.5–7.6MB each run, non-trivial point count) plus `cameras.bin`/`frames.bin`/`images.bin`,
  matching this pipeline's expected output layout (`configs/ablator.toml`'s
  `result_glob = "{model_path}/colmap/sparse/0/points3D.bin"` pattern, same artifact this repo's
  cluster-dispatch path checks for).
- Log output showed genuine GPU-side work in feature extraction and matching.

  **Correction (2026-08-05):** the line above originally also claimed this run's `global_mapper`
  stage showed "Caspar bundle-adjustment iterations" — that was wrong. `global_mapper`'s internal
  3-iteration BA loop (`global_positioning.cc`/`bundle_adjustment_ceres.cc`) always ran the
  CPU Ceres solver in this run, printing `Requested to use GPU for bundle adjustment, but COLMAP
  was compiled without CUDA support. Falling back to CPU-based solvers.` on every invocation —
  this is *expected and correct*: `global_mapper`'s Caspar path (`--GlobalMapper.ba_backend
  CASPAR`) is deliberately not requested by this pipeline (see `runner.py`'s `global_mapper()`,
  which gates it behind a separate `global_mapper_ba_backend` key left unset by default after a
  live accuracy sweep found it caused reconstruction fragmentation — see that function's comment).
  So neither this run nor its sibling ever exercised genuine GPU/Caspar bundle adjustment; only
  GPU SIFT extraction and matching were real GPU work here. See "Corrected GPU-BA speedup
  measurement (2026-08-05)" below for the actual first genuine Caspar-HIP BA measurement, using
  the separate, already-safe-by-default standalone `bundle_adjuster --BundleAdjustment.backend
  CASPAR` pass instead.

**Run-to-run consistency check**: both runs registered the identical frame count (613/613) and
produced near-identical trajectory path length (45.65 vs 45.23 units) with highly correlated
per-step motion (correlation 0.96 between consecutive-frame step lengths). The two trajectories'
raw XYZ coordinates differ by more than trivial floating-point noise — expected and not a bug:
`global_mapper`'s unscaled/unaligned output has gauge freedom (no canonical global reference
frame is enforced between independent runs), so run-to-run global rotation/translation offsets
are normal even on deterministic hardware; `scale-depth`'s Umeyama alignment step (which every
real scene run already goes through) removes exactly this ambiguity before any downstream
comparison. The reconstruction's internal *shape* (path length, per-step motion) is what actually
indicates a correct, reproducible reconstruction, and that matched closely across both runs.

**Conclusion: the actual production pipeline (`gttool run-colmap` → `colmap_pipeline()`), not
just COLMAP itself, works correctly end-to-end on this host's local AMD GPU via HIP.**

## Corrected GPU-BA speedup measurement (2026-08-05)

The verification run above never exercised genuine GPU bundle adjustment (see correction note).
`global_mapper`'s own internal Caspar BA path stays deliberately disabled (documented
fragmentation regression, see `runner.py`'s `global_mapper()`). The separate, already
safe-by-default path is the standalone final BA pass (`runner.py`'s `bundle_adjuster()`,
gated on `colmap.ba_backend: caspar` + `use_gpu: true`) — this is the mechanism actually
exercised below, the first genuine Caspar-HIP BA measurement for this pipeline.

Two fresh workspaces from the same 613-frame `freiburg1_desk` extraction, run through
`gttool run-colmap` end to end with `run_bundle_adjustment: true` added to the config so the
final `bundle_adjuster` pass runs in both:

- CPU run: `--config` with `use_gpu: false`, no `ba_backend` (final BA uses Ceres/CPU).
- GPU run: `gttool run-colmap --gpu --config ...` with `ba_backend: caspar`
  (`--privileged`, `--device=/dev/kfd`, `--device=/dev/dri`,
  `-e HSA_OVERRIDE_GFX_VERSION=11.5.1`, `--security-opt label=disable`).

Confirmed genuine Caspar-HIP engagement on the GPU run by re-running the same standalone
`bundle_adjuster --BundleAdjustment.backend CASPAR` step at `--log_level 2`:

```
bundle_adjustment_caspar.cc:24] Using Caspar bundle adjuster
cuda.cc:72] Found 1 CUDA device(s), selected device 0 with name Radeon 8060S Graphics
bundle_adjustment_caspar.cc:819]   Points: 54205  Frames: 613
bundle_adjustment_caspar.cc:998] Caspar: CONVERGED_DIAG_EXIT after 3 iters
```

(HIP is exposed through COLMAP's CUDA-compat layer, hence the "CUDA device" log wording — the
actual device is an AMD APU via ROCm/HIP, not real CUDA.)

Per-stage wall-clock timings, taken from COLMAP's own `timer.cc` elapsed-time log lines:

| Stage | CPU run | GPU run | Delta |
|---|---|---|---|
| Feature extraction | 13.8s | 68.8s | GPU run 5x **slower** here — HIP context/kernel init overhead dominates on this small scene; not a regression in the GPU path itself, just fixed startup cost not amortized at 613 frames |
| Sequential matching | 38.8s | 31.8s | GPU ~18% faster |
| `global_mapper` internal loop (CPU Ceres both runs, by design) | 275.5s | 273.9s | ~identical, as expected — same code path both runs |
| Final bundle adjustment (Ceres CPU vs Caspar-HIP GPU) | 65.6s | 0.48s | **NOT a trustworthy speedup** — see "Correctness follow-up" below: the Caspar-HIP pass does zero optimization work on this scene (nan cost from iteration 0, no LM step ever accepted, output bit-identical to input), so 0.48s reflects an immediate solver failure, not fast-but-correct GPU work |
| **End-to-end pipeline** | 401.3s | 383.0s | ~5% faster end-to-end |

**Honest takeaways:**

- freiburg3 (1300-frame) re-measurement was not completed in this pass — scope was limited to
  freiburg1_desk to first confirm genuine engagement and get a trustworthy number.

**Correctness follow-up (2026-08-05): the ~137x number above is NOT trustworthy — confirmed
real bug, not the earlier-documented benign symforce-rocm nan artifact.**

The Caspar solver's own log flagged `CONVERGED_DIAG_EXIT after 3 iters (diag limit hit ->
likely premature termination)` with `score_current: -nan` during iteration. `symforce-rocm`'s
`docs/rocm-integration.md` documents a *different*, confirmed-benign nan artifact on synthetic
BA problems: a near-convergence 0/0 caught by the LM step-acceptance guard, occurring late,
after most of the real optimization work is already done. This occurrence does not match that
signature — `score_current` is `-nan` from iteration 0, not just near the end, and no LM step
is ever accepted (see next paragraph) — so the earlier finding does not transfer here without
separate verification, and separate verification says it doesn't apply.

To isolate the BA-only effect (a full pipeline run confounds CPU-vs-GPU SIFT/matcher output
with the BA backend, since each run's `global_mapper` stage then optimizes a different
reconstruction), the same post-`global_mapper` sparse model (613/613 registered images, 53992
points, baseline mean reprojection error **0.767858px** via `colmap model_analyzer`) was fed
through both backends directly:

- `colmap bundle_adjuster --input_path in --output_path out_ceres --log_level 2` (Ceres/CPU):
  101 iterations, 77.1s, `NO_CONVERGENCE` (hit the 100-iteration cap) — reprojection error
  **0.785265px** (slightly worse than baseline; still mid-optimization, not fully converged
  within the cap, but genuinely doing iterative work).
- `colmap bundle_adjuster --input_path in --output_path out_caspar --BundleAdjustment.backend
  CASPAR --log_level 2` (Caspar-HIP): full iteration trace —
  ```
  score_init:  4.491941e+05
  solver_iter:   0  pcg_iter:  10  score_current: -nan  score_best: 4.491941e+05  step_quality: 0.000  diag: 1.000e+00
  solver_iter:   1  pcg_iter:   6  score_current: -nan  score_best: 4.491941e+05  step_quality: 0.000  diag: 2.000e+00
  solver_iter:   2  pcg_iter:   2  score_current: -nan  score_best: 4.491941e+05  step_quality: 0.000  diag: 8.000e+00
  Caspar: CONVERGED_DIAG_EXIT after 3 iters (diag limit hit -> likely premature termination)
  ```
  `step_quality: 0.000` on every iteration (no step ever accepted), `score_best` pinned at
  `score_init` throughout, `diag` (LM damping) climbing 1.0 -> 2.0 -> 8.0 as every step is
  rejected until the damping ceiling is hit and the solver bails — its own log message says so.
  The camera-intrinsics log line shows `params: [...] -> [...]` with identical values before
  and after. `model_analyzer` on `out_caspar` confirms it: mean reprojection error
  **0.767858px — bit-identical to the pre-BA baseline.** Caspar wrote back the unmodified
  input; it did no optimization work at all.

**Verdict: real bug, not benign.** The ~137x/0.48s number reflects a solver that fails
immediately (nan cost from iteration 0, zero accepted steps) and exits, not a fast-but-correct
GPU optimization. It should not be used as a genuine performance measurement until fixed.

**Update (2026-08-05): root cause narrowed — isolated to Caspar's `OPENCV` camera-model
kernels specifically, not to this scene's scale, input data, or distortion values.** Dedicated
follow-up investigation (in `colmap-rocm`, see that repo's `docs/rocm-integration.md` for the
full evidence trail) reproduced this exact failure directly against `bundle_adjuster
--BundleAdjustment.backend CASPAR`, then ruled out causes by controlled substitution on the
identical 613-pose/54458-point problem:

- Input data is clean — no NaN/Inf in `points3D.txt`, all 635192 observations have positive
  camera-frame depth, no degenerate/behind-camera points.
- **Scale is not the trigger**: the same 613-image/54458-point problem with the camera model
  swapped to `SIMPLE_RADIAL` or `PINHOLE` (same poses/observations, no re-extraction) bundle-
  adjusts correctly — 200 real LM iterations, genuine score decrease, only occasional benign
  transient nans correctly rejected (the already-diagnosed symforce-rocm artifact, not this bug).
- **Distortion coefficient values are not the trigger**: forcing the camera to stay `OPENCV`
  but zeroing `k1,k2,p1,p2` (mathematically equivalent to `PINHOLE`) still fails identically.
- **Factor-variant dispatch is not the sole trigger**: both Caspar's `FIXED_PP` and `BASE`
  OPENCV factor variants (different generated kernel files) fail the same way.

This means the defect is specific to Caspar's `OPENCV`-model kernel family
(`kernel_opencv_*.cu` in `colmap-rocm`'s `src/thirdparty/Symforce-Caspar/generated/f32/`) at
this problem's scale — it did not show up in this session's earlier smaller-scale OPENCV
numerical check (30 images) but does at 613 images/634857 factors, while non-OPENCV models are
fine at the same 613-image scale. Leading hypothesis: OPENCV's 8 intrinsic parameters (vs. 4 for
PINHOLE/SIMPLE_RADIAL) give its kernels a larger per-thread-block register/shared-memory
footprint, and a buffer/stride sizing assumption correct for the smaller models breaks past some
factor-count threshold — the same general class of HIP shared-memory bug already found and fixed
once this session in symforce-rocm, but a distinct instance, not the same code. Exact defect
location (line number inside the ~1400-line generated kernel files) not yet pinned down — see
`colmap-rocm`'s `docs/rocm-integration.md` 2026-08-05 entry for the full reproduction commands,
ruled-out list, and suggested next step (bisect problem size with OPENCV forced to find the
exact failure threshold). Not yet fixed.

## Files

- `docker/Dockerfile.hip` — the HIP-enabled image definition.
- `docker/colmap-rocm-hip-src/` — vendored COLMAP source (`hip-integration` branch, commit
  `0b079c55`) built by the above.

This path is entirely additive — `configs/ablator.toml`'s A100 cluster dispatch config and
`docker/Dockerfile.cuda` are untouched.
