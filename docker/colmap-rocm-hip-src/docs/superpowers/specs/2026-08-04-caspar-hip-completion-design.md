# Caspar-HIP Completion — Design

Date: 2026-08-04
Builds on: `docs/superpowers/specs/2026-08-04-colmap-rocm-integration-design.md` (prior
milestone: PatchMatch-HIP verified end-to-end on gfx1151; HIP SIFT and Caspar-HIP
wiring deferred with documented reasons).

## Problem

Four items remain from the prior milestone's deferrals plus newly-discovered
overlapping work:

1. Caspar-HIP's smoke test (`symforce-rocm`) only exercises 4 toy accessor kernels,
   not the actual bundle-adjustment solver path (PCG, `SortIndices`, `GraphSolver`).
2. Caspar-HIP was never wired into COLMAP's own build — COLMAP vendors Caspar as
   generated CUDA source (`src/thirdparty/Symforce-Caspar/generated/{f32,f64}`),
   gated CUDA-only in `CASPAR_ENABLED`.
3. HIP SIFT (`jeffdaily/rocm-sift-gpu`) was deferred — its rebase hit a 12-file
   conflict on the first commit against `hip-integration` (5-file abort threshold).
4. `colmap-rocm`'s own CUDA build path (`CUDA_ENABLED=ON`) was never compiled —
   the one hand-resolved rebase conflict (Task 1, a CMake arch-list template) is
   unverified.

Independently discovered: `~/git/rosbag-colmap-pipeline` (a sibling repo on this
machine) already has a **complete, cluster-verified** CUDA Caspar patch adding
native `OPENCV` camera-model support (`docker/patches/caspar-opencv/`, submitted
upstream as `colmap/colmap#4611`), built on a `caspar_generate.py`-regeneration
workflow. Verified live 2026-08-04 on the cps-gpu-cluster's A100 workers: 267/267
poses, zero scale-regime split vs. the CPU/Ceres baseline, 290x BA-stage speedup.
This is real, working prior art for exactly the regeneration mechanism a HIP-wired
Caspar in `colmap-rocm` would also need — item 2 should incorporate it rather than
solve the same problem twice.

## Key clarification (resolves an apparent design conflict)

Two things that looked like alternative approaches for item 2 are actually
independent, composable axes:

- **Which camera models are generated** — controlled by which `caspar_generate.py`
  variant runs. Upstream generates only PINHOLE/SIMPLE_RADIAL; `rosbag-colmap-pipeline`'s
  patched version adds OPENCV. `caslib.generate(out_dir, ...)` (no `use_hip` param)
  only emits generic CUDA-spelling `.cu`/`.h` source regardless of which models it
  covers — it does not compile anything.
- **CUDA vs. HIP compilation of that generated source** — controlled entirely at
  COLMAP's own build time: `LANGUAGE HIP` in `generated/f32/CMakeLists.txt` (same
  per-source mechanism PR #4420 already applies to `patch_match_cuda.cu`) plus a
  `cooperative_groups`/`cg::*`-aware compat header, informed by SymForce PR #465's
  HIP mappings but written for whatever's actually in `generated/f32/` — which after
  merging the OpenCV patch is the union of both projects' kernels, not either alone.

So: regenerate once with OpenCV support, then make COLMAP's build of that generated
tree HIP-capable. One pipeline, not two competing ones.

## Approach — four sub-projects, dependency-ordered

### 1. Caspar solver-path verification (`~/git/symforce-rocm`)

Extend `hip_smoke.py`/`hip_smoke_run_only.py` (or add a new script alongside them)
to exercise Caspar's actual bundle-adjustment solver: build a small synthetic
factor graph exercising `GraphSolver`, PCG iterations, and `SortIndices` — not
just the existing toy `example_kernel`'s 4 accessor types. Compare against a
CPU/Ceres or known-good reference where feasible (a tiny synthetic BAL-style
problem is enough — doesn't need real image data). This is the correctness gate
for trusting Caspar-HIP in anything downstream.

**Blocks:** sub-project 2 (don't wire an unverified solver path into COLMAP).

### 2. Caspar-HIP wired into COLMAP, merged with caspar-opencv (`~/git/colmap-rocm`)

- Port `rosbag-colmap-pipeline/docker/patches/caspar-opencv/` (patched
  `caspar_generate.py`, `bundle_adjustment_caspar.cc`, `caspar_model_adapter.h`)
  into `colmap-rocm`. That patch targets a COLMAP `4.1.1` pin; `colmap-rocm` is on
  current `main` — re-verify the dispatch-site line numbers/patterns (`BuildSizing()`,
  `CasparSolverSizing`, `CreateSolver()`'s positional-argument list) still apply,
  don't assume a clean drop-in.
- Regenerate `generated/{f32,f64}` from the merged (OpenCV-aware) generator.
- Add HIP compilation: HIP branch in `cmake/FindDependencies.cmake`'s
  `CASPAR_ENABLED` arch guard (currently `CMAKE_CUDA_ARCHITECTURES`-only), `LANGUAGE
  HIP` in `generated/f32/CMakeLists.txt`, and a Caspar-specific compat header
  covering `cooperative_groups`/`cg::reduce`/`cg::labeled_partition`, informed by
  (not copy-pasted from) PR #465's mappings — verified against sub-project 1's
  now-broader test coverage.
- Verify: `colmap mapper --BundleAdjustment.backend CASPAR` (or the actual flag
  name — confirm, don't assume) runs on real data with an OPENCV-model
  reconstruction on gfx1151, producing comparable output to the CPU/Ceres baseline.

**Depends on:** sub-project 1. **Independent of:** sub-projects 3, 4.

### 3. HIP SIFT (`~/git/colmap-rocm`)

Cherry-pick only the SIFT-specific commits from `jeffdaily/rocm-sift-gpu`,
skipping ones that duplicate PatchMatch-HIP compat-layer changes already in
`hip-integration` (the prior session's rebase attempt aborted specifically
because the first commit touched `cuda_flip/rotate/texture/transpose.h`,
`gpu_mat.h`, `patch_match_cuda.h`, `util/cuda*` — files PR #4420 already
rewrote). Identify which of the 10 commits are genuinely SIFT-only first.

**Independent** of sub-projects 1, 2, 4 — can run in parallel.

### 4. CUDA regression check (`~/git/colmap-rocm`, on cps-gpu-cluster)

Confirm `hip-integration`'s `CUDA_ENABLED=ON` configuration still compiles and
runs correctly — specifically the one hand-resolved conflict in Task 1's rebase
(a CMake arch-list template merge). Use the cluster's existing `ablator`
dispatch pattern (from `rosbag-colmap-pipeline`'s `docs/cluster-dispatch.md`) to
build and smoke-test on an A100 worker. Distinct from caspar-opencv's
already-proven verification — different repo/branch (current `main`, not the
`4.1.1` pin), never itself compiled.

**Independent** of sub-projects 1, 2, 3 — can run in parallel. Can run before or
after sub-project 2 lands (2 doesn't change the CUDA path's compilability by
design — HIP additions are additive, gated by `HIP_ENABLED`).

## Out of scope (unchanged from prior spec, plus new items)

- PyCOLMAP ROCm wheels/packaging, MIGraphX/ONNX feature path, upstreaming to
  `colmap/colmap`/`symforce-org/symforce`, ZLUDA (all carried over).
- Merging `colmap-rocm`'s eventual HIP+OpenCV Caspar work back into
  `rosbag-colmap-pipeline`'s own CUDA-only `caspar-opencv-native` branch, or
  vice versa — the two forks solve different problems (HIP-on-gfx1151 vs.
  CUDA-on-A100-cluster-in-production) and should stay independent even though
  sub-project 2 ports the OpenCV *generator* changes across.
- Split-calib Caspar variants beyond what `caspar-opencv`'s generator already
  registers (that patch's own README documents this as already handled — 15
  factor variants, 4 merged + 11 split).
