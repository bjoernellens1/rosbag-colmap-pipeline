# COLMAP ROCm Integration — Design

Date: 2026-08-04
Target hardware: AMD Ryzen AI Max+ PRO 395 / Radeon 8060S (gfx1151), this workstation, ROCm present, `gh` authenticated as `bjoernellens1`.

## Problem

We want GPU-accelerated COLMAP (dense MVS, bundle adjustment, and eventually feature
extraction/matching) on AMD ROCm hardware, for fast camera-trajectory/SfM preprocessing
ahead of Gaussian-splatting workflows. Rather than porting COLMAP/Caspar/SymForce from
scratch, three pieces of upstream work already exist and were verified live via `gh api`
on 2026-08-04:

- `colmap/colmap#4420` ("Add ROCm/HIP support for patch_match_stereo (AMD GPU)"),
  head `iShengnan/colmap:rocm-support` → `colmap/colmap:main`. 21 files, 8 commits,
  `mergeable: true`, `mergeable_state: blocked` (review/CI, not conflicts).
- `symforce-org/symforce#465` ("Add AMD GPU (ROCm/HIP) support to the Caspar backend"),
  head `jeffdaily/symforce:moat-port` → `symforce-org/symforce:main`. 18 files, 2 commits,
  currently `mergeable: false` / `mergeable_state: dirty` — needs rebase.
- `jeffdaily/colmap` branch `rocm-sift-gpu` — GPU SIFT extraction/matching prototype,
  10 commits ahead of colmap `main` but **115 commits behind** — stale, will need a
  nontrivial rebase.

Both PatchMatch-HIP (gfx1151-tested, per PR discussion) and Caspar-HIP (tested on
MI250/RDNA3/RDNA4 per PR description) report working numerics. The real gap is GLOMAP's
non-bundle-adjustment optimization stages (global positioning, rotation averaging, etc.),
which have no ROCm implementation anywhere — see Out of Scope below.

## Approach

Two forked repos under `bjoernellens1`, cloned to `~/git/`:

1. **`~/git/symforce-rocm`** — fork of `symforce-org/symforce`. Integration branch
   `hip-integration` = current `main` + rebased `jeffdaily:moat-port` (PR #465).
2. **`~/git/colmap-rocm`** — fork of `colmap/colmap`. Integration branch
   `hip-integration` = current `main` + rebased `iShengnan:rocm-support` (PR #4420) +
   rebased `jeffdaily:rocm-sift-gpu`. Depends on the HIP Caspar build produced by
   `symforce-rocm`.

### Docker base

Build inside a container based on `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`
(already pulled locally, 29.5GB) — this is the same battle-tested base already used by
`~/git/rosbag-colmap-pipeline`, `~/git/gaussian-splatting-lightning`, and
`~/git/splatograph` on this machine, with `ROCM_ARCH=gfx1151` and
`HSA_OVERRIDE_GFX_VERSION=11.5.1` per that convention.

`colmap-rocm` gets its own `Dockerfile` (modeled on `rosbag-colmap-pipeline`'s, which
already builds COLMAP from source on this same base image, currently CPU-only
`-DCUDA_ENABLED=OFF`), building from local source (`COPY`) instead of
`git clone --branch 4.1.1`, with `-DCUDA_ENABLED=OFF -DHIP_ENABLED=ON
-DCMAKE_HIP_ARCHITECTURES=gfx1151`. SymForce's HIP Caspar build is produced as an
earlier build stage (or a separately-built local artifact copied in via build context)
so `colmap-rocm`'s image can consume it without vendoring SymForce source into the
COLMAP repo.

### Integration order

1. Fork `colmap/colmap` and `symforce-org/symforce` to `bjoernellens1`; clone both
   locally; add `upstream`/contributor remotes.
2. `symforce-rocm`: rebase PR #465 onto current `main`, resolve the dirty-merge
   conflicts. Build `compile_caspar_library(use_hip=True, hip_arch="gfx1151")` inside
   the container; smoke-test only (library builds and loads), not a full BAL run yet.
3. `colmap-rocm`: rebase PR #4420 onto current `main` alone first (small, clean diff).
   Build the Dockerfile, run `patch_match_stereo` against a real dataset in-container
   on gfx1151, confirm sane depth/fused-point output.
4. Rebase `rocm-sift-gpu`'s 10 commits on top of the now-updated branch — expect real
   conflicts given the 115-commit drift from upstream `main`.
5. Wire the SymForce HIP Caspar build in as the mapper's bundle-adjustment backend.
6. Run full incremental SfM end-to-end inside the container — `feature_extractor` →
   `matcher` → `mapper` → `patch_match_stereo` — on a real dataset on this gfx1151 box,
   and compare against CPU/existing reference results.

### Success criteria (this milestone)

Full incremental SfM pipeline runs end-to-end on real data using the ROCm-accelerated
path (HIP SIFT if the rebase succeeds, HIP PatchMatch, HIP Caspar BA), producing a
reconstruction comparable in quality to the CPU/CUDA baseline, all inside the
`rocm/pytorch`-based container on this machine.

## Out of scope (future work)

Explicitly deferred to their own future spec/plan cycles — noted here so they aren't
lost:

- **GLOMAP global-positioning ROCm solver.** GLOMAP is now merged into `colmap/colmap`
  as `global_mapper`; its non-BA optimization stages (rotation averaging, global
  positioning, fixed-rotation constraints) have no ROCm/HIP implementation anywhere
  upstream as of 2026-08-04. Caspar-HIP only covers bundle adjustment, not these
  stages. A future project would need either a custom HIP PCG solver, new Caspar
  factor types for GLOMAP's formulations, or a ROCm backend for Ceres itself
  (no such backend currently exists — verified no relevant hipSPARSE/rocSOLVER/HIP
  Schur work in the Ceres repo). Estimated 1–3 months per the original analysis.
- **PyCOLMAP ROCm wheels / packaging.** No official ROCm PyCOLMAP wheel, Conda
  variant, or Docker image exists upstream. Packaging (wheel build, CI, multi-arch
  Docker images) is separate follow-on work once the integration branch is proven.
- **MIGraphX/ONNX Runtime feature path (ALIKED/LightGlue).** COLMAP supports these
  models via ONNX Runtime, and AMD's MIGraphX execution provider could accelerate them
  on ROCm, but no COLMAP-specific MIGraphX integration exists yet. Independent of the
  SIFT-HIP path — a possible alternative/complementary feature-extraction route.
- **Upstreaming.** No plan yet to push `hip-integration` branches back as PRs to
  `colmap/colmap` or `symforce-org/symforce` — this work stays in the forks until
  proven out locally.
- **ZLUDA.** Explicitly rejected as a path — an extra translation layer, unlikely to
  handle CUDA textures/CUB/cooperative-groups/generated Caspar kernels as reliably as
  the native HIP ports already in progress.
