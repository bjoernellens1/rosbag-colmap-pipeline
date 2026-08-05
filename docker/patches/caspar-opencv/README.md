# Caspar OpenCV camera-model patch

Submitted upstream: [colmap/colmap#4611](https://github.com/colmap/colmap/pull/4611) (addresses
[colmap/colmap#4371](https://github.com/colmap/colmap/issues/4371)). This directory can be
removed and `docker/Dockerfile.cuda` simplified back to a plain `git clone` once that merges and
this pipeline bumps its pinned COLMAP tag past it.

Adds native `OPENCV` camera-model support to COLMAP's Caspar GPU bundle-adjustment
backend (COLMAP >=4.1.0 only supports `PINHOLE`/`SIMPLE_RADIAL` upstream). Applied
during `docker/Dockerfile.cuda`'s COLMAP build step, overlaying these files onto the
cloned `colmap` source tree before `cmake`/`ninja`.

## Contents

- `caspar_generate.py` — patched copy of COLMAP's
  `src/thirdparty/Symforce-Caspar/caspar_generate.py`, adding an `opencv_core`/
  `opencv_split_core` symbolic model definition (radial + tangential distortion,
  matching `sensor/models.h`'s `OpenCVCameraModel::Distortion` exactly) and
  registering all 15 factor variants (4 merged + 11 split — split variants are
  required, not optional: COLMAP's own default `refine_principal_point = false`
  means real BA calls hit the `FIXED_PRINCIPAL_POINT` split variant, not the merged
  `BASE` variant). This is the *source of truth* — `generated_f32/` below is its
  output, regenerate with `python caspar_generate.py <out_dir> f32` if this file
  ever needs to change (needs `pip install symforce` + a working CUDA toolchain
  to fully validate, though generation itself doesn't need a GPU present).
- `generated_f32/` — output of the above (712 files, ~6MB) — verbatim replacement
  for COLMAP's `src/thirdparty/Symforce-Caspar/generated/f32/` (includes the
  unchanged PINHOLE/SIMPLE_RADIAL kernels too, since `caspar_generate.py`
  regenerates the whole library each run).
- `bundle_adjustment_caspar.cc` — patched copy of
  `src/colmap/estimators/bundle_adjustment_caspar.cc`. Two small additions to
  `BuildSizing()`: an `if (const ModelData* md = get_md(CameraModelId::kOpenCV))`
  block mirroring the existing `kPinhole` one, and an `OpenCV` pose-count lookup
  mirroring `kPinhole`'s. No other changes — this file's dispatch logic
  (`GetAdapter`, `AddFactorCore`, `SetupSolverData`, etc.) is already generic over
  `CameraModelId` via `ICasparModelAdapter`.
- `caspar_model_adapter.h` — patched copy of
  `src/colmap/estimators/caspar/caspar_model_adapter.h`. Adds:
  - `CasparSolverSizing` struct: `num_opencv_poses` + 16 `num_opencv_*` count
    fields (mirrors `SimpleRadial`'s field set, since OpenCV also uses the
    `focal_and_extra` split-key naming, unlike `Pinhole`'s shorter `focal` key).
  - `OpenCVAdapter` class implementing `ICasparModelAdapter` — mechanically
    derived from `SimpleRadialAdapter` (same split-key naming), with
    `FocalAndExtraSize()=6` (`[fx,fy,k1,k2,p1,p2]`, vs SimpleRadial's `[f,k]`=2),
    `PrincipalPointSize()=2` (`[cx,cy]`), and `Extract/WriteFocalAndExtra`/
    `Extract/WritePrincipalPoint` mapped to COLMAP's actual `OPENCV` param layout
    (`camera.params = [fx,fy,cx,cy,k1,k2,p1,p2]` — note this differs from the
    Caspar calib node's packed order `[fx,fy,k1,k2,p1,p2,cx,cy]`,
    focal_and_extra-then-principal_point per `SetupSolverData`'s merged-calib
    packing convention). All 110 unique `s.<Method>(...)` calls in this class
    were verified against the actual generated `solver.h` — zero mismatches.
  - `CreateCasparAdapter()`: one new `case CameraModelId::kOpenCV:` entry.
  - `CreateSolver()`: **completely rewritten positional argument list** — adding
    OpenCV's node types (which sort alphabetically before Pinhole/SimpleRadial)
    shifts every existing argument's position in the generated `GraphSolver`
    constructor, not just appends new ones at the end. All 58 positional
    arguments were verified 1:1 against the actual generated constructor
    signature in `generated_f32/solver.h` (name + position), not derived by
    pattern-matching — this was the single highest-risk part of the whole patch
    (a silent misordering would compile fine and produce wrong numbers, not an
    error).

## Verification performed

- `caspar_generate.py f32` codegen succeeds with exit 0, no shared-memory-budget
  errors (COLMAP's own 48KB-per-block check) — OpenCV's kernels declare the
  identical `__shared__` footprint (`inout_shared[16384]` + 3× `SharedIndex[1024]`
  index arrays) as PINHOLE's, despite OpenCV's calib node being 8 floats vs
  PINHOLE's 4 — the framework packs into a fixed-size buffer regardless of calib
  size, so this fits without any kernel-batching-size changes.
- All 110 `s.<Method>()` calls in `OpenCVAdapter` cross-checked against
  `generated_f32/solver.h` — 0 missing.
- All 58 `CreateSolver()` positional arguments cross-checked by name AND position
  against `generated_f32/solver.h`'s actual `GraphSolver` constructor — 0
  mismatches.
- NOT yet verified: actual compilation (needs the CUDA-enabled Docker build) or
  numerical correctness against the Ceres/CPU OPENCV baseline (needs a real BA
  run on the A100 cluster) — see the session plan
  (`/home/bjoern/.claude/plans/linear-puzzling-candy.md`) for the remaining
  verification steps.
