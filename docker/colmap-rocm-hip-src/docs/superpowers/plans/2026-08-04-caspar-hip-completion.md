# Caspar-HIP Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify Caspar's actual bundle-adjustment solver path on HIP, wire Caspar-HIP into `colmap-rocm`'s own COLMAP build merged with `rosbag-colmap-pipeline`'s proven OpenCV camera-model patch, land HIP SIFT where mechanically clean, and confirm the CUDA build path still works.

**Architecture:** Four dependency-ordered/parallel tracks across three repos (`symforce-rocm`, `colmap-rocm`, read-only reference to `rosbag-colmap-pipeline`) plus the cps-gpu-cluster for CUDA verification. Track A (solver verification) gates Track B (COLMAP wiring). Tracks C (HIP SIFT) and D (CUDA regression) are independent of A/B and each other.

**Tech Stack:** CMake 3.21+, HIP/ROCm 7.2, hipcc, SymForce codegen (`caslib.generate()`), Docker/Podman, `kubectl`/`ablator` (cluster dispatch), git rebase/cherry-pick.

## Global Constraints

- Target GPU architecture for HIP work: `gfx1151` only, on this machine.
- Base Docker image for `colmap-rocm`/`symforce-rocm` builds: `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`.
- `HSA_OVERRIDE_GFX_VERSION=11.5.1` passed via `-e` at `docker run` time (not baked into image `ENV`, per the prior milestone's final-review fix).
- Podman on this host needs `--security-opt label=disable` on bind mounts; `--group-add` needs numeric host GIDs (`39` for video, `105` for render — confirm via `getent group render video`, don't hardcode blindly on other hosts).
- `QT_QPA_PLATFORM=offscreen` and `--FeatureExtraction.use_gpu 0`/`--FeatureMatching.use_gpu 0` needed for headless COLMAP CLI runs (per prior milestone's Task 7 findings) — irrelevant to Caspar/BA work directly but needed if any task runs a full pipeline smoke test.
- Every rebase/port targets the CURRENT tip of the relevant branch at time of execution (`colmap-rocm:hip-integration`, `symforce-rocm:hip-integration`), not a stale snapshot.
- Push to `origin` only (the `bjoernellens1` forks) — never to `upstream` (`colmap/colmap`, `symforce-org/symforce`) or `rosbag-colmap-pipeline`'s own remote.
- `docs/rocm-integration.md` (in whichever repo a task modifies) gets a dated entry per task, same convention as the prior milestone.
- If a `git reset --hard <branch>` is used to fold a rebase onto an integration branch, first confirm every commit currently on the target branch is an ancestor of the source branch (a prior task in this ecosystem lost commits this way) — never reset blind.

---

## File Structure

- `~/git/symforce-rocm/hip_solver_smoke.py` — new script, Track A. Exercises `GraphSolver`/PCG/`SortIndices` on a small synthetic factor graph, HIP-compiled, gfx1151.
- `~/git/colmap-rocm/src/thirdparty/Symforce-Caspar/caspar_generate.py` — modified, Track B: merges `rosbag-colmap-pipeline`'s OpenCV `opencv_core`/registration additions.
- `~/git/colmap-rocm/src/thirdparty/Symforce-Caspar/generated/{f32,f64}/` — regenerated, Track B.
- `~/git/colmap-rocm/src/thirdparty/Symforce-Caspar/generated/f32/CMakeLists.txt` — modified, Track B: `LANGUAGE HIP` per-source mechanism.
- `~/git/colmap-rocm/cmake/FindDependencies.cmake` — modified, Track B: HIP branch in `CASPAR_ENABLED`'s arch guard.
- New: `~/git/colmap-rocm/src/colmap/estimators/caspar/caspar_hip_compat.h` (or similar — exact name decided in Track B Task 1) — Caspar-specific `cooperative_groups` HIP compat header.
- `~/git/colmap-rocm/src/colmap/estimators/bundle_adjustment_caspar.cc`, `src/colmap/estimators/caspar/caspar_model_adapter.h` — modified, Track B: OpenCV dispatch, ported from `rosbag-colmap-pipeline`'s patch and re-verified against current `main`.
- `~/git/colmap-rocm` `hip-integration` branch — Track C: cherry-picked SIFT-only commits from `jeffdaily/rocm-sift-gpu`.
- No new files for Track D — build/run verification only, results logged to `docs/rocm-integration.md`.

---

## Track A, Task 1: Extend Caspar-HIP smoke test to the solver path

**Files:**
- Create: `~/git/symforce-rocm/hip_solver_smoke.py`
- Modify: `~/git/symforce-rocm/docs/rocm-integration.md`

**Interfaces:**
- Consumes: `symforce-rocm` `hip-integration` branch as it stands (HIP Caspar build verified for the toy accessor kernel — commit `42bf1a18`).
- Produces: confirmation (or a documented failure) that Caspar's actual solver — `GraphSolver`, PCG, `SortIndices` — works correctly on HIP/gfx1151. Track B, Task 3 depends on this passing before wiring Caspar-HIP into COLMAP.

- [ ] **Step 1: Find a small existing BAL-style or synthetic-graph example to adapt**

```bash
cd ~/git/symforce-rocm
find symforce/caspar -iname "*.py" | xargs grep -l "GraphSolver\|SortIndices\|solver_tools" 2>/dev/null
find symforce/caspar/examples -maxdepth 2
```

Look for an existing example that builds a small multi-factor graph and calls the solver (not just a single kernel) — e.g. a tiny bundle-adjustment or least-squares example under `symforce/caspar/examples/`. If one exists, adapt it (same pattern as `hip_smoke.py` adapting `kernel_example/gen_and_run.py`); if none exists, use `caslib.compile(..., use_hip=True, hip_arch="gfx1151")` on a `CasparLibrary` built with `register_camera_model`-style factor registration (mirroring the `PinholeCalib`/`opencv_core` pattern from `caspar_generate.py`) plus a call into `GraphSolver`'s Python-exposed solve entry point — find the actual binding name via `grep -rn "GraphSolver" symforce/caspar/ --include=*.py`.

- [ ] **Step 2: Write a small synthetic problem with a known-correct answer**

Construct a trivial multi-camera, multi-point bundle-adjustment-shaped problem (e.g. 3-5 cameras, 10-20 points, synthetic noise-free projections) where the correct optimized parameters are known in advance (you generated them), so PCG convergence and the final solved values can be checked against ground truth, not just "did it run."

```python
# hip_solver_smoke.py (sketch — fill in with the real API found in Step 1)
import numpy as np
import torch
from symforce.caspar import CasparLibrary
# ... build caslib with a small factor graph, register real kernels ...
caslib.compile(out_dir, use_hip=True, hip_arch="gfx1151")
# ... load compiled lib, run solve, compare final parameter values against
#     the known ground truth within a reasonable tolerance (e.g. 1e-3) ...
```

- [ ] **Step 3: Run inside the ROCm container, verify convergence and correctness**

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 39 --group-add 105 \
  --security-opt label=disable \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -v ~/git/symforce-rocm:/workspace/symforce-rocm \
  -w /workspace/symforce-rocm \
  <the pip-install-ready image from the prior milestone's Task 2, or rebuild if unavailable> \
  python3 hip_solver_smoke.py
```

Expected: the solver converges (PCG relative error below its configured threshold, or a bounded number of LM iterations) and the final solved parameters match the known ground truth within tolerance. If it fails to converge or produces wrong values: this is a real correctness gap in the HIP port's solver path (as opposed to the individual kernels), not a smoke-test triviality — stop, write the full failure mode (does it crash, diverge, or converge to wrong values?) to `docs/rocm-integration.md` under `## BLOCKED`, and report BLOCKED rather than guessing at a fix.

- [ ] **Step 4: Log and commit**

```bash
cd ~/git/symforce-rocm
# append to docs/rocm-integration.md: what was tested (problem size, factor
# types, solver settings), pass/fail, final parameter error vs. ground truth
git add hip_solver_smoke.py docs/rocm-integration.md
git commit -m "test: verify Caspar-HIP solver path (GraphSolver/PCG) on gfx1151"
git push origin hip-integration
```

---

## Track B, Task 1: Port caspar-opencv's C++ dispatch changes into colmap-rocm

**Files:**
- Modify: `~/git/colmap-rocm/src/colmap/estimators/bundle_adjustment_caspar.cc`
- Modify: `~/git/colmap-rocm/src/colmap/estimators/caspar/caspar_model_adapter.h`

**Interfaces:**
- Consumes: `rosbag-colmap-pipeline/docker/patches/caspar-opencv/bundle_adjustment_caspar.cc` and `caspar_model_adapter.h` (read-only reference — do not modify that repo), `colmap-rocm`'s current `hip-integration` tip.
- Produces: `colmap-rocm` with `CameraModelId::kOpenCV` recognized by Caspar's dispatch logic. Track B, Task 2 (regeneration) and Task 4 (HIP compat header) both need this in place first, since the generated kernel tree and the C++ dispatch code must agree on which models exist.

- [ ] **Step 1: Diff the reference patch against colmap-rocm's current files**

```bash
diff ~/git/rosbag-colmap-pipeline/docker/patches/caspar-opencv/bundle_adjustment_caspar.cc \
     ~/git/colmap-rocm/src/colmap/estimators/bundle_adjustment_caspar.cc
diff ~/git/rosbag-colmap-pipeline/docker/patches/caspar-opencv/caspar_model_adapter.h \
     ~/git/colmap-rocm/src/colmap/estimators/caspar/caspar_model_adapter.h
```

The reference patch targets COLMAP `4.1.1`; `colmap-rocm` is on current `main`. Read both diffs in full before touching anything — the base files may have drifted (new dispatch sites added, refactored structure) since `4.1.1`. Do not assume the reference patch applies as a mechanical `git apply`.

- [ ] **Step 2: Apply the OpenCV additions, re-deriving positions where the base has drifted**

Using the reference patch's README.md (`rosbag-colmap-pipeline/docker/patches/caspar-opencv/README.md`) as the specification of *what* needs to change (the `BuildSizing()` `kOpenCV` block, `CasparSolverSizing`'s `num_opencv_*` fields, `OpenCVAdapter` class, `CreateCasparAdapter()`'s new case, `CreateSolver()`'s positional-argument reordering), apply the equivalent changes to `colmap-rocm`'s current files. If `colmap-rocm`'s `CreateSolver()` signature differs from the reference's, the positional-argument analysis (which node types sort alphabetically before which) needs to be redone against `colmap-rocm`'s own generated `solver.h`, not copied from the reference's — this was flagged as the single highest-risk part of the original patch precisely because it's silently wrong-compiling if done by pattern-matching instead of verification.

- [ ] **Step 3: Verify — do not skip this, it's what the original patch's README calls out as critical**

Cross-check every `s.<Method>()` call added to the adapter class against what will actually be in the regenerated `solver.h` (this requires Task 2's regeneration to have happened first for a full check — if doing Task 1 before Task 2, do a best-effort check against the CURRENT `generated/f32/solver.h` and re-verify after Task 2 regenerates it). Cross-check `CreateSolver()`'s full positional-argument list by name AND position, not just by count.

- [ ] **Step 4: Commit**

```bash
cd ~/git/colmap-rocm
git add src/colmap/estimators/bundle_adjustment_caspar.cc src/colmap/estimators/caspar/caspar_model_adapter.h
git commit -m "feat(caspar): port OpenCV camera-model dispatch from rosbag-colmap-pipeline's caspar-opencv patch"
git push origin hip-integration
```

Do not build yet — `generated/f32/` doesn't have OpenCV kernels until Task 2.

---

## Track B, Task 2: Regenerate Caspar kernels with OpenCV support

**Files:**
- Modify: `~/git/colmap-rocm/src/thirdparty/Symforce-Caspar/caspar_generate.py`
- Modify: `~/git/colmap-rocm/src/thirdparty/Symforce-Caspar/generated/{f32,f64}/` (regenerated wholesale)

**Interfaces:**
- Consumes: `rosbag-colmap-pipeline/docker/patches/caspar-opencv/caspar_generate.py` (reference for the `opencv_core` symbolic function and `register_camera_model` calls), Track B Task 1's dispatch changes (so the generated tree and dispatch code agree).
- Produces: a `generated/{f32,f64}` tree containing PINHOLE, SIMPLE_RADIAL, and OPENCV kernels (still CUDA-spelling — HIP compilation is Task 4). Task 3 (build+run verification, still CUDA at this point since HIP isn't wired yet) and Task 4 (HIP wiring) both depend on this.

- [ ] **Step 1: Port the `opencv_core` additions into colmap-rocm's `caspar_generate.py`**

```bash
diff ~/git/rosbag-colmap-pipeline/docker/patches/caspar-opencv/caspar_generate.py \
     ~/git/colmap-rocm/src/thirdparty/Symforce-Caspar/caspar_generate.py
```

Same caveat as Track B Task 1: the reference targets `4.1.1`, `colmap-rocm` is on current `main` — check whether `caspar_generate.py`'s structure (the `Pinhole*`/`SimpleRadial*` blocks this patch mirrors) has changed. Add `OpenCVPose`, `ConstOpenCVPose`, `OpenCVCalib(sf.V8)`, `ConstOpenCVCalib`, `ConstOpenCVSensorFromRig`, `opencv_core(...)` (COLMAP's exact radial+tangential distortion formula — cross-check against `colmap-rocm/src/colmap/sensor/models.h`'s current `OpenCVCameraModel::Distortion`, don't assume it's unchanged from `4.1.1`), `FIXABLE_OPENCV`, and the `register_camera_model(caslib, "opencv", opencv_core, FIXABLE_OPENCV, include_all_fixed=True)` call plus the split-variant registration (11 more, per the reference's README — "required, not optional" because COLMAP's default `refine_principal_point=false` hits the split variant).

- [ ] **Step 2: Generate and check the shared-memory budget**

```bash
cd ~/git/colmap-rocm/src/thirdparty/Symforce-Caspar
python3 caspar_generate.py /tmp/caspar-regen-f32 f32
```

This needs `symforce` installed (`pip install -e .` from `~/git/symforce-rocm`, or a plain `pip install symforce` if not testing the HIP-specific parts of the generator itself — codegen doesn't need HIP, only the later compile step does). Expected: exit 0, no shared-memory-budget error (COLMAP's 48KB-per-block check, called out in the reference README as the real go/no-go check — OpenCV's calib node is 8 floats vs. PINHOLE's 4, but the reference's prior verification found this still fits the fixed-size buffer). If it fails here: stop, report the exact error (kernel size vs. budget), do not attempt to force a fit by further splitting the calib node without discussing scope first — that expands this task significantly.

- [ ] **Step 3: Diff against the existing tree, then replace it**

```bash
diff -rq /tmp/caspar-regen-f32 ~/git/colmap-rocm/src/thirdparty/Symforce-Caspar/generated/f32
```

Confirm the diff is what's expected (new `*opencv*` files, existing `*pinhole*`/`*simple_radial*` files unchanged or only trivially different e.g. from an unrelated symforce/dependency version bump — investigate if PINHOLE/SIMPLE_RADIAL kernels changed substantively, that would be a regression signal). Then:

```bash
rm -rf ~/git/colmap-rocm/src/thirdparty/Symforce-Caspar/generated/f32
mv /tmp/caspar-regen-f32 ~/git/colmap-rocm/src/thirdparty/Symforce-Caspar/generated/f32
```

Repeat Steps 2-3 for `f64` if `CASPAR_USE_DOUBLE` is exercised anywhere in this branch (check `grep -rn CASPAR_USE_DOUBLE ~/git/colmap-rocm/CMakeLists.txt` — if it's off by default and untested, f32-only regeneration may be sufficient for now; note the decision either way).

- [ ] **Step 4: Commit**

```bash
cd ~/git/colmap-rocm
git add src/thirdparty/Symforce-Caspar/
git commit -m "feat(caspar): regenerate kernels with native OPENCV camera-model support"
git push origin hip-integration
```

---

## Track B, Task 3: (cut from critical path — folded into Track D as an optional extra check)

**Amended before dispatch:** this task originally asked to CUDA-build-verify the
OpenCV-augmented tree on the cluster before adding HIP on top, and pointed at
Track D's dispatch mechanism — which created a circular "coordinate with each
other" dependency (Track D's own step said to check whether Track B had landed
first). Track B Task 4 (HIP wiring) does not actually need this CUDA baseline to
proceed — it builds and debugs directly against `HIP_ENABLED=ON` on this gfx1151
host. Cut: Track B now runs Task 1 → Task 2 → **Task 4** → Task 5. If Track D
(independent, see below) happens to run and its dispatch mechanism turns out to
be readily reusable, use it there to *additionally* verify the OpenCV-augmented
CUDA build as a bonus check — but nothing in Track B blocks on it.

---

## Track B, Task 4: Add HIP compilation to CASPAR_ENABLED

**Files:**
- Modify: `~/git/colmap-rocm/cmake/FindDependencies.cmake`
- Modify: `~/git/colmap-rocm/src/thirdparty/Symforce-Caspar/generated/f32/CMakeLists.txt`
- Create: `~/git/colmap-rocm/src/colmap/estimators/caspar/caspar_hip_compat.h` (or adjust name/location to match this codebase's actual conventions once you're looking at the real file layout)

**Interfaces:**
- Consumes: Track A Task 1 (verified HIP solver-path mappings, informing what the compat header needs to cover), Track B Task 3 (known-good CUDA baseline to compare against).
- Produces: `colmap-rocm` buildable with `-DCASPAR_ENABLED=ON -DHIP_ENABLED=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151`. Task 5 (real BA run on gfx1151) depends on this.

- [ ] **Step 1: Add the HIP branch to CASPAR_ENABLED's arch guard**

```bash
cd ~/git/colmap-rocm
grep -n "CASPAR_ENABLED" cmake/FindDependencies.cmake
```

The current guard (from the prior milestone's investigation) is inside `if(CUDA_ENABLED AND CUDA_FOUND)` with a `CMAKE_CUDA_ARCHITECTURES`-based compute-capability check (`>= 70`). Add an equivalent `elseif(HIP_ENABLED)` branch (or restructure to an `if/elseif` covering both) that validates `CMAKE_HIP_ARCHITECTURES` is set to something Caspar-HIP has actually been tested on (gfx1151, per this session's work — don't silently accept an untested arch without at least a warning, mirroring the existing CUDA branch's pattern).

- [ ] **Step 2: Add LANGUAGE HIP to the generated kernel CMakeLists**

```bash
grep -n "LANGUAGE\|\.cu" src/thirdparty/Symforce-Caspar/generated/f32/CMakeLists.txt | head -20
```

Mirror the mechanism PR #4420 already uses for `patch_match_cuda.cu` — `set_source_files_properties(<files> PROPERTIES LANGUAGE HIP)` when `HIP_ENABLED`, applied to all the generated `.cu` files (now including the new `*opencv*` ones from Task 2).

- [ ] **Step 3: Write the Caspar-specific cooperative_groups compat header**

Based on Track A Task 1's now-verified solver path (which constructs are actually exercised — `cg::reduce`, `cg::labeled_partition`, shared-memory atomics, per SymForce PR #465's own mappings) plus a scan of what's actually used in the regenerated tree:

```bash
grep -rn "cooperative_groups\|cg::" src/thirdparty/Symforce-Caspar/generated/f32/ | grep -oP 'cg::\w+' | sort -u
```

Write a header mapping each construct found to its HIP equivalent (rocPRIM/hipCUB-backed where applicable, informed by — not copy-pasted from — `symforce-rocm`'s `cuda_to_hip.h` mappings for the analogous constructs, since the exact API surface may differ between SymForce's own runtime headers and COLMAP's vendored generated kernels). Include it from wherever the generated `.cu` files currently `#include <cooperative_groups.h>` when `COLMAP_HIP_ENABLED` is defined (mirror `cuda_to_hip.h`'s existing inclusion pattern for PatchMatch).

- [ ] **Step 4: Build and check for compile errors**

```bash
docker build -t colmap-rocm:caspar-hip --build-arg CMAKE_EXTRA_ARGS="-DCASPAR_ENABLED=ON" ~/git/colmap-rocm
```

(This needs `-DHIP_ENABLED=ON -DCUDA_ENABLED=OFF` already baked into the Dockerfile's base cmake invocation from the prior milestone — `CASPAR_ENABLED=ON` is the new addition via `CMAKE_EXTRA_ARGS`.) Expected: clean build. If HIP-specific compile errors occur in the generated kernels (missing compat-header coverage for a construct Step 3 missed): iterate Step 3, don't paper over with suppression flags. If a construct proves genuinely hard to map (e.g. a butterfly-reduction pattern PR #465's own comments flag as broken for non-contiguous groups, per the prior milestone's final review of `symforce-rocm`): stop, document the specific construct and why, report BLOCKED rather than shipping a silently-wrong reduction.

- [ ] **Step 5: Commit**

```bash
cd ~/git/colmap-rocm
git add cmake/FindDependencies.cmake src/thirdparty/Symforce-Caspar/generated/f32/CMakeLists.txt src/colmap/estimators/caspar/
git commit -m "feat(caspar): add HIP compilation path to CASPAR_ENABLED (gfx1151)"
git push origin hip-integration
```

---

## Track B, Task 5: Verify Caspar-HIP bundle adjustment on real data, gfx1151

**Files:** none new — verification only.

**Interfaces:**
- Consumes: Track B Task 4's build.
- Produces: confirmation Caspar-HIP BA produces correct results on this machine, closing out the prior milestone's Task 6 deferral for real.

- [ ] **Step 1: Build the full image and confirm the Caspar backend is selectable**

```bash
docker build -t colmap-rocm:hip --build-arg CMAKE_EXTRA_ARGS="-DCASPAR_ENABLED=ON" ~/git/colmap-rocm
docker run --rm --entrypoint bash colmap-rocm:hip -c "colmap mapper --help" 2>&1 | grep -i caspar
```

- [ ] **Step 2: Run mapper with Caspar-HIP BA on the OpenCV-model dataset from Track B Task 3, or a fresh one on this machine**

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 39 --group-add 105 \
  --security-opt label=disable \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e QT_QPA_PLATFORM=offscreen \
  -v <dataset-dir>:/workspace/data \
  colmap-rocm:hip \
  mapper --database_path /workspace/data/db.sqlite --image_path /workspace/data/images \
  --output_path /workspace/data/sparse_caspar \
  --Mapper.ba_global_backend CASPAR  # or the actual flag name confirmed in Step 1
```

- [ ] **Step 3: Compare against the CPU/Ceres baseline**

Run the same dataset through `mapper` with the default (Ceres) BA backend, compare final reprojection error / registered-image count / pose agreement, same standard the prior milestone's Task 7 and `rosbag-colmap-pipeline`'s caspar-opencv verification both used. Not required to match bit-for-bit — order-of-magnitude agreement, same registered-image count, no gross divergence.

- [ ] **Step 4: Log final results and commit**

```bash
cd ~/git/colmap-rocm
git add docs/rocm-integration.md
git commit -m "docs: verify Caspar-HIP bundle adjustment on real data, gfx1151 — closes prior Task 6 deferral"
git push origin hip-integration
```

---

## Track C, Task 1: Cherry-pick SIFT-only commits from jeffdaily/rocm-sift-gpu

**Files:** whatever the identified SIFT-only commits touch — determined in Step 1, not assumed in advance.

**Interfaces:**
- Consumes: `colmap-rocm`'s `hip-integration` tip (independent of Track A/B's state — can run any time).
- Produces: HIP-accelerated SIFT extraction/matching, or a documented reason it's still not feasible.

- [ ] **Step 1: Identify which of the 10 commits are SIFT-only vs. PatchMatch-duplicating**

```bash
cd ~/git/colmap-rocm
git fetch jeffdaily rocm-sift-gpu
git log --oneline upstream/main..jeffdaily/rocm-sift-gpu
git show --stat <each commit>
```

The prior rebase attempt aborted because the FIRST commit touched `cuda_flip/rotate/texture/transpose.h`, `gpu_mat.h`, `patch_match_cuda.h`, `util/cuda*.{cc,h}` — files PR #4420 already HIP-ported. Classify each of the 10 commits: does it touch only SIFT-specific files (`feature/sift*.{cc,h,cu}`, SiftGPU-related), or does it also touch the PatchMatch-overlapping compat layer? List this out explicitly before attempting anything.

- [ ] **Step 2: Attempt cherry-picking only the SIFT-only commits**

```bash
git checkout -b colmap-sift-cherrypick hip-integration
git cherry-pick <SIFT-only commit SHAs, in original order>
```

For commits that touch BOTH SIFT and the already-ported compat layer: do not cherry-pick them wholesale. Either split the commit's changes manually (extract just the SIFT-relevant hunks) or skip that commit and note what functionality is lost as a result. Use the same conflict-resolution rule as before: keep `hip-integration`'s existing (already-HIP-ported) compat-layer code, don't let a stale copy from `rocm-sift-gpu` overwrite it.

- [ ] **Step 3: Build and test**

```bash
docker build -t colmap-rocm:hip-sift ~/git/colmap-rocm
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 39 --group-add 105 \
  --security-opt label=disable -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e QT_QPA_PLATFORM=offscreen \
  -v <dataset-dir>:/workspace/data colmap-rocm:hip-sift \
  feature_extractor --database_path /workspace/data/db.sqlite --image_path /workspace/data/images \
  --FeatureExtraction.use_gpu 1
```

Expected: completes without HIP/OpenGL errors, produces real keypoints/descriptors (nonzero counts, comparable to the CPU SIFT run from the prior milestone's Task 7). If it still fails to build or run correctly after a genuine, scoped attempt: abort and document, same as the prior milestone's Task 5 fallback — this is a legitimate outcome, not a failure of the task.

- [ ] **Step 4: Fold onto hip-integration (success path) or document fallback, then log**

```bash
# success path:
cd ~/git/colmap-rocm
git checkout hip-integration
git log --oneline hip-integration colmap-sift-cherrypick  # confirm hip-integration's tip is an ancestor of the cherrypick branch before resetting
git reset --hard colmap-sift-cherrypick
git branch -D colmap-sift-cherrypick
git push --force-with-lease origin hip-integration
# either path:
# append to docs/rocm-integration.md: which commits were cherry-picked or
# skipped and why, build/test result
git add docs/rocm-integration.md
git commit -m "docs: log HIP SIFT cherry-pick attempt outcome"
git push origin hip-integration
```

---

## Track D, Task 1: CUDA regression check on cps-gpu-cluster

**Files:** none new — verification only.

**Interfaces:**
- Consumes: `colmap-rocm`'s `hip-integration` tip.
- Produces: confirmation `CUDA_ENABLED=ON` still compiles and runs on this branch, independent of any HIP-specific changes. Can run any time, in parallel with everything else.

- [ ] **Step 1: Confirm cluster access and dispatch mechanism**

```bash
kubectl get nodes -l accelerator=nvidia
kubectl get gitrepo -n fleet-local
```

Per `cps-gpu-cluster`'s own `CLAUDE.md`: GPU workers are at `.38/.43/.40/.41`, labeled `accelerator=nvidia,gpu-model=a100`. Check `~/git/rosbag-colmap-pipeline/docs/cluster-dispatch.md` for the exact `ablator`-based dispatch pattern already proven for COLMAP CUDA builds on this cluster.

**Resolve this before anything else in this task:** `ablator` (per `docs/cluster-dispatch.md`) dispatches a *prebuilt image tag* (`cuda-caspar-opencv-v2`) at a workspace — it does not itself build images. Find out where that image was actually built (`~/git/rosbag-colmap-pipeline/configs/ablator.toml`, `.github/workflows/`, or any build-related script/doc in that repo) before assuming `ablator`'s dispatch mechanism is reusable as-is for building *new* `colmap-rocm` source. If the build happens in CI or on a separate dev box with a CUDA toolchain, this task's real first step is "find or create a way to build a CUDA image from `colmap-rocm` source with access to a CUDA toolchain" — which may mean an in-cluster Kaniko/buildkit build (a different mechanism from `ablator`'s job-dispatch, since `cps-gpu-cluster`'s K3s pods run workloads, not necessarily privileged image builds) rather than local `docker build` (this gfx1151 host has no CUDA toolchain to build against natively either). If no readily-available CUDA build path exists within a reasonable scope: stop, document exactly what's missing, and report BLOCKED rather than inventing new cluster infrastructure — this is explicitly the lowest-value of the four sub-projects (it verifies a CMake merge conflict that only affects CUDA users of this fork, which is nobody today) and isn't worth a large infra detour.

- [ ] **Step 2: Build with CUDA_ENABLED=ON on a GPU worker**

Adapt whatever `Dockerfile.cuda`-equivalent pattern `rosbag-colmap-pipeline` uses, but building from `colmap-rocm`'s local source (`COPY . /opt/colmap_src`, matching this session's own Dockerfile pattern) instead of a `git clone --branch <tag>`, with `-DCUDA_ENABLED=ON -DHIP_ENABLED=OFF -DCASPAR_ENABLED=ON` (include Caspar here too if Track B has landed by this point — otherwise just verify the base CUDA build).

- [ ] **Step 3: Smoke-test on an A100**

Run a minimal `colmap --help` plus one real command (e.g. `feature_extractor` on a tiny image set) to confirm the binary actually executes on real CUDA hardware, not just compiles.

- [ ] **Step 4: Log and commit**

```bash
cd ~/git/colmap-rocm
# append to docs/rocm-integration.md: cluster dispatch command, build result,
# smoke-test result, explicit note this confirms Task 1's hand-resolved
# rebase conflict (CMakeLists.txt.jinja-equivalent arch-list merge) didn't
# regress the CUDA path
git add docs/rocm-integration.md
git commit -m "docs: verify CUDA_ENABLED build on cps-gpu-cluster A100 — confirms no regression from Task 1's rebase conflict resolution"
git push origin hip-integration
```

---

## Self-Review Notes

- **Spec coverage:** Track A covers spec sub-project 1 (solver verification). Track B Tasks 1-5 cover sub-project 2 (COLMAP wiring merged with caspar-opencv) in the order the spec's "Key clarification" section implies (dispatch code → regenerate → CUDA-verify baseline → HIP-wire → verify on real data). Track C covers sub-project 3 (HIP SIFT), with an explicit re-scoped approach (only SIFT-only commits) rather than repeating the whole-branch rebase that already failed once. Track D covers sub-project 4 (CUDA regression check), explicitly reusing `rosbag-colmap-pipeline`'s proven cluster-dispatch pattern rather than re-deriving one. Out-of-scope items from the spec are not tasked here.
- **Placeholder scan:** no bare TBD/TODO. The two spots requiring runtime discovery (Track B Task 4 Step 1's arch-guard exact structure, Track B Task 5 Step 1's exact BA-backend flag name) are framed as "confirm via inspection," consistent with how the prior milestone's plan handled the same kind of not-yet-verified-until-you-look details — not a placeholder for missing design, a acknowledgment that the design correctly depends on live inspection.
- **Type/name consistency:** file paths, branch names (`hip-integration`), Docker image tags, and env vars/flags match the prior milestone's established conventions throughout.
- **Cross-task dependency correctness:** Track A blocks Track B Task 4 (compat header informed by verified solver mappings) and implicitly Task 5 (don't trust BA results from an unverified solver). Track B Task 3 and Track D Task 1 both need cluster CUDA access — Task 3 explicitly says to reuse Track D's mechanism rather than duplicate it; whichever runs first should establish the pattern for the other.

## Parallelization Notes (for subagent-driven-development)

- **Track A** (`symforce-rocm`) and **Track C** (`colmap-rocm`, SIFT cherry-pick) and **Track D** (`colmap-rocm` + cluster, CUDA check) have no file overlap with each other and can all start immediately, concurrently — one worktree per repo/concern as established in the prior milestone (Track A in `symforce-rocm`, Track C and D both touch `colmap-rocm` but Track D is verification-only with no local file changes until its final docs commit, so it's safe to interleave with Track C in the same worktree, sequenced rather than truly parallel to avoid docs-file merge noise).
- **Track B** is strictly sequential within itself (Task 1 → 2 → 3 → 4 → 5) and Task 1 should not start until Track A Task 1 has at least produced its verification result (even if the result is BLOCKED — Track B Task 4's compat header needs to know what's actually been verified, or explicitly proceed with a documented risk if Track A is still in progress and time is tight).
- Recommended dispatch order: Track A, Track C, Track D all start together (3 concurrent efforts, 2 in the same `colmap-rocm` worktree sequenced). Track B starts once Track A completes, using the same `colmap-rocm` worktree Track C/D used (after their work lands, to avoid divergent branch state).
