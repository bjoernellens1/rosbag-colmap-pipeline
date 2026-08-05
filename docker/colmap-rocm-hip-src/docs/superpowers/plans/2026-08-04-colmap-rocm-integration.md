# COLMAP ROCm Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a full incremental SfM pipeline (feature extraction → matching → mapper → dense stereo) running end-to-end on this gfx1151 (Radeon 8060S) machine, GPU-accelerated via HIP for bundle adjustment (Caspar) and dense PatchMatch stereo, by integrating existing upstream ROCm work rather than writing new kernels.

**Architecture:** Two forked repos, each with an `hip-integration` branch layering existing open PRs on top of current upstream `main`: `~/git/symforce-rocm` (Caspar HIP backend, PR #465) and `~/git/colmap-rocm` (PatchMatch HIP, PR #4420, plus GPU SIFT, `jeffdaily/colmap:rocm-sift-gpu`). `colmap-rocm` builds inside a Docker image `FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`, consuming a HIP-built Caspar library from `symforce-rocm` as a separate build stage. Two independent tracks (SymForce, COLMAP-minus-SIFT) can proceed in parallel worktrees; SIFT rebase and final wiring are sequential dependents.

**Tech Stack:** CMake 3.21+, HIP/ROCm 7.2, hipcc, Docker, Python (SymForce codegen), C++17, git rebase.

## Global Constraints

- Target GPU architecture: `gfx1151` only (this machine). Do not attempt multi-arch in this plan.
- Base Docker image: `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` (already pulled locally). Do not substitute a different ROCm base image.
- `HSA_OVERRIDE_GFX_VERSION=11.5.1` must be set in the container per this machine's established convention (`gaussian-splatting-lightning`, `splatograph`).
- CUDA and HIP are mutually exclusive in COLMAP's CMake (`-DCUDA_ENABLED=OFF -DHIP_ENABLED=ON`) — never enable both.
- All rebases target current `upstream/main` at time of execution, not the PRs' original base commits.
- Every task's integration branch is `hip-integration` on `origin` (the `bjoernellens1` fork) — push after each task's commit.
- Do not push to `upstream` (`colmap/colmap`, `symforce-org/symforce`) or open PRs against them — out of scope per spec.
- No GLOMAP global-positioning ROCm work, no PyCOLMAP wheels, no MIGraphX/ONNX path in this plan — see spec's Out of Scope section.

---

## File Structure

- `~/git/symforce-rocm/` — forked SymForce repo, branch `hip-integration`. No new files beyond what PR #465 introduces (a HIP compatibility layer under SymForce's Caspar codegen/runtime dirs — exact paths determined by the PR's own diff, inspected in Task 1).
- `~/git/colmap-rocm/` — forked COLMAP repo, branch `hip-integration`.
  - Modified: `CMakeLists.txt`, `cmake/FindDependencies.cmake`, `src/colmap/util/*cuda*`, `src/colmap/mvs/*` (per PR #4420's diff).
  - New: `src/colmap/util/cuda_to_hip.h` (from PR #4420).
  - New: `Dockerfile` at repo root — builds the HIP-enabled COLMAP image, consuming SymForce's HIP Caspar build.
  - New: `docs/rocm-integration.md` — running notes: build commands, known issues, test results per task (append-only log, not prose you have to keep consistent — just append dated entries).

---

## Task 1: Rebase SymForce PR #465 onto current main

**Files:**
- Modify: whatever `git diff upstream/main jeffdaily/moat-port` touches in `~/git/symforce-rocm` (inspect first — see Step 1).
- Test: none yet (build verification is Task 2).

**Interfaces:**
- Consumes: nothing (first task in this track).
- Produces: `~/git/symforce-rocm` branch `hip-integration` containing PR #465's HIP support, rebased onto current `upstream/main`, pushed to `origin/hip-integration`. This is what Task 2 builds from.

- [ ] **Step 1: Inspect the PR's diff before rebasing**

```bash
cd ~/git/symforce-rocm
git fetch jeffdaily moat-port
git diff upstream/main jeffdaily/moat-port --stat
```

Read the full list of changed files. Note any files under codegen template directories (these are most likely to conflict with upstream changes).

- [ ] **Step 2: Attempt the rebase**

```bash
cd ~/git/symforce-rocm
git checkout -b symforce-hip-rebase jeffdaily/moat-port
git rebase upstream/main
```

Expected: PR #465's `mergeable_state` was `dirty` as of 2026-08-04, so expect conflicts. Resolve each conflict by keeping upstream's non-HIP-related changes and re-applying the HIP-specific hunks from the PR commit. Do not silently drop HIP code to resolve a conflict. If a conflict can't be resolved confidently: stop, write the full conflicting hunks, the file(s) involved, and exactly why you're unsure to `docs/rocm-integration.md` under a `## BLOCKED` heading, commit that file on a throwaway branch or just leave it uncommitted, and end the task here — do not guess and do not proceed to Step 3. This is a real stop condition, not a formality.

- [ ] **Step 3: Fold the rebased commits onto `hip-integration`**

```bash
git checkout hip-integration
git reset --hard symforce-hip-rebase
git branch -D symforce-hip-rebase
git log --oneline upstream/main..HEAD
```

Expected: 2 commits (matching PR #465's commit count) on top of current `upstream/main`.

- [ ] **Step 4: Push**

```bash
git push --force-with-lease origin hip-integration
```

- [ ] **Step 5: Log the outcome**

Append to `~/git/symforce-rocm/docs/rocm-integration.md` (create if absent):

```markdown
## 2026-08-04 — Task 1: Rebase PR #465

Rebased jeffdaily/moat-port (PR symforce-org/symforce#465) onto upstream/main
at commit <upstream main SHA>. Conflicts: <list files, or "none">.
Resolution notes: <one line per nontrivial conflict, or "clean rebase">.
```

```bash
git add docs/rocm-integration.md
git commit -m "docs: log PR #465 rebase onto upstream/main"
git push origin hip-integration
```

---

## Task 2: Build and smoke-test HIP Caspar in SymForce

**Files:**
- Test: none new — this is a build/run smoke test, not a unit test addition.

**Interfaces:**
- Consumes: `~/git/symforce-rocm` branch `hip-integration` (Task 1's output).
- Produces: confirmation that `compile_caspar_library(use_hip=True, hip_arch="gfx1151")` builds a loadable `.so` inside a ROCm container. Task 5 (Docker wiring) depends on knowing this command works and what it outputs.

- [ ] **Step 1: Find the actual HIP compile entry point**

```bash
cd ~/git/symforce-rocm
grep -rn "use_hip" --include="*.py" . | head -20
```

Confirm the exact function signature and default `hip_arch` handling — the spec's example (`compile_caspar_library(caslib, output_dir, use_hip=True, hip_arch="gfx1151")`) is from prior research, not verified against this codebase. If the signature differs, use the real one and note the difference in the log (Step 4).

- [ ] **Step 2: Run the build inside the ROCm container**

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v ~/git/symforce-rocm:/workspace/symforce-rocm \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
  bash -c "cd /workspace/symforce-rocm && pip install -e . && python3 -c \"
from symforce.codegen import <actual caspar module found in Step 1>
# call the real compile_caspar_library-equivalent found in Step 1
# with a trivial example graph, use_hip=True, hip_arch='gfx1151'
\""
```

Adjust the inline Python to the actual API surface found in Step 1 — do not guess at a nonexistent function name. If no example/test graph is bundled with the PR, use whatever minimal example SymForce's own Caspar tests use (check `test/` or `examples/` for `caspar` in the filename).

- [ ] **Step 3: Verify the output**

Confirm the build produces a `.so`/`.hsaco`-backed library without errors, and that a trivial Python import/dlopen of it succeeds inside the container. If it fails: stop, write the full error and command that produced it to `docs/rocm-integration.md` under `## BLOCKED`, and end the task here rather than attempting speculative fixes.

- [ ] **Step 4: Log the outcome**

```bash
cd ~/git/symforce-rocm
# append build command, output summary, and pass/fail to docs/rocm-integration.md
git add docs/rocm-integration.md
git commit -m "docs: log HIP Caspar build smoke test"
git push origin hip-integration
```

---

## Task 3: Rebase COLMAP PR #4420 onto current main

**Files:**
- Modify: `CMakeLists.txt`, `cmake/FindDependencies.cmake`, `src/colmap/exe/CMakeLists.txt`, `src/colmap/exe/mvs.cc`, `src/colmap/mvs/CMakeLists.txt`, `src/colmap/mvs/cuda_flip.h`, `src/colmap/mvs/cuda_rotate.h`, `src/colmap/mvs/cuda_texture.h`, `src/colmap/mvs/cuda_transpose.h`, `src/colmap/mvs/gpu_mat.h`, `src/colmap/mvs/gpu_mat_test.cu`, `src/colmap/mvs/patch_match_cuda.cu`, `src/colmap/mvs/patch_match_cuda.h`, `src/colmap/util/CMakeLists.txt`, `src/colmap/util/cuda.cc`, `src/colmap/util/cudacc.cc`, `src/colmap/util/cudacc.h`, `src/colmap/util/version.cc.in`, `doc/install.rst`, `README.md`.
- New: `src/colmap/util/cuda_to_hip.h`.

**Interfaces:**
- Consumes: `~/git/colmap-rocm` branch `hip-integration` (already created, tracking `upstream/main`, per prior setup).
- Produces: `HIP_ENABLED` CMake option (mutually exclusive with `CUDA_ENABLED`) wired through the build; `-DCUDA_ENABLED=OFF -DHIP_ENABLED=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151` becomes a valid CMake invocation. Task 4 builds using exactly this flag combination.

- [ ] **Step 1: Inspect the PR's diff**

```bash
cd ~/git/colmap-rocm
git fetch ishengnan rocm-support
git diff upstream/main ishengnan/rocm-support --stat
```

- [ ] **Step 2: Rebase**

```bash
git checkout -b colmap-patchmatch-rebase ishengnan/rocm-support
git rebase upstream/main
```

Expected: `mergeable: true` on GitHub as of 2026-08-04, so this should be closer to clean than Task 1's SymForce rebase, but "mergeable" per GitHub's API means no textual conflict against `main` at check time — it can still drift before you run this. Resolve any conflicts favoring upstream's non-HIP changes plus the PR's HIP-specific hunks, same rule as Task 1.

- [ ] **Step 3: Fold onto `hip-integration` and push**

```bash
git checkout hip-integration
git reset --hard colmap-patchmatch-rebase
git branch -D colmap-patchmatch-rebase
git log --oneline upstream/main..HEAD
git push --force-with-lease origin hip-integration
```

Expected: 8 commits (PR #4420's commit count) on top of current `upstream/main`.

- [ ] **Step 4: Log the outcome**

```bash
mkdir -p docs
# append to docs/rocm-integration.md: rebase result, conflicts, resolutions
git add docs/rocm-integration.md
git commit -m "docs: log PR #4420 rebase onto upstream/main"
git push origin hip-integration
```

---

## Task 4: Dockerfile + build + PatchMatch-HIP smoke test

**Files:**
- Create: `Dockerfile` (repo root).
- Test: manual run of `colmap patch_match_stereo` against a real dataset (no unit test framework applies to a CLI smoke test — this is the verification step).

**Interfaces:**
- Consumes: `~/git/colmap-rocm` branch `hip-integration` (Task 3's output). Does NOT yet consume SymForce's Caspar build (that's Task 5).
- Produces: a working `colmap-rocm:hip` Docker image with a HIP-enabled `colmap` binary. Task 6 (full pipeline run) uses this same image (extended in Task 5) as its runtime environment.

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# syntax=docker/dockerfile:1.7
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ROCM_ARCH=gfx1151 \
    HSA_OVERRIDE_GFX_VERSION=11.5.1

RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    build-essential \
    libeigen3-dev \
    libopencv-dev \
    libsqlite3-dev \
    libboost-all-dev \
    libceres-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgtest-dev \
    libgmock-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libcurl4-openssl-dev \
    libopenimageio-dev \
    openimageio-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

ARG CMAKE_EXTRA_ARGS=""

COPY . /opt/colmap_src
RUN mkdir -p /opt/colmap_src/build \
    && cd /opt/colmap_src/build \
    && cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ENABLED=OFF \
        -DHIP_ENABLED=ON \
        -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH} \
        ${CMAKE_EXTRA_ARGS} \
    && ninja -j "$(nproc)" \
    && ninja install \
    && if [ -z "${CMAKE_EXTRA_ARGS}" ]; then rm -rf /opt/colmap_src; fi

WORKDIR /workspace
ENTRYPOINT ["colmap"]
```

This intentionally mirrors `~/git/rosbag-colmap-pipeline/docker/Dockerfile`'s package list and build pattern (verified working on this same base image) — including keeping the full Qt5/CGAL/GUI dependency list installed and GUI_ENABLED at its default (ON), matching that reference exactly rather than introducing an untested `-DGUI_ENABLED=OFF` configuration nobody has verified. Swaps the pinned-tag `git clone` for local `COPY` source and `-DCUDA_ENABLED=OFF` for the HIP flags. `CMAKE_EXTRA_ARGS` lets Task 4 Step 3 opt into `-DTESTS_ENABLED=ON` without duplicating the whole Dockerfile, and skips deleting the build tree so `ctest` has binaries to run when tests are enabled.

- [ ] **Step 2: Build the image**

```bash
cd ~/git/colmap-rocm
docker build -t colmap-rocm:hip .
```

Expected: image builds successfully. If HIP-specific compile errors occur (e.g. from `patch_match_cuda.cu` being compiled as HIP), capture the exact error — do not paper over it with `-Wno-*` flags without understanding what's failing. If the error isn't a straightforward missing-include/missing-flag issue: stop, log the full error to `docs/rocm-integration.md` under `## BLOCKED`, and end the task here.

- [ ] **Step 3: Run the PR's own GPU unit tests as the smoke test (not a full dense run yet)**

No dense workspace (completed sparse model + undistorted images) exists on this machine yet — `patch_match_stereo` requires one, and building one is Task 7's job, after SIFT (Task 5) and Caspar (Task 6) are also wired in, so the first *full* dense run happens once, end-to-end, in Task 7 rather than twice. For this task, verify PatchMatch-HIP compiles and runs correctly at the unit level instead:

```bash
cd ~/git/colmap-rocm
docker build -t colmap-rocm:hip-tests --build-arg CMAKE_EXTRA_ARGS="-DTESTS_ENABLED=ON" .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  --entrypoint bash colmap-rocm:hip-tests -c \
  "cd /opt/colmap_src/build && ctest -R 'mvs|gpu_mat|patch_match' --output-on-failure"
```

This requires the Dockerfile to accept a `CMAKE_EXTRA_ARGS` build arg (add `ARG CMAKE_EXTRA_ARGS=""` and append `${CMAKE_EXTRA_ARGS}` to the `cmake` invocation in Step 1's Dockerfile, and stop deleting `/opt/colmap_src` when tests are enabled — add a conditional so `ctest` has binaries to run against). Expected: `gpu_mat_test` and any `patch_match`-related tests pass on gfx1151, confirming the HIP kernels execute correctly before Task 7 relies on them inside a full pipeline.

- [ ] **Step 4: Commit the Dockerfile and log results**

```bash
cd ~/git/colmap-rocm
git add Dockerfile docs/rocm-integration.md
git commit -m "build: add HIP-enabled Dockerfile, verify PatchMatch-HIP on gfx1151"
git push origin hip-integration
```

---

## Task 5: Rebase GPU SIFT branch onto hip-integration

**Files:**
- Modify: whatever `git diff upstream/main jeffdaily/rocm-sift-gpu --stat` shows (inspect first — this branch is 115 commits behind, so expect substantial drift in COLMAP's feature-extraction/matching code).

**Interfaces:**
- Consumes: `~/git/colmap-rocm` branch `hip-integration` (Task 4's output — PatchMatch-HIP already integrated).
- Produces: HIP-accelerated SIFT extraction/matching on top of the existing `hip-integration` branch. Task 6 uses this if it succeeds; if this task is abandoned per Step 4's fallback, Task 6 proceeds with CPU/OpenGL SIFT instead and the plan's success criteria are adjusted accordingly (documented, not silently dropped).

- [ ] **Step 1: Assess the drift before attempting a mechanical rebase**

```bash
cd ~/git/colmap-rocm
git fetch jeffdaily rocm-sift-gpu
git log --oneline jeffdaily/rocm-sift-gpu ^upstream/main | wc -l
git diff upstream/main jeffdaily/rocm-sift-gpu --stat
```

Read the diff. If it touches files also modified by PR #4420 (already on `hip-integration`), expect double conflicts (against both upstream drift AND our own PatchMatch changes).

- [ ] **Step 2: Attempt the rebase onto hip-integration (not upstream/main)**

```bash
git checkout -b colmap-sift-rebase jeffdaily/rocm-sift-gpu
git rebase hip-integration
```

This rebases SIFT's 10 commits onto our already-rebased PatchMatch work, so both land together. Resolve conflicts using the same rule as prior tasks: keep upstream/hip-integration's independent changes, re-apply the SIFT-HIP-specific hunks.

**Hard abort trigger (check after each commit's conflicts are resolved, via `git status` / `git diff --stat` during the rebase):** if more than 3 of the 10 commits produce conflicts, OR any single commit's conflict touches more than ~5 files, stop immediately and take the fallback path in Step 3 below — do not push through on a case-by-case "am I confident" judgment call. Given 115 commits of drift, that threshold is what separates "normal rebase friction" from "the branch has diverged too far to trust a mechanical resolution."

- [ ] **Step 3 (success path): Fold onto hip-integration and push**

```bash
git checkout hip-integration
git reset --hard colmap-sift-rebase
git branch -D colmap-sift-rebase
git push --force-with-lease origin hip-integration
```

- [ ] **Step 3 (fallback path, if the rebase proves impractical): Abandon and document**

If after a genuine attempt the conflict volume makes correctness unverifiable (not just "this is tedious" — actual "I can't tell if this is right"), stop:

```bash
git rebase --abort  # or git merge --abort, whichever is mid-flight
git checkout hip-integration
git branch -D colmap-sift-rebase
```

Log the decision and reasoning in `docs/rocm-integration.md`. This means Task 6 runs the pipeline with COLMAP's existing OpenGL/CPU SIFT instead of HIP SIFT — still a valid end-to-end ROCm-accelerated run (PatchMatch + Caspar are still HIP), just not full-HIP-frontend. Note this explicitly rather than letting it look like an oversight.

- [ ] **Step 4: Rebuild and smoke-test (success path only)**

```bash
docker build -t colmap-rocm:hip .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -v <dataset-dir>:/workspace/data \
  colmap-rocm:hip feature_extractor --database_path /workspace/data/db.sqlite \
  --image_path /workspace/data/images
```

Expected: extraction completes without HIP errors, produces nonzero keypoints/descriptors in the database.

- [ ] **Step 5: Log and commit**

```bash
cd ~/git/colmap-rocm
git add docs/rocm-integration.md
git commit -m "docs: log GPU SIFT rebase outcome (success or documented fallback)"
git push origin hip-integration
```

---

## Task 6: Wire Caspar-HIP into the Docker build and mapper — DEFERRED

**Amended 2026-08-04, after Step 1 was actually run.** This task's premise —
that SymForce ships a linkable HIP Caspar library COLMAP consumes via Docker
multi-stage `COPY --from` — is wrong. Verified by direct inspection:
- COLMAP vendors Caspar as **generated CUDA `.cu` source**,
  `src/thirdparty/Symforce-Caspar/generated/{f32,f64}/` (241 kernel files in
  `f32/` alone), compiled directly into the `colmap` binary — not linked as an
  external library.
- `CASPAR_ENABLED` (the flag Step 1 below correctly predicted) is wired **only
  inside `if(CUDA_ENABLED AND CUDA_FOUND)`** in `cmake/FindDependencies.cmake`
  — there is no HIP branch for it on `hip-integration` (PR #4420 only HIP-ported
  PatchMatch/MVS, not Caspar).
- The 241 vendored kernel files use `cooperative_groups`/`cg::reduce`/
  `cg::labeled_partition` — CUDA-specific constructs `cuda_to_hip.h` (PR #4420's
  compat header) has zero coverage of.

Full writeup: `docs/rocm-integration.md`, "Task 6: Caspar-HIP wiring — DEFERRED"
entry. Steps 2–4 below (the Docker multi-stage plan) are **not executed** — left
in place only as a record of the original, invalidated approach. Two real paths
for a future session are documented there: (1) add a HIP branch to
`CASPAR_ENABLED` plus a Caspar-specific `cooperative_groups` compat header, or
(2) regenerate `generated/{f32,f64}` from `caspar_generate.py` with
`use_hip=True`. Neither is in scope here.

**Consequence for Task 7:** the end-to-end run uses COLMAP's default Ceres CPU
bundle adjustment, not Caspar. Task 7 below is amended accordingly — it no
longer passes Caspar backend flags to `mapper`.

<details>
<summary>Original task text (not executed — kept for the record)</summary>

### Task 6 (original): Wire Caspar-HIP into the Docker build and mapper

**Files:**
- Modify: `Dockerfile` (add SymForce build stage).
- Modify: whatever COLMAP config/CMake toggle selects Caspar as the BA backend (locate via `grep -rn "caspar" -i src/ cmake/` inside the container or repo — PR #4420 doesn't add this, it must already exist in current upstream `main` since Caspar was integrated separately; confirm before assuming a flag name).

**Interfaces:**
- Consumes: `~/git/symforce-rocm` branch `hip-integration` (Task 2's verified HIP Caspar build command), `~/git/colmap-rocm` branch `hip-integration` (Task 5's output).
- Produces: a `colmap-rocm:hip` image where `colmap mapper` can select the Caspar-HIP backend for bundle adjustment. Task 7 (full pipeline run) depends on this.

- [ ] **Step 1: Confirm how current upstream COLMAP exposes Caspar as a BA backend**

```bash
cd ~/git/colmap-rocm
grep -rn -i "caspar" src/ cmake/ | grep -v "\.pyc"
```

Read what you find — likely a `Mapper.ba_local_backend`/`Mapper.ba_global_backend` option or a `CASPAR_ENABLED` CMake flag, per the spec's background research. Use the actual names found, not the ones from the earlier research summary if they differ.

- [ ] **Step 2: Extend the Dockerfile with a SymForce build stage**

```dockerfile
# --- SymForce HIP Caspar build stage ---
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 AS symforce-hip
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1 ROCM_ARCH=gfx1151
COPY --from=symforce-rocm-context . /opt/symforce_src
RUN cd /opt/symforce_src && pip install -e . \
    && python3 <build script from Task 2, adapted to write output to /opt/caspar_hip_lib>

# --- main COLMAP stage ---
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
# ... (rest as in Task 4) ...
COPY --from=symforce-hip /opt/caspar_hip_lib /opt/caspar_hip_lib
# wire /opt/caspar_hip_lib into whichever CMake variable Step 1 identified
```

`symforce-rocm-context` isn't a real build context name — Docker multi-stage `COPY --from` only works within a single build context. Since `symforce-rocm` and `colmap-rocm` are separate git repos, use one of: (a) a multi-context build (`docker build --build-context symforce=~/git/symforce-rocm ...`, Docker Buildx feature, then `COPY --from=symforce`), or (b) build the SymForce HIP artifact separately first (`docker build -t symforce-hip:latest ~/git/symforce-rocm` using a small Dockerfile there) and `COPY --from=symforce-hip:latest` in colmap-rocm's Dockerfile. Prefer (b) — it keeps each repo's Docker build self-contained and matches "each fork is its own unit" from the design. Write a minimal `~/git/symforce-rocm/Dockerfile` for this if one doesn't exist yet, using Task 2's verified build command as its `RUN` step.

- [ ] **Step 3: Build and verify Caspar loads**

```bash
docker build -t symforce-hip:latest ~/git/symforce-rocm
cd ~/git/colmap-rocm
docker build -t colmap-rocm:hip .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 colmap-rocm:hip mapper --help
```

Expected: `--help` output lists the Caspar backend option found in Step 1, and the binary doesn't crash on startup (a missing/mislinked `.so` typically shows up as a dynamic-link failure at exec time, not a graceful error — check `ldd` on the `colmap` binary inside the container if this happens).

- [ ] **Step 4: Commit and log**

```bash
cd ~/git/colmap-rocm
git add Dockerfile docs/rocm-integration.md
git commit -m "build: wire Caspar-HIP into colmap-rocm Docker image"
git push origin hip-integration
cd ~/git/symforce-rocm
git add Dockerfile docs/rocm-integration.md 2>/dev/null || git add Dockerfile
git commit -m "build: add standalone Dockerfile for HIP Caspar artifact export"
git push origin hip-integration
```

</details>

---

## Task 7: Full end-to-end incremental SfM run on gfx1151

**Amended 2026-08-04:** with Task 6 deferred, this run uses PatchMatch-HIP
(dense stereo, verified) + stock CPU/OpenGL SIFT (Task 5's documented fallback)
+ COLMAP's default Ceres CPU bundle adjustment (Caspar-HIP unavailable per
Task 6's deferral). One HIP-accelerated stage (dense stereo), not three — this
is the honest scope of "full incremental SfM end-to-end" on this branch today,
not the original three-HIP-stage vision. Caspar-HIP itself remains
independently verified as a standalone library (Task 2, in `symforce-rocm`) —
it just isn't wired into this COLMAP binary yet.

**Files:**
- No source changes. This task only runs the pipeline and records results.
- Modify: `docs/rocm-integration.md` (results log).

**Interfaces:**
- Consumes: `colmap-rocm:hip` image from Task 4 (PatchMatch-HIP verified 14/14 tests, gfx1151) + Task 5 (stock CPU/OpenGL SIFT, HIP SIFT deferred).
- Produces: a recorded end-to-end run — this is the plan's success criterion, nothing downstream depends on its output artifacts.

- [ ] **Step 1: Pick a real dataset and stage it**

No pre-existing sparse/dense COLMAP workspace exists on this machine as of this plan's
writing (checked `~/git/rosbag-colmap-pipeline` and `~/git/splatograph*`). Use
`~/git/rosbag-colmap-pipeline/data/workspaces/table1/rgb/` (451 numbered PNG frames,
already on this machine) instead of downloading anything new. Take a manageable subset
rather than all 451 — full incremental SfM over 451 frames is a poor smoke test (slow,
harder to debug a failure in):

```bash
mkdir -p /tmp/colmap-rocm-e2e/images
cd ~/git/rosbag-colmap-pipeline/data/workspaces/table1/rgb
ls *.png | awk 'NR % 15 == 1' | xargs -I{} cp {} /tmp/colmap-rocm-e2e/images/
ls /tmp/colmap-rocm-e2e/images | wc -l   # expect ~30 images
```

- [ ] **Step 2: Run the full pipeline inside the container, including dense stereo's prerequisites**

`patch_match_stereo` needs a completed sparse model plus undistorted images, not just a
folder of JPEGs/PNGs — run `image_undistorter` between `mapper` and
`patch_match_stereo`:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -v /tmp/colmap-rocm-e2e:/workspace/data \
  colmap-rocm:hip \
  bash -c "
    mkdir -p /workspace/data/sparse /workspace/data/dense &&
    colmap feature_extractor --database_path /workspace/data/db.sqlite --image_path /workspace/data/images &&
    colmap sequential_matcher --database_path /workspace/data/db.sqlite &&
    colmap mapper --database_path /workspace/data/db.sqlite --image_path /workspace/data/images --output_path /workspace/data/sparse &&
    colmap image_undistorter --image_path /workspace/data/images --input_path /workspace/data/sparse/0 --output_path /workspace/data/dense &&
    colmap patch_match_stereo --workspace_path /workspace/data/dense
  "
```

- [ ] **Step 3: Evaluate the result**

Check: nonzero registered images in the sparse model, reasonable reprojection error (not orders of magnitude off from what a CPU/CUDA baseline would produce, if one is available for comparison), nonzero dense depth coverage from `patch_match_stereo`. A crash, zero registered images, or all-NaN output is a real failure — go back to systematic-debugging on whichever stage failed, don't mark this task done anyway.

- [ ] **Step 4: Log final results and commit**

```bash
cd ~/git/colmap-rocm
# append full pipeline results to docs/rocm-integration.md: dataset used,
# registered image count, reprojection error, dense coverage %, wall-clock
# time per stage, and explicit note of which stages ran on HIP vs CPU
# (per Task 5's outcome)
git add docs/rocm-integration.md
git commit -m "docs: log full end-to-end ROCm SfM pipeline run on gfx1151"
git push origin hip-integration
```

---

## Self-Review Notes

- **Spec coverage:** Task 1–2 cover symforce-rocm setup+build (spec step 2). Task 3–4 cover colmap-rocm PatchMatch rebase+build+test (spec step 3). Task 5 covers SIFT rebase (spec step 4), with an explicit documented fallback since this is the plan's highest-risk step (115-commit drift) and the spec doesn't guarantee it succeeds. Task 6 covers Caspar wiring (spec step 5). Task 7 covers the full end-to-end run (spec step 6, success criteria). Out-of-scope items from the spec are not tasked here, matching the spec's Global Constraints.
- **Fork/clone setup** (spec step 1) is already done as of this plan's writing — both repos exist at `~/git/colmap-rocm` and `~/git/symforce-rocm` with `hip-integration` branches pushed to `origin`, tracking `upstream/main`. No task needed for it.
- **Placeholder scan:** no TBD/TODO left; the one deliberately open decision (Task 5's success-vs-fallback branch, Task 6's multi-context vs separate-build-stage choice) is resolved with a concrete recommended path, not left blank.
- **Type/name consistency:** `hip-integration` branch name, `colmap-rocm:hip` / `symforce-hip:latest` image tags, and `HSA_OVERRIDE_GFX_VERSION=11.5.1` / `ROCM_ARCH=gfx1151` env vars are used identically across all tasks.
- **Advisor-reachability fix:** dispatched subagents don't have the advisor tool. All "consult advisor" instructions were replaced with a concrete stop-and-log-to-`docs/rocm-integration.md`-under-`## BLOCKED` pattern; the coordinating session (not the subagent) checks that file between tasks and consults advisor itself if needed.
- **Worktree-per-task was wrong:** git can't have one branch checked out in two worktrees, and `hip-integration` is reset+force-pushed at the end of nearly every task. Fixed to one worktree per repo (two tracks total), tasks within a track run sequentially — see Parallelization Notes.
- **Dataset gap fixed:** confirmed no sparse/dense COLMAP workspace exists on this machine yet. Task 4's smoke test now uses the PR's own `ctest` suite (`-DTESTS_ENABLED=ON`) instead of a full dense run it had no data for; Task 7 now stages a real image subset (`rosbag-colmap-pipeline`'s `table1` frames) and runs the full `feature_extractor` → `matcher` → `mapper` → `image_undistorter` → `patch_match_stereo` chain, since dense stereo needs an undistorted sparse model as input, not a plain image folder.
- **GUI_ENABLED contradiction fixed:** Dockerfile no longer passes `-DGUI_ENABLED=OFF` while installing the full Qt5/CGAL dep list — now matches `rosbag-colmap-pipeline`'s verified-working default (GUI deps installed, flag left at its default ON).

## Parallelization Notes (for subagent-driven-development)

**One worktree per repo, not per task.** Every task in a track ends by resetting and
force-pushing the shared `hip-integration` branch; git only allows one worktree to have
a given branch checked out at a time, and two worktrees resetting the same branch is a
lost-update race. The real parallelism here is 2-way, not 7-way:

- **Track A** = Tasks 1–2, runs entirely in one worktree, e.g. `~/git/wt/symforce-hip`
  (or just `~/git/symforce-rocm` directly if no other work touches it concurrently).
  Tasks 1 and 2 run sequentially within this worktree.
- **Track B** = Tasks 3–4, runs entirely in one worktree, e.g. `~/git/wt/colmap-hip`.
  Tasks 3 and 4 run sequentially within this worktree.
- Track A and Track B have no file overlap (different repos) and dispatch concurrently.
- **Task 5** depends on Track B's Task 4 (rebases onto `hip-integration` after
  PatchMatch is already there) — do not start until Task 4's commit is pushed. Runs in
  the same `colmap-hip` worktree as Track B (sequential continuation, not a new
  worktree).
- **Task 6** depends on both Task 2 (needs the verified Caspar build command, from
  Track A) and Task 5 (needs the SIFT-or-fallback state of `hip-integration`) — do not
  start until both are pushed. Touches both repos (adds a `Dockerfile` to
  `symforce-rocm`, extends the one in `colmap-rocm`) — run it in the `colmap-hip`
  worktree, reading `symforce-rocm`'s state directly rather than opening a third
  worktree for a single-file addition.
- **Task 7** depends on Task 6 alone. Same `colmap-hip` worktree.

So: dispatch Track A and Track B concurrently (2 subagents). When Track B's Task 4
completes, dispatch Task 5 as a follow-on in the same worktree. Once both Task 2 and
Task 5 are done, dispatch Task 6, then Task 7 sequentially.
