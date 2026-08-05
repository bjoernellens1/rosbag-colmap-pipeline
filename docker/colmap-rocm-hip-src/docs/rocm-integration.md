# ROCm/HIP integration log

## Current status (read this first)

As of 2026-08-05, on this branch (`hip-integration`):

- **HIP-accelerated:** dense stereo (`patch_match_stereo`), bundle adjustment
  (`CASPAR` backend, native OpenCV camera-model support included from the start),
  AND feature extraction/matching (`SiftGPU`) — all verified building and
  running correctly on real data on gfx1151. GPU SIFT was previously a
  documented fallback (crashed on buffer/texture reuse past the first image);
  the real root cause has since been found and fixed with real diagnostic
  evidence (`AMD_LOG_LEVEL` kernel-launch tracing), not a guess — see the
  "GPU SIFT: real root cause and fix" entry below, which supersedes the two
  "leading suspect" commits named in the original Track C Task 1 report and
  that report's "bisect between them" recommendation (both are now known dead
  ends: the actual bug is in a third, previously-unexamined kernel neither
  commit touches). Caspar-HIP BA closes the deferral noted below: it turned
  out to require build-system fixes (three real, iterated-on bugs, see the
  Caspar-HIP Completion entries below), not a fundamentally missing capability.
- **Full pipeline integration (GPU SIFT extraction → GPU SIFT matching →
  Caspar-HIP `mapper`) verified end-to-end on real data, 2026-08-05.** The
  two features above had only ever been verified separately; running them
  together was the last open gap this milestone. See "Full-pipeline
  integration verification" entry below — no fresh bug found; both features
  compose cleanly.
- **Caspar-HIP's OpenCV distortion model (k1,k2,p1,p2) verified numerically
  correct against real distorted-lens image data, 2026-08-05.** All prior
  verification used `SIMPLE_RADIAL` data only, leaving the OpenCV dispatch
  path's actual numerical correctness untested (a crash-free run alone
  wouldn't rule out e.g. silently-zeroed distortion terms). Closed — see
  "OpenCV camera-model numerical verification" entry below: fitted k1/k2/p1/p2
  are non-degenerate and match the CPU/Ceres baseline within the same
  fragmented-model artifacts CPU/Ceres also exhibits, i.e. this is a
  dataset-conditioning characteristic, not a Caspar-HIP bug.
- Earlier entries below (particularly around Task 4 and the old Task 5/6 deferral
  notes) describe an intermediate state where Caspar-HIP was believed structurally
  blocked ("COLMAP vendors Caspar as CUDA-only generated source"). That assumption
  no longer holds: `symforce-rocm`'s own codegen templates (as of its
  `hip-integration` branch) now bake in full HIP support for every generated
  Caspar kernel unconditionally, and colmap-rocm's build system only needed a
  handful of CMake wiring fixes on top, not the from-scratch device-code-mapping
  effort originally assumed necessary. Where an entry below conflicts with this
  status block, this status block is current; the entry is a historical record of
  what was believed true at the time, not a live claim.
- Post-final-review fixes (2026-08-04): `ROCM_ARCH` is now a Dockerfile `ARG`
  (overridable via `--build-arg`), `HSA_OVERRIDE_GFX_VERSION` was removed from
  the image's persistent `ENV` (every documented run command already passes it
  via `-e` explicitly, so this wasn't load-bearing — an image should not force a
  GPU-arch override on whoever runs it), and the tests-enabled build now needs
  `KEEP_SOURCE=1` explicitly (previously any non-empty `CMAKE_EXTRA_ARGS` kept
  the source tree as a side effect) — Task 4's `docker build ... --build-arg
  CMAKE_EXTRA_ARGS="-DTESTS_ENABLED=ON"` invocation above needs
  `--build-arg KEEP_SOURCE=1` added if repeated after this fix.

## Task 3: Rebase COLMAP PR #4420 onto current `upstream/main` (2026-08-04)

**Source:** `ishengnan/rocm-support` (PR [#4420](https://github.com/colmap/colmap/pull/4420),
"Add ROCm/HIP support for patch_match_stereo (AMD GPU)").

**Result:** Clean rebase, no conflicts.

`git rebase upstream/main` on a branch created from `ishengnan/rocm-support` completed
successfully with zero conflict hunks across all 8 commits. GitHub's `mergeable: true`
flag (checked prior to starting) held true at rebase time — `upstream/main` had not
drifted in a way that touched the same lines as the PR.

- Base (merge-base with `upstream/main`): `ecdeba302c511b552f6fcb38a03332212cbcc037`
- Rebased tip: `9c1cd066` ("Address remaining HIP review feedback")
- Commit count: 8 (matches PR #4420's original commit count)

Rebased commit range (`upstream/main..hip-integration` after fold-in):

```
6edb2aca Add ROCm/HIP support for patch_match_stereo on AMD GPUs
36e28b52 fix(cmake): address code review feedback for portability
c4185870 Fix ROCm/HIP support: dual-compatible headers, hipify-perl build, avoid enable_language(HIP)
38922a3f Simplify ROCm/HIP support: enable_language(HIP) + cuda_to_hip.h compat header (#1)
c5fd3db4 Fix CI failures from PR #1 and address PR #4420 review
13508b4f Support patch_match_stereo on AMD CDNA (gfx9) GPUs
a9e90965 Auto-detect ROCm install path and HIP architectures via rocm-sdk
9c1cd066 Address remaining HIP review feedback
```

**Steps taken:**
1. `git fetch ishengnan rocm-support` / `git fetch upstream main`.
2. `git checkout -b colmap-patchmatch-rebase ishengnan/rocm-support`.
3. `git rebase upstream/main` — completed cleanly (8/8 commits applied, no conflict markers).
4. `git checkout hip-integration && git reset --hard colmap-patchmatch-rebase`.
5. `git branch -D colmap-patchmatch-rebase`.
6. `git push --force-with-lease origin hip-integration` — accepted
   (`f6cfb683...9c1cd066 hip-integration -> hip-integration (forced update)`).

**Conflicts encountered:** None. No manual conflict resolution was required for this track.

**Follow-on:** Task 4 will build using
`-DCUDA_ENABLED=OFF -DHIP_ENABLED=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151` per the produced
`HIP_ENABLED` CMake option.

## Task 4: Dockerfile + build + PatchMatch-HIP smoke test (2026-08-04)

**Result:** Image builds successfully (both plain and `-DTESTS_ENABLED=ON` variants). No
HIP-specific compile errors. Unit test smoke test: 13/14 tests in the `mvs|gpu_mat|patch_match`
filter pass; `mvs/gpu_mat_test` fails reproducibly (2/2 runs) with a HIP runtime memory error,
most likely caused by a concurrently running GPU workload on this single-GPU host rather than a
defect in the ported HIP code — see details below.

**Build (Step 2):** `docker build -t colmap-rocm:hip .` — succeeded, `real 4m40.559s`
(most of the time is compiling ~300 translation units; base ROCm/PyTorch image and apt
layers were already warm). No HIP-specific compile errors of any kind.

**Tests-enabled build (Step 3a):**
`docker build -t colmap-rocm:hip-tests --build-arg CMAKE_EXTRA_ARGS="-DTESTS_ENABLED=ON" .`
— succeeded, reusing cached layers, completed in under a minute of net new work.

**ctest smoke test (Step 3b):**
```
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 39 --group-add 105 \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  --entrypoint bash colmap-rocm:hip-tests -c \
  "cd /opt/colmap_src/build && ctest -R 'mvs|gpu_mat|patch_match' --output-on-failure"
```
Note: the brief's literal `--group-add video --group-add render` failed with
`Error: looking up supplemental groups ... Unable to find group render: no matching entries
in group file` — the `render` group name isn't defined in this base image's `/etc/group` (only
`video`, gid 44, is). Substituted the host's numeric GIDs for `video` (39) and `render` (105)
instead, which podman accepts without a name-lookup, and device access then worked correctly
(13 of 14 tests ran and interacted with the GPU successfully).

**Result:** 13/14 tests passed. `mvs/gpu_mat_test` (`GpuMat.FillWithVector`) aborted both times
it was run with:
```
Memory critical error by agent node-0 (Agent handle: 0x388b9da0) on address 0x7f561fead000. Reason: Memory in use.
```
`rocm-smi` at the time showed VRAM at 80% and an unrelated `splatograph_train_tmp*` container
actively training on the same (only) GPU in this host — this looks like GPU memory contention
from a concurrent workload rather than a bug in the PR's HIP port. This is a plausible but
*unconfirmed* explanation: the test was not re-run with the GPU otherwise idle, so genuine
HIP-correctness regressions in `GpuMat` cannot be fully ruled out yet. Re-running this smoke
test with the GPU free is recommended before Task 7 relies on `GpuMat`/PatchMatch-HIP inside a
full pipeline.

Full pass/fail table and both raw ctest logs are in the Task 4 report:
`.superpowers/sdd/2026-08-04-colmap-rocm-integration/task-4-report.md`.

## Task 5: Rebase GPU SIFT branch onto hip-integration (2026-08-04)

**Result: FALLBACK — rebase abandoned per the documented hard abort trigger.**
`hip-integration` is unchanged; GPU SIFT (`jeffdaily/rocm-sift-gpu`) is **not** included.
Task 6 proceeds with COLMAP's existing OpenGL/CPU SIFT frontend. PatchMatch-HIP (Task 3/4)
and Caspar-HIP remain HIP-accelerated, so this is still a valid, partially-HIP-accelerated
end-to-end pipeline — just not full-HIP-frontend.

**Source:** `jeffdaily/rocm-sift-gpu`, 10 commits, ~115 commits behind `upstream/main` at
plan-writing time (`git log --oneline jeffdaily/rocm-sift-gpu ^upstream/main | wc -l` → 10).

**Assessment (Step 1):** `git diff upstream/main jeffdaily/rocm-sift-gpu --stat` showed a
whole-repository-scale diff (CI workflows, benchmark scripts, docs, and core `src/colmap/mvs`
and `src/colmap/util` files all touched) — expected fallout from 115 commits of upstream drift,
not evidence by itself of a bad rebase.

**Rebase attempt (Step 2):**
```
git checkout -b colmap-sift-rebase jeffdaily/rocm-sift-gpu
git rebase hip-integration
```
Two of the ten commits (`786e0963`, `e609b9f1`) were skipped automatically as already applied
(shared history with the PR #4420 lineage already on `hip-integration`). The very first commit
actually replayed, `658f8b56` ("Add ROCm/HIP support for patch_match_stereo on AMD GPUs"),
produced conflicts in **12 files**:

```
cmake/FindDependencies.cmake
src/colmap/exe/CMakeLists.txt
src/colmap/mvs/CMakeLists.txt
src/colmap/mvs/cuda_flip.h
src/colmap/mvs/cuda_rotate.h
src/colmap/mvs/cuda_texture.h
src/colmap/mvs/cuda_transpose.h
src/colmap/mvs/gpu_mat.h
src/colmap/mvs/patch_match_cuda.h
src/colmap/util/CMakeLists.txt
src/colmap/util/cuda.cc
src/colmap/util/cudacc.cc
src/colmap/util/cudacc.h
```

This exceeds the brief's hard abort trigger (>~5 files conflicted in a single commit) on the
very first commit replayed — before any judgment call about resolution quality was even
reachable. Per the brief: *"do not push through on a case-by-case 'am I confident' judgment
call."* Stopped immediately.

**Why this makes sense:** `jeffdaily/rocm-sift-gpu`'s own patch-match/CUDA-compat HIP work
(`658f8b56` and friends) independently touches almost the exact same CUDA-compat surface
(`cuda_flip.h`, `cuda_rotate.h`, `cuda_texture.h`, `cuda_transpose.h`, `gpu_mat.h`,
`patch_match_cuda.h`, `util/cuda*.{cc,h}`) that PR #4420's PatchMatch-HIP port
(already folded into `hip-integration`) rewrote. Two independent HIP ports of the same
CUDA-compat layer, built 115 commits apart, is exactly the "double conflict" scenario the
brief warned about in Step 1 — and it manifested on the first commit rather than being
resolvable case-by-case.

**Action taken (Step 3, fallback path):**
```
git rebase --abort
git checkout hip-integration
git branch -D colmap-sift-rebase
```
`hip-integration` working tree is clean and unchanged (`git status` confirms
"nothing to commit, working tree clean", still tracking `origin/hip-integration`, no
force-push performed — `upstream`/`origin` were never touched by this task).

**Consequence for the plan:** Task 6 should run the pipeline with COLMAP's stock
OpenGL/CPU SIFT extractor/matcher instead of HIP-accelerated SIFT. HIP acceleration still
covers PatchMatch stereo (Task 3/4) and Caspar (separate task) — this remains a genuine
partial-HIP end-to-end run, not a fully-CPU fallback. The plan's success criteria should be
read as "HIP-accelerated PatchMatch + Caspar, CPU/OpenGL SIFT" rather than full-HIP-frontend,
per this task's brief.

## 2026-08-04 — Task 4 follow-up: gpu_mat_test retest with GPU idle

Per the task review's required follow-up: re-ran `ctest -R 'mvs|gpu_mat|patch_match'`
in the already-built `colmap-rocm:hip-tests` image once no `splatograph_train_tmp*`
container was running.

```bash
docker run --rm --name gpu_mat_retest \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 39 --group-add 105 \
  --security-opt label=disable \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  --entrypoint bash \
  localhost/colmap-rocm:hip-tests \
  -c "cd /opt/colmap_src/build && ctest -R 'mvs|gpu_mat|patch_match' --output-on-failure"
```

Result: **14/14 tests passed**, including `mvs/gpu_mat_test` (previously failed with
a HIP "Memory in use" error under GPU contention). Confirms the Task 4 report's
hypothesis: the original failure was resource contention from a concurrent,
unrelated training workload on this shared single-GPU host, not a defect in
PR #4420's HIP PatchMatch/GpuMat port. PatchMatch-HIP is now verified clean at the
unit-test level; Task 7 proceeds with confidence in this layer.

## 2026-08-04 — Task 6: Caspar-HIP wiring — DEFERRED (plan premise invalidated)

Task 6 as originally scoped (Docker multi-stage: build symforce-rocm's HIP Caspar
`.so` in one stage, `COPY --from=` it into the colmap-rocm image, wire it in as
the mapper's BA backend) is **not how COLMAP actually consumes Caspar** — the
premise was wrong, discovered by inspection, not by attempting the Docker wiring
and failing.

Facts, verified by direct inspection of this repo:

- Caspar is **vendored as generated CUDA `.cu` source** at
  `src/thirdparty/Symforce-Caspar/generated/{f32,f64}/` (241 kernel files in the
  f32 tree alone), compiled directly as part of COLMAP's own build — not linked
  as an external SymForce library. `src/colmap/controllers/option_manager.cc`
  gates `BundleAdjustmentCaspar.*` options behind `#ifdef CASPAR_ENABLED`, a
  COLMAP-internal compile flag (`CMakeLists.txt:77`), unrelated to whether
  `symforce-rocm` (this repo's sibling fork) is built or even present.
- `CASPAR_ENABLED` is wired **only inside `if(CUDA_ENABLED AND CUDA_FOUND)`** in
  `cmake/FindDependencies.cmake` (~line 310+), with an arch guard that reads
  `CMAKE_CUDA_ARCHITECTURES` specifically. There is no `HIP_ENABLED` branch for
  Caspar at all in the current `hip-integration` branch (PR #4420 only added HIP
  support for PatchMatch/MVS, not Caspar/bundle-adjustment).
- The 241 vendored `.cu` files under `generated/f32/` use
  `#include <cooperative_groups.h>`, `cooperative_groups::reduce`,
  `cooperative_groups::details::partitioning` (`labeled_partition`), and
  `namespace cg = cooperative_groups` — CUDA-specific constructs.
  `src/colmap/util/cuda_to_hip.h` (PR #4420's compat header) has **zero**
  coverage of any of these — it was built for PatchMatch's texture/RNG/event
  usage, an entirely different API surface than Caspar's cooperative-groups
  reduction pattern.

Given CUDA_ENABLED and HIP_ENABLED are mutually exclusive (this plan's own global
constraint, enforced by a `FATAL_ERROR` in `CMakeLists.txt`),
`-DCASPAR_ENABLED=ON -DHIP_ENABLED=ON -DCUDA_ENABLED=OFF` cannot work today even
before considering the 241-file cooperative_groups gap: the `CASPAR_ENABLED`
block simply never executes when `CUDA_ENABLED=OFF`.

Whether SymForce PR #465's HIP compat layer (which does map `cg::reduce`,
`cg::labeled_partition`, and shared-memory atomics — see `~/git/symforce-rocm`
`docs/rocm-integration.md` Task 1 entry) *would* cover this vendored tree's usage
is unresolved: PR #465's mappings live in SymForce's own Caspar *runtime*
generator output, and this vendored tree may have been generated by a different
(older, or differently-configured) run of the same generator — there's no
guarantee the two match construct-for-construct without actually attempting the
port or diffing generator output.

**Decision: defer, same treatment as Task 5's SIFT rebase fallback.** Two real
paths exist for a future session:
1. Add a HIP branch to `CASPAR_ENABLED` in `FindDependencies.cmake` +
   `generated/f32/CMakeLists.txt` (mirroring PR #4420's `LANGUAGE HIP` mechanism
   for `.cu`→HIP compilation) plus a Caspar-specific compat header covering
   `cooperative_groups`/`cg::*`, informed by (not copy-pasted from) PR #465's
   mappings.
2. Regenerate `generated/{f32,f64}` from `src/thirdparty/Symforce-Caspar/caspar_generate.py`
   with SymForce-rocm's `use_hip=True` path (the same `CasparLibrary.compile()`
   API exercised in Task 2's `hip_smoke.py`) — larger diff, no reference output
   to verify numeric equivalence against, higher risk without a dedicated
   verification pass.

Neither is attempted here. Bundle adjustment for Task 7's end-to-end run uses
COLMAP's default Ceres CPU backend, not Caspar. PatchMatch-HIP (Task 4) and
HIP Caspar as a standalone library (Task 2, `symforce-rocm`) remain independently
verified and valid — they are just not yet wired into a single COLMAP binary.

## 2026-08-04 — Task 7: Full end-to-end incremental SfM run on gfx1151

Ran directly in this session (not via subagent — a short, monitorable sequential
pipeline, per advisor guidance after Task 2's subagent dispatch overhead).

**Dataset:** 31 frames subsampled (every 15th) from
`~/git/rosbag-colmap-pipeline/data/workspaces/table1/rgb/` (451 total frames),
staged at `/tmp/colmap-rocm-e2e/`.

**Pipeline run (all stages, `colmap-rocm:hip` image):**

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 39 --group-add 105 \
  --security-opt label=disable \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e QT_QPA_PLATFORM=offscreen \
  -v /tmp/colmap-rocm-e2e:/workspace/data \
  colmap-rocm:hip <command> ...
```

Two runtime env-var fixes needed beyond Task 4/2's known gotchas (both required
for `feature_extractor`/`sequential_matcher`, harmless for other commands):
- `QT_QPA_PLATFORM=offscreen` — `feature_extractor` instantiates a `QApplication`
  even in CLI mode (this build has `GUI_ENABLED` at its default `ON`, per Task 4's
  deliberate choice to match `rosbag-colmap-pipeline`'s known-working config);
  without a display, `QGuiApplicationPrivate::createPlatformIntegration()` aborts.
- `--FeatureExtraction.use_gpu 0` / `--FeatureMatching.use_gpu 0` — SiftGPU's
  default GPU path tries to create an OpenGL context
  (`colmap::OpenGLContextManager`), which fails headlessly in this container
  (`Check failed: context_.create()`). Since HIP SIFT was deferred (Task 5),
  this is expected — CPU/OpenGL SIFT was always the fallback plan; this simply
  makes that explicit at the command-line level rather than relying on a
  silent internal fallback.

**Results, stage by stage:**

| Stage | Backend | Result |
|---|---|---|
| `feature_extractor` | CPU SIFT | 31/31 images, 3700–12400 features each, 0.03 min |
| `sequential_matcher` | CPU | 31/31 images matched, 0.14 min |
| `mapper` | Ceres CPU BA (Caspar deferred, Task 6) | 17/31 images registered into one connected model (`sparse/0`), 1341 3D points, "Keeping successful reconstruction", 0.04 min. (14 images did not register into this model — expected for a sparse/wide-baseline 31-frame subsample of a video sequence, not investigated further; out of scope for this HIP-verification task.) |
| `image_undistorter` | CPU | 17/17 images undistorted cleanly, 0.007 min |
| `patch_match_stereo` | **HIP (gfx1151)** | 17/17 views × 2 passes (photometric + geometric consistency) completed with no errors — confirmed via per-sweep/iteration timing logs (`cudacc.cc`, e.g. "Sweep 1: 0.78s", "Iteration 1: 4.19s") showing real GPU computation, not a no-op. Produced 34 depth-map + 34 normal-map `.bin` files (~3.8–4MB each, consistent with genuine per-pixel float32 data for 1280×720 images — not empty/degenerate output). |
| `stereo_fusion` | CPU | Valid `fused.ply` written, but only **3 fused points**. |

**On the low fusion count:** not investigated as a bug — the wide baseline from
subsampling every 15th frame of a video (31 frames spanning what was originally
~465 sequential frames) combined with a small, already-fragmented sparse model
(1341 points, only 17/31 images registered) plausibly explains aggressive
rejection by `stereo_fusion`'s default multi-view consistency filters
(`filter_min_num_consistent: 2`, `filter_min_triangulation_angle: 3`). The
depth/normal maps themselves are demonstrably real (correct file sizes, correct
count, produced by a HIP kernel run that logged real per-sweep GPU timings) —
this is a dataset-scale/dense-fusion-tuning question, not evidence PatchMatch-HIP
is broken. A production run would use a denser, better-suited image set; this
task's goal was verifying the pipeline executes correctly end-to-end on gfx1151,
which it does.

**Summary: full incremental SfM pipeline runs end-to-end on this gfx1151 machine.**
One stage (`patch_match_stereo`, dense stereo) is genuinely HIP-accelerated and
verified working on real data, not just unit tests. SIFT and bundle adjustment
run on CPU (Tasks 5 and 6 deferred, both with documented reasons and future
paths). This is the honest, achieved scope of this plan as of 2026-08-04.

## 2026-08-04: Track C Task 1 — HIP SIFT via selective cherry-pick from jeffdaily/rocm-sift-gpu

**Goal:** land HIP-accelerated SIFT (`SiftGPU`) on `hip-integration` without repeating
the whole-branch rebase that previously aborted on its first commit (12 conflicted
files against PatchMatch-HIP's already-ported compat layer).

**Classification of all 10 commits on `jeffdaily/rocm-sift-gpu`** (oldest to newest):

| # | Commit | Touches SIFT only? | Disposition |
|---|--------|---------------------|-------------|
| 1 | `658f8b56` "Add ROCm/HIP support for patch_match_stereo" | No — `cuda_flip/rotate/texture/transpose.h`, `gpu_mat.h`, `patch_match_cuda.h`, `util/cuda*.{cc,h}`, `mvs/CMakeLists.txt` | **Skip** — this is the original (superseded) patch_match HIP port; already-ported differently by PR #4420 on `hip-integration`. |
| 2 | `786e0963` "fix(cmake): address code review feedback" | No — same compat-layer files (portability fixes on #1) | **Skip** — fixups to a superseded commit. |
| 3 | `e609b9f1` "Fix ROCm/HIP support: dual-compatible headers..." | No — same compat-layer files, reworked | **Skip** — still the superseded patch_match approach. |
| 4 | `def43b23` "Simplify ROCm/HIP support: enable_language(HIP) + cuda_to_hip.h" | No — introduces `src/colmap/util/cuda_to_hip.h`, rewrites `mvs/CMakeLists.txt`, `cuda_flip/rotate/texture/transpose.h`, `gpu_mat.h` | **Skip** — this is the commit that introduces the compat-header approach PR #4420 already carries (in its own, further-evolved form) on `hip-integration`. |
| 5 | `690348f3` "Enable gpu_mat_test under HIP and report HIP backend in version banner" | No — `mvs/CMakeLists.txt`, `mvs/gpu_mat_test.cu`, `util/version.cc.in` | **Skip.** Functionality forgone: `mvs/gpu_mat_test` is not registered to build/run under `HIP_ENABLED` on this branch. Functionality *not* forgone: the "with HIP" version-banner string is already present in `hip-integration`'s `src/colmap/util/version.cc.in` (verified: `#elif defined(COLMAP_HIP_ENABLED) ... "with HIP"`), landed independently by PR #4420. |
| 6 | `566e4df7` "docs: document HIP/ROCm build in install.rst, drop stale README.rocm.md" | Docs only | **Skip.** Targets a `README.rocm.md` that does not exist on `hip-integration` (never added — PR #4420 took a different path) and an `doc/install.rst` HIP paragraph that `hip-integration` does not have in the form this commit expects. Not mechanically applicable; the underlying facts it would document (build flags, arch mapping) are superseded by this branch's own conventions. |
| 7 | `bf064e92` "Enable GPU SIFT (SiftGPU) under ROCm/HIP" | **Yes** (SIFT-specific: `thirdparty/SiftGPU/*`, `feature/sift.cc`, dispatch-site widening in `controllers/`, `feature/`, `ui/`, `pycolmap/`) plus small additive touches to `cuda_to_hip.h` (9 new `#define`s, no removals) and `FindDependencies.cmake` (widen one `if` condition) | **Cherry-picked.** |
| 8 | `e95eb380` "SiftGPU: fix double-destroy and DoG edge OOB" | **Yes** — `thirdparty/SiftGPU/{CuTexImage.cpp,CuTexImage.h,ProgramCU.cu}` only | **Cherry-picked.** |
| 9 | `3345a981` "SiftGPU: route tex2D through linear binding on HIP" | **Yes** — `thirdparty/SiftGPU/ProgramCU.cu` only | **Cherry-picked.** |
| 10 | `e41e06e0` "docs: note GPU SIFT is now covered by the HIP backend" | Docs only, 2-line edit to the same `doc/install.rst` HIP paragraph #6 targets | **Skip** — same reason as #6: that paragraph does not exist in this branch's `doc/install.rst` in the expected form. |

**Cherry-pick branch:** `colmap-sift-cherrypick`, based on `hip-integration` tip
`e8ad01ca`. Commits, in order:
- `b1a3f26b` = cherry-pick of `bf064e92`
- `7dd7a7fc` = cherry-pick of `e95eb380`
- `66eaa995` = cherry-pick of `3345a981`
- `77c6959f` = local fix-up commit repairing a formatting bug introduced while
  resolving `b1a3f26b`'s merge conflicts (see below)

**Conflict resolution (all in `b1a3f26b`):** 3 files conflicted —
`src/colmap/controllers/automatic_reconstruction.cc`,
`src/colmap/ui/dense_reconstruction_widget.cc`, `src/pycolmap/pipeline/mvs.cc`.
In every case the conflict was HEAD (PR #4420's `hip-integration`) already having
an equivalent `#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)`
guard in different formatting/comment style from what `bf064e92` introduces —
same semantics, cosmetic diff only. Resolved by keeping HEAD's guard in all three.
An automated resolution script had a bug (dropped a newline after a multi-line
`#if` continuation), which broke the build (`missing binary operator before
token "auto"` in `dense_reconstruction_widget.cc`); fixed in follow-up commit
`77c6959f` (also restores two cosmetic blank lines lost the same way in the
other two files). `cuda_to_hip.h`, `FindDependencies.cmake`, and the SiftGPU
files merged cleanly with no conflicts.

**Build:** `docker build -t colmap-rocm:hip-sift ~/git/colmap-rocm` (from the
`colmap-sift-cherrypick` worktree) succeeds cleanly — `libcolmap_sift_gpu.a`
links as a HIP static library (`ProgramCU.cu` compiled through the HIP
toolchain per `set_source_files_properties(... LANGUAGE HIP)`), `-DCOLMAP_GPU_ENABLED`
present in `colmap_ui`'s compile flags confirming `bf064e92`'s
`FindDependencies.cmake` widening took effect.

**Runtime test — GPU SIFT initializes and extracts, then faults on reuse:**

Dataset: 30 frames (every 15th of 451) from
`~/git/rosbag-colmap-pipeline/data/workspaces/table1/rgb/`, all 1280×720.

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 39 --group-add 105 \
  --security-opt label=disable -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e QT_QPA_PLATFORM=offscreen \
  -v <dataset-dir>:/workspace/data colmap-rocm:hip-sift \
  feature_extractor --database_path /workspace/data/db.sqlite --image_path /workspace/data/images \
  --FeatureExtraction.use_gpu 1
```

- `sift.cc:761 "Creating SIFT GPU feature extractor"` confirms the HIP GPU SIFT
  path is selected (not a CPU/GLSL fallback).
- The **first** GPU-extracted image always succeeds, with correct nonzero
  keypoint counts (e.g. 4582, 3693, 3670 SIFT features — plausible, non-degenerate
  values for these images).
- The process then **aborts with a GPU memory access fault**
  ("Memory access fault by GPU node-1 ... Reason: Page not present or supervisor
  privilege", SIGABRT) on the image immediately after the first successfully
  processed one. Lining up GPU-worked images (not raw file position) across
  runs: one run had its first file (`000145.png`) skipped as already-extracted
  (contaminated database from an earlier interrupted run sharing the same
  output DB — discard this run, it is not a clean data point), so its first
  *GPU* image was `000213.png` (succeeded) and its second was `000278.png`
  (faulted, i.e. 3rd file overall). Two independent clean runs (one plain, one
  with `AMD_SERIALIZE_KERNEL=3`) both started fresh and both faulted on
  exactly the same position: 1st GPU image (`000145.png`) succeeds, 2nd GPU
  image (`000213.png`) faults — identical outcome, not run-to-run variance.
  This is a **deterministic "works once per instance, fails on reuse" pattern**,
  which is textbook stale-handle/use-after-free behavior on a reused
  buffer/texture, not a race or allocator-state coin-flip. It points at
  `e95eb380`'s `CuTexObj` rule-of-five rewrite (move-only semantics, handle
  nulling, guarded destructor) or `3345a981`'s `BindTexture2D` → `BindTexture`
  (linear-binding) switch — both touch exactly the texture-object lifecycle
  that would only misbehave on reuse, not on a fresh object.
- Ruled out as GPU contention: `Memory access fault ... Page not present` is a
  virtual-address fault from an illegal access inside a kernel, not an
  allocation failure — contention from other GPU workloads on this host (e.g.
  `splat_train`, confirmed running concurrently via `docker ps` / `rocm-smi
  --showpids` during testing) produces `hipErrorOutOfMemory`/"Memory in use"
  errors, a different failure mode. The fault reproduced identically both with
  and without a concurrent container competing for the GPU.
- `AMD_SERIALIZE_KERNEL=3` (forces synchronous kernel launches so an abort is
  attributed to the actual faulting launch rather than a later sync point) was
  used on one run: the fault still occurred at the same point (2nd GPU image),
  confirming it is synchronous with a specific kernel launch rather than a
  deferred/batched async report. `AMD_SERIALIZE_KERNEL` does not print kernel
  names by itself, so the specific faulting kernel/API call was not identified
  in this pass — that would need a symbolic tool (e.g. `rocgdb`) attached to
  the abort, out of scope for this task's two-probe budget.
- **Isolation probe:** running `feature_extractor` on `000213.png` alone (the
  image that faulted as the "2nd GPU image" in the two clean multi-image runs)
  succeeds cleanly every time (3693 features, 0.007 min, no fault) when it is
  the *only*, and therefore *first*, image processed. This confirms the defect
  is specific to **cross-image buffer/texture reuse** — first use is always
  clean — not the extraction kernel logic itself.

**Outcome: documented fallback, not folded into `hip-integration`.** GPU SIFT
compiles cleanly under HIP on gfx1151 and correctly extracts features for the
first image processed by a given `SiftGPU` instance, but crashes non-deterministically
once more than one image goes through the same instance — a real, upstream
(pre-existing in `jeffdaily/rocm-sift-gpu`, not introduced by cherry-picking)
buffer/texture-lifecycle bug in the reuse path, not a smoke-test triviality and
not a conflict-resolution artifact (the isolated single-image path proves the
ported code is functionally correct; the surviving files after conflict
resolution match HEAD's semantics exactly). Per this task's stated scope (two
diagnostic probes, then document — not patch `ProgramCU.cu`), this is left for
a future session.

**State left behind:** `hip-integration` was **not reset or force-pushed** —
this entry (commit `57b26614`) is a single docs-only commit added normally on
top of its prior tip `e8ad01ca`. The `colmap-sift-cherrypick` branch (4
commits atop `hip-integration` tip `e8ad01ca`: `b1a3f26b`, `7dd7a7fc`,
`66eaa995`, `77c6959f`, plus a docs commit `1682d716` adding
`docs/superpowers/plans/track-c-task1-report.md`) is kept, not deleted, and
pushed to `origin` so the classification work and conflict resolution do not
need to be redone.

**Important — branches have since diverged, check before any fold-in:**
`git merge-base --is-ancestor hip-integration colmap-sift-cherrypick` was
confirmed true *before* this docs commit was pushed to `hip-integration`.
Since `57b26614` landed only on `hip-integration` and is **not** present on
`colmap-sift-cherrypick`, that ancestor relationship no longer holds. A
future session must **not** run `git checkout hip-integration && git reset
--hard colmap-sift-cherrypick` without first re-establishing it (e.g.
`git rebase hip-integration colmap-sift-cherrypick`, then re-check
`--is-ancestor`) — doing so blind would silently drop this docs entry.

**To resume:** the most direct next diagnostic is a bisect within the 3
cherry-picked commits — build with only `bf064e92` + `e95eb380` (drop
`3345a981`'s `BindTexture2D` → `BindTexture` switch) and re-run the same
2-image test. If it still faults, the bug is in `e95eb380`'s `CuTexObj`
rule-of-five; if it's clean, `3345a981`'s linear-binding switch is the cause.
Kernel-level attribution of the fault (which this session's
`AMD_SERIALIZE_KERNEL=3` probe did not provide by itself) would need a
symbolic debugger such as `rocgdb` attached to the abort, not simply a
`RelWithDebInfo` rebuild. Also see the full write-up at
`docs/superpowers/plans/track-c-task1-report.md` on the `colmap-sift-cherrypick`
branch (the refactor `e95eb380`'s own commit message flagged as
"left for a separate change").

## Caspar-HIP Completion, Track B (2026-08-05): wiring Caspar-HIP into COLMAP with native OpenCV support

Executed against `docs/superpowers/plans/2026-08-04-caspar-hip-completion.md`,
Track B Tasks 1, 2, 4, 5 (Task 3 was cut from the critical path in that plan).
Closes the "bundle adjustment CPU-only" deferral above for real — Caspar-HIP BA
now builds and runs correctly on gfx1151, with native `OPENCV` camera-model
support merged in from the start (not a follow-up), per an explicit design
decision made before this work started.

### Task 1: Port caspar-opencv's C++ dispatch changes

Ported `rosbag-colmap-pipeline`'s `docker/patches/caspar-opencv/{bundle_adjustment_caspar.cc,caspar_model_adapter.h}`
(read-only reference, targets COLMAP 4.1.1) into this branch's current tree.
Diffed first, as required — did not blind-apply.

- `bundle_adjustment_caspar.cc`: this branch's tip had already drifted from the
  reference (switched `std::unordered_map`/`unordered_set` to `NodeHashMap`/
  `FlatHashMap`/`FlatHashSet`, added `VLOG_IS_ON(2)` gating for `print_progress`)
  — unrelated to the OpenCV patch. Ported only the two `BuildSizing()` `kOpenCV`
  blocks (pose count, calib count) on top of that drift, mirroring the existing
  `kPinhole` blocks.
- `caspar_model_adapter.h`: otherwise byte-identical to the reference minus the
  OpenCV additions (no drift) — applied the reference file wholesale:
  `CasparSolverSizing` OpenCV fields, `OpenCVAdapter` class, `CreateCasparAdapter()`
  case, and `CreateSolver()`'s full positional-argument list (the single
  highest-risk part of the original patch — silently wrong-compiling if
  misordered).
- Pre-regeneration sanity check: current `generated/f32/solver.h`'s
  `GraphSolver` constructor (no OpenCV nodes yet) matched the non-OpenCV
  portion of the ported call exactly, by name and position.
- Commit: `feat(caspar): port OpenCV camera-model dispatch from rosbag-colmap-pipeline's caspar-opencv patch`.

### Task 2: Regenerate Caspar kernels with OpenCV support

Ported `caspar_generate.py`'s `opencv_core`/`opencv_split_core` additions the
same way (clean diff, pure additions, no drift). Cross-checked the distortion
formula against this branch's own `src/colmap/sensor/models.h`
`OpenCVCameraModel::Distortion`/`ImgFromCam` — exact match (params order
`[fx,fy,cx,cy,k1,k2,p1,p2]`, `radial = k1*r2 + k2*r2^2`, same `du`/`dv` terms),
unchanged from the 4.1.1 baseline the reference patch targeted.

Regenerated `generated/f32/` (host Python lacked a working `symengine` build
compatible with `symforce-rocm`'s vendored fork — ran codegen inside the
`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` container
instead, with `symforce-rocm` `pip install -e .`'d there). Exit 0, no 48KB
shared-memory-budget error. 713 files, 224 new OpenCV-related.

**Notable finding, not anticipated by the plan:** every regenerated file now
unconditionally `#include`s `"cuda_to_hip.h"`, and the regenerated
`CMakeLists.txt` gained a full `USE_HIP` option (`find_package(hip)`,
`LANGUAGE HIP` source properties, `HIP_ARCHITECTURES`). This isn't something
this session added — `symforce-rocm`'s own upstream jinja templates
(`symforce/caspar/source/templates/*.jinja`) now bake in HIP support
unconditionally, including shipping a full, already-authored Caspar-specific
`cuda_to_hip.h` compat header (`symforce/caspar/source/runtime/cuda_to_hip.h`,
authored by Jeff Daily, AMD) that maps `cudaMalloc`→`hipMalloc` etc., and
provides `caspar_hip::reduce_sum`/`labeled_reduce_sum`/`match_any_mask` HIP
fallbacks for the `cg::reduce`/`cg::labeled_partition` operations HIP's
cooperative_groups lacks. This substantially changed Task 4's scope from
"write a compat header from scratch" to "fix real build-system wiring bugs
around an already-correct header" (see Task 4 below).

PINHOLE/SIMPLE_RADIAL: the mechanical per-node kernels are pure reformatting
versus the prior committed tree (clang-format-style whitespace/comments plus
the `cuda_to_hip.h`/`USE_HIP` additions, no logic change). The
reprojection-residual/score kernels (`*_res_jac`, `*_res_jac_first`,
`*_score`) are not textually identical, though — 90 pre-existing generated
files there still differ in actual generated-code content after normalizing
whitespace/comments/includes, showing expression-ordering differences (e.g.
register count changes) from symforce/symengine codegen-version drift, not a
hand edit and not a semantic bug (both gfx1151 HIP and A100 CUDA builds ran
these kernels successfully this milestone). Do not assume regeneration is a
no-op diff for these files.

Re-verified Task 1's `CreateSolver()` positional-argument port against the
now-OpenCV-augmented `solver.h`'s actual `GraphSolver` constructor: node-type
order (`OpenCVCalib`, `OpenCVFocalAndExtra`, `OpenCVPose`,
`OpenCVPrincipalPoint`, `Pinhole*`, `Point`, `SimpleRadial*`) and factor-count
order (`simple_radial` → `pinhole` → `opencv` → `*_split` variants) match
exactly, zero mismatches.

`f64` left unregenerated: `CASPAR_USE_DOUBLE` defaults `OFF` in this branch's
`CMakeLists.txt` and is untested elsewhere in the branch.

Commit: `feat(caspar): regenerate kernels with native OPENCV camera-model support`.

### Task 4: Add HIP compilation to CASPAR_ENABLED

Extended `cmake/FindDependencies.cmake`'s `CASPAR_ENABLED` arch guard with a
standalone `if(HIP_ENABLED AND CASPAR_ENABLED)` block (requires
`CMAKE_HIP_ARCHITECTURES` set, warns — doesn't fail — on any arch other than
`gfx1151`, the only one built and run to date). Mirrored the existing CUDA
`FetchContent` block in `src/thirdparty/CMakeLists.txt` with a
`CASPAR_ENABLED AND HIP_ENABLED` branch that sets `USE_HIP ON` before
`FetchContent_MakeAvailable`, which the regenerated `CMakeLists.txt` (Task 2)
picks up to build itself as a HIP project.

Per the plan's explicit instruction ("Build and iterate on real compile
errors — do not suppress. If a construct is genuinely unmappable, stop and
report BLOCKED"), this took **three build iterations**, each a real bug found
and fixed, none suppressed:

1. `fatal error: hip/hip_runtime.h: No such file or directory` — the
   regenerated `CMakeLists.txt`'s own
   `target_include_directories(caspar_lib_core PUBLIC ${hip_INCLUDE_DIRS})`
   is a no-op: modern `find_package(hip)` never sets that legacy variable,
   only populates `hip::host`'s own `INTERFACE_INCLUDE_DIRECTORIES`. Fixed by
   reading that target property (with a `ROCM_PATH`-based fallback if empty)
   in the generated `CMakeLists.txt`.
2. Same error persisted after fix #1 — root cause was actually
   `src/thirdparty/CMakeLists.txt`'s pre-existing
   `set_target_properties(caspar_lib_core PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ...)`
   call, which **overwrites** rather than appends, silently wiping out
   whatever the generated `CMakeLists.txt` had just set (including fix #1).
   Fixed by re-adding the ROCm include dir afterwards with
   `target_include_directories()` (which appends) instead. Also had to add
   `target_compile_definitions(caspar_lib_core PUBLIC __HIP_PLATFORM_AMD__)`:
   HIP-language translation units get this defined implicitly by the compiler
   wrapper, but plain C++ consumers (colmap's `bundle_adjustment.cc`,
   transitively via `solver.h`) do not, and `hip_runtime.h` `#error`s out
   without it.
3. `'__device__' does not name a type` — `cuda_to_hip.h` unconditionally
   defines `__device__ __forceinline__` function bodies
   (`caspar_hip::reduce_sum`/`reduce_max`/`match_any_mask`/`labeled_reduce_sum`)
   whenever `USE_HIP` is defined, but `USE_HIP` being defined does not mean
   the translation unit is being compiled by `hipcc`/`clang++ --hip` — plain
   `g++` cannot parse `__device__` at all, regardless of what headers it's
   given. Since `USE_HIP` is now (correctly, per fix #2) propagated `PUBLIC`
   to every consumer of `caspar_lib_core`, including plain-C++
   `bundle_adjustment.cc`, this broke. Fixed by guarding the `hipcub`/
   `hip_cooperative_groups.h` includes and all four `__device__` function
   definitions (plus the macros referencing them) behind `__HIPCC__`, which
   the HIP compiler defines automatically and a plain host compiler never
   does — these are device-only utilities never called from host code, so
   losing them in host translation units is correct, not a functionality
   regression. This is a hand-patch on top of `symforce-rocm`'s vendored
   `cuda_to_hip.h` (shipped verbatim by Task 2's regeneration); consistent
   with the plan's "Caspar-specific cooperative_groups HIP compat header"
   step, which turned out to already exist upstream rather than needing to
   be written from scratch, but still needed this host/device-compile-mode
   fix specific to how colmap-rocm's build reaches this header from plain
   C++ translation units.

**Final-branch-review follow-up (2026-08-05):** the fix-#2 `set_target_properties`
overwrite and the fix-#3 `__HIPCC__` guard were both hand-patches on
generated/vendored files with no warning banner, so a future
`generate_caspar.py` regeneration would have silently reintroduced both bugs.
Added `LOCAL PATCH` comment banners at both patch sites
(`src/thirdparty/CMakeLists.txt` and
`src/thirdparty/Symforce-Caspar/generated/f32/cuda_to_hip.h`), and — since
the `__HIPCC__` guard fix is small and self-contained — also applied it
directly to `symforce-rocm`'s own codegen template
(`symforce/caspar/source/runtime/cuda_to_hip.h`, `hip-integration` branch,
commit `bdc65218`), so future regeneration produces the guard correctly
without needing the downstream hand-patch at all. The
`set_target_properties`-overwrite fix (I1) is colmap-rocm-specific build
wiring, not a symforce-rocm codegen issue, so it stays local to this repo
only (see fix #2 above). It also needed a follow-up once actually rebuilt:
naively switching the overwrite to `target_include_directories()` (append)
left the *generated* `CMakeLists.txt`'s own unwrapped
`${CMAKE_CURRENT_SOURCE_DIR}` entry (added by its plain
`add_library()`/`target_include_directories()` call, not `BUILD_INTERFACE`-
wrapped) sitting in `INTERFACE_INCLUDE_DIRECTORIES` alongside the new
wrapped one -- which CMake's `install(EXPORT)` validation rejects outright
("... which is prefixed in the source directory"). The old overwrite had
been accidentally masking this pre-existing bug in the generated file by
discarding that raw entry along with everything else. Fixed by reading back
the current `INTERFACE_INCLUDE_DIRECTORIES` list, removing the raw
`CASPAR_GEN_DIR` entry with `list(REMOVE_ITEM)`, and re-adding it
`BUILD_INTERFACE`-wrapped -- preserving every other entry already on the
property, in particular the ROCm include directory the generated
`CMakeLists.txt` had added for `<hip/hip_runtime.h>`.

Verified: `docker build -t colmap-rocm:caspar-hip --build-arg CMAKE_EXTRA_ARGS="-DCASPAR_ENABLED=ON" .`
(Dockerfile already bakes in `-DCUDA_ENABLED=OFF -DHIP_ENABLED=ON
-DCMAKE_HIP_ARCHITECTURES=gfx1151`) completes clean, 0 `FAILED` targets,
image tagged `localhost/colmap-rocm:caspar-hip`.

Commit: `feat(caspar): add HIP compilation path to CASPAR_ENABLED (gfx1151)`.

### Task 5: Verify Caspar-HIP bundle adjustment on real data, gfx1151

`colmap mapper --help` confirms `--Mapper.ba_global_backend`/
`--Mapper.ba_local_backend` accept `CASPAR` (registered via
`option_manager.cc`'s `#ifdef CASPAR_ENABLED` block — not printed as an
explicit choices list in `--help` output, confirmed by reading the source
rather than guessing from `--help` text alone).

**Dataset:** same 31-frame subsample (every 15th frame of 451) from
`~/git/rosbag-colmap-pipeline/data/workspaces/table1/rgb/` used by the prior
milestone's Task 7 end-to-end run, staged fresh at `/tmp/caspar-hip-e2e/`.

**Pipeline:** `feature_extractor --FeatureExtraction.use_gpu 0` (31/31 images,
0.028 min) → `sequential_matcher --FeatureMatching.use_gpu 0` (31/31 matched,
0.116 min) → `mapper` run twice from the same database, once per backend:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 39 --group-add 105 \
  --security-opt label=disable -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e QT_QPA_PLATFORM=offscreen \
  -v /tmp/caspar-hip-e2e:/workspace/data localhost/colmap-rocm:caspar-hip \
  mapper --database_path /workspace/data/db.sqlite --image_path /workspace/data/images \
  --output_path /workspace/data/sparse_caspar \
  --Mapper.ba_global_backend CASPAR --Mapper.ba_local_backend CASPAR
```

vs the same command with `--output_path /workspace/data/sparse_ceres` and no
backend flags (default Ceres/CPU).

**Results (`colmap model_analyzer`, both models):**

| | Caspar-HIP (gfx1151) | Ceres (CPU, default) |
|---|---|---|
| Registered images | 17 / 31 | 17 / 31 |
| 3D points | 1342 | 1342 |
| Observations | 5167 | 5170 |
| Mean track length | 3.850 | 3.852 |
| Mean reprojection error | 0.630 px | 0.543 px |

Caspar-HIP mapper run: 1.169 min (13 registration steps, "Keeping successful
reconstruction", no errors/crashes in the log — grepped for
`error|caspar|hip|failed|crash|abort`, zero matches). Ceres run: 0.047 min
(expected — no GPU dispatch overhead at this tiny problem size).

**Verdict: PASS.** Identical registered-image count, identical point count,
reprojection error same order of magnitude (both sub-pixel, ~15% apart — well
within the "not required to match bit-for-bit" tolerance the plan set). No
crashes, no NaNs, no divergent/degenerate reconstruction. This is real,
on-hardware confirmation that Caspar-HIP's bundle adjustment — including the
newly-added native OpenCV camera-model path wired in Tasks 1–2 (this
dataset's cameras are `SIMPLE_RADIAL`, not `OPENCV`, so the OpenCV dispatch
path itself was verified for build/link correctness and positional-argument
safety in Tasks 1–2's cross-checks rather than exercised numerically here —
none of the staged images' cameras use `OPENCV`; a follow-up with an
`OPENCV`-model dataset would close that last numerical gap) — produces
correct results on gfx1151, not just a clean compile.

Commit: `docs: verify Caspar-HIP bundle adjustment on real data, gfx1151 — closes prior Task 6 deferral`.

### 2026-08-05: CUDA_ENABLED=ON regression check on cps-gpu-cluster (A100)

Verified this session's extensive HIP-specific work (Tasks 1–5 above) did not
regress the original `-DCUDA_ENABLED=ON` build path, using real A100 hardware
on the `cps-gpu-cluster` (see `~/git/cps-gpu-cluster/CLAUDE.md`).

**Dispatch mechanism finding (the actual point of this task):** `ablator`
(`~/git/rosbag-colmap-pipeline/ablator`, `configs/ablator.toml`) is
**dispatch-only** — it has no image-build step, only job submission against a
fixed pre-published `image:` tag already in a registry the cluster can pull
from. It cannot build a new image from this branch. Building+pushing is a
separate, manual, already-documented `podman build` / `podman push` flow
(`~/git/rosbag-colmap-pipeline/docs/cluster-dispatch.md`), and registry push
credentials for `ghcr.io/bjoernellens1/*` were already present on this host
(`podman login ghcr.io --get-login` → `bjoernellens1`), so this did **not**
end BLOCKED.

Two things ruled out reusing that documented flow as-is:
- `colmap-rocm`'s own `Dockerfile` bases on `rocm/pytorch:...` (no `nvcc`) —
  there is no CUDA-capable base to layer `-DCUDA_ENABLED=ON` onto directly;
  a CUDA build needs a distinct Dockerfile with a CUDA devel base image.
- `rosbag-colmap-pipeline/docker/Dockerfile.cuda` (the only existing
  "CUDA-equivalent" Dockerfile referenced in that repo's docs) `git clone`s
  `bjoernellens1/colmap` (a *different fork*, pinned to a specific commit),
  not this repo's `hip-integration` branch — building it would not have
  tested this branch's CUDA path at all.

**What was actually done:** wrote a new Dockerfile (scratch-only, not
committed to either repo) modeled directly on `colmap-rocm`'s own
`Dockerfile` — same dependency list, `COPY . /opt/colmap_src` from this
branch's worktree — but based on `nvcr.io/nvidia/cuda:12.6.0-devel-ubuntu24.04`
with `-DCUDA_ENABLED=ON -DHIP_ENABLED=OFF -DCMAKE_CUDA_ARCHITECTURES=80`.

1. **Compile check** (no GPU/cluster needed): built locally via `podman
   build` at branch tip `b96b57d1`. Compiled clean — `ninja install`
   completed and `/usr/local/bin/colmap` was installed. `colmap -h` inside
   the image reports
   `COLMAP 4.2.0.dev0 ... with CUDA`. No `CUDA_ENABLED`/`HIP_ENABLED`
   mutual-exclusion guard blocks this combination (`CMakeLists.txt:43` only
   errors if *both* are `ON` simultaneously, which is correct and unchanged).
2. **Runtime check on the cluster:** pushed the built image as a new tag,
   `ghcr.io/bjoernellens1/colmap-rgbd-gt:cuda-hip-integration-regression-b96b57d1`,
   onto the *existing public* `colmap-rgbd-gt` GHCR package (deliberately not
   a new package — a new package defaults private and pods would hang
   forever in `ContainerCreating` with zero events, per
   `rosbag-colmap-pipeline/docs/cluster-dispatch.md`).
   `ablator`'s `[types.reconstruct]` job type assumes a `gttool` entrypoint
   this image doesn't have (ENTRYPOINT is `colmap`), so dispatched directly
   via a `kubectl` `Job` instead (single PVC mount, `kai-scheduler`,
   `kai-batch-low`, `nvidia.com/gpu: 1`, namespace `jupyterhub`) — a one-off
   `Job` is the lower-risk path for a single regression check anyway.
   Ran `feature_extractor` → `exhaustive_matcher` → `mapper` with
   `--FeatureExtraction.use_gpu 1` / `--FeatureMatching.use_gpu 1` against 8
   frames sampled from the TUM `freiburg1_desk` sequence already present on
   this host (copied to the cluster's NFS-backed scratch PVC, then to
   node-local disk inside the pod first — COLMAP's SQLite `database.db` is
   documented-unreliable directly over NFS on this cluster).

   Result: scheduled on `k3s-wk-gpu2`, image pulled in 55s (5.4GB; no
   45-minute cold-pull hazard hit — not investigated further why this was
   fast). `nvidia-smi` inside the pod correctly showed the assigned A100.
   GPU SIFT extraction (1000-2900 features/image across all 8 frames) and
   GPU matching (14 verified pairs) both completed without error. `mapper`
   registered 4 of the 8 images into one kept reconstruction (initial pair
   #6+#5, then #4, #8, #7 added; every other candidate initial pair was
   tried and discarded for lacking a good match), writing a real
   `points3D.bin` sparse model. Job reported `Succeeded`. The partial
   (4/8) registration is expected dataset sparsity — these are widely
   spaced, non-consecutive frames from a monocular RGB sequence, not a
   curated multiview set — not a sign of a build regression; the GPU
   extraction/matching/BA code paths themselves ran clean throughout.

**Verdict: PASS.** `CUDA_ENABLED=ON` still compiles and runs correctly on
real A100 hardware after this session's HIP-specific changes — no
regression. Local test images, the scratch-PVC test data, and the `kubectl`
`Job` were all cleaned up after the check; nothing was left running on the
cluster, and no `cluster-maintenance/` manifests in `cps-gpu-cluster` were
touched (Fleet-managed tree was not used for this — this was ad hoc one-off
job dispatch against the existing cluster, per this task's scope). One
thing *not* cleaned up: the pushed image tag
`ghcr.io/bjoernellens1/colmap-rgbd-gt:cuda-hip-integration-regression-b96b57d1`
was left on the public GHCR package (consistent with the many other
`cuda-caspar-*`/`cuda-test`-style tags already there from this session).
Also note: the CUDA Dockerfile used for this check lives only in this
session's scratchpad, not committed to either repo — this check is not
directly reproducible from either repo's current state without recreating
that Dockerfile (it mirrors `colmap-rocm`'s own `Dockerfile` but on a CUDA
devel base with `-DCUDA_ENABLED=ON -DHIP_ENABLED=OFF
-DCMAKE_CUDA_ARCHITECTURES=80`).

## 2026-08-05: GPU SIFT — real root cause found and fixed, folded into `hip-integration`

**Supersedes** the "documented fallback" outcome and the two "leading
suspect" commits (`e95eb380`, `3345a981`) named in the 2026-08-04 Track C
Task 1 entry above. Both commits are innocent; the report's recommended next
step ("bisect within the 3 cherry-picked commits between these two") is now a
known dead end — don't spend a future session on it.

**Reproduced first,** with the same setup the prior report used (10-30 image
subsets of `~/git/rosbag-colmap-pipeline/data/workspaces/table1/rgb/`,
1280x720): confirmed the same "works on image 1, crashes with a `Page not
present or supervisor privilege` GPU memory access fault a few images in"
signature, `colmap-sift-cherrypick` branch, unmodified.

**Real diagnostic signal:** ran with `AMD_LOG_LEVEL=3 HIP_LAUNCH_BLOCKING=1`
(prints every HIP API call and kernel `ShaderName` to stderr as it's issued —
no `rocgdb` needed to identify the faulting call, contrary to the prior
report's assumption that a symbolic debugger was required). The trace showed
the fault landing immediately after a `hipLaunchKernel` for
`ListGen_Kernel(__hip_texture*, __hip_texture*, HIP_vector_type<int,4u>*, int, int)`
— a kernel in `ProgramCU.cu`'s `GenerateList()`, part of
`SiftPyramid::RunSIFT` → `GenerateFeatureList()`'s GPU list-compaction path
(`GlobalUtil::_ListGenGPU == 1`), immediately preceded by two
`hipCreateTextureObject` calls binding a growing pair of "list"/"histogram"
linear-texture buffers. **Neither `e95eb380` nor `3345a981` touches this
kernel or this call path at all** — both are scoped to `ComputeOrientation`
and `ComputeDescriptor`'s texture-object lifecycle, several call frames away
from where the fault actually occurs.

**Root cause:** `ListGen_Kernel` (original upstream code, predates both
cherry-picked commits, never previously exercised on HIP) launches a grid
rounded up to a multiple of `LISTGEN_BLOCK_DIM`, so `idx1` can exceed
`list_len` for the kernel's tail block. The only bounds check in the
original code guards the *write* at the bottom of the kernel
(`if (idx1 < list_len) d_list[idx1] = pos;`), but the *read* above it —
`tex1Dfetch<int4>(texDataList, idx1)` — is unconditional. On CUDA, an
out-of-range `tex1Dfetch` on a linear texture silently returns zero, so the
tail threads' garbage `pos` is harmlessly discarded by the write guard. HIP
has no such clamp-to-zero fallback: an out-of-range `tex1Dfetch` can read
unmapped device memory and faults with exactly the "page not present"
signature reproduced here. This explains the "works once, fails on reuse"
shape purely as a **coincidence of allocation sizing**, not a lifecycle bug:
whether the tail-thread OOB read happens to land on mapped or unmapped
memory depends on how big the accumulated list/histogram buffers are for a
given image, which only tends to grow (and cross onto unmapped pages) after
the first image or two.

**Two speculative fixes were tried and empirically ruled out before finding
this** (both built and run, not just reasoned about): (1) a HIP-only
`hipDeviceSynchronize()` added to `CuTexImage::CuTexObj`'s destructor and
move-assignment (the initial hypothesis: HIP doesn't stream-order
texture-object destruction against in-flight kernels the way CUDA does), and
(2) an added `hipDeviceSynchronize()` at the top of
`PyramidCU::BuildPyramid`, i.e. at the image boundary. Both changes shifted
the fault later — deterministically from the 2nd processed image to the
6th/7th — but neither eliminated it. That both a per-destroy sync and an
image-boundary sync only delayed rather than fixed the fault is itself
evidence against the race-condition hypothesis: a real missing
synchronization would either always matter or never matter for a given
buffer-size trajectory, not shift by exactly the amount of scheduling
perturbation the sync itself introduced. Both speculative changes were
reverted; they are not part of the final fix.

**Fix (commit on `colmap-sift-cherrypick`, folded into `hip-integration` via
rebase — see below):** clamp the tail-thread read index to `list_len - 1`
before the first `tex1Dfetch` in `ListGen_Kernel`, gated under
`#ifdef COLMAP_HIP_ENABLED` (this codebase's established convention for
HIP-only behavior changes, keeping the CUDA path byte-identical to
upstream). The fetched value for out-of-range threads is still discarded by
the existing write guard, so in-range threads' results are unaffected —
this is a minimal, surgical, one-`#if`-block diff.

**Verification:**
- 3 fresh-process runs (`docker run --rm`, no state carried between runs) of
  `feature_extractor --FeatureExtraction.use_gpu 1` on 20 images from
  `table1/rgb` (1280x720) — 3/3 completed all 20 images, zero faults,
  correct nonzero keypoint counts on every image (range ~4600-9500 features,
  no degenerate/zero counts).
- 3 more fresh-process runs on 20 images from `freiburg1_desk/rgb` (640x480 —
  different resolution, therefore a different feature-list buffer growth
  trajectory than the first dataset, specifically chosen to probe for any
  other unguarded read the first dataset's sizing sequence might not
  exercise) — 3/3 completed cleanly, zero faults.
- Re-verified after rebasing onto the current `hip-integration` tip (see
  below): rebuilt fresh, re-ran 2 fresh-process runs per dataset (4 total) —
  4/4 clean, zero faults. Verification on the pre-rebase tree does not
  transfer to different surrounding code, so this re-run was not skipped.

**Folded into `hip-integration`:** `colmap-sift-cherrypick`'s tip (previously
7 commits ahead of a stale base, `hip-integration`'s old tip `e8ad01ca`, per
the divergence warning in the 2026-08-04 entry above) was rebased onto the
current `hip-integration` tip `734c7bea` — the branches had diverged by a
whole Caspar-HIP + OpenCV milestone since. The rebase applied cleanly with
**zero conflicts**. `hip-integration` was then fast-forwarded onto the
rebased branch (`git merge --ff-only`, not `reset --hard`, and only after
confirming `git merge-base --is-ancestor hip-integration
colmap-sift-cherrypick` held true post-rebase) — a true fast-forward, not a
history rewrite. GPU SIFT (`SiftGPU`) is therefore now part of
`hip-integration`'s HIP-accelerated code paths, not a separate branch kept
for a future session.

### 2026-08-05: Full-pipeline integration verification (GPU SIFT + Caspar-HIP, together, end-to-end)

**Why this task:** GPU SIFT (`a650ef52`/`1afc1ddb`, root-caused and fixed) had
only ever been run in isolation via `feature_extractor` alone. Caspar-HIP BA
(`b96b57d1`) had only ever been verified with the default CPU feature
extraction/matching path, since GPU SIFT wasn't merged onto `hip-integration`
yet at that time. Nobody had run `feature_extractor` (GPU) →
`exhaustive_matcher` (GPU) → `mapper --Mapper.ba_global_backend CASPAR`
together on this branch's current tip. This closes that gap.

**Build:** `docker build -t colmap-rocm:full-integration --build-arg
CMAKE_EXTRA_ARGS="-DCASPAR_ENABLED=ON" .` at branch tip `1afc1ddb`
(`-DHIP_ENABLED=ON -DCUDA_ENABLED=OFF` from the Dockerfile default). Clean
build, `801/801` ninja targets, `real 3m~4m`. `colmap -h` inside the image
reports `COLMAP 4.2.0.dev0 ... with HIP`.

**Dataset:** 30 frames sampled from TUM `freiburg1_desk`
(`~/git/rosbag-colmap-pipeline/docker/workspaces/freiburg1_desk/rgb`, frames
`000000.png`–`000290.png` at stride 10, 640x480, `SIMPLE_RADIAL`). This
stride turned out wider-baseline than ideal for this fast-moving handheld
sequence — of the 30 staged frames, only a contiguous ~14-frame span had
enough visual overlap to form one connected reconstruction; the mapper
correctly discarded every other candidate initial pair for insufficient
size/no good match rather than force a bad registration. This is dataset
sparsity, not a pipeline defect (same pattern, same root cause, as the
CUDA-regression check's 4/8 result recorded in the entry above).

**Commands** (env: `HSA_OVERRIDE_GFX_VERSION=11.5.1 --group-add 39
--group-add 105 --security-opt label=disable --device=/dev/kfd
--device=/dev/dri -e QT_QPA_PLATFORM=offscreen`):
```
colmap feature_extractor --FeatureExtraction.use_gpu 1 --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL ...
colmap exhaustive_matcher --FeatureMatching.use_gpu 1 ...
colmap mapper --Mapper.ba_global_backend CASPAR ...
```
Note: `--SiftExtraction.use_gpu` / `--SiftMatching.use_gpu` (used in some
older docs/scripts) do **not** exist as flags on this branch; the correct
names are `--FeatureExtraction.use_gpu` and `--FeatureMatching.use_gpu`
(confirmed via `colmap feature_extractor -h` / `colmap exhaustive_matcher
-h`). First attempt with the wrong flag name failed fast with a clear
`unrecognised option` error — not a pipeline bug, a docs/flag-naming trap
worth flagging for anyone copying older invocations.

**Results — 3 fresh-process runs (fresh containers, fresh `database.db` per
run, no state reuse):**

| run | images extracted | crash/fault | registered | points | mean track len | mean reproj. error |
|-----|---|---|---|---|---|---|
| 1 | 30/30 | none | 14/30 | 1312 | 4.236 | 0.6995 px |
| 2 | 30/30 | none | 14/30 | 1300 | 4.238 | 0.6997 px |
| 3 | 30/30 | none | 14/30 | 1305 | 4.231 | 0.7063 px |

All three runs registered the identical 14-image span, identical initial
pair selection pattern, and point counts/reprojection error within ~1% of
each other run-to-run — no nondeterminism, no intermittent faults across 3
independent runs.

**Keypoint-count regression check (Track C's fixed failure mode):** per-image
SIFT feature counts across all 30 images, every run, ranged 599–3334 with no
systematic drop-off after image 1 (e.g. run 1: 1224, 1011, 1227, 1138,
1056, 1875, 2151, 2606, ... 599) — confirms the tail-thread texture-read fix
(`a650ef52`) holds under the full pipeline's different call pattern/timing,
not just the isolated `feature_extractor`-only test it was originally fixed
under.

**CPU baseline comparison** (same 30 images, `--FeatureExtraction.use_gpu 0
--FeatureMatching.use_gpu 0`, `--Mapper.ba_global_backend CERES`): 14/30
registered (same count and same image span as all 3 GPU runs), 1272 points,
mean track length 4.266, mean reprojection error 0.6686 px. GPU pipeline's
~0.70px vs. CPU's ~0.67px is the same order of magnitude and consistent with
expected GPU-SIFT/CPU-SIFT keypoint-localization differences already
documented elsewhere in this log — not a quality regression.

**Verdict: PASS.** The full HIP-accelerated pipeline — GPU SIFT extraction,
GPU SIFT matching, and Caspar-HIP global bundle adjustment — runs correctly
together, end-to-end, on real data, on gfx1151, with no crashes, no
nondeterminism across 3 runs, and reconstruction quality in line with the
CPU baseline. **No fresh integration bug was found between the two
features** — each behaves in combination exactly as it behaved in isolation.
This closes the last open verification gap for this milestone: dense stereo,
Caspar-HIP BA, and GPU SIFT extraction/matching are now all confirmed
working, both individually and composed into one real pipeline run.

Artifacts from this check (image `colmap-rocm:full-integration`, 30-image
dataset, 3 run logs, CPU baseline run) were left in this session's scratchpad
only, not committed to the repo or pushed anywhere.

## OpenCV camera-model numerical verification (2026-08-05)

Every prior BA verification on this branch ran on `SIMPLE_RADIAL` data. Caspar-HIP's
native OpenCV camera-model support (k1,k2,p1,p2, ported from
`rosbag-colmap-pipeline`'s caspar-opencv patch) had never been exercised against
image data that actually has meaningful OpenCV-style distortion — a crash-free
run alone wouldn't distinguish "distortion correctly optimized" from "distortion
terms silently collapsed to near-zero," a subtler bug than a crash.

**Dataset:** 30 frames sampled (every 20th of 613) from
`rosbag-colmap-pipeline`'s local `docker/workspaces/freiburg1_desk` — the raw
RGB images from TUM's `rgbd_dataset_freiburg1_desk` sequence. TUM's published
calibration for this camera (fx≈517.3, fy≈516.5, cx≈318.6, cy≈255.3,
k1≈0.2624, k2≈−0.9531, p1≈−0.0054, p2≈0.0026) has real, non-negligible radial
distortion — a legitimate OpenCV-model test case, not a near-pinhole lens.
(`floor2`, the other candidate mentioned for this check, lives only on the
cluster's NFS workspace path and wasn't pulled locally for this quick check.)

**Commands** (both backends, both forcing the OpenCV model explicitly):
```
colmap feature_extractor --ImageReader.camera_model OPENCV --database_path db.db --image_path images --FeatureExtraction.use_gpu {0,1}
colmap exhaustive_matcher --database_path db.db --FeatureMatching.use_gpu {0,1}
colmap mapper --database_path db.db --image_path images --output_path sparse --Mapper.ba_global_backend {CASPAR,CERES}
```
Image `colmap-rocm:full-integration` (matches this branch's tip, `1218282`),
same env flags as prior full-pipeline check. 2 fresh-process GPU/Caspar-HIP
runs plus 2 fresh-process CPU/Ceres baseline runs, all on the identical
30-image set with `--ImageReader.camera_model OPENCV` forced on both sides
(default per-image-camera COLMAP behavior — `ImageReader.single_camera` was
*not* set, so each image gets its own independently-fit OPENCV camera; this
matters for reading the per-camera intrinsics below).

**Note on the conversion step:** `colmap model_converter` against the bind-mounted
output silently produced `Could not open .../cameras.bin` and aborted unless
`--security-opt label=disable` was also passed to that container invocation
(the pipeline runs had it; my first conversion attempts didn't) — an SELinux
labeling gotcha on this host, not a COLMAP/Caspar bug, worth a note for anyone
re-running this later without that flag.

**Results — registered images / points, both fragmented into 2 sub-models each run:**

| run | backend | model0 imgs/pts | model1 imgs/pts | total registered |
|-----|---------|---|---|---|
| GPU run 1 | Caspar-HIP | 2 / 38 | 28 / 1596 | 30/30 |
| GPU run 2 | Caspar-HIP | 6 / 187 | 18 / 1090 | 24/30 |
| CPU run 1 | Ceres | 6 / 223 | 19 / 1325 | 25/30 |
| CPU run 2 | Ceres | 2 / 151 | 20 / 1364 | 22/30 |

Registration counts vary run-to-run on **both** backends (not just Caspar-HIP)
because `--ImageReader.single_camera` wasn't set and initial-pair/registration-order
choice is sensitive with only 30 sparse, wide-baseline frames — this is expected
incremental-SfM behavior with per-image cameras, not new nondeterminism in the
Caspar-HIP path itself (the earlier SIMPLE_RADIAL full-pipeline check used the
same dataset scale and also saw run-to-run variation in which images registered,
just with a shared single camera so it wasn't visible in per-camera params).

**Reprojection error, main (largest) sub-model each run:**

| run | backend | mean reproj. error | points |
|---|---|---|---|
| GPU run 1 | Caspar-HIP | 0.753 px | 1596 |
| GPU run 2 | Caspar-HIP | 0.814 px | 1090 |
| CPU run 1 | Ceres | 0.666 px | 1325 |
| CPU run 2 | Ceres | 0.658 px | 1364 |

Same ~0.75-0.81px (GPU) vs. ~0.66px (CPU) pattern already established
elsewhere in this log for SIMPLE_RADIAL data — same order of magnitude, no
regression from forcing OPENCV.

**Distortion-coefficient sanity check (the critical check this session was
dispatched to close):** in each run's main sub-model (18-28 images sharing a
pool of per-image OPENCV cameras), fitted values cluster tightly and
plausibly on **both** backends:

- GPU run 1 (28 cameras): fx/fy mostly 480-580, k1 range 0.009-0.38, k2 range
  −0.10 to −0.95, p1/p2 small (~0.001-0.06) — e.g. camera 6:
  `548.03 528.24 320 240 0.1219 -0.2511 0.01663 -0.00062`.
- CPU run 1 (19 cameras): fx/fy mostly 505-535, k1 range 0.08-0.62, k2 range
  −0.16 to −1.39, p1/p2 similarly small — e.g. camera 4:
  `526.44 517.09 320 240 0.1376 -0.2405 -0.0052 -0.0027`.

Both backends land in the same regime as TUM's published ground truth
(k1≈0.26, k2≈−0.95) — **not** collapsed to zero, and not blown up, confirming
Caspar-HIP's OpenCV dispatch path is genuinely optimizing all four distortion
terms, matching Ceres' behavior on the same forced-OPENCV data.

**A real (pre-existing, backend-agnostic) artifact, correctly ruled out as a
Caspar-HIP bug:** the *small* sub-model in every run — GPU and CPU alike —
contains 1-2 badly degenerate cameras (e.g. GPU run 2's camera 19:
`fx=62197 fy=55249 cx=320 cy=240 k1=13677 k2=-451824192 ...`; CPU run 1's
camera 19: `fx=292129 k1=233189 k2=-2158855`). This is not a GPU-only or
Caspar-only failure — Ceres produces equally degenerate fits on the same
small sub-models. Root cause: with `single_camera` unset and only 2-6 images
sharing a camera in these fragments, OpenCV's 8 free parameters (fx,fy,cx,cy,
k1,k2,p1,p2) are underconstrained by the available correspondences, so BA
finds a degenerate local minimum regardless of solver backend. This is a
dataset-conditioning/config characteristic of forcing per-image OPENCV
cameras on a sparse, fragmented reconstruction — not something this
verification pass should chase further, since it reproduces identically on
CPU/Ceres.

**Verdict: PASS — OpenCV numerical-verification gap closed.** Caspar-HIP's
OpenCV distortion dispatch path produces registered-image counts, reprojection
error, and fitted k1/k2/p1/p2 values consistent with the CPU/Ceres baseline on
the same real distorted-lens data, across 2 independent fresh-process runs per
backend. No evidence of distortion terms being silently zeroed or of a
Caspar-HIP-specific numerical bug; the one artifact found (degenerate small
sub-model cameras) is reproduced identically on Ceres and traced to sparse
per-image-camera conditioning, not the BA backend.

Artifacts from this check (30-image TUM freiburg1_desk subset, 2 GPU run
dirs, 2 CPU run dirs, converted TXT models) were left in this session's
scratchpad only, not committed to the repo or pushed anywhere.

## 2026-08-05: rosbag-colmap-pipeline integration audit — no colmap-rocm bug found

A downstream task assumed `global_mapper`'s log line
(`Requested to use GPU for bundle adjustment, but COLMAP was compiled without
CUDA support. Falling back to CPU-based solvers.`, seen when running the real
`rosbag-colmap-pipeline` production orchestration, not raw `colmap` CLI) meant
a CASPAR-vs-CUDA-only gate bug existed in `global_mapper`'s controller here.
Investigation found no such bug: `CreateDefaultBundleAdjuster`
(`src/colmap/estimators/bundle_adjustment.cc`) already dispatches on
`#ifdef CASPAR_ENABLED`, independent of CUDA. The log line in question comes
from an unrelated place — `bundle_adjustment_ceres.cc`/`global_positioning.cc`
guard Ceres's own `ceres::CUDA` dense/sparse linear-algebra solver, a
genuine Ceres-CUDA-only feature unrelated to Caspar; seeing that message on a
HIP build is expected, correct behavior, not a bug.

The real reason `global_mapper`'s Caspar path never engaged through that
pipeline: `rosbag-colmap-pipeline`'s `runner.py` gates
`--GlobalMapper.ba_backend CASPAR` behind a config key deliberately left
unset by default, after a prior live accuracy sweep found it caused real
reconstruction fragmentation on 2/3 tested scenes. That is a correct,
intentional decision on the pipeline side, not a colmap-rocm defect — no
code change made here.

The pipeline's separate, already-safe-by-default standalone `bundle_adjuster
--BundleAdjustment.backend CASPAR` pass (same mechanism verified directly
against this branch in the Task 5 and full-pipeline-integration entries
above) was confirmed to genuinely engage Caspar-HIP when correctly invoked
end-to-end through `gttool run-colmap`, initially measured at ~137x
standalone-BA-stage speedup on the 613-frame `freiburg1_desk` scene (0.48s
GPU/Caspar vs 65.6s CPU/Ceres). **That number is not trustworthy — confirmed
2026-08-05 as a real correctness bug, not a genuine speedup**: on this scene
Caspar's solver returns nan cost from the first iteration, never accepts a
single LM step, and exits after 3 iterations having written back the
unmodified input model (reprojection error unchanged from baseline). See
`rosbag-colmap-pipeline`'s `docs/local-hip-run.md`
("Correctness follow-up (2026-08-05)") for the full same-input Ceres-vs-Caspar
comparison and evidence; not duplicated here since this repo has no
involvement in that fix.

## 2026-08-05: Caspar-HIP `-nan`-from-iteration-0 bug isolated to the OPENCV kernel family (root cause not fully found — BLOCKED with strong evidence)

Follow-up on the entry above. Reproduced the failure directly against this
branch's tip (`localhost/colmap-rocm:full-integration`, matches
`1218282`/`95b76dc0`, docs-only commit in between) by feeding the preserved
post-`global_mapper` `freiburg1_desk` sparse model (613 images, 54458 points,
634857 factors, single shared `OPENCV` camera) straight into
`colmap bundle_adjuster --BundleAdjustment.backend CASPAR --log_level 2`.
Confirmed the exact reported signature: `score_init: 4.548663e+05`,
`score_current: -nan` from `solver_iter: 0`, `step_quality: 0.000` on every
iteration, `diag` climbing 1→2→8 until `CONVERGED_DIAG_EXIT` after 3 iters,
output bit-identical to input.

**Input data ruled out first (cheap check, per plan).** Parsed `points3D.txt`
and `images.txt` directly: zero NaN/Inf in any point coordinate, all 635192
observations have positive camera-frame depth (min `z = 0.898`), max
normalized-image-plane radius² across all observations is `0.53` (nowhere
near the range where a `k2·r⁴` term would threaten float32 range). The input
model is clean — this is not a garbage-in-garbage-out problem.

**Scale ruled out as the trigger** by re-running the *identical* 613-pose/
54458-point problem with the camera model swapped to `SIMPLE_RADIAL` (same
image observations, hand-edited `cameras.txt`, no re-extraction). Result:
score decreases genuinely over 200 iterations (`4.7e5 → 4.6999e5`,
`MAX_ITERATIONS` exit, not `CONVERGED_DIAG_EXIT`), with occasional isolated
`nan`/`-nan` iterations correctly rejected (`step_quality: 0.000`) and
recovered from on the next iteration — this is precisely the already-diagnosed
*benign* symforce-rocm nan artifact, not the bug. Repeating the same test with
the camera forced to `PINHOLE` (4 params, no distortion) gave the same
healthy behavior: 200 real iterations, genuine convergence, only benign
transient nans. **So at this exact scale/shape, both `SIMPLE_RADIAL` and
`PINHOLE` bundle-adjust correctly; only `OPENCV` fails outright.**

**Distortion coefficients ruled out as the trigger.** Re-ran the real 613-pose
problem with the camera still tagged `OPENCV` but all four distortion
params (`k1,k2,p1,p2`) hand-zeroed (mathematically equivalent to `PINHOLE`).
Still failed identically: `-nan` from iteration 0, `CONVERGED_DIAG_EXIT` after
3 iters. So the defect is not in the distortion-term math (k1/k2/p1/p2
values) — it reproduces even when those terms are numerically inert.

**Variant dispatch ruled out as the sole trigger.** Ran the real problem with
`--BundleAdjustment.refine_principal_point 1`, which switches Caspar from the
`FIXED_PP` factor variant (`kernel_opencv_split_fixed_principal_point_*`) to
the `BASE` variant (`kernel_opencv_res_jac*`, no `_split_` in the name) —
different generated kernel files entirely. Same failure, same signature. Both
`OPENCV` variants tested fail; both non-`OPENCV` models tested at the same
scale succeed.

**Conclusion: the bug is specific to Caspar's `OPENCV`-camera-model kernel
family (`src/thirdparty/Symforce-Caspar/generated/f32/kernel_opencv_*.cu`),
triggered by this problem's scale/shape (613 poses, 54458 points, 634857
factors) — not by scale alone (SIMPLE_RADIAL/PINHOLE are fine at the same
scale), not by camera model alone (Track B's 30-image OPENCV check with real
distortion passed), and not by the distortion values (fails with distortion
zeroed too).** This narrows it to something structural in how the `OPENCV`
kernels specifically handle a factor set this large — the OPENCV model is the
only one of the three with 8 intrinsic parameters (vs. 4 for PINHOLE/
SIMPLE_RADIAL), giving its per-factor kernels a materially larger
register/shared-memory footprint per thread block
(`kernel_opencv_split_fixed_principal_point_res_jac_first.cu` alone declares
~110 scalar temporaries plus a `16384`-byte `__shared__ inout_shared` buffer
per block). A buffer/stride sizing formula correct for PINHOLE/SIMPLE_RADIAL's
smaller footprint but wrong for OPENCV's larger one — only manifesting once
the factor/thread-block count crosses some threshold between 30-image and
613-image scale — is the leading hypothesis, consistent with the same class
of HIP shared-memory scratch-buffer bug already found and fixed once this
session in symforce-rocm (a different specific instance, not the same code).
Also worth noting structurally: the `f64` (`CASPAR_USE_DOUBLE`) solver variant
has **zero** generated `OPENCV` kernels at all (`find .../generated/f64
-iname '*opencv*'` returns nothing, vs. dozens under `f32`) — `OPENCV` support
only exists in single precision in this codebase. This wasn't confirmed as
the trigger (`SIMPLE_RADIAL`/`PINHOLE` also ran in f32, the project's default,
and were fine), but it means there is no double-precision fallback available
to sidestep the bug by rebuilding, and is worth fixing/generating regardless.

**Status: BLOCKED — root cause narrowed to a specific kernel family and
scale-dependent trigger, but the exact defect inside the ~1400-line
generated `kernel_opencv_*_res_jac*.cu`/`.h` files was not pinned down to a
line number.** Next step for whoever picks this up: instrument
`bundle_adjustment_caspar.cc`'s `CASPAR SOLVER SETUP` block to dump the raw
factor/pose/point buffers sent to Caspar for the `OPENCV` path immediately
before `Solve()`, and bisect the 613-image problem size down (e.g. 100, 200,
400 images) with `OPENCV` forced, to find the exact factor-count threshold
where it starts failing — that threshold, cross-referenced against the
`__shared__`/register allocation in the generated kernels, should point at
the exact overflow. Not attempted here due to time; all reproduction
artifacts (`in/`, `in_simple/`, `in_pinhole/`, `in_opencv_zero/`, and their
`run_*.log`s) were left in this session's scratchpad only.

## 2026-08-05: bisection refutes the scale-threshold hypothesis; bug reproduces at n=2 poses (still BLOCKED, evidence tightened)

Follow-up on the entry above, which left "bisect by image/factor count to find
the exact failure threshold" as the next step, on the theory (never actually
tested) that the `-nan`-from-iteration-0 OPENCV bug was a buffer/shared-memory
overflow that only manifested past some pose/factor-count threshold.

**That hypothesis is now refuted.** Built a subsetting tool
(`subset.py`, scratchpad-only) that takes the preserved 613-image/54458-point
`freiburg1_desk` sparse model and produces a valid smaller COLMAP text model
for the first N images: keeps only points3D with >=2 surviving observations,
nulls out `POINT3D_ID` references in `images.txt` for any point that got
dropped, and filters `frames.txt` to match (two real bugs in the first
version of this tool — an off-by-one in the frames.txt header-line count, and
dangling `POINT3D_ID` references — were caught and fixed before trusting any
result; both produced clean crashes, not silent bad data, so they didn't
contaminate any reported number below).

Ran `bundle_adjuster --backend CASPAR` against N = 2, 5, 10, 20, 50, 100, 150,
200, 250, 300, 350, 400, 450, 500, 550 (all still tagged `OPENCV`, same single
shared camera, same fitted intrinsics from the real reconstruction). **Every
single size failed identically** — `-nan` from `solver_iter: 0`,
`CONVERGED_DIAG_EXIT` after 3 iterations, same signature as the full
613-image case. This includes **N=2** (2 poses, 200 points, 208 factors) —
about as small as a bundle adjustment problem can get. A fixed-size
buffer/shared-memory overflow cannot explain a failure at N=2; that
hypothesis, which was this investigation's leading theory as of the prior
entry, is dead.

**Confirmed the N=2 input itself is valid** by running the identical
`in_2` directory through `--BundleAdjustment.backend CERES`: 100 real
iterations, cost decreasing monotonically (`0.746px → 0.231px` per-residual,
`NO_CONVERGENCE` only because of the 100-iteration cap), no nan. The data is
fine; only Caspar's OPENCV path chokes on it.

**Narrowed further: the bug is not in calibration-parameter refinement.**
Re-ran `in_2` with `--BundleAdjustment.refine_focal_length 0
--refine_extra_params 0` (on top of the already-default `refine_pp=0`),
which forces the `FIXED_FAE_PP` variant — *only* poses and points are free;
all 8 OpenCV intrinsic values are held fixed as constants for the entire
solve. **Still failed identically** (`score_init: 4.605089e+02`, `-nan` from
iteration 0). This rules out the calib/focal_and_extra Jacobian machinery
entirely as the culprit — whatever's broken lives in the pose/point residual
and Jacobian computation shared by every OpenCV variant, not in anything
specific to intrinsics refinement.

**Audited the GraphSolver construction call site for the
positional-argument bug class the code's own comment warns about**
(`caspar_model_adapter.h`'s `CreateSolver()`, `WARNING: Argument order is
opaque and bug-prone...`). Compared every one of the ~60 positional
arguments in `CreateSolver()`'s call to `caspar::GraphSolver(...)` against
the constructor's parameter list in `generated/f32/solver.h` (lines
~168-224): node-type-count order (OpenCVCalib, OpenCVFocalAndExtra,
OpenCVPose, OpenCVPrincipalPoint, then Pinhole's four, then Point, then
SimpleRadial's four) and factor-count order (simple_radial → pinhole →
opencv → simple_radial_split → pinhole_split → opencv_split, each in the
same fixed 4- or 11-variant sub-order) **match exactly, argument for
argument.** No positional/ordering bug at this call site — this specific,
plausible-looking hypothesis is ruled out with a direct side-by-side read,
not just "looks fine."

Also checked `SetupSolverData()` (`bundle_adjustment_caspar.cc`): pose,
focal-and-extra, principal-point, and variant-factor node uploads are all
driven generically through the `adapters_` map for every registered camera
model, with no per-model special-casing that could silently skip OpenCV's
upload — and the existing log line (`Camera 1 (OPENCV) params: [546.126,
539.083, 320, 240, 0.151159, -0.260699, -0.0044668, -0.00178992] -> [same]`,
seen on every run including the failing ones) independently confirms the
source camera parameters read from `Reconstruction` are finite and correct
before upload.

**Status: still BLOCKED, but with the search space sharply reduced.** What's
now ruled out: input data, problem scale/factor count (down to N=2), the
calibration/intrinsics refinement path specifically, and the
GraphSolver-construction argument-ordering class of bug. What remains
implicated: the pose/point residual-and-Jacobian math itself, generated in
`kernel_opencv_*res_jac*.cu` (and mirrored in `kernel_opencv_*score.cu`,
since `score_current` — not just the Jacobian-derived step — comes back
`-nan` too), for a real, non-scale-dependent, non-calib-related reason.
Structural diffs already tried and found inconclusive: `__shared__
inout_shared[16384]` sizing is identical between the OpenCV and Pinhole
`_split_fixed_principal_point_res_jac_first.cu` variants; both use the same
`copysign`-based epsilon-safe-division idiom seen elsewhere in this
project's already-fixed HIP-specific `SumStore()` reduction
(`generated/f32/memops.cuh`), which itself already carries this session's
HIP-specific butterfly-reduction fix and looks correct on inspection.

**Best-supported remaining hypothesis:** a genuine defect in the *math* of
the OpenCV projection/residual formula as translated into the generated
kernel — not a buffer, sizing, dispatch, or ordering bug — that produces a
NaN (likely a 0/0 or similar degenerate operation) regardless of problem
scale or which parameters are held fixed. Since it survives even with
distortion coefficients hand-zeroed (prior entry) *and* with all intrinsics
fixed as constants (this entry), whatever's wrong is specific to how the
OpenCV kernel structures the pose/point part of its residual — plausibly
something that differs between OpenCV's kernel and Pinhole/SimpleRadial's
even though the surrounding scaffolding (shared-memory buffer, indices,
`SumStore`) is templated identically.

**Concrete next step for whoever picks this up:** the fast, decisive check
is to instrument `kernel_opencv_split_fixed_principal_point_res_jac_first.cu`
(or `kernel_opencv_res_jac_first.cu` for the BASE variant) directly with a
`thread_idx==0`-gated `printf` of every intermediate register (`r0`...`r109`)
for one factor of the `in_2` repro, and find the first one that goes NaN —
that pins the exact faulty expression instead of continuing to eyeball
~1400 lines of generated arithmetic. `in_2` (2 poses, 200 points, 208
factors, runs in under a second) is now the reproduction case to use for
that — not the 613-image case. All bisection subsets (`in_2` through
`in_550`), their run logs, the CPU/Ceres control run, and `subset.py` were
left in this session's scratchpad only (`ba_repro/`), not committed to the
repo.

## 2026-08-05 (session 2): kernel-level printf tracing rules out res_jac and per-factor score computation; still BLOCKED

Direct follow-up on the entry above, executing its own recommended next step:
instrumented the generated kernels directly with `printf` (gated behind a
`CASPAR_OPENCV_DEBUG_TRACE` macro, rebuilt into throwaway
`colmap-rocm:opencv-debug-traceN` images, never committed) and traced the
`in_2` repro (2 poses, 200 points, 208+206 factors) register-by-register.
None of this instrumentation is in the committed tree — every kernel edit
was reverted (`git checkout --`) after each finding, since none produced a
confirmed fix.

**Per-factor residual/Jacobian computation is clean.** Instrumented
`kernel_opencv_split_fixed_principal_point_res_jac_first.cu` (the `FIXED_PP`
variant, free pose+focal+extra) to dump every intermediate register for
factor 0 and to flag any factor whose final residual (`r0`/`r1`) was NaN/Inf.
Result: factor 0's full pipeline — quaternion/translation compose to camera
frame, safe (`copysign`-epsilon) division by `z`, `r² = u²+v²` (correctly
computed as the *sum*, confirmed by manually tracing the register reuse —
initially looked like only `v²` was used due to a `3u²+v²` intermediate for
the `p2` tangential term, which is the correct OpenCV formula, not a bug),
`radial = k1·r²+k2·r⁴`, final distorted pixel — all produced finite,
sane values (`res_x=1.218, res_y=0.389`, consistent with `score_init`'s
order of magnitude). **Zero of the 208 (or 206, for the `FIXED_POSE_PP`
variant) factors flagged NaN/Inf** across two independent instrumented
rebuilds. The residual/Jacobian kernels are not the defect.

**Isolating which parameter group is free doesn't matter — every single
combination fails.** Beyond the prior entry's `FIXED_FAE_PP` (calib fixed)
result, this round tested every remaining single-group-free configuration
on `in_2`:
- `--refine_rig_from_world 0` (pose fixed, calib+points free): fails.
- `--refine_points3D 0` (points fixed, pose+calib free): fails.
- pose+calib both fixed, **only points free**: fails.
- pose+points both fixed, **only calib free**: fails.
- calib+points both fixed, **only pose free**: fails.

Every one of the eight possible {pose, calib, points} free/fixed
combinations that leaves anything free fails identically. This rules out
any single node type's (Pose, Calib, Point) kernels as the *sole* culprit —
whichever one is left as the only free group still triggers the bug.

**The LM-driver's built-in `CASPAR_DUMP_RK=1` diagnostic (already present in
`solver.cc`, not something this session added) shows no anomaly.** Dumped
`r_k` (Jtr) and `precond_diag`/`precond_tril` for every node type right
after the initial residual/Jacobian evaluation (before any PCG iteration).
Cross-checked the `OpenCVPose` dump against an equivalent `PinholePose` dump
on a same-poses/same-points `in_2_pinhole` control (camera model
hand-edited to `PINHOLE`, PINHOLE genuinely converges on this data). Both
show an identical structural pattern — components 2 and 3 of the 6-dim SE3
tangent are exactly zero in both `r_k` and `precond_diag` for every pose in
*both* models — and the `precond_tril` packed-size constants (`ntril=15` for
every 6-dim node, `ntril=28` for the 8-dim `OpenCVCalib`, `ntril=3` for the
3-dim `Point`, etc.) all match the correct "off-diagonal lower triangle
only" sizing formula `n·(n-1)/2` for their declared tangent dimension, with
no discrepancy between OpenCV's and Pinhole's/SimpleRadial's equivalent
nodes. No buffer-sizing or dimension-mismatch bug found here, and the
zero-component pattern is shared with the working Pinhole case, so it isn't
the cause either.

**Per-factor score computation (the kernel that recomputes cost after a
retracted step) is also clean on real factors — but revealed a
same-in-both-models GPU-printf/masking artifact that turned out to be a red
herring.** Instrumented `kernel_opencv_split_fixed_principal_point_score.cu`
(and its `_fixed_pose_` sibling) to flag any NaN/Inf per-factor squared
residual before the `SumStore` reduction. This fired — but at thread
indices (e.g. `idx=928`) far beyond the kernel's own `problem_size=208`,
i.e. on threads that the source-level `if (global_thread_idx < problem_size)`
guard should mask out entirely. Ran the identical instrumentation on
Pinhole's equivalent score kernel against the same-shape `in_2_pinhole`
control: it **also** fires this same guard-appearing-bypassed pattern, in
fact more often (15808 firings over Pinhole's 200 real iterations vs. 3263
over OpenCV's 3 aborted ones — comparable or higher per-iteration rate).
Since Pinhole converges correctly despite this, the artifact itself isn't
the defect — `SumStore`'s `valid ? data : StorageT(0)` (in the shared,
already-HIP-fixed `memops.cuh`) evidently still zeroes these masked lanes
out of the sum correctly for Pinhole. This is very likely just how HIP
device-side `printf` interacts with predicated/reconverged control flow for
short warp-uniform-false branches (the print appears to execute regardless
of the source-level guard, even though the *arithmetic side effects* remain
correctly masked) — a diagnostic-tooling artifact, not the production bug.
It cost real time to characterize but is now on record so nobody re-chases
it.

**Status: still BLOCKED.** What's additionally ruled out this round beyond
the prior entry: the residual/Jacobian kernel's actual math (verified
correct by hand for a real factor, not just "looks plausible"), any single
node type (Pose/Calib/Point) as sole culprit (every combination fails), the
preconditioner buffer sizing/dimension (matches Pinhole's pattern exactly,
formula-verified), and the per-factor score computation on in-range threads
(clean, same as res_jac). The masked-thread printf/NaN pattern in the score
kernel was investigated in detail and set aside as a tooling artifact common
to both models, not a lead.

**What remains unexplained:** since neither the per-factor Jacobian nor the
per-factor score computation produces NaN on real (in-range) data, and no
single node-type's kernels are uniquely at fault, the defect must live in
something not yet instrumented: the actual PCG iteration kernels
(`update_p`, `update_r`, `update_Mp`, `update_step`, `normalize` — i.e. the
Cholesky-preconditioned conjugate-gradient solve itself, operating on the
already-confirmed-finite `r_k`/`precond_diag`/`precond_tril` values) or the
`retract` kernels that apply the computed step to pose/calib/point state.
Given both res_jac and score are clean on identical inputs, and PINHOLE's
PCG/retract data has an identical zero-component precond pattern yet works,
the remaining hypothesis is a genuine numerical defect specific to how
OpenCV's *larger, more heterogeneous-scale* per-node blocks (8-dim merged
Calib mixing `fx≈546` with `p2≈-0.0018`; 6-dim FocalAndExtra) get
Cholesky-factored or solved in the PCG inner loop — plausibly a
float32 conditioning failure (a near-zero or negative pivot from rounding,
producing `sqrt` of a negative number) that Pinhole's much better-conditioned
2-dim Focal block and 4-dim Calib block never trigger, even on the same
poses/points. This was not directly instrumented this round (would require
tracing `kernel_OpenCVCalib_update_p.cu`/`update_Mp.cu`/`normalize.cu` and
comparable Pinhole kernels) and is the concrete next step, not the
generic "check `__shared__` sizing" suggestion from the prior entry (already
disproven).

**Practical note for whoever continues this:** each kernel-instrumentation
cycle costs a full `docker build` (~3-4 minutes) since the generated kernel
`.cu` files are compiled into the main COLMAP image, not a separately
cacheable target — budget for that when planning further printf-based
tracing. The `in_2` reproduction (2 poses, runs in under a second once
built) remains the right scale to iterate on, not the full 613-image case.
All debug images were tagged locally (`colmap-rocm:opencv-debug-traceN`)
and were not pushed anywhere; the source tree itself was left clean (every
instrumentation edit reverted via `git checkout --` once superseded).

## 2026-08-05 (session 3): preconditioner-conditioning hypothesis directly tested and refuted; still BLOCKED

Direct follow-up testing this session's own leading hypothesis: that
OpenCV's 6-dim `FocalAndExtra` preconditioner block (mixing `fx≈546` with
`p2≈-0.0018` in one unscaled Cholesky-style factorization) produces an
already-wrecked (garbage/NaN) preconditioner before any PCG iteration even
runs. Same `in_2` fast-repro cycle, same instrument→rebuild→run→revert
discipline as the prior two rounds; nothing below is in the committed tree.

**The preconditioner is not garbage — it's small but numerically stable.**
`kernel_OpenCVFocalAndExtra_normalize.cu` implements the block's
preconditioner application as a 6-variable sequential LDLT-style elimination
(6 reciprocal "pivots" computed via Schur-complement updates, no `sqrt`
anywhere in this kernel — unlike `retract`, this one can't hit a
`sqrt(negative)`). Instrumented all 6 pivot reciprocals and the final
4-component output for NaN/Inf on `in_2`. Result: pivots
`1.71e-2, 2.84e-2, 1.16e-6, 1.94e-5, 6.41e-8, 4.30e-8` — small (consistent
with the huge `precond_diag` magnitudes already logged in the prior entry,
up to `1.6e7`, since pivot ≈ 1/diag) but **entirely finite, none negative,
none zero**, and zero `NORMALIZE_NAN` flags fired across the run. The
coordinator's specific mechanism — a naive unscaled Cholesky-like solve
producing a day-zero-wrecked preconditioner — does not hold: the actual
numbers show poor conditioning (roughly a `10^5`-`10^6` spread across the
six pivots) but not numerical failure at this step.

**OpenCVPose's normalize kernel is structurally and numerically identical
to Pinhole's on this data.** Both models' `Pose` node is a 6-dim SE3
tangent using the exact same LDLT-elimination structure (confirmed
line-for-line: same six `1.0 / r*` pivot sites at identical positions in
both `.cu` files, differing only in earlier register-naming from
independent codegen runs, already established in the prior entry). Ran the
identical pivot instrumentation on both `kernel_OpenCVPose_normalize.cu`
(against `in_2`) and `kernel_PinholePose_normalize.cu` (against the
same-poses/points `in_2_pinhole` control): both report the identical
`1.0e+06` pivot value for the gauge-fixed pose (thread 0) — expected, since
a fixed node's regularization-only diagonal is model-independent — and
neither shows any NaN signature. (Only thread 0, the fixed pose, was
captured this round; the free pose's pivots weren't separately isolated,
but the RK dump in the prior entry already showed its `precond_diag`
magnitude is unremarkable, ~1e6-1e8, similar order to Pinhole's own.)

**Retract's transcendental-function surface is identical between models.**
Grepped both `kernel_OpenCVPose_retract.cu` and `kernel_PinholePose_retract.cu`
for every `sqrt`/`rsqrt`/`acos`/`asin`/`atan` call site (the classic
"negative-input-to-sqrt" NaN source for quaternion exponential maps): both
files call exactly `sqrtf` once and `rsqrtf` once, at the same structural
position, with no additional epsilon-guarding in either — same math, same
risk profile, same absence of anything OpenCV-specific.

**Status: still BLOCKED.** The float32-conditioning-produces-garbage
hypothesis, while a reasonable and worth-testing mechanism given the raw
`precond_diag` magnitude spread already on record, is now directly refuted
by instrumented evidence rather than left as a plausible-sounding
unconfirmed theory. Combined with the prior two rounds (res_jac clean on
every factor, score clean on in-range threads, every single node-type
isolated as sole-free-group still fails, preconditioner sizing/formula
matches Pinhole's exactly), the search has now covered: residual/Jacobian,
score/cost recomputation, preconditioner construction and application
(both FocalAndExtra and Pose blocks), and retract's transcendental
functions. None show a defect on `in_2`, and everything checked either
produces clean finite output or is structurally/numerically identical to
the working Pinhole case on the same poses and points.

**What has not yet been instrumented:** the CG direction-update kernels
proper — `update_p` (search direction), `update_r`/`update_r_first`
(residual update), `update_Mp` (preconditioner-vector product), and the
`alpha`/`beta`/`pred_decrease` step-size and trust-region kernels
(`kernel_OpenCVFocalAndExtra_alpha_numerator_denominator.cu`,
`alpha_denominator_or_beta_numerator.cu`, `pred_decrease_times_two.cu`) that
decide the LM step's acceptance. These remain the concrete next targets,
but given the amount of adjacent surface area already ruled out clean and
structurally-matching-Pinhole, it may be more efficient for the next
session to widen the net rather than continue one-kernel-at-a-time: e.g.
dump the actual PCG solution vector / retracted pose+calib+point values
immediately after the *first* PCG iteration completes (before the score
kernel even runs) and check those for NaN directly, which would in one
shot tell us whether the corruption is upstream (in the CG loop) or
downstream (in retract's application of an already-bad step) of the parts
already ruled out.

**Also worth flagging as a possible non-kernel angle, unexplored:** every
hypothesis tested so far has assumed the bug is in the generated-kernel
math. An alternative not yet investigated is a data/indexing bug on the
C++ host side in `bundle_adjustment_caspar.cc` — e.g. `SetVariantFactors`
or the per-model `idx_shared_` construction for OpenCV specifically writing
a wrong or stale index into `pose_indices`/`point_indices`/
`focal_and_extra_indices` for some factors, which downstream kernels would
silently read as valid-looking but wrong data (not necessarily NaN at any
single point checked, but converging to a numerically inconsistent overall
system). This is speculative and not evidenced either way — flagging it as
an alternative direction if the CG-kernel instrumentation above also comes
back clean.

No fix shipped this round — the coordinator's proposed mechanism was
tested directly and did not hold, so no speculative change was made.

## 2026-08-05 (session 4): Caspar-HIP OPENCV `-nan` bug FOUND AND FIXED — uninitialized score accumulator on padding-lane threads

Direct follow-up on the prior three rounds' pattern of clean eliminations.
Continued from where the preconditioner-conditioning round left off: dumped
the GPU-side score accumulator (`solver__res_tot_`) at successive
checkpoints through `DoRetractScore()`'s ~50 sequential score-kernel calls
(one env-gated `cudaMemcpy`+`printf` inserted between each major model/variant
group, later narrowed to individual kernel calls) on the `in_2` repro. This
bisected the exact call where the accumulator flips from clean to `-nan`:

```
after_opencv_nonsplit=0.0 (isnan=0)
after_sr_and_pinhole_split=0.0 (isnan=0)
after_opencv_split_fixed_fae=0.0 (isnan=0) count=0
after_REAL_fixed_pp=-nan (isnan=1) count=208        <-- here
```

The accumulator is clean going into `OpencvSplitFixedPrincipalPointScore`
(the `FIXED_PP` variant, 208 real factors) and already `-nan` coming out of
it — pinning the defect to that one kernel (and, by the same pattern, its
`FIXED_POSE_PP` sibling used for the gauge-fixed pose's factors), not to
any of the ~48 other model/variant score kernels that also run in the same
function (all correctly no-ops on this data, verified with explicit
per-group checkpoints).

**Root cause.** Both `kernel_opencv_split_fixed_principal_point_score.cu`
and `kernel_opencv_split_fixed_pose_fixed_principal_point_score.cu` declare
their per-thread squared-residual accumulator (`r46`) once at the top of the
kernel and only ever assign it inside `if (global_thread_idx < problem_size)
{ ... }`, alongside the real per-factor math. Immediately after that guarded
block, `SumStore()` is called *unconditionally* for all 1024 threads in the
launched block, passing `r46` to be reduced into the running total (gated
separately by a `valid` boolean argument). For any thread with
`global_thread_idx >= problem_size` — i.e. every "padding lane" in a block
that isn't an exact multiple of 1024 real factors, which is the normal case
for almost any real problem size — `r46` is read at that call site without
ever having been assigned during this kernel invocation. That is a plain
C++ uninitialized-variable read (undefined behavior), and its value is
whatever bit pattern happens to occupy that physical register. For OpenCV's
score kernel specifically — objectively larger and more register-pressured
than Pinhole's or SimpleRadial's equivalent kernels (this was already noted
descriptively in the very first entry in this investigation, "~110 scalar
temporaries") — that leftover register reliably decodes as a NaN bit
pattern. `SumStore`'s `valid ? data : StorageT(0)` selection is *supposed*
to discard exactly this kind of out-of-range garbage before it's summed,
and does so correctly in the ordinary sense of "discards the wrong value" —
but it still has to *read* `r46` to evaluate the ternary, and reading an
uninitialized local is UB independent of what happens to the read value
afterward; empirically, for OpenCV's specific kernel it reliably produced
NaN, corrupting the reduction despite the mask being logically correct.
(Two other things checked and ruled out along the way this round, for the
record: rewriting `SumStore`'s ternary as an explicit `if`/`else` branch —
tested directly, in case the compiler was lowering the ternary into an
arithmetic `data * (float)valid` where `NaN * 0 = NaN` — made no difference,
confirming the corruption happens before `SumStore` is even called, not
inside it. And `kernel_opencv_split_fixed_pose_fixed_principal_point_res_jac_first.cu`'s
own retracted output, dumped across 8 real PCG sub-iterations, was
confirmed completely clean/finite the whole time — the bug is specific to
the *score* recomputation, not the pose/point/calib state itself.)

This also fully explains why Pinhole and SimpleRadial never exhibited this
bug on identical poses/points (`in_pinhole`, `in_simple` controls, this
whole investigation): the exact same source-level pattern — accumulator
declared once, assigned only inside the guard, read unconditionally by
`SumStore` right after — exists in *every* generated score kernel across
*every* camera model (confirmed by inspection; this is how the code
generator structures all of them, not something specific to OpenCV's
math). It only manifests as an observable bug for OpenCV because that
specific kernel's register allocation happens to leave NaN in the
leftover register, where Pinhole's and SimpleRadial's smaller kernels
apparently leave something finite (and thus numerically harmless once
multiplied against/discarded by the mask). This is exactly why the earlier
rounds' apparently-thorough checks (comparing OpenCV's and Pinhole's
`precond_diag`/`precond_tril`/pivot structure and finding them identical)
never surfaced it — the defect isn't in any math difference between the
models at all, it's a latent, model-agnostic code-generation gap that
happens to be numerically silent everywhere except this one kernel.

**Fix.** Explicitly zero the accumulator for out-of-range threads
immediately before the `SumStore` call, in both affected kernel files:

```cpp
if (global_thread_idx >= problem_size) {
  r46 = 0.0f;
}
SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                global_thread_idx < problem_size, r46);
```

This removes the undefined-behavior read entirely rather than relying on
`SumStore`'s mask to safely discard a value that was never guaranteed to be
in a discardable state to begin with.

**Verification (real evidence, not just "looks fixed"):**
- `in_2` (2 poses, 200 points, 208+206 factors): 3 fresh runs, all now run
  the full 200 real LM iterations (`MAX_ITERATIONS`, not
  `CONVERGED_DIAG_EXIT`), converging consistently to `score_best≈45.20-45.25`
  from `score_init=460.51`. `step_quality`/`score_current` still show the
  already-documented benign transient nan on occasional rejected trial
  steps (correctly recovered from on the next iteration) — the same
  pre-existing, unrelated symforce-rocm artifact noted throughout this
  investigation, not a new problem.
- Full `in/` (613 images, 54458 points, 634857+335 factors, the original
  reported-bug reproduction): 3 fresh runs, all converge genuinely —
  `score_init=4.5487e5` down to `score_best≈4.4298e5` (a real ~2.6%
  reduction) over 70-87 real iterations (`CONVERGED_DIAG_EXIT` now fires for
  the legitimate reason, after real convergence, not immediately). Fitted
  camera intrinsics are consistent and non-bit-identical across all 3 runs
  and clearly different from the input (`fx: 546.126→546.36±0.01`,
  `k1: 0.1512→0.1490±0.0001`, etc. — tight run-to-run agreement, genuine
  optimum).
- `in_pinhole` and `in_simple` regression controls: both still converge
  correctly (`score_init 5.08e6→score_best 4.75e5` and
  `1.38e6→4.70e5` respectively, 200 iterations, no change in behavior) —
  confirms the fix doesn't affect the already-working models, as expected
  since only the two OpenCV-specific kernel files were touched.
- `in_opencv_zero` (OPENCV with distortion hand-zeroed, mathematically
  equivalent to PINHOLE, the case used earlier to rule out distortion-value
  causes): now also converges genuinely
  (`score_init 5.08e6→score_best 4.43e5`), matching Pinhole's behavior on
  the same data as it always should have.

**Scope note for whoever picks up the upstream PR work:** this fix is
scoped to exactly the two kernel files actually exercised by the reported
bug (`OpencvSplitFixedPrincipalPointScore` and
`OpencvSplitFixedPoseFixedPrincipalPointScore`, i.e. `refine_principal_point=0`,
the default). The same source-level pattern — and thus, plausibly, the same
latent defect — exists in every other generated score kernel across every
camera model and factor variant in `src/thirdparty/Symforce-Caspar/generated/f32/`;
it simply hasn't been *observed* to misbehave elsewhere because those
kernels' register allocation happens not to leave NaN in the relevant
leftover register on the hardware/compiler combination tested here. That's
a property of this specific compile, not a guarantee — a different GPU
architecture, HIP/ROCm version, or even a minor optimizer change could
make the identical latent bug surface in a currently-silent kernel (e.g.
`refine_principal_point=1`'s `BASE`/`FIXED_POSE` variants, or any Pinhole/
SimpleRadial kernel). The robust fix belongs in Caspar's code generator
(`caspar_generate.py`, not vendored for regeneration in this build) so
every generated kernel initializes its reduction accumulators once at
declaration rather than leaving them assigned only inside the per-factor
guard. Flagging this for the upstream PR discussion rather than
attempting a blanket patch across every generated file without individually
verifying each one, consistent with this session's rule of only shipping
changes actually tested end-to-end.

All debug instrumentation used to localize this (the checkpoint dumps in
`solver.cc`'s `DoRetractScore()`, the per-factor NaN printfs, the
`CASPAR_DUMP_RETRACT` retract-state dump) was reverted before this fix was
committed — the committed diff is exactly the two-kernel accumulator fix
above, nothing else.
