# Track C, Task 1 Report: Cherry-pick SIFT-only commits from jeffdaily/rocm-sift-gpu

**Date:** 2026-08-04
**Status:** DONE (documented fallback — GPU SIFT compiles and runs, but has a
pre-existing buffer-reuse bug; not folded into `hip-integration`)

## Summary

Classified all 10 commits on `jeffdaily/rocm-sift-gpu` individually (per task
instructions, not repeating the prior whole-branch rebase attempt). 3 were
SIFT-only and cherry-picked cleanly onto a new branch `colmap-sift-cherrypick`
(based on `hip-integration` tip `e8ad01ca`). The build succeeds under HIP on
gfx1151 and GPU SIFT genuinely initializes and extracts correct-looking
keypoints on the first image processed — but the process then crashes with a
GPU memory access fault non-deterministically on the second or third image, in
what an isolation probe shows is specifically SiftGPU's cross-image
buffer/texture reuse path, not the extraction kernel logic itself. Per the
task's stated scope, this was diagnosed with two targeted probes and then
documented rather than patched. `colmap-sift-cherrypick` was **not** folded
into `hip-integration`; instead it is kept and pushed to `origin` for a future
session, and only a `docs/rocm-integration.md` entry landed on
`hip-integration` directly.

## Commit classification (all 10, oldest to newest)

| # | Commit | SIFT-only? | Disposition |
|---|--------|------------|-------------|
| 1 | `658f8b56` | No — original (superseded) patch_match compat layer | Skip |
| 2 | `786e0963` | No — fixups to #1 | Skip |
| 3 | `e609b9f1` | No — compat layer rework | Skip |
| 4 | `def43b23` | No — introduces `cuda_to_hip.h`/`enable_language(HIP)` approach, superseded by PR #4420's own version already on `hip-integration` | Skip |
| 5 | `690348f3` | No — `gpu_mat_test` + version banner | Skip. Forgoes `mvs/gpu_mat_test` under HIP only — the "with HIP" version banner is already present on `hip-integration` via PR #4420 (verified in `src/colmap/util/version.cc.in`). |
| 6 | `566e4df7` | Docs only | Skip — targets a `README.rocm.md`/`doc/install.rst` HIP paragraph that doesn't exist on `hip-integration` in the expected form |
| 7 | `bf064e92` | **Yes** (SiftGPU + small additive touches to already-ported `cuda_to_hip.h`/`FindDependencies.cmake`) | **Cherry-picked** |
| 8 | `e95eb380` | **Yes** — SiftGPU only | **Cherry-picked** |
| 9 | `3345a981` | **Yes** — SiftGPU only | **Cherry-picked** |
| 10 | `e41e06e0` | Docs only | Skip — same reason as #6 |

Full reasoning, conflict-resolution detail, and the runtime diagnosis are in
`docs/rocm-integration.md` on `hip-integration` (commit `57b26614`).

## What was built and tested

- Branch: `colmap-sift-cherrypick` (pushed to `origin`), 4 commits on top of
  `hip-integration` tip `e8ad01ca`:
  - `b1a3f26b` — cherry-pick of `bf064e92`, 3 conflicts resolved (all
    cosmetic — HEAD already had semantically-identical HIP guards from
    PR #4420)
  - `7dd7a7fc` — cherry-pick of `e95eb380`, clean
  - `66eaa995` — cherry-pick of `3345a981`, clean
  - `77c6959f` — local fix-up repairing a formatting bug my own conflict
    resolution introduced (dropped newline broke compilation), found by the
    build failing on the first attempt
- `docker build -t colmap-rocm:hip-sift ~/git/colmap-rocm` — succeeds cleanly.
  `libcolmap_sift_gpu.a` links as a HIP static library.
- `feature_extractor --FeatureExtraction.use_gpu 1` on a 30-image (1280x720)
  subset of `~/git/rosbag-colmap-pipeline/data/workspaces/table1/rgb/`:
  - Confirmed GPU SIFT path selected (`"Creating SIFT GPU feature extractor"`
    log line), not CPU/GLSL fallback.
  - First image always succeeds with correct nonzero keypoint counts.
  - Crashes with a GPU memory access fault (`Page not present or supervisor
    privilege`, SIGABRT) **deterministically on the 2nd image processed by a
    given `SiftGPU` instance**, always after the 1st succeeds — confirmed by
    two independent clean runs (one plain, one with `AMD_SERIALIZE_KERNEL=3`)
    both faulting at exactly the same position on exactly the same image
    (`000213.png` as the 2nd GPU-processed image). A third run showed the
    same pattern once its own contamination (an already-populated database
    causing the 1st file to be skipped) is accounted for — its "3rd file"
    was still its "2nd GPU-processed image." Not run-to-run nondeterminism;
    a consistent "works once per instance, fails on reuse" signature.
  - Ruled out GPU contention from other containers on this host (`splat_train`
    etc.) — this fault type (illegal in-kernel access) differs from the
    `hipErrorOutOfMemory`/"Memory in use" signature contention would produce,
    and reproduced identically with and without a concurrent container.
  - `AMD_SERIALIZE_KERNEL=3` probe: fault still occurs at the same point,
    confirming it's synchronous with a specific kernel launch rather than a
    deferred async report. `AMD_SERIALIZE_KERNEL` does not itself print
    kernel names, so the specific faulting call was not identified — that
    would need a symbolic debugger (e.g. `rocgdb`) attached to the abort.
  - Isolation probe: running `feature_extractor` on the faulting image alone
    (i.e. as the 1st and only image) succeeds cleanly every time — proves the
    defect is in cross-image buffer/texture **reuse**, not the per-image
    extraction kernels themselves.

## Why this is a documented fallback, not folded in

The bug is real, reproducible, and upstream (present in the cherry-picked
commits from `jeffdaily/rocm-sift-gpu` as-is, not introduced by the
conflict resolution — the isolated single-image path proves the ported code
is functionally correct on its own). The most likely root cause, based on
which files the cherry-picked commits touch and the shape of the bug
(fine on a fresh object, breaks on reuse): `e95eb380`'s `CuTexObj`
rule-of-five rewrite (move-only semantics, handle nulling, guarded
destructor) or `3345a981`'s `BindTexture2D` → `BindTexture` (linear-binding)
switch — both touch exactly the texture-object lifecycle. Per this task's
scope (identify + cherry-pick, or document why not — not an open-ended
debugging task), two targeted diagnostic probes were run and the result
documented rather than attempting a fix.

## Repo state left behind

- `hip-integration`: unchanged except for one docs-only commit
  (`57b26614`, `docs/rocm-integration.md`). **Not reset, not force-pushed.**
- `colmap-sift-cherrypick`: kept (not deleted), pushed to `origin`. Contains
  the 3 cherry-picked SIFT commits, 1 local fix-up commit, and this report
  (`1682d716`), ready for a future session to resume debugging without
  redoing the classification or conflict resolution.
- Safety check performed before pushing the docs commit to `hip-integration`:
  confirmed `hip-integration`'s tip (`e8ad01ca` at the time) was an ancestor
  of `colmap-sift-cherrypick`. **This is now stale**: `hip-integration`
  subsequently advanced to `57b26614` (the docs commit itself), which is
  *not* on `colmap-sift-cherrypick`. A future session must re-verify
  ancestry (or `git rebase hip-integration colmap-sift-cherrypick` first)
  before any `git reset --hard` fold-in — doing so blind would silently
  drop this docs entry.

## Recommended next step (for a future session)

Bisect within the 3 cherry-picked commits: build with only `bf064e92` +
`e95eb380` (i.e. drop `3345a981`'s `BindTexture2D` → `BindTexture` switch)
and re-run the same 2-image test. If it still faults, the bug is in
`e95eb380`'s `CuTexObj` rule-of-five rewrite; if it's clean, `3345a981`'s
linear-binding switch is the cause. For kernel-level attribution of the
fault itself, attach a symbolic debugger (`rocgdb`) to the abort — a
`RelWithDebInfo` rebuild alone will not surface the kernel name.
