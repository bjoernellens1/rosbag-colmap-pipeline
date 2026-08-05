# Running COLMAP on the A100 cluster

COLMAP reconstruction (`gttool run-colmap`) is the pipeline's slowest stage on CPU —
`configs/navigation.yaml`'s wider matching/loop-detection/longer final bundle-adjustment pass,
needed for correct reconstructions on harder scenes, makes this worse, not better, on CPU
(minutes for small scenes, tens of CPU-minutes to over an hour for larger ones). This pipeline
can dispatch `run-colmap` to the `cps-gpu-cluster` A100 nodes instead, via
[`ablator`](https://github.com/bjoernellens1/ablator) — the same job-queue orchestrator the
sibling `splatograph` project uses on this same cluster. **Verified working end-to-end
2026-08-04** — see the status note in `docs/superpowers/plans` history / commit log for the
full investigation; this doc covers the operational how-to.

The GPU accelerates COLMAP's `feature_extractor`/`sequential_matcher`/`global_mapper`'s global
positioning stage. Bundle adjustment currently still runs on CPU (see cuDSS warning below).
`scale-depth`/`depth-ba` are pure NumPy/SciPy and stay a cheap local/CPU step run afterward
against the same workspace.

## One-time setup

```bash
git submodule update --init ablator
uv pip install -e ablator/
```

`kubectl`/`helm` must already be configured against the cluster (`kubectl get nodes` should
list `k3s-wk-gpu1..4`) — see `ablator/docs/cluster-setup.md` for a from-scratch walkthrough
(Rancher access, kubeconfig) if not.

**The `colmap-rgbd-gt` GHCR package must be public.** Check at
`https://github.com/users/bjoernellens1/packages/container/colmap-rgbd-gt/settings` — if it's
private, cluster pods hang forever in `ContainerCreating` with zero events (no `ImagePullBackOff`,
nothing) since no `image_pull_secret` is configured. This can't be fixed via the GitHub API with
a normal token (returns 404) — it's a manual one-time web UI step.

## Data placement

Existing `data/workspaces/*` scenes stay on local disk — nothing about them needs to move. Only
a scene you intend to reconstruct on the cluster needs to live on NFS, since cluster pods can
only read/write the shared `cps-scratch1-tmp-v2-pvc` PVC (same NFS server backing this
machine's own `/mnt/cps_scratch1_tmp` automount):

```bash
gttool extract <bag> --workspace /mnt/cps_scratch1_tmp/bjoern/rosbag-colmap-pipeline/workspaces/<scene>
```

**Important**: `[types.reconstruct]`'s command copies the scene into `/local_workspace/{id}`
inside the pod (real node-local disk, the container's own writable layer) before running COLMAP,
and copies only the finished `colmap/`+`outputs/` back to NFS afterward. **Do not** run COLMAP
directly against the NFS-mounted path — COLMAP's `database.db` is SQLite, and SQLite over NFS is
unreliable (broken POSIX locking) — confirmed live, it produced two different confusing errors
(a bogus "directory doesn't exist" and later "SQLite error: disk I/O error") depending on what
state a prior failed attempt left the file in.

## Building the CUDA image

`docker/Dockerfile.cuda` builds COLMAP with `-DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=80`
(A100 compute capability) — everything else matches `docker/Dockerfile`'s CPU/ROCm build.

```bash
podman build -f docker/Dockerfile.cuda -t ghcr.io/bjoernellens1/colmap-rgbd-gt:cuda-<short-sha> .
podman push ghcr.io/bjoernellens1/colmap-rgbd-gt:cuda-<short-sha>
```

Use an immutable per-build tag, not `:latest` — update `configs/ablator.toml`'s
`[machines.a100cluster].image` to match. (This dev machine has an AMD GPU, not NVIDIA — building
this image doesn't need a GPU, but a true `--gpus`-enabled smoke test isn't possible locally;
verify on the cluster instead, e.g. `kubectl exec` into a debug pod with `nvidia.com/gpu: 1`
requested and run `nvidia-smi`.)

If apt hangs indefinitely on a package fetch during the build with zero progress: this build
host has intermittent flaky routing to Canonical's geo-DNS `archive.ubuntu.com`/
`security.ubuntu.com` (confirmed on both IPv4 and IPv6 at different times) — the Dockerfile
already switches to `de.archive.ubuntu.com` plus `Acquire::Retries`/short timeouts as a fix;
if it's still stuck, check `curl -sI http://archive.ubuntu.com` directly first.

### ⚠️ Do not use Ceres+cuDSS (GPU bundle adjustment) — confirmed broken, tried twice

A `Dockerfile.cuda` variant building Ceres from source with CUDA+cuDSS (NVIDIA's sparse Cholesky
library) for genuinely GPU-accelerated bundle adjustment (not just feature extraction/matching)
was attempted **twice, with two different cuDSS versions, and both are broken**:

1. **Attempt 1**: apt's cuDSS `0.8.0.10`. Live on the cluster, produced a corrupted
   reconstruction — trajectory fragmented into 39 disconnected segments, some up to 81m apart,
   scale-estimation confidence 0.
2. **Attempt 2**: after deep research identified that Ceres' unreleased cuDSS integration is
   only ever CI-tested against cuDSS `0.3.0.9` (five versions behind what apt shipped), rebuilt
   against NVIDIA's official `0.3.0.9` tarball instead — Ceres' own CI-pinned version. **Still
   broken**, and worse: 39-40 disconnected segments, some up to *201m* apart, scale confidence 0
   again.

Both attempts hit the identical self-contradictory Ceres log line: `"Linear solver failure.
Failed to compute a step: Success."` This rules out a simple cuDSS-version mismatch as the root
cause — since the exact CI-validated version reproduces the same corruption, this is a deeper,
currently-unresolved problem in Ceres' unreleased cuDSS integration itself (see
[ceres-solver/ceres-solver#1079](https://github.com/ceres-solver/ceres-solver/issues/1079) for a
related open upstream bug about cuDSS persisting numerical-factorization-error state across
calls, and [#1161](https://github.com/ceres-solver/ceres-solver/issues/1161) confirming no
tagged Ceres release has cuDSS support at all yet). Both builds themselves succeed cleanly and
`colmap -h` correctly reports `with CUDA` — this is a runtime numerical-correctness bug, not a
build/config mistake.

Also checked and ruled out: a COLMAP-side solver-configuration bug
([colmap/colmap#2758](https://github.com/colmap/colmap/pull/2758), which fixed how COLMAP wires
`sparse_linear_algebra_library_type = CUDA_SPARSE`) — read directly against COLMAP 4.1.1's actual
source (`bundle_adjustment_ceres.cc`): the fix is already included, and floor2's 267 images
legitimately clears every threshold (`min_num_images_gpu_solver=50`,
`max_num_images_direct_dense_gpu_solver=200` so it falls through to sparse,
`max_num_images_direct_sparse_gpu_solver=4000`) to select `SPARSE_SCHUR` + `CUDA_SPARSE`
correctly. So this isn't COLMAP misconfiguring Ceres either — the corruption is downstream of
that, inside Ceres' unreleased cuDSS integration and/or cuDSS itself. Research into NVIDIA's own
cuDSS release notes found a real history of *silent wrong-answer* bugs spanning exactly the
version range tried here (CUDSS-882, CUDSS-1003: wrong results, fixed in 0.6.0/0.7.0; CUDSS-1020,
CUDSS-1292/1293: wrong results, fixed in 0.8.0) — no version tested is confirmed clean, and no
one has publicly verified Ceres-master+cuDSS as numerically correct on Ampere.

**Do not point `configs/ablator.toml` at any `cuda-cudss-*` tag.** GPU bundle adjustment now uses
`cuda-caspar-*` images instead (see below) — Ceres+cuDSS is not worth revisiting unless Ceres
ships a tagged release with cuDSS support and cuDSS itself has moved well past the buggy version
range above.

### GPU bundle adjustment via Caspar — native OpenCV support, working

COLMAP >=4.1.0 ships **Caspar**, a native GPU bundle-adjustment solver bundled directly in COLMAP
(`src/thirdparty/Symforce-Caspar/`) — it is *not* a Ceres backend, so it sidesteps the whole
Ceres+cuDSS dependency chain above entirely. `docker/Dockerfile.cuda` builds COLMAP with
`-DCASPAR_ENABLED=ON` and otherwise uses apt's plain CPU-only `libceres-dev` (Ceres is still
needed for COLMAP's non-BA estimators, just not for BA itself).

**Caspar upstream only supports `PINHOLE`/`SIMPLE_RADIAL` camera models**, not this pipeline's
`OPENCV` (needed for real RGBD lens tangential distortion, `p1`/`p2`). Two workarounds were tried
and rejected: approximating the camera model as `SIMPLE_RADIAL` (dropping `p1`/`p2` — reproduced a
real 2-way scale-regime split), then undistorting RGB to `PINHOLE` before COLMAP ever saw it
(mathematically exact, but left a smaller residual split on navigation-corridor scenes — undistorting
the images at all shifted which SIFT features got matched, a scene-specific matcher-tuning
sensitivity, not a distortion-math bug).

**Fixed properly**: `docker/patches/caspar-opencv/` adds **native OpenCV camera-model support
directly to Caspar's CUDA kernels** — no image preprocessing at all. Caspar's camera models are
generated by [symforce](https://github.com/symforce-org/symforce)'s CUDA code generator from a
small symbolic Python function per model (`caspar_generate.py`); the patch adds an `opencv_core`
function (COLMAP's exact radial+tangential distortion formula, matching `sensor/models.h`'s
`OpenCVCameraModel::Distortion`) and registers all 15 factor variants (4 merged + 11 split — the
split variants are required, not optional, since COLMAP's own default `refine_principal_point =
false` means real BA calls hit the `FIXED_PRINCIPAL_POINT` split variant, not the merged `BASE`
one). See `docker/patches/caspar-opencv/README.md` for the full derivation, including the
riskiest part: `CreateSolver()`'s positional argument list to Caspar's generated constructor,
which is order-dependent and shifts entirely (not just appends) when a new camera model's node
types sort alphabetically before existing ones — verified 1:1 against the actual generated
constructor signature, not derived by pattern-matching.

Verified live 2026-08-04 — floor2 (267 frames, navigation corridor, the exact scene that showed
the residual split under the rectification workaround) reconstructed through the **full pipeline**
(`gttool run-colmap --gpu`, no image preprocessing) with:

| | poses | scale regimes | BA time |
|---|---|---|---|
| OPENCV/CPU baseline | 267/267 | 1 (no split) | ~122s |
| **OpenCV/Caspar (this fix)** | **267/267** | **1 (no split)** | **0.42s** |

Zero scale-regime split — matches the CPU baseline exactly, not an approximation — with the final
bundle-adjustment pass roughly **290x faster** (0.42s vs ~122s). `tableware1` (155 frames, tabletop
scan) was also verified clean under the earlier rectification approach and continues to work
under native OpenCV. `trolley_femto` (1599 frames) was benchmarked end-to-end: full reconstruction
in ~43 minutes (vs a CPU baseline where the BA pass alone took ~83 minutes).

`colmap/scale_regime_correction.py` still auto-corrects any split that does occur on some future
scene (independent per-regime metric anchoring) — always run `scale-depth` and check its log for
"internally-inconsistent scale regimes" after any cluster dispatch, same as every other scene
validated this session. `configs/navigation.yaml`'s `ba_backend: caspar` is enabled by default
(only takes effect when `--gpu` is also passed, so local CPU runs are unaffected) and controls
this standalone `bundle_adjuster` pass only.

### GlobalMapper's internal BA loop — real speedup, regression found and fixed

`global_mapper`'s own internal 3-iteration bundle-adjustment loop (separate from the standalone
`bundle_adjuster` pass above) was Ceres-CPU-only in COLMAP 4.1.1 — confirmed the dominant
wall-clock cost on large scenes (700-900s on 1600-2700 frame scenes), well past global positioning
and the final BA pass combined. `docker/Dockerfile.cuda` now builds from
[bjoernellens1/colmap's `caspar-opencv-support` branch](https://github.com/bjoernellens1/colmap/tree/caspar-opencv-support)
(pinned commit, not upstream's `4.1.1` tag) instead of the earlier overlay-patch approach — this
branch rebases our own OpenCV Caspar work (§ above, submitted upstream as
[colmap/colmap#4611](https://github.com/colmap/colmap/pull/4611)) on top of current `main`, which
includes [colmap/colmap#4484](https://github.com/colmap/colmap/pull/4484) "Support selecting
Caspar BA backend in global mapper" (merged after 4.1.1) — exposing `--GlobalMapper.ba_backend
CASPAR`.

**This flag is genuinely fast** (verified on `trolley_femto`: internal BA loop 11.1s vs 883.6s CPU,
~80x; `global_mapper` total 590s vs 2387s, ~4x; full pipeline 17.9min vs 45.5min, ~2.5x) but a live
accuracy sweep across three scenes found it introduces **real scale-regime-split regressions**:

| scene | frames | scale regimes (GlobalMapper Caspar) | baseline (standalone-BA-only Caspar) |
|---|---|---|---|
| tableware1 | 155 | 0 (clean) | 0 |
| floor2 | 267 | **2** | 0 |
| trolley_femto | 1599 | **5** | 0 |

Two of three scenes regressed from zero split to a real split — including `floor2`, which was
independently verified zero-split-clean with the (more mature, extensively validated) standalone
Caspar path. This is a real correctness gap in the newer, less-battle-tested upstream
`GlobalMapper.ba_backend` feature, not noise.

**Root cause and fix**: `global_mapper`'s internal loop runs each iteration in two stages —
a "fixed-rotation stage" (`constant_rig_from_world_rotation=true`, meant to stabilize positions
before touching rotation) followed by a full joint-optimization stage. Caspar's pose node is a
single retracted `Pose3` (rotation+translation together) with no mechanism to hold rotation
constant while translation is free — `constant_rig_from_world_rotation` is silently ignored when
`backend == CASPAR` (confirmed via source: zero references to it anywhere in
`bundle_adjustment_caspar.cc`), so the fixed-rotation stage silently degraded into a second full
joint-optimization pass instead of a stabilizing partial one. Fixed in `global_mapper.cc`'s
`IterativeBundleAdjustment` (commit `15906fd0` on `caspar-opencv-support`) by forcing Ceres for
just that one sub-stage when the configured backend is Caspar — the joint-optimization stage
(the actual dominant cost) still uses Caspar, preserving the speed win.

Re-verified clean on both regressed scenes after the fix:

| scene | frames | scale regimes (post-fix) | pre-fix |
|---|---|---|---|
| floor2 | 267 | 0 | 2 |
| trolley_femto | 1599 | 0 | 5 |

`trolley_femto`'s internal BA loop also dropped from 884s (CPU) to 298.6s (Caspar, post-fix) —
close to the earlier pre-fix speedup measurement, confirming the fix didn't sacrifice the
performance win.

**Consequence**: `colmap/runner.py`'s `global_mapper()` still gates this flag on a separate
`global_mapper_ba_backend` config key (checked independently of `ba_backend`, which stays on for
the safe standalone-BA path) — **not yet set in `configs/navigation.yaml`**, pending a decision on
defaulting it on now that the regression is fixed and re-verified. Global positioning and
retriangulation remain CPU-only regardless (confirmed via source: `global_positioning.cc`
hardcodes `SPARSE_SCHUR` with no Caspar option; `incremental_triangulator.cc` has no
threading/GPU hooks at all — nothing to flip there without further upstream work).

## Dispatching a job

```bash
ablator --config configs/ablator.toml plan specs/a100cluster_reconstruct_smoke.json
ablator --config configs/ablator.toml run --once
ablator --config configs/ablator.toml status floor2
ablator --config configs/ablator.toml collect floor2
```

`specs/a100cluster_reconstruct_smoke.json` targets `floor2` (267 frames, already known-good —
see `docs/workspace-structure.md`'s floor2 note) as the first real dispatch. After it completes,
run `scale-depth` locally against the same NFS workspace path and confirm the result matches
the already-verified floor2 baseline (267/267 poses, no scale-regime split, no disconnected
pose-outlier segments) before trusting the cluster path for other scenes — this is the
regression check that caught the cuDSS bug above; always run it after building/pushing a new
image, not just after a config change.

To retry a `done`/`quarantined` job: `ablator rerun <job_id>` (resets to pending), then dispatch
again with `run --once`. `ablator run --once` can take a while to return (it's not truly
fire-and-forget — it polls until the job either resolves or is claimed) so background it if
running interactively.

## Cluster facts worth knowing

- 8× A100-40GB across 4 worker nodes, KAI Scheduler, `batch` queue / `kai-batch-low` priority
  (lowest, preemptible — any custom priority class must stay under 100, or KAI treats it as
  non-preemptible and hard-caps it at zero quota).
- `mps = true` in `configs/ablator.toml` is required — GPU nodes run NVIDIA `Exclusive_Process`
  mode with no per-pod permission to change it; without MPS wiring, the first CUDA call in a job
  can fail with "device busy" even on a fully idle GPU.
- Public `ghcr.io/bjoernellens1/...` images pull directly on cluster nodes, no
  `image_pull_secret` needed (once the GHCR package visibility is actually public — see above).
- **Never set `pvc_scratch` directly** in `[machines.a100cluster]` — it makes ablator's manifest
  builder *also* auto-mount the same PVC a second time, read-only, as a `{scene}`-style
  "dataset" volume this job type never uses. Mounting one PVC twice under two different volume
  names hangs the pod forever in `ContainerCreating` with **zero** events (not even a `Pulling`
  event) — confirmed reproducible with a plain `busybox` pod, unrelated to image size. Use
  `extra_volumes` for a single explicit mount instead (see `configs/ablator.toml`).
- A shared in-cluster BuildKit pool exists (`buildkit-pool.ci.svc.cluster.local:1234`, reachable
  via `kubectl port-forward -n ci svc/buildkit-pool <local-port>:1234`) for faster/more reliable
  builds than this dev machine's own network — but it needs real Docker's `buildx` with the
  `remote` driver or raw `buildctl`, neither of which is available via this machine's
  podman-backed `docker` shim. Worth setting up properly if local builds keep hitting network
  issues.
