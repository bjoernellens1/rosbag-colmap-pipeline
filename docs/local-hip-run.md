# Running COLMAP locally on this host's AMD GPU (HIP/ROCm) — the default build

Alongside dispatching `run-colmap` to the `cps-gpu-cluster` A100 nodes (see
`docs/cluster-dispatch.md`), the default `docker/Dockerfile` image builds COLMAP with HIP
acceleration for this host's own AMD GPU (tested on gfx1151, Radeon 8060S) — no cluster/network
dependency, useful for local iteration. This doc covers the operational how-to; it does not touch
or replace the CUDA cluster-dispatch path in any way.

## Background: where the HIP-enabled COLMAP build comes from

`docker/Dockerfile` used to do `apt-get install colmap`, which on this host's Ubuntu 24.04 base
resolves to a CPU-only build — no CUDA, no HIP. It now builds COLMAP from source instead, using
[`bjoernellens1/colmap`](https://github.com/bjoernellens1/colmap)'s `hip-integration` branch,
which — as of the commit vendored here (`d6824820`) — has HIP-accelerated feature
extraction/matching (`SiftGPU`), dense stereo (`patch_match_stereo`), and bundle adjustment
(`CASPAR` backend, with native `OPENCV` camera-model support, the same GPU BA solver
`docker/Dockerfile.cuda` uses on the A100 cluster).

The COLMAP source is vendored as a plain directory copy at `docker/colmap-rocm-hip-src/` (not a
git submodule, not cloned at build time) — this pins the exact tested worktree state rather than
tracking a moving branch tip. Re-sync it if the upstream branch moves and you want the newer
state:

```bash
rsync -a --exclude='.git' --exclude=build <colmap-rocm worktree>/ docker/colmap-rocm-hip-src/
```

## Building

`docker/Dockerfile` builds COLMAP with:

```
-DHIP_ENABLED=ON -DCUDA_ENABLED=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCASPAR_ENABLED=ON
```

on top of `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` (so
`--system-site-packages` still picks up the base image's ROCm-enabled PyTorch). The GPU
architecture is a build arg (`ROCM_ARCH`, default `gfx1151`) — override it for a different AMD
GPU:

```bash
docker build -f docker/Dockerfile -t colmap-rgbd-gt .
# or, for a different AMD GPU target:
docker build -f docker/Dockerfile --build-arg ROCM_ARCH=<your-gfx-target> -t colmap-rgbd-gt .
```

`docker compose -f docker/docker-compose.yml build gttool` builds the same image via the
`gttool` service's default `dockerfile: docker/Dockerfile`.

(This host's `docker` CLI aliases to podman 5.8.4.) Build takes a few minutes — most of the time
is compiling ~300 translation units; layer caching makes rebuilds after a source-only change much
faster.

## Running

GPU-passthrough flags for this host's AMD GPU:

```bash
docker run --rm \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  --group-add 39 --group-add 105 \
  --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  -e QT_QPA_PLATFORM=offscreen \
  -v "$(pwd)/docker/workspaces:/app/workspaces" \
  --entrypoint gttool colmap-rgbd-gt \
  run-colmap /app/workspaces/<scene> --gpu --config /app/configs/tum_like.yaml
```

`--group-add 39`/`--group-add 105` are this host's numeric `video`/`render` GIDs — podman
accepts numeric GIDs without needing the group *names* to exist inside the container's
`/etc/group` (only `video` is defined there by default). `--gpu` on `run-colmap` sets
`colmap.use_gpu = true` in the config actually passed to `colmap_pipeline()` — this is a
pipeline-level flag, not CUDA-specific naming; it works identically for the HIP build.

## Verification (pipeline-level, not just standalone COLMAP)

Ran the real `gttool run-colmap` pipeline entry point (`src/colmap_rgbd_gt/pipelines/colmap_only.py`
via `cli.py`'s `run-colmap` command — actual production orchestration code, not raw `colmap` CLI
invocations) against the TUM `freiburg1_desk` scene (613 RGB-D frames), against a fresh build of
the now-default `docker/Dockerfile` image (via `docker compose build gttool`). Config:
`configs/tum_like.yaml` (`camera_model: OPENCV`, `matcher: sequential`), `--gpu`.

The run:

- Completed with container exit code 0.
- Registered **613/613 frames** (`COLMAP pipeline complete: 613 poses`, matching
  `outputs/trajectory_colmap_unscaled.txt` line counts) — full registration, no fragmentation.
- Produced a real sparse reconstruction at `colmap/sparse/0/points3D.bin` plus
  `cameras.bin`/`frames.bin`/`images.bin`, matching this pipeline's expected output layout
  (`configs/ablator.toml`'s `result_glob = "{model_path}/colmap/sparse/0/points3D.bin"` pattern,
  same artifact this repo's cluster-dispatch path checks for).
- Log output showed genuine GPU-side work throughout feature extraction, matching, global
  positioning, and iterative retriangulation/refinement — not a silent CPU fallback. (Bundle
  adjustment itself is a separate story: see the Caspar-HIP bug section below — `global_mapper`'s
  own internal BA loop stays on CPU/Ceres by default regardless of backend, an unrelated,
  documented fragmentation-regression gate, not a HIP limitation.)

This result is consistent with an earlier round of verification on the same scene (two
independent runs registering the same 613/613 frames with highly correlated per-step motion,
correlation 0.96) done before this build was folded in as the default, confirming the merge into
`docker/Dockerfile` didn't change reconstruction behavior.

**Conclusion: the actual production pipeline (`gttool run-colmap` → `colmap_pipeline()`), not
just COLMAP itself, works correctly end-to-end on this host's local AMD GPU via HIP, as the
default local build.**

## Caspar-HIP OPENCV bundle-adjustment bug: found, root-caused, and FIXED (2026-08-05)

The pipeline-level verification above exercised GPU SIFT extraction and matching, but not genuine
GPU bundle adjustment — `global_mapper`'s own internal Caspar BA path stays deliberately disabled
by default (a separate, documented fragmentation regression unrelated to HIP; see `runner.py`'s
`global_mapper()`). The standalone final BA pass (`bundle_adjuster --BundleAdjustment.backend
CASPAR`, gated on `colmap.ba_backend: caspar` + `use_gpu: true`) is the mechanism that actually
exercises Caspar-HIP.

An initial measurement of that pass on the 613-frame `freiburg1_desk` scene looked implausibly
fast (0.48s vs. 65.6s CPU/Ceres, ~137x). Investigation found this was **not a real speedup** — the
Caspar-HIP solver was failing immediately: `score_current: -nan` from iteration 0, `step_quality:
0.000` on every iteration (no LM step ever accepted), output bit-identical to the pre-BA input.
Ruled out as data/scale/distortion-value causes by controlled substitution (PINHOLE and
SIMPLE_RADIAL on the identical poses/points converged correctly; OPENCV with distortion
hand-zeroed — mathematically equivalent to PINHOLE — still failed identically) — isolating the
defect specifically to Caspar's `OPENCV`-model kernel family.

**Root cause** (bisected via GPU-side score-accumulator dumps through `DoRetractScore()`'s ~50
sequential score-kernel calls, pinning the exact failing call): two OpenCV-specific score kernels
(`kernel_opencv_split_fixed_principal_point_score.cu` and
`kernel_opencv_split_fixed_pose_fixed_principal_point_score.cu`) declare their per-thread
squared-residual accumulator once at the top of the kernel and only assign it inside a
problem-size guard, but `SumStore()` reads that accumulator *unconditionally* for every thread in
the launched block right after. For "padding lane" threads (`global_thread_idx >= problem_size` —
the normal case for almost any real problem size), the accumulator is read without ever having
been assigned during that invocation: a plain uninitialized-variable read (undefined behavior).
`SumStore`'s masking logic correctly discards the value afterward, but still has to read it first
to evaluate the mask — and for OpenCV's specific kernel (larger and more register-pressured than
Pinhole's/SimpleRadial's equivalents), that leftover register reliably decoded as NaN, corrupting
the whole reduction. The identical source-level pattern exists in every generated score kernel
across every camera model — it just happens to be numerically silent everywhere except this one.

**Fix** (`bjoernellens1/colmap`'s `hip-integration` branch, commit `b5ead5a9`): explicitly zero
the accumulator for out-of-range threads immediately before `SumStore` is called, in both affected
kernel files — removing the undefined-behavior read entirely.

**Verification** (3 fresh runs each, real numbers not just "looks fixed"):
- 2-pose/200-point minimal repro: all 3 runs now complete the full 200 real LM iterations,
  converging consistently to `score_best≈45.20-45.25` from `score_init=460.51`.
- Full 613-image/54458-point problem (the original bug reproduction): all 3 runs converge
  genuinely — `score_init=4.5487e5` down to `score_best≈4.4298e5` (~2.6% real reduction) over
  70-87 real iterations, with non-bit-identical, tightly-agreeing fitted camera intrinsics across
  runs (`fx: 546.126→546.36±0.01`, `k1: 0.1512→0.1490±0.0001`).
- PINHOLE, SIMPLE_RADIAL, and OpenCV-with-zeroed-distortion controls all still converge correctly
  — confirms the fix doesn't affect the already-working models.

This repo's `docker/colmap-rocm-hip-src/` vendored copy was updated 2026-08-05 from the pre-fix
`0b079c55` to the fixed `d6824820` (the docs commit immediately after the fix landed), and
re-verified directly against this pipeline's own data (not just the upstream repro): running
`bundle_adjuster --BundleAdjustment.backend CASPAR` on a real 361-frame OPENCV-model
reconstruction (`kitchen1`) through the rebuilt image now produces finite scores and a genuine
(non-bit-identical) camera-parameter refinement — mean reprojection error `0.803997px ->
0.803996px`, no NaN, no premature 3-iteration bailout.

**Scope note**: the same uninitialized-accumulator pattern exists in every other generated score
kernel across every camera model — currently unpatched (numerically silent so far, but not proven
safe by construction) and flagged as an open risk for the upstream code generator to address,
rather than blanket-patching files that weren't individually verified as affected.

## Files

- `docker/Dockerfile` — the default image definition (HIP-enabled COLMAP build).
- `docker/colmap-rocm-hip-src/` — vendored COLMAP source (`hip-integration` branch, commit
  `d6824820`) built by the above.

`configs/ablator.toml`'s A100 cluster dispatch config and `docker/Dockerfile.cuda` are separate
and untouched — see [Running COLMAP on the A100
Cluster](https://bjoernellens1.github.io/rosbag-colmap-pipeline/cluster-dispatch/) for that path.
