# Running COLMAP locally on this host's AMD GPU (HIP/ROCm) — the default build

Alongside dispatching `run-colmap` to the `cps-gpu-cluster` A100 nodes (see
`docs/cluster-dispatch.md`), the default `docker/Dockerfile` image builds COLMAP with HIP
acceleration for this host's own AMD GPU (tested on gfx1151, Radeon 8060S) — no cluster/network
dependency, useful for local iteration. **Verified working end-to-end 2026-08-05**, including a
real, production-code pipeline run (not just standalone `colmap` CLI calls) on a real scene,
across two independent builds/runs for consistency. This doc covers the operational how-to; it
does not touch or replace the CUDA cluster-dispatch path in any way.

## Background: where the HIP-enabled COLMAP build comes from

`docker/Dockerfile` used to do `apt-get install colmap`, which on this host's Ubuntu 24.04 base
resolves to a CPU-only build — no CUDA, no HIP. It now builds COLMAP from source instead, using
[`bjoernellens1/colmap`](https://github.com/bjoernellens1/colmap)'s `hip-integration` branch,
which — as of the commit vendored here (`0b079c55`) — has HIP-accelerated feature
extraction/matching (`SiftGPU`), dense stereo (`patch_match_stereo`), and bundle adjustment
(`CASPAR` backend, with native `OPENCV` camera-model support, the same GPU BA solver
`docker/Dockerfile.cuda` uses on the A100 cluster).

The COLMAP source is vendored as a plain directory copy at `docker/colmap-rocm-hip-src/` (not a
git submodule, not cloned at build time) — this pins the exact tested worktree state rather than
whatever `hip-integration`'s HEAD happens to be on a later build. Re-sync it if the upstream
branch moves and you want the newer state:

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

## Verification (2026-08-05)

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
- Log output showed genuine GPU-side work throughout (Caspar bundle-adjustment iterations,
  global positioning, iterative retriangulation/refinement stages), not a silent CPU fallback.

This result is consistent with an earlier round of verification on the same scene (two
independent runs registering the same 613/613 frames with highly correlated per-step motion,
correlation 0.96) done before this build was folded in as the default, confirming the merge into
`docker/Dockerfile` didn't change reconstruction behavior.

**Conclusion: the actual production pipeline (`gttool run-colmap` → `colmap_pipeline()`), not
just COLMAP itself, works correctly end-to-end on this host's local AMD GPU via HIP, as the
default local build.**

## Files

- `docker/Dockerfile` — the default image definition (HIP-enabled COLMAP build).
- `docker/colmap-rocm-hip-src/` — vendored COLMAP source (`hip-integration` branch, commit
  `0b079c55`) built by the above.

`configs/ablator.toml`'s A100 cluster dispatch config and `docker/Dockerfile.cuda` are separate
and untouched — see [Running COLMAP on the A100
Cluster](https://bjoernellens1.github.io/rosbag-colmap-pipeline/cluster-dispatch/) for that path.
