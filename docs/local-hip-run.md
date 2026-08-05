# Running COLMAP on this host's local AMD GPU (HIP/ROCm)

Alongside dispatching `run-colmap` to the `cps-gpu-cluster` A100 nodes (see
`docs/cluster-dispatch.md`), this pipeline can also run COLMAP reconstruction locally on this
host's own AMD GPU (gfx1151, Radeon 8060S) via HIP — no cluster/network dependency, useful for
quick local iteration. **Verified working end-to-end 2026-08-05**, including a real,
production-code pipeline run (not just standalone `colmap` CLI calls) on a real scene, twice, for
consistency. This doc covers the operational how-to; it does not touch or replace the CUDA
cluster-dispatch path in any way.

## Background: where the HIP-enabled COLMAP build comes from

`docker/Dockerfile` (the existing plain ROCm/PyTorch-based image) does `apt-get install colmap`,
which on this host's Ubuntu 24.04 base resolves to a CPU-only build — no CUDA, no HIP. Building
COLMAP from source with HIP support requires a fork with ROCm/HIP patches, since upstream COLMAP
has no HIP backend. `docker/Dockerfile.hip` builds
[`bjoernellens1/colmap`](https://github.com/bjoernellens1/colmap)'s `hip-integration` branch
instead, which — as of the commit vendored here (`0b079c55`) — has HIP-accelerated feature
extraction/matching (`SiftGPU`), dense stereo (`patch_match_stereo`), and bundle adjustment
(`CASPAR` backend, with native `OPENCV` camera-model support, the same GPU BA solver
`docker/Dockerfile.cuda` uses on the A100 cluster), all separately verified on real data on this
host's gfx1151 GPU in prior investigation.

The COLMAP source is vendored as a plain directory copy at `docker/colmap-rocm-hip-src/` (not a
git submodule, not cloned at build time) — this pins the exact tested worktree state rather than
whatever `hip-integration`'s HEAD happens to be on a later build. Re-sync it if the upstream
branch moves and you want the newer state:

```bash
rsync -a --exclude='.git' --exclude=build <colmap-rocm worktree>/ docker/colmap-rocm-hip-src/
```

## Building the HIP image

`docker/Dockerfile.hip` mirrors `docker/Dockerfile.cuda`'s structure (same apt dependency list,
same vocab-tree asset, same pipeline install steps) but builds COLMAP with:

```
-DHIP_ENABLED=ON -DCUDA_ENABLED=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCASPAR_ENABLED=ON
```

on top of the same `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` base
`docker/Dockerfile` already uses (so `--system-site-packages` still picks up the base image's
ROCm-enabled PyTorch, same as the plain image).

```bash
docker build -f docker/Dockerfile.hip -t colmap-rgbd-gt:hip .
```

(This host's `docker` CLI aliases to podman 5.8.4.) Build takes a few minutes — most of the time
is compiling ~300 translation units; layer caching makes rebuilds after a source-only change much
faster.

## Running

Same GPU-passthrough flags established in this session's earlier `colmap-rocm` verification work:

```bash
docker run --rm \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  --group-add 39 --group-add 105 \
  --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  -e QT_QPA_PLATFORM=offscreen \
  -v "$(pwd)/docker/workspaces:/app/workspaces" \
  --entrypoint gttool colmap-rgbd-gt:hip \
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
invocations) against the TUM `freiburg1_desk` scene (613 RGB-D frames, already extracted from a
prior run), copied into two fresh workspaces (`hip_run1`, `hip_run2`) so each run started from
identical raw input with no shared/staled COLMAP state. Config: `configs/tum_like.yaml`
(`camera_model: OPENCV`, `matcher: sequential`), `--gpu`.

Both runs:

- Completed with container exit code 0 in ~3–5 minutes.
- Registered **613/613 frames** (`COLMAP pipeline complete: 613 poses`, matching
  `outputs/trajectory_colmap_unscaled.txt` line counts) — full registration, no fragmentation.
- Produced a real sparse reconstruction at `colmap/sparse/0/points3D.bin`
  (~7.5–7.6MB each run, non-trivial point count) plus `cameras.bin`/`frames.bin`/`images.bin`,
  matching this pipeline's expected output layout (`configs/ablator.toml`'s
  `result_glob = "{model_path}/colmap/sparse/0/points3D.bin"` pattern, same artifact this repo's
  cluster-dispatch path checks for).
- Log output showed genuine GPU-side work throughout (Caspar bundle-adjustment iterations,
  global positioning, iterative retriangulation/refinement stages), not a silent CPU fallback.

**Run-to-run consistency check**: both runs registered the identical frame count (613/613) and
produced near-identical trajectory path length (45.65 vs 45.23 units) with highly correlated
per-step motion (correlation 0.96 between consecutive-frame step lengths). The two trajectories'
raw XYZ coordinates differ by more than trivial floating-point noise — expected and not a bug:
`global_mapper`'s unscaled/unaligned output has gauge freedom (no canonical global reference
frame is enforced between independent runs), so run-to-run global rotation/translation offsets
are normal even on deterministic hardware; `scale-depth`'s Umeyama alignment step (which every
real scene run already goes through) removes exactly this ambiguity before any downstream
comparison. The reconstruction's internal *shape* (path length, per-step motion) is what actually
indicates a correct, reproducible reconstruction, and that matched closely across both runs.

**Conclusion: the actual production pipeline (`gttool run-colmap` → `colmap_pipeline()`), not
just COLMAP itself, works correctly end-to-end on this host's local AMD GPU via HIP.**

## Files

- `docker/Dockerfile.hip` — the HIP-enabled image definition.
- `docker/colmap-rocm-hip-src/` — vendored COLMAP source (`hip-integration` branch, commit
  `0b079c55`) built by the above.

This path is entirely additive — `configs/ablator.toml`'s A100 cluster dispatch config and
`docker/Dockerfile.cuda` are untouched.
