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

### ⚠️ Do not use Ceres+cuDSS (GPU bundle adjustment) yet

A `Dockerfile.cuda` variant building Ceres from source with CUDA+cuDSS (NVIDIA's sparse Cholesky
library) for genuinely GPU-accelerated bundle adjustment was attempted and **is broken** — cuDSS
support only exists on Ceres' unreleased `master` branch (no tagged release has it), and live on
this cluster it silently produced a corrupted reconstruction (a reconstruction that fragmented
into 39 disconnected trajectory segments, some up to 81m apart, scale-estimation confidence 0)
while `global_mapper.log` showed a self-contradictory solver error: `"Linear solver failure.
Failed to compute a step: Success."` — a real bug in Ceres' in-development cuDSS integration,
not a build/config mistake (the build itself succeeds and `colmap -h` reports `with CUDA`
correctly). **Do not point `configs/ablator.toml` at a `cuda-cudss-*` tag** until Ceres ships a
stable, tagged release with cuDSS support and this has been re-validated. The current
`cuda-<sha>` image (CUDA feature-extraction/matching, CPU bundle adjustment) is the
known-good, verified-correct one to use.

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
