# Follow-up: optional dense reconstruction via COLMAP's HIP/ROCm PatchMatch port

**Status:** not started. Captured for future work, not yet verified or implemented.

## Why this exists

The pipeline currently produces sparse reconstructions only (camera trajectory +
sparse point cloud), which is all the primary deliverable (metric pseudo-GT
trajectory) needs. COLMAP's dense multi-view stereo step (`patch_match_stereo`)
is CUDA-only in upstream COLMAP, and this repo runs on AMD/ROCm hardware with no
CUDA available, so dense reconstruction was ruled out as infeasible here.

That may no longer be fully true: there is reportedly an open, unmerged COLMAP
pull request porting `patch_match_stereo` to HIP/ROCm, which would make AMD-GPU
dense reconstruction possible. **This needs independent verification before any
implementation work starts** — the specifics below (PR number, commit SHA, tested
GPU architectures) came from an external, unverified source and must be checked
against the actual COLMAP repository, not taken as fact.

## Verification checklist (do this first)

- [ ] Confirm a COLMAP PR matching this description actually exists and is open —
      check https://github.com/colmap/colmap/pulls (search "HIP", "ROCm",
      "patch_match_stereo AMD"). The PR number and commit hash below are
      **unverified** and may be wrong, stale, or fabricated.
- [ ] If found, confirm current mergeability/CI status and how actively it's
      maintained (a long-stalled unmerged PR is a maintenance risk to build on).
- [ ] Confirm which GPU architectures the PR has actually been tested on. The
      source claims testing on `gfx1100` (RX 7900 XTX) but NOT on this machine's
      APU architecture (`gfx1151`, Radeon 8060S / Ryzen AI Max+ 395) — ROCm
      officially supporting the *hardware* is not the same as this *specific PR*
      working on it. Do not assume gfx1151 works without testing.
- [ ] Confirm the claimed ROCm base image tag (`rocm/dev-ubuntu-24.04:7.2.4-complete`)
      exists on Docker Hub, the same way the existing `rocm/pytorch` tag was
      verified for the depth-BA Docker work (see git history / session notes) —
      don't assume it's real without checking.
- [ ] Before running on a full bag, validate on a small 10-20 image test set
      first, per the source's own recommendation.

## Proposed design (pending verification above)

Keep dense reconstruction strictly optional and downstream of the existing
trajectory pipeline — it must not become a dependency of the default `full`
command, and must not change how the metric trajectory is produced.

```text
Default trajectory pipeline (unchanged)
rosbag -> sparse COLMAP -> metric scaling -> optional depth-BA (kornia-rs)
       -> metric trajectory

Optional geometry pipeline (new, opt-in)
metric COLMAP model -> image_undistorter -> HIP patch_match_stereo
       -> stereo_fusion -> dense point cloud / optional mesh
```

Key point: dense MVS does not itself improve the trajectory — it consumes
already-estimated poses. So sparse reconstruction stays the correct default;
dense becomes an additional output for users who also want scene geometry
(e.g. for evaluating reconstruction quality, not just trajectory accuracy).

### New Docker image (do not touch the existing CPU image)

Add `docker/Dockerfile.hip` as a *separate* image from `docker/Dockerfile` (the
ROCm/PyTorch-based CPU-COLMAP image already in use). The existing image installs
COLMAP via `apt-get install colmap` (vanilla, no HIP support); a HIP-enabled
COLMAP needs to be built from source against the PR branch once it's verified to
exist. Do not replace the stable CPU image — most users/CI don't need or want a
from-source COLMAP build with an unmerged patch.

Sketch (specifics need re-verification per the checklist above, especially the
apt package list — COLMAP's actual build deps should be checked against its
current `docs/install.md`, not assumed from memory):

```dockerfile
FROM rocm/dev-ubuntu-24.04:7.2.4-complete   # VERIFY this tag exists

ARG COLMAP_HIP_COMMIT=<verify-this-before-use>

ENV CMAKE_PREFIX_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH}

# ... apt-get install build deps (verify against COLMAP's current docs) ...

RUN git clone https://github.com/colmap/colmap.git /opt/colmap \
    && cd /opt/colmap \
    && git fetch origin pull/<PR_NUMBER>/head:rocm-support \
    && git checkout "${COLMAP_HIP_COMMIT}"

RUN cmake -S /opt/colmap -B /opt/colmap/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_ENABLED=OFF \
      -DHIP_ENABLED=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
      -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF \
    && cmake --build /opt/colmap/build --parallel \
    && cmake --install /opt/colmap/build
```

### docker-compose service

A separate `dense-amd` service (not replacing `gttool`/`dev`), with the same
`/dev/kfd` + `/dev/dri` + `group_add: [video]` GPU passthrough already added to
`docker/docker-compose.yml` for the depth-BA work, plus `group_add: [render]`
and `security_opt: [seccomp=unconfined]` per AMD's documented ROCm container
setup (verify this against current AMD docs, not just the source's claim).

### Prerequisite: a real metric-model writer

This is the part most likely to actually be missing today and worth checking
first, independent of the HIP question: dense MVS needs a full COLMAP model
(cameras + images + points3D with tracks) at metric scale, not just a TUM
trajectory export. Check whether `optimization/depth_ba.py`'s
`DepthBAResult.to_colmap_model()` (added in the depth-BA work) already produces
a valid, complete metric COLMAP model suitable for `image_undistorter`, or
whether `pipelines/scale_only.py`'s plain scale-estimation path also needs an
equivalent "write back a scaled COLMAP model" step (today it only scales the
TUM trajectory export, not the COLMAP model files themselves). This is
independent, useful groundwork regardless of whether the HIP PatchMatch port
pans out.

### New CLI command (pending everything above)

```bash
gttool dense-colmap <workspace>
gttool full <bag> --dense   # optional flag on the full pipeline
```

Config sketch:

```yaml
dense:
  enabled: false
  backend: colmap_hip
  input_model: sparse_metric/0   # requires the metric-model writer above
  max_image_size: 1920
  geometric_consistency: true
  fusion:
    input_type: geometric
  mesh:
    enabled: true
    method: poisson
```

### Two distinct dense outputs worth producing (if this gets built)

| Reference | Strength | Weakness |
|---|---|---|
| COLMAP HIP MVS (`outputs/dense_colmap_rgb_mvs.ply`) | Independent dense RGB geometry | Texture/lighting dependent indoors |
| Orbbec depth-fusion (`outputs/dense_orbbec_rgbd_fusion.ply`) | Metric, dense, usually complete | Correlated with the RGB-D input being evaluated |

Agreement between the two is a useful confidence signal but neither is
independent ground truth on its own.

## Recommended next step

Before writing any code: verify the COLMAP PR actually exists and get its real
number/commit/tested-architecture list. If it doesn't exist or is abandoned,
this entire follow-up is moot and the honest fallback stays what's documented in
the main README today — sparse-only, dense MVS not available on this hardware.
