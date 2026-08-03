# Mapper Selection: global vs. incremental

`colmap:mapper_type` selects which COLMAP reconstruction algorithm
`run-colmap` (and the `run-colmap` stage inside `full`) uses. **`global` is
the default** (`ColmapRunner.run_full_pipeline`'s code-level default, not
just this repo's `configs/default.yaml`) as of 2026-08-02.

- **`global`** → `colmap global_mapper`. This is COLMAP's built-in global
  Structure-from-Motion pipeline — the functionality that used to live in the
  separate GLOMAP project (`colmap/glomap` on GitHub) before GLOMAP was
  merged into COLMAP itself and archived. It solves rotation averaging +
  global positioning + iterative bundle adjustment over the *entire* image
  set in one shot, rather than growing the reconstruction incrementally.
  Requires COLMAP ≥4.0 (see [Installation](installation.md)).
- **`incremental`** → `colmap mapper`, COLMAP's classic image-by-image
  incremental SfM. Available as an explicit opt-in for a scene that
  genuinely needs it (see "When incremental might still be right" below).

## Why global is the default

Real, controlled comparison from this project's own use (2026-08-02), same
already-extracted-and-matched `database.db` fed to both mappers on two
production scenes:

| Scene | Frames | Incremental mapper | Global mapper |
|---|---|---|---|
| kitchen | 2056 | 598.8 CPU-minutes (~10 hours), then gave up ("No good initial image pair found"), 3 disconnected sub-models | 48 minutes, **100% connected** (2056/2056 images, 1 model), 98564 points, 0.878px mean reprojection error |
| hallway (long corridor, real content difficulty — Open3D odometry fitness degraded 0.86→0.37 across the trajectory) | 5139 | Fragmented into 5 disconnected sub-models even after retuning matching (wider `sequential_overlap`, `loop_detection` enabled) — largest held only ~518 frames (~10%) | Reconstructed as a single connected model |

No scene tested during this comparison did better with incremental than
global.

**Why global tends to win on difficult/long sequences:** incremental SfM
grows the model image-by-image, via repeated partial re-triangulation and
*local* bundle adjustment. A single weakly-connected segment (motion blur,
low parallax, a textureless stretch) can break the growing chain — COLMAP
then either re-seeds a new, disconnected sub-model from a different starting
pair, or, in a worse case, fails to find any good re-seed pair at all and
just stops (kitchen scene above). Global SfM instead sees the whole problem
at once and solves it as one joint optimization, so a locally weak segment
gets compensated by consistency constraints from the rest of the image set
rather than causing a hard break in the reconstruction.

## When incremental might still be right

Global reconstruction methods can, in some cases, be more sensitive to a bad
match or outlier propagating *globally* through the joint optimization,
rather than staying locally contained the way it would in an incremental
pipeline. Incremental mapper is kept as a non-default option
(`mapper_type: incremental`) for exactly this reason — if a future scene
shows the opposite pattern (global mapper produces a worse or fragmented
result, incremental succeeds cleanly), that is a legitimate, scene-specific
reason to override the default, not a sign that something is broken or that
the default choice was wrong in general.
