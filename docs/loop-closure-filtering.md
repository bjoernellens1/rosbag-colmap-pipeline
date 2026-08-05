# Loop-Closure Filtering: fixing false-positive vocab-tree matches

## Root cause (kitchen1, 2026-08-05)

`kitchen1`'s reconstruction fragmented badly: 43 spatially-separated trajectory segments, only
214/361 frames in the majority cluster, worst segment 12.35m from the rest.
`colmap/pose_outliers.py`'s fragmentation check correctly refused to auto-resolve it (effective
majority 214 vs. rest 147, needed 3.0x dominance).

Traced to COLMAP's `rotation_averaging` step flagging 328/3948 (8.3%) relative-pose edges as
invalid (>10° error) — an unusually high fraction. Root cause: `--SequentialMatching.loop_detection`
(vocab-tree retrieval, matching each frame against visually-similar past frames, not just
temporally-adjacent ones) found visually-similar-but-wrong pairs on kitchen1's repetitive,
low-texture surfaces (cabinets, tile, countertops), and COLMAP verified those pairs with the exact
same two-view geometric-verification thresholds used for genuine, high-overlap sequential-neighbor
pairs — it has no built-in concept of "stricter verification for loop-closure-retrieved pairs."
Confirmed live: disabling `loop_detection` entirely fixed kitchen1 (1 segment, 361/361 frames,
`trajectory_sanity.json` passed).

## Why disabling loop_detection globally isn't the fix

`docs/mapper-selection.md`'s data shows scenes that only reconnect into a single model *with*
`sequential_overlap`/`loop_detection` tuning — hallway's incremental-mapper run fragmented into 5
disconnected sub-models even with that tuning; `floor3` similarly depends on loop-closure
reconnection for its longer/looping trajectory. Turning loop detection off scene-wide trades one
fragmentation failure mode (false loop closures) for another (missed genuine reconnections).

COLMAP itself offers no way to apply stricter verification specifically to loop-closure-retrieved
pairs within one `sequential_matcher` invocation — `--TwoViewGeometry.*`/`--SiftMatching.*`
thresholds are global. Tightening them to fight kitchen1-style false positives would also raise the
bar for legitimate sequential matches on already-weak-overlap stretches — the exact scenes
`sequential_overlap`/`loop_detection` were widened to rescue in the first place.

## The fix: `colmap/loop_closure_filter.py`

Runs after `sequential_matcher` succeeds (only when `loop_detection` actually applied) and before
mapping. For every geometrically-verified pair in `database.db`'s `two_view_geometries` table, it
classifies the pair by **sequence-rank gap** (each image's ordinal position in the frame_id-sorted
keyframe sequence, not raw frame_id — keyframe selection skips frames non-uniformly, so raw
frame_id distance and sequence-rank distance are not the same thing):

- **Sequential-window pairs** (`rank_gap <= sequential_overlap`, or an exact
  `quadratic_overlap` step `sequential_overlap * 2^k`) are never touched, regardless of match
  strength — COLMAP's own default verification already accepted them.
- **Loop-closure-only pairs** (rank gap not explainable by either window — could only be a
  vocab-tree retrieval) are held to a stricter `loop_closure_min_inliers` /
  `loop_closure_min_inlier_ratio` threshold and dropped (from both `matches` and
  `two_view_geometries`) if they fall short.

Two implementation pitfalls found and fixed during validation (now covered by
`tests/test_loop_closure_filter.py`):

1. Classifying by raw `frame_id` gap instead of sequence-rank gap misclassified many genuine
   sequential pairs as "loop" pairs whenever keyframe selection had skipped frames between them.
2. `quadratic_overlap` matches at *exact discrete* offsets (`overlap*2`, `overlap*4`, ...), not a
   continuous range up to the largest such offset — treating it as a continuous range let real
   false loop-closure pairs slip through unclassified.

A per-run sidecar report, `outputs/loop_closure_filter.json`, records how many loop pairs were
checked/dropped/kept, for QA visibility (same convention as `pose_outlier_filter.json` /
`scale_regime_correction.json`).

## Cross-scene validation

| scene | frames | baseline | with filter | notes |
|---|---|---|---|---|
| kitchen1 | 361 | 43 segments, sanity **failed** | **1 segment, passed** | fix target |
| floor3 | 790 | 1 segment (needs loop_detection) | **1 segment, 790/790, passed** | regression check — 64% of loop pairs dropped, reconnection still preserved |
| hallway | 2326 (of 2729 keyframes) | 44 segments | 6 segments, all spatially-adjacent/auto-merged, passed | informational only — hallway has known-bad camera calibration and a doorway-heavy trajectory (confirmed by project owner); not loop-closure-driven alone, disabling `loop_detection` entirely didn't fix it either. Improved but not treated as a target for this fix. |

Default thresholds (`loop_closure_min_inliers: 30`, `loop_closure_min_inlier_ratio: 0.35`, set in
`configs/navigation.yaml`) were validated against kitchen1 and floor3 — the fix-target and
regression-check pair. They were not tuned to hallway specifically; see the caveat above.
