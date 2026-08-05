"""Post-matching filter for false-positive loop-closure pairs.

Root-caused 2026-08-05 on a real scene (kitchen1): COLMAP's vocab-tree
loop-closure retrieval (`--SequentialMatching.loop_detection`) found
visually-similar-but-wrong image pairs on kitchen1's repetitive,
low-texture surfaces (cabinets, tile, countertops), and those pairs went
through the SAME geometric-verification thresholds as genuine,
high-overlap sequential-neighbor pairs. The result: rotation averaging
flagged 328/3948 (8.3%) of relative-pose edges as invalid, and the
reconstruction fragmented into 43 spatially-separated segments (only
214/361 frames in the majority). Disabling loop detection entirely fixed
kitchen1 (1 segment, 361/361 frames) but isn't generalizable -- other
scenes (hallway, per docs/mapper-selection.md) only reconnect into one
model WITH loop detection enabled, so turning it off scene-wide trades
one fragmentation failure mode for another.

COLMAP itself has no mechanism to verify loop-closure-retrieved pairs
more strictly than temporally-adjacent sequential pairs within one
`sequential_matcher` invocation -- `--TwoViewGeometry.*`/
`--SiftMatching.*` verification thresholds are global, so tightening
them to fight kitchen1-style false loop closures would also raise the
bar for legitimate sequential matches on already-weak-overlap stretches
(the same fragile scenes `sequential_overlap`/`loop_detection` were
widened to rescue in the first place).

This module runs after `sequential_matcher` and before mapping: it
inspects `database.db`'s already-computed `two_view_geometries` (COLMAP's
own verified-inlier count per pair) and applies a STRICTER threshold only
to pairs that could not have been found by the sequential window (i.e.
pairs whose frame-index gap exceeds `sequential_overlap` -- these can only
be loop-detection retrievals). Genuine sequential-neighbor pairs are never
touched, regardless of how weak their match is -- COLMAP's own default
verification already accepted them, and this module has no stronger
prior about those than COLMAP itself does.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colmap_rgbd_gt.colmap.database import COLMAPDatabase
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MIN_INLIERS = 30
DEFAULT_MIN_INLIER_RATIO = 0.35


@dataclass
class LoopClosureFilterResult:
    action_taken: bool = False
    reason: str = ""
    n_pairs_checked: int = 0
    n_loop_pairs: int = 0
    n_sequential_pairs: int = 0
    n_dropped: int = 0
    dropped_pairs: list[dict[str, Any]] = field(default_factory=list)


def _frame_id_from_image_name(name: str) -> int:
    # Same convention as pose_extract.py's frame_id derivation --
    # filenames are zero-padded frame indices, e.g. "000123.png".
    return int(name.split(".")[0])


def filter_loop_closure_matches(
    database_path: Path,
    sequential_overlap: int,
    quadratic_overlap: bool = False,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
) -> LoopClosureFilterResult:
    """Drop loop-closure-only pairs (sequence-rank gap beyond what
    `sequential_overlap`/`quadratic_overlap` could explain) whose
    verified-inlier count or inlier ratio falls below the given
    thresholds. Sequential/quadratic-window pairs are always kept -- this
    only prunes matches that vocab-tree retrieval could have introduced.

    Uses each image's RANK in the frame_id-sorted sequence, not its raw
    frame_id, to determine whether a pair falls inside the sequential
    window -- `--SequentialMatching.overlap` counts neighbors by ordinal
    position in the registered keyframe sequence, and keyframe selection
    (max_frame_gap) skips frames non-uniformly, so raw frame_id distance
    and sequence rank distance are not the same thing. When
    `quadratic_overlap` is enabled, COLMAP additionally matches each image
    against neighbors at EXACT discrete offsets (`overlap * 2^k` for
    k=1,2,3,...) -- those specific rank gaps are also genuine
    sequential-window matches and are exempted individually. This is a
    discrete set, not a continuous range: a rank gap that doesn't land on
    the linear window or one of these exact steps is not explainable by
    sequential_matcher's window at all, however small it looks, and can
    only be a loop-closure retrieval.
    """
    with COLMAPDatabase(database_path) as db:
        images = db.get_images()
        image_id_to_frame_id = {
            img["image_id"]: _frame_id_from_image_name(img["name"]) for img in images
        }
        # `--SequentialMatching.overlap` counts neighbors by ORDINAL
        # POSITION in the registered keyframe sequence, not raw frame-id
        # distance -- keyframe selection (max_frame_gap) skips frames
        # non-uniformly, so two adjacent keyframes can have a large raw
        # frame_id gap while still being well inside the sequential
        # matching window. Rank images by frame_id (their actual capture
        # order) and use RANK gap for the sequential/loop split, matching
        # what sequential_matcher itself actually did.
        ordered_image_ids = [
            img_id for img_id, _ in sorted(image_id_to_frame_id.items(), key=lambda kv: kv[1])
        ]
        image_id_to_rank = {img_id: rank for rank, img_id in enumerate(ordered_image_ids)}

        # COLMAP's quadratic window only matches at EXACT discrete offsets
        # (overlap*2, overlap*4, overlap*8, ...) from each image, not a
        # continuous range up to that reach -- a rank gap of, say, 250
        # with overlap=20 is not explainable by either the linear window
        # (1-20) or any single quadratic step (40, 80, 160, 320, ...), so
        # it must be a loop-closure retrieval. Build the small discrete
        # set of quadratic offsets rather than a coarse upper bound, which
        # would let real loop-closure pairs slip through misclassified as
        # "reachable".
        quadratic_rank_gaps: set[int] = set()
        if quadratic_overlap:
            total_images = len(ordered_image_ids)
            step = sequential_overlap
            while step * 2 < total_images:
                step *= 2
                quadratic_rank_gaps.add(step)

        match_counts = db.get_match_counts()
        verified_pairs = db.get_verified_pairs()

        n_sequential = 0
        n_loop = 0
        to_drop: list[int] = []
        dropped_info: list[dict[str, Any]] = []

        for pair in verified_pairs:
            pair_id = pair["pair_id"]
            num_inliers = pair["num_inliers"]
            image_id1, image_id2 = COLMAPDatabase.pair_id_to_image_ids(pair_id)

            frame_id1 = image_id_to_frame_id.get(image_id1)
            frame_id2 = image_id_to_frame_id.get(image_id2)
            rank1 = image_id_to_rank.get(image_id1)
            rank2 = image_id_to_rank.get(image_id2)
            if frame_id1 is None or frame_id2 is None or rank1 is None or rank2 is None:
                continue

            rank_gap = abs(rank2 - rank1)
            if rank_gap <= sequential_overlap or rank_gap in quadratic_rank_gaps:
                n_sequential += 1
                continue

            n_loop += 1
            num_matches = match_counts.get(pair_id, num_inliers)
            inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0

            if num_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
                to_drop.append(pair_id)
                dropped_info.append({
                    "pair_id": pair_id,
                    "frame_id1": frame_id1,
                    "frame_id2": frame_id2,
                    "rank_gap": rank_gap,
                    "num_inliers": num_inliers,
                    "num_matches": num_matches,
                    "inlier_ratio": round(inlier_ratio, 3),
                })

        if to_drop:
            db.delete_pairs(to_drop)
            logger.warning(
                f"loop_closure_filter: dropped {len(to_drop)}/{n_loop} loop-closure "
                f"pair(s) below threshold (min_inliers={min_inliers}, "
                f"min_inlier_ratio={min_inlier_ratio}) -- likely false-positive "
                f"vocab-tree matches on repetitive/low-texture content. "
                f"{n_sequential} sequential-neighbor pair(s) left untouched."
            )

        return LoopClosureFilterResult(
            action_taken=bool(to_drop),
            reason=(
                f"dropped {len(to_drop)} loop-closure pair(s) below threshold"
                if to_drop
                else "no loop-closure pair fell below threshold"
            ),
            n_pairs_checked=len(verified_pairs),
            n_loop_pairs=n_loop,
            n_sequential_pairs=n_sequential,
            n_dropped=len(to_drop),
            dropped_pairs=dropped_info,
        )
