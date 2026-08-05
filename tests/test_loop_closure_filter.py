"""Tests for colmap/loop_closure_filter.py's loop-closure pair pruning.

Root-cause case: kitchen1's repetitive, low-texture surfaces produced
false-positive vocab-tree loop-closure matches that fragmented the
reconstruction. This filter must drop weak loop-closure-only pairs while
never touching genuine sequential-window pairs -- including pairs only
reachable via `quadratic_overlap`'s exponentially-spaced matching, and
regardless of gaps in raw frame_id introduced by sparse keyframe
selection (sequence RANK, not raw frame_id, determines the window).
"""

import sqlite3

import pytest

from colmap_rgbd_gt.colmap.loop_closure_filter import filter_loop_closure_matches


def _make_database(tmp_path, images, pairs):
    """images: list of (image_id, frame_id) -- name encodes frame_id like
    real keyframe exports ("000123.png"). pairs: list of
    (image_id1, image_id2, num_inliers, num_matches)."""
    db_path = tmp_path / "database.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL, camera_id INTEGER)"
    )
    conn.execute(
        "CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)"
    )
    conn.execute(
        "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER, "
        "cols INTEGER, data BLOB, config INTEGER)"
    )
    for image_id, frame_id in images:
        conn.execute(
            "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, 1)",
            (image_id, f"{frame_id:06d}.png"),
        )

    max_num_images = 2147483647
    for image_id1, image_id2, num_inliers, num_matches in pairs:
        assert image_id1 < image_id2
        pair_id = image_id1 * max_num_images + image_id2
        conn.execute(
            "INSERT INTO matches (pair_id, rows, cols, data) VALUES (?, ?, 2, NULL)",
            (pair_id, num_matches),
        )
        conn.execute(
            "INSERT INTO two_view_geometries (pair_id, rows, cols, data, config) VALUES (?, ?, 2, NULL, 2)",
            (pair_id, num_inliers),
        )
    conn.commit()
    conn.close()
    return db_path


def test_sequential_pairs_never_touched_even_if_weak(tmp_path):
    # 5 consecutive images, overlap=2 -- all pairs within rank gap 2 are
    # "sequential" and must survive even with a very weak match.
    images = [(i, i * 10) for i in range(5)]  # sparse raw frame_ids: 0,10,20,30,40
    pairs = [(0, 1, 3, 20), (1, 2, 3, 20), (2, 3, 3, 20), (3, 4, 3, 20)]
    db_path = _make_database(tmp_path, images, pairs)

    result = filter_loop_closure_matches(
        db_path, sequential_overlap=2, quadratic_overlap=False,
        min_inliers=30, min_inlier_ratio=0.35,
    )

    assert result.n_sequential_pairs == 4
    assert result.n_loop_pairs == 0
    assert result.n_dropped == 0


def test_weak_loop_pair_dropped_strong_kept(tmp_path):
    # 10 images, overlap=2 -- pair (0, 9) has rank gap 9, well beyond the
    # sequential window, so it can only be a loop-closure retrieval.
    images = [(i, i) for i in range(10)]
    pairs = [
        (0, 9, 10, 50),   # weak: 10 inliers, ratio 0.2 -- should be dropped
        (1, 8, 40, 50),   # strong: 40 inliers, ratio 0.8 -- should be kept
        (0, 1, 3, 5),      # sequential neighbor, weak but never checked
    ]
    db_path = _make_database(tmp_path, images, pairs)

    result = filter_loop_closure_matches(
        db_path, sequential_overlap=2, quadratic_overlap=False,
        min_inliers=30, min_inlier_ratio=0.35,
    )

    assert result.n_sequential_pairs == 1
    assert result.n_loop_pairs == 2
    assert result.n_dropped == 1
    assert result.dropped_pairs[0]["frame_id1"] == 0
    assert result.dropped_pairs[0]["frame_id2"] == 9


def test_quadratic_overlap_window_not_misclassified_as_loop(tmp_path):
    # 200 images, overlap=10, quadratic_overlap=True -- COLMAP itself would
    # match e.g. rank gap 40 (10*2*2) as part of the quadratic window, not
    # loop detection. A weak match at that rank gap must NOT be dropped.
    images = [(i, i) for i in range(200)]
    pairs = [
        (0, 40, 5, 50),    # rank gap 40 = quadratic step (10*2*2) -- must survive
        (0, 100, 5, 50),   # rank gap 100 -- between steps 80 and 160, NOT exact -- must be dropped
        (0, 199, 5, 50),   # rank gap 199, beyond any quadratic step -- must be dropped
    ]
    db_path = _make_database(tmp_path, images, pairs)

    result = filter_loop_closure_matches(
        db_path, sequential_overlap=10, quadratic_overlap=True,
        min_inliers=30, min_inlier_ratio=0.35,
    )

    dropped_frame_pairs = {(d["frame_id1"], d["frame_id2"]) for d in result.dropped_pairs}
    assert (0, 40) not in dropped_frame_pairs
    assert (0, 100) in dropped_frame_pairs
    assert (0, 199) in dropped_frame_pairs


def test_sparse_keyframe_frame_id_gap_does_not_affect_rank_classification(tmp_path):
    # Mirrors kitchen1's real bug: keyframe selection skips raw frames
    # non-uniformly, so two RANK-adjacent keyframes can have a huge raw
    # frame_id gap. Classification must use rank, not frame_id, gap.
    images = [(0, 0), (1, 1200), (2, 1205)]  # ranks 0,1,2; frame_id gaps huge
    pairs = [(0, 1, 5, 50)]  # rank gap 1 (sequential), frame_id gap 1200

    db_path = _make_database(tmp_path, images, pairs)

    result = filter_loop_closure_matches(
        db_path, sequential_overlap=2, quadratic_overlap=False,
        min_inliers=30, min_inlier_ratio=0.35,
    )

    assert result.n_sequential_pairs == 1
    assert result.n_loop_pairs == 0
    assert result.n_dropped == 0
