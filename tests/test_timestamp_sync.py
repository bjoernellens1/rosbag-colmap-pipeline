"""Test timestamp synchronization."""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_synchronize_rgb_depth_basic():
    from colmap_rgbd_gt.dataset.synchronization import synchronize_rgb_depth

    rgb_times = [(1000000000, "000000.png"), (1033000000, "000001.png"), (1066000000, "000002.png")]
    depth_times = [(1001000000, "000000.png"), (1034000000, "000001.png"), (1067000000, "000002.png")]

    associations = synchronize_rgb_depth(rgb_times, depth_times, max_dt_ns=50000000)

    assert len(associations) == 3
    assert associations[0].depth_timestamp_ns == 1001000000
    assert associations[1].depth_timestamp_ns == 1034000000


def test_synchronize_rgb_depth_frame_id_parsed_from_filename_not_position():
    # RGB export with min_frame_stride > 1 skips frames, so filenames are
    # sparse (e.g. every 15th) -- Association.frame_id must come from the
    # filename, not from this list's position, or downstream depth lookups
    # (keyed by the embedded frame index) silently misalign.
    from colmap_rgbd_gt.dataset.synchronization import synchronize_rgb_depth

    rgb_times = [
        (1000000000, "000000.png"),
        (1750000000, "000015.png"),
        (2500000000, "000030.png"),
    ]
    depth_times = [
        (1001000000, "000000.png"),
        (1751000000, "000001.png"),
        (2501000000, "000002.png"),
    ]

    associations = synchronize_rgb_depth(rgb_times, depth_times, max_dt_ns=50000000)

    assert [a.frame_id for a in associations] == [0, 15, 30]


def test_synchronize_rgb_depth_no_match():
    from colmap_rgbd_gt.dataset.synchronization import synchronize_rgb_depth

    rgb_times = [(1000000000, "000000.png")]
    depth_times = [(2000000000, "000000.png")]

    associations = synchronize_rgb_depth(rgb_times, depth_times, max_dt_ns=33000000)

    assert len(associations) == 1
    assert associations[0].depth_timestamp_ns is None


def test_association_csv_export(tmp_path):
    from colmap_rgbd_gt.dataset.synchronization import (
        Association,
        export_associations_csv
    )

    associations = [
        Association(0, 1000000000, 1001000000, None, 1000000),
        Association(1, 1033000000, 1034000000, None, 1000000),
    ]

    output_path = tmp_path / "associations.csv"
    export_associations_csv(associations, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "1000000000,1001000000" in content


def test_align_depth_to_rgb_frames_rekeys_by_frame_id(tmp_path):
    from colmap_rgbd_gt.dataset.synchronization import (
        Association,
        align_depth_to_rgb_frames,
    )

    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()

    # Depth's own independent numbering (3 depth msgs for 2 rgb frames,
    # different index space than rgb frame_id).
    depth_data = [
        (1001000000, "000000.png"),
        (1020000000, "000001.png"),
        (1034000000, "000002.png"),
    ]
    for _, fname in depth_data:
        (depth_dir / fname).write_bytes(fname.encode())

    # RGB frame 0 matches depth msg 0 (ts=1001000000, "000000.png");
    # RGB frame 1 matches depth msg 2 (ts=1034000000, "000002.png") --
    # i.e. NOT depth's own "000001.png", proving index-based lookup would
    # have been wrong.
    associations = [
        Association(0, 1000000000, 1001000000, None, 1000000),
        Association(1, 1033000000, 1034000000, None, 1000000),
    ]

    align_depth_to_rgb_frames(depth_dir, depth_data, associations)

    assert (depth_dir / "000000.png").read_bytes() == b"000000.png"
    assert (depth_dir / "000001.png").read_bytes() == b"000002.png"
    # Depth msg 1 ("000001.png", unmatched to any rgb frame) should not survive.
    assert not (depth_dir / "000002.png").exists()
    assert len(list(depth_dir.glob("*.png"))) == 2


def test_align_depth_to_rgb_frames_skips_unmatched(tmp_path):
    from colmap_rgbd_gt.dataset.synchronization import (
        Association,
        align_depth_to_rgb_frames,
    )

    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    depth_data = [(1001000000, "000000.png")]
    (depth_dir / "000000.png").write_bytes(b"data")

    associations = [Association(0, 1000000000, None, None, None)]

    align_depth_to_rgb_frames(depth_dir, depth_data, associations)

    assert list(depth_dir.glob("*.png")) == []
