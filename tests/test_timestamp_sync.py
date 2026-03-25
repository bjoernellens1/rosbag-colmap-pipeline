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
