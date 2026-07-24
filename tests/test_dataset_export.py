"""Tests for the parallelized dataset export functions.

Regression coverage: export_depth_frames and export_rgb_frames were
refactored to process/write frames via a thread pool instead of fully
sequentially. These tests verify the parallel version produces the same
ordered, correct output as the original sequential version would.
"""

import numpy as np
import cv2

from colmap_rgbd_gt.ingest.dataset_export import export_depth_frames, export_rgb_frames


class _FakeMsg:
    def __init__(self, encoding, height, width, data):
        self.encoding = encoding
        self.height = height
        self.width = width
        self.data = data


class _FakeReader:
    def __init__(self, messages):
        self._messages = messages  # list of (timestamp, msg)

    def get_messages(self, topic):
        return iter(self._messages)


def _make_depth_msg(value: int, height=8, width=8):
    arr = np.full((height, width), value, dtype=np.uint16)
    return _FakeMsg("16uc1", height, width, arr.tobytes())


def test_export_depth_frames_preserves_order_under_parallel_processing(tmp_path):
    # 30 frames, distinct depth values -- if parallel writes raced or
    # results were mis-ordered, frame N's content wouldn't match frame N's
    # expected value.
    messages = [
        (1000 + i, _make_depth_msg(value=1000 + i))
        for i in range(30)
    ]
    reader = _FakeReader(messages)
    output_dir = tmp_path / "depth"

    result = export_depth_frames(
        reader, output_dir, "depth_topic",
        {"unit_scale_to_meters": 0.001, "min_depth_m": 0.0, "max_depth_m": 100.0, "num_workers": 4},
    )

    assert len(result) == 30
    # Result order matches ascending frame_idx (== ascending timestamp here).
    timestamps = [ts for ts, _ in result]
    assert timestamps == sorted(timestamps)

    for i, (ts, fname) in enumerate(result):
        assert fname == f"{i:06d}.png"
        img = cv2.imread(str(output_dir / fname), cv2.IMREAD_UNCHANGED)
        # depth value (1000+i) mm survives the mm-round-trip encoding.
        assert img[0, 0] == 1000 + i


def test_export_depth_frames_respects_stride(tmp_path):
    messages = [(i, _make_depth_msg(value=500)) for i in range(10)]
    reader = _FakeReader(messages)
    output_dir = tmp_path / "depth"

    result = export_depth_frames(
        reader, output_dir, "depth_topic",
        {"min_frame_stride": 3, "min_depth_m": 0.0, "max_depth_m": 10.0},
    )

    assert len(result) == 4  # frames 0, 3, 6, 9
    assert [fname for _, fname in result] == ["000000.png", "000003.png", "000006.png", "000009.png"]


def _make_rgb_msg(color_value: int, height=16, width=16):
    arr = np.full((height, width, 3), color_value, dtype=np.uint8)
    return _FakeMsg("bgr8", height, width, arr.tobytes())


def test_export_rgb_frames_writes_all_and_preserves_order(tmp_path):
    messages = [(100 + i, _make_rgb_msg(color_value=i % 256)) for i in range(20)]
    reader = _FakeReader(messages)
    output_dir = tmp_path / "rgb"

    result = export_rgb_frames(reader, output_dir, "rgb_topic", {"num_workers": 4})

    assert len(result) == 20
    timestamps = [ts for ts, _ in result]
    assert timestamps == sorted(timestamps)
    for i, (ts, fname) in enumerate(result):
        assert fname == f"{i:06d}.png"
        assert (output_dir / fname).exists()
