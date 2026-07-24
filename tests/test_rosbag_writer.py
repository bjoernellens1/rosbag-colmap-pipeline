"""Tests for write_processed_bag (export/rosbag_writer.py)."""

import numpy as np
import pytest

rosbags = pytest.importorskip("rosbags")

from rosbags.rosbag2 import Reader as ReaderV2, Writer as WriterV2, StoragePlugin
from rosbags.typesys import get_typestore
from rosbags.typesys.stores import Stores

from colmap_rgbd_gt.export.rosbag_writer import write_processed_bag


def _write_synthetic_source_bag(path, n_messages=5):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    String = ts.types["std_msgs/msg/String"]
    with WriterV2(path, version=9, storage_plugin=StoragePlugin.MCAP) as w:
        conn = w.add_connection("/chat", String.__msgtype__, typestore=ts)
        for i in range(n_messages):
            msg = String(data=f"hello{i}")
            w.write(conn, i * 1000, ts.serialize_cdr(msg, String.__msgtype__))


def _write_rgb_csv(path, n_frames):
    with open(path, "w") as f:
        f.write("timestamp_ns,filename\n")
        for i in range(n_frames):
            f.write(f"{i * 1000},{i:06d}.png\n")


def test_write_processed_bag_copies_originals_and_adds_gt_topics(tmp_path):
    src_bag = tmp_path / "src.mcap"
    _write_synthetic_source_bag(src_bag, n_messages=5)

    rgb_csv = tmp_path / "rgb.csv"
    _write_rgb_csv(rgb_csv, n_frames=5)

    trajectory = [
        {"frame_id": i, "R": np.eye(3), "t": np.array([float(i), 0.0, 0.0])}
        for i in range(5)
    ]

    dst_bag = tmp_path / "src_processed.mcap"
    n_copied = write_processed_bag(src_bag, trajectory, rgb_csv, dst_bag)

    assert n_copied == 5
    assert dst_bag.exists()

    with ReaderV2(dst_bag) as r:
        topics = set(r.topics.keys())
        assert "/chat" in topics
        assert "/gt/pose" in topics
        assert "/gt/path" in topics

        chat_count = 0
        pose_count = 0
        path_count = 0
        for conn, ts_ns, rawdata in r.messages():
            if conn.topic == "/chat":
                chat_count += 1
            elif conn.topic == "/gt/pose":
                pose_count += 1
            elif conn.topic == "/gt/path":
                path_count += 1

        assert chat_count == 5
        assert pose_count == 5
        assert path_count == 1


def test_write_processed_bag_skips_frames_without_timestamp(tmp_path):
    src_bag = tmp_path / "src.mcap"
    _write_synthetic_source_bag(src_bag, n_messages=2)

    rgb_csv = tmp_path / "rgb.csv"
    _write_rgb_csv(rgb_csv, n_frames=2)  # only frame_id 0, 1 have timestamps

    trajectory = [
        {"frame_id": i, "R": np.eye(3), "t": np.array([float(i), 0.0, 0.0])}
        for i in range(5)  # 5 trajectory entries, only 2 have timestamps
    ]

    dst_bag = tmp_path / "dst.mcap"
    write_processed_bag(src_bag, trajectory, rgb_csv, dst_bag)

    with ReaderV2(dst_bag) as r:
        pose_count = sum(1 for conn, _, _ in r.messages() if conn.topic == "/gt/pose")
        assert pose_count == 2
