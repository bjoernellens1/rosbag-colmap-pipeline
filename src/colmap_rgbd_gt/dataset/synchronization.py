"""Timestamp synchronization between RGB and depth frames."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class Association:
    frame_id: int
    rgb_timestamp_ns: int
    depth_timestamp_ns: int | None
    camera_info_timestamp_ns: int | None
    delta_rgb_depth_ns: int | None


def synchronize_rgb_depth(
    rgb_times: list[tuple[int, str]],
    depth_times: list[tuple[int, str]],
    max_dt_ns: int = 33_000_000,
) -> list[Association]:
    rgb_ts = np.array([t[0] for t in rgb_times], dtype=np.int64)
    depth_ts = np.array([t[0] for t in depth_times], dtype=np.int64)

    associations = []

    for frame_id, (rgb_time, rgb_name) in enumerate(rgb_times):
        deltas = np.abs(depth_ts - rgb_time)
        min_idx = np.argmin(deltas)
        min_delta = deltas[min_idx]

        if min_delta <= max_dt_ns:
            associations.append(Association(
                frame_id=frame_id,
                rgb_timestamp_ns=rgb_time,
                depth_timestamp_ns=int(depth_ts[min_idx]),
                camera_info_timestamp_ns=None,
                delta_rgb_depth_ns=int(min_delta),
            ))
        else:
            associations.append(Association(
                frame_id=frame_id,
                rgb_timestamp_ns=rgb_time,
                depth_timestamp_ns=None,
                camera_info_timestamp_ns=None,
                delta_rgb_depth_ns=None,
            ))

    return associations


def associate_camera_info(
    rgb_times: list[tuple[int, str]],
    info_times: list[int],
    max_dt_ns: int = 100_000_000,
) -> list[int | None]:
    rgb_ts = np.array([t[0] for t in rgb_times], dtype=np.int64)
    info_ts = np.array(info_times, dtype=np.int64)

    associations = []

    for rgb_time, _ in rgb_times:
        deltas = np.abs(info_ts - rgb_time)
        min_idx = np.argmin(deltas)
        min_delta = deltas[min_idx]

        if min_delta <= max_dt_ns:
            associations.append(int(info_ts[min_idx]))
        else:
            associations.append(None)

    return associations


def export_associations_csv(associations: list[Association], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("frame_id,rgb_timestamp_ns,depth_timestamp_ns,camera_info_timestamp_ns,delta_rgb_depth_ns\n")
        for assoc in associations:
            depth_ts = str(assoc.depth_timestamp_ns) if assoc.depth_timestamp_ns else ""
            info_ts = str(assoc.camera_info_timestamp_ns) if assoc.camera_info_timestamp_ns else ""
            delta = str(assoc.delta_rgb_depth_ns) if assoc.delta_rgb_depth_ns else ""
            f.write(f"{assoc.frame_id},{assoc.rgb_timestamp_ns},{depth_ts},{info_ts},{delta}\n")


def export_timestamps_csv(
    timestamps: list[tuple[int, str]],
    path: Path,
    columns: list[str] = ["timestamp_ns", "filename"]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(columns) + "\n")
        for ts, fname in timestamps:
            f.write(f"{ts},{fname}\n")
