"""Timestamp synchronization between RGB and depth frames."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil
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
    """Associate each RGB frame with its nearest-in-time depth frame.

    `Association.frame_id` is parsed from the RGB filename (the original
    extraction-time frame index, e.g. "000615.png" -> 615) rather than
    taken from this loop's position -- RGB export can skip frames
    (`images.min_frame_stride`), so the position in `rgb_times` diverges
    from the embedded frame index whenever stride > 1. Callers (COLMAP
    image naming, `Workspace.get_depth_path`, `align_depth_to_rgb_frames`)
    all key by the embedded frame index, so this must match.
    """
    rgb_ts = np.array([t[0] for t in rgb_times], dtype=np.int64)
    depth_ts = np.array([t[0] for t in depth_times], dtype=np.int64)

    associations = []

    for rgb_time, rgb_name in rgb_times:
        frame_id = int(Path(rgb_name).stem)
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


def align_depth_to_rgb_frames(
    depth_dir: Path,
    depth_data: list[tuple[int, str]],
    associations: list[Association],
) -> None:
    """Re-key depth PNG files by RGB `frame_id`.

    RGB and depth come from separate topics with independent, generally
    non-identical frame counts/rates (e.g. a 2314-message RGB stream and a
    3045-message depth stream over the same bag). `export_depth_frames`
    numbers depth files by its own independent per-message counter, but
    downstream consumers (`Workspace.get_depth_path`, used by
    `estimate_global_scale` and the depth-BA pipeline) look up depth by the
    COLMAP/RGB `frame_id`. Without this re-keying step, that lookup would
    silently read whichever depth frame happens to share the same index as
    an RGB frame -- not the depth frame actually closest in time to it.

    After this runs, `depth_dir/{frame_id:06d}.png` is the depth frame
    time-matched (per `synchronize_rgb_depth`) to RGB frame `frame_id`;
    depth frames from the original independent numbering that don't
    survive re-keying (no matching RGB frame within tolerance) are removed.
    """
    depth_by_ts = {ts: fname for ts, fname in depth_data}

    staged_paths = []
    for assoc in associations:
        if assoc.depth_timestamp_ns is None:
            continue
        src_name = depth_by_ts.get(assoc.depth_timestamp_ns)
        if src_name is None:
            continue
        src = depth_dir / src_name
        if not src.exists():
            continue
        staged = depth_dir / f"{assoc.frame_id:06d}.png.aligned"
        shutil.copyfile(src, staged)
        staged_paths.append(staged)

    for f in depth_dir.glob("*.png"):
        f.unlink()

    for f in staged_paths:
        f.rename(f.with_suffix(""))


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
