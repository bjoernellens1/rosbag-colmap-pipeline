"""TUM RGB-D trajectory format export."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from colmap_rgbd_gt.utils.transforms import rotation_matrix_to_quaternion
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TUMEntry:
    timestamp: float
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


def pose_to_tum_entry(pose: dict[str, Any], scale: float = 1.0) -> TUMEntry:
    ts = pose.get("timestamp_ns", pose.get("frame_id", 0))
    if isinstance(ts, int):
        ts = ts / 1e9

    t = pose["t"] * scale
    R = pose["R"]

    q = rotation_matrix_to_quaternion(R)

    return TUMEntry(
        timestamp=ts,
        tx=float(t[0]),
        ty=float(t[1]),
        tz=float(t[2]),
        qx=float(q[0]),
        qy=float(q[1]),
        qz=float(q[2]),
        qw=float(q[3]),
    )


def export_tum(
    poses: list[dict[str, Any]],
    path: Path,
    timestamps: list[float] | None = None,
    scale: float = 1.0
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for i, pose in enumerate(poses):
        entry = pose_to_tum_entry(pose, scale)

        if timestamps and i < len(timestamps):
            entry = TUMEntry(
                timestamp=timestamps[i],
                tx=entry.tx,
                ty=entry.ty,
                tz=entry.tz,
                qx=entry.qx,
                qy=entry.qy,
                qz=entry.qz,
                qw=entry.qw,
            )

        entries.append(entry)

    entries.sort(key=lambda e: e.timestamp)

    with open(path, "w") as f:
        for entry in entries:
            f.write(
                f"{entry.timestamp:.6f} "
                f"{entry.tx:.6f} {entry.ty:.6f} {entry.tz:.6f} "
                f"{entry.qx:.6f} {entry.qy:.6f} {entry.qz:.6f} {entry.qw:.6f}\n"
            )

    logger.info(f"Exported {len(entries)} poses to {path}")


def export_trajectory_tum(
    trajectory: list[dict[str, Any]],
    path: Path,
    scale: float = 1.0
) -> None:
    export_tum(trajectory, path, scale=scale)
