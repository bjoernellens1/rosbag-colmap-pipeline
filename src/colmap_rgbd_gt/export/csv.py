"""CSV trajectory export."""

from pathlib import Path
from typing import Any

from colmap_rgbd_gt.utils.transforms import rotation_matrix_to_quaternion
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def export_csv(
    poses: list[dict[str, Any]],
    path: Path,
    scale: float = 1.0,
    include_rotation_matrix: bool = False,
    include_camera_center: bool = False
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "frame_id"]

    if include_rotation_matrix:
        header.extend([f"R{i}{j}" for i in range(1, 4) for j in range(1, 4)])

    if include_camera_center:
        header.extend(["cx", "cy", "cz"])

    rows = []
    for pose in poses:
        frame_id = pose.get("frame_id", 0)
        ts = pose.get("timestamp_ns", frame_id)

        t = pose["t"] * scale
        R = pose["R"]
        q = rotation_matrix_to_quaternion(R)

        row = [
            ts,
            t[0], t[1], t[2],
            q[0], q[1], q[2], q[3],
            frame_id
        ]

        if include_rotation_matrix:
            row.extend(R.flatten().tolist())

        if include_camera_center:
            c = -R.T @ t
            row.extend([c[0], c[1], c[2]])

        rows.append((ts, row))

    rows.sort(key=lambda x: x[0])

    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for _, row in rows:
            f.write(",".join(str(v) for v in row) + "\n")

    logger.info(f"Exported {len(rows)} poses to {path}")
