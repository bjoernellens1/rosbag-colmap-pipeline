"""Static TF tree reading and chain composition.

Used to recover the real physical depth-sensor -> color-sensor extrinsic
from a bag's `/tf_static` topic, independent of (and not trusting) whatever
`frame_id` a driver's depth `camera_info` message happens to claim -- see
`depth_registration.py` for why that field can be mislabeled.
"""

from collections import deque
from typing import Any, Iterable

import numpy as np

from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)

TF_STATIC_TOPIC = "/tf_static"


def _quat_xyzw_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix. Normalizes defensively."""
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm == 0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> quaternion (x, y, z, w). Inverse of the
    rotation half of `transform_to_matrix`; used for manifest provenance
    (recording a resolved transform in the same convention it was read in)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def transform_to_matrix(
    translation: Iterable[float], rotation_xyzw: Iterable[float]
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a translation and quaternion.

    Follows the `parent_T_child` convention used throughout this module:
    ``p_parent_homogeneous = T @ p_child_homogeneous``.
    """
    tx, ty, tz = translation
    T = np.eye(4)
    T[:3, :3] = _quat_xyzw_to_rotmat(*rotation_xyzw)
    T[:3, 3] = [tx, ty, tz]
    return T


def read_static_transforms(
    reader: Any, topic: str = TF_STATIC_TOPIC
) -> dict[tuple[str, str], np.ndarray]:
    """Read all `/tf_static` messages and return every declared edge.

    Returns a dict keyed by ``(parent_frame, child_frame)`` (i.e.
    ``TransformStamped.header.frame_id`` -> ``.child_frame_id``) mapping to
    the 4x4 ``parent_T_child`` matrix, such that
    ``p_parent_homogeneous = T @ p_child_homogeneous``. A bag with no
    `/tf_static` messages (or the topic missing entirely) yields an empty
    dict -- callers must treat that as "no static transforms available",
    not an error.
    """
    transforms: dict[tuple[str, str], np.ndarray] = {}
    try:
        messages = reader.get_messages(topic)
    except Exception as e:
        logger.warning(f"Could not read {topic!r}: {e}")
        return transforms

    for _timestamp, msg in messages:
        tf_list = getattr(msg, "transforms", None)
        if not tf_list:
            continue
        for tf in tf_list:
            try:
                parent = tf.header.frame_id
                child = tf.child_frame_id
                t = tf.transform.translation
                r = tf.transform.rotation
                T = transform_to_matrix((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))
            except AttributeError as e:
                logger.warning(f"Skipping malformed TransformStamped in {topic!r}: {e}")
                continue
            transforms[(parent, child)] = T

    return transforms


def find_transform_chain(
    transforms: dict[tuple[str, str], np.ndarray],
    from_frame: str,
    to_frame: str,
) -> np.ndarray | None:
    """Compose a transform chain across the TF graph, treating every edge as
    invertible (undirected traversal), from `from_frame` to `to_frame`.

    Returns a 4x4 matrix ``T`` such that
    ``p_to_frame_homogeneous = T @ p_from_frame_homogeneous``, or ``None``
    if no path connects the two frames (or `from_frame` never appears in
    `transforms` at all).
    """
    if from_frame == to_frame:
        return np.eye(4)

    # adjacency[node] = list of (neighbor, T_neighbor_from_node), where
    # T_neighbor_from_node maps a point expressed in `node`'s frame into
    # `neighbor`'s frame.
    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}
    for (parent, child), T_parent_child in transforms.items():
        # T_parent_child maps child-frame points into parent frame.
        adjacency.setdefault(child, []).append((parent, T_parent_child))
        adjacency.setdefault(parent, []).append((child, np.linalg.inv(T_parent_child)))

    if from_frame not in adjacency:
        return None

    visited = {from_frame}
    queue: deque[tuple[str, np.ndarray]] = deque([(from_frame, np.eye(4))])
    while queue:
        node, T_node_from_from = queue.popleft()
        if node == to_frame:
            return T_node_from_from
        for neighbor, T_neighbor_from_node in adjacency.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, T_neighbor_from_node @ T_node_from_from))

    return None
