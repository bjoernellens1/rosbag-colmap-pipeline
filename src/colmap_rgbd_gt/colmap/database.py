"""COLMAP database handling."""

import sqlite3
from pathlib import Path
from typing import Any
import numpy as np

from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


class COLMAPDatabase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._conn = None

    def connect(self) -> None:
        self._conn = sqlite3.connect(str(self.path))

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def get_images(self) -> list[dict[str, Any]]:
        if not self._conn:
            raise RuntimeError("Database not connected")

        cursor = self._conn.cursor()
        cursor.execute("SELECT image_id, name, camera_id FROM images")
        rows = cursor.fetchall()

        return [
            {"image_id": row[0], "name": row[1], "camera_id": row[2]}
            for row in rows
        ]

    def get_keypoints(self, image_id: int) -> np.ndarray:
        if not self._conn:
            raise RuntimeError("Database not connected")

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
            (image_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return np.array([])

        rows, cols, data = row
        keypoints = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
        return keypoints

    def get_matches(self, image_id: int) -> np.ndarray:
        if not self._conn:
            raise RuntimeError("Database not connected")

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT rows, cols, data FROM matches WHERE image_id1 = ?",
            (image_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return np.array([])

        rows, cols, data = row
        matches = np.frombuffer(data, dtype=np.uint32).reshape(rows, cols)
        return matches

    # COLMAP's own pair-id encoding (see COLMAP's database.py reference
    # script / src/colmap/scene/database.cc `ImagePairToPairId`):
    # pair_id = image_id1 * kMaxNumImages + image_id2, image_id1 < image_id2.
    _MAX_NUM_IMAGES = 2147483647

    @classmethod
    def pair_id_to_image_ids(cls, pair_id: int) -> tuple[int, int]:
        image_id2 = pair_id % cls._MAX_NUM_IMAGES
        image_id1 = (pair_id - image_id2) // cls._MAX_NUM_IMAGES
        return image_id1, image_id2

    def get_match_counts(self) -> dict[int, int]:
        """pair_id -> number of putative (pre-verification) matches, from
        the `matches` table (raw output of feature matching, before
        `two_view_geometries` geometric verification)."""
        if not self._conn:
            raise RuntimeError("Database not connected")

        cursor = self._conn.cursor()
        cursor.execute("SELECT pair_id, rows FROM matches")
        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_verified_pairs(self) -> list[dict[str, Any]]:
        """pair_id + inlier count for every geometrically-verified pair in
        `two_view_geometries` (COLMAP only keeps a row here once a pair
        passes verification, so `rows` is the inlier count for that pair)."""
        if not self._conn:
            raise RuntimeError("Database not connected")

        cursor = self._conn.cursor()
        cursor.execute("SELECT pair_id, rows FROM two_view_geometries")
        return [{"pair_id": row[0], "num_inliers": row[1]} for row in cursor.fetchall()]

    def delete_pairs(self, pair_ids: list[int]) -> None:
        """Remove pairs from both `matches` and `two_view_geometries` --
        used to prune false-positive loop-closure matches before mapping
        (see colmap/loop_closure_filter.py). Sequential-neighbor pairs are
        never passed here."""
        if not self._conn:
            raise RuntimeError("Database not connected")
        if not pair_ids:
            return

        cursor = self._conn.cursor()
        placeholders = ",".join("?" * len(pair_ids))
        cursor.execute(f"DELETE FROM matches WHERE pair_id IN ({placeholders})", pair_ids)
        cursor.execute(f"DELETE FROM two_view_geometries WHERE pair_id IN ({placeholders})", pair_ids)
        self._conn.commit()

    def get_camera(self, camera_id: int) -> dict[str, Any]:
        if not self._conn:
            raise RuntimeError("Database not connected")

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT camera_id, model, width, height, params FROM cameras WHERE camera_id = ?",
            (camera_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return {}

        params = np.frombuffer(row[4], dtype=np.float64)
        return {
            "camera_id": row[0],
            "model": row[1],
            "width": row[2],
            "height": row[3],
            "params": params,
        }
