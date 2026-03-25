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
