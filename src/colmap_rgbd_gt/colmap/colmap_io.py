"""COLMAP text file I/O."""

from pathlib import Path
from typing import Any
import numpy as np
import struct


def read_cameras_text(path: Path) -> dict[int, dict[str, Any]]:
    cameras = {}

    if not path.exists():
        return cameras

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]

            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }

    return cameras


def read_images_text(path: Path) -> dict[int, dict[str, Any]]:
    images = {}

    if not path.exists():
        return images

    with open(path, "r") as f:
        pending_image_id = None
        for line in f:
            if pending_image_id is not None:
                # POINTS2D line: may be blank (no 2D points registered for
                # this image) -- must NOT be skipped by the blank-line check
                # below, or subsequent lines would misalign.
                parts = line.strip().split()
                xys = []
                point3d_ids = []
                for i in range(0, len(parts), 3):
                    x, y = float(parts[i]), float(parts[i + 1])
                    point3d_id = int(parts[i + 2])
                    xys.append([x, y])
                    point3d_ids.append(point3d_id)
                images[pending_image_id]["xys"] = np.array(xys, dtype=np.float64).reshape(-1, 2)
                images[pending_image_id]["point3d_ids"] = np.array(point3d_ids, dtype=np.int64)
                pending_image_id = None
                continue

            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]

            images[image_id] = {
                "image_id": image_id,
                "qvec": [qw, qx, qy, qz],
                "tvec": [tx, ty, tz],
                "camera_id": camera_id,
                "name": name,
                "xys": np.zeros((0, 2), dtype=np.float64),
                "point3d_ids": np.zeros((0,), dtype=np.int64),
            }
            pending_image_id = image_id

    return images


def read_points3d_text(path: Path) -> dict[int, dict[str, Any]]:
    points = {}

    if not path.exists():
        return points

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()
            point_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            error = float(parts[7])

            track = []
            i = 8
            while i < len(parts):
                image_id = int(parts[i])
                point2d_idx = int(parts[i + 1])
                track.append((image_id, point2d_idx))
                i += 2

            points[point_id] = {
                "point_id": point_id,
                "xyz": [x, y, z],
                "rgb": [r, g, b],
                "error": error,
                "image_ids": [t[0] for t in track],
                "point2d_idxs": [t[1] for t in track],
            }

    return points


def write_cameras_text(path: Path, cameras: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")

        for cam in sorted(cameras.values(), key=lambda c: c["camera_id"]):
            params_str = " ".join(str(p) for p in cam["params"])
            f.write(f"{cam['camera_id']} {cam['model']} {cam['width']} {cam['height']} {params_str}\n")


def write_images_text(path: Path, images: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for img in sorted(images.values(), key=lambda i: i["image_id"]):
            qvec = img["qvec"]
            tvec = img["tvec"]
            f.write(f"{img['image_id']} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} ")
            f.write(f"{tvec[0]} {tvec[1]} {tvec[2]} {img['camera_id']} {img['name']}\n")
            f.write("\n")


def write_points3d_text(path: Path, points: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        for point in sorted(points.values(), key=lambda p: p["point_id"]):
            x, y, z = point["xyz"]
            r, g, b = point["rgb"]
            error = point["error"]
            track = " ".join(
                f"{image_id} {point2d_idx}"
                for image_id, point2d_idx in zip(point["image_ids"], point["point2d_idxs"])
            )
            f.write(f"{point['point_id']} {x} {y} {z} {r} {g} {b} {error} {track}\n")


def read_model_binary(path: Path) -> dict[str, Any]:
    path = Path(path)

    cameras = {}
    images = {}
    points3d = {}

    def read_next_bytes(f, num_bytes, format_char_sequence):
        data = f.read(num_bytes)
        return struct.unpack(format_char_sequence, data)

    with open(path / "cameras.bin", "rb") as f:
        num_cameras = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(f, 24, "iiQQ")
            model_names = ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"]
            model = model_names[model_id] if model_id < len(model_names) else "UNKNOWN"
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": model,
                "width": width,
                "height": height,
                "params": [],
            }

    return {"cameras": cameras, "images": images, "points3d": points3d}
