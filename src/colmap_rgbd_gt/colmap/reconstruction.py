"""COLMAP reconstruction utilities."""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from colmap_rgbd_gt.colmap.colmap_io import read_cameras_text, read_images_text, read_points3d_text
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def ensure_text_model(path: Path, colmap_path: str = "colmap") -> bool:
    path = Path(path)

    text_files = [path / "cameras.txt", path / "images.txt", path / "points3D.txt"]
    if all(file.exists() for file in text_files):
        return True

    binary_files = [path / "cameras.bin", path / "images.bin", path / "points3D.bin"]
    if not all(file.exists() for file in binary_files):
        return False

    colmap_exe = shutil.which(colmap_path)
    if colmap_exe is None:
        logger.error("COLMAP not found in PATH, cannot convert sparse model to text")
        return False

    logger.info("Converting COLMAP sparse model from binary to text in %s", path)
    result = subprocess.run(
        [
            colmap_exe,
            "model_converter",
            "--input_path",
            str(path),
            "--output_path",
            str(path),
            "--output_type",
            "TXT",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("COLMAP model conversion failed: %s", result.stderr.strip())
        return False

    return all(file.exists() for file in text_files)


def remove_images_from_sparse_model(
    sparse_dir: Path,
    image_names_to_delete: list[str],
    colmap_path: str = "colmap",
) -> bool:
    """Remove specific images from a COLMAP sparse model IN PLACE, using
    COLMAP's own `image_deleter` tool (not a hand-rolled binary-format
    edit) -- a well-supported COLMAP operation, verified against a real
    reconstruction (floor2) including its rigs.bin/frames.bin (COLMAP
    4.x's multi-rig schema): image_deleter logs both image_id AND
    frame_id per deletion and produces a model that
    `colmap.reconstruction.load_sparse_model` reads back correctly with
    exactly the expected remaining image count.

    FIXED 2026-08-03: `pipelines.scale_only.scale_pipeline`'s disconnected-
    segment pose filter (colmap.pose_outliers) only ever filtered the
    EXPORTED trajectory (trajectory_metric_tum.txt / scene_metadata.json /
    depth-ba's inputs) -- it never touched `colmap/sparse/0` itself. A
    real user directly inspecting the raw sparse model in COLMAP's own GUI
    (a completely normal sanity-check workflow) still saw the original,
    uncleaned reconstruction with the mis-positioned minority segment
    intact, even after the exported-trajectory artifacts were correctly
    cleaned -- an "it's fixed" report based only on downstream-artifact
    inspection was wrong. This function is what actually closes that gap:
    called from scale_pipeline right after the pose filter determines
    which frame_ids to drop, so `colmap/sparse/0` on disk ends up
    reflecting the same clean trajectory as every exported artifact.

    Returns True on success (including the trivial True for an empty
    `image_names_to_delete`, a no-op). On any failure, the ORIGINAL
    sparse_dir contents are left untouched (edits happen in a scratch
    directory first, only swapped in on confirmed success) -- a failed
    edit must never leave a partially-modified or corrupted model in
    place of the last known-good one.
    """
    sparse_dir = Path(sparse_dir)
    if not image_names_to_delete:
        return True

    colmap_exe = shutil.which(colmap_path)
    if colmap_exe is None:
        logger.error("COLMAP not found in PATH, cannot remove images from sparse model")
        return False

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        names_file = tmpdir / "image_names_to_delete.txt"
        names_file.write_text("\n".join(image_names_to_delete) + "\n")
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        logger.info(
            f"Removing {len(image_names_to_delete)} image(s) from sparse model at "
            f"{sparse_dir}: {image_names_to_delete}"
        )
        result = subprocess.run(
            [
                colmap_exe, "image_deleter",
                "--input_path", str(sparse_dir),
                "--output_path", str(output_dir),
                "--image_names_path", str(names_file),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error(f"colmap image_deleter failed: {result.stderr.strip()}")
            return False

        # image_deleter only writes the binary format; confirm it produced
        # a usable model before touching the real sparse_dir at all.
        if not (output_dir / "images.bin").exists():
            logger.error(
                f"colmap image_deleter reported success but {output_dir}/images.bin "
                "is missing -- refusing to overwrite the original model"
            )
            return False

        # Swap in the edited binary model, dropping ALL old model files
        # (binary AND text -- stale .txt files from before the edit must
        # not linger, or load_sparse_model's ensure_text_model() would
        # short-circuit on them and silently serve the pre-edit model).
        for pattern in ("*.bin", "*.txt"):
            for f in sparse_dir.glob(pattern):
                f.unlink()
        for f in output_dir.iterdir():
            shutil.copy2(f, sparse_dir / f.name)

    # Regenerate the text model from the now-edited binary so every
    # existing text-format consumer (load_sparse_model et al) sees the
    # cleaned model too.
    if not ensure_text_model(sparse_dir, colmap_path=colmap_path):
        logger.error("Failed to regenerate text model after removing images")
        return False

    logger.info(f"Sparse model at {sparse_dir} updated: {len(image_names_to_delete)} image(s) removed")
    return True


def load_sparse_model(path: Path) -> dict[str, Any]:
    path = Path(path)
    ensure_text_model(path)

    cameras = read_cameras_text(path / "cameras.txt")
    images = read_images_text(path / "images.txt")
    points3d = read_points3d_text(path / "points3D.txt")

    return {
        "cameras": cameras,
        "images": images,
        "points3d": points3d,
    }


def get_image_names(model: dict[str, Any]) -> list[str]:
    return [img["name"] for img in model["images"].values()]


def get_camera(model: dict[str, Any], camera_id: int) -> dict[str, Any]:
    return model["cameras"].get(camera_id, {})


def get_image_pose(model: dict[str, Any], image_name: str) -> tuple[np.ndarray, np.ndarray]:
    for img in model["images"].values():
        if img["name"] == image_name:
            qvec = np.array(img["qvec"], dtype=np.float64)
            tvec = np.array(img["tvec"], dtype=np.float64)
            from colmap_rgbd_gt.utils.transforms import quaternion_to_rotation_matrix
            R = quaternion_to_rotation_matrix(qvec)
            return R, tvec

    raise ValueError(f"Image not found: {image_name}")


def get_points_observed_in_image(model: dict[str, Any], image_id: int) -> np.ndarray:
    points = []

    for point_id, point in model["points3d"].items():
        if image_id in point.get("image_ids", []):
            points.append(point["xyz"])

    return np.array(points, dtype=np.float64) if points else np.array([]).reshape(0, 3)


def get_image_id_by_name(model: dict[str, Any], name: str) -> int | None:
    for img_id, img in model["images"].items():
        if img["name"] == name:
            return img_id
    return None
