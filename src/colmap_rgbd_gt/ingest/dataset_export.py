"""Dataset export utilities."""

from pathlib import Path
from typing import Any
import cv2
import numpy as np

from colmap_rgbd_gt.ingest.bag_reader import BagReader
from colmap_rgbd_gt.ingest.image_decode import decode_image
from colmap_rgbd_gt.ingest.depth_decode import decode_depth_image, filter_depth_range, depth_to_uint16
from colmap_rgbd_gt.ingest.camera_info import extract_camera_info
from colmap_rgbd_gt.ingest.keyframe_selection import KeyframeSelector
from colmap_rgbd_gt.utils.io import ensure_dir, save_json
from colmap_rgbd_gt.utils.time import ros_time_to_nanoseconds
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def export_rgb_frames(
    reader: BagReader,
    output_dir: Path,
    topic: str,
    config: dict[str, Any]
) -> list[tuple[int, str]]:
    output_dir = ensure_dir(output_dir)
    max_frames = config.get("max_frames")
    fmt = config.get("export_format", "png")

    keyframe_config = config.get("keyframe_selection", {})
    use_keyframes = keyframe_config.get("enabled", False)
    selector = (
        KeyframeSelector(
            min_match_ratio=keyframe_config.get("min_match_ratio", 0.5),
            max_frame_gap=keyframe_config.get("max_frame_gap", 30),
            orb_features=keyframe_config.get("orb_features", 500),
        )
        if use_keyframes
        else None
    )
    stride = config.get("min_frame_stride", 1)

    timestamps_paths = []
    frame_idx = 0

    for timestamp, msg in reader.get_messages(topic):
        if not use_keyframes and frame_idx % stride != 0:
            frame_idx += 1
            continue

        if max_frames:
            limit = len(timestamps_paths) if use_keyframes else frame_idx
            if limit >= max_frames:
                break

        try:
            img = decode_image(msg)

            if use_keyframes and not selector.should_select(img):
                frame_idx += 1
                continue

            fname = f"{frame_idx:06d}.{fmt}"
            fpath = output_dir / fname

            if fmt == "png":
                cv2.imwrite(str(fpath), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(str(fpath), img)

            ts_ns = timestamp
            timestamps_paths.append((ts_ns, fname))

            if frame_idx % 100 == 0:
                logger.debug(f"Exported RGB frame {frame_idx}")

        except Exception as e:
            logger.warning(f"Failed to decode frame {frame_idx}: {e}")

        frame_idx += 1

    logger.info(f"Exported {len(timestamps_paths)} RGB frames to {output_dir}")
    return timestamps_paths


def export_depth_frames(
    reader: BagReader,
    output_dir: Path,
    topic: str,
    config: dict[str, Any]
) -> list[tuple[int, str]]:
    output_dir = ensure_dir(output_dir)
    max_frames = config.get("max_frames")
    stride = config.get("min_frame_stride", 1)
    scale = config.get("unit_scale_to_meters", 0.001)
    min_m = config.get("min_depth_m", 0.2)
    max_m = config.get("max_depth_m", 8.0)

    timestamps_paths = []
    frame_idx = 0

    for timestamp, msg in reader.get_messages(topic):
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        if max_frames and frame_idx >= max_frames:
            break

        try:
            depth_m = decode_depth_image(msg, scale=scale)
            depth_m = filter_depth_range(depth_m, min_m=min_m, max_m=max_m)
            depth_uint16 = depth_to_uint16(depth_m, scale=1000.0)

            fname = f"{frame_idx:06d}.png"
            fpath = output_dir / fname
            cv2.imwrite(str(fpath), depth_uint16, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            ts_ns = timestamp
            timestamps_paths.append((ts_ns, fname))

        except Exception as e:
            logger.warning(f"Failed to decode depth frame {frame_idx}: {e}")

        frame_idx += 1

    logger.info(f"Exported {len(timestamps_paths)} depth frames to {output_dir}")
    return timestamps_paths


def export_camera_info(
    reader: BagReader,
    output_dir: Path,
    topic: str
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)

    for timestamp, msg in reader.get_messages(topic):
        info = extract_camera_info(msg)

        save_json(output_dir / "camera_info.json", info)
        save_json(output_dir / "intrinsics.json", {
            "fx": info["K"][0],
            "fy": info["K"][4],
            "cx": info["K"][2],
            "cy": info["K"][5],
            "width": info["width"],
            "height": info["height"],
        })
        save_json(output_dir / "distortion.json", {
            "model": info["distortion_model"],
            "coeffs": info["D"],
        })

        logger.info(f"Exported camera info from {topic}")
        return info

    logger.warning(f"No camera info messages found on topic {topic}")
    return {}


def export_timestamps_csv(
    timestamps: list[tuple[int, str]],
    output_path: Path,
    columns: list[str] = ["timestamp_ns", "filename"]
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for ts, fname in timestamps:
            f.write(f"{ts},{fname}\n")
