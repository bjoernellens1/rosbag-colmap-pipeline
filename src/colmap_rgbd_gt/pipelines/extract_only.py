"""Extraction pipeline."""

from pathlib import Path
from typing import Any

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.ingest.bag_reader import BagReader
from colmap_rgbd_gt.ingest.topic_discovery import select_best_topics
from colmap_rgbd_gt.ingest.tf_static import read_static_transforms, rotmat_to_quat_xyzw
from colmap_rgbd_gt.ingest.dataset_export import (
    export_rgb_frames,
    export_depth_frames,
    export_camera_info,
    export_timestamps_csv,
    read_camera_info,
    resolve_depth_registration,
)
from colmap_rgbd_gt.dataset.schema import Workspace
from colmap_rgbd_gt.dataset.manifest import Manifest
from colmap_rgbd_gt.dataset.synchronization import (
    synchronize_rgb_depth,
    align_depth_to_rgb_frames,
    export_associations_csv,
)

logger = get_logger(__name__)


def extract_pipeline(bag_path: Path, workspace: Path, config: dict[str, Any]) -> bool:
    bag_path = Path(bag_path)
    workspace = Path(workspace)

    if not bag_path.exists():
        logger.error(f"Bag file not found: {bag_path}")
        return False

    ws = Workspace(workspace)
    ws.create()

    try:
        reader = BagReader(bag_path)

        topics = select_best_topics(reader, config)
        logger.info(f"Selected topics: {topics}")

        rgb_topic = topics.get("rgb")
        depth_topic = topics.get("depth")
        camera_info_topic = topics.get("camera_info")
        depth_camera_info_topic = topics.get("depth_camera_info")

        if not rgb_topic:
            logger.error("No RGB topic found")
            reader.close()
            return False

        images_config = config.get("images", {})
        depth_config = config.get("depth", {})
        storage_id = reader.storage_id

        # RGB, depth, and camera-info reads must stay sequential against a
        # single BagReader: concurrent BagReader instances on the same file
        # were tried (each topic read on its own thread) but caused
        # intermittent "decompression error: Data corruption detected" on
        # a real (chunk-compressed) mcap file under concurrent load --
        # opening/listing topics concurrently is safe, but concurrent
        # message-chunk decompression is not. The actual CPU-bound
        # per-frame work (decode/encode/write) is still parallelized
        # within each of these calls via a thread pool, which doesn't
        # touch the bag reader concurrently.
        logger.info("Extracting RGB frames...")
        rgb_data = export_rgb_frames(reader, ws.layout.rgb, rgb_topic, images_config)

        camera_info = {}
        if camera_info_topic:
            logger.info("Extracting camera info...")
            camera_info = export_camera_info(reader, ws.layout.camera, camera_info_topic)

        # Resolved once, before the per-frame depth loop -- not per-frame.
        # `depth_registration` stays None (a provable no-op, see
        # `resolve_depth_registration`) unless `depth.registration.enabled`
        # is explicitly set, so scenes not opting in are unaffected.
        depth_registration = None
        registration_provenance: dict[str, Any] | None = None
        reg_cfg = (depth_config or {}).get("registration", {}) or {}
        if depth_topic:
            registration_provenance = {"applied": False, "reason": "disabled"}
        if depth_topic and reg_cfg.get("enabled", False):
            depth_camera_info = None
            if depth_camera_info_topic:
                logger.info("Reading depth camera info for registration...")
                depth_camera_info = read_camera_info(reader, depth_camera_info_topic)
            else:
                logger.warning(
                    "Depth registration enabled but no depth_camera_info topic "
                    "resolved; falling back to unregistered depth output."
                )

            logger.info("Reading /tf_static for depth->color extrinsic...")
            static_transforms = read_static_transforms(reader)

            depth_registration, reg_reason = resolve_depth_registration(
                depth_config, depth_camera_info, camera_info, static_transforms
            )
            if depth_registration is None:
                logger.warning(
                    f"Depth registration enabled but not applied ({reg_reason}); "
                    "falling back to unregistered depth output."
                )
            T_resolved = (
                depth_registration["T_color_from_depth"]
                if depth_registration is not None
                else None
            )
            registration_provenance = {
                "applied": depth_registration is not None,
                "reason": reg_reason,
                "depth_camera_info": depth_camera_info,
                "transform_translation_m": (
                    T_resolved[:3, 3].tolist() if T_resolved is not None else None
                ),
                "transform_rotation_quat_xyzw": (
                    list(rotmat_to_quat_xyzw(T_resolved[:3, :3]))
                    if T_resolved is not None
                    else None
                ),
            }

        depth_data = []
        if depth_topic:
            logger.info("Extracting depth frames...")
            depth_data = export_depth_frames(
                reader, ws.layout.depth, depth_topic, depth_config,
                registration=depth_registration,
            )

        reader.close()

        logger.info("Synchronizing timestamps...")
        sync_config = config.get("sync", {})
        max_dt = sync_config.get("max_rgb_depth_dt_sec", 0.03)
        max_dt_ns = int(max_dt * 1e9)

        associations = synchronize_rgb_depth(rgb_data, depth_data, max_dt_ns=max_dt_ns)

        export_associations_csv(associations, ws.layout.timestamps / "associations.csv")
        export_timestamps_csv(rgb_data, ws.layout.timestamps / "rgb.csv")
        if depth_data:
            export_timestamps_csv(depth_data, ws.layout.timestamps / "depth.csv")
            logger.info("Re-keying depth frames to RGB frame_id...")
            align_depth_to_rgb_frames(ws.layout.depth, depth_data, associations)

        start_ns, end_ns = (0, 0)
        if rgb_data:
            start_ns = rgb_data[0][0]
            end_ns = rgb_data[-1][0]

        manifest = Manifest.from_extraction(
            bag_path=str(bag_path),
            topics=topics,
            frame_count=len(rgb_data),
            time_range=(start_ns, end_ns),
            camera_info=camera_info,
            settings=config,
            storage_id=storage_id,
            depth_registration=registration_provenance,
        )
        manifest.save(ws.layout.manifest)

        logger.info(
            f"Extraction complete: {len(rgb_data)} RGB frames, {len(depth_data)} depth frames"
        )
        return True

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False
