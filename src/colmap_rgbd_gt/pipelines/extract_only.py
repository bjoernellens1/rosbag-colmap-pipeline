"""Extraction pipeline."""

from pathlib import Path
from typing import Any

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.ingest.bag_reader import BagReader
from colmap_rgbd_gt.ingest.topic_discovery import select_best_topics
from colmap_rgbd_gt.ingest.dataset_export import (
    export_rgb_frames,
    export_depth_frames,
    export_camera_info,
    export_timestamps_csv,
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

        depth_data = []
        if depth_topic:
            logger.info("Extracting depth frames...")
            depth_data = export_depth_frames(reader, ws.layout.depth, depth_topic, depth_config)

        camera_info = {}
        if camera_info_topic:
            logger.info("Extracting camera info...")
            camera_info = export_camera_info(reader, ws.layout.camera, camera_info_topic)

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
        )
        manifest.save(ws.layout.manifest)

        logger.info(
            f"Extraction complete: {len(rgb_data)} RGB frames, {len(depth_data)} depth frames"
        )
        return True

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False
