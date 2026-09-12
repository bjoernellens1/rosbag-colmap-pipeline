"""Manifest handling for workspace metadata."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json

from colmap_rgbd_gt.utils.io import load_json, save_json


@dataclass
class Manifest:
    version: str = "1.0"
    bag_path: str = ""
    bag_storage_id: str = ""
    topics: dict[str, str] = field(default_factory=dict)
    frame_count: int = 0
    time_range_ns: tuple[int, int] = (0, 0)
    camera_info: dict[str, Any] | None = None
    sync_settings: dict[str, Any] = field(default_factory=dict)
    extraction_settings: dict[str, Any] = field(default_factory=dict)
    # Depth-to-color registration provenance (see
    # ingest/depth_registration.py): None when there was no depth topic to
    # consider registration for at all; otherwise a dict with at least
    # "applied" (bool) and "reason" (str) -- "disabled" when
    # depth.registration.enabled is false (the default) -- plus the depth
    # camera_info used and the resolved transform, when applied, for
    # debuggability without re-reading the bag.
    depth_registration: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        data = load_json(path)
        return cls(
            version=data.get("version", "1.0"),
            bag_path=data.get("bag_path", ""),
            bag_storage_id=data.get("bag_storage_id", ""),
            topics=data.get("topics", {}),
            frame_count=data.get("frame_count", 0),
            time_range_ns=tuple(data.get("time_range_ns", [0, 0])),
            camera_info=data.get("camera_info"),
            sync_settings=data.get("sync_settings", {}),
            extraction_settings=data.get("extraction_settings", {}),
            depth_registration=data.get("depth_registration"),
        )

    def save(self, path: Path) -> None:
        data = {
            "version": self.version,
            "bag_path": self.bag_path,
            "bag_storage_id": self.bag_storage_id,
            "topics": self.topics,
            "frame_count": self.frame_count,
            "time_range_ns": list(self.time_range_ns),
            "camera_info": self.camera_info,
            "sync_settings": self.sync_settings,
            "extraction_settings": self.extraction_settings,
            "depth_registration": self.depth_registration,
        }
        save_json(path, data)

    @classmethod
    def from_extraction(
        cls,
        bag_path: str,
        topics: dict[str, str],
        frame_count: int,
        time_range: tuple[int, int],
        camera_info: dict[str, Any] | None,
        settings: dict[str, Any],
        storage_id: str = "ros1",
        depth_registration: dict[str, Any] | None = None,
    ) -> "Manifest":
        sync = settings.get("sync", {})
        extraction = settings.get("images", {})
        return cls(
            bag_path=bag_path,
            bag_storage_id=storage_id,
            topics=topics,
            frame_count=frame_count,
            time_range_ns=time_range,
            camera_info=camera_info,
            sync_settings=sync,
            extraction_settings=extraction,
            depth_registration=depth_registration,
        )
