"""Workspace schema for canonical dataset format."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkspaceLayout:
    manifest: Path
    rgb: Path
    depth: Path
    camera: Path
    timestamps: Path
    colmap: Path
    outputs: Path

    @classmethod
    def from_root(cls, root: Path) -> "WorkspaceLayout":
        root = Path(root)
        return cls(
            manifest=root / "manifest.json",
            rgb=root / "rgb",
            depth=root / "depth",
            camera=root / "camera",
            timestamps=root / "timestamps",
            colmap=root / "colmap",
            outputs=root / "outputs",
        )


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.layout = WorkspaceLayout.from_root(self.root)

    def create(self) -> "Workspace":
        self.root.mkdir(parents=True, exist_ok=True)
        for path in [
            self.layout.rgb,
            self.layout.depth,
            self.layout.camera,
            self.layout.timestamps,
            self.layout.colmap,
            self.layout.outputs,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self

    def validate(self) -> bool:
        required = [self.layout.manifest, self.layout.rgb, self.layout.depth]
        return all(p.exists() for p in required)

    def get_rgb_path(self, frame_id: int) -> Path:
        return self.layout.rgb / f"{frame_id:06d}.png"

    def get_depth_path(self, frame_id: int) -> Path:
        return self.layout.depth / f"{frame_id:06d}.png"

    def get_manifest_path(self) -> Path:
        return self.layout.manifest

    def exists(self) -> bool:
        return self.root.exists() and self.validate()


@dataclass
class FrameInfo:
    frame_id: int
    timestamp_ns: int
    rgb_path: str
    depth_path: str | None
    camera_info_available: bool
