"""Pipelines module initialization."""

from colmap_rgbd_gt.pipelines.extract_only import extract_pipeline
from colmap_rgbd_gt.pipelines.colmap_only import colmap_pipeline
from colmap_rgbd_gt.pipelines.scale_only import scale_pipeline
from colmap_rgbd_gt.pipelines.full_pipeline import full_pipeline

__all__ = [
    "extract_pipeline",
    "colmap_pipeline",
    "scale_pipeline",
    "full_pipeline",
]
