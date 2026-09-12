"""Depth-to-color pixel registration.

Some RGB-D drivers publish a "raw" depth stream that is NOT pixel-aligned
with the color image -- the depth sensor and color sensor are physically
offset by a small stereo baseline, so a 3D point visible in both cameras
projects to different pixel coordinates in each. Every downstream consumer
of `workspace/depth/*.png` in this codebase (scaling/correspondences.py's
`_sample_depth_at_pixels`, optimization/depth_ba.py's
`build_ba_observations`) assumes depth and color share a pixel grid. This
module performs the actual reprojection so that assumption becomes true.

Verified concretely on the kitchen1 bag (Orbbec Femto): `/camera/depth/
camera_info` reports `frame_id="camera_color_optical_frame"` with K
identical to color's -- almost certainly a driver metadata quirk for the
"raw" depth stream, not a real registration -- while `/tf_static` carries a
real ~3.2cm physical baseline between `camera_depth_frame` and
`camera_color_frame`. An independent edge-alignment QC tool found a
depth-dependent-parallax pixel shift (never a single constant offset)
ranging (0,-2) to (6,-6) across sampled frames, confirming the raw stream
is not actually registered. See `configs/orbbec_kitchen1_registered.yaml`
and `ingest/tf_static.py` for how the extrinsic is recovered.
"""

from typing import Any

import numpy as np

from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def register_depth_to_color(
    depth_m: np.ndarray,
    depth_K: np.ndarray,
    color_K: np.ndarray,
    color_width: int,
    color_height: int,
    T_color_from_depth: np.ndarray,
) -> np.ndarray:
    """Warp a metric depth image into the color camera's pixel grid.

    For every depth pixel with a valid (> 0) reading: unproject to a 3D
    point in the depth optical frame using `depth_K`, transform it into the
    color optical frame via `T_color_from_depth` (4x4, maps depth-frame
    points into the color frame), then project into the color pixel grid
    with `color_K`. When multiple source pixels land on the same target
    pixel, the nearest (smallest transformed Z -- occlusion-correct) one
    wins. Points that fall outside the color image bounds, or land behind
    the color camera, are dropped.

    Vectorized (no per-pixel Python loop): unprojection/transform/
    projection are done as batched matrix ops over all valid pixels at
    once, and the "nearest wins" z-buffer is implemented by sorting
    candidates far-to-near and relying on plain numpy fancy-index
    assignment overwriting earlier (farther) writes with later (nearer)
    ones for duplicate target indices.

    Args:
        depth_m: (H, W) metric depth in meters, 0 = invalid/no reading.
        depth_K: 3x3 depth camera intrinsic matrix.
        color_K: 3x3 color camera intrinsic matrix.
        color_width, color_height: target (color) image dimensions.
        T_color_from_depth: 4x4 homogeneous transform, depth frame -> color
            frame.

    Returns:
        (color_height, color_width) float64 array of metric depth in the
        color camera's optical frame and pixel grid, 0 where no valid
        source point projected there.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    output = np.zeros((color_height, color_width), dtype=np.float64)

    valid_v, valid_u = np.nonzero(depth_m > 0)
    if valid_v.size == 0:
        return output

    z = depth_m[valid_v, valid_u]

    fx_d, fy_d = depth_K[0, 0], depth_K[1, 1]
    cx_d, cy_d = depth_K[0, 2], depth_K[1, 2]

    x = (valid_u.astype(np.float64) - cx_d) / fx_d * z
    y = (valid_v.astype(np.float64) - cy_d) / fy_d * z

    pts_depth = np.stack([x, y, z, np.ones_like(z)], axis=0)  # (4, N)
    pts_color = T_color_from_depth @ pts_depth  # (4, N)
    xc, yc, zc = pts_color[0], pts_color[1], pts_color[2]

    in_front = zc > 0
    xc, yc, zc = xc[in_front], yc[in_front], zc[in_front]
    if zc.size == 0:
        return output

    fx_c, fy_c = color_K[0, 0], color_K[1, 1]
    cx_c, cy_c = color_K[0, 2], color_K[1, 2]

    u_proj = fx_c * xc / zc + cx_c
    v_proj = fy_c * yc / zc + cy_c

    u_int = np.round(u_proj).astype(np.int64)
    v_int = np.round(v_proj).astype(np.int64)

    in_bounds = (
        (u_int >= 0) & (u_int < color_width) & (v_int >= 0) & (v_int < color_height)
    )
    u_int, v_int, zc = u_int[in_bounds], v_int[in_bounds], zc[in_bounds]
    if zc.size == 0:
        return output

    # Farthest first, so that when we assign in array order, nearer points
    # (smaller z, later in this order) overwrite farther ones sharing the
    # same target pixel -- a simple vectorized z-buffer.
    order = np.argsort(-zc)
    flat_idx = v_int[order] * color_width + u_int[order]
    output_flat = output.reshape(-1)
    output_flat[flat_idx] = zc[order]

    return output


def resolve_registration_transform(
    depth_config: dict[str, Any],
    color_camera_info: dict[str, Any] | None,
    static_transforms: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray | None, str]:
    """Resolve the depth->color transform for registration, or explain why not.

    Returns ``(T_color_from_depth, reason)``. `T_color_from_depth` is
    ``None`` when registration cannot proceed (config disabled, frames
    missing from the TF graph, etc.) -- callers must treat that as a
    no-op fallback, not an error.
    """
    from colmap_rgbd_gt.ingest.tf_static import find_transform_chain

    reg_config = (depth_config or {}).get("registration", {}) or {}
    if not reg_config.get("enabled", False):
        return None, "disabled"

    if not color_camera_info:
        return None, "no color camera_info available"

    depth_frame_id = reg_config.get("depth_frame_id") or "camera_depth_optical_frame"
    color_frame_id = reg_config.get("color_frame_id") or "camera_color_optical_frame"

    if not static_transforms:
        return None, "no /tf_static transforms available"

    T_color_from_depth = find_transform_chain(static_transforms, depth_frame_id, color_frame_id)
    if T_color_from_depth is None:
        return (
            None,
            f"no TF chain found from {depth_frame_id!r} to {color_frame_id!r}",
        )

    return T_color_from_depth, "ok"
