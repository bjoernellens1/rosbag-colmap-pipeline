"""Tests for depth-to-color pixel registration.

Covers the standalone warp (`register_depth_to_color`), the TF-chain
composition helper (`find_transform_chain`), and the extraction wiring
(`export_depth_frames` with a resolved registration) -- including the
critical regression guard that the default-disabled path is a provable
no-op, byte-identical to the pre-registration behavior.
"""

import numpy as np
import cv2

from colmap_rgbd_gt.ingest.depth_registration import (
    register_depth_to_color,
    resolve_registration_transform,
)
from colmap_rgbd_gt.ingest.tf_static import find_transform_chain, transform_to_matrix
from colmap_rgbd_gt.ingest.dataset_export import export_depth_frames, resolve_depth_registration


def _identity_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


class TestRegisterDepthToColor:
    def test_identity_transform_is_near_noop(self):
        # Flat plane at 2m, same intrinsics/resolution on both "cameras",
        # identity transform -- registration must not corrupt an
        # already-aligned depth image (allowing for pixel-grid resampling
        # rounding at non-integer projections, which shouldn't occur here
        # since depth_K == color_K and T is identity: every source pixel
        # projects back to its own integer coordinate exactly).
        h, w = 64, 64
        depth = np.full((h, w), 2.0, dtype=np.float64)
        K = _identity_K()
        T_identity = np.eye(4)

        registered = register_depth_to_color(depth, K, K, w, h, T_identity)

        np.testing.assert_allclose(registered, depth)

    def test_translation_matches_analytic_expectation_at_principal_point(self):
        # A frontal plane at depth Z0, depth camera and color camera share
        # orientation and intrinsics but color is translated by `baseline`
        # along +X relative to depth (i.e. color sits to the depth
        # camera's right). A 3D point that appears at the depth camera's
        # principal point (its optical axis) is, in the color camera's
        # frame, shifted by -baseline in X -- so it no longer lines up
        # with the color camera's principal point, but the KNOWN 3D point
        # that DOES land on the color camera's principal point (directly
        # in front of the color sensor) has the same Z0 depth (frontal
        # plane), letting us assert the registered depth at the color
        # image's principal point pixel equals Z0.
        h, w = 200, 200
        fx = fy = 300.0
        cx, cy = 100.0, 100.0
        K = _identity_K(fx, fy, cx, cy)
        Z0 = 3.0

        depth = np.full((h, w), Z0, dtype=np.float64)

        baseline = 0.05  # 5cm, color camera is +X relative to depth camera
        T_color_from_depth = np.eye(4)
        T_color_from_depth[0, 3] = -baseline  # p_color = p_depth - baseline*X

        registered = register_depth_to_color(depth, K, K, w, h, T_color_from_depth)

        # The color camera's principal point pixel (cx, cy) should see
        # depth Z0 (the plane is frontal and flat, so every visible point
        # on it is at Z0 in both frames -- transforming by a pure X
        # translation changes X, not Z).
        assert registered[int(cy), int(cx)] == Z0

    def test_out_of_frame_point_is_dropped(self):
        # A single valid pixel that, after a large translation, projects
        # outside the color image bounds must be dropped (left as 0/invalid
        # in the output), not wrapped/clamped/left stale.
        h, w = 32, 32
        K = _identity_K(fx=500.0, fy=500.0, cx=16.0, cy=16.0)
        depth = np.zeros((h, w), dtype=np.float64)
        depth[16, 16] = 1.0  # at the principal point

        # Huge lateral shift pushes the reprojected point far outside the
        # color frame.
        T_color_from_depth = np.eye(4)
        T_color_from_depth[0, 3] = -100.0

        registered = register_depth_to_color(depth, K, K, w, h, T_color_from_depth)

        assert np.all(registered == 0)

    def test_zbuffer_keeps_nearest_point_on_collision(self):
        # Two source points that project to the same target pixel: the
        # nearer one (smaller transformed Z) must win.
        h, w = 8, 8
        K = _identity_K(fx=100.0, fy=100.0, cx=4.0, cy=4.0)
        depth = np.zeros((h, w), dtype=np.float64)
        # Two different source pixels, chosen so that after unprojection
        # and an identity transform, they'd naturally land on different
        # target pixels -- instead we directly construct a transform-free
        # collision by placing two points along the same optical ray at
        # different depths is impossible with distinct pixels, so instead
        # verify via a translation that maps two distinct depth pixels to
        # the same rounded color pixel.
        depth[4, 4] = 5.0   # far point at the principal point
        depth[4, 5] = 1.0   # near point one pixel over

        # A small translation in X chosen to make pixel (4,5)'s reprojection
        # round to the same integer pixel as (4,4)'s.
        T = np.eye(4)
        # unprojected X for (row=4, col=5): (5-4)/100 * 1.0 = 0.01
        # unprojected X for (row=4, col=4): (4-4)/100 * 5.0 = 0.0
        # Shift color camera so both reproject near col=4: no shift needed
        # for (4,4) (already at cx); shift (4,5)'s point by -0.01 in X to
        # also land at col=4: T translates depth-frame points by -0.01 in X.
        T[0, 3] = -0.01

        registered = register_depth_to_color(depth, K, K, w, h, T)

        # Both source points now round to color pixel (row=4, col=4).
        # The near point (Z=1.0) must win over the far point (Z=5.0).
        assert registered[4, 4] == 1.0


class TestFindTransformChain:
    def _fake_transforms(self):
        # A -> B -> C -> D chain, plus an isolated node.
        return {
            ("A", "B"): transform_to_matrix((1.0, 0.0, 0.0), (0, 0, 0, 1)),
            ("B", "C"): transform_to_matrix((0.0, 2.0, 0.0), (0, 0, 0, 1)),
            ("C", "D"): transform_to_matrix((0.0, 0.0, 3.0), (0, 0, 0, 1)),
        }

    def test_multi_hop_forward_chain(self):
        transforms = self._fake_transforms()
        T = find_transform_chain(transforms, "D", "A")
        assert T is not None
        # D -> A the "wrong way": since edges are (parent, child) with
        # parent_T_child, and D is the "innermost child", going D->A
        # composes the same translations. Sanity: applying T to the origin
        # of D should give D's position expressed in A's frame, i.e.
        # (1, 2, 3).
        origin_D = np.array([0.0, 0.0, 0.0, 1.0])
        p_in_A = T @ origin_D
        np.testing.assert_allclose(p_in_A[:3], [1.0, 2.0, 3.0])

    def test_inversion_when_only_reverse_edge_stored(self):
        # Only the (parent="A", child="B") edge is stored, as
        # parent_T_child (maps a point in B's frame into A's frame). Asking
        # for A -> B (parent -> child) requires the traversal to invert
        # that stored edge, since only child -> parent is directly usable
        # without inversion.
        transforms = {("A", "B"): transform_to_matrix((5.0, 0.0, 0.0), (0, 0, 0, 1))}

        T_A_to_B = find_transform_chain(transforms, "A", "B")
        assert T_A_to_B is not None
        origin_A = np.array([0.0, 0.0, 0.0, 1.0])
        p_in_B = T_A_to_B @ origin_A
        # A's origin sits at (-5,0,0) in B's frame (inverse of B's origin
        # sitting at (5,0,0) in A's frame).
        np.testing.assert_allclose(p_in_B[:3], [-5.0, 0.0, 0.0])

    def test_same_frame_is_identity(self):
        transforms = self._fake_transforms()
        T = find_transform_chain(transforms, "B", "B")
        np.testing.assert_allclose(T, np.eye(4))

    def test_unreachable_frame_returns_none(self):
        transforms = self._fake_transforms()
        assert find_transform_chain(transforms, "A", "not_in_graph") is None
        assert find_transform_chain(transforms, "nonexistent", "A") is None


class TestResolveRegistrationTransform:
    def test_disabled_by_default(self):
        T, reason = resolve_registration_transform({}, {"K": [1, 0, 0, 0, 1, 0, 0, 0, 1], "width": 10, "height": 10}, {})
        assert T is None
        assert reason == "disabled"

    def test_missing_tf_falls_back(self):
        depth_config = {"registration": {"enabled": True}}
        color_info = {"K": [1, 0, 0, 0, 1, 0, 0, 0, 1], "width": 10, "height": 10}
        T, reason = resolve_registration_transform(depth_config, color_info, {})
        assert T is None
        assert "tf_static" in reason


class _FakeDepthMsg:
    def __init__(self, encoding, height, width, data):
        self.encoding = encoding
        self.height = height
        self.width = width
        self.data = data


class _FakeReader:
    def __init__(self, messages):
        self._messages = messages

    def get_messages(self, topic):
        return iter(self._messages)


def _make_depth_msg(value_mm: int, height=8, width=8):
    arr = np.full((height, width), value_mm, dtype=np.uint16)
    return _FakeDepthMsg("16uc1", height, width, arr.tobytes())


class TestExportDepthFramesRegistrationWiring:
    def test_default_disabled_path_is_byte_identical(self, tmp_path):
        # Critical regression guard: calling export_depth_frames WITHOUT
        # the `registration` kwarg (as every pre-existing caller/test
        # does) must produce byte-identical output to before registration
        # support existed.
        messages = [(1000 + i, _make_depth_msg(2000)) for i in range(5)]

        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        config = {"min_depth_m": 0.0, "max_depth_m": 10.0}

        result_a = export_depth_frames(_FakeReader(messages), out_a, "depth", config)
        result_b = export_depth_frames(
            _FakeReader(messages), out_b, "depth", config, registration=None
        )

        assert len(result_a) == len(result_b) == 5
        for (_, fname_a), (_, fname_b) in zip(result_a, result_b):
            img_a = cv2.imread(str(out_a / fname_a), cv2.IMREAD_UNCHANGED)
            img_b = cv2.imread(str(out_b / fname_b), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(img_a, img_b)
            assert img_a[0, 0] == 2000

    def test_registration_enabled_changes_output(self, tmp_path):
        h, w = 16, 16
        messages = [(1000, _make_depth_msg(2000, height=h, width=w))]
        config = {"min_depth_m": 0.0, "max_depth_m": 10.0}

        K = _identity_K(fx=100.0, fy=100.0, cx=8.0, cy=8.0)
        T = np.eye(4)
        T[0, 3] = -0.5  # nontrivial shift -- moves content within the frame

        registration = {
            "depth_K": K,
            "color_K": K,
            "color_width": w,
            "color_height": h,
            "T_color_from_depth": T,
        }

        out_disabled = tmp_path / "disabled"
        out_enabled = tmp_path / "enabled"

        result_disabled = export_depth_frames(
            _FakeReader(messages), out_disabled, "depth", config
        )
        result_enabled = export_depth_frames(
            _FakeReader(messages), out_enabled, "depth", config, registration=registration
        )

        img_disabled = cv2.imread(
            str(out_disabled / result_disabled[0][1]), cv2.IMREAD_UNCHANGED
        )
        img_enabled = cv2.imread(
            str(out_enabled / result_enabled[0][1]), cv2.IMREAD_UNCHANGED
        )
        assert not np.array_equal(img_disabled, img_enabled)

    def test_resolve_depth_registration_disabled_is_none(self):
        registration, reason = resolve_depth_registration(
            {"registration": {"enabled": False}}, None, None, None
        )
        assert registration is None
        assert reason == "disabled"
