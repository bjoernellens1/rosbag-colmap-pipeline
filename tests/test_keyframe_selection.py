"""Tests for motion-adaptive keyframe selection (ingest/keyframe_selection.py)."""

import numpy as np
import cv2

from colmap_rgbd_gt.ingest.keyframe_selection import KeyframeSelector


def _textured_image(seed=0, size=200):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    # add some structure so ORB has features to find
    cv2.rectangle(img, (30, 30), (100, 100), (255, 255, 255), -1)
    cv2.circle(img, (150, 150), 30, (0, 0, 0), -1)
    return img


def test_first_frame_always_selected():
    selector = KeyframeSelector()
    img = _textured_image()

    assert selector.should_select(img) is True


def test_identical_frame_not_selected_again():
    selector = KeyframeSelector(min_match_ratio=0.5)
    img = _textured_image()

    assert selector.should_select(img) is True
    # Same image again -- full overlap, should not become a new keyframe.
    assert selector.should_select(img) is False


def test_completely_different_frame_selected():
    selector = KeyframeSelector(min_match_ratio=0.5)
    img1 = _textured_image(seed=1)
    img2 = _textured_image(seed=99)

    assert selector.should_select(img1) is True
    assert selector.should_select(img2) is True


def test_max_frame_gap_forces_selection():
    selector = KeyframeSelector(min_match_ratio=0.0, max_frame_gap=3)
    img = _textured_image()

    assert selector.should_select(img) is True  # frame 0, keyframe
    assert selector.should_select(img) is False  # frame 1, high overlap (ratio threshold 0.0 never trips)
    assert selector.should_select(img) is False  # frame 2
    assert selector.should_select(img) is True  # frame 3, gap forces selection


def test_blank_image_handled_gracefully():
    selector = KeyframeSelector()
    blank = np.zeros((100, 100, 3), dtype=np.uint8)

    # Should not raise even if ORB finds no features.
    result = selector.should_select(blank)
    assert isinstance(result, bool)
