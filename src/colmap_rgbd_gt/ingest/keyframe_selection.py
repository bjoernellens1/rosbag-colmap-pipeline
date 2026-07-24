"""Motion-adaptive keyframe selection.

A fixed frame stride assumes roughly constant camera motion speed across
the whole bag. In practice motion speed varies (turns, stops, fast
segments), so a stride tuned for the average case is too sparse exactly
where it matters most -- during fast motion, consecutive kept frames can
lose all shared features, which is what causes COLMAP's incremental mapper
to lose tracking and split the reconstruction into disconnected models.

`KeyframeSelector` instead measures actual visual overlap between each
candidate frame and the last selected keyframe, selecting a new keyframe
whenever overlap drops below a threshold -- densely during fast motion,
sparsely when the camera is nearly stationary -- while a hard
`max_frame_gap` still guarantees forward progress even over a textureless
stretch with no reliable matches at all.

Overlap is measured via a RANSAC-verified homography inlier ratio, not raw
ORB match count: repetitive scene texture (floor tiles, wall patterns)
produces many spurious nearest-neighbor matches between visually distant
frames, which inflates a raw match-count ratio and causes the selector to
wait too long -- exactly where real overlap has already dropped. Lowe's
ratio test (on knn matches) plus a RANSAC-inlier count against a fitted
homography filters those out, leaving a proxy that actually tracks
geometric overlap.
"""

import cv2
import numpy as np


class KeyframeSelector:
    def __init__(
        self,
        min_match_ratio: float = 0.5,
        max_frame_gap: int = 30,
        orb_features: int = 500,
        lowe_ratio: float = 0.75,
        min_good_matches: int = 8,
        ransac_reproj_threshold: float = 5.0,
    ):
        self._orb = cv2.ORB_create(nfeatures=orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # no crossCheck: needed for knnMatch
        self._min_match_ratio = min_match_ratio
        self._max_frame_gap = max_frame_gap
        self._lowe_ratio = lowe_ratio
        self._min_good_matches = min_good_matches
        self._ransac_reproj_threshold = ransac_reproj_threshold
        self._last_kp: tuple | None = None
        self._last_des: np.ndarray | None = None
        self._frames_since_keyframe = 0

    def _geometric_overlap_ratio(self, kp, des) -> float:
        """RANSAC-verified homography inlier ratio against the last keyframe.

        Returns 0.0 (forces a new keyframe) if there are too few reliable
        matches to fit a homography at all.
        """
        knn_matches = self._matcher.knnMatch(self._last_des, des, k=2)

        good = []
        for pair in knn_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self._lowe_ratio * n.distance:
                good.append(m)

        if len(good) < self._min_good_matches:
            return 0.0

        src_pts = np.float32([self._last_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self._ransac_reproj_threshold)
        if mask is None:
            return 0.0

        inliers = int(mask.sum())
        denom = min(len(self._last_kp), len(kp))
        return inliers / max(denom, 1)

    def should_select(self, image_bgr: np.ndarray) -> bool:
        """Returns True if `image_bgr` should be kept as a new keyframe.

        Call once per candidate frame, in order -- this is stateful
        (compares against the last *selected* keyframe, not the last
        frame seen).
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        self._frames_since_keyframe += 1

        if self._last_des is None or des is None or len(des) < self._min_good_matches:
            is_keyframe = True
        elif self._frames_since_keyframe >= self._max_frame_gap:
            is_keyframe = True
        else:
            overlap_ratio = self._geometric_overlap_ratio(kp, des)
            is_keyframe = overlap_ratio < self._min_match_ratio

        if is_keyframe:
            self._last_kp, self._last_des = kp, des
            self._frames_since_keyframe = 0

        return is_keyframe
