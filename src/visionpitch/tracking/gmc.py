"""Global (camera) motion compensation.

Broadcast football is shot with a panning, tilting, zooming camera. Between two
consecutive frames a stationary point on the pitch can move 30-60 px near the
frame edges. A tracker that assumes a static camera spends its entire motion
budget on camera movement and has nothing left to model the players, which shows
up as ID switches during fast pans and lost tracks after a zoom.

This module estimates the frame-to-frame background warp and hands it to the
Kalman filter, which transports every track's state into the new frame's
coordinates before association.

Three estimators, in decreasing order of the usual accuracy/cost trade-off:

``sparseOptFlow``
    Shi-Tomasi corners tracked with pyramidal Lucas-Kanade, then a partial
    affine fit under RANSAC. Cheap and robust; the default.
``ecc``
    Direct intensity alignment (Enhanced Correlation Coefficient). More accurate
    on low-texture frames but iterative and markedly slower.
``orb``
    ORB features with descriptor matching. Handles larger baselines, useful if
    frames are being sampled with a large stride.

The pitch is a large, moving-object-sparse, textured plane, so a *partial
affine* (rotation + uniform scale + translation) model is the right amount of
freedom: a full homography would be seduced by the players and by the strong
perspective gradient across the frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionpitch.common.logging import get_logger

log = get_logger("tracking.gmc")

IDENTITY_WARP = np.eye(2, 3, dtype=np.float64)


class GlobalMotionCompensator:
    """Estimates the background affine warp between consecutive frames."""

    def __init__(
        self,
        method: str = "sparseOptFlow",
        downscale: int = 2,
        max_corners: int = 1000,
        quality_level: float = 0.01,
        min_distance: int = 8,
        ransac_threshold: float = 3.0,
    ) -> None:
        self.method = method
        self.downscale = max(1, int(downscale))
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_threshold = ransac_threshold

        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None
        self._orb: cv2.ORB | None = None
        self._matcher: cv2.BFMatcher | None = None
        self.n_failures = 0

        if method == "orb":
            self._orb = cv2.ORB_create(nfeatures=2000)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # -- helpers ------------------------------------------------------------ #

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            gray = cv2.resize(
                gray,
                (gray.shape[1] // self.downscale, gray.shape[0] // self.downscale),
                interpolation=cv2.INTER_AREA,
            )
        return gray

    def _rescale(self, warp: np.ndarray) -> np.ndarray:
        """Lift a warp estimated at reduced resolution back to full resolution.

        Only the translation column scales -- the rotation/scale block is
        resolution-independent.
        """
        if self.downscale == 1:
            return warp
        out = warp.copy()
        out[:2, 2] *= self.downscale
        return out

    @staticmethod
    def _detection_mask(shape: tuple[int, int], boxes: np.ndarray | None, scale: int) -> np.ndarray:
        """Mask out detected objects so the fit sees background, not players.

        This is the single most important detail in the whole module. Twenty-two
        players moving coherently up the pitch look exactly like camera motion to
        an unmasked estimator, and the tracker then compensates for the players'
        own movement -- actively making association worse than no GMC at all.
        """
        mask = np.full(shape, 255, dtype=np.uint8)
        if boxes is None or len(boxes) == 0:
            return mask
        h, w = shape
        for box in np.asarray(boxes).reshape(-1, 4):
            x1, y1, x2, y2 = (box / scale).astype(int)
            # Dilate slightly: motion blur and shadow extend past the box.
            pad = 4
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    # -- estimators --------------------------------------------------------- #

    def _apply_sparse_flow(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._prev_gray is None or self._prev_points is None or len(self._prev_points) < 8:
            return IDENTITY_WARP.copy()

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None
        )
        if next_points is None or status is None:
            return IDENTITY_WARP.copy()

        good_prev = self._prev_points[status.ravel() == 1]
        good_next = next_points[status.ravel() == 1]
        if len(good_prev) < 8:
            return IDENTITY_WARP.copy()

        warp, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold
        )
        if warp is None:
            return IDENTITY_WARP.copy()
        if inliers is not None and inliers.sum() < 0.25 * len(good_prev):
            # A weak consensus means the fit latched onto moving objects.
            return IDENTITY_WARP.copy()
        return warp.astype(np.float64)

    def _apply_ecc(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._prev_gray is None:
            return IDENTITY_WARP.copy()
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        try:
            _, warp = cv2.findTransformECC(
                self._prev_gray, gray, warp, cv2.MOTION_EUCLIDEAN, criteria, mask, 5
            )
        except cv2.error as exc:
            # ECC diverges on low-texture or heavily-occluded frames. That is a
            # normal outcome, not a crash, but it must be counted.
            log.debug("ECC failed to converge: %s", exc)
            self.n_failures += 1
            return IDENTITY_WARP.copy()
        return warp.astype(np.float64)

    def _apply_orb(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._prev_gray is None or self._orb is None or self._matcher is None:
            return IDENTITY_WARP.copy()
        kp_prev, des_prev = self._orb.detectAndCompute(self._prev_gray, None)
        kp_next, des_next = self._orb.detectAndCompute(gray, mask)
        if des_prev is None or des_next is None or len(kp_prev) < 8 or len(kp_next) < 8:
            return IDENTITY_WARP.copy()

        matches = self._matcher.match(des_prev, des_next)
        if len(matches) < 8:
            return IDENTITY_WARP.copy()
        matches = sorted(matches, key=lambda m: m.distance)[: max(8, len(matches) // 2)]
        src = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        dst = np.float32([kp_next[m.trainIdx].pt for m in matches])
        warp, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold
        )
        return IDENTITY_WARP.copy() if warp is None else warp.astype(np.float64)

    # -- public API --------------------------------------------------------- #

    def apply(self, image: np.ndarray, detection_boxes: np.ndarray | None = None) -> np.ndarray:
        """Estimate the warp from the previous frame to ``image``.

        Returns a 2x3 affine matrix in full-resolution image coordinates.
        Returns identity on the first frame and whenever estimation fails --
        never ``None``, so callers cannot forget to handle the failure case.
        """
        if self.method == "none":
            return IDENTITY_WARP.copy()

        gray = self._preprocess(image)
        mask = self._detection_mask(gray.shape, detection_boxes, self.downscale)

        if self.method == "sparseOptFlow":
            warp = self._apply_sparse_flow(gray, mask)
            self._prev_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                mask=mask,
                blockSize=3,
            )
        elif self.method == "ecc":
            warp = self._apply_ecc(gray, mask)
        elif self.method == "orb":
            warp = self._apply_orb(gray, mask)
        else:
            raise ValueError(f"unknown gmc method {self.method!r}")

        self._prev_gray = gray
        return self._rescale(warp)

    def reset(self) -> None:
        """Forget history. Call on a shot change -- there is no motion to track
        across a cut, and estimating one produces a nonsense warp."""
        self._prev_gray = None
        self._prev_points = None
