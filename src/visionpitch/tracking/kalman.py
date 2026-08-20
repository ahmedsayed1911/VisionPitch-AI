"""Kalman filter for bounding-box tracking.

State is ``[cx, cy, aspect, height, vcx, vcy, vaspect, vheight]``. Parameterising
by aspect ratio and height rather than width and height is the SORT/ByteTrack
convention and matters for football: a player's height in pixels is a smooth
function of their depth in the frame, whereas their width jumps whenever their
arms or legs extend. Tying process noise to height therefore scales uncertainty
with distance from camera, which is what we want on a wide broadcast shot.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

#: 0.95 quantile of the chi-square distribution, by degrees of freedom.
CHI2_95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592}


def xyxy_to_state(box: np.ndarray) -> np.ndarray:
    """``[x1, y1, x2, y2]`` -> ``[cx, cy, aspect, height]``."""
    x1, y1, x2, y2 = box
    w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], dtype=np.float64)


def state_to_xyxy(state: np.ndarray) -> np.ndarray:
    """``[cx, cy, aspect, height, ...]`` -> ``[x1, y1, x2, y2]``."""
    cx, cy, a, h = state[:4]
    h = max(1e-6, h)
    w = max(1e-6, a * h)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


class KalmanBoxFilter:
    """Constant-velocity filter over box centre, aspect and height."""

    def __init__(self, position_weight: float = 1.0 / 20, velocity_weight: float = 1.0 / 160):
        self.ndim = 4
        self._position_weight = position_weight
        self._velocity_weight = velocity_weight

        # Constant-velocity transition: position += velocity each step.
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        # We observe position only.
        self._H = np.eye(4, 8)

    # -- lifecycle ---------------------------------------------------------- #

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Start a track from one observation, with velocity unknown."""
        mean = np.concatenate([measurement, np.zeros(4)])
        h = measurement[3]
        std = np.array(
            [
                2 * self._position_weight * h,
                2 * self._position_weight * h,
                1e-2,
                2 * self._position_weight * h,
                10 * self._velocity_weight * h,
                10 * self._velocity_weight * h,
                1e-5,
                10 * self._velocity_weight * h,
            ]
        )
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array(
            [
                self._position_weight * h,
                self._position_weight * h,
                1e-2,
                self._position_weight * h,
                self._velocity_weight * h,
                self._velocity_weight * h,
                1e-5,
                self._velocity_weight * h,
            ]
        )
        Q = np.diag(np.square(std))
        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + Q
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array(
            [
                self._position_weight * h,
                self._position_weight * h,
                1e-1,
                self._position_weight * h,
            ]
        )
        R = np.diag(np.square(std))
        return self._H @ mean, self._H @ covariance @ self._H.T + R

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        # Solve rather than invert: the projected covariance can be poorly
        # conditioned when a track has been coasting through a long occlusion.
        kalman_gain = np.linalg.solve(
            projected_cov.T, (covariance @ self._H.T).T
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    # -- gating ------------------------------------------------------------- #

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        """Squared Mahalanobis distance from the prediction to each measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)
        measurements = np.atleast_2d(measurements)
        try:
            chol = np.linalg.cholesky(projected_cov)
        except np.linalg.LinAlgError:
            return np.full(measurements.shape[0], np.inf)
        diff = (measurements - projected_mean).T
        z = solve_triangular(chol, diff, lower=True)
        return np.sum(z * z, axis=0)

    def apply_motion_compensation(
        self, mean: np.ndarray, covariance: np.ndarray, warp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rigid-transform the filter state into the new frame's coordinates.

        Without this, every track's Kalman prediction is expressed in the
        previous frame's pixel grid. On a panning broadcast camera that is the
        dominant source of association failure -- the whole scene translates by
        tens of pixels between frames while the players barely move relative to
        the pitch.
        """
        if warp is None:
            return mean, covariance

        R = warp[:2, :2]
        t = warp[:2, 2]

        # Build the 8x8 equivalent: centre translates and rotates, height and
        # aspect scale, and velocities rotate but do not translate.
        R8 = np.eye(8)
        R8[0:2, 0:2] = R
        R8[4:6, 4:6] = R
        scale = float(np.sqrt(abs(np.linalg.det(R)))) or 1.0
        R8[3, 3] = scale
        R8[7, 7] = scale

        new_mean = R8 @ mean
        new_mean[0:2] += t
        new_cov = R8 @ covariance @ R8.T
        return new_mean, new_cov
