"""Constant-acceleration Kalman filter for the ball.

Why constant acceleration rather than constant velocity
-------------------------------------------------------
A football in flight is under gravity and drag, and a rolling ball decelerates.
A constant-velocity model treats all of that as process noise, so its prediction
lags systematically on any lofted pass -- exactly when the ball is hardest to
detect and the prediction matters most for the ROI crop. A constant-acceleration
model in image space absorbs the projected gravity term and tracks parabolic
flight far better.

State: ``[x, y, vx, vy, ax, ay]`` in image pixels, per frame.
"""

from __future__ import annotations

import numpy as np


class BallKalmanFilter:
    """Constant-acceleration filter with Mahalanobis gating."""

    def __init__(self, process_noise: float = 9.0, measurement_noise: float = 4.0) -> None:
        self.q = float(process_noise)
        self.r = float(measurement_noise)

        dt = 1.0  # one frame; the estimator works in frame units throughout
        self.F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt * dt, 0],
                [0, 1, 0, dt, 0, 0.5 * dt * dt],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
            dtype=np.float64,
        )
        # Continuous white-noise-acceleration discretisation, per axis.
        g = np.array(
            [
                [dt**4 / 4, dt**3 / 2, dt**2 / 2],
                [dt**3 / 2, dt**2, dt],
                [dt**2 / 2, dt, 1.0],
            ]
        )
        self.Q = np.zeros((6, 6))
        for axis in (0, 1):
            idx = [axis, axis + 2, axis + 4]
            self.Q[np.ix_(idx, idx)] = g * self.q
        self.R = np.eye(2) * self.r

        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e4
        self.initialised = False

    # -- lifecycle ---------------------------------------------------------- #

    def reset(self) -> None:
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e4
        self.initialised = False

    def initiate(self, position: tuple[float, float]) -> None:
        self.x = np.array([position[0], position[1], 0.0, 0.0, 0.0, 0.0])
        # Position is well known, velocity and acceleration are not.
        self.P = np.diag([self.r, self.r, 400.0, 400.0, 100.0, 100.0])
        self.initialised = True

    # -- filtering ---------------------------------------------------------- #

    def predict(self, steps: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Advance the state ``steps`` frames. Mutates the filter."""
        for _ in range(max(1, steps)):
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def peek(self, steps: int = 1) -> tuple[float, float]:
        """Predicted position ``steps`` frames ahead, without mutating state.

        This is what the detection stage uses to place its ROI crop.
        """
        x = self.x.copy()
        F = np.linalg.matrix_power(self.F, max(1, steps))
        x = F @ x
        return float(x[0]), float(x[1])

    def update(self, measurement: tuple[float, float]) -> None:
        z = np.asarray(measurement, dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.initialised = True

    # -- gating ------------------------------------------------------------- #

    def gating_distance(self, measurement: tuple[float, float]) -> float:
        """Squared Mahalanobis distance from the prediction to a candidate."""
        if not self.initialised:
            return 0.0
        z = np.asarray(measurement, dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        innovation = z - self.H @ self.x
        try:
            return float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            return float("inf")

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def position_uncertainty(self) -> float:
        """1-sigma positional uncertainty in pixels, as a single scalar."""
        return float(np.sqrt(max(0.0, 0.5 * (self.P[0, 0] + self.P[1, 1]))))
