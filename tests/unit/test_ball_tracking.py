"""Ball trajectory estimation.

These tests encode the behaviours the brief calls out explicitly: reject
implausible jumps, survive short occlusions, and *refuse to invent a position*
during a long disappearance.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionpitch.ball_tracking.kalman import BallKalmanFilter
from visionpitch.ball_tracking.trajectory import BallTrajectoryEstimator
from visionpitch.common.config import Config
from visionpitch.common.types import BBox, Detection, ObjectClass


def ball_at(frame: int, x: float, y: float, conf: float = 0.8) -> Detection:
    return Detection(frame, ObjectClass.BALL, BBox(x - 5, y - 5, x + 5, y + 5), conf, "test")


def run(estimator, detections_by_frame, n_frames, width=1920):
    frames = list(range(n_frames))
    timestamps = {f: f / 25.0 for f in frames}
    return estimator.estimate(detections_by_frame, frames, timestamps, width)


class TestBallKalmanFilter:
    def test_tracks_constant_velocity_exactly(self) -> None:
        kf = BallKalmanFilter()
        kf.initiate((100.0, 100.0))
        last_step = 24
        for step in range(1, last_step + 1):
            kf.predict()
            kf.update((100.0 + 10 * step, 100.0 + 5 * step))
        # The filter has absorbed the measurement at ``last_step``; peeking one
        # frame ahead must predict ``last_step + 1``.
        x, y = kf.peek(1)
        assert x == pytest.approx(100.0 + 10 * (last_step + 1), abs=8.0)
        assert y == pytest.approx(100.0 + 5 * (last_step + 1), abs=8.0)

    def test_models_acceleration(self) -> None:
        """A constant-velocity filter would lag on a parabola; this must not."""
        kf = BallKalmanFilter()
        kf.initiate((0.0, 0.0))
        positions = [(t * 10.0, 0.5 * 2.0 * t * t) for t in range(1, 30)]
        for pos in positions:
            kf.predict()
            kf.update(pos)
        predicted_y = kf.peek(1)[1]
        truth_y = 0.5 * 2.0 * 30 * 30
        assert abs(predicted_y - truth_y) < 0.10 * truth_y

    def test_uncertainty_grows_while_coasting(self) -> None:
        kf = BallKalmanFilter()
        kf.initiate((100.0, 100.0))
        start = kf.position_uncertainty
        for _ in range(15):
            kf.predict()
        assert kf.position_uncertainty > start

    def test_gating_rejects_a_distant_candidate(self) -> None:
        kf = BallKalmanFilter()
        kf.initiate((100.0, 100.0))
        for step in range(1, 12):
            kf.predict()
            kf.update((100.0 + 5 * step, 100.0))
        assert kf.gating_distance((160.0, 100.0)) < kf.gating_distance((1500.0, 900.0))

    def test_peek_does_not_mutate(self) -> None:
        kf = BallKalmanFilter()
        kf.initiate((50.0, 50.0))
        before = kf.x.copy()
        kf.peek(5)
        assert np.array_equal(kf.x, before)


class TestTrajectoryEstimator:
    def test_clean_trajectory_is_fully_observed(self, config: Config) -> None:
        estimator = BallTrajectoryEstimator(config)
        detections = {f: [ball_at(f, 100 + 8 * f, 300.0)] for f in range(40)}
        states = run(estimator, detections, 40)
        assert sum(1 for s in states.values() if s.observed) >= 38
        assert all(s.position is not None for s in states.values())

    def test_short_occlusion_is_interpolated_and_flagged(self, config: Config) -> None:
        detections = {}
        for f in range(40):
            if 15 <= f < 21:  # 6-frame gap, under the limit
                continue
            detections[f] = [ball_at(f, 100 + 8 * f, 300.0)]

        states = run(BallTrajectoryEstimator(config), detections, 40)
        for f in range(15, 21):
            assert states[f].position is not None, f"frame {f} should be interpolated"
            assert states[f].interpolated is True
            assert states[f].observed is False
            # An inferred position must never claim to be an observation.
            assert states[f].confidence < 1.0
        assert states[17].position[0] == pytest.approx(100 + 8 * 17, abs=6.0)

    def test_long_disappearance_is_left_unknown(self, config: Config) -> None:
        """The brief is explicit: do not hallucinate a ball position."""
        gap = config.ball_tracking.max_interpolation_gap_frames + 15
        detections = {}
        for f in range(60):
            if 20 <= f < 20 + gap:
                continue
            detections[f] = [ball_at(f, 100 + 8 * f, 300.0)]

        states = run(BallTrajectoryEstimator(config), detections, 60)
        unknown = [f for f in range(21, 20 + gap - 1) if states[f].position is None]
        assert unknown, "a long gap must produce unknown positions, not a guess"
        for f in unknown:
            assert states[f].interpolated is False
            assert states[f].confidence == 0.0
            assert states[f].uncertainty_px == float("inf")

    def test_teleporting_false_positive_is_rejected(self, config: Config) -> None:
        detections = {}
        for f in range(40):
            candidates = [ball_at(f, 100 + 8 * f, 300.0, conf=0.7)]
            if f == 20:
                # A high-confidence blob on the far side of the frame. A greedy
                # tracker takes it; a whole-sequence search must not.
                candidates.append(ball_at(f, 1800.0, 950.0, conf=0.95))
            detections[f] = candidates

        states = run(BallTrajectoryEstimator(config), detections, 40)
        assert states[20].position is not None
        assert states[20].position[0] == pytest.approx(100 + 8 * 20, abs=30.0)

    def test_recovers_after_the_ball_leaves_and_returns(self, config: Config) -> None:
        """Two disjoint sightings separated by a long absence must both survive.

        A single global best path can only keep one of them; this is the
        regression test for that bug.
        """
        detections = {}
        for f in range(0, 30):
            detections[f] = [ball_at(f, 100 + 5 * f, 300.0)]
        for f in range(90, 130):
            detections[f] = [ball_at(f, 1000 - 5 * (f - 90), 500.0)]

        states = run(BallTrajectoryEstimator(config), detections, 130)
        first = sum(1 for f in range(0, 30) if states[f].observed)
        second = sum(1 for f in range(90, 130) if states[f].observed)
        assert first >= 25, "first segment was discarded"
        assert second >= 35, "second segment was discarded"
        assert states[60].position is None, "the long absence must stay unknown"

    def test_no_detections_at_all_is_handled(self, config: Config) -> None:
        states = run(BallTrajectoryEstimator(config), {}, 20)
        assert len(states) == 20
        assert all(s.position is None for s in states.values())

    def test_speed_limit_scales_with_frame_width(self, config: Config) -> None:
        """A 4K clip must not be judged by 1080p pixel speeds."""
        estimator = BallTrajectoryEstimator(config)
        step_hd = estimator._max_step(1920, 1)
        step_4k = estimator._max_step(3840, 1)
        assert step_4k == pytest.approx(2 * step_hd)

    def test_quality_report_totals_are_consistent(self, config: Config) -> None:
        detections = {f: [ball_at(f, 100 + 8 * f, 300.0)] for f in range(0, 30, 2)}
        states = run(BallTrajectoryEstimator(config), detections, 40)
        report = BallTrajectoryEstimator.quality_report(states)
        assert report["observed"] + report["interpolated"] + report["unknown"] == report["frames"]
        assert 0.0 <= report["observed_ratio"] <= 1.0
