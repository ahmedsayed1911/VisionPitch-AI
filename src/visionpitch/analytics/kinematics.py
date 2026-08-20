"""Physical statistics from tracked pitch positions.

The naive approach -- difference consecutive positions, divide by dt, sum the
magnitudes -- is wrong in a specific and severe way. Position noise of even
20 cm at 30 fps differentiates to 6 m/s of pure noise, so a stationary player
accumulates hundreds of metres of "distance covered" and shows a top speed
faster than the world record. Every published match-data pipeline smooths
before differentiating; this one does too, and additionally:

* uses **only** rows Phase 1B deemed physically trustworthy (``valid``)
* refuses to bridge gaps, because a player who was untracked for two seconds
  did not travel in a straight line during them
* discards physically impossible samples rather than clipping them, since a
  12 m/s sample is a tracking error and clipping it to 12 keeps the error
* reports the **coverage** every number rests on

A player tracked for 14 usable frames gets a distance figure with coverage
0.05, not a confident total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from visionpitch.analytics.types import (
    MAX_PLAUSIBLE_ACCEL_M_S2,
    MAX_PLAUSIBLE_SPEED_M_S,
    SPEED_ZONES,
    SPRINT_MIN_DURATION_S,
    SPRINT_MIN_SPEED_M_S,
    Metric,
    MetricBasis,
)
from visionpitch.common.logging import get_logger

log = get_logger("analytics.kinematics")


@dataclass
class Segment:
    """A contiguous run of usable positions for one track.

    Contiguity matters: kinematics may only be computed within a segment. A gap
    between segments is unmeasured time, not zero movement and not straight-line
    movement.
    """

    frames: np.ndarray
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        return float(self.times[-1] - self.times[0]) if self.n > 1 else 0.0


@dataclass
class KinematicProfile:
    """Physical output for one track, with the coverage behind it."""

    track_id: int
    n_rows_total: int = 0
    n_rows_usable: int = 0
    n_segments: int = 0
    tracked_duration_s: float = 0.0
    measured_duration_s: float = 0.0

    distance_m: float = 0.0
    distance_by_zone_m: dict[str, float] = field(default_factory=dict)
    mean_speed_m_s: float = 0.0
    top_speed_m_s: float = 0.0
    n_sprints: int = 0
    sprint_distance_m: float = 0.0
    n_accelerations: int = 0
    n_decelerations: int = 0
    mean_position: tuple[float, float] | None = None
    #: per-frame speed, for heatmaps and work-rate timelines
    speed_by_frame: dict[int, float] = field(default_factory=dict)
    #: samples discarded as physically impossible
    n_rejected_samples: int = 0

    @property
    def coverage(self) -> float:
        """Share of this track's rows that contributed to the physical numbers."""
        return self.n_rows_usable / self.n_rows_total if self.n_rows_total else 0.0

    def metric(self, value: float | int, unit: str, samples: int | None = None) -> Metric:
        return Metric(
            value=value,
            coverage=self.coverage,
            # Confidence is the coverage tempered by how much continuous time was
            # actually measured: 200 frames scattered across 200 one-frame
            # segments support nothing, however good the coverage ratio looks.
            confidence=float(
                np.clip(self.coverage * min(1.0, self.measured_duration_s / 5.0), 0.0, 1.0)
            ),
            n_samples=samples if samples is not None else self.n_rows_usable,
            basis=MetricBasis.VALID_ONLY,
            unit=unit,
        )


def extract_segments(
    track_rows: pd.DataFrame, fps: float, max_gap_frames: int = 5
) -> list[Segment]:
    """Split one track's usable rows into contiguous runs.

    ``max_gap_frames`` tolerates the occasional dropped frame inside an
    otherwise continuous run; anything longer starts a new segment, because the
    player's path across it is unknown.
    """
    if track_rows.empty:
        return []

    ordered = track_rows.sort_values("frame_idx")
    frames = ordered.frame_idx.to_numpy(dtype=np.int64)
    times = ordered.timestamp_s.to_numpy(dtype=np.float64)
    xs = ordered.pitch_x.to_numpy(dtype=np.float64)
    ys = ordered.pitch_y.to_numpy(dtype=np.float64)

    breaks = np.flatnonzero(np.diff(frames) > max_gap_frames) + 1
    segments: list[Segment] = []
    for chunk in np.split(np.arange(len(frames)), breaks):
        if len(chunk) < 2:
            continue
        segments.append(
            Segment(frames[chunk], times[chunk], xs[chunk], ys[chunk])
        )
    _ = fps
    return segments


#: Standard deviation of a single projected player position, in metres.
#:
#: Measured, not assumed. On the validation clip the median frame-to-frame
#: positional step of a *valid* row is 0.717 m, which would imply 21.5 m/s if it
#: were real motion. Since a footballer covers about 0.2 m per frame at 30 fps,
#: essentially all of that step is measurement noise: calibration contributes
#: ~0.44 m of frame-to-frame instability on its own, and bounding-box jitter adds
#: more. Treating a 0.5 m sigma as the measurement noise is therefore
#: conservative rather than pessimistic.
POSITION_NOISE_M = 0.5

#: Process noise as an acceleration, m/s^2. A footballer changes velocity by a
#: few m/s^2; this is what tells the smoother how much of an observed jump to
#: believe.
PROCESS_ACCEL_M_S2 = 3.0


def _rts_smooth(
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    position_noise: float = POSITION_NOISE_M,
    process_accel: float = PROCESS_ACCEL_M_S2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kalman filter plus RTS smoother over a constant-velocity model.

    Returns smoothed ``(x, y, vx, vy)``.

    A moving average was not sufficient here. With noise several times larger
    than the per-frame signal, a boxcar filter either leaves the noise in (short
    window) or destroys real accelerations (long window), and in both cases the
    *velocity* it implies is still obtained by differencing, which re-amplifies
    whatever noise survived.

    A Kalman smoother estimates velocity as part of the state instead of
    differentiating position, and the Rauch-Tung-Striebel backward pass uses the
    whole segment rather than only the past. That is the standard tool for this
    problem and the reason published match-data pipelines report plausible
    sprint speeds from noisy tracking.
    """
    n = len(times)
    if n < 3:
        return x.copy(), y.copy(), np.zeros(n), np.zeros(n)

    # State [x, vx, y, vy]
    dim = 4
    H = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
    R = np.eye(2) * position_noise**2

    means = np.zeros((n, dim))
    covs = np.zeros((n, dim, dim))
    pred_means = np.zeros((n, dim))
    pred_covs = np.zeros((n, dim, dim))
    transitions = np.zeros((n, dim, dim))

    state = np.array([x[0], 0.0, y[0], 0.0])
    cov = np.diag([position_noise**2, 4.0, position_noise**2, 4.0])
    means[0] = state
    covs[0] = cov
    pred_means[0] = state
    pred_covs[0] = cov
    transitions[0] = np.eye(dim)

    for i in range(1, n):
        dt = max(1e-3, float(times[i] - times[i - 1]))
        F = np.eye(dim)
        F[0, 1] = dt
        F[2, 3] = dt
        q = process_accel**2
        block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * q
        Q = np.zeros((dim, dim))
        Q[0:2, 0:2] = block
        Q[2:4, 2:4] = block

        state = F @ state
        cov = F @ cov @ F.T + Q
        pred_means[i] = state
        pred_covs[i] = cov
        transitions[i] = F

        z = np.array([x[i], y[i]])
        S = H @ cov @ H.T + R
        K = cov @ H.T @ np.linalg.inv(S)
        state = state + K @ (z - H @ state)
        cov = (np.eye(dim) - K @ H) @ cov
        means[i] = state
        covs[i] = cov

    # RTS backward pass.
    smoothed_means = means.copy()
    smoothed_covs = covs.copy()
    for i in range(n - 2, -1, -1):
        F = transitions[i + 1]
        try:
            gain = covs[i] @ F.T @ np.linalg.inv(pred_covs[i + 1])
        except np.linalg.LinAlgError:
            continue
        smoothed_means[i] = means[i] + gain @ (smoothed_means[i + 1] - pred_means[i + 1])
        smoothed_covs[i] = covs[i] + gain @ (smoothed_covs[i + 1] - pred_covs[i + 1]) @ gain.T

    return (
        smoothed_means[:, 0],
        smoothed_means[:, 2],
        smoothed_means[:, 1],
        smoothed_means[:, 3],
    )


def _segment_kinematics(
    segment: Segment, smoothing_window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Smoothed speed, acceleration and step distance for one segment.

    Returns ``(speed, accel, step_distance, n_rejected)``, all indexed between
    samples so element *i* describes the interval from sample *i* to *i+1*.
    """
    _ = smoothing_window  # retained for API compatibility; the smoother is adaptive

    sx, sy, vx, vy = _rts_smooth(segment.times, segment.x, segment.y)

    # Speed comes from the estimated velocity state, not from differencing
    # positions. This is the whole point of the smoother: differencing
    # re-introduces exactly the noise it just removed.
    speed_at_sample = np.hypot(vx, vy)
    speed = 0.5 * (speed_at_sample[:-1] + speed_at_sample[1:])

    dt = np.diff(segment.times)
    dt = np.where(dt <= 0, 1e-6, dt)

    # Distance is integrated from the smoothed *path*, which is the quantity
    # actually travelled, rather than from the raw positions whose jitter would
    # add several hundred metres over a match.
    step = np.hypot(np.diff(sx), np.diff(sy))

    impossible = speed > MAX_PLAUSIBLE_SPEED_M_S
    n_rejected = int(impossible.sum())
    speed = np.where(impossible, np.nan, speed)
    step = np.where(impossible, np.nan, step)

    accel = np.full_like(speed, np.nan)
    if len(speed) > 1:
        accel[1:] = np.diff(speed) / dt[1:]
        accel = np.where(np.abs(accel) > MAX_PLAUSIBLE_ACCEL_M_S2, np.nan, accel)

    return speed, accel, step, n_rejected


def compute_kinematics(
    track_id: int,
    valid_rows: pd.DataFrame,
    all_rows_count: int,
    fps: float,
    smoothing_window: int = 7,
) -> KinematicProfile:
    """Physical profile for one track from its usable rows only."""
    profile = KinematicProfile(track_id=track_id, n_rows_total=all_rows_count)
    if valid_rows.empty:
        return profile

    profile.n_rows_usable = len(valid_rows)
    ordered = valid_rows.sort_values("frame_idx")
    profile.tracked_duration_s = float(
        ordered.timestamp_s.iloc[-1] - ordered.timestamp_s.iloc[0]
    )
    profile.mean_position = (
        float(ordered.pitch_x.mean()),
        float(ordered.pitch_y.mean()),
    )

    segments = extract_segments(ordered, fps)
    profile.n_segments = len(segments)
    if not segments:
        return profile

    zone_totals = {name: 0.0 for name, _, _ in SPEED_ZONES}
    all_speeds: list[np.ndarray] = []
    total_distance = 0.0
    n_accel = n_decel = 0
    sprint_count = 0
    sprint_distance = 0.0

    for segment in segments:
        speed, accel, step, rejected = _segment_kinematics(segment, smoothing_window)
        profile.n_rejected_samples += rejected
        profile.measured_duration_s += segment.duration_s

        usable = np.isfinite(step)
        total_distance += float(np.nansum(step))

        for i in np.flatnonzero(usable):
            s = speed[i]
            for name, lo, hi in SPEED_ZONES:
                if lo <= s < hi:
                    zone_totals[name] += float(step[i])
                    break

        all_speeds.append(speed[np.isfinite(speed)])

        # Speed is defined between samples; attribute it to the later frame so a
        # heatmap keyed by frame lines up with the position at that frame.
        for i, frame_idx in enumerate(segment.frames[1:]):
            if np.isfinite(speed[i]):
                profile.speed_by_frame[int(frame_idx)] = float(speed[i])

        finite_accel = accel[np.isfinite(accel)]
        n_accel += int((finite_accel > 2.0).sum())
        n_decel += int((finite_accel < -2.0).sum())

        count, distance = _count_sprints(segment, speed, step)
        sprint_count += count
        sprint_distance += distance

    speeds = np.concatenate(all_speeds) if all_speeds else np.zeros(0)
    profile.distance_m = float(total_distance)
    profile.distance_by_zone_m = {k: round(v, 2) for k, v in zone_totals.items()}
    profile.mean_speed_m_s = float(speeds.mean()) if speeds.size else 0.0
    # 99th percentile, not the maximum: the maximum of a differentiated noisy
    # signal is the largest error, not the fastest the player ran.
    profile.top_speed_m_s = float(np.percentile(speeds, 99)) if speeds.size > 10 else (
        float(speeds.max()) if speeds.size else 0.0
    )
    profile.n_accelerations = n_accel
    profile.n_decelerations = n_decel
    profile.n_sprints = sprint_count
    profile.sprint_distance_m = float(sprint_distance)
    return profile


def _count_sprints(
    segment: Segment, speed: np.ndarray, step: np.ndarray
) -> tuple[int, float]:
    """Count sustained sprints, not individual fast frames.

    A single frame above the sprint threshold is far more likely to be a
    tracking glitch than a sprint, so a run must be held for a minimum duration
    before it counts.
    """
    above = np.isfinite(speed) & (speed >= SPRINT_MIN_SPEED_M_S)
    if not above.any():
        return 0, 0.0

    count = 0
    distance = 0.0
    run_start: int | None = None
    for i, flag in enumerate(above):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            duration = float(segment.times[i] - segment.times[run_start])
            if duration >= SPRINT_MIN_DURATION_S:
                count += 1
                distance += float(np.nansum(step[run_start:i]))
            run_start = None
    if run_start is not None:
        duration = float(segment.times[-1] - segment.times[run_start])
        if duration >= SPRINT_MIN_DURATION_S:
            count += 1
            distance += float(np.nansum(step[run_start:]))
    return count, distance


def compute_all(
    valid_players: pd.DataFrame,
    all_players: pd.DataFrame,
    fps: float,
    smoothing_window: int = 7,
) -> dict[int, KinematicProfile]:
    """Physical profiles for every track that has any usable rows."""
    totals = all_players.groupby("track_id").size().to_dict()
    profiles: dict[int, KinematicProfile] = {}

    for track_id, rows in valid_players.groupby("track_id"):
        tid = int(track_id)
        profiles[tid] = compute_kinematics(
            tid, rows, int(totals.get(track_id, len(rows))), fps, smoothing_window
        )

    # Tracks with no usable rows still get a profile, so a consumer iterating
    # players finds an explicit zero-coverage entry rather than a missing key.
    for track_id, total in totals.items():
        profiles.setdefault(
            int(track_id), KinematicProfile(track_id=int(track_id), n_rows_total=int(total))
        )

    rejected = sum(p.n_rejected_samples for p in profiles.values())
    if rejected:
        log.info("kinematics: discarded %d physically impossible samples", rejected)
    return profiles
