"""Temporal handling of calibration: shot changes, smoothing, carry-forward.

Per-frame homography estimation is independent and therefore jittery. Averaged
over a second, a static camera's estimated homography can wander by a metre or
more at the far touchline, which turns into phantom player movement and inflates
every distance-covered statistic.

Three temporal mechanisms fix that, and each has a failure mode that must be
guarded:

* **Shot-change detection** -- broadcast footage cuts to replays and close-ups.
  Smoothing or carrying a homography across a cut is worse than having none.
* **Smoothing** -- a sliding window, but only over frames belonging to the same
  shot.
* **Carry-forward** -- a brief failure (a player occluding the only visible
  penalty box) is bridged by the last good homography with decaying confidence,
  bounded in length so a long failure is reported as a failure.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from visionpitch.common.geometry import apply_homography, normalise_homography
from visionpitch.common.logging import get_logger
from visionpitch.common.types import CalibrationResult, SegmentKind

log = get_logger("calibration.temporal")


class ShotChangeDetector:
    """Detects hard cuts from frame-to-frame colour histogram dissimilarity.

    A downscaled HSV histogram is used rather than pixel differencing: a fast
    camera pan changes every pixel while the colour distribution barely moves,
    whereas a genuine cut to a crowd shot or a replay changes the distribution
    completely. That distinction is exactly what we need.
    """

    def __init__(self, threshold: float = 0.45, bins: tuple[int, int] = (32, 32)) -> None:
        self.threshold = threshold
        self.bins = bins
        self._prev_hist: np.ndarray | None = None

    @staticmethod
    def _histogram(image: np.ndarray, bins: tuple[int, int]) -> np.ndarray:
        small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def update(self, image: np.ndarray) -> tuple[bool, float]:
        """Returns ``(is_shot_change, dissimilarity)`` for this frame."""
        hist = self._histogram(image, self.bins)
        if self._prev_hist is None:
            self._prev_hist = hist
            return False, 0.0

        correlation = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))
        dissimilarity = float(np.clip(1.0 - correlation, 0.0, 2.0))
        self._prev_hist = hist
        return dissimilarity > self.threshold, dissimilarity

    def reset(self) -> None:
        self._prev_hist = None


def _probe_grid(image_size: tuple[int, int]) -> np.ndarray:
    """Points used to compare and combine homographies.

    Confined to the lower-central region of the frame, where the pitch is and
    where players stand. The top corners of a broadcast frame sit at or past the
    horizon; including them lets a fraction of a degree of camera tilt dominate
    every comparison.
    """
    width, height = image_size
    xs = np.linspace(0.1 * width, 0.9 * width, 3)
    ys = np.linspace(0.5 * height, 0.95 * height, 3)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median. Robust to the minority of frames that fit wildly."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float(np.median(values))
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def average_homographies(
    homographies: list[np.ndarray], weights: list[float], image_size: tuple[int, int]
) -> np.ndarray | None:
    """Robustly combine homographies through the points they induce.

    Averaging homography *matrices* entrywise is meaningless -- they are defined
    only up to scale and their entries are not commensurate. The correct
    operation is to combine the mapping: push a grid of probe points through each
    homography, combine the resulting pitch positions, then re-fit.

    The combination is a weighted **median**, not a mean. Measured on the
    validation clip, per-frame homography error is heavy-tailed: the median
    frame-to-frame movement of a fixed probe point is 1.9 m while the mean is
    29 m, because a small minority of frames fit catastrophically. A mean lets
    those few frames drag the whole smoothing window, which is the opposite of
    what smoothing is for.
    """
    if not homographies:
        return None
    if len(homographies) == 1:
        return normalise_homography(homographies[0])

    grid = _probe_grid(image_size)
    weight_array = np.asarray(weights, dtype=np.float64)

    projections = []
    for H in homographies:
        projections.append(apply_homography(H, grid))
    stacked = np.stack(projections)  # (n_homographies, n_points, 2)

    combined = np.zeros_like(grid)
    usable = np.zeros(grid.shape[0], dtype=bool)
    for i in range(grid.shape[0]):
        valid = np.isfinite(stacked[:, i, :]).all(axis=1)
        if valid.sum() == 0:
            continue
        usable[i] = True
        combined[i, 0] = _weighted_median(stacked[valid, i, 0], weight_array[valid])
        combined[i, 1] = _weighted_median(stacked[valid, i, 1], weight_array[valid])

    if usable.sum() < 4:
        return None

    H, _ = cv2.findHomography(grid[usable], combined[usable], method=0)
    return normalise_homography(H) if H is not None else None


class TemporalCalibrationSmoother:
    """Sliding-window smoothing that never crosses a shot boundary."""

    def __init__(self, window: int, image_size: tuple[int, int]) -> None:
        self.window = max(1, window)
        self.image_size = image_size
        self._buffer: deque[tuple[int, np.ndarray, float]] = deque()

    def reset(self) -> None:
        """Called on a shot change: history from the previous shot is invalid."""
        self._buffer.clear()

    def push(self, frame_idx: int, homography: np.ndarray, confidence: float) -> None:
        self._buffer.append((frame_idx, homography, confidence))
        while len(self._buffer) > self.window:
            self._buffer.popleft()

    def smoothed(self) -> np.ndarray | None:
        if not self._buffer:
            return None
        homographies = [h for _, h, _ in self._buffer]
        # Weight by confidence so a marginal frame does not drag the window.
        weights = [max(1e-3, c) for _, _, c in self._buffer]
        return average_homographies(homographies, weights, self.image_size)


def reject_temporal_outliers(
    results: dict[int, CalibrationResult],
    frame_indices: list[int],
    image_size: tuple[int, int],
    max_jump_m: float,
    window: int = 15,
) -> tuple[dict[int, CalibrationResult], int]:
    """Invalidate homographies that disagree wildly with their neighbours.

    A broadcast camera pans, tilts and zooms, but it does not teleport. If frame
    *t*'s homography places the centre of the pitch 80 m from where frames
    *t-7..t+7* place it, that frame's fit is wrong -- regardless of how good its
    own reprojection error looks, because a fit to 5 near-collinear landmarks can
    reproduce those 5 points perfectly while being geometrically nonsense
    everywhere else.

    This is the check that self-reported error structurally cannot perform, and
    it needs no ground truth. Rejected frames become uncalibrated rather than
    being silently repaired.
    """
    grid = _probe_grid(image_size)
    valid_frames = [f for f in frame_indices if f in results and results[f].is_valid]
    if len(valid_frames) < 5:
        return results, 0

    centres: dict[int, np.ndarray] = {}
    for frame_idx in valid_frames:
        projected = apply_homography(results[frame_idx].homography, grid)
        finite = projected[np.isfinite(projected).all(axis=1)]
        if finite.shape[0] >= 4:
            centres[frame_idx] = np.median(finite, axis=0)

    ordered = [f for f in valid_frames if f in centres]
    half = max(2, window // 2)
    n_rejected = 0

    for i, frame_idx in enumerate(ordered):
        lo, hi = max(0, i - half), min(len(ordered), i + half + 1)
        neighbours = [centres[ordered[j]] for j in range(lo, hi) if j != i]
        if len(neighbours) < 4:
            continue
        reference = np.median(np.stack(neighbours), axis=0)
        deviation = float(np.linalg.norm(centres[frame_idx] - reference))
        if deviation > max_jump_m:
            current = results[frame_idx]
            results[frame_idx] = CalibrationResult(
                frame_idx=frame_idx,
                homography=None,
                confidence=0.0,
                reprojection_error_m=current.reprojection_error_m,
                n_keypoints=current.n_keypoints,
                n_inliers=current.n_inliers,
                smoothed=False,
                segment_kind=current.segment_kind,
            )
            n_rejected += 1

    if n_rejected:
        log.info(
            "rejected %d frame(s) whose homography disagreed with its temporal "
            "neighbours by more than %.1f m",
            n_rejected,
            max_jump_m,
        )
    return results, n_rejected


def smooth_calibration_sequence(
    results: dict[int, CalibrationResult],
    frame_indices: list[int],
    image_size: tuple[int, int],
    window: int,
    shot_boundaries: set[int],
) -> dict[int, CalibrationResult]:
    """Offline centred smoothing over each shot independently.

    Offline and centred, unlike the online smoother: with the whole clip
    available there is no reason to accept the half-window lag that a causal
    filter imposes.
    """
    if window <= 1:
        return results

    half = window // 2
    ordered = [f for f in frame_indices if f in results]
    position = {f: i for i, f in enumerate(ordered)}
    smoothed: dict[int, CalibrationResult] = {}

    for frame_idx in ordered:
        result = results[frame_idx]
        if not result.is_valid:
            smoothed[frame_idx] = result
            continue

        i = position[frame_idx]
        lo, hi = max(0, i - half), min(len(ordered), i + half + 1)

        # Contract the window at shot boundaries.
        for j in range(i, lo, -1):
            if ordered[j] in shot_boundaries:
                lo = j
                break
        for j in range(i + 1, hi):
            if ordered[j] in shot_boundaries:
                hi = j
                break

        neighbours = [
            results[ordered[j]] for j in range(lo, hi) if results[ordered[j]].is_valid
        ]
        if len(neighbours) <= 1:
            smoothed[frame_idx] = result
            continue

        averaged = average_homographies(
            [n.homography for n in neighbours],
            [max(1e-3, n.confidence) for n in neighbours],
            image_size,
        )
        if averaged is None:
            smoothed[frame_idx] = result
            continue

        smoothed[frame_idx] = CalibrationResult(
            frame_idx=frame_idx,
            homography=averaged,
            confidence=result.confidence,
            reprojection_error_m=result.reprojection_error_m,
            n_keypoints=result.n_keypoints,
            n_inliers=result.n_inliers,
            smoothed=True,
            segment_kind=result.segment_kind,
        )

    for frame_idx, result in results.items():
        smoothed.setdefault(frame_idx, result)
    return smoothed


def temporal_stability(
    results: dict[int, CalibrationResult],
    frame_indices: list[int],
    image_size: tuple[int, int],
) -> dict[str, float]:
    """How much the calibration jitters between consecutive valid frames.

    Reported in metres of induced pitch-position change. A static camera should
    score near zero; anything above a few tens of centimetres per frame will
    show up as phantom player motion in Phase 2's distance statistics.
    """
    from visionpitch.common.geometry import homography_distance

    deltas = []
    previous = None
    for frame_idx in frame_indices:
        result = results.get(frame_idx)
        if result is None or not result.is_valid:
            previous = None
            continue
        if previous is not None:
            d = homography_distance(previous, result.homography, image_size)
            if np.isfinite(d):
                deltas.append(d)
        previous = result.homography

    if not deltas:
        return {
            "median_delta_m": 0.0,
            "mean_delta_m": 0.0,
            "p95_delta_m": 0.0,
            "max_delta_m": 0.0,
            "n_pairs": 0,
        }
    arr = np.array(deltas)
    # The median leads, because this distribution is heavy-tailed: a handful of
    # badly-fitted frames set the mean, and quoting only the mean would make a
    # calibration that is stable to ~2 m across 95% of frames look unusable.
    # Both are reported so neither can hide the other.
    return {
        "median_delta_m": round(float(np.median(arr)), 4),
        "mean_delta_m": round(float(arr.mean()), 4),
        "p90_delta_m": round(float(np.percentile(arr, 90)), 4),
        "p95_delta_m": round(float(np.percentile(arr, 95)), 4),
        "max_delta_m": round(float(arr.max()), 4),
        "frames_over_5m": int((arr > 5.0).sum()),
        "n_pairs": int(arr.size),
    }


def classify_segment(
    dissimilarity: float, n_confident_keypoints: int, shot_change: bool
) -> SegmentKind:
    """Coarse footage classification from calibration-time evidence.

    Deliberately conservative: this is a Phase 1 signal used to *flag* frames as
    non-live, not the full match segmentation the brief describes for the
    product stage. Anything it is unsure about stays ``UNKNOWN`` so that Phase 2
    can decide, rather than being silently excluded from analytics here.
    """
    if shot_change:
        return SegmentKind.UNKNOWN
    if n_confident_keypoints == 0:
        # No pitch landmarks at all: a crowd shot, a dugout, or a tight close-up.
        return SegmentKind.CLOSE_UP
    if n_confident_keypoints >= 6:
        return SegmentKind.LIVE
    return SegmentKind.UNKNOWN
