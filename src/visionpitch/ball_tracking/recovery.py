"""Track-before-detect recovery of sub-threshold ball evidence.

Phase 2D, Part 4.

Why this stage exists
---------------------
The Phase 2D failure audit measured, on the held-out multi-corpus test split,
that of 267 missed balls:

* 173 (64.8%) were behind or inside a player box
* 65 (24.3%) had a detector candidate at confidence 0.001 that the 0.08
  operating threshold discarded
* 21 (7.9%) were low-contrast
* 3 (1.1%) were visible, clean, and simply missed

So the detector is not mostly blind. It is mostly *unlucky or over-cautious*.
Lowering the global threshold recovers some of that, but at a precision cost
paid on every frame including the easy ones. This stage instead spends the
extra sensitivity only where the trajectory says the ball should be, which is a
few hundred pixels out of two million.

The rule that keeps it honest
-----------------------------
**A single frame of weak evidence never becomes an observation.** A recovered
position must be supported by ``min_supporting_frames`` consecutive frames whose
evidence peaks are mutually consistent with a plausible ball trajectory. One
bright blob near the prediction is exactly the false positive the Viterbi
trajectory search was built to reject, and letting this stage inject it would
undo that.

Recovered positions are labelled ``BallStateKind.RECOVERED`` and never merged
into the observed count. Everything downstream can tell them apart, and the
reports do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from visionpitch.common.logging import get_logger

log = get_logger("ball_tracking.recovery")

RECOVERY_SCHEMA_VERSION = "1.0.0"


class RecoveryMethod(str, Enum):
    """How the weak evidence was found. Recorded per candidate."""

    FRAME_DIFFERENCE = "frame_difference"
    OPTICAL_FLOW = "optical_flow"
    APPEARANCE_TEMPLATE = "appearance_template"
    COMBINED = "combined"


@dataclass
class RecoveryConfig:
    """Search and confirmation parameters.

    ``search_radius_px`` is deliberately tight. The point of track-before-detect
    is that the prior is strong; widening the window until it always contains
    the ball also guarantees it contains a distractor.
    """

    #: half-width of the search window around the predicted position
    search_radius_px: float = 48.0
    #: consecutive frames of consistent evidence required to accept a recovery
    min_supporting_frames: int = 3
    #: longest gap this stage will attempt to bridge at all
    max_gap_frames: int = 15
    #: peak response must exceed this multiple of the local median response
    min_response_ratio: float = 3.0
    #: implied speed above this (px/frame) is not a ball, it is a distractor
    max_step_px: float = 60.0
    #: how far a candidate may sit from the predicted position and still count
    max_deviation_px: float = 36.0
    #: recovered positions are capped at this confidence, always below detector
    max_confidence: float = 0.45
    #: candidate blob area bounds, in pixels squared
    min_blob_area_px2: float = 4.0
    max_blob_area_px2: float = 2000.0

    def to_dict(self) -> dict:
        return {
            "search_radius_px": self.search_radius_px,
            "min_supporting_frames": self.min_supporting_frames,
            "max_gap_frames": self.max_gap_frames,
            "min_response_ratio": self.min_response_ratio,
            "max_step_px": self.max_step_px,
            "max_deviation_px": self.max_deviation_px,
            "max_confidence": self.max_confidence,
            "min_blob_area_px2": self.min_blob_area_px2,
            "max_blob_area_px2": self.max_blob_area_px2,
        }


@dataclass
class RecoveryCandidate:
    """One frame's weak evidence, before temporal confirmation."""

    frame_idx: int
    position: tuple[float, float]
    method: RecoveryMethod
    response: float
    response_ratio: float
    deviation_px: float
    blob_area_px2: float

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "position": [round(self.position[0], 2), round(self.position[1], 2)],
            "method": self.method.value,
            "response_ratio": round(self.response_ratio, 3),
            "deviation_px": round(self.deviation_px, 2),
            "blob_area_px2": round(self.blob_area_px2, 1),
        }


@dataclass
class RecoveredObservation:
    """A confirmed recovery, with the evidence that justified it."""

    frame_idx: int
    position: tuple[float, float]
    confidence: float
    method: RecoveryMethod
    supporting_frames: list[int] = field(default_factory=list)
    trajectory_consistency: float = 0.0
    observability_state: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "position": [round(self.position[0], 2), round(self.position[1], 2)],
            "confidence": round(self.confidence, 4),
            "method": self.method.value,
            "supporting_frames": self.supporting_frames,
            "trajectory_consistency": round(self.trajectory_consistency, 4),
            "observability_state": self.observability_state,
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "note": "recovered, not detected; never counted as a direct observation",
        }


class TrackBeforeDetect:
    """Searches for ball evidence the detector was too cautious to report."""

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.cfg = config or RecoveryConfig()

    # -- single-frame evidence ------------------------------------------------ #

    def _search_frame(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        following: np.ndarray | None,
        predicted: tuple[float, float],
    ) -> RecoveryCandidate | None:
        """Strongest motion-difference peak within the search window.

        Three-frame differencing rather than two: a two-frame difference lights
        up both where the object *was* and where it *is*, and on a small fast
        object those two blobs are indistinguishable. Intersecting the
        before-and-after differences leaves only the current position.
        """
        height, width = current.shape[:2]
        radius = int(self.cfg.search_radius_px)
        px, py = predicted
        x1 = int(max(0, px - radius))
        y1 = int(max(0, py - radius))
        x2 = int(min(width, px + radius))
        y2 = int(min(height, py + radius))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        window = current[y1:y2, x1:x2]
        previous_window = previous[y1:y2, x1:x2]
        difference = cv2.absdiff(window, previous_window)
        if following is not None:
            forward = cv2.absdiff(window, following[y1:y2, x1:x2])
            difference = cv2.min(difference, forward)

        difference = cv2.GaussianBlur(difference, (3, 3), 0)
        median = float(np.median(difference))
        peak = float(difference.max())
        # A flat window has no evidence in it; the ratio would be meaningless.
        if peak <= 1.0:
            return None
        ratio = peak / max(1.0, median)
        if ratio < self.cfg.min_response_ratio:
            return None

        _, mask = cv2.threshold(
            difference, max(8.0, peak * 0.6), 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not (self.cfg.min_blob_area_px2 <= area <= self.cfg.max_blob_area_px2):
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            cx = x1 + moments["m10"] / moments["m00"]
            cy = y1 + moments["m01"] / moments["m00"]
            deviation = float(np.hypot(cx - px, cy - py))
            if deviation > self.cfg.max_deviation_px:
                continue
            if best is None or deviation < best[2]:
                best = ((cx, cy), area, deviation)

        if best is None:
            return None
        (cx, cy), area, deviation = best
        return RecoveryCandidate(
            frame_idx=-1, position=(cx, cy), method=RecoveryMethod.FRAME_DIFFERENCE,
            response=peak, response_ratio=ratio, deviation_px=deviation,
            blob_area_px2=area,
        )

    # -- temporal confirmation ------------------------------------------------ #

    def _confirm(self, candidates: list[RecoveryCandidate]) -> list[RecoveredObservation]:
        """Accept only runs of mutually consistent candidates.

        Consistency means each consecutive pair implies a step a ball could
        actually take. A run of blobs jumping around the search window is a
        flickering distractor, however strong each individual response is.
        """
        if len(candidates) < self.cfg.min_supporting_frames:
            return []

        runs: list[list[RecoveryCandidate]] = []
        current: list[RecoveryCandidate] = []
        for candidate in sorted(candidates, key=lambda c: c.frame_idx):
            if not current:
                current = [candidate]
                continue
            previous = current[-1]
            gap = candidate.frame_idx - previous.frame_idx
            step = float(np.hypot(
                candidate.position[0] - previous.position[0],
                candidate.position[1] - previous.position[1],
            ))
            if gap == 1 and step <= self.cfg.max_step_px:
                current.append(candidate)
            else:
                runs.append(current)
                current = [candidate]
        if current:
            runs.append(current)

        confirmed: list[RecoveredObservation] = []
        for run in runs:
            if len(run) < self.cfg.min_supporting_frames:
                continue
            steps = [
                float(np.hypot(
                    b.position[0] - a.position[0], b.position[1] - a.position[1]
                ))
                for a, b in zip(run, run[1:], strict=False)
            ]
            # Consistency: low variability of step size means smooth motion.
            spread = float(np.std(steps)) if len(steps) > 1 else 0.0
            mean_step = float(np.mean(steps)) if steps else 0.0
            consistency = 1.0 / (1.0 + spread / max(1.0, mean_step))

            frames = [c.frame_idx for c in run]
            for candidate in run:
                strength = min(1.0, candidate.response_ratio / (self.cfg.min_response_ratio * 3))
                closeness = 1.0 - min(1.0, candidate.deviation_px / self.cfg.max_deviation_px)
                confidence = min(
                    self.cfg.max_confidence,
                    self.cfg.max_confidence * strength * closeness * consistency,
                )
                confirmed.append(
                    RecoveredObservation(
                        frame_idx=candidate.frame_idx,
                        position=candidate.position,
                        confidence=confidence,
                        method=candidate.method,
                        supporting_frames=frames,
                        trajectory_consistency=consistency,
                    )
                )
        return confirmed

    # -- entry point ---------------------------------------------------------- #

    def recover(
        self,
        gap_frames: list[int],
        frame_accessor,
        predicted_positions: dict[int, tuple[float, float]],
        observability: dict[int, str] | None = None,
    ) -> list[RecoveredObservation]:
        """Attempt recovery over one contiguous gap.

        ``frame_accessor(frame_idx)`` returns a greyscale frame or ``None``.
        ``predicted_positions`` must come from the trajectory estimator, not
        from this module: recovery may not choose where to look based on what it
        has already found, or it will walk itself onto a distractor.

        Frames the observability model calls out-of-frame or off-pitch are
        skipped. There is nothing to recover there, and searching anyway would
        find whatever noise happened to be brightest.
        """
        if not gap_frames or len(gap_frames) > self.cfg.max_gap_frames:
            return []

        skip = {"likely_outside_frame", "not_on_pitch"}
        candidates: list[RecoveryCandidate] = []
        for frame_idx in sorted(gap_frames):
            if observability and observability.get(frame_idx) in skip:
                continue
            predicted = predicted_positions.get(frame_idx)
            if predicted is None:
                continue
            previous = frame_accessor(frame_idx - 1)
            current = frame_accessor(frame_idx)
            following = frame_accessor(frame_idx + 1)
            if previous is None or current is None:
                continue
            candidate = self._search_frame(previous, current, following, predicted)
            if candidate is None:
                continue
            candidate.frame_idx = frame_idx
            candidates.append(candidate)

        confirmed = self._confirm(candidates)
        for observation in confirmed:
            if observability:
                observation.observability_state = observability.get(
                    observation.frame_idx, "unknown"
                )
        if confirmed:
            log.debug(
                "recovered %d/%d gap frame(s) with %d candidate(s)",
                len(confirmed), len(gap_frames), len(candidates),
            )
        return confirmed
