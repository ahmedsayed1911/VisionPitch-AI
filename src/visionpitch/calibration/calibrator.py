"""Calibration stage orchestration.

Runs the keypoint model, solves and validates a homography per frame, tracks
shot changes, bridges brief failures, and finally applies offline centred
smoothing per shot.

The stage never fabricates a calibration. A frame ends in exactly one of three
states, all of them recorded:

* **valid** -- a homography that passed all three validation checks
* **carried** -- the previous shot-consistent homography, reused for a bounded
  number of frames with decayed confidence, flagged ``smoothed``
* **invalid** -- ``homography=None``. Objects in that frame get null pitch
  coordinates and ``validation_status = no_calibration``.
"""

from __future__ import annotations

import numpy as np

from visionpitch.calibration.homography import estimate_homography
from visionpitch.calibration.keypoints import KeypointObservation, PitchKeypointDetector
from visionpitch.calibration.temporal import (
    ShotChangeDetector,
    classify_segment,
    reject_temporal_outliers,
    smooth_calibration_sequence,
    temporal_stability,
)
from visionpitch.common.config import Config
from visionpitch.common.logging import StageCounters, get_logger
from visionpitch.common.types import CalibrationResult, SegmentKind
from visionpitch.pitch.geometry import PitchConfiguration

log = get_logger("calibration")


class Calibrator:
    """Per-frame calibration with temporal consistency."""

    def __init__(
        self, config: Config, pitch: PitchConfiguration, image_size: tuple[int, int]
    ) -> None:
        self.config = config
        self.cfg = config.calibration
        self.pitch = pitch
        self.image_size = image_size
        self.counters = StageCounters("calibration")

        self.detector = PitchKeypointDetector(config, n_expected=pitch.n_vertices)
        self.shot_detector = ShotChangeDetector(threshold=self.cfg.shot_change_threshold)

        self.results: dict[int, CalibrationResult] = {}
        self.shot_boundaries: set[int] = set()
        self.rejections: dict[str, int] = {}
        #: frame -> 2x3 affine mapping the previous processed frame onto it.
        #: Supplied by the pipeline, which already computes it for the tracker.
        self.warps: dict[int, np.ndarray] = {}
        #: frame -> image region the solve was actually constrained over
        self.support: dict[int, tuple[float, float, float, float]] = {}
        self.propagation_report: dict = {}

        self._last_valid: tuple[int, np.ndarray, float] | None = None

    # -- per-frame ---------------------------------------------------------- #

    def _solve(self, observation: KeypointObservation | None, frame_idx: int):
        if observation is None:
            return None, 0

        indices, points, confidences = observation.confident(self.cfg.keypoint_conf_threshold)
        if indices.size < self.cfg.min_keypoints:
            return None, int(indices.size)

        # Record where the evidence was, so projections outside it can be
        # flagged as extrapolation rather than presented at face value.
        from visionpitch.calibration.propagation import support_region

        region = support_region(points, self.image_size)
        if region is not None:
            self.support[frame_idx] = region

        fit = estimate_homography(
            image_points=points,
            pitch_indices=indices,
            pitch=self.pitch,
            image_size=self.image_size,
            keypoint_confidences=confidences,
            ransac_threshold_m=self.cfg.ransac_threshold_m,
            max_reprojection_error_m=self.cfg.max_reprojection_error_m,
            min_keypoints=self.cfg.min_keypoints,
        )
        if fit.rejection_reason:
            self.rejections[fit.rejection_reason] = (
                self.rejections.get(fit.rejection_reason, 0) + 1
            )
        return fit, int(indices.size)

    def process_batch(
        self, images: list[np.ndarray], frame_indices: list[int]
    ) -> list[CalibrationResult]:
        """Calibrate a batch of frames. Shot detection is applied in order."""
        if not self.cfg.enabled:
            return [self._invalid(f, SegmentKind.UNKNOWN) for f in frame_indices]

        shot_flags = []
        for image, frame_idx in zip(images, frame_indices, strict=True):
            is_cut, _ = self.shot_detector.update(image)
            if is_cut:
                self.shot_boundaries.add(frame_idx)
                # A cut invalidates carry-forward: the previous shot's camera
                # pose says nothing about this one.
                self._last_valid = None
            shot_flags.append(is_cut)

        observations = self.detector.detect_batch(images, frame_indices)

        out: list[CalibrationResult] = []
        for observation, frame_idx, is_cut in zip(
            observations, frame_indices, shot_flags, strict=True
        ):
            fit, n_confident = self._solve(observation, frame_idx)
            segment = classify_segment(0.0, n_confident, is_cut)

            if fit is not None and fit.ok and fit.confidence >= self.cfg.min_confidence:
                result = CalibrationResult(
                    frame_idx=frame_idx,
                    homography=fit.homography,
                    confidence=fit.confidence,
                    reprojection_error_m=fit.reprojection_error_m,
                    n_keypoints=fit.n_keypoints,
                    n_inliers=fit.n_inliers,
                    smoothed=False,
                    segment_kind=segment,
                )
                self._last_valid = (frame_idx, fit.homography, fit.confidence)
                self.counters.ok()
            elif fit is not None and fit.ok:
                # Solved, but below the confidence floor. Keep it -- flagged --
                # rather than discard: a weak calibration is still far better
                # than none for coarse zone analytics, provided downstream code
                # can see that it is weak.
                result = CalibrationResult(
                    frame_idx=frame_idx,
                    homography=fit.homography,
                    confidence=fit.confidence,
                    reprojection_error_m=fit.reprojection_error_m,
                    n_keypoints=fit.n_keypoints,
                    n_inliers=fit.n_inliers,
                    smoothed=False,
                    segment_kind=segment,
                )
                self._last_valid = (frame_idx, fit.homography, fit.confidence)
                self.counters.warn("low_confidence_calibration")
            else:
                result = self._carry_or_fail(frame_idx, segment, fit, n_confident)

            self.results[frame_idx] = result
            out.append(result)
        return out

    def _carry_or_fail(
        self, frame_idx: int, segment: SegmentKind, fit, n_confident: int
    ) -> CalibrationResult:
        if self._last_valid is not None and self.cfg.max_carry_forward_frames > 0:
            last_frame, last_H, last_conf = self._last_valid
            age = frame_idx - last_frame
            if 0 < age <= self.cfg.max_carry_forward_frames:
                # Linear decay to zero at the carry limit: by the time we stop
                # carrying, the confidence has already told downstream code not
                # to trust it.
                decayed = last_conf * max(0.0, 1.0 - age / (self.cfg.max_carry_forward_frames + 1))
                self.counters.warn("carried_forward")
                return CalibrationResult(
                    frame_idx=frame_idx,
                    homography=last_H,
                    confidence=float(decayed),
                    reprojection_error_m=float("nan"),
                    n_keypoints=n_confident,
                    n_inliers=0,
                    smoothed=True,
                    segment_kind=segment,
                )

        reason = fit.rejection_reason if fit is not None else "insufficient_keypoints"
        self.counters.fail(reason or "unknown")
        return self._invalid(frame_idx, segment, n_confident)

    @staticmethod
    def _invalid(
        frame_idx: int, segment: SegmentKind, n_keypoints: int = 0
    ) -> CalibrationResult:
        return CalibrationResult(
            frame_idx=frame_idx,
            homography=None,
            confidence=0.0,
            reprojection_error_m=float("nan"),
            n_keypoints=n_keypoints,
            n_inliers=0,
            smoothed=False,
            segment_kind=segment,
        )

    # -- offline pass ------------------------------------------------------- #

    def finalise(self, frame_indices: list[int]) -> dict[int, CalibrationResult]:
        """Reject temporal outliers, then smooth, then fill missing frames."""
        for frame_idx in frame_indices:
            self.results.setdefault(frame_idx, self._invalid(frame_idx, SegmentKind.UNKNOWN))

        # Outlier rejection must run *before* smoothing: smoothing a window that
        # contains a catastrophic fit spreads it across its neighbours.
        self.results, n_rejected = reject_temporal_outliers(
            self.results, frame_indices, self.image_size, self.cfg.max_temporal_jump_m
        )
        if n_rejected:
            self.counters.fail("temporal_outlier", n_rejected)
            self.rejections["temporal_outlier"] = n_rejected

        # Propagation runs after outlier rejection so a bad solve cannot become
        # the anchor for a whole neighbourhood, and before smoothing so the
        # propagated frames participate in it.
        if self.cfg.propagate_from_motion and self.warps:
            from visionpitch.calibration.propagation import propagate_calibration

            self.results, self.propagation_report = propagate_calibration(
                self.results,
                frame_indices,
                self.warps,
                max_propagation_frames=self.cfg.max_propagation_frames,
                confidence_decay_per_frame=self.cfg.propagation_decay_per_frame,
                min_anchor_confidence=self.cfg.min_confidence,
            )

        if self.cfg.temporal_smoothing:
            self.results = smooth_calibration_sequence(
                self.results,
                frame_indices,
                self.image_size,
                self.cfg.smoothing_window,
                self.shot_boundaries,
            )
        return self.results

    # -- reporting ---------------------------------------------------------- #

    def report(self, frame_indices: list[int]) -> dict:
        """Calibration data-quality summary for the manifest and evaluation."""
        total = len(frame_indices)
        valid = [self.results[f] for f in frame_indices if self.results.get(f, None) and
                 self.results[f].is_valid]
        confident = [r for r in valid if r.confidence >= self.cfg.min_confidence]
        errors = np.array(
            [r.reprojection_error_m for r in valid if np.isfinite(r.reprojection_error_m)]
        )

        return {
            "frames": total,
            "valid_frames": len(valid),
            "valid_ratio": round(len(valid) / total, 4) if total else 0.0,
            "confident_frames": len(confident),
            "confident_ratio": round(len(confident) / total, 4) if total else 0.0,
            "mean_reprojection_error_m": (
                round(float(errors.mean()), 4) if errors.size else None
            ),
            "p95_reprojection_error_m": (
                round(float(np.percentile(errors, 95)), 4) if errors.size else None
            ),
            "mean_confidence": (
                round(float(np.mean([r.confidence for r in valid])), 4) if valid else 0.0
            ),
            "shot_changes": len(self.shot_boundaries),
            "rejections": dict(self.rejections),
            "temporal_stability": temporal_stability(
                self.results, frame_indices, self.image_size
            ),
        }
