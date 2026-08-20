"""Calibration metrics.

Two kinds of number, and the distinction matters:

**Self-reported** -- coverage, confidence, temporal stability. Available on any
clip with no annotation at all, and enough to catch a run where calibration
mostly failed.

**Ground-truthed** -- reprojection error against *manually marked* landmarks,
and pitch-position error in metres. These require annotation, and only these can
tell you the homography is systematically wrong rather than merely unstable. A
homography fitted to its own keypoints will always report a small error against
those same keypoints; measuring against independently marked points is the only
way to detect a consistent bias, such as a mis-ordered landmark set.
"""

from __future__ import annotations

import numpy as np

from visionpitch.common.geometry import apply_homography, reprojection_errors
from visionpitch.common.logging import get_logger
from visionpitch.common.types import CalibrationResult
from visionpitch.evaluation.ground_truth import GroundTruth
from visionpitch.pitch.geometry import PitchConfiguration

log = get_logger("evaluation.calibration")


def evaluate_calibration(
    results: dict[int, CalibrationResult],
    frame_indices: list[int],
    pitch: PitchConfiguration,
    image_size: tuple[int, int],
    ground_truth: GroundTruth | None = None,
    min_confidence: float = 0.4,
) -> dict:
    """Coverage, stability, and -- where annotated -- true reprojection error."""
    from visionpitch.calibration.temporal import temporal_stability

    total = len(frame_indices)
    valid = [results[f] for f in frame_indices if f in results and results[f].is_valid]
    confident = [r for r in valid if r.confidence >= min_confidence]
    self_errors = np.array(
        [r.reprojection_error_m for r in valid if np.isfinite(r.reprojection_error_m)]
    )

    report: dict = {
        "frames": total,
        "valid_frames": len(valid),
        "valid_frame_percentage": round(100 * len(valid) / total, 2) if total else 0.0,
        "confident_frames": len(confident),
        "confident_frame_percentage": (
            round(100 * len(confident) / total, 2) if total else 0.0
        ),
        "mean_confidence": (
            round(float(np.mean([r.confidence for r in valid])), 4) if valid else 0.0
        ),
        "self_reported_reprojection_error_m": {
            "mean": round(float(self_errors.mean()), 4) if self_errors.size else None,
            "median": round(float(np.median(self_errors)), 4) if self_errors.size else None,
            "p95": round(float(np.percentile(self_errors, 95)), 4) if self_errors.size else None,
        },
        "temporal_stability": temporal_stability(results, frame_indices, image_size),
        "smoothed_frames": sum(1 for r in valid if r.smoothed),
    }

    if ground_truth is None or not ground_truth.calibration:
        report["ground_truth_available"] = False
        report["note"] = (
            "no manually marked landmarks: reprojection error is self-reported "
            "against the model's own keypoints and cannot detect a systematic bias"
        )
        return report

    # -- measured against independent annotation ----------------------------- #
    per_frame_errors: list[float] = []
    all_point_errors: list[float] = []
    n_evaluated = 0
    n_uncalibrated = 0

    for frame_idx, (points, indices) in ground_truth.calibration.items():
        result = results.get(frame_idx)
        if result is None or not result.is_valid:
            n_uncalibrated += 1
            continue
        world = pitch.vertices[indices]
        errors = reprojection_errors(result.homography, points, world)
        finite = errors[np.isfinite(errors)]
        if finite.size == 0:
            continue
        per_frame_errors.append(float(finite.mean()))
        all_point_errors.extend(finite.tolist())
        n_evaluated += 1

    errors_arr = np.array(all_point_errors)
    report["ground_truth_available"] = True
    report["ground_truth_frames"] = len(ground_truth.calibration)
    report["ground_truth_frames_evaluated"] = n_evaluated
    report["ground_truth_frames_uncalibrated"] = n_uncalibrated
    report["pitch_position_error_m"] = {
        "mean": round(float(errors_arr.mean()), 4) if errors_arr.size else None,
        "median": round(float(np.median(errors_arr)), 4) if errors_arr.size else None,
        "p95": round(float(np.percentile(errors_arr, 95)), 4) if errors_arr.size else None,
        "max": round(float(errors_arr.max()), 4) if errors_arr.size else None,
        "n_points": int(errors_arr.size),
    }

    # A large *median* error means a systematic problem -- most likely a
    # landmark-ordering mismatch between the model and the pitch model -- rather
    # than noise. Say so explicitly; it is the failure that silently invalidates
    # every physical metric downstream.
    median = report["pitch_position_error_m"]["median"]
    if median is not None and median > 5.0:
        report["diagnosis"] = (
            f"median pitch-position error of {median:.1f}m is far too large to be "
            f"noise. The most likely cause is a mismatch between the keypoint "
            f"model's landmark ordering and PitchConfiguration.vertices."
        )
        log.error(report["diagnosis"])

    return report


def measure_position_error(
    homography: np.ndarray,
    image_points: np.ndarray,
    expected_pitch_points: np.ndarray,
) -> dict:
    """Error of projecting known image points to known pitch positions."""
    projected = apply_homography(homography, image_points)
    valid = np.isfinite(projected).all(axis=1)
    if not valid.any():
        return {"n": 0, "mean_m": None, "max_m": None}
    errors = np.linalg.norm(projected[valid] - expected_pitch_points[valid], axis=1)
    return {
        "n": int(valid.sum()),
        "mean_m": round(float(errors.mean()), 4),
        "median_m": round(float(np.median(errors)), 4),
        "max_m": round(float(errors.max()), 4),
    }
