"""Propagating camera calibration across frames that could not be solved.

The problem
-----------
Per-frame homography estimation succeeds only when enough pitch landmarks are
visible. On the validation clip that is ~75% of frames, and only ~60% at a
confidence worth trusting. The remaining frames are not hard because the camera
did something strange — they are hard because the camera happened to be pointing
at a stretch of empty grass with no lines in it. The *camera pose* on those
frames is perfectly well determined by its neighbours.

The mechanism
-------------
Frame-to-frame background motion is already estimated elsewhere in the pipeline
for the tracker's benefit (``tracking.gmc``). That warp is exactly what is
needed here. If ``H_t`` maps image *t* to the pitch, and ``W`` maps image *t*
onto image *t+1*, then a point in frame *t+1* corresponds to ``W^-1 p`` in frame
*t*, so::

    H_{t+1} = H_t @ W^-1

Chaining that from a solved anchor frame fills the gaps. Because each step
compounds the previous step's error, propagation is bounded in length and its
confidence decays, so a long unsolved stretch still ends up uncalibrated rather
than confidently wrong.

Anchors are chosen as the *highest-confidence* solved frames, and propagation
runs outward in both directions, so a gap is filled from whichever side has
better evidence.
"""

from __future__ import annotations

import numpy as np

from visionpitch.common.geometry import normalise_homography
from visionpitch.common.logging import get_logger
from visionpitch.common.types import CalibrationResult

log = get_logger("calibration.propagation")


def affine_to_homography(warp: np.ndarray) -> np.ndarray:
    """Lift a 2x3 affine warp to a 3x3 projective matrix."""
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = np.asarray(warp, dtype=np.float64)
    return H


def propagate_calibration(
    results: dict[int, CalibrationResult],
    frame_indices: list[int],
    warps: dict[int, np.ndarray],
    max_propagation_frames: int = 45,
    confidence_decay_per_frame: float = 0.015,
    min_anchor_confidence: float = 0.45,
    min_output_confidence: float = 0.20,
) -> tuple[dict[int, CalibrationResult], dict]:
    """Fill unsolved frames by chaining motion warps from solved neighbours.

    ``warps[f]`` is the 2x3 affine mapping frame ``f-1`` onto frame ``f``.

    Returns the updated results and a report. Frames that were already solved
    are never overwritten — propagation only ever adds.
    """
    report = {
        "anchors": 0,
        "propagated": 0,
        "skipped_no_anchor": 0,
        "skipped_too_far": 0,
        "skipped_singular_warp": 0,
        "skipped_low_confidence": 0,
    }

    order = list(frame_indices)
    position = {f: i for i, f in enumerate(order)}

    anchors = [
        f
        for f in order
        if f in results and results[f].is_valid and results[f].confidence >= min_anchor_confidence
    ]
    report["anchors"] = len(anchors)
    if not anchors:
        log.warning("no calibration anchors above confidence %.2f", min_anchor_confidence)
        return results, report

    # For each unsolved frame, the nearest anchor on each side.
    anchor_positions = np.array([position[f] for f in anchors])

    #: candidate[frame] = (distance, homography, confidence)
    candidates: dict[int, tuple[int, np.ndarray, float]] = {}

    for direction in (1, -1):
        for anchor in anchors:
            anchor_pos = position[anchor]
            current = normalise_homography(results[anchor].homography)
            confidence = results[anchor].confidence

            for step in range(1, max_propagation_frames + 1):
                pos = anchor_pos + direction * step
                if pos < 0 or pos >= len(order):
                    break
                frame = order[pos]

                # Composing toward frame f uses the warp recorded *for* f when
                # moving forward, and the warp recorded for the previous frame
                # when moving backward.
                warp_frame = frame if direction == 1 else order[pos + 1]
                warp = warps.get(warp_frame)
                if warp is None:
                    break

                W = affine_to_homography(warp)
                try:
                    if direction == 1:
                        current = current @ np.linalg.inv(W)
                    else:
                        current = current @ W
                except np.linalg.LinAlgError:
                    report["skipped_singular_warp"] += 1
                    break

                current = normalise_homography(current)
                confidence = results[anchor].confidence * max(
                    0.0, 1.0 - confidence_decay_per_frame * step
                )
                if confidence < min_output_confidence:
                    break

                # Never overwrite a frame that solved on its own evidence.
                if frame in results and results[frame].is_valid:
                    break

                existing = candidates.get(frame)
                if existing is None or step < existing[0]:
                    candidates[frame] = (step, current.copy(), confidence)

    for frame, (step, homography, confidence) in candidates.items():
        previous = results.get(frame)
        results[frame] = CalibrationResult(
            frame_idx=frame,
            homography=homography,
            confidence=float(confidence),
            reprojection_error_m=float("nan"),
            n_keypoints=previous.n_keypoints if previous else 0,
            n_inliers=0,
            # Flagged as smoothed: this homography was inferred from motion, not
            # measured from landmarks, and nothing downstream should mistake it
            # for a solve.
            smoothed=True,
            segment_kind=previous.segment_kind if previous else None,  # type: ignore[arg-type]
        )
        report["propagated"] += 1
        _ = step

    unsolved = sum(1 for f in order if f not in results or not results[f].is_valid)
    report["still_unsolved"] = unsolved
    log.info(
        "calibration propagation: %d anchors filled %d frame(s); %d still unsolved",
        report["anchors"],
        report["propagated"],
        unsolved,
    )
    _ = anchor_positions
    return results, report


def support_region(
    keypoints: np.ndarray | None, image_size: tuple[int, int], expand: float = 0.35
) -> tuple[float, float, float, float] | None:
    """Image region the homography is actually constrained over.

    A homography fitted to landmarks clustered in one corner of the frame is
    well determined there and extrapolates badly elsewhere — most severely
    toward the horizon, where a fraction of a degree of camera tilt moves the
    projected point by tens of metres. Callers use this to mark projections
    outside the supported region rather than presenting them at face value.
    """
    if keypoints is None or len(keypoints) < 3:
        return None
    points = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    lo = lo - expand * span
    hi = hi + expand * span

    width, height = image_size
    return (
        float(max(0.0, lo[0])),
        float(max(0.0, lo[1])),
        float(min(width, hi[0])),
        float(min(height, hi[1])),
    )


def extrapolation_risk(
    point: tuple[float, float], region: tuple[float, float, float, float] | None
) -> float:
    """0 inside the supported region, growing to 1 well outside it.

    Expressed as a continuous risk rather than a boolean so downstream code can
    choose its own tolerance instead of inheriting one baked in here.

    An unknown region returns 0.0, not a middling value: absence of evidence
    about the support is not evidence of extrapolation, and returning something
    above a caller's threshold would silently mark *every* row extrapolated on
    any frame whose support was not recorded — including propagated frames,
    which are already flagged by their own confidence and ``smoothed`` fields.
    """
    if region is None:
        return 0.0
    x, y = point
    x1, y1, x2, y2 = region
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    dx = max(x1 - x, 0.0, x - x2) / width
    dy = max(y1 - y, 0.0, y - y2) / height
    return float(np.clip(np.hypot(dx, dy), 0.0, 1.0))
