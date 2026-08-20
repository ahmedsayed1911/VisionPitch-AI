"""Production wiring for the temporal ball-fusion layer.

The layer itself lives in :mod:`visionpitch.ball_tracking.fusion` and knows
nothing about the pipeline. This module is the adapter: it translates pipeline
objects into the layer's inputs, translates the layer's output back into the
``Detection`` list the trajectory estimator already consumes, and records what
it did.

Where it sits
-------------
::

    detector candidates  ->  fuse_detections()   [unchanged]
                         ->  THIS MODULE          [opt-in]
                         ->  BallTrajectoryEstimator
                         ->  possession / events  [unchanged]

Nothing downstream changes shape. The estimator receives the same
``dict[int, list[Detection]]`` it always has; fusion only ever *removes* or
*merges* entries in it. That is what makes the four-way comparison a comparison
of fusion and nothing else.

What survives a frame
---------------------
Two distinct operations, and it matters that they are distinct:

* **suppression** collapses duplicate hypotheses. All surviving merged
  candidates are handed on, not just the best one -- the trajectory search is
  a lattice and pruning it to one node per frame would be a different change
  with different effects. The isolated ablation could not measure this because
  it scored top-1 selection; here it reaches the search.
* **temporal verification** judges the frame's *best* candidate. If that
  candidate is rejected, the whole frame is emptied. A frame whose strongest
  evidence is a pitch marking is not improved by keeping its second-strongest.

Camera motion
-------------
Real GMC, from :class:`~visionpitch.tracking.gmc.GlobalMotionCompensator` via
``MultiObjectTracker.motion_warps`` -- the same estimates, the same frame
indices and the same failure cases the person tracker used. No second
estimation pass, and no phase-correlation substitute.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from visionpitch.ball_tracking.fp_filter import FilterConfig
from visionpitch.ball_tracking.fusion import (
    FUSION_SCHEMA_VERSION,
    BallFusion,
    FusionConfig,
    SuppressionMethod,
    summarise,
    suppress,
)
from visionpitch.common.logging import get_logger
from visionpitch.common.types import BBox, Detection, ObjectClass

if TYPE_CHECKING:  # pragma: no cover - typing only
    from visionpitch.common.config import BallFusionConfig

log = get_logger("pipeline.ball_fusion")

#: Written into the manifest so a stored run states which adapter produced it.
BALL_FUSION_STAGE_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def build_fusion_config(cfg: BallFusionConfig) -> FusionConfig:
    """Translate the pydantic run config into the layer's dataclass config.

    The disable flags are expressed as thresholds the corresponding test can
    never fail, rather than as branches inside the layer. One code path runs in
    every configuration, so an ablation row cannot accidentally exercise
    different logic from the row it is being compared against.
    """
    temporal = FilterConfig(
        min_support_frames=cfg.min_support_frames,
        trust_confidence=cfg.trust_confidence,
        camera_motion_px=cfg.camera_motion_px,
    )
    if not cfg.temporal_filter_enabled:
        temporal = FilterConfig(
            min_support_frames=0,
            trust_confidence=0.0,
            max_step_px=1e9,
            camera_motion_px=1e9,
            max_size_ratio=1e9,
        )
    elif not cfg.camera_motion_enabled:
        # Only the two camera-dependent tests are neutralised; persistence,
        # step and size checks stay exactly as they are.
        temporal = FilterConfig(
            min_support_frames=cfg.min_support_frames,
            trust_confidence=cfg.trust_confidence,
            camera_motion_px=1e9,
        )
    return FusionConfig(
        suppression=SuppressionMethod(cfg.suppression),
        merge_radius_px=cfg.merge_radius_px,
        temporal=temporal,
        min_camera_confidence=cfg.min_camera_confidence,
    )


# --------------------------------------------------------------------------- #
# Real camera motion
# --------------------------------------------------------------------------- #


def camera_motion_from_warps(
    warps: dict[int, np.ndarray], width: int, height: int
) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    """Frame-to-frame background displacement from the tracker's GMC warps.

    The warp is a 2x3 partial affine mapping the previous processed frame onto
    this one. The ball occupies a small neighbourhood, so its local displacement
    is well approximated by the warp evaluated at the frame centre; using the
    full affine per candidate would be more precise and is not what the
    person tracker does, and the point of this wiring is that both paths see the
    *same* motion.

    Confidence is **binary, not graded**. ``GlobalMotionCompensator.apply``
    returns exactly identity on the first frame and on every documented failure
    (too few corners, weak RANSAC consensus, ECC divergence) and exposes no
    inlier ratio. So this reports 1.0 when an estimate exists and 0.0 when the
    estimator fell back, which is availability. It is not a quality score and is
    not presented as one.
    """
    shifts: dict[int, tuple[float, float]] = {}
    confidence: dict[int, float] = {}
    centre = np.array([width / 2.0, height / 2.0, 1.0])

    for frame_idx, warp in warps.items():
        matrix = np.asarray(warp, dtype=np.float64)
        if matrix.shape != (2, 3):
            confidence[frame_idx] = 0.0
            continue
        moved = matrix @ centre
        dx = float(moved[0] - centre[0])
        dy = float(moved[1] - centre[1])
        identity = bool(
            np.allclose(matrix, np.eye(2, 3, dtype=np.float64), atol=1e-9)
        )
        shifts[frame_idx] = (dx, dy)
        confidence[frame_idx] = 0.0 if identity else 1.0
    return shifts, confidence


def gmc_provenance(
    warps: dict[int, np.ndarray],
    confidence: dict[int, float],
    frame_indices: list[int],
    method: str,
    downscale: int,
    enabled: bool,
) -> dict:
    """What the manifest needs to say the ball and player paths agree."""
    available = sum(1 for f in frame_indices if confidence.get(f, 0.0) > 0.0)
    return {
        "source": "tracking.gmc.GlobalMotionCompensator",
        "method": method if enabled else "disabled",
        "downscale": downscale,
        "enabled": enabled,
        "shared_with_person_tracker": True,
        "n_warps": len(warps),
        "n_frames": len(frame_indices),
        "frames_with_estimate": available,
        "estimate_ratio": round(available / max(1, len(frame_indices)), 4),
        "confidence_kind": "binary_availability",
        "note": (
            "identical estimates, frame indexing and failure cases as the "
            "person tracker; identity warps are the estimator's documented "
            "fallback and are gated out, never applied as zero motion"
        ),
    }


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


def _to_detection(
    frame_idx: int, x: float, y: float, radius: float, confidence: float, source: str
) -> Detection:
    return Detection(
        frame_idx=frame_idx,
        object_class=ObjectClass.BALL,
        bbox=BBox(x - radius, y - radius, x + radius, y + radius),
        confidence=float(min(1.0, max(0.0, confidence))),
        source=source,
    )


def run_ball_fusion(
    detections: dict[int, list[Detection]],
    frame_indices: list[int],
    cfg: BallFusionConfig,
    camera_shifts: dict[int, tuple[float, float]] | None = None,
    camera_confidence: dict[int, float] | None = None,
    cut_frames: set[int] | None = None,
    observability: dict[int, str] | None = None,
) -> tuple[dict[int, list[Detection]], dict]:
    """Re-derive ball candidates through the temporal fusion layer.

    Returns the replacement candidate map and a report. ``detections`` is not
    mutated -- the caller keeps the legacy candidates, which is what lets the
    same cached detector output be scored under both engines.
    """
    started = time.perf_counter()
    config = build_fusion_config(cfg)

    by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    for frame_idx in frame_indices:
        packed: list[tuple[float, float, float, float]] = []
        for det in detections.get(frame_idx, []):
            if det.object_class is not ObjectClass.BALL:
                continue
            cx, cy = det.bbox.center
            radius = max(det.bbox.width, det.bbox.height) / 2.0
            packed.append((float(cx), float(cy), float(radius), float(det.confidence)))
        by_frame[frame_idx] = packed

    fusion = BallFusion(config)
    frames = fusion.run(
        by_frame,
        frame_indices,
        camera_shifts=camera_shifts if cfg.camera_motion_enabled else None,
        camera_confidence=camera_confidence if cfg.camera_motion_enabled else None,
        cut_frames=cut_frames if cfg.camera_cut_reset_enabled else None,
        observability=observability if cfg.observability_enabled else None,
    )

    out: dict[int, list[Detection]] = {}
    n_in = n_out = n_rejected_frames = n_merged_away = 0
    for frame_idx in frame_indices:
        packed = by_frame[frame_idx]
        n_in += len(packed)
        verdict = frames[frame_idx]
        if not verdict.kind.counts_as_observed:
            # The frame's strongest evidence did not survive verification, or
            # there was none. Either way this frame contributes no candidate.
            out[frame_idx] = []
            if packed:
                n_rejected_frames += 1
            continue

        # Hand on every surviving hypothesis, not only the winner: the
        # trajectory search is what arbitrates between them.
        merged = suppress(packed, config, frame_idx)
        n_merged_away += sum(m.n_merged - 1 for m in merged)
        kept = [
            _to_detection(
                frame_idx, m.x, m.y, m.radius_px, m.confidence,
                f"fusion:{m.merge_method}",
            )
            for m in merged
        ]
        out[frame_idx] = kept
        n_out += len(kept)

    elapsed = time.perf_counter() - started
    report = {
        "stage_version": BALL_FUSION_STAGE_VERSION,
        "schema_version": FUSION_SCHEMA_VERSION,
        "engine": "temporal",
        "config": cfg.model_dump(),
        "config_fingerprint": cfg.fingerprint(),
        "layer_fingerprint": config.fingerprint(),
        "candidates_in": n_in,
        "candidates_out": n_out,
        "candidates_merged_away": n_merged_away,
        "frames_emptied_by_verification": n_rejected_frames,
        "candidates_per_frame_in": round(n_in / max(1, len(frame_indices)), 4),
        "candidates_per_frame_out": round(n_out / max(1, len(frame_indices)), 4),
        "duplicate_candidate_rate": round(
            n_merged_away / max(1, n_in), 4
        ),
        "fusion_ms_per_frame": round(1000 * elapsed / max(1, len(frame_indices)), 4),
        **summarise(frames),
    }
    return out, report


def legacy_report(
    detections: dict[int, list[Detection]], frame_indices: list[int]
) -> dict:
    """The same candidate-level counts for the legacy engine.

    Reported so the four-way comparison has matching columns on both sides
    rather than a populated block against an absent one.
    """
    n_in = sum(
        sum(1 for d in detections.get(f, []) if d.object_class is ObjectClass.BALL)
        for f in frame_indices
    )
    frames_with_any = sum(1 for f in frame_indices if detections.get(f))
    return {
        "stage_version": BALL_FUSION_STAGE_VERSION,
        "engine": "legacy",
        "note": (
            "IoU de-duplication inside detection.fusion.fuse_detections, no "
            "temporal verification; unchanged from before this option existed"
        ),
        "candidates_in": n_in,
        "candidates_out": n_in,
        "candidates_merged_away": 0,
        "frames_emptied_by_verification": 0,
        "candidates_per_frame_in": round(n_in / max(1, len(frame_indices)), 4),
        "candidates_per_frame_out": round(n_in / max(1, len(frame_indices)), 4),
        "frames_with_a_candidate": frames_with_any,
    }
