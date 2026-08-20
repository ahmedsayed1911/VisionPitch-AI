"""Production temporal ball-candidate fusion.

Wires together three things that already existed separately and one that did
not:

* **centre-distance duplicate suppression** (new) -- replacing IoU de-duplication
* the **temporal false-positive filter** (existed, never wired into production)
* **camera-motion compensation** (existed for the person tracker only)
* the **observability model** (existed, used only in reporting)

Why IoU de-duplication was the wrong tool
-----------------------------------------
``detection.fusion._dedupe`` suppresses ball candidates by IoU at 0.5. On an
11 px ball two detections 8 px apart overlap by almost nothing, so IoU keeps both
and calls them distinct hypotheses. The Candidate C audit measured the
consequence directly: **26.3% of all false positives were duplicates** -- the
single largest cause after players, and one that costs no recall to fix.

Centre distance does not degrade with object size in the same way. Two candidates
whose centres are within a ball-and-a-bit of each other are one hypothesis
whatever their boxes say.

What this layer must never do
-----------------------------
It never proposes a position. Merging combines candidates the detector already
produced; verification removes or downgrades. Interpolation stays where it is,
in the trajectory estimator, and remains labelled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from visionpitch.ball_tracking.fp_filter import (
    Candidate,
    CandidateState,
    FilterConfig,
    TemporalFalsePositiveFilter,
)
from visionpitch.common.logging import get_logger

log = get_logger("ball_tracking.fusion")

FUSION_SCHEMA_VERSION = "1.0.0"


class SuppressionMethod(str, Enum):
    """How duplicate candidates are collapsed."""

    NONE = "none"
    IOU = "iou"
    CENTRE_DISTANCE = "centre_distance"
    #: centre clustering plus confidence-weighted centre averaging
    WEIGHTED_CENTRE = "weighted_centre"


class ObservationKind(str, Enum):
    """Deliberately more granular than ``BallStateKind``.

    A consumer must be able to tell a detector sighting from one that only
    survived because its neighbours agreed, and both from an interpolation.
    """

    OBSERVED_DIRECT = "observed_direct"
    OBSERVED_TEMPORALLY_VERIFIED = "observed_temporally_verified"
    RECOVERED_WEAK_EVIDENCE = "recovered_weak_evidence"
    INTERPOLATED = "interpolated"
    OCCLUDED = "occluded"
    OUTSIDE_FRAME = "outside_frame"
    UNKNOWN = "unknown"

    @property
    def is_direct(self) -> bool:
        """Only a detector sighting that passed verification on its own."""
        return self is ObservationKind.OBSERVED_DIRECT

    @property
    def has_position(self) -> bool:
        return self in (
            ObservationKind.OBSERVED_DIRECT,
            ObservationKind.OBSERVED_TEMPORALLY_VERIFIED,
            ObservationKind.RECOVERED_WEAK_EVIDENCE,
            ObservationKind.INTERPOLATED,
        )

    @property
    def counts_as_observed(self) -> bool:
        """Direct or temporally verified -- image evidence in this frame."""
        return self in (
            ObservationKind.OBSERVED_DIRECT,
            ObservationKind.OBSERVED_TEMPORALLY_VERIFIED,
        )


@dataclass
class FusionConfig:
    """Every knob, so a mode can be fingerprinted."""

    suppression: SuppressionMethod = SuppressionMethod.WEIGHTED_CENTRE
    #: Candidates whose centres are within this are one hypothesis. Sized from
    #: the measured ball: median 14 px across on this broadcast, so ~1.5 ball
    #: diameters tolerates localisation jitter without merging a ball with a
    #: nearby sock.
    merge_radius_px: float = 22.0
    iou_threshold: float = 0.5
    #: candidates kept per frame after suppression
    max_candidates_per_frame: int = 3
    temporal: FilterConfig = field(default_factory=FilterConfig)
    #: apply camera compensation only when the estimate is trustworthy
    min_camera_confidence: float = 0.35
    enabled: bool = True

    def to_dict(self) -> dict:
        payload = {
            "schema_version": FUSION_SCHEMA_VERSION,
            "suppression": self.suppression.value,
            "merge_radius_px": self.merge_radius_px,
            "iou_threshold": self.iou_threshold,
            "max_candidates_per_frame": self.max_candidates_per_frame,
            "min_camera_confidence": self.min_camera_confidence,
            "enabled": self.enabled,
            "temporal": self.temporal.to_dict(),
        }
        return payload

    def fingerprint(self) -> str:
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]


@dataclass
class MergedCandidate:
    """One physical ball hypothesis, with the detections behind it."""

    frame_idx: int
    x: float
    y: float
    radius_px: float
    confidence: float
    source_ids: list[int] = field(default_factory=list)
    merge_method: str = SuppressionMethod.NONE.value
    uncertainty_px: float = 0.0
    n_merged: int = 1

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["x"] = round(self.x, 2)
        payload["y"] = round(self.y, 2)
        payload["confidence"] = round(self.confidence, 4)
        payload["uncertainty_px"] = round(self.uncertainty_px, 2)
        return payload


# --------------------------------------------------------------------------- #
# Duplicate suppression
# --------------------------------------------------------------------------- #


def suppress(
    detections: list[tuple[float, float, float, float]],
    config: FusionConfig,
    frame_idx: int = 0,
) -> list[MergedCandidate]:
    """Collapse duplicates. ``detections`` is ``(x, y, radius, confidence)``.

    Provenance is retained: every merged candidate records which input indices
    it came from, how they were combined, and how far apart they were.
    """
    if not detections:
        return []

    order = sorted(range(len(detections)), key=lambda i: -detections[i][3])
    if config.suppression is SuppressionMethod.NONE:
        return [
            MergedCandidate(
                frame_idx=frame_idx, x=detections[i][0], y=detections[i][1],
                radius_px=detections[i][2], confidence=detections[i][3],
                source_ids=[i], merge_method=SuppressionMethod.NONE.value,
            )
            for i in order
        ][: config.max_candidates_per_frame]

    clusters: list[list[int]] = []
    for index in order:
        x, y, radius, _ = detections[index]
        joined = False
        for cluster in clusters:
            hx, hy, hr, _ = detections[cluster[0]]
            if config.suppression is SuppressionMethod.IOU:
                # Box IoU on the equivalent squares.
                ax1, ay1, ax2, ay2 = x - radius, y - radius, x + radius, y + radius
                bx1, by1, bx2, by2 = hx - hr, hy - hr, hx + hr, hy + hr
                iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                ih = max(0.0, min(ay2, by2) - max(ay1, by1))
                inter = iw * ih
                union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
                close = union > 0 and inter / union > config.iou_threshold
            else:
                close = float(np.hypot(x - hx, y - hy)) <= config.merge_radius_px
            if close:
                cluster.append(index)
                joined = True
                break
        if not joined:
            clusters.append([index])

    merged: list[MergedCandidate] = []
    for cluster in clusters:
        members = [detections[i] for i in cluster]
        weights = np.array([m[3] for m in members], dtype=float)
        xs = np.array([m[0] for m in members])
        ys = np.array([m[1] for m in members])
        radii = np.array([m[2] for m in members])

        if (
            config.suppression is SuppressionMethod.WEIGHTED_CENTRE
            and weights.sum() > 0
        ):
            cx = float((xs * weights).sum() / weights.sum())
            cy = float((ys * weights).sum() / weights.sum())
            radius = float((radii * weights).sum() / weights.sum())
        else:
            # Highest-confidence member wins outright.
            cx, cy, radius = float(xs[0]), float(ys[0]), float(radii[0])

        # Confidence is the strongest member's, not the sum. Two weak
        # detections of the same blob are not strong evidence -- they are one
        # weak detection seen twice.
        confidence = float(weights.max())
        spread = (
            float(np.hypot(xs - cx, ys - cy).max()) if len(members) > 1 else 0.0
        )
        merged.append(MergedCandidate(
            frame_idx=frame_idx, x=cx, y=cy, radius_px=radius,
            confidence=confidence, source_ids=sorted(cluster),
            merge_method=config.suppression.value,
            uncertainty_px=max(spread, radius / 2.0), n_merged=len(members),
        ))

    merged.sort(key=lambda m: -m.confidence)
    return merged[: config.max_candidates_per_frame]


# --------------------------------------------------------------------------- #
# Camera compensation
# --------------------------------------------------------------------------- #


def stabilise(
    candidates: dict[int, MergedCandidate],
    camera_shifts: dict[int, tuple[float, float]],
    camera_confidence: dict[int, float] | None = None,
    min_confidence: float = 0.35,
    cut_frames: set[int] | None = None,
) -> tuple[dict[int, tuple[float, float]], dict[int, bool]]:
    """Cumulative camera displacement, so candidates can be compared in a
    stabilised frame.

    Returns the stabilised coordinates and a per-frame flag saying whether the
    compensation was trusted. **Image coordinates are never overwritten** -- the
    stabilised view is an additional coordinate system, because every consumer
    downstream works in image space.
    """
    camera_confidence = camera_confidence or {}
    cuts = cut_frames or set()
    stabilised: dict[int, tuple[float, float]] = {}
    trusted: dict[int, bool] = {}

    offset = np.zeros(2)
    for frame_idx in sorted(candidates):
        if frame_idx in cuts:
            # A cut invalidates the accumulated offset entirely.
            offset = np.zeros(2)
        shift = camera_shifts.get(frame_idx)
        confidence = camera_confidence.get(frame_idx, 1.0 if shift else 0.0)
        ok = shift is not None and confidence >= min_confidence
        if ok:
            offset = offset + np.array(shift)
        trusted[frame_idx] = ok
        candidate = candidates[frame_idx]
        stabilised[frame_idx] = (
            float(candidate.x + offset[0]), float(candidate.y + offset[1])
        )
    return stabilised, trusted


# --------------------------------------------------------------------------- #
# The layer
# --------------------------------------------------------------------------- #


@dataclass
class FusedFrame:
    frame_idx: int
    kind: ObservationKind
    x: float | None = None
    y: float | None = None
    radius_px: float | None = None
    detector_confidence: float = 0.0
    temporal_confidence: float = 0.0
    fusion_confidence: float = 0.0
    source_ids: list[int] = field(default_factory=list)
    merge_method: str = ""
    n_merged: int = 0
    uncertainty_px: float = 0.0
    camera_confidence: float = 0.0
    trajectory_id: int | None = None
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        for key in ("x", "y", "radius_px"):
            if payload[key] is not None:
                payload[key] = round(payload[key], 2)
        for key in ("detector_confidence", "temporal_confidence",
                    "fusion_confidence", "uncertainty_px", "camera_confidence"):
            payload[key] = round(payload[key], 4)
        return payload


class BallFusion:
    """Suppression, then camera-aware temporal verification."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.cfg = config or FusionConfig()
        self.filter = TemporalFalsePositiveFilter(self.cfg.temporal)

    def run(
        self,
        detections_by_frame: dict[int, list[tuple[float, float, float, float]]],
        frame_indices: list[int],
        camera_shifts: dict[int, tuple[float, float]] | None = None,
        camera_confidence: dict[int, float] | None = None,
        cut_frames: set[int] | None = None,
        observability: dict[int, str] | None = None,
        plausible_region: tuple[float, float, float, float] | None = None,
    ) -> dict[int, FusedFrame]:
        camera_shifts = camera_shifts or {}
        camera_confidence = camera_confidence or {}
        observability = observability or {}

        # -- 1. suppression --------------------------------------------------- #
        best: dict[int, MergedCandidate] = {}
        merged_all: dict[int, list[MergedCandidate]] = {}
        for frame_idx in frame_indices:
            merged = suppress(
                detections_by_frame.get(frame_idx, []), self.cfg, frame_idx
            )
            merged_all[frame_idx] = merged
            if merged:
                best[frame_idx] = merged[0]

        # -- 2. camera compensation ------------------------------------------- #
        _, trusted = stabilise(
            best, camera_shifts, camera_confidence,
            self.cfg.min_camera_confidence, cut_frames,
        )

        # -- 3. temporal verification ----------------------------------------- #
        usable_shifts = {
            frame_idx: shift for frame_idx, shift in camera_shifts.items()
            if trusted.get(frame_idx, False)
        }
        filtered = self.filter.filter_sequence(
            {
                frame_idx: Candidate(
                    frame_idx=frame_idx, x=m.x, y=m.y,
                    confidence=m.confidence, radius_px=m.radius_px,
                )
                for frame_idx, m in best.items()
            },
            frame_indices,
            camera_shifts=usable_shifts,
            plausible_region=plausible_region,
        )

        # -- 4. assemble ------------------------------------------------------- #
        out: dict[int, FusedFrame] = {}
        for frame_idx in frame_indices:
            verdict = filtered[frame_idx]
            merged = best.get(frame_idx)
            observable = observability.get(frame_idx, "")

            if merged is None:
                kind = (
                    ObservationKind.OUTSIDE_FRAME
                    if observable == "likely_outside_frame"
                    else ObservationKind.OCCLUDED
                    if observable in (
                        "likely_occluded", "likely_hidden_by_players"
                    )
                    else ObservationKind.UNKNOWN
                )
                out[frame_idx] = FusedFrame(
                    frame_idx=frame_idx, kind=kind,
                    camera_confidence=camera_confidence.get(frame_idx, 0.0),
                )
                continue

            support = verdict.support_frames
            # Temporal confidence: how much the neighbourhood agrees, saturating
            # at the support the filter requires.
            temporal = min(
                1.0, support / max(1, self.cfg.temporal.min_support_frames)
            )
            if verdict.state is CandidateState.REJECTED:
                out[frame_idx] = FusedFrame(
                    frame_idx=frame_idx, kind=ObservationKind.UNKNOWN,
                    detector_confidence=merged.confidence,
                    temporal_confidence=temporal, fusion_confidence=0.0,
                    source_ids=merged.source_ids, merge_method=merged.merge_method,
                    n_merged=merged.n_merged,
                    camera_confidence=camera_confidence.get(frame_idx, 0.0),
                    rejection_reasons=[r.value for r in verdict.reasons],
                )
                continue

            kind = (
                ObservationKind.OBSERVED_TEMPORALLY_VERIFIED
                if verdict.state is CandidateState.TEMPORALLY_VERIFIED
                else ObservationKind.OBSERVED_DIRECT
            )
            out[frame_idx] = FusedFrame(
                frame_idx=frame_idx, kind=kind, x=merged.x, y=merged.y,
                radius_px=merged.radius_px,
                detector_confidence=merged.confidence,
                temporal_confidence=temporal,
                # Fusion confidence combines both, and is never higher than the
                # detector's own: temporal agreement corroborates evidence, it
                # does not create it.
                fusion_confidence=merged.confidence * (0.6 + 0.4 * temporal),
                source_ids=merged.source_ids, merge_method=merged.merge_method,
                n_merged=merged.n_merged, uncertainty_px=merged.uncertainty_px,
                camera_confidence=camera_confidence.get(frame_idx, 0.0),
            )
        return out


def summarise(frames: dict[int, FusedFrame]) -> dict:
    kinds: dict[str, int] = {}
    reasons: dict[str, int] = {}
    merged_counts: list[int] = []
    for frame in frames.values():
        kinds[frame.kind.value] = kinds.get(frame.kind.value, 0) + 1
        for reason in frame.rejection_reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
        if frame.n_merged:
            merged_counts.append(frame.n_merged)
    total = len(frames) or 1
    observed = sum(1 for f in frames.values() if f.kind.counts_as_observed)
    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "n_frames": len(frames),
        "by_kind": dict(sorted(kinds.items())),
        "by_rejection_reason": dict(sorted(reasons.items())),
        "observed_coverage": round(observed / total, 4),
        "direct_coverage": round(
            sum(1 for f in frames.values() if f.kind.is_direct) / total, 4
        ),
        "mean_merged_per_frame": (
            round(float(np.mean(merged_counts)), 3) if merged_counts else 0.0
        ),
        "duplicate_rate": round(
            float(np.mean([c > 1 for c in merged_counts])), 4
        ) if merged_counts else 0.0,
    }
