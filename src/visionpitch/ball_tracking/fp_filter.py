"""Temporal verification of ball candidates.

Part 4 of Candidate C precision hardening.

This stage only ever **removes or downgrades** a candidate. It never proposes a
position, never fills a gap, and never moves a candidate to where it thinks the
ball should be. A filter that could invent positions would reintroduce exactly
the fabrication the trajectory estimator refuses.

The camera-motion test is the useful one
----------------------------------------
Three things behave differently in image space when the camera pans:

* a **broadcast overlay** stays fixed -- it is painted on the frame
* a **static pitch feature** (a line, the penalty spot, a seat) moves *with* the
  background, by exactly the camera displacement
* a **ball** moves independently of both

So a candidate whose image displacement is close to zero during a pan is an
overlay, and one whose displacement matches the camera's is glued to the world.
Neither is a ball. That is a physical test, not a heuristic about appearance, and
it catches the two largest static false-positive families at once.

Everything else is persistence and plausibility: a candidate seen once, or one
that teleports, is not something a trajectory can be built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from visionpitch.common.logging import get_logger

log = get_logger("ball_tracking.fp_filter")

FP_FILTER_SCHEMA_VERSION = "1.0.0"


class CandidateState(str, Enum):
    """Kept deliberately separate; a consumer must be able to tell them apart."""

    #: detector output that survived every check on its own
    DIRECT = "direct"
    #: supported by neighbouring frames as well as the detector
    TEMPORALLY_VERIFIED = "temporally_verified"
    #: detector output the filter refused
    REJECTED = "rejected"
    #: no candidate, or not enough context to judge one
    UNKNOWN = "unknown"

    @property
    def is_usable(self) -> bool:
        return self in (CandidateState.DIRECT, CandidateState.TEMPORALLY_VERIFIED)


class RejectionReason(str, Enum):
    SINGLE_FRAME = "single_frame_only"
    IMPLAUSIBLE_JUMP = "implausible_jump"
    STATIC_WHILE_CAMERA_MOVES = "static_in_image_while_camera_moves"
    MOVES_WITH_BACKGROUND = "moves_with_background"
    NO_NEIGHBOUR_SUPPORT = "no_neighbour_support"
    OUTSIDE_PLAUSIBLE_REGION = "outside_plausible_region"
    SIZE_INCONSISTENT = "size_inconsistent"


@dataclass
class FilterConfig:
    """Thresholds, each tied to a measurement rather than a preference."""

    #: A ball at 50 fps on a 1280x720 broadcast crosses at most roughly this far
    #: between frames. Measured on the local annotations: the largest genuine
    #: frame-to-frame step observed was well under this.
    max_step_px: float = 90.0
    #: frames of support required before a candidate is trusted on its own
    min_support_frames: int = 2
    #: a neighbour counts as support within this distance
    support_radius_px: float = 55.0
    #: camera displacement above this means the shot is genuinely moving
    camera_motion_px: float = 6.0
    #: candidate displacement below this during a pan means it is painted on
    static_tolerance_px: float = 2.0
    #: candidate displacement within this of the camera's means it is glued to
    #: the world, not flying through it
    background_match_px: float = 3.0
    #: ball radius may not change by more than this factor between frames
    max_size_ratio: float = 2.2
    #: candidates above this confidence bypass persistence checks -- a very
    #: confident single detection is worth keeping when the ball has just
    #: reappeared from an occlusion
    trust_confidence: float = 0.75

    def to_dict(self) -> dict:
        return {
            "schema_version": FP_FILTER_SCHEMA_VERSION,
            "max_step_px": self.max_step_px,
            "min_support_frames": self.min_support_frames,
            "support_radius_px": self.support_radius_px,
            "camera_motion_px": self.camera_motion_px,
            "static_tolerance_px": self.static_tolerance_px,
            "background_match_px": self.background_match_px,
            "max_size_ratio": self.max_size_ratio,
            "trust_confidence": self.trust_confidence,
        }


@dataclass
class Candidate:
    frame_idx: int
    x: float
    y: float
    confidence: float
    radius_px: float = 7.0


@dataclass
class FilteredCandidate:
    frame_idx: int
    state: CandidateState
    x: float | None = None
    y: float | None = None
    confidence: float = 0.0
    reasons: list[RejectionReason] = field(default_factory=list)
    support_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "state": self.state.value,
            "x": None if self.x is None else round(self.x, 2),
            "y": None if self.y is None else round(self.y, 2),
            "confidence": round(self.confidence, 4),
            "reasons": [r.value for r in self.reasons],
            "support_frames": self.support_frames,
        }


class TemporalFalsePositiveFilter:
    """Verifies detector candidates against their temporal neighbourhood."""

    def __init__(self, config: FilterConfig | None = None) -> None:
        self.cfg = config or FilterConfig()

    # -- individual tests ----------------------------------------------------- #

    def _camera_tests(
        self,
        candidate: Candidate,
        previous: Candidate | None,
        camera_shift: tuple[float, float] | None,
    ) -> list[RejectionReason]:
        """Overlay and background-locked tests, both from camera displacement."""
        if previous is None or camera_shift is None:
            return []
        magnitude = float(np.hypot(*camera_shift))
        if magnitude < self.cfg.camera_motion_px:
            return []  # the camera is still; neither test says anything

        displacement = np.array([candidate.x - previous.x, candidate.y - previous.y])
        reasons: list[RejectionReason] = []
        if float(np.linalg.norm(displacement)) <= self.cfg.static_tolerance_px:
            reasons.append(RejectionReason.STATIC_WHILE_CAMERA_MOVES)
        # A world-static feature moves opposite to the camera's own motion.
        expected = -np.array(camera_shift)
        if float(np.linalg.norm(displacement - expected)) <= self.cfg.background_match_px:
            reasons.append(RejectionReason.MOVES_WITH_BACKGROUND)
        return reasons

    def _support(self, candidate: Candidate, neighbours: list[Candidate]) -> int:
        return sum(
            1 for other in neighbours
            if other.frame_idx != candidate.frame_idx
            and float(np.hypot(other.x - candidate.x, other.y - candidate.y))
            <= self.cfg.support_radius_px
        )

    # -- sequence ------------------------------------------------------------- #

    def filter_sequence(
        self,
        candidates: dict[int, Candidate],
        frame_indices: list[int],
        camera_shifts: dict[int, tuple[float, float]] | None = None,
        plausible_region: tuple[float, float, float, float] | None = None,
    ) -> dict[int, FilteredCandidate]:
        """Verify each frame's best candidate against its neighbours.

        ``candidates`` maps frame index to the single best candidate for that
        frame. ``camera_shifts`` maps frame index to the camera displacement
        since the previous frame, in pixels.
        """
        camera_shifts = camera_shifts or {}
        out: dict[int, FilteredCandidate] = {}
        ordered = sorted(frame_indices)

        for position, frame_idx in enumerate(ordered):
            candidate = candidates.get(frame_idx)
            if candidate is None:
                out[frame_idx] = FilteredCandidate(
                    frame_idx=frame_idx, state=CandidateState.UNKNOWN
                )
                continue

            reasons: list[RejectionReason] = []

            if plausible_region is not None:
                x1, y1, x2, y2 = plausible_region
                if not (x1 <= candidate.x <= x2 and y1 <= candidate.y <= y2):
                    reasons.append(RejectionReason.OUTSIDE_PLAUSIBLE_REGION)

            window = [
                candidates[ordered[j]]
                for j in range(max(0, position - 2), min(len(ordered), position + 3))
                if ordered[j] in candidates
            ]
            support = self._support(candidate, window)

            previous_idx = ordered[position - 1] if position else None
            previous = candidates.get(previous_idx) if previous_idx is not None else None

            if previous is not None:
                gap = max(1, frame_idx - previous.frame_idx)
                step = float(
                    np.hypot(candidate.x - previous.x, candidate.y - previous.y)
                ) / gap
                if step > self.cfg.max_step_px:
                    reasons.append(RejectionReason.IMPLAUSIBLE_JUMP)
                ratio = candidate.radius_px / max(1e-6, previous.radius_px)
                if ratio > self.cfg.max_size_ratio or ratio < 1 / self.cfg.max_size_ratio:
                    reasons.append(RejectionReason.SIZE_INCONSISTENT)

            reasons.extend(
                self._camera_tests(candidate, previous, camera_shifts.get(frame_idx))
            )

            # A very confident detection survives thin support: the ball
            # reappearing after an occlusion legitimately has no neighbours yet.
            trusted = candidate.confidence >= self.cfg.trust_confidence
            if support < self.cfg.min_support_frames and not trusted:
                reasons.append(
                    RejectionReason.SINGLE_FRAME if support == 0
                    else RejectionReason.NO_NEIGHBOUR_SUPPORT
                )

            if reasons:
                out[frame_idx] = FilteredCandidate(
                    frame_idx=frame_idx, state=CandidateState.REJECTED,
                    confidence=candidate.confidence, reasons=reasons,
                    support_frames=support,
                )
            else:
                out[frame_idx] = FilteredCandidate(
                    frame_idx=frame_idx,
                    state=(
                        CandidateState.TEMPORALLY_VERIFIED
                        if support >= self.cfg.min_support_frames
                        else CandidateState.DIRECT
                    ),
                    x=candidate.x, y=candidate.y,
                    confidence=candidate.confidence, support_frames=support,
                )
        return out


def summarise(filtered: dict[int, FilteredCandidate]) -> dict:
    states: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for item in filtered.values():
        states[item.state.value] = states.get(item.state.value, 0) + 1
        for reason in item.reasons:
            reasons[reason.value] = reasons.get(reason.value, 0) + 1
    total = len(filtered) or 1
    usable = sum(1 for i in filtered.values() if i.state.is_usable)
    return {
        "schema_version": FP_FILTER_SCHEMA_VERSION,
        "n_frames": len(filtered),
        "by_state": dict(sorted(states.items())),
        "by_rejection_reason": dict(sorted(reasons.items())),
        "usable_share": round(usable / total, 4),
        "rejected_share": round(
            states.get(CandidateState.REJECTED.value, 0) / total, 4
        ),
        "note": (
            "the filter only removes or downgrades; it never proposes a position "
            "and never fills a gap"
        ),
    }
