"""Core analytics types.

The Phase 1B measurements imposed constraints that this module encodes in the
*type system* rather than in documentation:

* the ball is observed, interpolated or unknown -- never silently one of them
* every metric carries the coverage it was computed over
* physical statistics may only use rows whose validation status is ``valid``

A convention would be violated within a week. A type cannot be serialised
without its coverage, so the constraint survives contact with future code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BallStateKind(str, Enum):
    """How the ball's position for a frame was arrived at.

    Phase 1B measured direct ball observation at 60.2% of frames, bounded
    interpolation at a further 27.9%, and 11.9% genuinely unknown. Every
    ball-dependent record carries which of these it rests on, so a consumer can
    never mistake an inference for a sighting.
    """

    OBSERVED = "observed"
    #: Phase 2D. Found by track-before-detect search along the predicted path,
    #: from evidence too weak for the detector to have reported on its own, and
    #: confirmed across several frames. Stronger than an interpolation -- there
    #: is real image evidence at this location -- but weaker than a detection,
    #: and never to be counted as one.
    RECOVERED = "recovered"
    INTERPOLATED = "interpolated"
    UNKNOWN = "unknown"

    @property
    def is_known(self) -> bool:
        return self is not BallStateKind.UNKNOWN

    @property
    def is_direct(self) -> bool:
        """Whether the detector reported this position itself.

        Use this, not ``is_known``, whenever the question is "did we actually
        see the ball" -- for reporting coverage, for weighting confidence, and
        for deciding whether a possession call rests on evidence or on inference.
        """
        return self is BallStateKind.OBSERVED

    @property
    def has_image_evidence(self) -> bool:
        """Whether some pixel evidence supports this position.

        True for observed and recovered, false for interpolated: an
        interpolation is a statement about physics between two sightings, not
        about anything visible in this frame.
        """
        return self in (BallStateKind.OBSERVED, BallStateKind.RECOVERED)


#: Team labels that name no team.
#:
#: Defined once, as a set of *sentinels*, rather than by listing the real ids.
#: The pipeline's classifier emits "A" and "B", and several decision rules used
#: to test ``team_id in ("A", "B")`` directly. That silently discarded every
#: result whenever the same code ran against a corpus labelling teams
#: "left"/"right" -- possession contests were never detected, and passes were
#: never separated from turnovers -- with no error and no warning. Anything not
#: named here is a team.
UNASSIGNED_TEAMS = frozenset({"", "unknown", "none", "contested"})


def is_team(team_id: str | None) -> bool:
    """Whether ``team_id`` identifies one of the two sides."""
    return bool(team_id) and team_id not in UNASSIGNED_TEAMS


class PossessionState(str, Enum):
    """Temporal possession state.

    ``UNKNOWN`` is first-class and common: it is what the engine reports when
    the ball's whereabouts are unknown, and it must never be silently folded
    into whichever team held the ball last.
    """

    CONTROLLED = "controlled"
    CONTESTED = "contested"
    LOOSE_BALL = "loose_ball"
    UNKNOWN = "unknown"
    OUT_OF_PLAY = "out_of_play"


class EventType(str, Enum):
    """Football events detected in Phase 2."""

    BALL_TOUCH = "ball_touch"
    PASS = "pass"
    PASS_SUCCESSFUL = "pass_successful"
    PASS_FAILED = "pass_failed"
    PASS_PROGRESSIVE = "pass_progressive"
    PASS_LONG = "pass_long"
    PASS_BACK = "pass_back"
    CROSS = "cross"
    CARRY = "carry"
    DRIBBLE_CANDIDATE = "dribble_candidate"
    SHOT = "shot"
    SHOT_ON_TARGET = "shot_on_target"
    GOAL_CANDIDATE = "goal_candidate"
    INTERCEPTION = "interception"
    RECOVERY = "recovery"
    CLEARANCE = "clearance"
    TURNOVER = "turnover"
    BALL_OUT = "ball_out"
    RESTART = "restart"
    LOOSE_BALL = "loose_ball"
    CONTESTED_POSSESSION = "contested_possession"


class Confidence(str, Enum):
    """Coarse band for filtering. The numeric confidence is always kept too."""

    HIGH = "high"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"

    @staticmethod
    def band(value: float) -> Confidence:
        if value >= 0.70:
            return Confidence.HIGH
        if value >= 0.45:
            return Confidence.PROBABLE
        return Confidence.UNCERTAIN


class MetricBasis(str, Enum):
    """Which rows a metric was computed from.

    Phase 1B constraint 4: physical statistics must not use rows whose
    validation status is anything but ``valid``. Constraint 6 permits
    team-level metrics to use extrapolated positions *if clearly labelled* --
    this is that label.
    """

    #: only rows with validation_status == 'valid'
    VALID_ONLY = "valid_only"
    #: valid plus extrapolated; permitted for team aggregates, labelled as such
    INCLUDES_EXTRAPOLATED = "includes_extrapolated"
    #: derived from events rather than per-frame geometry
    EVENT_DERIVED = "event_derived"
    #: counted from raw tracking without needing pitch coordinates
    IMAGE_SPACE = "image_space"


@dataclass(slots=True, frozen=True)
class Metric:
    """A number that cannot be quoted without knowing what it rests on.

    ``coverage`` is the fraction of the relevant population that contributed.
    For a player's distance covered it is the share of that player's tracked
    frames that were usable; for a team's possession it is the share of match
    time in which the ball's location was known.

    A metric with ``coverage`` of 0.1 is not wrong -- it is a measurement over
    a tenth of the data, and the consumer needs to know that before comparing
    it with anything.
    """

    value: float | int | None
    coverage: float = 0.0
    confidence: float = 0.0
    n_samples: int = 0
    basis: MetricBasis = MetricBasis.VALID_ONLY
    unit: str = ""

    @property
    def is_reportable(self) -> bool:
        """Whether there is enough behind this number to show it at all."""
        return self.value is not None and self.n_samples > 0 and self.coverage > 0.0

    @property
    def band(self) -> Confidence:
        return Confidence.band(self.confidence)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "coverage": round(self.coverage, 4),
            "confidence": round(self.confidence, 4),
            "n_samples": self.n_samples,
            "basis": self.basis.value,
            "unit": self.unit,
            "reportable": self.is_reportable,
        }

    @staticmethod
    def unavailable(unit: str = "", basis: MetricBasis = MetricBasis.VALID_ONLY) -> Metric:
        """A metric that could not be computed. Explicitly not zero.

        Zero and 'no data' are different facts, and collapsing them is how a
        player who was never tracked ends up reported as having run 0 metres.
        """
        return Metric(value=None, coverage=0.0, confidence=0.0, n_samples=0,
                      basis=basis, unit=unit)


@dataclass(slots=True)
class CoverageProfile:
    """The four coverages Phase 1B constraint 5 requires on every player metric."""

    #: share of the analysis window in which this track existed at all
    tracking: float = 0.0
    #: share of this track's frames with usable pitch coordinates
    pitch: float = 0.0
    #: share of this track's frames in which the ball's position was known
    ball: float = 0.0
    #: track-level team/identity confidence carried from Phase 1
    identity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tracking": round(self.tracking, 4),
            "pitch": round(self.pitch, 4),
            "ball": round(self.ball, 4),
            "identity": round(self.identity, 4),
        }

    @property
    def worst(self) -> float:
        return min(self.tracking, self.pitch, self.ball, self.identity)


@dataclass
class Evidence:
    """Why the engine believes an event happened.

    Events are model output, not observations. Carrying the evidence makes a
    reviewer able to disagree with a specific number rather than with the
    verdict as a whole.
    """

    reasons: list[str] = field(default_factory=list)
    measurements: dict[str, float] = field(default_factory=dict)

    def add(self, reason: str, **measurements: float) -> Evidence:
        self.reasons.append(reason)
        self.measurements.update(measurements)
        return self

    def to_dict(self) -> dict:
        return {
            "reasons": list(self.reasons),
            "measurements": {k: round(float(v), 4) for k, v in self.measurements.items()},
        }


@dataclass(slots=True)
class ClipReference:
    """Where in the video an event can be seen.

    Every event carries one so a dashboard can seek without recomputing
    anything, and so an analyst reviewing a disputed event lands on the right
    frame rather than approximately near it.
    """

    frame_start: int
    frame_end: int
    time_start_s: float
    time_end_s: float

    def to_dict(self) -> dict:
        return {
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "time_start_s": round(self.time_start_s, 3),
            "time_end_s": round(self.time_end_s, 3),
        }


@dataclass
class FootballEvent:
    """One detected football event."""

    event_id: str
    event_type: EventType
    frame_idx: int
    timestamp_s: float
    team_id: str
    #: primary actor
    track_id: int | None = None
    player_name: str = ""
    #: secondary actor: pass recipient, dribbled opponent, tackler
    related_track_id: int | None = None
    related_player_name: str = ""
    related_team_id: str = ""

    confidence: float = 0.0
    #: what fraction of the frames this event was inferred from had a known ball
    ball_coverage: float = 0.0
    ball_state: BallStateKind = BallStateKind.UNKNOWN

    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    distance_m: float | None = None
    duration_s: float | None = None

    evidence: Evidence = field(default_factory=Evidence)
    clip: ClipReference | None = None

    @property
    def band(self) -> Confidence:
        return Confidence.band(self.confidence)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 3),
            "team_id": self.team_id,
            "track_id": self.track_id,
            "player_name": self.player_name,
            "related_track_id": self.related_track_id,
            "related_player_name": self.related_player_name,
            "related_team_id": self.related_team_id,
            "confidence": round(self.confidence, 4),
            "confidence_band": self.band.value,
            "ball_coverage": round(self.ball_coverage, 4),
            "ball_state": self.ball_state.value,
            "start_x": self.start_x,
            "start_y": self.start_y,
            "end_x": self.end_x,
            "end_y": self.end_y,
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
            "evidence": self.evidence.to_dict(),
            "clip": self.clip.to_dict() if self.clip else None,
        }


@dataclass
class PossessionSpan:
    """A contiguous stretch of one possession state."""

    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    state: PossessionState
    team_id: str = "unknown"
    track_id: int | None = None
    player_name: str = ""
    confidence: float = 0.0
    ball_coverage: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": round(self.start_time_s, 3),
            "end_time_s": round(self.end_time_s, 3),
            "duration_s": round(self.duration_s, 3),
            "state": self.state.value,
            "team_id": self.team_id,
            "track_id": self.track_id,
            "player_name": self.player_name,
            "confidence": round(self.confidence, 4),
            "ball_coverage": round(self.ball_coverage, 4),
        }


#: Speed bands used for distance-by-zone, in m/s. Boundaries follow the
#: conventional football classification (walking / jogging / running /
#: high-speed running / sprinting) rather than arbitrary quantiles, so the
#: numbers are comparable with published match data.
SPEED_ZONES: tuple[tuple[str, float, float], ...] = (
    ("walking", 0.0, 2.0),
    ("jogging", 2.0, 4.0),
    ("running", 4.0, 5.5),
    ("high_speed", 5.5, 7.0),
    ("sprinting", 7.0, float("inf")),
)

#: A sprint is sustained high speed, not a single noisy frame above threshold.
SPRINT_MIN_SPEED_M_S = 7.0
SPRINT_MIN_DURATION_S = 0.7

#: Physically implausible for a footballer; anything above is tracking error.
MAX_PLAUSIBLE_SPEED_M_S = 12.0
MAX_PLAUSIBLE_ACCEL_M_S2 = 8.0
