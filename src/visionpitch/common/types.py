"""Core in-memory types shared across pipeline stages.

Design note
-----------
Raw *observations* (detections) are kept structurally separate from *inferred*
entities (tracks, teams, pitch positions).  Nothing in this module encodes a
football event -- events are Phase 2 and consume the storage schema, not these
objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class ObjectClass(str, Enum):
    """Detected object category. Mirrors the multiclass detector's label set."""

    BALL = "ball"
    GOALKEEPER = "goalkeeper"
    PLAYER = "player"
    REFEREE = "referee"

    @property
    def is_person(self) -> bool:
        return self is not ObjectClass.BALL


class TeamId(str, Enum):
    """Team assignment.

    ``A``/``B`` are arbitrary discovered labels, not real club identities; the
    user may rename them downstream without re-running detection.
    """

    A = "A"
    B = "B"
    NONE = "none"  # referees, and the ball
    UNKNOWN = "unknown"  # person whose team could not be resolved confidently


class Role(str, Enum):
    OUTFIELD = "outfield"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    """Per-record trust signal. Never silently drop a bad record -- label it."""

    VALID = "valid"
    #: geometry present but calibration confidence below threshold
    LOW_CALIBRATION = "low_calibration"
    #: no usable homography for this frame; pitch coords are null
    NO_CALIBRATION = "no_calibration"
    #: position filled by the temporal model rather than observed
    INTERPOLATED = "interpolated"
    #: projected well outside the image region the pitch landmarks constrained.
    #: Common for far-side players near the horizon, where homography error
    #: grows sharply. The coordinate is provided but should not be used for
    #: physical measurement without accounting for it.
    EXTRAPOLATED = "extrapolated"
    #: observed but flagged implausible by a temporal consistency check
    IMPLAUSIBLE = "implausible"
    #: frame classified as replay/close-up/non-live footage
    NON_LIVE = "non_live"


class SegmentKind(str, Enum):
    """Coarse footage classification. Tactical analytics run on LIVE only."""

    LIVE = "live"
    REPLAY = "replay"
    CLOSE_UP = "close_up"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class BBox:
    """Axis-aligned box in image pixel coordinates, ``xyxy`` convention."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    @property
    def ground_contact(self) -> tuple[float, float]:
        """Estimated point where the object meets the ground.

        For a person this is the bottom-centre of the box, which projects far
        more accurately through a ground-plane homography than the box centre
        (the centre sits ~90cm above the pitch and produces a systematic
        depth-dependent error of several metres).
        """
        return (0.5 * (self.x1 + self.x2), self.y2)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def clip(self, width: int, height: int) -> BBox:
        return BBox(
            max(0.0, min(self.x1, width - 1.0)),
            max(0.0, min(self.y1, height - 1.0)),
            max(0.0, min(self.x2, width - 1.0)),
            max(0.0, min(self.y2, height - 1.0)),
        )

    def iou(self, other: BBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def from_xyxy(v) -> BBox:
        return BBox(float(v[0]), float(v[1]), float(v[2]), float(v[3]))


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Detection:
    """A single raw detector output. Carries no inferred football semantics."""

    frame_idx: int
    object_class: ObjectClass
    bbox: BBox
    confidence: float
    #: which detector produced it, e.g. "multiclass" / "ball_hires" / "coco"
    source: str = "multiclass"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")


@dataclass(slots=True)
class FrameDetections:
    """All detections for one frame, plus the frame's identity."""

    frame_idx: int
    timestamp_s: float
    detections: list[Detection] = field(default_factory=list)

    def of_class(self, *classes: ObjectClass) -> list[Detection]:
        wanted = set(classes)
        return [d for d in self.detections if d.object_class in wanted]

    @property
    def persons(self) -> list[Detection]:
        return [d for d in self.detections if d.object_class.is_person]


# --------------------------------------------------------------------------- #
# Tracks
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TrackObservation:
    """One frame of a track's life."""

    frame_idx: int
    timestamp_s: float
    bbox: BBox
    det_confidence: float
    track_confidence: float
    #: True when the tracker predicted this box rather than observing it
    interpolated: bool = False


@dataclass(slots=True)
class Track:
    """A temporally-linked sequence of observations of one object."""

    track_id: int
    object_class: ObjectClass
    observations: list[TrackObservation] = field(default_factory=list)
    #: resolved downstream by team_classification, with confidence
    team_id: TeamId = TeamId.UNKNOWN
    team_confidence: float = 0.0
    role: Role = Role.UNKNOWN
    role_confidence: float = 0.0
    #: optional identity, resolved by the reid stage
    jersey_number: int | None = None
    jersey_confidence: float = 0.0
    #: confidence-weighted detector class votes accumulated over the track's
    #: life, keyed by :class:`ObjectClass` value.
    #:
    #: ``object_class`` alone is the class of the *birth* detection and never
    #: changes afterwards, so a single spurious ``referee`` box at the moment a
    #: track is created labels a player a referee for the rest of the clip.
    #: Role resolution reads these votes instead.
    class_votes: dict[str, float] = field(default_factory=dict)
    #: raw per-class observation counts backing ``class_votes``
    class_counts: dict[str, int] = field(default_factory=dict)

    def class_share(self, object_class: ObjectClass) -> float:
        """Fraction of this track's weighted class evidence for ``object_class``."""
        total = sum(self.class_votes.values())
        if total <= 0.0:
            return 1.0 if self.object_class is object_class else 0.0
        return float(self.class_votes.get(object_class.value, 0.0) / total)

    def class_frames(self, object_class: ObjectClass) -> int:
        """Number of observed frames the detector called ``object_class``."""
        if not self.class_counts:
            return self.length if self.object_class is object_class else 0
        return int(self.class_counts.get(object_class.value, 0))

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame_idx

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame_idx

    @property
    def length(self) -> int:
        return len(self.observations)

    @property
    def span(self) -> int:
        """Frames between first and last sighting (>= length when fragmented)."""
        return self.last_frame - self.first_frame + 1

    def observation_at(self, frame_idx: int) -> TrackObservation | None:
        # Observations are appended in frame order, so bisect is valid.
        lo, hi = 0, len(self.observations) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            f = self.observations[mid].frame_idx
            if f == frame_idx:
                return self.observations[mid]
            if f < frame_idx:
                lo = mid + 1
            else:
                hi = mid - 1
        return None


@dataclass(slots=True)
class BallState:
    """Estimated ball state for one frame.

    ``observed`` distinguishes a real detection from a model-filled position.
    ``position`` is ``None`` when the estimator refuses to guess -- which is a
    valid and desirable output during long disappearances.
    """

    frame_idx: int
    timestamp_s: float
    position: tuple[float, float] | None  # image coords of ball centre
    bbox: BBox | None
    velocity: tuple[float, float] | None  # px/frame, image space
    confidence: float
    observed: bool
    interpolated: bool
    #: 1-sigma positional uncertainty in pixels; grows through occlusion
    uncertainty_px: float = 0.0


@dataclass(slots=True)
class CalibrationResult:
    """Ground-plane homography for one frame.

    ``homography`` maps *image* pixels to *pitch* coordinates in metres.
    """

    frame_idx: int
    homography: np.ndarray | None  # 3x3, image -> pitch (metres)
    confidence: float
    #: mean reprojection error over inlier keypoints, in metres
    reprojection_error_m: float
    n_keypoints: int
    n_inliers: int
    smoothed: bool = False
    segment_kind: SegmentKind = SegmentKind.UNKNOWN

    @property
    def is_valid(self) -> bool:
        return self.homography is not None
