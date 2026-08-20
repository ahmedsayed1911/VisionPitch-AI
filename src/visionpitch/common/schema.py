"""The canonical game-state schema.

This is the contract between Phase 1 and every later phase. Phase 2 analytics,
Phase 3 tactical models and the Phase 4 product all read *this table* and never
re-decode the video.

One row = one object observed (or estimated) in one frame.

Design rules encoded here
-------------------------
* every row carries frame number **and** timestamp, so events remain seekable
* raw observation and inference are distinguishable (``source``, ``interpolated``)
* four independent confidences are stored rather than collapsed into one, because
  a confident detection under a bad homography is a different failure from a
  weak detection under a good one
* ``validation_status`` never lets an uncertain row masquerade as a clean one
* pitch coordinates are nullable: an uncalibrated frame stores ``null``, not a
  fabricated position
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pyarrow as pa

from visionpitch.common.types import ObjectClass, Role, TeamId, ValidationStatus

SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Row model
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class GameStateRow:
    """One object in one frame."""

    # -- identity of the observation ---------------------------------------- #
    video_id: str
    frame_idx: int
    timestamp_s: float
    #: wall-clock position within the match when known; null before segmentation
    match_clock_s: float | None

    # -- what was seen ------------------------------------------------------ #
    object_class: str  # ObjectClass value
    track_id: int | None  # null for an unassociated detection
    team_id: str  # TeamId value
    role: str  # Role value
    jersey_number: int | None
    jersey_confidence: float

    # -- where, in image space ---------------------------------------------- #
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    #: ground-contact point used for projection (bottom-centre for people)
    image_x: float
    image_y: float

    # -- where, in world space ---------------------------------------------- #
    pitch_x: float | None  # metres along pitch length
    pitch_y: float | None  # metres across pitch width
    #: same position normalised to [0, 1]; convenience for visualisation
    pitch_x_norm: float | None
    pitch_y_norm: float | None

    # -- how much to trust it ----------------------------------------------- #
    detection_confidence: float
    tracking_confidence: float
    team_confidence: float
    calibration_confidence: float

    # -- provenance --------------------------------------------------------- #
    interpolated: bool
    validation_status: str  # ValidationStatus value
    segment_kind: str  # SegmentKind value
    source: str  # which detector/estimator produced this row

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Arrow schema
# --------------------------------------------------------------------------- #

GAME_STATE_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int32(), nullable=False),
        pa.field("timestamp_s", pa.float64(), nullable=False),
        pa.field("match_clock_s", pa.float64(), nullable=True),
        pa.field("object_class", pa.string(), nullable=False),
        pa.field("track_id", pa.int32(), nullable=True),
        pa.field("team_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("jersey_number", pa.int32(), nullable=True),
        pa.field("jersey_confidence", pa.float32(), nullable=False),
        pa.field("bbox_x1", pa.float32(), nullable=False),
        pa.field("bbox_y1", pa.float32(), nullable=False),
        pa.field("bbox_x2", pa.float32(), nullable=False),
        pa.field("bbox_y2", pa.float32(), nullable=False),
        pa.field("image_x", pa.float32(), nullable=False),
        pa.field("image_y", pa.float32(), nullable=False),
        pa.field("pitch_x", pa.float32(), nullable=True),
        pa.field("pitch_y", pa.float32(), nullable=True),
        pa.field("pitch_x_norm", pa.float32(), nullable=True),
        pa.field("pitch_y_norm", pa.float32(), nullable=True),
        pa.field("detection_confidence", pa.float32(), nullable=False),
        pa.field("tracking_confidence", pa.float32(), nullable=False),
        pa.field("team_confidence", pa.float32(), nullable=False),
        pa.field("calibration_confidence", pa.float32(), nullable=False),
        pa.field("interpolated", pa.bool_(), nullable=False),
        pa.field("validation_status", pa.string(), nullable=False),
        pa.field("segment_kind", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ],
    metadata={"visionpitch_schema_version": SCHEMA_VERSION},
)


#: Raw detector output, stored before any association or inference. Keeping this
#: separate from the game state is what makes tracking re-runnable without
#: re-running detection.
DETECTIONS_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int32(), nullable=False),
        pa.field("timestamp_s", pa.float64(), nullable=False),
        pa.field("object_class", pa.string(), nullable=False),
        pa.field("bbox_x1", pa.float32(), nullable=False),
        pa.field("bbox_y1", pa.float32(), nullable=False),
        pa.field("bbox_x2", pa.float32(), nullable=False),
        pa.field("bbox_y2", pa.float32(), nullable=False),
        pa.field("confidence", pa.float32(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ],
    metadata={"visionpitch_schema_version": SCHEMA_VERSION},
)


#: Per-frame calibration, stored once per frame rather than once per object.
CALIBRATION_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int32(), nullable=False),
        pa.field("timestamp_s", pa.float64(), nullable=False),
        # Row-major 3x3 image->pitch homography, or null when the frame could not
        # be calibrated.
        #
        # A *variable*-length list, deliberately, even though every non-null
        # value has exactly 9 entries. pyarrow's fixed-size-list encoding does
        # not round-trip nulls here: a null is written back as a zero-length
        # list, and reading the file then fails with "Expected all lists to be of
        # size=9". Since uncalibrated frames are normal and common, the fixed
        # form produces tables that write cleanly and are unreadable afterwards.
        # Length is validated on read instead - see tables.homography_from_row.
        pa.field("homography", pa.list_(pa.float64()), nullable=True),
        pa.field("confidence", pa.float32(), nullable=False),
        pa.field("reprojection_error_m", pa.float32(), nullable=False),
        pa.field("n_keypoints", pa.int32(), nullable=False),
        pa.field("n_inliers", pa.int32(), nullable=False),
        pa.field("smoothed", pa.bool_(), nullable=False),
        pa.field("segment_kind", pa.string(), nullable=False),
    ],
    metadata={"visionpitch_schema_version": SCHEMA_VERSION},
)


#: One row per **processed frame**, whether or not anything was detected in it.
#:
#: Without this, a consumer reading game_state.parquet cannot distinguish "this
#: frame was processed and contained nothing" from "this frame was never
#: processed" -- the frame simply has no rows either way. Measured on the
#: validation clip, 67 of 1350 frames were silently absent, and any per-frame
#: rate computed from game_state alone was therefore wrong in the denominator.
FRAMES_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int32(), nullable=False),
        pa.field("timestamp_s", pa.float64(), nullable=False),
        pa.field("n_persons", pa.int32(), nullable=False),
        pa.field("n_ball_rows", pa.int32(), nullable=False),
        pa.field("ball_observed", pa.bool_(), nullable=False),
        pa.field("calibration_confidence", pa.float32(), nullable=False),
        pa.field("calibration_valid", pa.bool_(), nullable=False),
        pa.field("calibration_propagated", pa.bool_(), nullable=False),
        pa.field("segment_kind", pa.string(), nullable=False),
        # which chunk produced it, for full-match runs
        pa.field("chunk_index", pa.int32(), nullable=True),
    ],
    metadata={"visionpitch_schema_version": SCHEMA_VERSION},
)


#: One row per track, holding track-level inferences (team, role, jersey).
TRACKS_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("track_id", pa.int32(), nullable=False),
        pa.field("object_class", pa.string(), nullable=False),
        pa.field("first_frame", pa.int32(), nullable=False),
        pa.field("last_frame", pa.int32(), nullable=False),
        pa.field("n_observations", pa.int32(), nullable=False),
        pa.field("team_id", pa.string(), nullable=False),
        pa.field("team_confidence", pa.float32(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("role_confidence", pa.float32(), nullable=False),
        pa.field("jersey_number", pa.int32(), nullable=True),
        pa.field("jersey_confidence", pa.float32(), nullable=False),
        # human-readable temporary identity, e.g. "Team A - Player #10"
        pa.field("display_name", pa.string(), nullable=False),
    ],
    metadata={"visionpitch_schema_version": SCHEMA_VERSION},
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def empty_row(video_id: str, frame_idx: int, timestamp_s: float) -> GameStateRow:
    """A fully-null row used by tests and as a construction template."""
    return GameStateRow(
        video_id=video_id,
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        match_clock_s=None,
        object_class=ObjectClass.PLAYER.value,
        track_id=None,
        team_id=TeamId.UNKNOWN.value,
        role=Role.UNKNOWN.value,
        jersey_number=None,
        jersey_confidence=0.0,
        bbox_x1=0.0,
        bbox_y1=0.0,
        bbox_x2=0.0,
        bbox_y2=0.0,
        image_x=0.0,
        image_y=0.0,
        pitch_x=None,
        pitch_y=None,
        pitch_x_norm=None,
        pitch_y_norm=None,
        detection_confidence=0.0,
        tracking_confidence=0.0,
        team_confidence=0.0,
        calibration_confidence=0.0,
        interpolated=False,
        validation_status=ValidationStatus.NO_CALIBRATION.value,
        segment_kind="unknown",
        source="none",
    )


def display_name(team_id: str, jersey_number: int | None, track_id: int, role: str) -> str:
    """Stable human-readable identity, without inventing a jersey number.

    The brief is explicit: never fabricate a number when confidence is low. The
    fallback encodes the team letter and the track id instead, which is stable
    across the run and unambiguous to a reviewer.
    """
    if role == Role.REFEREE.value:
        return f"Referee {track_id}"
    label = {
        TeamId.A.value: "Team A",
        TeamId.B.value: "Team B",
    }.get(team_id, "Unknown team")
    suffix = " Goalkeeper" if role == Role.GOALKEEPER.value else ""
    if jersey_number is not None:
        return f"{label}{suffix} - Player #{jersey_number}"
    letter = team_id.upper() if team_id in (TeamId.A.value, TeamId.B.value) else "X"
    return f"{label}{suffix} - Player {letter}{track_id:02d}"
