"""Ground-truth schema for human-reviewed broadcast ball annotation.

Broadcast ball annotation workflow, steps 5 and 6.

The one rule this module exists to enforce
------------------------------------------
**A model prediction is never ground truth.** Predictions and annotations live in
separate files with separate types, and there is no code path that promotes one
into the other. A reviewer confirming a prediction produces an *annotation whose
coordinates happen to match* — recorded as a human decision with its own
timestamp, not as an endorsed prediction.

That distinction matters because this dataset exists to measure models. If a
model's own output can leak into the answer key, the measurement is worthless and
nothing downstream would reveal it.

Absence is first class
----------------------
"No ball here" is four different statements and they are not interchangeable:

* ``NOT_VISIBLE`` — the ball is in play but hidden, in a pile of players or
  behind someone. A detector that finds nothing here is correct.
* ``OUTSIDE_FRAME`` — the ball is out of shot. A detector that finds nothing is
  correct, and a detector that finds something is hallucinating.
* ``AMBIGUOUS`` — the reviewer cannot tell. Excluded from scoring rather than
  guessed, because a coin-flip label is worse than no label.
* ``IGNORE_*`` — the frame is a replay or non-live footage and should not be
  scored at all.

Collapsing these into "no annotation" would let a detector that never fires score
identically to one that fires correctly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from visionpitch.common.logging import get_logger

log = get_logger("annotation.schema")

#: 1.1.0 adds the optional ``radius_px`` field. Additive only: annotations
#: written under 1.0.0 load unchanged and simply carry no radius.
ANNOTATION_SCHEMA_VERSION = "1.1.0"

#: Radius bounds, in source pixels. The floor rejects a sub-pixel "ball"; the
#: ceiling rejects a stray drag. Measured on this video's own detector
#: proposals, the median ball is 14.15 px across (radius 7.07) and the 95th
#: percentile is 26.6 px across, so 60 px of radius is far outside anything real.
MIN_BALL_RADIUS_PX = 1.0
MAX_BALL_RADIUS_PX = 60.0
#: Starting radius for a fresh click, from the same measurement.
DEFAULT_BALL_RADIUS_PX = 7.0


class BallVisibility(str, Enum):
    VISIBLE = "visible"
    NOT_VISIBLE = "not_visible"
    OUTSIDE_FRAME = "outside_frame"
    AMBIGUOUS = "ambiguous"

    @property
    def requires_coordinates(self) -> bool:
        return self is BallVisibility.VISIBLE

    @property
    def forbids_coordinates(self) -> bool:
        """States where a coordinate would be a contradiction, not a detail."""
        return self in (BallVisibility.OUTSIDE_FRAME, BallVisibility.NOT_VISIBLE)

    @property
    def is_scorable(self) -> bool:
        """Whether a detector can be right or wrong on this frame at all."""
        return self in (
            BallVisibility.VISIBLE,
            BallVisibility.NOT_VISIBLE,
            BallVisibility.OUTSIDE_FRAME,
        )


class IgnoreReason(str, Enum):
    NONE = "none"
    REPLAY = "ignore_replay"
    NON_LIVE = "ignore_non_live"

    @property
    def excludes_from_scoring(self) -> bool:
        return self is not IgnoreReason.NONE


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    FIRST_PASS = "first_pass"
    #: flagged by the quality-control queue; needs a second independent look
    NEEDS_SECOND_REVIEW = "needs_second_review"
    CONFIRMED = "confirmed"


class SamplingCategory(str, Enum):
    """Why this frame was chosen. Recorded so results can be broken down."""

    WIDE_SHOT = "wide_shot"
    MIDFIELD_PLAY = "midfield_play"
    PENALTY_AREA = "penalty_area"
    NEAR_GOAL = "near_goal"
    FAST_TRANSITION = "fast_transition"
    LONG_BALL = "long_ball"
    AERIAL_BALL = "aerial_ball"
    SHORT_PASS = "short_pass"
    CAMERA_PAN = "camera_pan"
    CAMERA_ZOOM = "camera_zoom"
    MOTION_BLUR = "motion_blur"
    PLAYER_OCCLUSION = "player_occlusion"
    CROWDED_SCENE = "crowded_scene"
    BALL_NEAR_BODY = "ball_near_body"
    BALL_NEAR_LINE = "ball_near_line"
    LOW_CONTRAST = "low_contrast"
    BROADCAST_GRAPHIC = "broadcast_graphic"
    BALL_OUT_OF_FRAME = "ball_out_of_frame"
    UNOBSERVABLE = "unobservable"
    TEMPORAL_WINDOW = "temporal_window"
    CLOSE_UP_NEGATIVE = "close_up_negative"
    CROWD_NEGATIVE = "crowd_negative"


@dataclass
class FrameSample:
    """A frame selected for review. Immutable once the package is built."""

    frame_id: str
    frame_idx: int
    timestamp_s: float
    image_path: str
    shot_index: int
    shot_type: str
    sampling_category: SamplingCategory
    sampling_reason: str
    is_live_play_candidate: bool
    likely_slow_motion: bool
    #: frames belonging to one short consecutive run share this id
    window_id: str | None = None
    source_content_hash: str = ""
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["sampling_category"] = self.sampling_category.value
        return payload

    @staticmethod
    def from_dict(data: dict) -> FrameSample:
        payload = dict(data)
        payload["sampling_category"] = SamplingCategory(payload["sampling_category"])
        return FrameSample(**payload)


@dataclass
class ModelPrediction:
    """A proposal. Stored apart from ground truth and never edited by a reviewer."""

    frame_id: str
    model_label: str
    model_fingerprint: str
    centre_x: float | None
    centre_y: float | None
    confidence: float
    bbox: list[float] | None = None
    uncertainty_px: float | None = None

    def to_dict(self) -> dict:
        return {**asdict(self), "record_type": "prediction"}

    @staticmethod
    def from_dict(data: dict) -> ModelPrediction:
        payload = {k: v for k, v in data.items() if k != "record_type"}
        return ModelPrediction(**payload)


@dataclass
class BallAnnotation:
    """One human decision about one frame."""

    frame_id: str
    visibility: BallVisibility
    ignore_reason: IgnoreReason = IgnoreReason.NONE
    centre_x: float | None = None
    centre_y: float | None = None
    #: Visible ball radius in source-image pixels, set by the reviewer.
    #:
    #: This is the *ball*, not the marker drawn on screen. The marker scales with
    #: zoom so it stays clickable; the radius stored here does not. Recording it
    #: makes the dataset able to answer questions no fixed-size annotation can:
    #: recall by true ball scale, and whether a detector's errors concentrate on
    #: the small end. Measured on this video's own proposals, the median ball is
    #: 14.15 px across, so the default is 7 px.
    radius_px: float | None = None
    bbox: list[float] | None = None
    #: reviewer's own confidence, not a model's
    annotation_confidence: float = 1.0
    ambiguity_reason: str = ""
    reviewer: str = "anonymous"
    review_status: ReviewStatus = ReviewStatus.FIRST_PASS
    reviewed_at: str = ""
    #: true when the reviewer accepted a proposal unchanged. Recorded for audit
    #: -- it is still a human decision, but a cheaper one, and a dataset made
    #: entirely of these would be measuring the proposing model against itself.
    accepted_proposal_from: str | None = None
    schema_version: str = ANNOTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.reviewed_at:
            self.reviewed_at = datetime.now(UTC).isoformat()
        # A radius and a box are two views of one measurement, so the box is
        # derived rather than stored independently -- two fields that can
        # disagree eventually do.
        if (
            self.bbox is None
            and self.radius_px is not None
            and self.centre_x is not None
            and self.centre_y is not None
        ):
            self.bbox = [
                self.centre_x - self.radius_px, self.centre_y - self.radius_px,
                self.centre_x + self.radius_px, self.centre_y + self.radius_px,
            ]

    @property
    def is_scorable(self) -> bool:
        return self.visibility.is_scorable and not self.ignore_reason.excludes_from_scoring

    @property
    def diameter_px(self) -> float | None:
        """Visible ball diameter, for size-stratified reporting."""
        return self.radius_px * 2 if self.radius_px is not None else None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["visibility"] = self.visibility.value
        payload["ignore_reason"] = self.ignore_reason.value
        payload["review_status"] = self.review_status.value
        payload["record_type"] = "annotation"
        return payload

    @staticmethod
    def from_dict(data: dict) -> BallAnnotation:
        payload = {k: v for k, v in data.items() if k != "record_type"}
        payload["visibility"] = BallVisibility(payload["visibility"])
        payload["ignore_reason"] = IgnoreReason(payload.get("ignore_reason", "none"))
        payload["review_status"] = ReviewStatus(
            payload.get("review_status", ReviewStatus.FIRST_PASS.value)
        )
        return BallAnnotation(**payload)


class AnnotationError(ValueError):
    """A rejected annotation. Never a warning -- bad rows do not get stored."""


def validate(
    annotation: BallAnnotation, sample: FrameSample
) -> None:
    """Reject anything internally contradictory. Raises on the first problem."""
    if annotation.frame_id != sample.frame_id:
        raise AnnotationError(
            f"frame id mismatch: annotation {annotation.frame_id!r} vs sample "
            f"{sample.frame_id!r}"
        )

    has_coordinates = annotation.centre_x is not None and annotation.centre_y is not None

    if annotation.visibility.requires_coordinates and not has_coordinates:
        raise AnnotationError(
            f"{annotation.frame_id}: visibility 'visible' requires a centre"
        )
    if annotation.visibility.forbids_coordinates and has_coordinates:
        raise AnnotationError(
            f"{annotation.frame_id}: visibility {annotation.visibility.value!r} "
            f"cannot carry a centre; that is a contradiction, not extra detail"
        )

    if has_coordinates:
        if not (0 <= annotation.centre_x <= sample.width):
            raise AnnotationError(
                f"{annotation.frame_id}: x={annotation.centre_x} outside "
                f"[0, {sample.width}]"
            )
        if not (0 <= annotation.centre_y <= sample.height):
            raise AnnotationError(
                f"{annotation.frame_id}: y={annotation.centre_y} outside "
                f"[0, {sample.height}]"
            )

    if annotation.radius_px is not None:
        if not annotation.visibility.requires_coordinates:
            raise AnnotationError(
                f"{annotation.frame_id}: a radius implies the ball is visible"
            )
        if annotation.radius_px < MIN_BALL_RADIUS_PX:
            raise AnnotationError(
                f"{annotation.frame_id}: radius {annotation.radius_px} below the "
                f"{MIN_BALL_RADIUS_PX} px floor; a ball smaller than one pixel is "
                f"not a measurement"
            )
        if annotation.radius_px > MAX_BALL_RADIUS_PX:
            raise AnnotationError(
                f"{annotation.frame_id}: radius {annotation.radius_px} above the "
                f"{MAX_BALL_RADIUS_PX} px ceiling; that is larger than any ball on "
                f"a broadcast frame and is almost certainly a stray drag"
            )

    if annotation.bbox is not None:
        if len(annotation.bbox) != 4:
            raise AnnotationError(f"{annotation.frame_id}: bbox must be [x1,y1,x2,y2]")
        x1, y1, x2, y2 = annotation.bbox
        if x2 <= x1 or y2 <= y1:
            raise AnnotationError(f"{annotation.frame_id}: bbox has non-positive extent")
        if not annotation.visibility.requires_coordinates:
            raise AnnotationError(
                f"{annotation.frame_id}: a bbox implies the ball is visible"
            )

    if (
        annotation.visibility is BallVisibility.AMBIGUOUS
        and not annotation.ambiguity_reason
    ):
        raise AnnotationError(
            f"{annotation.frame_id}: 'ambiguous' requires a reason, so the queue "
            f"can tell an unclear ball from an unclear rule"
        )

    if not 0.0 <= annotation.annotation_confidence <= 1.0:
        raise AnnotationError(
            f"{annotation.frame_id}: annotation_confidence out of range"
        )


class AnnotationStore:
    """Append-only JSONL storage that survives being interrupted.

    Append-only rather than rewrite-in-place: a review session that is killed
    halfway through must not corrupt hours of prior work, and re-annotating a
    frame must leave the earlier decision on record.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.root / "samples.jsonl"
        self.predictions_path = self.root / "predictions.jsonl"
        self.annotations_path = self.root / "annotations.jsonl"
        self.manifest_path = self.root / "package.json"

    # -- samples and predictions (written once, then read-only) --------------- #

    def write_samples(self, samples: list[FrameSample]) -> None:
        seen: set[str] = set()
        with self.samples_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                if sample.frame_id in seen:
                    raise AnnotationError(f"duplicate frame id {sample.frame_id}")
                seen.add(sample.frame_id)
                handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    def load_samples(self) -> dict[str, FrameSample]:
        out: dict[str, FrameSample] = {}
        if not self.samples_path.exists():
            return out
        for line in self.samples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = FrameSample.from_dict(json.loads(line))
            if sample.frame_id in out:
                raise AnnotationError(f"duplicate frame id {sample.frame_id}")
            out[sample.frame_id] = sample
        return out

    def write_predictions(self, predictions: list[ModelPrediction]) -> None:
        with self.predictions_path.open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                # ``default=float`` because detector outputs arrive as numpy
                # scalars. Without it a float32 that slipped through fails only
                # here, at the end of a ten-minute build.
                handle.write(
                    json.dumps(
                        prediction.to_dict(), ensure_ascii=False, default=float
                    ) + "\n"
                )

    def load_predictions(self) -> dict[str, list[ModelPrediction]]:
        out: dict[str, list[ModelPrediction]] = {}
        if not self.predictions_path.exists():
            return out
        for line in self.predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prediction = ModelPrediction.from_dict(json.loads(line))
            out.setdefault(prediction.frame_id, []).append(prediction)
        return out

    # -- annotations (append-only) -------------------------------------------- #

    def append(self, annotation: BallAnnotation, sample: FrameSample) -> None:
        validate(annotation, sample)
        with self.annotations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(annotation.to_dict(), ensure_ascii=False) + "\n")

    def load_annotations(self) -> dict[str, BallAnnotation]:
        """Latest decision per frame; earlier ones stay in the file."""
        out: dict[str, BallAnnotation] = {}
        if not self.annotations_path.exists():
            return out
        for line in self.annotations_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            annotation = BallAnnotation.from_dict(json.loads(line))
            out[annotation.frame_id] = annotation
        return out

    def history(self) -> list[BallAnnotation]:
        if not self.annotations_path.exists():
            return []
        return [
            BallAnnotation.from_dict(json.loads(line))
            for line in self.annotations_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # -- package manifest ------------------------------------------------------ #

    def write_manifest(self, payload: dict) -> Path:
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.manifest_path

    def manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def assert_source_matches(self, content_hash: str) -> None:
        """Refuse to mix annotations from a different video."""
        stored = self.manifest().get("source_content_hash")
        if stored and stored != content_hash:
            raise AnnotationError(
                f"this package was built from video {stored[:16]} but the current "
                f"video hashes to {content_hash[:16]}"
            )

    def fingerprint(self) -> str:
        """Content hash of the annotations, for split and report provenance."""
        annotations = self.load_annotations()
        payload = json.dumps(
            {k: annotations[k].to_dict() for k in sorted(annotations)},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def progress(self) -> dict:
        samples = self.load_samples()
        annotations = self.load_annotations()
        by_status: dict[str, int] = {}
        by_visibility: dict[str, int] = {}
        for annotation in annotations.values():
            by_status[annotation.review_status.value] = (
                by_status.get(annotation.review_status.value, 0) + 1
            )
            by_visibility[annotation.visibility.value] = (
                by_visibility.get(annotation.visibility.value, 0) + 1
            )
        return {
            "n_samples": len(samples),
            "n_annotated": len(annotations),
            "n_remaining": len(samples) - len(annotations),
            "by_review_status": dict(sorted(by_status.items())),
            "by_visibility": dict(sorted(by_visibility.items())),
            "n_scorable": sum(1 for a in annotations.values() if a.is_scorable),
            "annotation_fingerprint": self.fingerprint(),
        }
