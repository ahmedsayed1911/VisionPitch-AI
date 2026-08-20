"""Quality control over first-pass annotations.

Broadcast ball annotation workflow, step 6.

First-pass annotation is fast and therefore occasionally wrong. Rather than
re-reviewing everything, this ranks frames by how likely a second look is to
change something, and by how much that change would matter.

The flags are deliberately *evidence about the annotation*, not about the models.
A frame where both detectors disagree with the human is not evidence the human is
wrong — it is exactly the case the dataset exists to capture — but it is a case
worth confirming, because it will carry disproportionate weight in every metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from visionpitch.annotation.schema import (
    BallAnnotation,
    BallVisibility,
    FrameSample,
    ModelPrediction,
)
from visionpitch.common.logging import get_logger

log = get_logger("annotation.quality")

QUALITY_SCHEMA_VERSION = "1.0.0"

#: Both detectors this far from the human point: worth confirming.
DISAGREEMENT_PX = 40.0
#: Implied ball speed above this, between consecutive annotated frames in one
#: window, is not a ball -- it is a misplaced click. 1280x720 broadcast, ~11 px
#: ball; a genuine fast ball crosses roughly 60 px per frame at 50 fps.
MAX_STEP_PX_PER_FRAME = 90.0


class QualityFlag(str, Enum):
    DETECTOR_DISAGREEMENT = "detector_disagreement"
    TEMPORAL_JUMP = "temporal_jump"
    ISOLATED_POSITION = "isolated_position"
    NEIGHBOUR_INCONSISTENT = "neighbour_inconsistent"
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS = "ambiguous"
    ACCEPTED_PROPOSAL_ONLY = "accepted_proposal_only"

    @property
    def weight(self) -> float:
        """How strongly this flag argues for a second look."""
        return {
            QualityFlag.TEMPORAL_JUMP: 3.0,
            QualityFlag.NEIGHBOUR_INCONSISTENT: 2.5,
            QualityFlag.ISOLATED_POSITION: 2.0,
            QualityFlag.DETECTOR_DISAGREEMENT: 1.5,
            QualityFlag.AMBIGUOUS: 1.0,
            QualityFlag.LOW_CONFIDENCE: 1.0,
            QualityFlag.ACCEPTED_PROPOSAL_ONLY: 0.5,
        }[self]


@dataclass
class QualityItem:
    frame_id: str
    flags: list[QualityFlag] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def priority(self) -> float:
        return sum(flag.weight for flag in self.flags)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "flags": [f.value for f in self.flags],
            "priority": round(self.priority, 2),
            "detail": self.detail,
        }


def build_queue(
    samples: dict[str, FrameSample],
    annotations: dict[str, BallAnnotation],
    predictions: dict[str, list[ModelPrediction]],
) -> list[QualityItem]:
    """Rank annotated frames by how much a second review would be worth."""
    items: dict[str, QualityItem] = {}

    def flag(frame_id: str, value: QualityFlag, **detail) -> None:
        item = items.setdefault(frame_id, QualityItem(frame_id=frame_id))
        if value not in item.flags:
            item.flags.append(value)
        item.detail.update(detail)

    for frame_id, annotation in annotations.items():
        if annotation.visibility is BallVisibility.AMBIGUOUS:
            flag(frame_id, QualityFlag.AMBIGUOUS, reason=annotation.ambiguity_reason)
        if annotation.annotation_confidence < 0.75:
            flag(
                frame_id, QualityFlag.LOW_CONFIDENCE,
                confidence=annotation.annotation_confidence,
            )
        if annotation.accepted_proposal_from:
            flag(
                frame_id, QualityFlag.ACCEPTED_PROPOSAL_ONLY,
                accepted_from=annotation.accepted_proposal_from,
            )

        if annotation.visibility is BallVisibility.VISIBLE and annotation.centre_x is not None:
            distances = []
            for prediction in predictions.get(frame_id, []):
                if prediction.centre_x is None:
                    continue
                distances.append(
                    math.dist(
                        (annotation.centre_x, annotation.centre_y),
                        (prediction.centre_x, prediction.centre_y),
                    )
                )
            # Only flag when EVERY model that fired disagrees. One model being
            # wrong is ordinary; both being wrong is either a hard frame worth
            # keeping or a misplaced click worth checking.
            if distances and min(distances) > DISAGREEMENT_PX:
                flag(
                    frame_id, QualityFlag.DETECTOR_DISAGREEMENT,
                    nearest_prediction_px=round(min(distances), 1),
                )

    # -- temporal checks, inside annotated windows ----------------------------- #
    windows: dict[str, list[str]] = {}
    for frame_id, sample in samples.items():
        if sample.window_id and frame_id in annotations:
            windows.setdefault(sample.window_id, []).append(frame_id)

    for window_id, frame_ids in windows.items():
        ordered = sorted(frame_ids, key=lambda f: samples[f].frame_idx)
        visible = [
            f for f in ordered
            if annotations[f].visibility is BallVisibility.VISIBLE
            and annotations[f].centre_x is not None
        ]

        for a, b in zip(visible, visible[1:], strict=False):
            gap = max(1, samples[b].frame_idx - samples[a].frame_idx)
            step = math.dist(
                (annotations[a].centre_x, annotations[a].centre_y),
                (annotations[b].centre_x, annotations[b].centre_y),
            ) / gap
            if step > MAX_STEP_PX_PER_FRAME:
                for frame_id in (a, b):
                    flag(
                        frame_id, QualityFlag.TEMPORAL_JUMP,
                        step_px_per_frame=round(step, 1), window=window_id,
                    )

        # A single visible frame surrounded by "not visible" inside one short
        # window is usually a stray click or a missed ball, either way worth a
        # second look.
        for index, frame_id in enumerate(ordered):
            if annotations[frame_id].visibility is not BallVisibility.VISIBLE:
                continue
            neighbours = [
                ordered[j] for j in (index - 1, index + 1) if 0 <= j < len(ordered)
            ]
            if neighbours and all(
                annotations[n].visibility is not BallVisibility.VISIBLE
                for n in neighbours
            ):
                flag(frame_id, QualityFlag.ISOLATED_POSITION, window=window_id)

        # Midpoint test: a point far off the line between its neighbours.
        for index in range(1, len(visible) - 1):
            previous, current, following = visible[index - 1], visible[index], visible[index + 1]
            px, py = annotations[previous].centre_x, annotations[previous].centre_y
            nx, ny = annotations[following].centre_x, annotations[following].centre_y
            cx, cy = annotations[current].centre_x, annotations[current].centre_y
            expected = ((px + nx) / 2, (py + ny) / 2)
            deviation = math.dist((cx, cy), expected)
            span = math.dist((px, py), (nx, ny))
            if deviation > max(40.0, span):
                flag(
                    frame_id=current, value=QualityFlag.NEIGHBOUR_INCONSISTENT,
                    deviation_px=round(deviation, 1), window=window_id,
                )

    ranked = sorted(items.values(), key=lambda item: -item.priority)
    log.info(
        "quality queue: %d frame(s) flagged out of %d annotated",
        len(ranked), len(annotations),
    )
    return ranked


def summarise(queue: list[QualityItem], n_annotated: int) -> dict:
    counts: dict[str, int] = {}
    for item in queue:
        for flag in item.flags:
            counts[flag.value] = counts.get(flag.value, 0) + 1
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "n_annotated": n_annotated,
        "n_flagged": len(queue),
        "share_flagged": round(len(queue) / n_annotated, 4) if n_annotated else 0.0,
        "by_flag": dict(sorted(counts.items())),
        "thresholds": {
            "detector_disagreement_px": DISAGREEMENT_PX,
            "max_step_px_per_frame": MAX_STEP_PX_PER_FRAME,
        },
        "queue": [item.to_dict() for item in queue],
    }
