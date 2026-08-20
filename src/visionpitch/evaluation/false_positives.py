"""False-positive taxonomy for ball detection.

Candidate C's blocking weakness is false positives, and "too many false
positives" is not an actionable statement. This assigns every unmatched
prediction to a cause, using measurable image evidence at the prediction's
location, so hard-negative mining targets what actually fires rather than what
seems plausible.

What is measured and what is inferred
-------------------------------------
Categories are assigned from evidence: grass coverage, containment in a detected
person box and where inside it, straight-line structure, blob roundness, local
texture, and position relative to the pitch. All of that is measurement.

The *naming* is inference. A bright round blob on grass with no line through it
is called a penalty spot because that is overwhelmingly what it is on a football
pitch, not because anything verified it. The taxonomy is a mining aid, and the
categories that drive decisions are the ones with a physical test behind them --
inside a person box, on a line, off the pitch.

Temporal categories are separate on purpose
-------------------------------------------
``DUPLICATE_CANDIDATE``, ``TRAJECTORY_INCONSISTENT`` and one-frame persistence
cannot be judged from a still. They are only assigned when the caller supplies
sequence context, and are reported against their own denominator so a still-image
audit cannot be mistaken for a temporal one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import cv2
import numpy as np

from visionpitch.common.logging import get_logger

log = get_logger("evaluation.false_positives")

FP_TAXONOMY_VERSION = "1.0.0"

#: Grass hue range in HSV. Calibrated in the annotation audit, where confirmed
#: play frames ran 0.45-0.78 green fraction and ball-free close-ups under 0.05.
GREEN_LOW = (30, 40, 40)
GREEN_HIGH = (90, 255, 255)

#: A patch this green is on the pitch surface.
ON_GRASS_FRACTION = 0.35
#: Straight-line responses at or above this mean a pitch marking runs through.
LINE_RESPONSES = 2
#: Laplacian variance below this reads as a blur smear rather than an object.
BLUR_VARIANCE = 35.0
#: Two predictions closer than this are the same thing counted twice.
DUPLICATE_PX = 30.0


class FalsePositiveKind(str, Enum):
    PITCH_LINE = "pitch_line"
    PENALTY_SPOT = "penalty_spot"
    PLAYER_SOCKS_BOOTS = "player_socks_or_boots"
    JERSEY_MARKING = "jersey_marking"
    ADVERTISING_BOARD = "advertising_board"
    BROADCAST_GRAPHIC = "broadcast_graphic"
    CROWD_HIGHLIGHT = "crowd_highlight"
    GOAL_NET = "goal_net"
    WHITE_SEAT = "white_seat"
    COMPRESSION_ARTIFACT = "compression_artifact"
    MOTION_BLUR_ARTIFACT = "motion_blur_artifact"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    TRAJECTORY_INCONSISTENT = "trajectory_inconsistent_candidate"
    OUTSIDE_PITCH = "outside_pitch_candidate"
    UNKNOWN = "unknown"

    @property
    def is_temporal(self) -> bool:
        """Whether the category needs sequence context to be assigned at all."""
        return self in (
            FalsePositiveKind.DUPLICATE_CANDIDATE,
            FalsePositiveKind.TRAJECTORY_INCONSISTENT,
        )

    @property
    def minable(self) -> bool:
        """Whether a crop of this is useful as a training hard negative.

        Duplicates and trajectory failures are *fusion* problems, not appearance
        problems -- mining crops of them would teach the detector to suppress
        real balls.
        """
        return not self.is_temporal and self is not FalsePositiveKind.UNKNOWN


@dataclass
class FalsePositive:
    """One unmatched prediction, with the evidence behind its label."""

    frame_id: str
    domain: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    kind: FalsePositiveKind
    green_fraction: float
    line_responses: int
    blur_variance: float
    roundness: float
    inside_person: bool
    person_vertical_position: float | None
    relative_y: float
    #: frames in a row this location produced a candidate; None without context
    persistence: int | None = None
    accepted_by_fusion: bool | None = None

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["area_px2"] = round(self.area, 1)
        return payload


def _patch(image: np.ndarray, x: float, y: float, radius: int):
    h, w = image.shape[:2]
    x1, y1 = int(max(0, x - radius)), int(max(0, y - radius))
    x2, y2 = int(min(w, x + radius)), int(min(h, y + radius))
    return image[y1:y2, x1:x2]


def measure(
    image: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    person_boxes: np.ndarray,
) -> dict:
    """Evidence at a prediction's location. No naming, only measurement."""
    h, w = image.shape[:2]
    side = max(6.0, max(width, height))
    patch = _patch(image, x, y, int(side * 2))
    context = _patch(image, x, y, int(side * 5))

    green = 0.0
    if context.size:
        mask = cv2.inRange(cv2.cvtColor(context, cv2.COLOR_BGR2HSV), GREEN_LOW, GREEN_HIGH)
        green = float(mask.mean() / 255.0)

    lines = 0
    blur = 0.0
    roundness = 0.0
    if patch.size:
        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        edges = cv2.Canny(grey, 60, 180)
        found = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=max(10, int(side)),
            minLineLength=max(10, int(side * 1.5)), maxLineGap=3,
        )
        lines = 0 if found is None else int(len(np.asarray(found).reshape(-1, 4)))

        # Roundness of the brightest connected blob: a ball is round, a line is
        # not, a boot is not.
        _, binary = cv2.threshold(
            grey, max(120, int(grey.mean() + 1.5 * grey.std())), 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(biggest))
            perimeter = float(cv2.arcLength(biggest, True))
            if perimeter > 0:
                roundness = float(4 * np.pi * area / (perimeter * perimeter))

    inside = False
    vertical = None
    for box in person_boxes:
        bx1, by1, bx2, by2 = box[:4]
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            inside = True
            span = max(1.0, by2 - by1)
            vertical = float((y - by1) / span)
            break

    return {
        "green_fraction": green,
        "line_responses": lines,
        "blur_variance": blur,
        "roundness": roundness,
        "inside_person": inside,
        "person_vertical_position": vertical,
        "relative_y": float(y / max(1, h)),
    }


def classify(evidence: dict, width: float, height: float) -> FalsePositiveKind:
    """Assign one category, most physically-grounded test first.

    Order matters: containment in a person box is a hard geometric fact and
    outranks appearance, because a white blob inside a player is a sock or a
    shirt logo whatever else it looks like.
    """
    green = evidence["green_fraction"]
    on_grass = green >= ON_GRASS_FRACTION
    relative_y = evidence["relative_y"]

    # 1. inside a detected person -- geometry, not appearance
    if evidence["inside_person"]:
        vertical = evidence["person_vertical_position"] or 0.5
        return (
            FalsePositiveKind.PLAYER_SOCKS_BOOTS if vertical >= 0.6
            else FalsePositiveKind.JERSEY_MARKING
        )

    # 2. off the pitch surface entirely
    if green < 0.05:
        # Above the pitch is crowd or boards; broadcast overlays cluster at the
        # very top and bottom of the frame.
        if relative_y < 0.10 or relative_y > 0.93:
            return FalsePositiveKind.BROADCAST_GRAPHIC
        if relative_y < 0.35:
            return (
                FalsePositiveKind.WHITE_SEAT if evidence["roundness"] > 0.55
                else FalsePositiveKind.CROWD_HIGHLIGHT
            )
        return FalsePositiveKind.ADVERTISING_BOARD

    if not on_grass:
        # Partially green: the boundary band -- boards, netting, the goal frame.
        if evidence["line_responses"] >= 4:
            return FalsePositiveKind.GOAL_NET
        if relative_y < 0.4:
            return FalsePositiveKind.ADVERTISING_BOARD
        return FalsePositiveKind.OUTSIDE_PITCH

    # 3. on grass
    if evidence["line_responses"] >= LINE_RESPONSES:
        return FalsePositiveKind.PITCH_LINE
    if evidence["blur_variance"] < BLUR_VARIANCE:
        return (
            FalsePositiveKind.COMPRESSION_ARTIFACT if evidence["roundness"] < 0.3
            else FalsePositiveKind.MOTION_BLUR_ARTIFACT
        )
    if evidence["roundness"] >= 0.55 and max(width, height) <= 22:
        return FalsePositiveKind.PENALTY_SPOT
    return FalsePositiveKind.UNKNOWN


def mark_duplicates(
    predictions: list[tuple[float, float, float]], threshold: float = DUPLICATE_PX
) -> list[bool]:
    """Flag predictions that duplicate a higher-confidence neighbour.

    ``predictions`` is ``(x, y, confidence)``. The highest-confidence member of
    a cluster is kept; the rest are duplicates.
    """
    order = sorted(range(len(predictions)), key=lambda i: -predictions[i][2])
    duplicate = [False] * len(predictions)
    kept: list[int] = []
    for index in order:
        x, y, _ = predictions[index]
        if any(
            float(np.hypot(x - predictions[k][0], y - predictions[k][1])) <= threshold
            for k in kept
        ):
            duplicate[index] = True
        else:
            kept.append(index)
    return duplicate


def summarise(items: list[FalsePositive], n_frames: int) -> dict:
    """Counts, rates and distributions, grouped by cause."""
    if not items:
        return {
            "schema_version": FP_TAXONOMY_VERSION,
            "n_false_positives": 0, "n_frames": n_frames, "by_kind": {},
        }

    by_kind: dict[str, list[FalsePositive]] = {}
    for item in items:
        by_kind.setdefault(item.kind.value, []).append(item)

    table = {}
    for kind, group in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        confidences = np.array([g.confidence for g in group])
        areas = np.array([g.area for g in group])
        ys = np.array([g.relative_y for g in group])
        domains: dict[str, int] = {}
        for g in group:
            domains[g.domain] = domains.get(g.domain, 0) + 1
        persistences = [g.persistence for g in group if g.persistence is not None]
        table[kind] = {
            "count": len(group),
            "pct_of_false_positives": round(len(group) / len(items), 4),
            "per_frame": round(len(group) / max(1, n_frames), 4),
            "confidence": {
                "median": round(float(np.median(confidences)), 4),
                "p10": round(float(np.percentile(confidences, 10)), 4),
                "p90": round(float(np.percentile(confidences, 90)), 4),
            },
            "estimated_ball_size_px2": {
                "median": round(float(np.median(areas)), 1),
                "p90": round(float(np.percentile(areas, 90)), 1),
            },
            "image_region_relative_y": {
                "median": round(float(np.median(ys)), 3),
                "share_upper_third": round(float((ys < 0.333).mean()), 3),
            },
            "by_domain": dict(sorted(domains.items())),
            "temporal_persistence": {
                "n_with_context": len(persistences),
                "median_frames": (
                    round(float(np.median(persistences)), 2) if persistences else None
                ),
                "share_single_frame": (
                    round(float(np.mean([p <= 1 for p in persistences])), 3)
                    if persistences else None
                ),
            },
            "minable_as_hard_negative": FalsePositiveKind(kind).minable,
        }

    return {
        "schema_version": FP_TAXONOMY_VERSION,
        "n_false_positives": len(items),
        "n_frames": n_frames,
        "false_positives_per_frame": round(len(items) / max(1, n_frames), 4),
        "by_kind": table,
        "caveat": (
            "category names are inferred from image evidence, not verified. The "
            "physically-grounded ones -- inside a person box, on a line, off the "
            "pitch -- carry the decisions; the rest guide mining only."
        ),
    }
