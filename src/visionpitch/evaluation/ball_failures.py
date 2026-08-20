"""Ball failure taxonomy and the image evidence that assigns it.

Phase 2D, Part 1.

Every ground-truth ball the detector missed is placed in exactly one category by
a fixed priority order. Two design decisions matter for how the results read:

**Priority, not multi-label.** A 40 px^2 ball behind a defender satisfies both
"tiny" and "occluded". It is reported as occluded, because a detector that gets
better at tiny balls still cannot see a ball that is not visible. The order
below runs from *the ball was not there to be seen* down to *the ball was there
and the detector failed*, so the buckets nearest the bottom are the ones a
better model can actually fix.

**Threshold rejection is separated from blindness.** Each image is scored twice,
at the operating confidence and at a floor of 0.001. A ball with a candidate at
the floor but not at the operating point was found and then rejected -- fixable
by calibration, and cheap. A ball with no candidate anywhere was not seen at
all. Phase 2B's taxonomy conflated these and concluded "detector blindness"
without being able to measure how much of it was really thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

#: Below this the ball is smaller than roughly 12x12 px. Measured on the
#: multi-corpus test split: 63.7% of Roboflow balls and 27.0% of SN-GSR balls
#: fall here, and Phase 2B measured recall of 0.00 in this bucket.
TINY_AREA_PX2 = 150.0
SMALL_AREA_PX2 = 400.0
MEDIUM_AREA_PX2 = 2000.0

#: Variance of the Laplacian below this is a blurred patch. Calibrated against
#: the distribution over ball patches rather than the usual whole-image value,
#: because an 11 px patch has far less structure than a full frame.
BLUR_VARIANCE = 40.0

#: Michelson contrast between the ball patch and its surrounding ring.
LOW_CONTRAST = 0.15

#: A ball centre this far inside a player box is occluded rather than merely
#: nearby. Player boxes include a lot of air, so containment is required.
OCCLUSION_INSET_PX = 2.0

#: Ball centre within this many pixels of a player box counts as near-player
#: confusion territory.
NEAR_PLAYER_PX = 20.0


class FailureCategory(str, Enum):
    """One bucket per ground-truth ball."""

    DETECTED = "detected"

    # -- the ball was not available to be detected --------------------------- #
    OUTSIDE_FRAME = "ball_outside_visible_frame"
    PLAYER_OCCLUSION = "player_occlusion"
    GENUINELY_UNOBSERVABLE = "genuinely_unobservable"

    # -- present but degraded ------------------------------------------------ #
    MOTION_BLUR = "motion_blur"
    LOW_CONTRAST = "low_contrast"
    TINY_SCALE = "tiny_scale_ball"
    NEAR_PLAYER_CONFUSION = "near_player_confusion"
    PITCH_LINE_CONFUSION = "pitch_line_confusion"

    # -- present and clean; the pipeline lost it ----------------------------- #
    DETECTOR_THRESHOLD_REJECTION = "detector_threshold_rejection"
    DETECTOR_MISS_UNEXPLAINED = "detector_miss_unexplained"


#: Categories a better *detector* can address, as opposed to ones describing a
#: ball that was never visible. Reported separately so the headline "detector
#: blindness" figure is not inflated by frames where nothing could be done.
ADDRESSABLE = frozenset({
    FailureCategory.MOTION_BLUR,
    FailureCategory.LOW_CONTRAST,
    FailureCategory.TINY_SCALE,
    FailureCategory.NEAR_PLAYER_CONFUSION,
    FailureCategory.PITCH_LINE_CONFUSION,
    FailureCategory.DETECTOR_THRESHOLD_REJECTION,
    FailureCategory.DETECTOR_MISS_UNEXPLAINED,
})


def size_bucket(area_px2: float) -> str:
    if area_px2 < TINY_AREA_PX2:
        return "1_tiny_lt150"
    if area_px2 < SMALL_AREA_PX2:
        return "2_small_150_400"
    if area_px2 < MEDIUM_AREA_PX2:
        return "3_medium_400_2000"
    return "4_large_gt2000"


@dataclass
class GroundTruthBall:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class FrameEvidence:
    """Measurements at the ball's location, used only to explain a miss."""

    blur_variance: float
    contrast: float
    inside_player_box: bool
    nearest_player_px: float
    line_strength: float
    touches_frame_edge: bool
    floor_distance_px: float
    floor_confidence: float
    match_px: float

    @property
    def found_below_threshold(self) -> bool:
        """A candidate existed at the confidence floor but not at operating conf."""
        return self.floor_distance_px <= self.match_px

    @staticmethod
    def measure(
        frame: np.ndarray,
        truth: GroundTruthBall,
        player_boxes: np.ndarray,
        floor_distance_px: float,
        floor_confidence: float,
        match_px: float,
    ) -> FrameEvidence:
        height, width = frame.shape[:2]
        radius = max(4, int(round(max(truth.w, truth.h))))

        x1 = int(max(0, truth.cx - radius))
        y1 = int(max(0, truth.cy - radius))
        x2 = int(min(width, truth.cx + radius))
        y2 = int(min(height, truth.cy + radius))
        patch = frame[y1:y2, x1:x2]

        if patch.size == 0:
            return FrameEvidence(
                blur_variance=0.0, contrast=0.0, inside_player_box=False,
                nearest_player_px=float("inf"), line_strength=0.0,
                touches_frame_edge=True, floor_distance_px=floor_distance_px,
                floor_confidence=floor_confidence, match_px=match_px,
            )

        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(grey, cv2.CV_64F).var())

        # Michelson contrast between the ball's own pixels and a surrounding
        # ring, which is what a small-object detector actually keys on.
        inner = max(1, int(round(min(truth.w, truth.h) / 2)))
        centre_y, centre_x = grey.shape[0] // 2, grey.shape[1] // 2
        core = grey[
            max(0, centre_y - inner): centre_y + inner + 1,
            max(0, centre_x - inner): centre_x + inner + 1,
        ]
        ring_mask = np.ones(grey.shape, dtype=bool)
        ring_mask[
            max(0, centre_y - inner): centre_y + inner + 1,
            max(0, centre_x - inner): centre_x + inner + 1,
        ] = False
        core_mean = float(core.mean()) if core.size else 0.0
        ring_mean = float(grey[ring_mask].mean()) if ring_mask.any() else 0.0
        denominator = core_mean + ring_mean
        contrast = abs(core_mean - ring_mean) / denominator if denominator > 0 else 0.0

        # Straight-line structure near the ball: pitch markings are the most
        # common thing a ball detector confuses a ball with.
        edges = cv2.Canny(grey, 60, 180)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=max(8, radius),
            minLineLength=max(8, radius), maxLineGap=3,
        )
        line_strength = float(len(lines)) if lines is not None else 0.0

        inside = False
        nearest = float("inf")
        for box in player_boxes:
            bx1, by1, bx2, by2 = box[:4]
            if (
                bx1 + OCCLUSION_INSET_PX <= truth.cx <= bx2 - OCCLUSION_INSET_PX
                and by1 + OCCLUSION_INSET_PX <= truth.cy <= by2 - OCCLUSION_INSET_PX
            ):
                inside = True
                nearest = 0.0
                break
            dx = max(bx1 - truth.cx, 0.0, truth.cx - bx2)
            dy = max(by1 - truth.cy, 0.0, truth.cy - by2)
            nearest = min(nearest, float(np.hypot(dx, dy)))

        margin = max(truth.w, truth.h)
        touches_edge = (
            truth.cx - truth.w / 2 <= margin
            or truth.cy - truth.h / 2 <= margin
            or truth.cx + truth.w / 2 >= width - margin
            or truth.cy + truth.h / 2 >= height - margin
        )

        return FrameEvidence(
            blur_variance=blur_variance,
            contrast=contrast,
            inside_player_box=inside,
            nearest_player_px=nearest,
            line_strength=line_strength,
            touches_frame_edge=touches_edge,
            floor_distance_px=floor_distance_px,
            floor_confidence=floor_confidence,
            match_px=match_px,
        )


def classify_miss(truth: GroundTruthBall, evidence: FrameEvidence) -> FailureCategory:
    """Assign one category, most-specific first.

    The order is the argument of this module: causes that make the ball
    unavailable come before causes that make it hard, which come before "the
    detector simply missed it". Reversing it would report almost everything as
    a tiny-scale failure, since almost every ball is tiny.
    """
    # 1. Was it even there to be seen?
    if evidence.inside_player_box:
        return FailureCategory.PLAYER_OCCLUSION
    if evidence.touches_frame_edge:
        return FailureCategory.OUTSIDE_FRAME

    # 2. Was it found and then thrown away? Cheapest possible fix, so it is
    #    tested before any explanation that blames the image.
    if evidence.found_below_threshold:
        return FailureCategory.DETECTOR_THRESHOLD_REJECTION

    # 3. Degraded but present.
    if evidence.blur_variance < BLUR_VARIANCE and evidence.contrast < LOW_CONTRAST:
        return FailureCategory.GENUINELY_UNOBSERVABLE
    if evidence.blur_variance < BLUR_VARIANCE:
        return FailureCategory.MOTION_BLUR
    if evidence.contrast < LOW_CONTRAST:
        return FailureCategory.LOW_CONTRAST
    if evidence.nearest_player_px <= NEAR_PLAYER_PX:
        return FailureCategory.NEAR_PLAYER_CONFUSION
    if evidence.line_strength >= 3.0:
        return FailureCategory.PITCH_LINE_CONFUSION
    if truth.area < TINY_AREA_PX2:
        return FailureCategory.TINY_SCALE

    # 4. Visible, in focus, contrasty, clear of players and lines, big enough.
    return FailureCategory.DETECTOR_MISS_UNEXPLAINED
