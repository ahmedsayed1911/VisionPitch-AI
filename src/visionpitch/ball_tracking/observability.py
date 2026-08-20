"""Explicit ball observability modelling.

Phase 2D, Part 2.

The problem this exists to fix
-----------------------------
Every ball metric this project reports divides by "frames". That denominator
lumps together three completely different situations:

* the ball was in shot, unobstructed, and the detector missed it
* the ball was behind a player, or off the edge of the frame
* the camera cut to a replay and there is no pitch on screen at all

The first is a detector failure. The second and third are not failures of
anything -- they are the game and the broadcast. Scoring them identically makes
a good detector look bad on tightly-cut footage and hides real misses on wide
static shots. Worse, it makes "effective ball coverage" uninterpretable: you
cannot tell whether 43% means the detector is weak or the broadcast is busy.

This module labels every frame with *whether the ball could have been seen*,
using only evidence available without knowing where the ball is. It then lets
every downstream rate be reported twice: over all frames, and over frames where
the ball was plausibly observable.

What it must never do
---------------------
**It never produces a ball position.** Not an estimate, not a prior, not a hint
to the detector about where to look. Its entire output is a label per frame. If
it emitted coordinates it would become exactly the fabrication the trajectory
estimator is careful to avoid, laundered through a different module.

The distinction that matters is between *missed but likely visible* and
*genuinely unobservable*. Only the first is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from visionpitch.common.logging import get_logger

log = get_logger("ball_tracking.observability")

OBSERVABILITY_SCHEMA_VERSION = "1.0.0"


class Observability(str, Enum):
    """Whether the ball could have been detected in this frame."""

    #: pitch in shot, ball not near a player boundary, nothing obscuring it
    LIKELY_VISIBLE = "likely_visible"
    #: the ball's expected location is inside or under a player box
    LIKELY_OCCLUDED = "likely_occluded"
    #: trajectory heads off the frame, or the last sighting was at the edge
    LIKELY_OUTSIDE_FRAME = "likely_outside_frame"
    #: a dense cluster of players where the ball is statistically hidden
    LIKELY_HIDDEN_BY_PLAYERS = "likely_hidden_by_players"
    #: camera motion or ball speed high enough to smear the ball
    LIKELY_MOTION_BLURRED = "likely_motion_blurred"
    #: replay, close-up, crowd shot -- no pitch geometry to work with
    NOT_ON_PITCH = "not_on_pitch"
    #: evidence is insufficient to say either way
    UNCERTAIN = "uncertain"

    @property
    def is_fair_denominator(self) -> bool:
        """Whether a missed ball in this frame counts as a detector failure.

        Occlusion, out-of-frame and off-pitch frames are excluded. A detector
        cannot find a ball that is not there to find, and counting those frames
        against it produces a metric that rewards footage choice over accuracy.
        """
        return self in (
            Observability.LIKELY_VISIBLE,
            Observability.LIKELY_MOTION_BLURRED,
            Observability.UNCERTAIN,
        )

    @property
    def is_explained_absence(self) -> bool:
        """Whether an absent ball here is expected rather than surprising."""
        return self in (
            Observability.LIKELY_OCCLUDED,
            Observability.LIKELY_OUTSIDE_FRAME,
            Observability.LIKELY_HIDDEN_BY_PLAYERS,
            Observability.NOT_ON_PITCH,
        )


@dataclass
class ObservabilityConfig:
    """Thresholds, each with the measurement that set it.

    ``player_density_radius_px`` and ``crowded_player_count`` come from the
    observed relationship between local player count and detection success: the
    ball is found far less often inside a cluster than in open play, and five
    players inside a 120 px radius is where that effect becomes the dominant
    term rather than a modifier.
    """

    #: expected ball position within this margin of the frame edge
    frame_edge_margin_px: float = 24.0
    #: players within this radius of the expected ball position count as a crowd
    player_density_radius_px: float = 120.0
    crowded_player_count: int = 5
    #: camera motion above this many pixels per frame smears a small object
    camera_motion_blur_px: float = 18.0
    #: a frame with fewer than this many pitch keypoints is probably not pitch
    min_pitch_keypoints: int = 3
    #: calibration confidence below this suggests no usable pitch in shot
    min_pitch_confidence: float = 0.10
    #: how many frames back a sighting may be and still constrain the expectation
    max_extrapolation_frames: int = 12
    #: cap on how far an expectation is propagated before it is worthless
    max_expected_drift_px: float = 300.0

    def to_dict(self) -> dict:
        return {
            "frame_edge_margin_px": self.frame_edge_margin_px,
            "player_density_radius_px": self.player_density_radius_px,
            "crowded_player_count": self.crowded_player_count,
            "camera_motion_blur_px": self.camera_motion_blur_px,
            "min_pitch_keypoints": self.min_pitch_keypoints,
            "min_pitch_confidence": self.min_pitch_confidence,
            "max_extrapolation_frames": self.max_extrapolation_frames,
            "max_expected_drift_px": self.max_expected_drift_px,
        }


@dataclass
class FrameObservability:
    """One frame's label, plus why."""

    frame_idx: int
    state: Observability
    #: 0..1 -- how strongly the evidence supports the label
    confidence: float = 0.0
    reason: str = ""
    #: how many frames since the last direct observation, if any
    frames_since_observation: int | None = None
    #: players near the expected position, when an expectation exists
    local_player_count: int | None = None

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "state": self.state.value,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "frames_since_observation": self.frames_since_observation,
            "local_player_count": self.local_player_count,
        }


@dataclass
class ObservabilityReport:
    """Per-frame labels and the summary rates they support."""

    frames: dict[int, FrameObservability] = field(default_factory=dict)
    config: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    schema_version: str = OBSERVABILITY_SCHEMA_VERSION

    def state_of(self, frame_idx: int) -> Observability:
        entry = self.frames.get(frame_idx)
        return entry.state if entry else Observability.UNCERTAIN

    @property
    def fair_frames(self) -> set[int]:
        """Frames where a missing ball is genuinely the detector's fault."""
        return {
            idx for idx, entry in self.frames.items()
            if entry.state.is_fair_denominator
        }

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {state.value: 0 for state in Observability}
        for entry in self.frames.values():
            tally[entry.state.value] += 1
        return tally

    def summary(self, observed_frames: set[int]) -> dict:
        """Coverage over all frames and over observable frames.

        ``observed_frames`` is the set with a *direct* ball observation.
        """
        total = len(self.frames)
        fair = self.fair_frames
        observed_fair = len(observed_frames & fair)
        explained = sum(
            1 for e in self.frames.values() if e.state.is_explained_absence
        )
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "n_frames": total,
            "counts": self.counts(),
            "n_observable_frames": len(fair),
            "observable_fraction": round(len(fair) / total, 4) if total else 0.0,
            "n_explained_absence": explained,
            "raw_ball_coverage": (
                round(len(observed_frames) / total, 4) if total else 0.0
            ),
            "observability_conditioned_coverage": (
                round(observed_fair / len(fair), 4) if fair else 0.0
            ),
            "note": (
                "raw coverage divides by every frame; conditioned coverage divides "
                "only by frames in which the ball could plausibly have been seen. "
                "The gap between them is broadcast difficulty, not detector error."
            ),
        }


class ObservabilityEstimator:
    """Labels frames without ever proposing a ball position."""

    def __init__(self, config: ObservabilityConfig | None = None) -> None:
        self.cfg = config or ObservabilityConfig()

    # -- expectation ---------------------------------------------------------- #

    @staticmethod
    def _expected_position(
        last_seen: tuple[int, float, float] | None,
        last_velocity: tuple[float, float] | None,
        frame_idx: int,
        cfg: ObservabilityConfig,
    ) -> tuple[float, float] | None:
        """Where the ball would be if it kept going, or ``None``.

        This is used *only* to decide which part of the image to reason about.
        It is never written anywhere, never returned to a caller, and never
        reaches the trajectory estimator. Extrapolation is abandoned once it
        outruns either the frame budget or the drift cap, because an expectation
        that could be anywhere constrains nothing.
        """
        if last_seen is None:
            return None
        seen_idx, x, y = last_seen
        elapsed = frame_idx - seen_idx
        if elapsed <= 0 or elapsed > cfg.max_extrapolation_frames:
            return None
        if last_velocity is None:
            return (x, y)
        drift = float(np.hypot(last_velocity[0], last_velocity[1]) * elapsed)
        if drift > cfg.max_expected_drift_px:
            return None
        return (x + last_velocity[0] * elapsed, y + last_velocity[1] * elapsed)

    # -- per-frame labelling -------------------------------------------------- #

    def label_frame(
        self,
        frame_idx: int,
        frame_size: tuple[int, int],
        expected: tuple[float, float] | None,
        player_boxes: np.ndarray,
        camera_motion_px: float,
        calibration_confidence: float,
        n_pitch_keypoints: int,
        frames_since_observation: int | None,
    ) -> FrameObservability:
        width, height = frame_size

        # 1. Is there any pitch in shot? A replay or crowd cutaway is not a
        #    frame in which the detector failed.
        if (
            n_pitch_keypoints < self.cfg.min_pitch_keypoints
            and calibration_confidence < self.cfg.min_pitch_confidence
        ):
            return FrameObservability(
                frame_idx, Observability.NOT_ON_PITCH, 0.8,
                f"{n_pitch_keypoints} pitch keypoints, calibration "
                f"{calibration_confidence:.2f}",
                frames_since_observation,
            )

        # 2. Without an expectation there is nothing location-specific to say.
        #    Camera motion is still frame-wide evidence, so it is checked first.
        if camera_motion_px >= self.cfg.camera_motion_blur_px:
            return FrameObservability(
                frame_idx, Observability.LIKELY_MOTION_BLURRED, 0.6,
                f"camera motion {camera_motion_px:.1f} px/frame",
                frames_since_observation,
            )

        if expected is None:
            return FrameObservability(
                frame_idx, Observability.UNCERTAIN, 0.3,
                "no recent sighting to extrapolate from",
                frames_since_observation,
            )

        ex, ey = expected

        # 3. Has it left the picture?
        margin = self.cfg.frame_edge_margin_px
        if not (margin <= ex <= width - margin and margin <= ey <= height - margin):
            return FrameObservability(
                frame_idx, Observability.LIKELY_OUTSIDE_FRAME, 0.7,
                f"expected position ({ex:.0f}, {ey:.0f}) outside the frame margin",
                frames_since_observation,
            )

        # 4. Is a player on top of it?
        inside = 0
        nearby = 0
        for box in player_boxes:
            bx1, by1, bx2, by2 = box[:4]
            if bx1 <= ex <= bx2 and by1 <= ey <= by2:
                inside += 1
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if np.hypot(cx - ex, cy - ey) <= self.cfg.player_density_radius_px:
                nearby += 1

        if inside:
            return FrameObservability(
                frame_idx, Observability.LIKELY_OCCLUDED, 0.75,
                f"expected position inside {inside} player box(es)",
                frames_since_observation, nearby,
            )
        if nearby >= self.cfg.crowded_player_count:
            return FrameObservability(
                frame_idx, Observability.LIKELY_HIDDEN_BY_PLAYERS, 0.6,
                f"{nearby} players within {self.cfg.player_density_radius_px:.0f} px",
                frames_since_observation, nearby,
            )

        return FrameObservability(
            frame_idx, Observability.LIKELY_VISIBLE, 0.7,
            "pitch in shot, expected position clear of players and frame edge",
            frames_since_observation, nearby,
        )

    # -- sequence ------------------------------------------------------------- #

    def label_sequence(
        self,
        frame_indices: list[int],
        frame_size: tuple[int, int],
        ball_observations: dict[int, tuple[float, float]],
        player_boxes_by_frame: dict[int, np.ndarray],
        camera_motion_by_frame: dict[int, float] | None = None,
        calibration_confidence_by_frame: dict[int, float] | None = None,
        pitch_keypoints_by_frame: dict[int, int] | None = None,
        cut_frames: set[int] | None = None,
    ) -> ObservabilityReport:
        """Label a whole clip.

        ``ball_observations`` holds only **direct** observations. Interpolated
        or recovered positions are deliberately excluded: letting them seed the
        expectation would let the model justify its own guesses.
        """
        camera_motion_by_frame = camera_motion_by_frame or {}
        calibration_confidence_by_frame = calibration_confidence_by_frame or {}
        pitch_keypoints_by_frame = pitch_keypoints_by_frame or {}
        cuts = cut_frames or set()

        report = ObservabilityReport(config=self.cfg)
        last_seen: tuple[int, float, float] | None = None
        last_velocity: tuple[float, float] | None = None

        for frame_idx in sorted(frame_indices):
            # A camera cut invalidates everything spatial that came before it.
            if frame_idx in cuts:
                last_seen = None
                last_velocity = None

            observation = ball_observations.get(frame_idx)
            if observation is not None:
                if last_seen is not None and frame_idx > last_seen[0]:
                    gap = frame_idx - last_seen[0]
                    last_velocity = (
                        (observation[0] - last_seen[1]) / gap,
                        (observation[1] - last_seen[2]) / gap,
                    )
                last_seen = (frame_idx, observation[0], observation[1])
                report.frames[frame_idx] = FrameObservability(
                    frame_idx, Observability.LIKELY_VISIBLE, 1.0,
                    "ball directly observed in this frame", 0,
                )
                continue

            expected = self._expected_position(
                last_seen, last_velocity, frame_idx, self.cfg
            )
            report.frames[frame_idx] = self.label_frame(
                frame_idx=frame_idx,
                frame_size=frame_size,
                expected=expected,
                player_boxes=player_boxes_by_frame.get(frame_idx, np.zeros((0, 4))),
                camera_motion_px=camera_motion_by_frame.get(frame_idx, 0.0),
                calibration_confidence=calibration_confidence_by_frame.get(frame_idx, 1.0),
                n_pitch_keypoints=pitch_keypoints_by_frame.get(frame_idx, 32),
                frames_since_observation=(
                    frame_idx - last_seen[0] if last_seen else None
                ),
            )

        counts = report.counts()
        log.info(
            "observability over %d frame(s): %s",
            len(report.frames),
            {k: v for k, v in counts.items() if v},
        )
        return report
