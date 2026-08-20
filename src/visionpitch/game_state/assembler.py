"""Assembles the canonical per-frame game state.

This is where the four independent evidence streams -- tracks, ball trajectory,
team labels, calibration -- are joined into the single table Phase 2 consumes.

The join is where honesty about uncertainty is either preserved or lost, so the
rules are explicit:

* a track observation under an invalid homography keeps its image coordinates
  and gets ``pitch_x = pitch_y = NULL``. It is not dropped -- the player was
  genuinely there -- and it is not projected with a guessed homography.
* the four confidences stay separate all the way into the table. Collapsing them
  into one number would make it impossible for Phase 2 to distinguish "this
  player's position is precise but their team is a guess" from the reverse.
* people are projected from their **ground contact point**, the ball from its
  centre. A ball in flight is above the ground plane, so its projected pitch
  position is systematically wrong by an amount proportional to its height --
  which is why the ball's row is flagged and Phase 2 must treat aerial phases
  with care. Recovering true ball height needs a second view or a ballistic
  model, and is out of Phase 1 scope.
"""

from __future__ import annotations

import numpy as np

from visionpitch.calibration.propagation import extrapolation_risk
from visionpitch.common.geometry import apply_homography
from visionpitch.common.logging import StageCounters, get_logger
from visionpitch.common.schema import GameStateRow
from visionpitch.common.types import (
    BallState,
    CalibrationResult,
    ObjectClass,
    Role,
    SegmentKind,
    TeamId,
    Track,
    ValidationStatus,
)
from visionpitch.pitch.geometry import PitchConfiguration

log = get_logger("game_state")


class GameStateAssembler:
    """Joins every stage's output into :class:`GameStateRow` records."""

    def __init__(
        self,
        video_id: str,
        pitch: PitchConfiguration,
        min_calibration_confidence: float,
        support_regions: dict[int, tuple[float, float, float, float]] | None = None,
        max_extrapolation_risk: float = 0.35,
    ) -> None:
        self.video_id = video_id
        self.pitch = pitch
        self.min_calibration_confidence = min_calibration_confidence
        #: per-frame image region the homography was actually constrained over
        self.support_regions = support_regions or {}
        self.max_extrapolation_risk = max_extrapolation_risk
        self.counters = StageCounters("game_state")

    # -- projection --------------------------------------------------------- #

    def _project(
        self, calibration: CalibrationResult | None, point: tuple[float, float]
    ) -> tuple[float | None, float | None, float | None, float | None, float]:
        """Project a point, returning coordinates plus an extrapolation risk.

        Returns ``(x, y, nx, ny, risk)``. ``risk`` is 0 where the homography was
        constrained by landmarks and rises toward 1 outside that region; callers
        turn it into a validation status rather than silently dropping the row.

        Previously, points outside a hard containment margin were discarded
        entirely, which is why far-side players had 53% coverage against 95%
        near the camera — the rows vanished without explanation. Keeping them
        with an explicit risk preserves the information and still lets Phase 2
        exclude them.
        """
        if calibration is None or not calibration.is_valid:
            return None, None, None, None, 1.0

        projected = apply_homography(calibration.homography, np.array([point]))[0]
        if not np.isfinite(projected).all():
            self.counters.warn("projection_behind_horizon")
            return None, None, None, None, 1.0

        x, y = float(projected[0]), float(projected[1])
        # A coordinate this far out is not an extrapolation, it is a broken
        # homography for this point, and no confidence annotation makes it
        # useful.
        if not self.pitch.contains(x, y, margin=60.0):
            self.counters.warn("projection_far_off_pitch")
            return None, None, None, None, 1.0

        risk = extrapolation_risk(point, self.support_regions.get(calibration.frame_idx))
        if risk > self.max_extrapolation_risk:
            self.counters.warn("projection_extrapolated")

        nx, ny = self.pitch.normalise(x, y)
        return x, y, nx, ny, risk

    def _status(
        self,
        calibration: CalibrationResult | None,
        interpolated: bool,
        min_confidence: float,
        extrapolation: float = 0.0,
    ) -> ValidationStatus:
        if interpolated:
            return ValidationStatus.INTERPOLATED
        if calibration is None or not calibration.is_valid:
            return ValidationStatus.NO_CALIBRATION
        if calibration.segment_kind in (SegmentKind.REPLAY, SegmentKind.CLOSE_UP):
            return ValidationStatus.NON_LIVE
        # Extrapolation is reported ahead of low confidence: a well-solved
        # homography evaluated far outside its evidence is the more specific and
        # more actionable problem of the two.
        if extrapolation > self.max_extrapolation_risk:
            return ValidationStatus.EXTRAPOLATED
        if calibration.confidence < min_confidence:
            return ValidationStatus.LOW_CALIBRATION
        return ValidationStatus.VALID

    # -- assembly ----------------------------------------------------------- #

    def assemble(
        self,
        tracks: dict[int, Track],
        ball_states: dict[int, BallState],
        calibration: dict[int, CalibrationResult],
        timestamps: dict[int, float],
        frame_indices: list[int],
    ) -> list[GameStateRow]:
        """Produce every row for the processed frame range."""
        # Index track observations by frame so the outer loop is over frames,
        # which keeps rows grouped by frame in the written table and makes
        # Phase 2's per-frame scans sequential rather than random-access.
        by_frame: dict[int, list[tuple[Track, object]]] = {}
        for track in tracks.values():
            for obs in track.observations:
                by_frame.setdefault(obs.frame_idx, []).append((track, obs))

        rows: list[GameStateRow] = []
        for frame_idx in frame_indices:
            timestamp = timestamps.get(frame_idx, 0.0)
            calib = calibration.get(frame_idx)
            segment = calib.segment_kind if calib else SegmentKind.UNKNOWN
            calib_conf = calib.confidence if calib else 0.0

            for track, obs in by_frame.get(frame_idx, []):
                rows.append(
                    self._person_row(
                        track, obs, frame_idx, timestamp, calib, calib_conf, segment
                    )
                )

            ball = ball_states.get(frame_idx)
            if ball is not None and ball.position is not None:
                rows.append(
                    self._ball_row(ball, frame_idx, timestamp, calib, calib_conf, segment)
                )

            self.counters.ok()

        log.info("assembled %d game-state rows over %d frames", len(rows), len(frame_indices))
        return rows

    def _person_row(
        self,
        track: Track,
        obs,
        frame_idx: int,
        timestamp: float,
        calib: CalibrationResult | None,
        calib_conf: float,
        segment: SegmentKind,
    ) -> GameStateRow:
        image_x, image_y = obs.bbox.ground_contact
        pitch_x, pitch_y, nx, ny, risk = self._project(calib, (image_x, image_y))

        return GameStateRow(
            video_id=self.video_id,
            frame_idx=frame_idx,
            timestamp_s=timestamp,
            match_clock_s=None,
            object_class=track.object_class.value,
            track_id=track.track_id,
            team_id=track.team_id.value,
            role=track.role.value,
            jersey_number=track.jersey_number,
            jersey_confidence=track.jersey_confidence,
            bbox_x1=obs.bbox.x1,
            bbox_y1=obs.bbox.y1,
            bbox_x2=obs.bbox.x2,
            bbox_y2=obs.bbox.y2,
            image_x=image_x,
            image_y=image_y,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            pitch_x_norm=nx,
            pitch_y_norm=ny,
            detection_confidence=obs.det_confidence,
            tracking_confidence=obs.track_confidence,
            team_confidence=track.team_confidence,
            calibration_confidence=calib_conf,
            interpolated=obs.interpolated,
            validation_status=self._status(
                calib, obs.interpolated, self.min_calibration_confidence, risk
            ).value,
            segment_kind=segment.value,
            source="tracker",
        )

    def _ball_row(
        self,
        ball: BallState,
        frame_idx: int,
        timestamp: float,
        calib: CalibrationResult | None,
        calib_conf: float,
        segment: SegmentKind,
    ) -> GameStateRow:
        image_x, image_y = ball.position
        pitch_x, pitch_y, nx, ny, risk = self._project(calib, (image_x, image_y))
        bbox = ball.bbox

        return GameStateRow(
            video_id=self.video_id,
            frame_idx=frame_idx,
            timestamp_s=timestamp,
            match_clock_s=None,
            object_class=ObjectClass.BALL.value,
            track_id=None,
            team_id=TeamId.NONE.value,
            role=Role.BALL.value,
            jersey_number=None,
            jersey_confidence=0.0,
            bbox_x1=bbox.x1 if bbox else image_x,
            bbox_y1=bbox.y1 if bbox else image_y,
            bbox_x2=bbox.x2 if bbox else image_x,
            bbox_y2=bbox.y2 if bbox else image_y,
            image_x=image_x,
            image_y=image_y,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            pitch_x_norm=nx,
            pitch_y_norm=ny,
            detection_confidence=ball.confidence if ball.observed else 0.0,
            tracking_confidence=ball.confidence,
            team_confidence=0.0,
            calibration_confidence=calib_conf,
            interpolated=ball.interpolated,
            validation_status=self._status(
                calib, ball.interpolated, self.min_calibration_confidence, risk
            ).value,
            segment_kind=segment.value,
            source="ball_trajectory",
        )

    # -- reporting ---------------------------------------------------------- #

    @staticmethod
    def quality_report(rows: list[GameStateRow]) -> dict:
        """Data-quality summary. The numbers a reviewer should read first."""
        if not rows:
            return {"rows": 0}

        total = len(rows)
        people = [r for r in rows if r.object_class != ObjectClass.BALL.value]
        with_pitch = sum(1 for r in rows if r.pitch_x is not None)
        interpolated = sum(1 for r in rows if r.interpolated)

        status_counts: dict[str, int] = {}
        for row in rows:
            status_counts[row.validation_status] = status_counts.get(row.validation_status, 0) + 1

        frames = {r.frame_idx for r in rows}
        return {
            "rows": total,
            "frames": len(frames),
            "person_rows": len(people),
            "ball_rows": total - len(people),
            "rows_with_pitch_coordinates": with_pitch,
            "pitch_coordinate_ratio": round(with_pitch / total, 4),
            "interpolated_rows": interpolated,
            "interpolated_ratio": round(interpolated / total, 4),
            "validation_status_counts": status_counts,
            "mean_people_per_frame": (
                round(len(people) / max(1, len(frames)), 2)
            ),
        }
