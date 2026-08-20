"""Temporal possession state machine.

Why not nearest player
----------------------
Nearest-player possession is wrong in the situations that matter most. A
defender standing two metres from a ball rolling away from them is not in
possession. Two players a metre apart contesting a bouncing ball are not "the
closer one's possession". A ball travelling at 15 m/s past a stationary player
is nobody's. The nearest-player rule assigns confident possession in all three
cases, and those are precisely the moments that generate the pass, turnover and
interception events downstream.

The evidence actually used
--------------------------
* distance from player to ball, in metres
* the ball's speed, and whether it is moving *with* the player or past them
* how long the candidate has been the nearest player -- control is temporal
* whether a second player is close enough to contest
* who held the ball previously, which resists single-frame flicker

The state is then smoothed with a minimum dwell time, because a possession that
lasts three frames is a detection artefact, not a touch.

Why proximity is measured in image space
----------------------------------------
Not in metres, deliberately, and this was a measured decision rather than a
convenience. Projecting the ball onto the pitch assumes it is *on the ground*.
It frequently is not, and a lofted ball's ground-plane projection races across
the pitch at a speed proportional to its height. Measured on the validation
clip, the ball's projected pitch speed has a **median of 34 m/s** -- about
123 km/h, sustained -- and 83% of frames exceed any sane "the ball is
travelling" threshold. Possession decided in that space reports the ball as
loose for 30 of 42 seconds.

Image-space distance has the opposite problem: it is not scale-invariant, since
50 px near the camera is a different real distance from 50 px at the far
touchline. The fix is to normalise by the **player's bounding-box height**,
which is a direct measurement of that player's depth: a footballer is about
1.8 m tall, so a distance of one box-height is about 1.8 m wherever they stand.

The result is a proximity measure that needs no homography at all, and is
therefore immune to both the calibration error and the ball-height error that
make the metric version unusable. Pitch coordinates are still used for event
*geometry* -- pass length, progression -- where the ball is on the ground and
the errors are tolerable.

Unknown is first class
----------------------
When the ball's position is unknown the state is ``UNKNOWN``. It is never
carried forward from the last known holder, and it is never interpolated. On
the validation clip the ball is unknown in roughly 9% of frames, and those
frames must not silently become possession for whoever last had it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from visionpitch.analytics.context import AnalysisContext
from visionpitch.analytics.types import (
    BallStateKind,
    Metric,
    MetricBasis,
    PossessionSpan,
    PossessionState,
    is_team,
)
from visionpitch.common.logging import get_logger

log = get_logger("analytics.possession")


@dataclass
class PossessionConfig:
    """Thresholds, in metres and seconds so they are physically meaningful."""

    # Distances are in units of the nearest player's bounding-box height, which
    # is roughly 1.8 m of real distance regardless of where on the pitch the
    # player stands. See the module docstring for why this is not in metres.
    #: Within this many player-heights the player can plausibly be in control.
    #:
    #: Measured, not chosen. Swept against the SN-GSR possession reference over
    #: the 34 training sequences of the Phase 2C clip-disjoint split: team F1
    #: peaks at 0.6 (0.751) with a plateau from 0.5 to 0.8, and falls away in
    #: both directions -- 0.589 at 2.0, 0.350 at 0.2. The previous default of
    #: 1.6 scored 0.620, because a radius that generous calls the ball
    #: controlled while it is still in flight: at 1.6 the engine produced 1,179
    #: false negatives on loose balls and *zero* false positives, meaning it
    #: almost never admitted the ball was free.
    control_radius_heights: float = 0.6
    #: beyond this many player-heights nobody is in contact with the ball
    loose_radius_heights: float = 4.0
    #: A second player this much closer makes the ball contested.
    #:
    #: Kept below ``control_radius_heights``. Above it the test is vacuous --
    #: two players both inside the radius can differ by at most the radius, so
    #: any opponent in contact range would be a contest.
    contest_margin_heights: float = 0.3
    #: above this the ball is travelling, in player-heights per second
    travelling_speed_heights_s: float = 6.0
    #: a candidate must hold the ball this long before it counts as control
    min_control_s: float = 0.20
    #: possession spans shorter than this are smoothed away as flicker
    min_span_s: float = 0.16
    #: ball further outside the touchline than this is out of play
    out_of_play_margin_m: float = 1.5
    #: Calibration confidence required before an out-of-play call is believed.
    #:
    #: Measured on the SN-BAS segment: 19.5% of projected ball positions land
    #: between the touchline and the implausibility margin, against a ground
    #: truth of 2 balls out in three minutes. A single frame's projection
    #: drifting past a line is not evidence that play stopped, and taking it at
    #: face value produced 36 ball-out events where 2 occurred.
    out_of_play_min_calibration: float = 0.45
    #: The ball must stay outside for this long before play is called dead.
    out_of_play_min_duration_s: float = 0.40
    #: Beyond this the projection is broken, not the ball genuinely out.
    #:
    #: On the validation clip 25% of ball projections land off the pitch, with
    #: an extreme of x = 163.9 m on a 105 m pitch. A ball 58 m past the goal line
    #: is a calibration failure; calling it 'out of play' manufactures dead-ball
    #: time and, worse, a restart event afterwards. Positions beyond this margin
    #: are reported as UNKNOWN, which is what they are.
    implausible_projection_margin_m: float = 15.0
    #: interpolated ball positions are usable but weigh less in confidence
    interpolated_confidence_factor: float = 0.6


@dataclass
class FramePossession:
    """Per-frame possession decision, before smoothing."""

    frame_idx: int
    timestamp_s: float
    state: PossessionState
    team_id: str = "unknown"
    track_id: int | None = None
    confidence: float = 0.0
    ball_state: BallStateKind = BallStateKind.UNKNOWN
    #: in units of the nearest player's bounding-box height (~1.8 m each)
    nearest_distance_heights: float | None = None
    contest_distance_heights: float | None = None
    ball_speed_heights_s: float | None = None


class PossessionEngine:
    """Assigns possession per frame, then consolidates into spans."""

    def __init__(self, context: AnalysisContext, config: PossessionConfig | None = None):
        self.ctx = context
        self.cfg = config or PossessionConfig()
        self._calib_conf: dict[int, float] | None = None
        self._ball_image = self._index_ball_image()
        self._positions = self._index_positions()
        self._ball_speed = self._compute_ball_speed()

    def _index_ball_image(self) -> dict[int, tuple[float, float]]:
        """Ball centre in image pixels for every frame it was located in."""
        out: dict[int, tuple[float, float]] = {}
        for row in self.ctx.ball.itertuples(index=False):
            out[int(row.frame_idx)] = (float(row.image_x), float(row.image_y))
        return out

    def _calibration_confidence(self, frame_idx: int) -> float:
        """Calibration confidence for a frame, from the stored frames table."""
        if self._calib_conf is None:
            frames = self.ctx.frames
            self._calib_conf = (
                dict(zip(frames.frame_idx.astype(int),
                         frames.calibration_confidence.astype(float), strict=True))
                if "calibration_confidence" in frames.columns else {}
            )
        return self._calib_conf.get(frame_idx, 0.0)

    def _sustained_out_of_play(self, frame_idx: int) -> bool:
        """Whether the ball stays outside the pitch, not just this frame.

        A projection that wanders past the touchline for a frame or two is
        noise; play stopping is a state that persists.
        """
        need = max(1, int(self.cfg.out_of_play_min_duration_s * self.ctx.fps))
        checked = outside = 0
        for candidate in range(frame_idx, frame_idx + need * 2):
            position = self.ctx.ball_position(candidate)
            if position is None:
                continue
            checked += 1
            if self._is_out_of_play(*position) and not self._is_implausible(*position):
                outside += 1
            if checked >= need:
                break
        return checked > 0 and outside >= max(1, int(0.7 * checked))

    def _reference_height(self, frame_idx: int) -> float:
        """Median player box height in a frame, as the local depth scale.

        Used to convert an image-space speed into player-heights per second.
        The median rather than the nearest player's own height, because the ball
        may be nowhere near anyone and the scale still has to be defined.
        """
        candidates = self._positions.get(frame_idx, [])
        if not candidates:
            return 0.0
        return float(np.median([c[3] for c in candidates]))

    # -- indexing ------------------------------------------------------------ #

    def _index_positions(self) -> dict[int, list[tuple[int, float, float, float, str]]]:
        """frame -> [(track_id, image_x, image_y, box_height_px, team)].

        Every tracked person is included, whatever their calibration status:
        proximity is judged in image space, so a row with no usable pitch
        coordinate still answers "was this player near the ball". That matters
        because only a third of person rows carry trustworthy pitch coordinates.
        """
        out: dict[int, list[tuple[int, float, float, float, str]]] = {}
        for row in self.ctx.players.itertuples(index=False):
            if pd.isna(row.track_id):
                continue
            height = float(row.bbox_y2 - row.bbox_y1)
            if height <= 1.0:
                continue
            out.setdefault(int(row.frame_idx), []).append(
                (
                    int(row.track_id),
                    float(row.image_x),
                    float(row.image_y),
                    height,
                    str(row.team_id),
                )
            )
        return out

    def _ball_image_position(self, frame_idx: int) -> tuple[float, float] | None:
        """Ball centre in image pixels, or ``None`` when not known."""
        return self._ball_image.get(frame_idx)

    def _compute_ball_speed(self) -> dict[int, float]:
        """Ball speed in player-heights per second, from its image trajectory.

        Image space, not pitch space. The ball's ground-plane projection races
        across the pitch whenever the ball is airborne -- measured median 34 m/s
        on the validation clip -- so a metric speed answers a question about the
        ball's height as much as about its speed. Its image motion does not have
        that failure mode.

        Smoothed over contiguous runs with the same Kalman/RTS smoother used for
        players, in pixels. Runs are never bridged: a gap means the ball's path
        is unknown, and interpolating across it would invent exactly what the
        Phase 1 ball estimator declined to invent.
        """
        from visionpitch.analytics.kinematics import _rts_smooth

        speeds: dict[int, float] = {}
        run: list[tuple[int, float, float, float]] = []

        def flush() -> None:
            if len(run) >= 3:
                times = np.array([r[1] for r in run])
                xs = np.array([r[2] for r in run])
                ys = np.array([r[3] for r in run])
                # Pixel-space noise for a small, motion-blurred object.
                _, _, vx, vy = _rts_smooth(
                    times, xs, ys, position_noise=6.0, process_accel=900.0
                )
                for (frame_idx, _, _, _), sx, sy in zip(run, vx, vy, strict=True):
                    reference = self._reference_height(frame_idx)
                    if reference > 0:
                        speeds[frame_idx] = float(np.hypot(sx, sy)) / reference
            run.clear()

        previous_frame: int | None = None
        for frame_idx in self.ctx.frame_indices:
            position = self._ball_image_position(frame_idx)
            if position is None or (
                previous_frame is not None and frame_idx - previous_frame > 3
            ):
                flush()
            if position is not None:
                run.append(
                    (frame_idx, self.ctx.timestamps.get(frame_idx, 0.0), *position)
                )
                previous_frame = frame_idx
            else:
                previous_frame = None
        flush()
        return speeds

    # -- per-frame decision --------------------------------------------------- #

    def _is_out_of_play(self, x: float, y: float) -> bool:
        margin = self.cfg.out_of_play_margin_m
        pitch = self.ctx.pitch
        return (
            x < -margin
            or x > pitch.length + margin
            or y < -margin
            or y > pitch.width + margin
        )

    def _is_implausible(self, x: float, y: float) -> bool:
        """Whether a projected ball position is too far out to be a real ball."""
        margin = self.cfg.implausible_projection_margin_m
        pitch = self.ctx.pitch
        return (
            x < -margin
            or x > pitch.length + margin
            or y < -margin
            or y > pitch.width + margin
        )

    def _decide(self, frame_idx: int, previous_owner: int | None,
                held_for_s: float) -> FramePossession:
        timestamp = self.ctx.timestamps.get(frame_idx, 0.0)
        ball_state = self.ctx.ball_state(frame_idx)
        image_position = self._ball_image_position(frame_idx)

        if image_position is None or not ball_state.is_known:
            return FramePossession(
                frame_idx, timestamp, PossessionState.UNKNOWN, ball_state=ball_state
            )

        # Out of play is judged in pitch space -- it is a question about the
        # touchline, which only exists there -- but only when the projection is
        # plausible enough to answer it. An implausible projection means the
        # homography failed, not that the ball left the field.
        pitch_position = self.ctx.ball_position(frame_idx)
        if (
            pitch_position is not None
            and not self._is_implausible(*pitch_position)
            and self._calibration_confidence(frame_idx)
            >= self.cfg.out_of_play_min_calibration
            and self._sustained_out_of_play(frame_idx)
        ):
            return FramePossession(
                frame_idx, timestamp, PossessionState.OUT_OF_PLAY,
                confidence=0.6, ball_state=ball_state,
            )

        bx, by = image_position
        candidates = self._positions.get(frame_idx, [])
        if not candidates:
            return FramePossession(
                frame_idx, timestamp, PossessionState.LOOSE_BALL,
                confidence=0.3, ball_state=ball_state,
            )

        # Distance in units of each player's own box height, which is that
        # player's depth scale. One height is about 1.8 m wherever they stand.
        distances = sorted(
            (
                (float(np.hypot(px - bx, py - by)) / height, tid, team)
                for tid, px, py, height, team in candidates
            ),
            key=lambda item: item[0],
        )
        nearest_distance, nearest_id, nearest_team = distances[0]
        contest_distance = distances[1][0] if len(distances) > 1 else float("inf")
        contest_team = distances[1][2] if len(distances) > 1 else ""
        ball_speed = self._ball_speed.get(frame_idx)

        base = FramePossession(
            frame_idx, timestamp, PossessionState.UNKNOWN,
            ball_state=ball_state,
            nearest_distance_heights=nearest_distance,
            contest_distance_heights=contest_distance,
            ball_speed_heights_s=ball_speed,
        )

        # A ball travelling fast is in flight between players, not under
        # anyone's control, however close someone happens to be standing.
        if ball_speed is not None and ball_speed > self.cfg.travelling_speed_heights_s:
            base.state = PossessionState.LOOSE_BALL
            base.confidence = 0.55
            return base

        if nearest_distance > self.cfg.loose_radius_heights:
            base.state = PossessionState.LOOSE_BALL
            base.confidence = float(np.clip(nearest_distance / 8.0, 0.3, 0.8))
            return base

        # Contested requires two *opponents* both genuinely within contact range
        # of the ball -- not merely two players whose distances happen to be
        # similar. Requiring only similar distances made 17 of 42 seconds
        # contested, because in a crowded frame the second-nearest player is
        # almost always within a comparable distance whether or not they are
        # anywhere near the ball. And two team-mates near the ball is possession
        # for their team, not a contest.
        both_in_contact = (
            nearest_distance <= self.cfg.control_radius_heights
            and contest_distance <= self.cfg.control_radius_heights
        )
        opposing = (
            is_team(nearest_team)
            and is_team(contest_team)
            and nearest_team != contest_team
        )
        if (
            both_in_contact
            and opposing
            and contest_distance - nearest_distance < self.cfg.contest_margin_heights
        ):
            base.state = PossessionState.CONTESTED
            base.team_id = "contested"
            base.confidence = 0.5
            return base

        if nearest_distance <= self.cfg.control_radius_heights:
            confidence = self._control_confidence(
                nearest_distance, contest_distance, ball_speed, ball_state,
                nearest_id == previous_owner, held_for_s,
            )
            base.state = PossessionState.CONTROLLED
            base.team_id = nearest_team
            base.track_id = nearest_id
            base.confidence = confidence
            return base

        base.state = PossessionState.LOOSE_BALL
        base.confidence = 0.45
        return base

    def _control_confidence(
        self,
        nearest_distance: float,
        contest_distance: float,
        ball_speed: float | None,
        ball_state: BallStateKind,
        is_previous_owner: bool,
        held_for_s: float,
    ) -> float:
        """Blend the independent signals into one control confidence.

        Distances are in player-heights; speed is in player-heights per second.
        """
        proximity = float(
            np.clip(1.0 - nearest_distance / self.cfg.control_radius_heights, 0, 1)
        )
        separation = float(np.clip((contest_distance - nearest_distance) / 2.0, 0, 1))
        stillness = (
            1.0 if ball_speed is None
            else float(np.clip(1.0 - ball_speed / self.cfg.travelling_speed_heights_s, 0, 1))
        )
        # Control is temporal: a player who has held the ball for half a second
        # is far more likely to be in possession than one who is merely nearest
        # this frame.
        duration = float(np.clip(held_for_s / 1.0, 0.0, 1.0))
        continuity = 1.0 if is_previous_owner else 0.6

        confidence = (
            0.35 * proximity + 0.20 * separation + 0.20 * stillness + 0.25 * duration
        ) * continuity

        if ball_state is BallStateKind.INTERPOLATED:
            confidence *= self.cfg.interpolated_confidence_factor
        return float(np.clip(confidence, 0.0, 1.0))

    # -- sequencing ----------------------------------------------------------- #

    def per_frame(self) -> list[FramePossession]:
        """Possession decision for every processed frame."""
        results: list[FramePossession] = []
        previous_owner: int | None = None
        held_since: float | None = None

        for frame_idx in self.ctx.frame_indices:
            timestamp = self.ctx.timestamps.get(frame_idx, 0.0)
            held_for = (timestamp - held_since) if held_since is not None else 0.0
            decision = self._decide(frame_idx, previous_owner, held_for)

            if decision.state is PossessionState.CONTROLLED:
                if decision.track_id != previous_owner:
                    held_since = timestamp
                previous_owner = decision.track_id
            elif decision.state in (PossessionState.LOOSE_BALL, PossessionState.OUT_OF_PLAY):
                # Deliberately keep ``previous_owner`` through UNKNOWN and
                # CONTESTED: a brief loss of the ball's position should not
                # reset the continuity bonus for the player who had it. A
                # genuine loose ball should.
                previous_owner = None
                held_since = None

            results.append(decision)
        return results

    def spans(self, per_frame: list[FramePossession] | None = None) -> list[PossessionSpan]:
        """Consolidate per-frame decisions into smoothed spans."""
        decisions = per_frame if per_frame is not None else self.per_frame()
        if not decisions:
            return []

        raw: list[PossessionSpan] = []
        current: PossessionSpan | None = None
        confidences: list[float] = []
        ball_known: list[bool] = []

        for decision in decisions:
            key = (decision.state, decision.team_id, decision.track_id)
            if current is not None and (
                current.state,
                current.team_id,
                current.track_id,
            ) == key:
                current.end_frame = decision.frame_idx
                current.end_time_s = decision.timestamp_s
            else:
                if current is not None:
                    current.confidence = float(np.mean(confidences))
                    current.ball_coverage = float(np.mean(ball_known))
                    raw.append(current)
                current = PossessionSpan(
                    start_frame=decision.frame_idx,
                    end_frame=decision.frame_idx,
                    start_time_s=decision.timestamp_s,
                    end_time_s=decision.timestamp_s,
                    state=decision.state,
                    team_id=decision.team_id,
                    track_id=decision.track_id,
                    player_name=self.ctx.display_names.get(decision.track_id or -1, ""),
                )
                confidences = []
                ball_known = []
            confidences.append(decision.confidence)
            ball_known.append(decision.ball_state.is_known)

        if current is not None:
            current.confidence = float(np.mean(confidences))
            current.ball_coverage = float(np.mean(ball_known))
            raw.append(current)

        return self._smooth(raw)

    def _smooth(self, spans: list[PossessionSpan]) -> list[PossessionSpan]:
        """Absorb spans too short to be real.

        A three-frame possession is a detection artefact. Absorbing it into the
        neighbour it most resembles avoids generating a spurious turnover and
        turnover-back pair, which would otherwise appear as two events.
        """
        if not spans:
            return spans

        smoothed: list[PossessionSpan] = []
        for span in spans:
            if (
                span.duration_s < self.cfg.min_span_s
                and smoothed
                and span.state is not PossessionState.UNKNOWN
            ):
                # Never absorb into UNKNOWN: that would invent knowledge.
                if smoothed[-1].state is not PossessionState.UNKNOWN:
                    smoothed[-1].end_frame = span.end_frame
                    smoothed[-1].end_time_s = span.end_time_s
                    continue
            smoothed.append(span)
        return smoothed

    # -- aggregates ------------------------------------------------------------ #

    def summary(self, spans: list[PossessionSpan]) -> dict:
        """Team and player possession shares, with explicit unknown time."""
        total = sum(s.duration_s for s in spans)
        if total <= 0:
            return {"total_s": 0.0, "teams": {}, "players": {}, "states": {}}

        by_state: dict[str, float] = {}
        by_team: dict[str, float] = {}
        by_player: dict[int, float] = {}

        for span in spans:
            by_state[span.state.value] = by_state.get(span.state.value, 0.0) + span.duration_s
            if span.state is PossessionState.CONTROLLED and is_team(span.team_id):
                by_team[span.team_id] = by_team.get(span.team_id, 0.0) + span.duration_s
                if span.track_id is not None:
                    by_player[span.track_id] = (
                        by_player.get(span.track_id, 0.0) + span.duration_s
                    )

        controlled = sum(by_team.values())
        unknown_s = by_state.get(PossessionState.UNKNOWN.value, 0.0)

        return {
            "total_s": round(total, 2),
            "controlled_s": round(controlled, 2),
            "unknown_s": round(unknown_s, 2),
            # The share of match time in which possession could be determined at
            # all. Every possession percentage below is a share of this, not of
            # the match, and reporting one without the other is misleading.
            "determinable_ratio": round(controlled / total, 4) if total else 0.0,
            "unknown_ratio": round(unknown_s / total, 4) if total else 0.0,
            "states": {k: round(v, 2) for k, v in by_state.items()},
            "teams": {
                team: {
                    "seconds": round(seconds, 2),
                    "share_of_controlled": round(seconds / controlled, 4) if controlled else 0.0,
                    "share_of_match": round(seconds / total, 4),
                }
                for team, seconds in sorted(by_team.items())
            },
            "players": {
                int(track_id): round(seconds, 2)
                for track_id, seconds in sorted(by_player.items(), key=lambda kv: -kv[1])
            },
        }

    def team_possession_metric(self, spans: list[PossessionSpan], team_id: str) -> Metric:
        summary = self.summary(spans)
        team = summary["teams"].get(team_id)
        if team is None:
            return Metric.unavailable(unit="%", basis=MetricBasis.EVENT_DERIVED)
        return Metric(
            value=round(100 * team["share_of_controlled"], 2),
            coverage=summary["determinable_ratio"],
            confidence=self.ctx.ball_coverage,
            n_samples=int(team["seconds"] * self.ctx.fps),
            basis=MetricBasis.EVENT_DERIVED,
            unit="%",
        )


def run(context: AnalysisContext, config: PossessionConfig | None = None):
    """Compute possession for a run. Returns ``(spans, per_frame, summary)``."""
    engine = PossessionEngine(context, config)
    per_frame = engine.per_frame()
    spans = engine.spans(per_frame)
    summary = engine.summary(spans)
    log.info(
        "possession: %d spans, %.1f%% determinable, %.1f%% unknown",
        len(spans),
        100 * summary["determinable_ratio"],
        100 * summary["unknown_ratio"],
    )
    return spans, per_frame, summary
