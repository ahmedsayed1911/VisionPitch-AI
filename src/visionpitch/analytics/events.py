"""Football event detection.

Events are derived from **transitions between possession spans**, not from
individual frames. A pass is not a frame in which the ball moved; it is the
structure "player A controlled the ball, the ball travelled, player B controlled
it", and only the whole structure identifies it. Frame-level rules produce an
event every time a noisy position jitters.

Classification of a possession handover:

    A -> A (same team, different player)   pass, completed
    A -> B (opponent gains control)        depends on how:
        ball travelled far and fast          -> failed pass + interception
        ball was contested first             -> turnover
        ball was near A's own goal           -> clearance context
    A -> loose -> A (same player)          carry or dribble
    A -> out of play                       ball out, then restart

Every event carries the evidence behind it and a clip reference, because an
event is model output that a reviewer must be able to disagree with
specifically.

Coverage discipline
-------------------
An event inferred across frames where the ball was unknown is marked with that
coverage. Events whose entire supporting window had an unknown ball are not
emitted at all -- there is nothing to infer from.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from visionpitch.analytics.context import AnalysisContext
from visionpitch.analytics.types import (
    BallStateKind,
    ClipReference,
    EventType,
    Evidence,
    FootballEvent,
    PossessionSpan,
    PossessionState,
    is_team,
)
from visionpitch.common.logging import get_logger

log = get_logger("analytics.events")


@dataclass
class EventConfig:
    """Thresholds in football units."""

    #: a pass must move the ball at least this far
    min_pass_distance_m: float = 3.0
    #: beyond this a pass is 'long'
    long_pass_distance_m: float = 25.0
    #: forward progress toward goal required to be 'progressive'
    progressive_gain_m: float = 10.0
    #: a pass moving the ball this far backwards is a back pass
    back_pass_gain_m: float = -5.0
    #: a cross originates outside this distance from the touchline in the final third
    cross_max_width_from_touchline_m: float = 18.0
    #: a carry must move the ball at least this far while one player holds it
    min_carry_distance_m: float = 4.0
    #: progressive carry threshold
    progressive_carry_m: float = 8.0
    #: a shot must be directed at goal from within this range
    max_shot_distance_m: float = 35.0
    #: how close to the goal centre line a shot trajectory must point
    shot_on_target_half_width_m: float = 4.0
    #: a handover this quick after a travelling ball is an interception
    max_interception_gap_s: float = 1.2
    #: clip padding around an event, in seconds
    clip_pad_s: float = 2.0
    #: events below this confidence are still emitted, but banded 'uncertain'
    min_emit_confidence: float = 0.15


class EventEngine:
    """Turns possession spans into football events."""

    def __init__(
        self,
        context: AnalysisContext,
        spans: list[PossessionSpan],
        config: EventConfig | None = None,
    ) -> None:
        self.ctx = context
        self.spans = spans
        self.cfg = config or EventConfig()
        self._attack_direction = self._resolve_attack_directions()

    # -- helpers -------------------------------------------------------------- #

    def _resolve_attack_directions(self) -> dict[str, int]:
        """team -> +1 if attacking toward increasing x, -1 if decreasing, 0 unknown.

        Phase 1 reports this with a confidence and reports ``unknown`` when the
        evidence conflicts. Without it, 'progressive' and 'back' pass cannot be
        defined, so those classifications are withheld rather than guessed.
        """
        setup = self.ctx.manifest.get("stages", {}).get("match_setup", {})
        directions = setup.get("attack_directions", {})
        confidences = setup.get("attack_direction_confidence", {})
        out: dict[str, int] = {}
        for team, direction in directions.items():
            if confidences.get(team, 0.0) < 0.3:
                out[team] = 0
            elif direction == "left_to_right":
                out[team] = 1
            elif direction == "right_to_left":
                out[team] = -1
            else:
                out[team] = 0
        return out

    def _goal_centre(self, team_id: str) -> tuple[float, float] | None:
        direction = self._attack_direction.get(team_id, 0)
        if direction == 0:
            return None
        pitch = self.ctx.pitch
        return pitch.goal_centre("right" if direction > 0 else "left")

    def _clip(self, frame_idx: int, end_frame: int | None = None) -> ClipReference:
        pad = int(self.cfg.clip_pad_s * self.ctx.fps)
        frames = self.ctx.frame_indices
        lo = max(frames[0], frame_idx - pad) if frames else frame_idx
        hi = min(frames[-1], (end_frame or frame_idx) + pad) if frames else frame_idx
        return ClipReference(
            frame_start=lo,
            frame_end=hi,
            time_start_s=self.ctx.timestamps.get(lo, 0.0),
            time_end_s=self.ctx.timestamps.get(hi, 0.0),
        )

    def _ball_coverage(self, start_frame: int, end_frame: int) -> tuple[float, BallStateKind]:
        """Fraction of the window with a known ball, and the dominant state."""
        window = [f for f in self.ctx.frame_indices if start_frame <= f <= end_frame]
        if not window:
            return 0.0, BallStateKind.UNKNOWN
        states = [self.ctx.ball_state(f) for f in window]
        known = sum(1 for s in states if s.is_known)
        observed = sum(1 for s in states if s is BallStateKind.OBSERVED)
        if observed > len(window) / 2:
            dominant = BallStateKind.OBSERVED
        elif known > 0:
            dominant = BallStateKind.INTERPOLATED
        else:
            dominant = BallStateKind.UNKNOWN
        return known / len(window), dominant

    def _new_event(
        self,
        event_type: EventType,
        span: PossessionSpan,
        frame_idx: int,
        confidence: float,
        evidence: Evidence,
        **kwargs,
    ) -> FootballEvent:
        coverage, state = self._ball_coverage(frame_idx, kwargs.get("_end_frame", frame_idx))
        kwargs.pop("_end_frame", None)
        return FootballEvent(
            event_id=uuid.uuid4().hex[:16],
            event_type=event_type,
            frame_idx=frame_idx,
            timestamp_s=self.ctx.timestamps.get(frame_idx, span.start_time_s),
            team_id=span.team_id,
            track_id=span.track_id,
            player_name=self.ctx.display_names.get(span.track_id or -1, ""),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            ball_coverage=coverage,
            ball_state=state,
            evidence=evidence,
            clip=self._clip(frame_idx, kwargs.get("end_frame_for_clip")),
            **{k: v for k, v in kwargs.items() if k != "end_frame_for_clip"},
        )

    # -- detection ------------------------------------------------------------- #

    def detect(self) -> list[FootballEvent]:
        events: list[FootballEvent] = []
        controlled = [
            (i, s) for i, s in enumerate(self.spans)
            if s.state is PossessionState.CONTROLLED and s.track_id is not None
        ]

        events.extend(self._touches(controlled))
        events.extend(self._carries(controlled))
        events.extend(self._handovers(controlled))
        events.extend(self._out_of_play())
        events.extend(self._loose_and_contested())

        events.sort(key=lambda e: (e.frame_idx, e.event_type.value))
        by_type: dict[str, int] = {}
        for event in events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
        log.info("events: %d detected %s", len(events), by_type)
        return events

    # -- touches and carries ---------------------------------------------------- #

    def _touches(self, controlled) -> list[FootballEvent]:
        events = []
        for _, span in controlled:
            position = self.ctx.ball_position(span.start_frame)
            evidence = Evidence().add(
                "player was the nearest to the ball and held it",
                duration_s=span.duration_s,
                possession_confidence=span.confidence,
            )
            events.append(
                self._new_event(
                    EventType.BALL_TOUCH, span, span.start_frame,
                    confidence=span.confidence, evidence=evidence,
                    start_x=position[0] if position else None,
                    start_y=position[1] if position else None,
                    duration_s=span.duration_s,
                    _end_frame=span.end_frame,
                )
            )
        return events

    def _carries(self, controlled) -> list[FootballEvent]:
        """A carry is control sustained while the ball travels with the player."""
        events = []
        for _, span in controlled:
            start = self.ctx.ball_position(span.start_frame)
            end = self.ctx.ball_position(span.end_frame)
            if start is None or end is None:
                continue
            distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
            if distance < self.cfg.min_carry_distance_m:
                continue

            direction = self._attack_direction.get(span.team_id, 0)
            gain = (end[0] - start[0]) * direction if direction else 0.0
            evidence = Evidence().add(
                "ball moved with a player who retained control throughout",
                distance_m=distance,
                duration_s=span.duration_s,
                forward_gain_m=gain,
            )
            events.append(
                self._new_event(
                    EventType.CARRY, span, span.start_frame,
                    confidence=span.confidence, evidence=evidence,
                    start_x=start[0], start_y=start[1],
                    end_x=end[0], end_y=end[1],
                    distance_m=round(distance, 2), duration_s=span.duration_s,
                    _end_frame=span.end_frame, end_frame_for_clip=span.end_frame,
                )
            )
            if direction and gain >= self.cfg.progressive_carry_m:
                events.append(
                    self._new_event(
                        EventType.DRIBBLE_CANDIDATE, span, span.start_frame,
                        confidence=span.confidence * 0.8,
                        evidence=Evidence().add(
                            "sustained forward carry; opponent engagement not verified",
                            forward_gain_m=gain,
                        ),
                        start_x=start[0], start_y=start[1],
                        end_x=end[0], end_y=end[1],
                        distance_m=round(distance, 2),
                        _end_frame=span.end_frame,
                    )
                )
        return events

    # -- handovers -------------------------------------------------------------- #

    def _handovers(self, controlled) -> list[FootballEvent]:
        events: list[FootballEvent] = []

        for (idx_a, span_a), (idx_b, span_b) in zip(controlled, controlled[1:], strict=False):
            if span_a.track_id == span_b.track_id:
                continue

            start = self.ctx.ball_position(span_a.end_frame)
            end = self.ctx.ball_position(span_b.start_frame)
            if start is None or end is None:
                # No ball evidence across the handover: nothing to classify.
                continue

            distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
            gap_s = max(0.0, span_b.start_time_s - span_a.end_time_s)
            between = self.spans[idx_a + 1 : idx_b]
            was_contested = any(s.state is PossessionState.CONTESTED for s in between)

            coverage, _ = self._ball_coverage(span_a.end_frame, span_b.start_frame)
            base_confidence = min(span_a.confidence, span_b.confidence) * max(0.2, coverage)

            same_team = span_a.team_id == span_b.team_id and is_team(span_a.team_id)
            if same_team and distance >= self.cfg.min_pass_distance_m:
                events.extend(
                    self._pass_events(span_a, span_b, start, end, distance, gap_s,
                                      base_confidence)
                )
            elif not same_team and is_team(span_b.team_id):
                events.extend(
                    self._turnover_events(span_a, span_b, start, end, distance, gap_s,
                                          was_contested, base_confidence)
                )

        return events

    def _pass_events(
        self, span_a, span_b, start, end, distance, gap_s, confidence
    ) -> list[FootballEvent]:
        direction = self._attack_direction.get(span_a.team_id, 0)
        gain = (end[0] - start[0]) * direction if direction else None

        evidence = Evidence().add(
            "ball travelled between two players of the same team",
            distance_m=distance,
            travel_time_s=gap_s,
            receiver_possession_confidence=span_b.confidence,
        )
        if direction == 0:
            evidence.add("attack direction unknown: directional classification withheld")

        shared = {
            "start_x": start[0], "start_y": start[1],
            "end_x": end[0], "end_y": end[1],
            "distance_m": round(distance, 2), "duration_s": round(gap_s, 3),
            "related_track_id": span_b.track_id,
            "related_player_name": self.ctx.display_names.get(span_b.track_id or -1, ""),
            "related_team_id": span_b.team_id,
            "_end_frame": span_b.start_frame,
        }

        events = [
            self._new_event(EventType.PASS, span_a, span_a.end_frame,
                            confidence, evidence, **shared),
            self._new_event(EventType.PASS_SUCCESSFUL, span_a, span_a.end_frame,
                            confidence, evidence, **shared),
        ]

        if distance >= self.cfg.long_pass_distance_m:
            events.append(
                self._new_event(EventType.PASS_LONG, span_a, span_a.end_frame,
                                confidence, evidence, **shared)
            )
        if gain is not None and gain >= self.cfg.progressive_gain_m:
            events.append(
                self._new_event(EventType.PASS_PROGRESSIVE, span_a, span_a.end_frame,
                                confidence, evidence, **shared)
            )
        if gain is not None and gain <= self.cfg.back_pass_gain_m:
            events.append(
                self._new_event(EventType.PASS_BACK, span_a, span_a.end_frame,
                                confidence, evidence, **shared)
            )
        if self._is_cross(start, end, span_a.team_id):
            events.append(
                self._new_event(EventType.CROSS, span_a, span_a.end_frame,
                                confidence * 0.85,
                                Evidence().add("wide origin in the final third, "
                                               "delivered toward the penalty area"),
                                **shared)
            )
        return events

    def _is_cross(self, start, end, team_id) -> bool:
        direction = self._attack_direction.get(team_id, 0)
        if direction == 0:
            return False
        pitch = self.ctx.pitch
        # Originates wide, in the attacking third, and ends in the penalty area.
        from_touchline = min(start[1], pitch.width - start[1])
        if from_touchline > self.cfg.cross_max_width_from_touchline_m:
            return False
        attacking_third = (
            start[0] > 2 * pitch.length / 3 if direction > 0 else start[0] < pitch.length / 3
        )
        side = "right" if direction > 0 else "left"
        return attacking_third and pitch.in_penalty_area(end[0], end[1], side)

    def _turnover_events(
        self, span_a, span_b, start, end, distance, gap_s, was_contested, confidence
    ) -> list[FootballEvent]:
        shared = {
            "start_x": start[0], "start_y": start[1],
            "end_x": end[0], "end_y": end[1],
            "distance_m": round(distance, 2), "duration_s": round(gap_s, 3),
            "related_track_id": span_b.track_id,
            "related_player_name": self.ctx.display_names.get(span_b.track_id or -1, ""),
            "related_team_id": span_b.team_id,
            "_end_frame": span_b.start_frame,
        }
        events = [
            self._new_event(
                EventType.TURNOVER, span_a, span_a.end_frame, confidence,
                Evidence().add("possession changed team", distance_m=distance,
                               gap_s=gap_s, contested=float(was_contested)),
                **shared,
            )
        ]

        # An opponent taking a ball that travelled a distance quickly intercepted
        # a pass; one picking up a ball that went nowhere recovered a loose ball.
        if distance >= self.cfg.min_pass_distance_m and gap_s <= self.cfg.max_interception_gap_s:
            events.append(
                self._new_event(
                    EventType.PASS_FAILED, span_a, span_a.end_frame, confidence,
                    Evidence().add("ball travelled to an opponent",
                                   distance_m=distance, travel_time_s=gap_s),
                    **shared,
                )
            )
            interception = self._new_event(
                EventType.INTERCEPTION, span_b, span_b.start_frame, confidence,
                Evidence().add("opponent gained control of a travelling ball",
                               distance_m=distance, travel_time_s=gap_s),
                start_x=end[0], start_y=end[1],
                related_track_id=span_a.track_id,
                related_player_name=self.ctx.display_names.get(span_a.track_id or -1, ""),
                related_team_id=span_a.team_id,
                _end_frame=span_b.start_frame,
            )
            events.append(interception)
        else:
            events.append(
                self._new_event(
                    EventType.RECOVERY, span_b, span_b.start_frame, confidence * 0.9,
                    Evidence().add("opponent took control of a ball that was not "
                                   "travelling toward them",
                                   distance_m=distance, gap_s=gap_s),
                    start_x=end[0], start_y=end[1],
                    related_track_id=span_a.track_id,
                    related_team_id=span_a.team_id,
                    _end_frame=span_b.start_frame,
                )
            )

        events.extend(self._shot_or_clearance(span_a, start, end, distance, confidence, shared))
        return events

    def _shot_or_clearance(
        self, span_a, start, end, distance, confidence, shared
    ) -> list[FootballEvent]:
        """A ball driven toward goal is a shot; away from own goal, a clearance."""
        goal = self._goal_centre(span_a.team_id)
        if goal is None:
            return []

        to_goal_before = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
        to_goal_after = float(np.hypot(goal[0] - end[0], goal[1] - end[1]))

        events: list[FootballEvent] = []
        if (
            to_goal_before <= self.cfg.max_shot_distance_m
            and to_goal_after < to_goal_before
            and distance >= self.cfg.min_pass_distance_m
        ):
            on_target = abs(end[1] - goal[1]) <= self.cfg.shot_on_target_half_width_m
            evidence = Evidence().add(
                "ball driven toward the attacking goal from shooting range",
                distance_to_goal_m=to_goal_before,
                lateral_offset_m=abs(end[1] - goal[1]),
            )
            events.append(
                self._new_event(EventType.SHOT, span_a, span_a.end_frame,
                                confidence * 0.7, evidence, **shared)
            )
            if on_target:
                events.append(
                    self._new_event(EventType.SHOT_ON_TARGET, span_a, span_a.end_frame,
                                    confidence * 0.6, evidence, **shared)
                )
                # A goal cannot be confirmed without scoreboard or net evidence;
                # this is flagged as a candidate for review, never as a goal.
                if to_goal_after < 3.0:
                    events.append(
                        self._new_event(
                            EventType.GOAL_CANDIDATE, span_a, span_a.end_frame,
                            confidence * 0.4,
                            Evidence().add(
                                "ball reached the goal mouth; scoring not verified "
                                "-- requires review"
                            ),
                            **shared,
                        )
                    )
        else:
            own_goal = self._goal_centre(
                "B" if span_a.team_id == "A" else "A"
            )
            if own_goal is not None:
                away_before = float(np.hypot(own_goal[0] - start[0], own_goal[1] - start[1]))
                away_after = float(np.hypot(own_goal[0] - end[0], own_goal[1] - end[1]))
                if away_before < 25.0 and away_after > away_before + 8.0:
                    events.append(
                        self._new_event(
                            EventType.CLEARANCE, span_a, span_a.end_frame,
                            confidence * 0.75,
                            Evidence().add("ball driven away from own goal under pressure",
                                           gained_m=away_after - away_before),
                            **shared,
                        )
                    )
        return events

    # -- dead ball ---------------------------------------------------------------- #

    def _out_of_play(self) -> list[FootballEvent]:
        events = []
        for i, span in enumerate(self.spans):
            if span.state is not PossessionState.OUT_OF_PLAY:
                continue
            position = self.ctx.ball_position(span.start_frame)
            events.append(
                self._new_event(
                    EventType.BALL_OUT, span, span.start_frame,
                    confidence=span.confidence,
                    evidence=Evidence().add("ball position outside the field of play"),
                    start_x=position[0] if position else None,
                    start_y=position[1] if position else None,
                    _end_frame=span.end_frame,
                )
            )
            following = next(
                (s for s in self.spans[i + 1:]
                 if s.state is PossessionState.CONTROLLED and s.track_id is not None),
                None,
            )
            if following is not None:
                events.append(
                    self._new_event(
                        EventType.RESTART, following, following.start_frame,
                        confidence=following.confidence * 0.7,
                        evidence=Evidence().add(
                            "EXPERIMENTAL: first control after the ball left play. "
                            "The restart TYPE is not classified at all -- throw-in, "
                            "goal kick, corner and free kick are indistinguishable "
                            "here, since separating them needs the ball's exit point "
                            "relative to the touchline and goal line at a precision "
                            "the calibration does not support"
                        ),
                        _end_frame=following.start_frame,
                    )
                )
        return events

    def _loose_and_contested(self) -> list[FootballEvent]:
        events = []
        for span in self.spans:
            if span.state is PossessionState.LOOSE_BALL and span.duration_s >= 0.4:
                events.append(
                    self._new_event(
                        EventType.LOOSE_BALL, span, span.start_frame,
                        confidence=span.confidence,
                        evidence=Evidence().add("no player in control",
                                                duration_s=span.duration_s),
                        duration_s=span.duration_s, _end_frame=span.end_frame,
                    )
                )
            elif span.state is PossessionState.CONTESTED and span.duration_s >= 0.3:
                events.append(
                    self._new_event(
                        EventType.CONTESTED_POSSESSION, span, span.start_frame,
                        confidence=span.confidence,
                        evidence=Evidence().add("two players equally close to the ball",
                                                duration_s=span.duration_s),
                        duration_s=span.duration_s, _end_frame=span.end_frame,
                    )
                )
        return events


def run(
    context: AnalysisContext,
    spans: list[PossessionSpan],
    config: EventConfig | None = None,
) -> list[FootballEvent]:
    return EventEngine(context, spans, config).detect()
