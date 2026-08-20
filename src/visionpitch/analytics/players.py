"""Per-player, goalkeeper and team analytics.

Every number is a :class:`Metric`, so it cannot be reported without the coverage
it rests on. Physical statistics use only ``valid`` rows (Phase 1B constraint 4);
counting statistics derived from events do not need pitch coordinates at all and
say so through their ``basis``.

The four coverages required by constraint 5 -- tracking, pitch, ball, identity --
are attached to every player profile rather than to individual metrics, because
they describe the player's data rather than any one measurement.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from visionpitch.analytics.context import AnalysisContext
from visionpitch.analytics.kinematics import KinematicProfile
from visionpitch.analytics.types import (
    CoverageProfile,
    EventType,
    FootballEvent,
    Metric,
    MetricBasis,
    PossessionSpan,
    PossessionState,
    is_team,
)
from visionpitch.common.logging import get_logger

log = get_logger("analytics.players")

#: Event types that count as a defensive action for zonal aggregation.
DEFENSIVE_EVENTS = (EventType.INTERCEPTION, EventType.RECOVERY, EventType.CLEARANCE)


@dataclass
class PlayerProfile:
    """Everything the dashboard shows for one player."""

    track_id: int
    display_name: str
    team_id: str
    role: str
    jersey_number: int | None
    coverage: CoverageProfile
    metrics: dict[str, Metric] = field(default_factory=dict)
    #: (timestamp, event_type, event_id) so the UI can seek from any statistic
    event_links: list[tuple[float, str, str]] = field(default_factory=list)
    average_position: tuple[float, float] | None = None
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "display_name": self.display_name,
            "team_id": self.team_id,
            "role": self.role,
            "jersey_number": self.jersey_number,
            "coverage": self.coverage.to_dict(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "average_position": (
                {"x": round(self.average_position[0], 2),
                 "y": round(self.average_position[1], 2)}
                if self.average_position else None
            ),
            "first_seen_s": round(self.first_seen_s, 2),
            "last_seen_s": round(self.last_seen_s, 2),
            "event_links": [
                {"timestamp_s": round(t, 3), "event_type": e, "event_id": i}
                for t, e, i in self.event_links
            ],
        }


def _count(events: list[FootballEvent], event_type: EventType, track_id: int) -> int:
    return sum(1 for e in events if e.event_type is event_type and e.track_id == track_id)


def _count_related(events: list[FootballEvent], event_type: EventType, track_id: int) -> int:
    return sum(1 for e in events if e.event_type is event_type and e.related_track_id == track_id)


def build_player_profiles(
    context: AnalysisContext,
    kinematics: dict[int, KinematicProfile],
    events: list[FootballEvent],
    spans: list[PossessionSpan],
) -> dict[int, PlayerProfile]:
    """One profile per tracked person."""
    duration = max(1e-6, context.duration_s)

    possession_time: dict[int, float] = defaultdict(float)
    for span in spans:
        if span.state is PossessionState.CONTROLLED and span.track_id is not None:
            possession_time[span.track_id] += span.duration_s

    events_by_track: dict[int, list[FootballEvent]] = defaultdict(list)
    for event in events:
        if event.track_id is not None:
            events_by_track[event.track_id].append(event)
        if event.related_track_id is not None:
            events_by_track[event.related_track_id].append(event)

    profiles: dict[int, PlayerProfile] = {}

    for track_id, kin in kinematics.items():
        rows = context.players[context.players.track_id == track_id]
        if rows.empty:
            continue

        tracked_frames = len(rows)
        ball_known = sum(
            1 for f in rows.frame_idx.astype(int) if context.ball_state(int(f)).is_known
        )
        coverage = CoverageProfile(
            tracking=tracked_frames / max(1, context.n_frames),
            pitch=kin.coverage,
            ball=ball_known / max(1, tracked_frames),
            identity=context.track_identity_confidence.get(track_id, 0.0),
        )

        track_events = events_by_track.get(track_id, [])
        passes = _count(track_events, EventType.PASS, track_id)
        completed = _count(track_events, EventType.PASS_SUCCESSFUL, track_id)
        failed = _count(track_events, EventType.PASS_FAILED, track_id)

        # Counting metrics do not need pitch coordinates, but they *do* depend
        # on the ball having been locatable, so their coverage is the ball
        # coverage rather than the pitch coverage.
        # Loop variables are bound as defaults rather than captured: a closure
        # that reads them from the enclosing scope silently follows the loop if
        # it is ever called later, which is a bug waiting for a refactor.
        def event_metric(
            value: int, unit: str = "", _cov=coverage, _n=len(track_events)
        ) -> Metric:
            return Metric(
                value=value,
                coverage=_cov.ball,
                confidence=_cov.ball * max(0.2, _cov.identity),
                n_samples=_n,
                basis=MetricBasis.EVENT_DERIVED,
                unit=unit,
            )

        metrics: dict[str, Metric] = {
            # -- physical: valid rows only ---------------------------------- #
            "distance_m": kin.metric(round(kin.distance_m, 1), "m"),
            "mean_speed_m_s": kin.metric(round(kin.mean_speed_m_s, 2), "m/s"),
            "top_speed_m_s": kin.metric(round(kin.top_speed_m_s, 2), "m/s"),
            "sprints": kin.metric(kin.n_sprints, "count"),
            "sprint_distance_m": kin.metric(round(kin.sprint_distance_m, 1), "m"),
            "accelerations": kin.metric(kin.n_accelerations, "count"),
            "decelerations": kin.metric(kin.n_decelerations, "count"),
            # -- possession -------------------------------------------------- #
            "possession_time_s": Metric(
                value=round(possession_time.get(track_id, 0.0), 2),
                coverage=coverage.ball,
                confidence=coverage.ball,
                n_samples=tracked_frames,
                basis=MetricBasis.EVENT_DERIVED,
                unit="s",
            ),
            "touches": event_metric(_count(track_events, EventType.BALL_TOUCH, track_id)),
            # -- passing ------------------------------------------------------ #
            "pass_attempts": event_metric(passes),
            "passes_completed": event_metric(completed),
            "passes_failed": event_metric(failed),
            "progressive_passes": event_metric(
                _count(track_events, EventType.PASS_PROGRESSIVE, track_id)
            ),
            "long_passes": event_metric(_count(track_events, EventType.PASS_LONG, track_id)),
            "back_passes": event_metric(_count(track_events, EventType.PASS_BACK, track_id)),
            "crosses": event_metric(_count(track_events, EventType.CROSS, track_id)),
            "passes_received": event_metric(
                _count_related(track_events, EventType.PASS_SUCCESSFUL, track_id)
            ),
            # -- carrying ------------------------------------------------------ #
            "carries": event_metric(_count(track_events, EventType.CARRY, track_id)),
            "dribble_candidates": event_metric(
                _count(track_events, EventType.DRIBBLE_CANDIDATE, track_id)
            ),
            # -- defending ------------------------------------------------------ #
            "interceptions": event_metric(
                _count(track_events, EventType.INTERCEPTION, track_id)
            ),
            "recoveries": event_metric(_count(track_events, EventType.RECOVERY, track_id)),
            "clearances": event_metric(_count(track_events, EventType.CLEARANCE, track_id)),
            "possession_lost": event_metric(
                _count(track_events, EventType.TURNOVER, track_id)
            ),
            # -- attacking output ------------------------------------------------ #
            "shots": event_metric(_count(track_events, EventType.SHOT, track_id)),
            "shots_on_target": event_metric(
                _count(track_events, EventType.SHOT_ON_TARGET, track_id)
            ),
        }

        metrics["pass_accuracy_pct"] = (
            Metric(
                value=round(100 * completed / passes, 1),
                coverage=coverage.ball,
                confidence=coverage.ball * min(1.0, passes / 5.0),
                n_samples=passes,
                basis=MetricBasis.EVENT_DERIVED,
                unit="%",
            )
            if passes > 0
            else Metric.unavailable("%", MetricBasis.EVENT_DERIVED)
        )

        # Minutes played is the tracked span, not the match length: a track is
        # not a player, and a player split across three tracks would otherwise
        # be credited with three times their time on the pitch.
        metrics["minutes_tracked"] = Metric(
            value=round((rows.timestamp_s.max() - rows.timestamp_s.min()) / 60.0, 2),
            coverage=coverage.tracking,
            confidence=coverage.identity,
            n_samples=tracked_frames,
            basis=MetricBasis.IMAGE_SPACE,
            unit="min",
        )

        for name, zone_distance in kin.distance_by_zone_m.items():
            metrics[f"distance_{name}_m"] = kin.metric(round(zone_distance, 1), "m")

        profiles[track_id] = PlayerProfile(
            track_id=track_id,
            display_name=context.display_names.get(track_id, f"Track {track_id}"),
            team_id=context.track_teams.get(track_id, "unknown"),
            role=context.track_roles.get(track_id, "unknown"),
            jersey_number=None,
            coverage=coverage,
            metrics=metrics,
            event_links=sorted(
                (e.timestamp_s, e.event_type.value, e.event_id) for e in track_events
            ),
            average_position=kin.mean_position,
            first_seen_s=float(rows.timestamp_s.min()),
            last_seen_s=float(rows.timestamp_s.max()),
        )

    _ = duration
    log.info("built %d player profiles", len(profiles))
    return profiles


# --------------------------------------------------------------------------- #
# Goalkeepers
# --------------------------------------------------------------------------- #


@dataclass
class GoalkeeperProfile:
    """Goalkeeper-specific analytics."""

    track_id: int
    display_name: str
    team_id: str
    coverage: CoverageProfile
    metrics: dict[str, Metric] = field(default_factory=dict)
    distribution_map: list[dict] = field(default_factory=list)
    save_map: list[dict] = field(default_factory=list)
    shot_map: list[dict] = field(default_factory=list)
    average_position: tuple[float, float] | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "display_name": self.display_name,
            "team_id": self.team_id,
            "coverage": self.coverage.to_dict(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "distribution_map": self.distribution_map,
            "save_map": self.save_map,
            "shot_map": self.shot_map,
            "average_position": (
                {"x": round(self.average_position[0], 2),
                 "y": round(self.average_position[1], 2)}
                if self.average_position else None
            ),
            "note": self.note,
        }


def build_goalkeeper_profiles(
    context: AnalysisContext,
    players: dict[int, PlayerProfile],
    events: list[FootballEvent],
) -> dict[int, GoalkeeperProfile]:
    """Goalkeeper analytics for tracks Phase 1 identified as goalkeepers.

    Several of these are *candidates* rather than verified outcomes. A save
    cannot be distinguished from a shot that missed without knowing whether the
    ball crossed the line, and Phase 1 explicitly does not determine that. They
    are labelled accordingly rather than presented as facts.
    """
    keepers = {
        track_id: profile
        for track_id, profile in players.items()
        if profile.role == "goalkeeper"
    }
    if not keepers:
        log.info("no goalkeeper tracks in this run; goalkeeper analytics is empty")
        return {}

    profiles: dict[int, GoalkeeperProfile] = {}
    for track_id, player in keepers.items():
        own_events = [e for e in events if e.track_id == track_id]
        faced = [
            e for e in events
            if e.event_type is EventType.SHOT and e.related_team_id == player.team_id
        ]

        distributions = [
            e for e in own_events
            if e.event_type in (EventType.PASS, EventType.CLEARANCE)
        ]
        completed = [
            e for e in own_events if e.event_type is EventType.PASS_SUCCESSFUL
        ]
        long_distributions = [
            e for e in distributions
            if (e.distance_m or 0) >= 25.0
        ]

        def metric(value, unit="", n=0, _cov=player.coverage) -> Metric:
            return Metric(
                value=value,
                coverage=_cov.ball,
                confidence=_cov.ball * _cov.identity,
                n_samples=n,
                basis=MetricBasis.EVENT_DERIVED,
                unit=unit,
            )

        metrics = {
            "shots_faced": metric(len(faced), "count", len(faced)),
            "save_candidates": metric(
                sum(1 for e in faced if e.event_type is not EventType.GOAL_CANDIDATE),
                "count", len(faced),
            ),
            "goal_candidates_conceded": metric(
                sum(1 for e in events
                    if e.event_type is EventType.GOAL_CANDIDATE
                    and e.related_team_id == player.team_id),
                "count",
            ),
            "distributions": metric(len(distributions), "count", len(distributions)),
            "distributions_completed": metric(len(completed), "count", len(completed)),
            "long_distributions": metric(
                len(long_distributions), "count", len(long_distributions)
            ),
            "recoveries": player.metrics.get("recoveries", Metric.unavailable("count")),
            "clearances": player.metrics.get("clearances", Metric.unavailable("count")),
            "distance_m": player.metrics.get("distance_m", Metric.unavailable("m")),
        }
        metrics["distribution_accuracy_pct"] = (
            Metric(
                value=round(100 * len(completed) / len(distributions), 1),
                coverage=player.coverage.ball,
                confidence=player.coverage.ball * min(1.0, len(distributions) / 5.0),
                n_samples=len(distributions),
                basis=MetricBasis.EVENT_DERIVED,
                unit="%",
            )
            if distributions
            else Metric.unavailable("%", MetricBasis.EVENT_DERIVED)
        )

        profiles[track_id] = GoalkeeperProfile(
            track_id=track_id,
            display_name=player.display_name,
            team_id=player.team_id,
            coverage=player.coverage,
            metrics=metrics,
            distribution_map=[
                {
                    "event_id": e.event_id, "timestamp_s": round(e.timestamp_s, 3),
                    "start_x": e.start_x, "start_y": e.start_y,
                    "end_x": e.end_x, "end_y": e.end_y,
                    "completed": e.event_type is EventType.PASS_SUCCESSFUL,
                    "distance_m": e.distance_m,
                }
                for e in distributions if e.start_x is not None
            ],
            shot_map=[
                {
                    "event_id": e.event_id, "timestamp_s": round(e.timestamp_s, 3),
                    "x": e.start_x, "y": e.start_y,
                    "on_target": e.event_type is EventType.SHOT_ON_TARGET,
                }
                for e in faced if e.start_x is not None
            ],
            save_map=[],
            average_position=player.average_position,
            note=(
                "EXPERIMENTAL — this engine has never been measured against "
                "goalkeeper ground truth. Phase 2C established that SoccerNet-GSR "
                "does annotate goalkeepers (3,206 instances across 49 sequences), "
                "so this is now measurable; it has simply not been measured. Treat "
                "every number below as unvalidated. "
                "Saves are not distinguished from shots that missed: both require "
                "goal-line evidence the pipeline does not have, so 'save candidates' "
                "is an upper bound, not a save count. Goals conceded are candidates "
                "for review, never a scoreline. Sweeper actions, cross claims and "
                "punches are not implemented."
            ),
        )

    log.info("built %d goalkeeper profile(s)", len(profiles))
    return profiles


# --------------------------------------------------------------------------- #
# Teams
# --------------------------------------------------------------------------- #


@dataclass
class TeamProfile:
    team_id: str
    n_players: int
    coverage: CoverageProfile
    metrics: dict[str, Metric] = field(default_factory=dict)
    average_positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    attack_direction: str = "unknown"
    attack_direction_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "team_id": self.team_id,
            "n_players": self.n_players,
            "coverage": self.coverage.to_dict(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "average_positions": {
                str(tid): {"x": round(p[0], 2), "y": round(p[1], 2)}
                for tid, p in self.average_positions.items()
            },
            "attack_direction": self.attack_direction,
            "attack_direction_confidence": round(self.attack_direction_confidence, 3),
        }


def build_team_profiles(
    context: AnalysisContext,
    players: dict[int, PlayerProfile],
    events: list[FootballEvent],
    possession_summary: dict,
) -> dict[str, TeamProfile]:
    """Aggregate team analytics."""
    setup = context.manifest.get("stages", {}).get("match_setup", {})
    directions = setup.get("attack_directions", {})
    direction_conf = setup.get("attack_direction_confidence", {})

    profiles: dict[str, TeamProfile] = {}
    # Teams are read from the data rather than assumed to be named "A" and "B".
    # Hard-coding the pair meant this returned no profiles at all whenever the
    # classifier's vocabulary differed, which is silent and total data loss.
    for team_id in sorted({p.team_id for p in players.values() if is_team(p.team_id)}):
        members = [p for p in players.values() if p.team_id == team_id]
        if not members:
            continue

        team_events = [e for e in events if e.team_id == team_id]
        passes = sum(1 for e in team_events if e.event_type is EventType.PASS)
        completed = sum(1 for e in team_events if e.event_type is EventType.PASS_SUCCESSFUL)

        coverage = CoverageProfile(
            tracking=float(np.mean([p.coverage.tracking for p in members])),
            pitch=float(np.mean([p.coverage.pitch for p in members])),
            ball=float(np.mean([p.coverage.ball for p in members])),
            identity=float(np.mean([p.coverage.identity for p in members])),
        )

        def summed(name: str, _members=members, _cov=coverage) -> Metric:
            values = [
                p.metrics[name].value for p in _members
                if name in p.metrics and p.metrics[name].is_reportable
            ]
            if not values:
                return Metric.unavailable(basis=MetricBasis.INCLUDES_EXTRAPOLATED)
            return Metric(
                value=round(float(sum(values)), 1),
                coverage=_cov.pitch,
                confidence=_cov.pitch,
                n_samples=len(values),
                # Team aggregates pool players of differing coverage, so the
                # total is a lower bound on the team's real output. Constraint 6
                # permits this at team level provided it is labelled.
                basis=MetricBasis.INCLUDES_EXTRAPOLATED,
                unit=_members[0].metrics[name].unit if name in _members[0].metrics else "",
            )

        possession = possession_summary.get("teams", {}).get(team_id, {})
        metrics = {
            "possession_pct": Metric(
                value=round(100 * possession.get("share_of_controlled", 0.0), 1),
                coverage=possession_summary.get("determinable_ratio", 0.0),
                confidence=context.ball_coverage,
                n_samples=int(possession.get("seconds", 0.0) * context.fps),
                basis=MetricBasis.EVENT_DERIVED,
                unit="%",
            ),
            "possession_time_s": Metric(
                value=round(possession.get("seconds", 0.0), 1),
                coverage=possession_summary.get("determinable_ratio", 0.0),
                confidence=context.ball_coverage,
                n_samples=len(members),
                basis=MetricBasis.EVENT_DERIVED,
                unit="s",
            ),
            "pass_attempts": Metric(
                value=passes, coverage=coverage.ball, confidence=coverage.ball,
                n_samples=passes, basis=MetricBasis.EVENT_DERIVED, unit="count",
            ),
            "passes_completed": Metric(
                value=completed, coverage=coverage.ball, confidence=coverage.ball,
                n_samples=completed, basis=MetricBasis.EVENT_DERIVED, unit="count",
            ),
            "pass_accuracy_pct": (
                Metric(
                    value=round(100 * completed / passes, 1),
                    coverage=coverage.ball, confidence=coverage.ball,
                    n_samples=passes, basis=MetricBasis.EVENT_DERIVED, unit="%",
                )
                if passes else Metric.unavailable("%", MetricBasis.EVENT_DERIVED)
            ),
            "progressive_passes": Metric(
                value=sum(1 for e in team_events
                          if e.event_type is EventType.PASS_PROGRESSIVE),
                coverage=coverage.ball, confidence=coverage.ball, n_samples=passes,
                basis=MetricBasis.EVENT_DERIVED, unit="count",
            ),
            "interceptions": Metric(
                value=sum(1 for e in team_events if e.event_type is EventType.INTERCEPTION),
                coverage=coverage.ball, confidence=coverage.ball, n_samples=len(team_events),
                basis=MetricBasis.EVENT_DERIVED, unit="count",
            ),
            "recoveries": Metric(
                value=sum(1 for e in team_events if e.event_type is EventType.RECOVERY),
                coverage=coverage.ball, confidence=coverage.ball, n_samples=len(team_events),
                basis=MetricBasis.EVENT_DERIVED, unit="count",
            ),
            "turnovers": Metric(
                value=sum(1 for e in team_events if e.event_type is EventType.TURNOVER),
                coverage=coverage.ball, confidence=coverage.ball, n_samples=len(team_events),
                basis=MetricBasis.EVENT_DERIVED, unit="count",
            ),
            "distance_m": summed("distance_m"),
            "sprints": summed("sprints"),
            "final_third_entries": _zone_entries(context, events, team_id, third=True),
            "penalty_area_entries": _zone_entries(context, events, team_id, third=False),
        }

        profiles[team_id] = TeamProfile(
            team_id=team_id,
            n_players=len(members),
            coverage=coverage,
            metrics=metrics,
            average_positions={
                p.track_id: p.average_position
                for p in members if p.average_position is not None
            },
            attack_direction=directions.get(team_id, "unknown"),
            attack_direction_confidence=float(direction_conf.get(team_id, 0.0)),
        )

    log.info("built %d team profile(s)", len(profiles))
    return profiles


def _zone_entries(
    context: AnalysisContext, events: list[FootballEvent], team_id: str, third: bool
) -> Metric:
    """Count ball entries into the final third or the penalty area.

    Requires a known attack direction; without one the question has no answer
    and the metric is unavailable rather than zero.
    """
    setup = context.manifest.get("stages", {}).get("match_setup", {})
    direction = setup.get("attack_directions", {}).get(team_id)
    confidence = setup.get("attack_direction_confidence", {}).get(team_id, 0.0)
    if direction not in ("left_to_right", "right_to_left") or confidence < 0.3:
        return Metric.unavailable("count", MetricBasis.EVENT_DERIVED)

    pitch = context.pitch
    side = "right" if direction == "left_to_right" else "left"
    entries = 0
    inside = False
    for event in sorted(
        (e for e in events if e.team_id == team_id and e.end_x is not None),
        key=lambda e: e.timestamp_s,
    ):
        if third:
            now_inside = (
                event.end_x > 2 * pitch.length / 3 if side == "right"
                else event.end_x < pitch.length / 3
            )
        else:
            now_inside = pitch.in_penalty_area(event.end_x, event.end_y or 0.0, side)
        if now_inside and not inside:
            entries += 1
        inside = now_inside

    return Metric(
        value=entries, coverage=context.ball_coverage, confidence=float(confidence),
        n_samples=len(events), basis=MetricBasis.EVENT_DERIVED, unit="count",
    )


def summarise_counts(events: list[FootballEvent]) -> dict[str, int]:
    return dict(Counter(e.event_type.value for e in events))
