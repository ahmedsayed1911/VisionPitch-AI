"""Heatmaps, passing networks and the synchronized timeline.

Heatmaps are computed on a fixed pitch grid so that any two are directly
comparable -- between halves, between phases, and between players. A heatmap
normalised to its own maximum looks identical whether it was built from 4000
samples or 12, which is why every grid here carries its sample count and
coverage alongside the density.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from visionpitch.analytics.context import AnalysisContext
from visionpitch.analytics.kinematics import KinematicProfile
from visionpitch.analytics.types import (
    EventType,
    FootballEvent,
    PossessionSpan,
    PossessionState,
    is_team,
)
from visionpitch.common.logging import get_logger

log = get_logger("analytics.spatial")

HeatmapKind = Literal[
    "position", "touches", "possession", "carries", "pass_origin",
    "pass_destination", "defensive_actions", "sprints", "influence",
]

PhaseFilter = Literal["all", "in_possession", "out_of_possession", "attacking", "defending"]

#: Grid resolution. 12x8 cells of ~8.75 x 8.5 m each: fine enough to show
#: positional tendency, coarse enough that a player with a few dozen samples
#: still produces a readable surface rather than scattered dots.
GRID_X = 12
GRID_Y = 8


@dataclass
class Heatmap:
    kind: str
    track_id: int | None
    team_id: str | None
    grid: list[list[float]]
    n_samples: int
    coverage: float
    time_range_s: tuple[float, float]
    phase: str = "all"
    half: int | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "track_id": self.track_id,
            "team_id": self.team_id,
            "grid": self.grid,
            "grid_x": GRID_X,
            "grid_y": GRID_Y,
            "n_samples": self.n_samples,
            "coverage": round(self.coverage, 4),
            "time_range_s": [round(self.time_range_s[0], 2), round(self.time_range_s[1], 2)],
            "phase": self.phase,
            "half": self.half,
            # A surface built from a handful of samples is a picture of noise.
            "reportable": self.n_samples >= 10,
        }


def _blank_grid() -> np.ndarray:
    return np.zeros((GRID_Y, GRID_X), dtype=np.float64)


def _accumulate(grid: np.ndarray, xs, ys, pitch, weights=None) -> int:
    """Bin pitch positions into the grid. Returns the number binned."""
    n = 0
    weights = weights if weights is not None else np.ones(len(xs))
    for x, y, w in zip(xs, ys, weights, strict=True):
        if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
            continue
        cx = int(np.clip(x / pitch.length * GRID_X, 0, GRID_X - 1))
        cy = int(np.clip(y / pitch.width * GRID_Y, 0, GRID_Y - 1))
        grid[cy, cx] += w
        n += 1
    return n


class HeatmapEngine:
    """Builds every supported heatmap with filtering."""

    def __init__(
        self,
        context: AnalysisContext,
        events: list[FootballEvent],
        spans: list[PossessionSpan],
        kinematics: dict[int, KinematicProfile],
    ) -> None:
        self.ctx = context
        self.events = events
        self.spans = spans
        self.kinematics = kinematics
        self._possession_frames = self._index_possession_frames()

    def _index_possession_frames(self) -> dict[str, set[int]]:
        """team -> frames in which that team was in controlled possession."""
        out: dict[str, set[int]] = defaultdict(set)
        for span in self.spans:
            if span.state is PossessionState.CONTROLLED and is_team(span.team_id):
                out[span.team_id].update(range(span.start_frame, span.end_frame + 1))
        return out

    def _filter_rows(
        self,
        rows: pd.DataFrame,
        time_range: tuple[float, float] | None,
        half: int | None,
        phase: PhaseFilter,
        team_id: str | None,
    ) -> pd.DataFrame:
        out = rows
        if time_range is not None:
            out = out[(out.timestamp_s >= time_range[0]) & (out.timestamp_s <= time_range[1])]
        if half is not None:
            out = out[out.timestamp_s.map(self.ctx.half_of) == half]
        if phase != "all" and is_team(team_id):
            owned = self._possession_frames.get(team_id, set())
            in_possession = out.frame_idx.isin(owned)
            if phase in ("in_possession", "attacking"):
                out = out[in_possession]
            elif phase in ("out_of_possession", "defending"):
                out = out[~in_possession]
        return out

    def build(
        self,
        kind: HeatmapKind,
        track_id: int | None = None,
        team_id: str | None = None,
        time_range: tuple[float, float] | None = None,
        half: int | None = None,
        phase: PhaseFilter = "all",
    ) -> Heatmap:
        pitch = self.ctx.pitch
        grid = _blank_grid()
        n = 0

        resolved_team = team_id or (
            self.ctx.track_teams.get(track_id, None) if track_id is not None else None
        )

        if kind in ("position", "sprints", "influence"):
            # Positional surfaces use only rows Phase 1B deemed physically
            # trustworthy: an extrapolated far-side position would smear the
            # surface toward the horizon.
            rows = self.ctx.valid_players
            if track_id is not None:
                rows = rows[rows.track_id == track_id]
            elif resolved_team:
                rows = rows[rows.team_id == resolved_team]
            rows = self._filter_rows(rows, time_range, half, phase, resolved_team)

            if kind == "sprints":
                speeds = self.kinematics.get(track_id) if track_id is not None else None
                if speeds is None:
                    rows = rows.iloc[0:0]
                else:
                    fast = {f for f, s in speeds.speed_by_frame.items() if s >= 5.5}
                    rows = rows[rows.frame_idx.isin(fast)]

            weights = None
            if kind == "influence" and track_id is not None:
                profile = self.kinematics.get(track_id)
                if profile is not None:
                    # Weight by speed: a player standing still influences less
                    # space than one arriving at pace.
                    weights = np.array([
                        1.0 + profile.speed_by_frame.get(int(f), 0.0) / 5.0
                        for f in rows.frame_idx
                    ])
            n = _accumulate(grid, rows.pitch_x.to_numpy(), rows.pitch_y.to_numpy(),
                            pitch, weights)

        elif kind == "possession":
            frames = set()
            for span in self.spans:
                if span.state is not PossessionState.CONTROLLED:
                    continue
                if track_id is not None and span.track_id != track_id:
                    continue
                if resolved_team and span.team_id != resolved_team:
                    continue
                frames.update(range(span.start_frame, span.end_frame + 1))
            xs, ys = [], []
            for frame_idx in sorted(frames):
                position = self.ctx.ball_position(frame_idx)
                if position:
                    xs.append(position[0])
                    ys.append(position[1])
            n = _accumulate(grid, xs, ys, pitch)

        else:
            selectors: dict[str, tuple] = {
                "touches": ((EventType.BALL_TOUCH,), "start"),
                "carries": ((EventType.CARRY,), "start"),
                "pass_origin": ((EventType.PASS,), "start"),
                "pass_destination": ((EventType.PASS_SUCCESSFUL,), "end"),
                "defensive_actions": (
                    (EventType.INTERCEPTION, EventType.RECOVERY, EventType.CLEARANCE),
                    "start",
                ),
            }
            types, which = selectors[kind]
            xs, ys = [], []
            for event in self.events:
                if event.event_type not in types:
                    continue
                if track_id is not None and event.track_id != track_id:
                    continue
                if resolved_team and event.team_id != resolved_team:
                    continue
                if time_range and not (
                    time_range[0] <= event.timestamp_s <= time_range[1]
                ):
                    continue
                if half is not None and self.ctx.half_of(event.timestamp_s) != half:
                    continue
                x = event.start_x if which == "start" else event.end_x
                y = event.start_y if which == "start" else event.end_y
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            n = _accumulate(grid, xs, ys, pitch)

        total = grid.sum()
        normalised = (grid / total).tolist() if total > 0 else grid.tolist()

        times = list(self.ctx.timestamps.values())
        span = time_range or ((min(times), max(times)) if times else (0.0, 0.0))
        coverage = (
            self.kinematics[track_id].coverage
            if track_id is not None and track_id in self.kinematics
            else self.ctx.ball_coverage
        )

        return Heatmap(
            kind=kind, track_id=track_id, team_id=resolved_team, grid=normalised,
            n_samples=n, coverage=coverage, time_range_s=span, phase=phase, half=half,
        )


# --------------------------------------------------------------------------- #
# Passing network
# --------------------------------------------------------------------------- #


@dataclass
class PassingNetwork:
    team_id: str
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    window: str = "full_match"
    most_influential: int | None = None
    most_used_connection: tuple[int, int] | None = None
    isolated_players: list[int] = field(default_factory=list)
    dominant_side: str = "unknown"
    n_passes: int = 0

    def to_dict(self) -> dict:
        return {
            "team_id": self.team_id,
            "window": self.window,
            "nodes": self.nodes,
            "edges": self.edges,
            "most_influential": self.most_influential,
            "most_used_connection": list(self.most_used_connection)
            if self.most_used_connection else None,
            "isolated_players": self.isolated_players,
            "dominant_side": self.dominant_side,
            "n_passes": self.n_passes,
        }


def build_passing_network(
    context: AnalysisContext,
    events: list[FootballEvent],
    players: dict,
    team_id: str,
    time_range: tuple[float, float] | None = None,
    half: int | None = None,
    window_label: str = "full_match",
) -> PassingNetwork:
    """Player-to-player passing graph with centrality."""
    passes = [
        e for e in events
        if e.event_type is EventType.PASS_SUCCESSFUL
        and e.team_id == team_id
        and e.track_id is not None
        and e.related_track_id is not None
    ]
    if time_range:
        passes = [e for e in passes if time_range[0] <= e.timestamp_s <= time_range[1]]
    if half is not None:
        passes = [e for e in passes if context.half_of(e.timestamp_s) == half]

    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    progressive: dict[tuple[int, int], int] = defaultdict(int)
    for event in passes:
        key = (event.track_id, event.related_track_id)
        edge_counts[key] += 1

    progressive_ids = {
        (e.track_id, e.related_track_id)
        for e in events
        if e.event_type is EventType.PASS_PROGRESSIVE and e.team_id == team_id
    }
    for key in edge_counts:
        if key in progressive_ids:
            progressive[key] += 1

    involved = {tid for pair in edge_counts for tid in pair}
    members = {
        tid: profile for tid, profile in players.items() if profile.team_id == team_id
    }

    # Degree centrality normalised by the largest possible degree. With a
    # handful of passes this is a weak statistic, which is why n_passes travels
    # with the network.
    degree: dict[int, int] = defaultdict(int)
    for (a, b), count in edge_counts.items():
        degree[a] += count
        degree[b] += count
    max_degree = max(degree.values()) if degree else 1

    nodes = []
    for tid, profile in members.items():
        position = profile.average_position
        nodes.append({
            "track_id": tid,
            "display_name": profile.display_name,
            "x": round(position[0], 2) if position else None,
            "y": round(position[1], 2) if position else None,
            "passes": degree.get(tid, 0),
            "centrality": round(degree.get(tid, 0) / max_degree, 4) if max_degree else 0.0,
            "coverage": profile.coverage.to_dict(),
        })

    edges = [
        {
            "source": a, "target": b, "count": count,
            "progressive": progressive.get((a, b), 0),
            "weight": round(count / max(1, len(passes)), 4),
        }
        for (a, b), count in sorted(edge_counts.items(), key=lambda kv: -kv[1])
    ]

    positions = [
        profile.average_position for profile in members.values()
        if profile.average_position is not None
    ]
    dominant = "unknown"
    if positions:
        mean_y = float(np.mean([p[1] for p in positions]))
        half_width = context.pitch.width / 2
        if abs(mean_y - half_width) > context.pitch.width * 0.08:
            dominant = "left" if mean_y < half_width else "right"
        else:
            dominant = "central"

    return PassingNetwork(
        team_id=team_id,
        nodes=nodes,
        edges=edges,
        window=window_label,
        most_influential=max(degree, key=degree.get) if degree else None,
        most_used_connection=max(edge_counts, key=edge_counts.get) if edge_counts else None,
        isolated_players=sorted(set(members) - involved),
        dominant_side=dominant,
        n_passes=len(passes),
    )


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #


def build_timeline(
    context: AnalysisContext,
    events: list[FootballEvent],
    spans: list[PossessionSpan],
) -> dict:
    """Synchronized, filterable, seekable match timeline.

    Every entry carries the frame and timestamp needed to seek, plus the fields
    a UI filters on, so filtering never requires recomputation.
    """
    entries = []
    for event in sorted(events, key=lambda e: e.timestamp_s):
        entries.append({
            "event_id": event.event_id,
            "type": event.event_type.value,
            "timestamp_s": round(event.timestamp_s, 3),
            "frame_idx": event.frame_idx,
            "team_id": event.team_id,
            "track_id": event.track_id,
            "player_name": event.player_name,
            "related_track_id": event.related_track_id,
            "related_player_name": event.related_player_name,
            "confidence": round(event.confidence, 4),
            "confidence_band": event.band.value,
            "ball_coverage": round(event.ball_coverage, 4),
            "ball_state": event.ball_state.value,
            "half": context.half_of(event.timestamp_s),
            "clip": event.clip.to_dict() if event.clip else None,
        })

    possession_track = [
        {
            "start_s": round(s.start_time_s, 3),
            "end_s": round(s.end_time_s, 3),
            "state": s.state.value,
            "team_id": s.team_id,
            "track_id": s.track_id,
            "confidence": round(s.confidence, 4),
        }
        for s in spans
    ]

    return {
        "duration_s": round(context.duration_s, 2),
        "fps": round(context.fps, 4),
        "events": entries,
        "possession": possession_track,
        "filters": {
            "types": sorted({e["type"] for e in entries}),
            "teams": sorted({e["team_id"] for e in entries if e["team_id"]}),
            "players": sorted(
                {e["track_id"] for e in entries if e["track_id"] is not None}
            ),
            "confidence_bands": ["high", "probable", "uncertain"],
            "halves": sorted({e["half"] for e in entries}),
        },
    }
