"""Analytics orchestration.

Reads a completed Phase 1 run, produces the full analytics artefact set, and
writes it beside the vision outputs. Never touches the video, so a threshold
change is a one-second experiment rather than a ninety-second one.

Artefacts, all under ``<run_dir>/analytics/``::

    events.parquet          one row per detected event
    possession.parquet      one row per possession span
    player_stats.json       per-player metrics with coverage
    goalkeeper_stats.json   goalkeeper-specific analytics
    team_stats.json         team aggregates
    heatmaps.json           precomputed surfaces for every player and team
    networks.json           passing networks, full match and per half
    timeline.json           synchronized, filterable event timeline
    summary.json            match overview plus the data-quality header
    manifest.json           analytics schema version, config, provenance

The web layer reads these and recomputes nothing, so the dashboard and the
exports cannot disagree.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from visionpitch.analytics import events as event_module
from visionpitch.analytics import possession as possession_module
from visionpitch.analytics.context import AnalysisContext, load_context
from visionpitch.analytics.kinematics import compute_all
from visionpitch.analytics.players import (
    build_goalkeeper_profiles,
    build_player_profiles,
    build_team_profiles,
    summarise_counts,
)
from visionpitch.analytics.spatial import (
    HeatmapEngine,
    build_passing_network,
    build_timeline,
)
from visionpitch.analytics.types import is_team
from visionpitch.common.logging import get_logger

log = get_logger("analytics.runner")

ANALYTICS_SCHEMA_VERSION = "2.0.0"

#: Heatmaps precomputed for every player. Kept to the ones a dashboard shows by
#: default; the engine can build any of the nine on demand through the API.
DEFAULT_PLAYER_HEATMAPS = ("position", "touches", "carries", "defensive_actions")
DEFAULT_TEAM_HEATMAPS = ("position", "possession", "pass_origin", "pass_destination")


EVENTS_SCHEMA = pa.schema([
    pa.field("event_id", pa.string(), nullable=False),
    pa.field("event_type", pa.string(), nullable=False),
    pa.field("frame_idx", pa.int32(), nullable=False),
    pa.field("timestamp_s", pa.float64(), nullable=False),
    pa.field("team_id", pa.string(), nullable=False),
    pa.field("track_id", pa.int32(), nullable=True),
    pa.field("player_name", pa.string(), nullable=False),
    pa.field("related_track_id", pa.int32(), nullable=True),
    pa.field("related_player_name", pa.string(), nullable=False),
    pa.field("related_team_id", pa.string(), nullable=False),
    pa.field("confidence", pa.float32(), nullable=False),
    pa.field("confidence_band", pa.string(), nullable=False),
    pa.field("ball_coverage", pa.float32(), nullable=False),
    pa.field("ball_state", pa.string(), nullable=False),
    pa.field("start_x", pa.float32(), nullable=True),
    pa.field("start_y", pa.float32(), nullable=True),
    pa.field("end_x", pa.float32(), nullable=True),
    pa.field("end_y", pa.float32(), nullable=True),
    pa.field("distance_m", pa.float32(), nullable=True),
    pa.field("duration_s", pa.float32(), nullable=True),
    pa.field("clip_frame_start", pa.int32(), nullable=True),
    pa.field("clip_frame_end", pa.int32(), nullable=True),
    pa.field("clip_time_start_s", pa.float64(), nullable=True),
    pa.field("clip_time_end_s", pa.float64(), nullable=True),
    pa.field("evidence", pa.string(), nullable=False),
], metadata={"visionpitch_analytics_version": ANALYTICS_SCHEMA_VERSION})


POSSESSION_SCHEMA = pa.schema([
    pa.field("start_frame", pa.int32(), nullable=False),
    pa.field("end_frame", pa.int32(), nullable=False),
    pa.field("start_time_s", pa.float64(), nullable=False),
    pa.field("end_time_s", pa.float64(), nullable=False),
    pa.field("duration_s", pa.float64(), nullable=False),
    pa.field("state", pa.string(), nullable=False),
    pa.field("team_id", pa.string(), nullable=False),
    pa.field("track_id", pa.int32(), nullable=True),
    pa.field("player_name", pa.string(), nullable=False),
    pa.field("confidence", pa.float32(), nullable=False),
    pa.field("ball_coverage", pa.float32(), nullable=False),
], metadata={"visionpitch_analytics_version": ANALYTICS_SCHEMA_VERSION})


@dataclass
class AnalyticsResult:
    run_dir: Path
    analytics_dir: Path
    n_events: int = 0
    n_spans: int = 0
    n_players: int = 0
    n_goalkeepers: int = 0
    timings_s: dict[str, float] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _events_table(events) -> pa.Table:
    rows = []
    for e in events:
        d = e.to_dict()
        clip = d.pop("clip") or {}
        rows.append({
            **{k: v for k, v in d.items() if k != "evidence"},
            "clip_frame_start": clip.get("frame_start"),
            "clip_frame_end": clip.get("frame_end"),
            "clip_time_start_s": clip.get("time_start_s"),
            "clip_time_end_s": clip.get("time_end_s"),
            "evidence": json.dumps(d["evidence"]),
        })
    columns = []
    for field_ in EVENTS_SCHEMA:
        columns.append(pa.array([r.get(field_.name) for r in rows], type=field_.type))
    return pa.Table.from_arrays(columns, schema=EVENTS_SCHEMA)


def _possession_table(spans) -> pa.Table:
    rows = [s.to_dict() for s in spans]
    columns = []
    for field_ in POSSESSION_SCHEMA:
        columns.append(pa.array([r.get(field_.name) for r in rows], type=field_.type))
    return pa.Table.from_arrays(columns, schema=POSSESSION_SCHEMA)


def run_analytics(
    run_dir: str | Path,
    context: AnalysisContext | None = None,
) -> AnalyticsResult:
    """Produce the complete Phase 2 analytics artefact set for a run."""
    run_dir = Path(run_dir)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    ctx = context or load_context(run_dir)
    timings["load"] = round(time.perf_counter() - started, 3)

    mark = time.perf_counter()
    spans, per_frame, possession_summary = possession_module.run(ctx)
    timings["possession"] = round(time.perf_counter() - mark, 3)

    mark = time.perf_counter()
    detected = event_module.run(ctx, spans)
    timings["events"] = round(time.perf_counter() - mark, 3)

    mark = time.perf_counter()
    kinematics = compute_all(ctx.valid_players, ctx.players, ctx.fps)
    timings["kinematics"] = round(time.perf_counter() - mark, 3)

    mark = time.perf_counter()
    players = build_player_profiles(ctx, kinematics, detected, spans)
    goalkeepers = build_goalkeeper_profiles(ctx, players, detected)
    teams = build_team_profiles(ctx, players, detected, possession_summary)
    timings["profiles"] = round(time.perf_counter() - mark, 3)

    mark = time.perf_counter()
    heatmap_engine = HeatmapEngine(ctx, detected, spans, kinematics)
    heatmaps: dict[str, list[dict]] = {"players": [], "teams": []}
    for track_id, profile in players.items():
        if profile.coverage.tracking < 0.01:
            continue
        for kind in DEFAULT_PLAYER_HEATMAPS:
            heatmaps["players"].append(
                heatmap_engine.build(kind, track_id=track_id).to_dict()
            )
    for team_id in teams:
        for kind in DEFAULT_TEAM_HEATMAPS:
            heatmaps["teams"].append(
                heatmap_engine.build(kind, team_id=team_id).to_dict()
            )
    timings["heatmaps"] = round(time.perf_counter() - mark, 3)

    mark = time.perf_counter()
    networks = []
    for team_id in teams:
        networks.append(
            build_passing_network(ctx, detected, players, team_id).to_dict()
        )
        for half in sorted({ctx.half_of(t) for t in ctx.timestamps.values()}):
            networks.append(
                build_passing_network(
                    ctx, detected, players, team_id, half=half,
                    window_label=f"half_{half}",
                ).to_dict()
            )
    timeline = build_timeline(ctx, detected, spans)
    timings["spatial"] = round(time.perf_counter() - mark, 3)

    # -- persist -------------------------------------------------------------- #
    analytics_dir = run_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    pq.write_table(_events_table(detected), analytics_dir / "events.parquet",
                   compression="zstd")
    outputs["events"] = str(analytics_dir / "events.parquet")
    pq.write_table(_possession_table(spans), analytics_dir / "possession.parquet",
                   compression="zstd")
    outputs["possession"] = str(analytics_dir / "possession.parquet")

    outputs["players"] = str(_write_json(
        analytics_dir / "player_stats.json",
        {str(k): v.to_dict() for k, v in players.items()},
    ))
    outputs["goalkeepers"] = str(_write_json(
        analytics_dir / "goalkeeper_stats.json",
        {str(k): v.to_dict() for k, v in goalkeepers.items()},
    ))
    outputs["teams"] = str(_write_json(
        analytics_dir / "team_stats.json",
        {k: v.to_dict() for k, v in teams.items()},
    ))
    outputs["heatmaps"] = str(_write_json(analytics_dir / "heatmaps.json", heatmaps))
    outputs["networks"] = str(_write_json(analytics_dir / "networks.json", networks))
    outputs["timeline"] = str(_write_json(analytics_dir / "timeline.json", timeline))

    summary = {
        "video_id": ctx.video_id,
        "context": ctx.summary(),
        "possession": possession_summary,
        "event_counts": summarise_counts(detected),
        "n_players": len(players),
        "n_goalkeepers": len(goalkeepers),
        "teams": sorted(teams),
        # The header a consumer must read before trusting anything below it.
        "data_quality": {
            "ball_known_pct": round(100 * ctx.ball_coverage, 1),
            "ball_observed_pct": round(100 * ctx.ball_observed_coverage, 1),
            "valid_player_row_pct": round(
                100 * len(ctx.valid_players) / max(1, len(ctx.players)), 1
            ),
            "possession_determinable_pct": round(
                100 * possession_summary.get("determinable_ratio", 0.0), 1
            ),
            "tracks_without_team": sum(
                1 for p in players.values() if not is_team(p.team_id)
            ),
            "warnings": _quality_warnings(ctx, players, possession_summary),
        },
    }
    outputs["summary"] = str(_write_json(analytics_dir / "summary.json", summary))

    manifest = {
        "analytics_schema_version": ANALYTICS_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "vision_config_fingerprint": ctx.manifest.get("config_fingerprint"),
        "vision_models": ctx.manifest.get("models", {}),
        "timings_s": timings,
        "counts": {
            "events": len(detected),
            "possession_spans": len(spans),
            "players": len(players),
            "goalkeepers": len(goalkeepers),
        },
    }
    outputs["manifest"] = str(_write_json(analytics_dir / "manifest.json", manifest))

    timings["total"] = round(time.perf_counter() - started, 3)
    log.info(
        "analytics complete in %.2fs: %d events, %d spans, %d players",
        timings["total"], len(detected), len(spans), len(players),
    )
    _ = per_frame

    return AnalyticsResult(
        run_dir=run_dir,
        analytics_dir=analytics_dir,
        n_events=len(detected),
        n_spans=len(spans),
        n_players=len(players),
        n_goalkeepers=len(goalkeepers),
        timings_s=timings,
        summary=summary,
        outputs=outputs,
    )


def _quality_warnings(ctx, players, possession_summary) -> list[str]:
    """The caveats a reader must see before quoting any number here."""
    warnings: list[str] = []

    determinable = possession_summary.get("determinable_ratio", 0.0)
    if determinable < 0.5:
        warnings.append(
            f"Possession could be determined for only {100 * determinable:.0f}% of the "
            f"analysed time. Possession shares are shares of that subset, not of the match."
        )
    if ctx.ball_observed_coverage < 0.7:
        warnings.append(
            f"The ball was directly observed in {100 * ctx.ball_observed_coverage:.0f}% of "
            f"frames; the rest is interpolated or unknown. Every ball-dependent metric "
            f"inherits this."
        )
    valid_ratio = len(ctx.valid_players) / max(1, len(ctx.players))
    if valid_ratio < 0.5:
        warnings.append(
            f"Only {100 * valid_ratio:.0f}% of player rows have pitch coordinates "
            f"trustworthy enough for physical statistics. Distance and speed totals are "
            f"lower bounds."
        )
    unteamed = sum(1 for p in players.values() if not is_team(p.team_id))
    if unteamed > 0.2 * max(1, len(players)):
        warnings.append(
            f"{unteamed} of {len(players)} tracks have no confident team assignment and "
            f"are excluded from team aggregates."
        )
    if not any(p.role == "goalkeeper" for p in players.values()):
        warnings.append(
            "No goalkeeper was identified in this footage, so goalkeeper analytics is empty."
        )
    return warnings


def _unused() -> None:  # pragma: no cover
    _ = asdict
