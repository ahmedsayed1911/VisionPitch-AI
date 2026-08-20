"""The analysis context: Phase 1 outputs loaded once, filtered correctly.

Every analytics module reads from here rather than touching Parquet directly,
so the Phase 1B filtering rules are applied in exactly one place. In particular
``valid_players`` is the *only* view physical statistics may use, and it is a
distinct object from ``players`` so that using the wrong one is a visible
mistake at the call site rather than a forgotten predicate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from visionpitch.analytics.types import BallStateKind
from visionpitch.common.logging import get_logger
from visionpitch.pitch.geometry import PitchConfiguration

log = get_logger("analytics.context")

#: Statuses whose pitch coordinates are trustworthy enough for physical metrics.
PHYSICAL_STATUSES = ("valid",)
#: Additionally admissible for team-level aggregates, when labelled.
TEAM_LEVEL_STATUSES = ("valid", "extrapolated")


@dataclass
class AnalysisContext:
    """Everything Phase 2 needs from a completed Phase 1 run."""

    run_dir: Path
    video_id: str
    fps: float
    pitch: PitchConfiguration

    #: every person row, unfiltered
    players: pd.DataFrame
    #: person rows admissible for physical statistics
    valid_players: pd.DataFrame
    #: ball rows, with a ball_state column
    ball: pd.DataFrame
    #: one row per processed frame
    frames: pd.DataFrame
    #: track-level identity
    tracks: pd.DataFrame

    manifest: dict = field(default_factory=dict)

    # -- derived lookups ------------------------------------------------------ #
    ball_by_frame: dict[int, tuple[float, float, BallStateKind, float]] = field(
        default_factory=dict
    )
    timestamps: dict[int, float] = field(default_factory=dict)
    display_names: dict[int, str] = field(default_factory=dict)
    track_teams: dict[int, str] = field(default_factory=dict)
    track_roles: dict[int, str] = field(default_factory=dict)
    track_identity_confidence: dict[int, float] = field(default_factory=dict)

    # -- summary -------------------------------------------------------------- #

    @property
    def frame_indices(self) -> list[int]:
        return sorted(self.timestamps)

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def duration_s(self) -> float:
        if not self.timestamps:
            return 0.0
        return max(self.timestamps.values()) - min(self.timestamps.values())

    @property
    def ball_coverage(self) -> float:
        """Share of frames whose ball position is known at all."""
        if not self.n_frames:
            return 0.0
        known = sum(1 for v in self.ball_by_frame.values() if v[2].is_known)
        return known / self.n_frames

    @property
    def ball_observed_coverage(self) -> float:
        """Share of frames whose ball position was directly observed."""
        if not self.n_frames:
            return 0.0
        observed = sum(
            1 for v in self.ball_by_frame.values() if v[2] is BallStateKind.OBSERVED
        )
        return observed / self.n_frames

    def ball_state(self, frame_idx: int) -> BallStateKind:
        entry = self.ball_by_frame.get(frame_idx)
        return entry[2] if entry else BallStateKind.UNKNOWN

    def ball_position(self, frame_idx: int) -> tuple[float, float] | None:
        """Pitch position of the ball, or ``None`` when it is not known."""
        entry = self.ball_by_frame.get(frame_idx)
        if entry is None or not entry[2].is_known:
            return None
        if entry[0] is None or entry[1] is None:
            return None
        return (entry[0], entry[1])

    def half_of(self, timestamp_s: float) -> int:
        """1 or 2. Without detected half boundaries everything is half 1.

        Phase 1 only detects a halftime switch on clips long enough to contain
        one, and correctly reports none otherwise. Guessing a midpoint would
        put a fabricated boundary into every short clip.
        """
        switches = (self.manifest.get("stages", {})
                    .get("match_setup", {})
                    .get("direction_switch_frames") or [])
        if not switches:
            return 1
        first_switch_time = min(switches) / max(1e-6, self.fps)
        return 2 if timestamp_s >= first_switch_time else 1

    def summary(self) -> dict:
        return {
            "video_id": self.video_id,
            "fps": round(self.fps, 4),
            "frames": self.n_frames,
            "duration_s": round(self.duration_s, 2),
            "player_rows": len(self.players),
            "valid_player_rows": len(self.valid_players),
            "valid_row_ratio": round(
                len(self.valid_players) / max(1, len(self.players)), 4
            ),
            "tracks": len(self.tracks),
            "ball_coverage": round(self.ball_coverage, 4),
            "ball_observed_coverage": round(self.ball_observed_coverage, 4),
        }


def load_context(run_dir: str | Path) -> AnalysisContext:
    """Load a completed Phase 1 run for analysis.

    Reads only stored artefacts -- never the video -- so analytics can be
    re-run in about a second against a run that took a minute and a half to
    produce.
    """
    run_dir = Path(run_dir)
    game_state_path = run_dir / "game_state.parquet"
    if not game_state_path.exists():
        raise FileNotFoundError(
            f"no game_state.parquet in {run_dir}; run `visionpitch analyse` first"
        )

    game_state = pd.read_parquet(game_state_path)
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )

    frames_path = run_dir / "frames.parquet"
    if frames_path.exists():
        frames = pd.read_parquet(frames_path)
    else:
        # Phase 1 runs predating the frames table: reconstruct what we can, and
        # say so, rather than silently using the game state's frame set as the
        # denominator (which omits frames that detected nothing).
        log.warning(
            "no frames.parquet in %s; frame coverage denominators are taken from "
            "game_state and will be optimistic for frames with no detections",
            run_dir.name,
        )
        frames = (
            game_state[["video_id", "frame_idx", "timestamp_s"]]
            .drop_duplicates("frame_idx")
            .assign(n_persons=0, n_ball_rows=0, ball_observed=False,
                    calibration_confidence=0.0, calibration_valid=False,
                    calibration_propagated=False, segment_kind="unknown",
                    chunk_index=None)
        )

    tracks_path = run_dir / "tracks.parquet"
    tracks = (
        pd.read_parquet(tracks_path) if tracks_path.exists() else pd.DataFrame()
    )

    fps = float(manifest.get("video", {}).get("fps", 25.0))
    pitch_cfg = manifest.get("stages", {}).get("pitch", {})
    pitch = PitchConfiguration(
        length=float(pitch_cfg.get("length_m", 105.0)),
        width=float(pitch_cfg.get("width_m", 68.0)),
    )

    persons = game_state[game_state.object_class != "ball"].copy()
    valid_players = persons[
        persons.validation_status.isin(PHYSICAL_STATUSES) & persons.pitch_x.notna()
    ].copy()

    ball = game_state[game_state.object_class == "ball"].copy()
    ball["ball_state"] = np.where(
        ball.interpolated, BallStateKind.INTERPOLATED.value, BallStateKind.OBSERVED.value
    )

    timestamps = dict(
        zip(frames.frame_idx.astype(int), frames.timestamp_s.astype(float), strict=True)
    )

    ball_by_frame: dict[int, tuple[float, float, BallStateKind, float]] = {}
    for row in ball.itertuples(index=False):
        state = (
            BallStateKind.INTERPOLATED if row.interpolated else BallStateKind.OBSERVED
        )
        px = float(row.pitch_x) if pd.notna(row.pitch_x) else None
        py = float(row.pitch_y) if pd.notna(row.pitch_y) else None
        if px is None or py is None:
            # A ball seen in the image but without a trustworthy pitch position
            # is not usable for spatial analytics. Treated as unknown here
            # rather than dropped, so coverage reflects it.
            state = BallStateKind.UNKNOWN
        ball_by_frame[int(row.frame_idx)] = (px, py, state, float(row.tracking_confidence))

    for frame_idx in timestamps:
        ball_by_frame.setdefault(frame_idx, (None, None, BallStateKind.UNKNOWN, 0.0))

    display_names: dict[int, str] = {}
    track_teams: dict[int, str] = {}
    track_roles: dict[int, str] = {}
    identity_conf: dict[int, float] = {}
    if not tracks.empty:
        for row in tracks.itertuples(index=False):
            tid = int(row.track_id)
            display_names[tid] = str(row.display_name)
            track_teams[tid] = str(row.team_id)
            track_roles[tid] = str(row.role)
            identity_conf[tid] = float(row.team_confidence)

    context = AnalysisContext(
        run_dir=run_dir,
        video_id=str(game_state.video_id.iloc[0]) if len(game_state) else run_dir.name,
        fps=fps,
        pitch=pitch,
        players=persons,
        valid_players=valid_players,
        ball=ball,
        frames=frames,
        tracks=tracks,
        manifest=manifest,
        ball_by_frame=ball_by_frame,
        timestamps=timestamps,
        display_names=display_names,
        track_teams=track_teams,
        track_roles=track_roles,
        track_identity_confidence=identity_conf,
    )

    log.info(
        "analysis context: %d frames, %d player rows (%d usable), ball known in %.1f%% "
        "of frames (%.1f%% directly observed)",
        context.n_frames,
        len(persons),
        len(valid_players),
        100 * context.ball_coverage,
        100 * context.ball_observed_coverage,
    )
    return context
