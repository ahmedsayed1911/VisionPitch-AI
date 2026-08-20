"""Reviewer corrections applied to a finished run.

The brief requires manual team correction *without re-running detection*. That
is achievable because the expensive stages wrote their results to Parquet: only
the track-level labels and the game-state join need redoing, which takes seconds
rather than the minutes or hours a full reprocess would.

Corrections are stored separately from the model's own output
(``corrections.json`` alongside the tables) so that the difference between what
the model said and what a human said stays visible and can be used as training
signal later.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger
from visionpitch.common.schema import display_name
from visionpitch.common.types import Role, TeamId, ValidationStatus
from visionpitch.evaluation.report import _load_calibration, _load_tracks
from visionpitch.game_state.assembler import GameStateAssembler
from visionpitch.pitch.geometry import PitchConfiguration
from visionpitch.storage.tables import (
    read_table,
    track_rows,
    write_game_state,
    write_tracks,
)

log = get_logger("pipeline.correct")


def apply_track_corrections(run_dir: str | Path, corrections_path: str | Path) -> int:
    """Apply track-level overrides and rebuild the game state.

    Returns the number of corrections applied.
    """
    run_dir = Path(run_dir)
    corrections = json.loads(Path(corrections_path).read_text(encoding="utf-8"))
    if not isinstance(corrections, dict):
        raise ValueError("corrections file must be a JSON object keyed by track id")

    config = Config.model_validate(
        yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    )
    pitch = PitchConfiguration(length=config.pitch.length_m, width=config.pitch.width_m)

    game_state_path = run_dir / "game_state.parquet"
    calibration_path = run_dir / "calibration.parquet"
    for path in (game_state_path, calibration_path):
        if not path.exists():
            raise FileNotFoundError(f"required table missing: {path}")

    tracks = _load_tracks(game_state_path)
    calibration = _load_calibration(calibration_path)

    applied = 0
    for raw_id, override in corrections.items():
        track = tracks.get(int(raw_id))
        if track is None:
            log.warning("correction for unknown track %s ignored", raw_id)
            continue
        if "team_id" in override:
            track.team_id = TeamId(override["team_id"])
            track.team_confidence = 1.0
        if "role" in override:
            track.role = Role(override["role"])
            track.role_confidence = 1.0
        if "jersey_number" in override:
            value = override["jersey_number"]
            track.jersey_number = int(value) if value is not None else None
            track.jersey_confidence = 1.0
        applied += 1

    # -- rebuild ------------------------------------------------------------- #
    gs = read_table(game_state_path).to_pydict()
    video_id = gs["video_id"][0] if gs["video_id"] else ""
    timestamps = {
        int(f): float(t) for f, t in zip(gs["frame_idx"], gs["timestamp_s"], strict=True)
    }
    frame_indices = sorted(timestamps)

    # The ball is rebuilt from the stored rows rather than re-estimated: nothing
    # about a team correction changes the ball's trajectory.
    ball_states = _ball_states_from_table(gs)

    # Preserve the extrapolation marking from the original run. The support
    # regions were computed during calibration and are not recoverable here, so
    # they are read back from the stored rows: a row already marked EXTRAPOLATED
    # stays that way. Rebuilding without them would silently promote every
    # far-side row to VALID, which is the opposite error to the one that made a
    # chunked run report zero usable rows, and just as wrong.
    previously_extrapolated = {
        (int(f), int(t))
        for f, t, s in zip(gs["frame_idx"], gs["track_id"], gs["validation_status"],
                           strict=True)
        if t is not None and s == ValidationStatus.EXTRAPOLATED.value
    }

    assembler = GameStateAssembler(video_id, pitch, config.calibration.min_confidence)
    rows = assembler.assemble(tracks, ball_states, calibration, timestamps, frame_indices)
    for row in rows:
        if (
            row.track_id is not None
            and (row.frame_idx, row.track_id) in previously_extrapolated
            and row.validation_status == ValidationStatus.VALID.value
        ):
            row.validation_status = ValidationStatus.EXTRAPOLATED.value

    write_game_state(rows, game_state_path, config.storage.compression)
    write_tracks(
        track_rows(video_id, tracks.values()),
        run_dir / "tracks.parquet",
        config.storage.compression,
    )

    record = {
        "applied_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_applied": applied,
        "corrections": corrections,
        "resulting_identities": {
            str(t.track_id): display_name(
                t.team_id.value, t.jersey_number, t.track_id, t.role.value
            )
            for t in tracks.values()
        },
    }
    history_path = run_dir / "corrections.json"
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    history.append(record)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    log.info("applied %d correction(s); game_state and tracks rebuilt", applied)
    return applied


def _ball_states_from_table(data: dict) -> dict:
    from visionpitch.common.types import BallState, BBox

    states: dict[int, BallState] = {}
    for i in range(len(data["frame_idx"])):
        if data["object_class"][i] != "ball":
            continue
        frame_idx = int(data["frame_idx"][i])
        states[frame_idx] = BallState(
            frame_idx=frame_idx,
            timestamp_s=float(data["timestamp_s"][i]),
            position=(float(data["image_x"][i]), float(data["image_y"][i])),
            bbox=BBox(
                float(data["bbox_x1"][i]),
                float(data["bbox_y1"][i]),
                float(data["bbox_x2"][i]),
                float(data["bbox_y2"][i]),
            ),
            velocity=None,
            confidence=float(data["tracking_confidence"][i]),
            observed=not bool(data["interpolated"][i]),
            interpolated=bool(data["interpolated"][i]),
        )
    return states
