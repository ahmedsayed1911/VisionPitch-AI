"""Parquet/JSON writers and readers for the canonical tables.

Writers accept plain dataclass rows and enforce the Arrow schema on the way out,
so a stage that forgets a field or writes the wrong dtype fails loudly at the
write rather than silently producing an unreadable table for Phase 2.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from visionpitch.common.logging import get_logger
from visionpitch.common.schema import (
    CALIBRATION_SCHEMA,
    DETECTIONS_SCHEMA,
    GAME_STATE_SCHEMA,
    TRACKS_SCHEMA,
    GameStateRow,
)
from visionpitch.common.types import (
    CalibrationResult,
    Detection,
    ObjectClass,
    Role,
    TeamId,
    Track,
)

log = get_logger("storage.tables")

_COMPRESSION_MAP = {"zstd": "zstd", "snappy": "snappy", "gzip": "gzip", "none": None}


# --------------------------------------------------------------------------- #
# Generic
# --------------------------------------------------------------------------- #


def _rows_to_table(rows: Sequence[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build a schema-conformant Arrow table, column by column.

    Column-wise construction is used instead of ``from_pylist`` because it lets
    Arrow cast each column against its declared type and raise a precise error
    naming the offending field.
    """
    if not rows:
        return schema.empty_table()

    columns = []
    for field in schema:
        values = [row.get(field.name) for row in rows]
        try:
            columns.append(pa.array(values, type=field.type))
        except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError) as exc:
            raise ValueError(
                f"column {field.name!r} does not conform to declared type "
                f"{field.type}: {exc}"
            ) from exc
    return pa.Table.from_arrays(columns, schema=schema)


def _write(
    rows: Sequence[dict[str, Any]],
    schema: pa.Schema,
    path: str | Path,
    compression: str = "zstd",
    fmt: str = "parquet",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        path = path.with_suffix(".json")
        path.write_text(json.dumps(list(rows), indent=2, default=_json_default), encoding="utf-8")
        log.info("wrote %d rows -> %s", len(rows), path.name)
        return path

    table = _rows_to_table(rows, schema)
    pq.write_table(
        table,
        path,
        compression=_COMPRESSION_MAP.get(compression, "zstd"),
        version="2.6",
    )
    log.info("wrote %d rows -> %s", table.num_rows, path.name)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def read_table(path: str | Path) -> pa.Table:
    """Read a table back, validating the schema version it was written with."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"table not found: {path}")
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    version = meta.get(b"visionpitch_schema_version")
    if version is not None:
        log.debug("%s written with schema version %s", path.name, version.decode())
    return table


# --------------------------------------------------------------------------- #
# Detections
# --------------------------------------------------------------------------- #


def detection_rows(
    video_id: str, frame_idx: int, timestamp_s: float, detections: Iterable[Detection]
) -> list[dict[str, Any]]:
    return [
        {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "timestamp_s": timestamp_s,
            "object_class": d.object_class.value,
            "bbox_x1": d.bbox.x1,
            "bbox_y1": d.bbox.y1,
            "bbox_x2": d.bbox.x2,
            "bbox_y2": d.bbox.y2,
            "confidence": d.confidence,
            "source": d.source,
        }
        for d in detections
    ]


def write_detections(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    compression: str = "zstd",
    fmt: str = "parquet",
) -> Path:
    return _write(rows, DETECTIONS_SCHEMA, path, compression, fmt)


# --------------------------------------------------------------------------- #
# Tracks
# --------------------------------------------------------------------------- #


def track_rows(video_id: str, tracks: Iterable[Track]) -> list[dict[str, Any]]:
    from visionpitch.common.schema import display_name

    rows = []
    for t in tracks:
        if not t.observations:
            continue
        rows.append(
            {
                "video_id": video_id,
                "track_id": t.track_id,
                "object_class": t.object_class.value,
                "first_frame": t.first_frame,
                "last_frame": t.last_frame,
                "n_observations": t.length,
                "team_id": t.team_id.value,
                "team_confidence": t.team_confidence,
                "role": t.role.value,
                "role_confidence": t.role_confidence,
                "jersey_number": t.jersey_number,
                "jersey_confidence": t.jersey_confidence,
                "display_name": display_name(
                    t.team_id.value, t.jersey_number, t.track_id, t.role.value
                ),
            }
        )
    return rows


def write_tracks(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    compression: str = "zstd",
    fmt: str = "parquet",
) -> Path:
    return _write(rows, TRACKS_SCHEMA, path, compression, fmt)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def calibration_rows(
    video_id: str, timestamps: dict[int, float], results: Iterable[CalibrationResult]
) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        rows.append(
            {
                "video_id": video_id,
                "frame_idx": r.frame_idx,
                "timestamp_s": timestamps.get(r.frame_idx, 0.0),
                "homography": (
                    [float(v) for v in np.asarray(r.homography).ravel()]
                    if r.homography is not None
                    else None
                ),
                "confidence": r.confidence,
                "reprojection_error_m": r.reprojection_error_m,
                "n_keypoints": r.n_keypoints,
                "n_inliers": r.n_inliers,
                "smoothed": r.smoothed,
                "segment_kind": r.segment_kind.value,
            }
        )
    return rows


def frame_rows(
    video_id: str,
    frame_indices: Sequence[int],
    timestamps: dict[int, float],
    rows: Sequence[GameStateRow],
    calibration: dict,
    chunk_of: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    """One record per processed frame, including frames with nothing in them."""
    from visionpitch.common.types import ObjectClass

    persons: dict[int, int] = {}
    ball_rows: dict[int, int] = {}
    ball_observed: dict[int, bool] = {}
    for row in rows:
        if row.object_class == ObjectClass.BALL.value:
            ball_rows[row.frame_idx] = ball_rows.get(row.frame_idx, 0) + 1
            if not row.interpolated:
                ball_observed[row.frame_idx] = True
        else:
            persons[row.frame_idx] = persons.get(row.frame_idx, 0) + 1

    out = []
    for frame_idx in frame_indices:
        result = calibration.get(frame_idx)
        out.append(
            {
                "video_id": video_id,
                "frame_idx": int(frame_idx),
                "timestamp_s": float(timestamps.get(frame_idx, 0.0)),
                "n_persons": int(persons.get(frame_idx, 0)),
                "n_ball_rows": int(ball_rows.get(frame_idx, 0)),
                "ball_observed": bool(ball_observed.get(frame_idx, False)),
                "calibration_confidence": float(result.confidence) if result else 0.0,
                "calibration_valid": bool(result.is_valid) if result else False,
                "calibration_propagated": bool(result.smoothed) if result else False,
                "segment_kind": result.segment_kind.value if result else "unknown",
                "chunk_index": (chunk_of or {}).get(frame_idx),
            }
        )
    return out


def write_frames(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    compression: str = "zstd",
    fmt: str = "parquet",
) -> Path:
    from visionpitch.common.schema import FRAMES_SCHEMA

    return _write(rows, FRAMES_SCHEMA, path, compression, fmt)


def homography_from_row(raw) -> np.ndarray | None:
    """Rebuild a 3x3 homography from a stored row, or ``None``.

    The stored column is a variable-length list, so length is checked here
    rather than by the schema. A row that is neither null nor 9 long means the
    file was written by something that did not respect this contract, and is
    surfaced rather than reshaped into nonsense.
    """
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64).ravel()
    if values.size == 0:
        return None
    if values.size != 9:
        raise ValueError(
            f"homography column holds {values.size} values; expected 9 (row-major 3x3)"
        )
    return values.reshape(3, 3)


def write_calibration(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    compression: str = "zstd",
    fmt: str = "parquet",
) -> Path:
    return _write(rows, CALIBRATION_SCHEMA, path, compression, fmt)


# --------------------------------------------------------------------------- #
# Game state
# --------------------------------------------------------------------------- #


def write_game_state(
    rows: Sequence[GameStateRow | dict[str, Any]],
    path: str | Path,
    compression: str = "zstd",
    fmt: str = "parquet",
) -> Path:
    payload = [r.to_dict() if isinstance(r, GameStateRow) else r for r in rows]
    return _write(payload, GAME_STATE_SCHEMA, path, compression, fmt)


def load_game_state(path: str | Path):
    """Read the game state into pandas. The entry point Phase 2 will use."""
    return read_table(path).to_pandas()


# --------------------------------------------------------------------------- #
# Round-trip helpers used by tracking / team classification resume
# --------------------------------------------------------------------------- #


def detections_from_table(table: pa.Table) -> dict[int, list[Detection]]:
    """Rebuild per-frame detections from a stored detections table."""
    from visionpitch.common.types import BBox

    data = table.to_pydict()
    out: dict[int, list[Detection]] = {}
    for i in range(table.num_rows):
        frame_idx = int(data["frame_idx"][i])
        out.setdefault(frame_idx, []).append(
            Detection(
                frame_idx=frame_idx,
                object_class=ObjectClass(data["object_class"][i]),
                bbox=BBox(
                    float(data["bbox_x1"][i]),
                    float(data["bbox_y1"][i]),
                    float(data["bbox_x2"][i]),
                    float(data["bbox_y2"][i]),
                ),
                confidence=float(data["confidence"][i]),
                source=str(data["source"][i]),
            )
        )
    return out


def apply_track_labels(tracks: dict[int, Track], table: pa.Table) -> None:
    """Re-apply stored team/role/jersey labels onto in-memory tracks.

    This is what makes manual team correction cheap: edit the tracks table, then
    rebuild the game state without re-running detection or tracking.
    """
    data = table.to_pydict()
    for i in range(table.num_rows):
        track_id = int(data["track_id"][i])
        track = tracks.get(track_id)
        if track is None:
            continue
        track.team_id = TeamId(data["team_id"][i])
        track.team_confidence = float(data["team_confidence"][i])
        track.role = Role(data["role"][i])
        track.role_confidence = float(data["role_confidence"][i])
        jersey = data["jersey_number"][i]
        track.jersey_number = int(jersey) if jersey is not None else None
        track.jersey_confidence = float(data["jersey_confidence"][i])
