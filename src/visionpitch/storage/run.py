"""Run directories, stage checkpoints and the run manifest.

A *run* is one (video, resolved config) pair. Its directory holds every
intermediate artefact, so a later phase -- or a re-run after a crash -- can pick
up from stored results instead of re-decoding the video. Rule 19 of the brief.

Layout::

    outputs/<video_id>/<config_fingerprint>/
        config.yaml            fully resolved configuration
        manifest.json          provenance, model versions, data-quality summary
        detections.parquet     raw detector output
        tracks.parquet         one row per track, with inferred team/role
        calibration.parquet    one row per frame
        game_state.parquet     the canonical Phase 1 deliverable
        checkpoints/<stage>/   shard files + state.json for resume
        video/                 annotated.mp4, radar.mp4, combined.mp4
        evaluation/            metric reports
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionpitch.common.config import Config, save_config
from visionpitch.common.logging import get_logger
from visionpitch.common.schema import SCHEMA_VERSION
from visionpitch.ingestion.video import VideoMetadata

log = get_logger("storage")

STAGES = (
    "detection",
    "tracking",
    "ball_tracking",
    "team_classification",
    "calibration",
    "game_state",
    "visualization",
)


# --------------------------------------------------------------------------- #


@dataclass
class StageState:
    """Resume state for one stage."""

    stage: str
    completed: bool = False
    last_frame_idx: int | None = None
    n_shards: int = 0
    n_records: int = 0
    counters: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "completed": self.completed,
            "last_frame_idx": self.last_frame_idx,
            "n_shards": self.n_shards,
            "n_records": self.n_records,
            "counters": self.counters,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> StageState:
        return StageState(
            stage=data["stage"],
            completed=bool(data.get("completed", False)),
            last_frame_idx=data.get("last_frame_idx"),
            n_shards=int(data.get("n_shards", 0)),
            n_records=int(data.get("n_records", 0)),
            counters=data.get("counters", {}),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class RunManifest:
    """Everything needed to explain, reproduce or distrust a set of results."""

    video: dict[str, Any]
    config_fingerprint: str
    schema_version: str
    created_at: str
    visionpitch_version: str
    python_version: str
    platform: str
    torch_version: str | None = None
    cuda_device: str | None = None
    models: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "config_fingerprint": self.config_fingerprint,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "visionpitch_version": self.visionpitch_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "cuda_device": self.cuda_device,
            "models": self.models,
            "stages": self.stages,
            "data_quality": self.data_quality,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #


class RunContext:
    """Filesystem handle for one run, plus checkpoint bookkeeping."""

    def __init__(self, config: Config, metadata: VideoMetadata) -> None:
        self.config = config
        self.metadata = metadata
        self.fingerprint = config.fingerprint()
        self.root = (
            Path(config.storage.output_dir) / metadata.video_id / self.fingerprint
        ).resolve()
        self._manifest: RunManifest | None = None

    # -- directories -------------------------------------------------------- #

    def ensure(self) -> RunContext:
        for sub in ("", "checkpoints", "video", "evaluation"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        save_config(self.config, self.root / "config.yaml")
        return self

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def checkpoint_dir(self, stage: str) -> Path:
        d = self.root / "checkpoints" / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def table_path(self, name: str) -> Path:
        suffix = "json" if self.config.storage.format == "json" else "parquet"
        return self.root / f"{name}.{suffix}"

    # -- stage state -------------------------------------------------------- #

    def _state_path(self, stage: str) -> Path:
        return self.checkpoint_dir(stage) / "state.json"

    def load_state(self, stage: str) -> StageState:
        path = self._state_path(stage)
        if not path.exists():
            return StageState(stage=stage)
        try:
            return StageState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError) as exc:
            # A truncated state file means the process died mid-write. Discard it
            # and restart the stage rather than resuming from garbage.
            log.warning("discarding unreadable checkpoint for stage %r: %s", stage, exc)
            return StageState(stage=stage)

    def save_state(self, state: StageState) -> None:
        state.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        path = self._state_path(state.stage)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        # Atomic replace: a crash can never leave a half-written state file.
        tmp.replace(path)

    def is_complete(self, stage: str) -> bool:
        return self.load_state(stage).completed

    def mark_complete(self, stage: str, counters: dict[str, Any] | None = None) -> None:
        state = self.load_state(stage)
        state.completed = True
        if counters:
            state.counters = counters
        self.save_state(state)

    def reset_stage(self, stage: str) -> None:
        """Invalidate a stage and every stage that depends on it."""
        try:
            start = STAGES.index(stage)
        except ValueError as exc:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}") from exc
        for downstream in STAGES[start:]:
            path = self._state_path(downstream)
            if path.exists():
                path.unlink()
            for shard in self.checkpoint_dir(downstream).glob("shard_*"):
                shard.unlink()
        log.info("reset stage %r and %d downstream stage(s)", stage, len(STAGES) - start - 1)

    # -- manifest ----------------------------------------------------------- #

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def manifest(self) -> RunManifest:
        if self._manifest is None:
            if self.manifest_path.exists():
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self._manifest = RunManifest(
                    video=data["video"],
                    config_fingerprint=data["config_fingerprint"],
                    schema_version=data["schema_version"],
                    created_at=data["created_at"],
                    visionpitch_version=data["visionpitch_version"],
                    python_version=data["python_version"],
                    platform=data["platform"],
                    torch_version=data.get("torch_version"),
                    cuda_device=data.get("cuda_device"),
                    models=data.get("models", {}),
                    stages=data.get("stages", {}),
                    data_quality=data.get("data_quality", {}),
                    warnings=data.get("warnings", []),
                )
            else:
                self._manifest = self._new_manifest()
        return self._manifest

    def _new_manifest(self) -> RunManifest:
        from visionpitch import __version__

        torch_version, cuda_device = None, None
        try:
            import torch

            torch_version = torch.__version__
            if torch.cuda.is_available():
                cuda_device = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 - torch is optional at manifest time
            pass

        return RunManifest(
            video=self.metadata.to_dict(),
            config_fingerprint=self.fingerprint,
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            visionpitch_version=__version__,
            python_version=sys.version.split()[0],
            platform=f"{platform.system()} {platform.release()}",
            torch_version=torch_version,
            cuda_device=cuda_device,
        )

    def update_manifest(self, **updates: Any) -> None:
        manifest = self.manifest()
        for key, value in updates.items():
            current = getattr(manifest, key, None)
            if isinstance(current, dict) and isinstance(value, dict):
                current.update(value)
            elif isinstance(current, list) and isinstance(value, list):
                current.extend(value)
            else:
                setattr(manifest, key, value)
        self.write_manifest()

    def write_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest().to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)
