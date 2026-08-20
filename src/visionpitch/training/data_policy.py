"""Fail-closed enforcement for training, validation, and final-holdout data."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

POLICY_SCHEMA = "VISIONPITCH_TRAINING_DATA_V1"
CANONICAL_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}


class DataBoundaryError(RuntimeError):
    """Raised when a data source cannot be proven safe for model development."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class TrainingDataPolicy:
    """Allow only official TRAIN/VALID data backed by complete provenance."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.official = {
            "train": (self.project_root / "data/SoccerNetGS/train").resolve(),
            "valid": (self.project_root / "data/SoccerNetGS/valid").resolve(),
        }
        self.legacy_test = (self.project_root / "data/eval/gsr").resolve()
        official_test = (self.project_root / "data/SoccerNetGS/test").resolve()
        self.test_roots = (self.legacy_test, official_test)
        self.test_sequence_ids = {
            path.name.casefold()
            for root in self.test_roots
            if root.exists()
            for path in root.iterdir()
            if path.is_dir()
        }
        self.holdout_sequence_ids: set[str] = set()
        self.holdout_paths: list[Path] = []
        holdout = self.project_root / "configs/evaluation/final_holdout_policy.json"
        if holdout.exists():
            payload = json.loads(holdout.read_text(encoding="utf-8"))
            for item in payload.get("holdout_items", []):
                sequence_id = item.get("sequence_id")
                if sequence_id:
                    self.holdout_sequence_ids.add(str(sequence_id).casefold())
                source = item.get("source_path")
                if source:
                    self.holdout_paths.append((self.project_root / source).resolve())

    def assert_sequence_allowed(self, sequence_id: str) -> None:
        folded = sequence_id.casefold()
        if folded in self.test_sequence_ids:
            raise DataBoundaryError(f"TEST sequence is quarantined: {sequence_id}")
        if folded in self.holdout_sequence_ids:
            raise DataBoundaryError(f"Final-holdout sequence is frozen: {sequence_id}")

    def assert_artifact_path_allowed(self, path: Path) -> Path:
        """Prevent exporters and trainers from writing inside protected roots."""
        resolved = path.resolve()
        for forbidden in (*self.test_roots, *self.holdout_paths):
            if _inside(resolved, forbidden):
                raise DataBoundaryError(f"Protected TEST/holdout path is forbidden: {path}")
        return resolved

    def assert_source_allowed(self, source: Path, split: str, sequence_id: str) -> Path:
        if split not in self.official:
            raise DataBoundaryError(f"Only official train/valid splits are allowed, got: {split}")
        self.assert_sequence_allowed(sequence_id)
        resolved = source.resolve()
        for forbidden in (*self.test_roots, *self.holdout_paths):
            if _inside(resolved, forbidden):
                raise DataBoundaryError(f"Protected TEST/holdout path is forbidden: {source}")
        allowed_root = self.official[split]
        if not _inside(resolved, allowed_root):
            raise DataBoundaryError(
                f"Source is not inside official {split.upper()}: {source} (expected {allowed_root})"
            )
        return resolved

    def validate_manifest(self, manifest_path: Path) -> dict:
        manifest_path = manifest_path.resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != POLICY_SCHEMA:
            raise DataBoundaryError("Missing or unsupported training provenance schema")
        names = {int(key): value for key, value in payload.get("class_names", {}).items()}
        if names != CANONICAL_NAMES:
            raise DataBoundaryError(f"Unexpected class mapping: {names}")
        if set(payload.get("source_splits", [])) != {"train", "valid"}:
            raise DataBoundaryError("Manifest must contain exactly official TRAIN and VALID")

        frame_file = manifest_path.parent / payload.get("frames_file", "")
        if not frame_file.is_file():
            raise DataBoundaryError("Manifest frame provenance file is missing")
        seen_sources: set[tuple[str, str, str]] = set()
        sequences: dict[str, set[str]] = {"train": set(), "valid": set()}
        count = 0
        with frame_file.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    split = row["source_split"]
                    sequence = row["source_sequence"]
                    frame = str(row["source_frame"])
                    source = self.project_root / row["source_path"]
                    exported_image = manifest_path.parent / row["exported_image"]
                    exported_label = manifest_path.parent / row["exported_label"]
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise DataBoundaryError(
                        f"Incomplete provenance at {frame_file}:{line_number}"
                    ) from exc
                self.assert_source_allowed(source, split, sequence)
                if not exported_image.is_file() or not exported_label.is_file():
                    raise DataBoundaryError(f"Missing exported pair at manifest row {line_number}")
                key = (split, sequence, frame)
                if key in seen_sources:
                    raise DataBoundaryError(f"Duplicate source frame in export: {key}")
                seen_sources.add(key)
                sequences[split].add(sequence)
                count += 1
        if sequences["train"] & sequences["valid"]:
            raise DataBoundaryError("TRAIN/VALID sequence overlap detected")
        if count != int(payload.get("exported_frames", -1)):
            raise DataBoundaryError("Manifest summary does not match frame provenance")
        return payload

    def validate_dataset_yaml(self, data_yaml: Path) -> dict:
        data_yaml = data_yaml.resolve()
        self.assert_artifact_path_allowed(data_yaml)
        if not data_yaml.is_file():
            raise DataBoundaryError(f"Dataset config not found: {data_yaml}")
        config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        dataset_root = Path(config.get("path", data_yaml.parent))
        if not dataset_root.is_absolute():
            dataset_root = (data_yaml.parent / dataset_root).resolve()
        manifest = dataset_root / "manifest.json"
        if not manifest.is_file():
            raise DataBoundaryError(
                f"Fail-closed: dataset has no canonical provenance manifest: {manifest}"
            )
        payload = self.validate_manifest(manifest)
        configured_names = config.get("names", {})
        if isinstance(configured_names, list):
            configured_names = dict(enumerate(configured_names))
        configured_names = {int(key): value for key, value in configured_names.items()}
        if configured_names != CANONICAL_NAMES:
            raise DataBoundaryError(
                f"dataset.yaml class mapping is not canonical: {configured_names}"
            )
        return payload


def find_project_root(start: Path) -> Path:
    """Find the repository root without relying on the caller's working directory."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/visionpitch").is_dir():
            return candidate
    raise DataBoundaryError(f"Could not locate VisionPitch project root from {start}")
