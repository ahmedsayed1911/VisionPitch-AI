"""The training boundary must fail closed around TEST and final holdout data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from visionpitch.training.data_policy import (
    CANONICAL_NAMES,
    POLICY_SCHEMA,
    DataBoundaryError,
    TrainingDataPolicy,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _manifest(project: Path, source: Path, split: str, sequence: str) -> Path:
    dataset = project / "data/export"
    image = _touch(dataset / "images/train/example.jpg")
    label = _touch(dataset / "labels/train/example.txt")
    row = {
        "source_split": split,
        "source_sequence": sequence,
        "source_frame": "000001",
        "source_path": source.relative_to(project).as_posix(),
        "exported_image": image.relative_to(dataset).as_posix(),
        "exported_label": label.relative_to(dataset).as_posix(),
    }
    (dataset / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    payload = {
        "schema": POLICY_SCHEMA,
        "class_names": {str(key): value for key, value in CANONICAL_NAMES.items()},
        "source_splits": ["train", "valid"],
        "frames_file": "frames.jsonl",
        "exported_frames": 1,
    }
    manifest = dataset / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    (dataset / "dataset.yaml").write_text(
        yaml.safe_dump({"path": str(dataset), "names": CANONICAL_NAMES}), encoding="utf-8"
    )
    return manifest


def test_rejects_test_path_even_when_declared_train(tmp_path: Path) -> None:
    source = _touch(tmp_path / "data/SoccerNetGS/test/SNGS-TEST/img1/000001.jpg")
    policy = TrainingDataPolicy(tmp_path)

    with pytest.raises(DataBoundaryError, match="TEST sequence|Protected TEST"):
        policy.assert_source_allowed(source, "train", "SNGS-TEST")


def test_rejects_known_test_sequence_id_on_train_path(tmp_path: Path) -> None:
    (tmp_path / "data/SoccerNetGS/test/SNGS-TEST").mkdir(parents=True)
    source = _touch(tmp_path / "data/SoccerNetGS/train/SNGS-TEST/img1/000001.jpg")
    policy = TrainingDataPolicy(tmp_path)

    with pytest.raises(DataBoundaryError, match="TEST sequence"):
        policy.assert_source_allowed(source, "train", "SNGS-TEST")


def test_rejects_file_from_wrong_official_split(tmp_path: Path) -> None:
    source = _touch(tmp_path / "data/SoccerNetGS/valid/SNGS-1/img1/000001.jpg")
    policy = TrainingDataPolicy(tmp_path)

    with pytest.raises(DataBoundaryError, match="official TRAIN"):
        policy.assert_source_allowed(source, "train", "SNGS-1")


def test_rejects_export_output_inside_test(tmp_path: Path) -> None:
    policy = TrainingDataPolicy(tmp_path)

    with pytest.raises(DataBoundaryError, match="Protected TEST"):
        policy.assert_artifact_path_allowed(tmp_path / "data/eval/gsr/export")


def test_dataset_yaml_without_provenance_fails_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "data/ad_hoc"
    dataset.mkdir(parents=True)
    config = dataset / "dataset.yaml"
    config.write_text(yaml.safe_dump({"path": str(dataset)}), encoding="utf-8")

    with pytest.raises(DataBoundaryError, match="Fail-closed"):
        TrainingDataPolicy(tmp_path).validate_dataset_yaml(config)


def test_manifest_rejects_test_origin(tmp_path: Path) -> None:
    source = _touch(tmp_path / "data/SoccerNetGS/test/SNGS-TEST/img1/000001.jpg")
    manifest = _manifest(tmp_path, source, "train", "SNGS-TEST")

    with pytest.raises(DataBoundaryError, match="TEST sequence|Protected TEST"):
        TrainingDataPolicy(tmp_path).validate_manifest(manifest)


def test_canonical_manifest_is_accepted(tmp_path: Path) -> None:
    source = _touch(tmp_path / "data/SoccerNetGS/train/SNGS-1/img1/000001.jpg")
    manifest = _manifest(tmp_path, source, "train", "SNGS-1")

    payload = TrainingDataPolicy(tmp_path).validate_manifest(manifest)

    assert payload["schema"] == POLICY_SCHEMA
