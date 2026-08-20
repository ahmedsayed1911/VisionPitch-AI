"""Configuration layering and the storage schema contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

from visionpitch.common.config import AnalysisMode, Config, load_config, save_config
from visionpitch.common.schema import (
    GAME_STATE_SCHEMA,
    SCHEMA_VERSION,
    GameStateRow,
    display_name,
    empty_row,
)
from visionpitch.common.types import BBox, ObjectClass, Role, TeamId, Track, TrackObservation
from visionpitch.storage.tables import track_rows, write_game_state, write_tracks


class TestConfigLayering:
    def test_defaults_load_without_a_file(self) -> None:
        config = load_config(config_root=Path("/nonexistent"))
        assert config.mode is AnalysisMode.BALANCED
        assert config.detection.imgsz == 1280

    def test_mode_overlay_is_applied(self, repo_root: Path) -> None:
        fast = load_config(config_root=repo_root, mode=AnalysisMode.FAST_PREVIEW)
        accurate = load_config(config_root=repo_root, mode=AnalysisMode.MAX_ACCURACY)
        assert fast.ingestion.frame_stride == 3
        assert fast.ball_detection.enabled is False
        assert accurate.ball_detection.enabled is True
        assert accurate.detection.augment is True

    def test_inference_resolution_never_exceeds_training_resolution(
        self, repo_root: Path
    ) -> None:
        """Regression guard for a measured finding: the shipped checkpoint was
        fine-tuned at 1280, and running above it lost referee detections."""
        for mode in AnalysisMode:
            config = load_config(config_root=repo_root, mode=mode)
            assert config.detection.imgsz <= 1280, f"{mode.value} exceeds training imgsz"

    def test_cli_set_overrides_win(self, repo_root: Path) -> None:
        config = load_config(
            config_root=repo_root,
            cli_sets=["detection.conf_threshold=0.42", "runtime.half_precision=false"],
        )
        assert config.detection.conf_threshold == 0.42
        assert config.runtime.half_precision is False

    def test_malformed_set_is_rejected(self, repo_root: Path) -> None:
        with pytest.raises(ValueError, match="key.path=value"):
            load_config(config_root=repo_root, cli_sets=["nonsense"])

    def test_unknown_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Config.model_validate({"detection": {"not_a_real_key": 1}})

    def test_out_of_range_values_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Config.model_validate({"detection": {"conf_threshold": 1.5}})
        with pytest.raises(ValidationError):
            Config.model_validate({"pitch": {"length_m": 500}})

    def test_end_time_must_follow_start_time(self) -> None:
        with pytest.raises(ValidationError, match="end_time_s"):
            Config.model_validate({"ingestion": {"start_time_s": 10.0, "end_time_s": 5.0}})

    def test_fingerprint_is_stable_and_sensitive(self, repo_root: Path) -> None:
        a = load_config(config_root=repo_root)
        b = load_config(config_root=repo_root)
        assert a.fingerprint() == b.fingerprint()
        c = load_config(config_root=repo_root, cli_sets=["detection.imgsz=960"])
        assert c.fingerprint() != a.fingerprint()

    def test_round_trips_through_yaml(self, repo_root: Path, tmp_path: Path) -> None:
        config = load_config(config_root=repo_root, mode=AnalysisMode.MAX_ACCURACY)
        path = tmp_path / "config.yaml"
        save_config(config, path)
        restored = Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert restored.fingerprint() == config.fingerprint()

    def test_every_shipped_config_is_valid(self, repo_root: Path) -> None:
        for mode in AnalysisMode:
            load_config(config_root=repo_root, mode=mode)


class TestSchema:
    def test_row_matches_arrow_schema_field_for_field(self) -> None:
        row = empty_row("v", 0, 0.0)
        assert set(row.to_dict()) == {f.name for f in GAME_STATE_SCHEMA}

    def test_schema_version_is_recorded(self) -> None:
        assert GAME_STATE_SCHEMA.metadata[b"visionpitch_schema_version"].decode() == SCHEMA_VERSION

    def test_write_and_read_round_trip(self, tmp_path: Path) -> None:
        rows = [empty_row("clip", f, f / 25.0) for f in range(5)]
        rows[2].pitch_x, rows[2].pitch_y = 52.5, 34.0
        path = write_game_state(rows, tmp_path / "gs.parquet")

        table = pq.read_table(path)
        assert table.num_rows == 5
        assert table.column("pitch_x").to_pylist()[2] == pytest.approx(52.5)
        # A null pitch coordinate must survive as null, not as 0.0.
        assert table.column("pitch_x").to_pylist()[0] is None

    def test_wrong_dtype_fails_loudly(self, tmp_path: Path) -> None:
        row = empty_row("clip", 0, 0.0).to_dict()
        row["frame_idx"] = "not an integer"
        with pytest.raises(ValueError, match="frame_idx"):
            write_game_state([row], tmp_path / "bad.parquet")

    def test_calibration_round_trips_with_null_homographies(self, tmp_path: Path) -> None:
        """Uncalibrated frames are normal, and the table must survive them.

        Regression test: the homography column was once a fixed-size list, which
        wrote nulls as zero-length lists. The file was produced without error and
        then failed to read with "Expected all lists to be of size=9", silently
        destroying the calibration output every run that had an uncalibrated
        frame -- which is every real run.
        """
        from visionpitch.common.types import CalibrationResult
        from visionpitch.storage.tables import (
            calibration_rows,
            homography_from_row,
            write_calibration,
        )

        H = np.eye(3)
        results = [
            CalibrationResult(0, H, 0.8, 0.4, 10, 9),
            CalibrationResult(1, None, 0.0, float("nan"), 0, 0),  # uncalibrated
            CalibrationResult(2, H * 2, 0.6, 0.5, 8, 7),
        ]
        rows = calibration_rows("clip", {0: 0.0, 1: 0.04, 2: 0.08}, results)
        path = write_calibration(rows, tmp_path / "calibration.parquet")

        table = pq.read_table(path)  # must not raise
        assert table.num_rows == 3

        column = table.column("homography").to_pylist()
        assert column[1] is None
        assert homography_from_row(column[0]).shape == (3, 3)
        assert homography_from_row(column[1]) is None
        assert np.allclose(homography_from_row(column[2]), H * 2)

    def test_malformed_homography_is_reported_not_reshaped(self) -> None:
        from visionpitch.storage.tables import homography_from_row

        with pytest.raises(ValueError, match="expected 9"):
            homography_from_row([1.0, 2.0, 3.0])

    def test_json_format_is_supported(self, tmp_path: Path) -> None:
        rows = [empty_row("clip", 0, 0.0)]
        path = write_game_state(rows, tmp_path / "gs", fmt="json")
        assert json.loads(path.read_text(encoding="utf-8"))[0]["video_id"] == "clip"


class TestDisplayName:
    def test_uses_the_jersey_number_when_known(self) -> None:
        assert display_name("A", 10, 3, "outfield") == "Team A - Player #10"

    def test_never_invents_a_number(self) -> None:
        name = display_name("B", None, 7, "outfield")
        assert "#" not in name
        assert "B07" in name

    def test_goalkeeper_and_referee_are_distinguished(self) -> None:
        assert "Goalkeeper" in display_name("A", 1, 4, "goalkeeper")
        assert display_name("none", None, 9, "referee") == "Referee 9"

    def test_is_stable_for_the_same_track(self) -> None:
        assert display_name("A", None, 12, "outfield") == display_name("A", None, 12, "outfield")


class TestTrackTable:
    def test_track_rows_carry_inferred_labels(self, tmp_path: Path) -> None:
        track = Track(
            track_id=5,
            object_class=ObjectClass.GOALKEEPER,
            observations=[TrackObservation(0, 0.0, BBox(0, 0, 10, 20), 0.9, 0.9, False)],
            team_id=TeamId.B,
            team_confidence=0.77,
            role=Role.GOALKEEPER,
            role_confidence=0.9,
        )
        rows = track_rows("clip", [track])
        assert rows[0]["team_id"] == "B"
        assert rows[0]["display_name"] == "Team B Goalkeeper - Player B05"

        path = write_tracks(rows, tmp_path / "tracks.parquet")
        assert pq.read_table(path).num_rows == 1

    def test_empty_tracks_are_skipped(self) -> None:
        empty = Track(track_id=1, object_class=ObjectClass.PLAYER, observations=[])
        assert track_rows("clip", [empty]) == []


def test_track_observation_lookup_is_correct() -> None:
    track = Track(
        track_id=1,
        object_class=ObjectClass.PLAYER,
        observations=[
            TrackObservation(f, f / 25.0, BBox(f, 0, f + 10, 20), 0.9, 0.9, False)
            for f in (0, 5, 10, 17, 30)
        ],
    )
    assert track.observation_at(17).bbox.x1 == 17
    assert track.observation_at(11) is None
    assert track.first_frame == 0 and track.last_frame == 30
    assert track.length == 5 and track.span == 31


def test_game_state_row_is_slotted_and_complete() -> None:
    row = GameStateRow(**empty_row("v", 1, 0.04).to_dict())
    assert row.frame_idx == 1
    assert np.isclose(row.timestamp_s, 0.04)

class TestConfigExtends:
    """`extends:` layering, and the silent mis-resolution it exists to prevent."""

    def test_extends_merges_the_parent_underneath(self, tmp_path) -> None:
        configs = tmp_path / "configs"
        configs.mkdir()
        (configs / "base.yaml").write_text(
            "detection:\n  imgsz: 1280\n  conf_threshold: 0.25\n"
            "calibration:\n  imgsz: 960\n",
            encoding="utf-8",
        )
        (configs / "child.yaml").write_text(
            "extends: base.yaml\ndetection:\n  conf_threshold: 0.5\n",
            encoding="utf-8",
        )
        cfg = load_config(config_path=configs / "child.yaml", config_root=tmp_path)
        assert cfg.detection.conf_threshold == 0.5, "child should win"
        assert cfg.detection.imgsz == 1280, "parent key should survive"
        assert cfg.calibration.imgsz == 960, "parent section should survive"

    def test_a_missing_parent_is_an_error_not_a_silent_default(self, tmp_path) -> None:
        configs = tmp_path / "configs"
        configs.mkdir()
        (configs / "child.yaml").write_text("extends: nope.yaml\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="nope.yaml"):
            load_config(config_path=configs / "child.yaml", config_root=tmp_path)

    def test_circular_extends_is_rejected(self, tmp_path) -> None:
        configs = tmp_path / "configs"
        configs.mkdir()
        (configs / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
        (configs / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
        with pytest.raises(ValueError, match="circular"):
            load_config(config_path=configs / "a.yaml", config_root=tmp_path)

    def test_showcase_config_keeps_the_validated_analysis_settings(self) -> None:
        """The regression this mechanism was added for.

        `config_path` replaces the default file rather than overriding it, so a
        visualization-only overlay used to resolve every analysis setting to its
        field default. Both the wrong calibration imgsz and the wrong ball
        checkpoint are individually legal values, so nothing raised -- the run
        just quietly stopped using the validated stack.
        """
        showcase = Path("configs/showcase.yaml")
        if not showcase.exists():  # pragma: no cover - repo layout guard
            pytest.skip("configs/showcase.yaml not present")
        cfg = load_config(config_path=showcase, mode="balanced")
        default = load_config(mode="balanced")

        assert cfg.calibration.imgsz == default.calibration.imgsz == 960
        assert cfg.ball_detection.model_path == default.ball_detection.model_path
        assert cfg.detection.model_path == default.detection.model_path
        assert cfg.detection.class_conf_overrides == default.detection.class_conf_overrides
        # and it still does its own job
        assert cfg.visualization.showcase.enabled
        assert not cfg.visualization.write_annotated_video
