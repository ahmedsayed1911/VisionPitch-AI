"""Integration tests.

Split by requirement so the cheap ones always run:

* ingestion and game-state assembly need only a synthetic video
* the full pipeline needs the checkpoints and a real clip
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from visionpitch.common.config import AnalysisMode, load_config
from visionpitch.common.types import (
    BallState,
    BBox,
    CalibrationResult,
    ObjectClass,
    Role,
    TeamId,
    Track,
    TrackObservation,
    ValidationStatus,
)
from visionpitch.game_state.assembler import GameStateAssembler
from visionpitch.ingestion.video import VideoReader, probe_video


class TestIngestion:
    def test_probes_a_real_file(self, synthetic_video: Path) -> None:
        meta = probe_video(synthetic_video)
        assert meta.width == 320 and meta.height == 240
        assert meta.fps == pytest.approx(25.0, abs=0.01)
        assert meta.frame_count == 50
        assert meta.video_id.startswith("synthetic_")

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            probe_video("does_not_exist.mp4")

    def test_reads_every_frame(self, synthetic_video: Path, repo_root: Path) -> None:
        config = load_config(config_root=repo_root)
        meta = probe_video(synthetic_video)
        with VideoReader(meta, config.ingestion) as reader:
            frames = list(reader)
        assert len(frames) == 50
        assert [f.idx for f in frames] == list(range(50))

    def test_stride_preserves_absolute_frame_indices(
        self, synthetic_video: Path, repo_root: Path
    ) -> None:
        """The 10th processed frame at stride 3 is source frame 27, and its
        timestamp must reflect that -- otherwise every event shifts in time."""
        config = load_config(config_root=repo_root, cli_sets=["ingestion.frame_stride=3"])
        meta = probe_video(synthetic_video)
        with VideoReader(meta, config.ingestion) as reader:
            frames = list(reader)
        assert [f.idx for f in frames[:4]] == [0, 3, 6, 9]
        assert frames[9].idx == 27
        assert frames[9].timestamp_s == pytest.approx(27 / meta.fps)

    def test_time_range_selection(self, synthetic_video: Path, repo_root: Path) -> None:
        config = load_config(
            config_root=repo_root,
            cli_sets=["ingestion.start_time_s=0.4", "ingestion.end_time_s=1.0"],
        )
        meta = probe_video(synthetic_video)
        with VideoReader(meta, config.ingestion) as reader:
            frames = list(reader)
        assert frames[0].idx == 10
        assert frames[-1].idx < 25

    def test_max_frames_caps_output(self, synthetic_video: Path, repo_root: Path) -> None:
        config = load_config(config_root=repo_root, cli_sets=["ingestion.max_frames=7"])
        meta = probe_video(synthetic_video)
        with VideoReader(meta, config.ingestion) as reader:
            assert len(list(reader)) == 7

    def test_resume_snaps_onto_the_sampling_lattice(
        self, synthetic_video: Path, repo_root: Path
    ) -> None:
        config = load_config(config_root=repo_root, cli_sets=["ingestion.frame_stride=4"])
        meta = probe_video(synthetic_video)
        with VideoReader(meta, config.ingestion, resume_from_frame=13) as reader:
            frames = list(reader)
        # 13 is not on the stride-4 lattice; the next lattice point is 16.
        assert frames[0].idx == 16
        assert all(f.idx % 4 == 0 for f in frames)


class TestGameStateAssembly:
    def _inputs(self, homography: np.ndarray | None):
        track = Track(
            track_id=1,
            object_class=ObjectClass.PLAYER,
            observations=[
                TrackObservation(f, f / 25.0, BBox(600, 400, 640, 500), 0.9, 0.8, False)
                for f in range(3)
            ],
            team_id=TeamId.A,
            team_confidence=0.8,
            role=Role.OUTFIELD,
        )
        calibration = {
            f: CalibrationResult(f, homography, 0.7 if homography is not None else 0.0,
                                 0.3, 10, 9)
            for f in range(3)
        }
        ball = {
            f: BallState(f, f / 25.0, (640.0, 450.0), BBox(635, 445, 645, 455),
                         None, 0.6, True, False)
            for f in range(3)
        }
        return {1: track}, ball, calibration, {f: f / 25.0 for f in range(3)}, [0, 1, 2]

    def test_projects_players_when_calibrated(
        self, pitch, synthetic_homography: np.ndarray
    ) -> None:
        assembler = GameStateAssembler("clip", pitch, 0.4)
        rows = assembler.assemble(*self._inputs(synthetic_homography))
        players = [r for r in rows if r.object_class == "player"]
        assert len(players) == 3
        for row in players:
            assert row.pitch_x is not None and row.pitch_y is not None
            assert pitch.contains(row.pitch_x, row.pitch_y, margin=25.0)
            assert row.validation_status == ValidationStatus.VALID.value

    def test_nulls_pitch_coordinates_when_uncalibrated(self, pitch) -> None:
        """The player was really there, so the row stays -- but the position
        must be null, not fabricated from a guessed homography."""
        assembler = GameStateAssembler("clip", pitch, 0.4)
        rows = assembler.assemble(*self._inputs(None))
        players = [r for r in rows if r.object_class == "player"]
        assert len(players) == 3
        for row in players:
            assert row.pitch_x is None and row.pitch_y is None
            assert row.image_x > 0  # image-space evidence is retained
            assert row.validation_status == ValidationStatus.NO_CALIBRATION.value

    def test_projects_from_ground_contact_not_box_centre(
        self, pitch, synthetic_homography: np.ndarray
    ) -> None:
        assembler = GameStateAssembler("clip", pitch, 0.4)
        rows = assembler.assemble(*self._inputs(synthetic_homography))
        player = next(r for r in rows if r.object_class == "player")
        assert player.image_y == pytest.approx(500.0)  # bbox bottom
        assert player.image_x == pytest.approx(620.0)  # bbox centre-x

    def test_low_confidence_calibration_is_flagged(
        self, pitch, synthetic_homography: np.ndarray
    ) -> None:
        tracks, ball, _, timestamps, frames = self._inputs(synthetic_homography)
        weak = {f: CalibrationResult(f, synthetic_homography, 0.1, 0.3, 6, 5) for f in frames}
        rows = GameStateAssembler("clip", pitch, 0.4).assemble(
            tracks, ball, weak, timestamps, frames
        )
        player = next(r for r in rows if r.object_class == "player")
        assert player.validation_status == ValidationStatus.LOW_CALIBRATION.value
        assert player.pitch_x is not None  # still usable, just flagged

    def test_quality_report_is_self_consistent(
        self, pitch, synthetic_homography: np.ndarray
    ) -> None:
        rows = GameStateAssembler("clip", pitch, 0.4).assemble(
            *self._inputs(synthetic_homography)
        )
        report = GameStateAssembler.quality_report(rows)
        assert report["rows"] == len(rows)
        assert report["person_rows"] + report["ball_rows"] == report["rows"]
        assert 0.0 <= report["pitch_coordinate_ratio"] <= 1.0


@pytest.mark.slow
@pytest.mark.needs_models
@pytest.mark.needs_clip
class TestFullPipeline:
    """End-to-end on a real clip. Skipped when models or clip are absent."""

    def test_produces_a_complete_run(
        self, validation_clip, models_available, repo_root: Path, tmp_path: Path
    ) -> None:
        if validation_clip is None:
            pytest.skip("validation clip not downloaded")
        if not models_available:
            pytest.skip("model checkpoints not downloaded")

        from visionpitch.pipeline.runner import Phase1Pipeline

        config = load_config(
            config_root=repo_root,
            mode=AnalysisMode.BALANCED,
            overrides={
                "ingestion": {"max_frames": 60},
                "storage": {"output_dir": str(tmp_path)},
                "visualization": {
                    "write_annotated_video": True,
                    "write_tactical_map": True,
                    "write_combined_video": False,
                },
            },
        )
        result = Phase1Pipeline(config).run(validation_clip)

        # -- the artefacts the phase promises ---------------------------------
        for key in ("game_state", "tracks", "calibration", "detections", "summary"):
            assert key in result.outputs, f"missing output: {key}"
            assert Path(result.outputs[key]).exists()
        assert Path(result.outputs["video_annotated"]).exists()
        assert Path(result.outputs["video_radar"]).exists()

        assert len(result.frame_indices) == 60
        assert result.rows, "no game-state rows produced"
        assert result.tracks, "no tracks produced"

        # -- provenance --------------------------------------------------------
        manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["config_fingerprint"] == config.fingerprint()
        assert manifest["models"]["detector"]["weights_sha256"]
        assert "data_quality" in manifest

        # -- every row must be internally consistent ---------------------------
        for row in result.rows:
            assert row.frame_idx in result.timestamps
            assert row.timestamp_s == pytest.approx(result.timestamps[row.frame_idx])
            assert 0.0 <= row.detection_confidence <= 1.0
            assert 0.0 <= row.calibration_confidence <= 1.0
            assert row.validation_status in {v.value for v in ValidationStatus}
            if row.pitch_x is None:
                assert row.pitch_y is None

    def test_chunked_matches_single_pass(
        self, validation_clip, models_available, repo_root: Path, tmp_path: Path
    ) -> None:
        """Chunked processing must cover the same frames without duplicating them.

        Track *counts* are allowed to differ: a seam can fragment an identity
        that a single pass kept whole. What may never differ is the frame
        coverage or the presence of duplicate rows, because those are
        correctness properties of the merge rather than quality properties of
        the tracker.
        """
        if validation_clip is None or not models_available:
            pytest.skip("models or clip unavailable")

        from visionpitch.pipeline.chunked_runner import ChunkedPipeline
        from visionpitch.pipeline.runner import Phase1Pipeline

        common = {
            "ingestion": {"end_time_s": 8.0},
            "visualization": {
                "write_annotated_video": False,
                "write_tactical_map": False,
                "write_combined_video": False,
            },
        }

        single_config = load_config(
            config_root=repo_root,
            overrides={**common, "storage": {"output_dir": str(tmp_path / "single")}},
        )
        single = Phase1Pipeline(single_config).run(validation_clip)

        chunked_config = load_config(
            config_root=repo_root,
            overrides={
                **common,
                "storage": {"output_dir": str(tmp_path / "chunked")},
                "chunking": {"enabled": True, "chunk_frames": 90, "overlap_frames": 30},
            },
        )
        chunked = ChunkedPipeline(chunked_config).run(validation_clip)

        assert len(chunked.chunks) >= 2, "the test must actually exercise a seam"

        single_frames = set(single.frame_indices)
        chunked_gs = pd.read_parquet(Path(chunked.outputs["game_state"]))
        chunked_frames_table = pd.read_parquet(Path(chunked.outputs["frames"]))

        # -- every processed frame is accounted for exactly once -------------- #
        listed = chunked_frames_table.frame_idx.tolist()
        assert len(listed) == len(set(listed)), "a frame is listed twice"
        assert set(listed) == single_frames, "chunked coverage differs from single pass"

        # -- no duplicated object rows at the seams --------------------------- #
        person = chunked_gs[chunked_gs.track_id.notna()]
        duplicates = person.duplicated(subset=["frame_idx", "track_id"]).sum()
        assert duplicates == 0, f"{duplicates} duplicate (frame, track) rows"

        ball = chunked_gs[chunked_gs.object_class == "ball"]
        assert ball.frame_idx.duplicated().sum() == 0, "duplicate ball rows"

        # -- the merge produced a usable amount of data ----------------------- #
        assert chunked.tracks > 0
        assert len(chunked_gs) > 0.7 * len(single.rows), (
            f"chunked produced {len(chunked_gs)} rows vs single-pass "
            f"{len(single.rows)}; the merge is losing data"
        )

    def test_chunked_run_resumes(
        self, validation_clip, models_available, repo_root: Path, tmp_path: Path
    ) -> None:
        """A completed chunk is not reprocessed on a second invocation."""
        if validation_clip is None or not models_available:
            pytest.skip("models or clip unavailable")

        from visionpitch.pipeline.chunked_runner import ChunkedPipeline

        config = load_config(
            config_root=repo_root,
            overrides={
                "ingestion": {"end_time_s": 6.0},
                "storage": {"output_dir": str(tmp_path)},
                "chunking": {"enabled": True, "chunk_frames": 60, "overlap_frames": 20},
                "visualization": {
                    "write_annotated_video": False,
                    "write_tactical_map": False,
                    "write_combined_video": False,
                },
            },
        )
        first = ChunkedPipeline(config).run(validation_clip)
        state = first.run_dir / "checkpoints" / "chunked_state.json"
        assert state.exists()
        recorded = json.loads(state.read_text(encoding="utf-8"))["completed_chunks"]
        assert len(recorded) == len(first.chunks)

        second = ChunkedPipeline(config).run(validation_clip, resume=True)
        assert second.timings["total"] < first.timings["total"], (
            "resume did not skip completed chunks"
        )

    def test_rerun_is_reproducible(
        self, validation_clip, models_available, repo_root: Path, tmp_path: Path
    ) -> None:
        if validation_clip is None or not models_available:
            pytest.skip("models or clip unavailable")

        from visionpitch.pipeline.runner import Phase1Pipeline

        config = load_config(
            config_root=repo_root,
            overrides={
                "ingestion": {"max_frames": 25},
                "storage": {"output_dir": str(tmp_path)},
                "visualization": {
                    "write_annotated_video": False,
                    "write_tactical_map": False,
                    "write_combined_video": False,
                },
            },
        )
        first = Phase1Pipeline(config).run(validation_clip)
        second = Phase1Pipeline(config).run(validation_clip)

        assert first.run_dir == second.run_dir, "same config must map to the same run"
        assert len(first.rows) == len(second.rows)
        assert len(first.tracks) == len(second.tracks)
