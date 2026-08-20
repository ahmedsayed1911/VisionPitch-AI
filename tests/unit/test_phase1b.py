"""Phase 1B: chunking, global association, calibration propagation, integrity."""

from __future__ import annotations

import numpy as np
import pytest

from visionpitch.calibration.propagation import (
    affine_to_homography,
    extrapolation_risk,
    propagate_calibration,
    support_region,
)
from visionpitch.common.config import Config, TrackingConfig
from visionpitch.common.geometry import apply_homography
from visionpitch.common.types import (
    BBox,
    CalibrationResult,
    ObjectClass,
    Track,
    TrackObservation,
    ValidationStatus,
)
from visionpitch.pipeline.chunking import (
    link_identities,
    merge_tracks,
    plan_chunks,
    trim_tracks_to_owned,
)
from visionpitch.tracking.association import associate_tracklets

IMAGE_SIZE = (1280, 720)


def track(track_id: int, frames, x0=100.0, step=4.0, y=300.0, h=90.0) -> Track:
    return Track(
        track_id=track_id,
        object_class=ObjectClass.PLAYER,
        observations=[
            TrackObservation(
                f, f / 25.0,
                BBox(x0 + step * f, y, x0 + step * f + 40, y + h),
                0.9, 0.9, False,
            )
            for f in frames
        ],
    )


# --------------------------------------------------------------------------- #
# Chunk planning
# --------------------------------------------------------------------------- #


class TestPlanChunks:
    def test_ownership_tiles_the_range_exactly_once(self) -> None:
        """The merge depends on this: every frame owned by exactly one chunk."""
        chunks = plan_chunks(0, 1000, chunk_frames=300, overlap_frames=50)
        owned = []
        for chunk in chunks:
            owned.extend(range(chunk.owned_start, chunk.owned_end))
        assert owned == list(range(1000))
        assert len(owned) == len(set(owned)), "a frame is owned twice"

    def test_chunks_overlap_for_warmup(self) -> None:
        chunks = plan_chunks(0, 1000, chunk_frames=300, overlap_frames=50)
        assert chunks[0].warmup_frames == 0  # nothing precedes the first chunk
        for chunk in chunks[1:]:
            assert chunk.warmup_frames == 50
            assert chunk.start_frame < chunk.owned_start

    def test_handles_a_range_shorter_than_one_chunk(self) -> None:
        chunks = plan_chunks(0, 120, chunk_frames=500, overlap_frames=50)
        assert len(chunks) == 1
        assert chunks[0].owned_start == 0 and chunks[0].owned_end == 120

    def test_respects_a_non_zero_start(self) -> None:
        chunks = plan_chunks(500, 900, chunk_frames=200, overlap_frames=40)
        assert chunks[0].owned_start == 500
        assert chunks[0].start_frame == 500  # cannot read before the range
        assert chunks[-1].owned_end == 900

    def test_rejects_overlap_that_would_stall(self) -> None:
        with pytest.raises(ValueError, match="smaller than"):
            plan_chunks(0, 1000, chunk_frames=100, overlap_frames=100)

    def test_rejects_non_positive_chunk(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            plan_chunks(0, 100, chunk_frames=0, overlap_frames=0)

    def test_owns_predicate_matches_the_range(self) -> None:
        chunk = plan_chunks(0, 600, chunk_frames=300, overlap_frames=30)[1]
        assert not chunk.owns(chunk.owned_start - 1)
        assert chunk.owns(chunk.owned_start)
        assert chunk.owns(chunk.owned_end - 1)
        assert not chunk.owns(chunk.owned_end)


# --------------------------------------------------------------------------- #
# Cross-chunk identity
# --------------------------------------------------------------------------- #


class TestLinkIdentities:
    def test_links_the_same_player_across_a_seam(self) -> None:
        earlier = {1: track(1, range(80, 120))}
        later = {7: track(7, range(100, 140))}  # same trajectory, new id
        mapping = link_identities(earlier, later, overlap_lo=100, overlap_hi=120)
        assert mapping == {7: 1}

    def test_does_not_link_two_different_players(self) -> None:
        earlier = {1: track(1, range(80, 120), x0=100.0)}
        later = {7: track(7, range(100, 140), x0=900.0)}
        assert link_identities(earlier, later, 100, 120) == {}

    def test_requires_several_shared_frames(self) -> None:
        """One coincidental overlap must not authorise an identity merge."""
        earlier = {1: track(1, range(80, 101))}
        later = {7: track(7, range(100, 140))}
        assert link_identities(earlier, later, 100, 120, min_shared_frames=5) == {}

    def test_no_overlap_links_nothing(self) -> None:
        earlier = {1: track(1, range(0, 50))}
        later = {7: track(7, range(60, 100))}
        assert link_identities(earlier, later, 60, 60) == {}

    def test_assignment_is_one_to_one(self) -> None:
        earlier = {1: track(1, range(80, 120), x0=100.0),
                   2: track(2, range(80, 120), x0=500.0)}
        later = {7: track(7, range(100, 140), x0=100.0),
                 8: track(8, range(100, 140), x0=500.0)}
        mapping = link_identities(earlier, later, 100, 120)
        assert mapping == {7: 1, 8: 2}
        assert len(set(mapping.values())) == len(mapping)


class TestMerge:
    def test_overlap_contributes_only_once(self) -> None:
        """A frame processed by two chunks must not appear twice."""
        accumulated = {1: track(1, range(0, 120))}
        incoming = {7: track(7, range(100, 140))}
        merged, _ = merge_tracks(accumulated, incoming, {7: 1}, id_offset=100)

        frames = [o.frame_idx for o in merged[1].observations]
        assert len(frames) == len(set(frames)), "duplicate frames after merge"
        assert frames == sorted(frames)
        assert max(frames) == 139

    def test_unmatched_tracks_get_globally_unique_ids(self) -> None:
        accumulated = {1: track(1, range(0, 50))}
        incoming = {1: track(1, range(60, 100))}  # id collision, different object
        merged, next_id = merge_tracks(accumulated, incoming, {}, id_offset=500)
        assert set(merged) == {1, 501}
        assert next_id == 501

    def test_trim_drops_observations_outside_ownership(self) -> None:
        tracks = {1: track(1, range(0, 200))}
        trimmed = trim_tracks_to_owned(tracks, 50, 150)
        frames = [o.frame_idx for o in trimmed[1].observations]
        assert min(frames) == 50 and max(frames) == 149

    def test_trim_removes_tracks_left_empty(self) -> None:
        tracks = {1: track(1, range(0, 40))}
        assert trim_tracks_to_owned(tracks, 100, 200) == {}


# --------------------------------------------------------------------------- #
# Global association
# --------------------------------------------------------------------------- #


class TestGlobalAssociation:
    def _config(self, **overrides) -> TrackingConfig:
        base = {"association": "global", "assoc_max_gap_frames": 60,
                "assoc_max_image_only_gap_frames": 60}
        base.update(overrides)
        return TrackingConfig(**base)

    def test_rejoins_a_fragmented_track(self) -> None:
        tracks = {1: track(1, range(0, 30)), 2: track(2, range(40, 80))}
        merged, id_map, report = associate_tracklets(tracks, self._config())
        assert len(merged) == 1
        assert id_map == {2: 1}
        assert report.merges == 1

    def test_refuses_a_teleport(self) -> None:
        tracks = {
            1: track(1, range(0, 30), x0=100.0, step=0.0),
            2: track(2, range(40, 80), x0=1200.0, step=0.0),
        }
        merged, _, report = associate_tracklets(tracks, self._config())
        assert len(merged) == 2
        assert report.refusals["image_distance_exceeded"] > 0

    def test_refuses_temporally_overlapping_tracks(self) -> None:
        """Two tracks alive at the same time are two different players."""
        tracks = {1: track(1, range(0, 50)), 2: track(2, range(20, 70))}
        merged, _, report = associate_tracklets(tracks, self._config())
        assert len(merged) == 2
        assert report.refusals["temporal_overlap"] > 0

    def test_refuses_inconsistent_size(self) -> None:
        tracks = {
            1: track(1, range(0, 30), h=90.0, step=0.0),
            2: track(2, range(35, 70), h=300.0, step=0.0),
        }
        merged, _, report = associate_tracklets(tracks, self._config())
        assert len(merged) == 2
        assert report.refusals["size_inconsistent"] > 0

    def test_refuses_a_class_mismatch(self) -> None:
        a = track(1, range(0, 30))
        b = track(2, range(35, 70))
        b.object_class = ObjectClass.REFEREE
        merged, _, report = associate_tracklets({1: a, 2: b}, self._config())
        assert len(merged) == 2
        assert report.refusals["class_mismatch"] > 0

    def test_long_gaps_need_pitch_evidence(self) -> None:
        """The safety rule: a long gap may not be bridged on pixels alone."""
        config = self._config(assoc_max_image_only_gap_frames=10, assoc_max_gap_frames=60)
        tracks = {1: track(1, range(0, 30)), 2: track(2, range(70, 110))}
        merged, _, report = associate_tracklets(tracks, config)
        assert len(merged) == 2
        assert report.refusals["long_gap_without_pitch_evidence"] > 0

    def test_collapses_a_chain_of_three(self) -> None:
        tracks = {
            1: track(1, range(0, 25)),
            2: track(2, range(30, 55)),
            3: track(3, range(60, 90)),
        }
        merged, id_map, _ = associate_tracklets(tracks, self._config())
        assert len(merged) == 1
        # Every absorbed id must resolve to the surviving id, not to a
        # intermediate that no longer exists.
        assert set(id_map.values()) == set(merged)

    def test_greedy_and_global_are_both_selectable(self) -> None:
        from visionpitch.tracking.postprocess import clean_tracks

        tracks = {1: track(1, range(0, 30)), 2: track(2, range(40, 80))}
        for strategy in ("greedy", "global", "none"):
            config = TrackingConfig(association=strategy, min_track_length=1)
            cleaned, _, report = clean_tracks(dict(tracks), config)
            assert report["association"] == strategy
            assert cleaned


# --------------------------------------------------------------------------- #
# Calibration propagation
# --------------------------------------------------------------------------- #


class TestPropagation:
    def test_affine_lift_is_correct(self) -> None:
        warp = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
        H = affine_to_homography(warp)
        moved = apply_homography(H, np.array([[10.0, 10.0]]))[0]
        assert np.allclose(moved, [15.0, 7.0])

    def test_fills_a_gap_between_anchors(
        self, synthetic_homography: np.ndarray
    ) -> None:
        frames = list(range(20))
        results = {
            f: CalibrationResult(f, synthetic_homography.copy(), 0.8, 0.2, 10, 9)
            for f in (0, 1, 2, 17, 18, 19)
        }
        for f in frames:
            results.setdefault(f, CalibrationResult(f, None, 0.0, float("nan"), 0, 0))

        # A static camera: identity warps.
        warps = {f: np.eye(2, 3) for f in frames}
        filled, report = propagate_calibration(results, frames, warps)

        assert report["propagated"] > 0
        assert all(filled[f].is_valid for f in frames)
        # Propagated frames must be flagged, never presented as solves.
        assert filled[10].smoothed is True
        assert filled[10].n_inliers == 0

    def test_never_overwrites_a_real_solve(
        self, synthetic_homography: np.ndarray
    ) -> None:
        frames = list(range(10))
        solved = CalibrationResult(5, synthetic_homography.copy(), 0.9, 0.1, 12, 11)
        results = {
            0: CalibrationResult(0, synthetic_homography.copy(), 0.8, 0.2, 10, 9),
            5: solved,
        }
        for f in frames:
            results.setdefault(f, CalibrationResult(f, None, 0.0, float("nan"), 0, 0))

        filled, _ = propagate_calibration(
            results, frames, {f: np.eye(2, 3) for f in frames}
        )
        assert filled[5] is solved

    def test_confidence_decays_with_distance(
        self, synthetic_homography: np.ndarray
    ) -> None:
        frames = list(range(30))
        results = {0: CalibrationResult(0, synthetic_homography.copy(), 0.9, 0.1, 12, 11)}
        for f in frames:
            results.setdefault(f, CalibrationResult(f, None, float("nan"), 0.0, 0, 0))

        filled, _ = propagate_calibration(
            results, frames, {f: np.eye(2, 3) for f in frames}
        )
        near = filled[3].confidence
        far = filled[20].confidence
        assert near > far, "confidence must decay as error compounds"

    def test_stops_before_inventing_a_distant_pose(
        self, synthetic_homography: np.ndarray
    ) -> None:
        frames = list(range(200))
        results = {0: CalibrationResult(0, synthetic_homography.copy(), 0.9, 0.1, 12, 11)}
        for f in frames:
            results.setdefault(f, CalibrationResult(f, None, 0.0, float("nan"), 0, 0))

        filled, report = propagate_calibration(
            results, frames, {f: np.eye(2, 3) for f in frames},
            max_propagation_frames=20,
        )
        assert not filled[150].is_valid, "propagation must be bounded"
        assert report["still_unsolved"] > 0

    def test_tracks_a_panning_camera(self, synthetic_homography: np.ndarray) -> None:
        """A translating camera must be followed, not ignored."""
        frames = list(range(12))
        results = {0: CalibrationResult(0, synthetic_homography.copy(), 0.9, 0.1, 12, 11)}
        for f in frames:
            results.setdefault(f, CalibrationResult(f, None, 0.0, float("nan"), 0, 0))

        shift = 4.0
        warps = {f: np.array([[1.0, 0.0, shift], [0.0, 1.0, 0.0]]) for f in frames}
        filled, _ = propagate_calibration(results, frames, warps)

        # A pitch point seen at image x under H0 appears at x+shift a frame
        # later, so the propagated homography must map the shifted pixel to the
        # same pitch coordinate.
        probe = np.array([[640.0, 600.0]])
        original = apply_homography(synthetic_homography, probe)[0]
        moved = apply_homography(filled[1].homography, probe + np.array([shift, 0.0]))[0]
        assert np.allclose(original, moved, atol=0.2)


class TestExtrapolationRisk:
    def test_zero_inside_the_supported_region(self) -> None:
        region = (100.0, 100.0, 500.0, 400.0)
        assert extrapolation_risk((300.0, 250.0), region) == 0.0

    def test_grows_outside(self) -> None:
        region = (100.0, 100.0, 500.0, 400.0)
        near = extrapolation_risk((520.0, 250.0), region)
        far = extrapolation_risk((900.0, 250.0), region)
        assert 0.0 < near < far

    def test_unknown_region_does_not_imply_extrapolation(self) -> None:
        """Absence of evidence about the support is not evidence of extrapolation.

        Returning a middling value here would mark every row on every frame
        whose support was not recorded as extrapolated, including all
        motion-propagated frames.
        """
        assert extrapolation_risk((0.0, 0.0), None) == 0.0

    def test_support_region_covers_the_keypoints(self) -> None:
        points = np.array([[400.0, 500.0], [800.0, 520.0], [600.0, 620.0]])
        region = support_region(points, IMAGE_SIZE)
        assert region is not None
        x1, y1, x2, y2 = region
        assert x1 <= 400.0 and x2 >= 800.0
        assert y1 <= 500.0 and y2 >= 620.0

    def test_support_region_needs_enough_points(self) -> None:
        assert support_region(np.array([[1.0, 2.0]]), IMAGE_SIZE) is None
        assert support_region(None, IMAGE_SIZE) is None


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #


class TestFrameTable:
    def test_records_every_processed_frame_including_empty_ones(
        self, tmp_path, pitch
    ) -> None:
        """The gap this closes: an empty frame and an unprocessed frame used to
        be indistinguishable, because neither produced any row."""
        import pyarrow.parquet as pq

        from visionpitch.game_state.assembler import GameStateAssembler
        from visionpitch.storage.tables import frame_rows, write_frames

        frames = [0, 1, 2, 3, 4]
        timestamps = {f: f / 25.0 for f in frames}
        tracks = {1: track(1, [0, 1])}  # frames 2-4 detect nothing
        calibration = {
            f: CalibrationResult(f, np.eye(3), 0.7, 0.2, 8, 8) for f in frames
        }

        rows = GameStateAssembler("clip", pitch, 0.4).assemble(
            tracks, {}, calibration, timestamps, frames
        )
        assert {r.frame_idx for r in rows} == {0, 1}

        path = write_frames(
            frame_rows("clip", frames, timestamps, rows, calibration),
            tmp_path / "frames.parquet",
        )
        table = pq.read_table(path).to_pydict()
        assert table["frame_idx"] == frames, "every processed frame must appear"
        assert table["n_persons"] == [1, 1, 0, 0, 0]
        assert all(table["calibration_valid"])

    def test_extrapolated_rows_are_flagged_not_dropped(self, pitch) -> None:
        from visionpitch.game_state.assembler import GameStateAssembler

        frames = [0]
        # Landmarks constrained the lower-right of the frame only.
        support = {0: (600.0, 500.0, 1200.0, 700.0)}
        far_side = Track(
            track_id=1,
            object_class=ObjectClass.PLAYER,
            observations=[
                TrackObservation(0, 0.0, BBox(60, 200, 100, 290), 0.9, 0.9, False)
            ],
        )
        import cv2

        world = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
        image = np.array([[150.0, 700.0], [1150.0, 700.0], [900.0, 250.0], [400.0, 250.0]])
        H, _ = cv2.findHomography(image, world, method=0)

        assembler = GameStateAssembler(
            "clip", pitch, 0.4, support_regions=support, max_extrapolation_risk=0.35
        )
        rows = assembler.assemble(
            {1: far_side},
            {},
            {0: CalibrationResult(0, H, 0.8, 0.2, 8, 8)},
            {0: 0.0},
            frames,
        )
        assert len(rows) == 1
        assert rows[0].validation_status == ValidationStatus.EXTRAPOLATED.value
        assert rows[0].pitch_x is not None, "the row must be kept, only flagged"


def test_chunking_config_defaults_are_sane(config: Config) -> None:
    assert config.chunking.overlap_frames < config.chunking.chunk_frames
    assert config.chunking.enabled is False  # opt-in
