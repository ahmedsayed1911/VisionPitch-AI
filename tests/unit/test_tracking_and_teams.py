"""Tracker association, offline cleaning, and team discovery."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from visionpitch.common.config import Config, TrackingConfig
from visionpitch.common.types import BBox, Detection, ObjectClass, Track, TrackObservation
from visionpitch.tracking.appearance import (
    ExponentialFeatureBank,
    TorsoHistogramAppearance,
    cosine_distance,
)
from visionpitch.tracking.gmc import GlobalMotionCompensator
from visionpitch.tracking.kalman import state_to_xyxy, xyxy_to_state
from visionpitch.tracking.postprocess import clean_tracks, drop_short_tracks, stitch_tracks
from visionpitch.tracking.tracker import MultiObjectTracker


def person(frame: int, x: float, y: float, conf: float = 0.9) -> Detection:
    return Detection(frame, ObjectClass.PLAYER, BBox(x, y, x + 40, y + 90), conf)


def make_track(track_id: int, frames: range, x0: float, step: float = 4.0) -> Track:
    return Track(
        track_id=track_id,
        object_class=ObjectClass.PLAYER,
        observations=[
            TrackObservation(
                f, f / 25.0, BBox(x0 + step * f, 300, x0 + step * f + 40, 390), 0.9, 0.9, False
            )
            for f in frames
        ],
    )


class TestKalmanStateConversion:
    def test_round_trips(self) -> None:
        box = np.array([100.0, 200.0, 140.0, 290.0])
        assert np.allclose(state_to_xyxy(xyxy_to_state(box)), box)

    def test_parameterises_by_aspect_and_height(self) -> None:
        state = xyxy_to_state(np.array([0.0, 0.0, 40.0, 80.0]))
        assert state[2] == pytest.approx(0.5)  # aspect
        assert state[3] == pytest.approx(80.0)  # height


class TestTracker:
    def test_maintains_one_id_for_a_moving_player(self, config: Config) -> None:
        tracker = MultiObjectTracker(config)
        image = np.zeros((720, 1280, 3), np.uint8)
        for f in range(30):
            tracker.update([person(f, 100 + 8 * f, 300)], f, f / 25.0, image)
        tracks = tracker.finalise()
        assert len(tracks) == 1
        assert next(iter(tracks.values())).length >= 25

    def test_keeps_two_players_apart(self, config: Config) -> None:
        tracker = MultiObjectTracker(config)
        image = np.zeros((720, 1280, 3), np.uint8)
        for f in range(30):
            tracker.update(
                [person(f, 100 + 5 * f, 300), person(f, 700 - 5 * f, 320)], f, f / 25.0, image
            )
        assert len(tracker.finalise()) == 2

    def test_survives_a_brief_occlusion(self, config: Config) -> None:
        tracker = MultiObjectTracker(config)
        image = np.zeros((720, 1280, 3), np.uint8)
        for f in range(40):
            detections = [] if 15 <= f < 20 else [person(f, 100 + 8 * f, 300)]
            tracker.update(detections, f, f / 25.0, image)
        tracks = tracker.finalise()
        assert len(tracks) == 1, "a 5-frame occlusion must not create a new identity"

    def test_low_confidence_boxes_do_not_start_tracks(self, config: Config) -> None:
        tracker = MultiObjectTracker(config)
        image = np.zeros((720, 1280, 3), np.uint8)
        for f in range(20):
            tracker.update([person(f, 100 + 5 * f, 300, conf=0.2)], f, f / 25.0, image)
        assert tracker.finalise() == {}

    def test_birth_indices_are_relative_to_high_confidence_subset(self, config: Config) -> None:
        """Interleaved low boxes must not shift indices used for new tracks."""
        tracker = MultiObjectTracker(config)
        active = tracker.update(
            [
                person(0, 900, 300, conf=0.2),
                person(0, 100, 300, conf=0.9),
                person(0, 1050, 300, conf=0.2),
                person(0, 500, 300, conf=0.9),
            ],
            0,
            0.0,
        )

        assert np.allclose(sorted(track.bbox_array[0] for track in active), [100.0, 500.0])

    def test_trailing_predictions_are_trimmed(self, config: Config) -> None:
        """A track must not report confident positions after its last sighting."""
        tracker = MultiObjectTracker(config)
        image = np.zeros((720, 1280, 3), np.uint8)
        for f in range(20):
            tracker.update([person(f, 100 + 5 * f, 300)], f, f / 25.0, image)
        for f in range(20, 40):
            tracker.update([], f, f / 25.0, image)

        tracks = tracker.finalise()
        for track in tracks.values():
            assert track.observations[-1].interpolated is False
            assert track.last_frame <= 20


class TestPostprocess:
    def test_stitches_a_fragmented_track(self) -> None:
        tracks = {
            1: make_track(1, range(0, 20), 100.0),
            2: make_track(2, range(28, 50), 100.0),
        }
        cfg = TrackingConfig(stitch_max_gap_frames=30, stitch_max_distance_px=200.0)
        stitched, id_map = stitch_tracks(tracks, cfg)
        assert len(stitched) == 1
        assert id_map == {2: 1}

    def test_does_not_stitch_across_a_long_gap(self) -> None:
        tracks = {
            1: make_track(1, range(0, 20), 100.0),
            2: make_track(2, range(200, 220), 100.0),
        }
        stitched, id_map = stitch_tracks(
            tracks, TrackingConfig(stitch_max_gap_frames=30)
        )
        assert len(stitched) == 2
        assert id_map == {}

    def test_does_not_stitch_across_a_teleport(self) -> None:
        tracks = {
            1: make_track(1, range(0, 20), 100.0, step=0.0),
            2: make_track(2, range(25, 45), 1200.0, step=0.0),
        }
        stitched, _ = stitch_tracks(
            tracks, TrackingConfig(stitch_max_gap_frames=30, stitch_max_distance_px=120.0)
        )
        assert len(stitched) == 2

    def test_drops_short_tracks(self) -> None:
        tracks = {1: make_track(1, range(0, 30), 100.0), 2: make_track(2, range(0, 2), 500.0)}
        kept, dropped = drop_short_tracks(tracks, min_length=5)
        assert dropped == 1 and set(kept) == {1}

    def test_id_map_never_points_at_a_dropped_track(self) -> None:
        """A crop keyed to an absorbed id must not follow it into the void."""
        tracks = {
            1: make_track(1, range(0, 3), 100.0),
            2: make_track(2, range(5, 8), 100.0),
        }
        cfg = TrackingConfig(min_track_length=50, stitch_max_gap_frames=30,
                             stitch_max_distance_px=400.0)
        cleaned, id_map, _ = clean_tracks(tracks, cfg)
        assert cleaned == {}
        assert all(dst in cleaned for dst in id_map.values())

    def test_report_totals_are_consistent(self) -> None:
        tracks = {1: make_track(1, range(0, 30), 100.0), 2: make_track(2, range(0, 2), 900.0)}
        cleaned, _, report = clean_tracks(tracks, TrackingConfig())
        assert report["tracks_in"] == 2
        assert report["tracks_out"] == len(cleaned)


class TestGlobalMotionCompensation:
    def _textured(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (rng.random((360, 640, 3)) * 255).astype(np.uint8)

    def test_first_frame_returns_identity(self) -> None:
        gmc = GlobalMotionCompensator(downscale=1)
        warp = gmc.apply(self._textured())
        assert np.allclose(warp, np.eye(2, 3))

    def test_static_scene_gives_near_identity(self) -> None:
        image = self._textured()
        gmc = GlobalMotionCompensator(downscale=1)
        gmc.apply(image)
        warp = gmc.apply(image.copy())
        assert np.allclose(warp[:2, :2], np.eye(2), atol=0.05)
        assert np.abs(warp[:2, 2]).max() < 2.0

    def test_recovers_a_known_translation(self) -> None:
        image = self._textured(7)
        shift = 12
        shifted = np.roll(image, shift, axis=1)

        gmc = GlobalMotionCompensator(downscale=1)
        gmc.apply(image)
        warp = gmc.apply(shifted)
        assert warp[0, 2] == pytest.approx(shift, abs=3.0)

    def test_detection_mask_excludes_boxes(self) -> None:
        mask = GlobalMotionCompensator._detection_mask(
            (100, 200), np.array([[50.0, 20.0, 90.0, 60.0]]), scale=1
        )
        assert mask[40, 70] == 0
        assert mask[5, 5] == 255

    def test_reset_forgets_history(self) -> None:
        gmc = GlobalMotionCompensator(downscale=1)
        gmc.apply(self._textured())
        gmc.reset()
        assert np.allclose(gmc.apply(self._textured(3)), np.eye(2, 3))


class TestAppearance:
    def _player(self, shirt_bgr: tuple[int, int, int]) -> np.ndarray:
        image = np.zeros((200, 120, 3), np.uint8)
        image[:, :] = (40, 130, 45)  # grass
        image[30:110, 30:90] = shirt_bgr  # torso
        return image

    def test_same_kit_is_closer_than_different_kit(self) -> None:
        extractor = TorsoHistogramAppearance()
        box = np.array([[0.0, 0.0, 120.0, 200.0]])

        red_a = extractor.extract(self._player((30, 30, 200)), box)
        red_b = extractor.extract(self._player((35, 35, 190)), box)
        blue = extractor.extract(self._player((200, 40, 30)), box)

        assert cosine_distance(red_a, red_b)[0, 0] < cosine_distance(red_a, blue)[0, 0]

    def test_brightness_change_does_not_break_identity(self) -> None:
        """The same shirt in sun and in shadow must stay the same player.

        Regression test: the descriptor once binned on HSV *value*, so a shirt
        crossing a shadow line jumped bins and scored as maximally dissimilar.
        """
        extractor = TorsoHistogramAppearance()
        box = np.array([[0.0, 0.0, 120.0, 200.0]])

        sunlit = extractor.extract(self._player((30, 30, 210)), box)
        shaded = extractor.extract(self._player((18, 18, 130)), box)
        other_kit = extractor.extract(self._player((200, 40, 30)), box)

        assert cosine_distance(sunlit, shaded)[0, 0] < 0.35
        assert cosine_distance(sunlit, shaded)[0, 0] < cosine_distance(sunlit, other_kit)[0, 0]

    def test_all_grass_crop_yields_no_evidence(self) -> None:
        grass = np.zeros((200, 120, 3), np.uint8)
        grass[:, :] = (40, 130, 45)
        feature = TorsoHistogramAppearance().extract(grass, np.array([[0.0, 0.0, 120.0, 200.0]]))
        assert not np.any(feature)
        assert np.isnan(cosine_distance(feature, feature)[0, 0])

    def test_missing_appearance_abstains_from_tracker_cost(self, config: Config) -> None:
        tracker = MultiObjectTracker(config)
        tracker.update([person(0, 100, 300)], 0, 0.0)
        detection = person(1, 104, 300)
        missing = np.zeros((1, tracker._appearance.dim), dtype=np.float32)

        geometry_only = tracker._cost(tracker._tracked, [detection], None)
        with_missing_appearance = tracker._cost(tracker._tracked, [detection], missing)

        assert np.allclose(with_missing_appearance, geometry_only)

    def test_feature_bank_ignores_low_confidence_updates(self) -> None:
        bank = ExponentialFeatureBank(momentum=0.9, min_confidence=0.5)
        good = np.array([1.0, 0.0, 0.0], np.float32)
        bad = np.array([0.0, 1.0, 0.0], np.float32)
        bank.update(1, good, confidence=0.9)
        bank.update(1, bad, confidence=0.1)  # an occluded crop
        assert bank.get(1, 3)[0] > 0.9


class TestTeamClassification:
    def _frame_with_two_kits(self) -> tuple[np.ndarray, list[tuple[int, tuple]]]:
        image = np.zeros((720, 1280, 3), np.uint8)
        image[:, :] = (45, 135, 50)
        boxes = []
        for i in range(6):
            x = 100 + i * 150
            colour = (40, 40, 210) if i % 2 == 0 else (210, 60, 40)
            image[300:390, x : x + 40] = colour
            boxes.append((i + 1, (float(x), 300.0, float(x + 40), 390.0)))
        return image, boxes

    def test_discovers_two_teams_from_two_kits(self, config: Config, pitch) -> None:
        from visionpitch.team_classification.classifier import TeamClassifier

        config = config.model_copy(deep=True)
        config.team_classification.method = "color"
        config.team_classification.min_votes = 2

        classifier = TeamClassifier(config, pitch)
        image, boxes = self._frame_with_two_kits()

        crops_by_track: dict[int, list] = {}
        for frame in range(0, 20, 5):
            for track_id, box in boxes:
                crop = classifier.extractor.extract(image, box, track_id, frame)
                assert crop is not None
                crops_by_track.setdefault(track_id, []).append(crop)

        classifier.fit([c for crops in crops_by_track.values() for c in crops])

        tracks = {
            track_id: Track(
                track_id=track_id,
                object_class=ObjectClass.PLAYER,
                observations=[TrackObservation(0, 0.0, BBox(*box), 0.9, 0.9, False)],
            )
            for track_id, box in boxes
        }
        classifier.assign_from_crops(crops_by_track, tracks, calibration=None)

        odd = {tracks[i].team_id for i in (1, 3, 5)}
        even = {tracks[i].team_id for i in (2, 4, 6)}
        assert len(odd) == 1 and len(even) == 1
        assert odd != even, "the two kits must land in different teams"

    def test_persistent_referees_are_never_given_a_team(
        self, config: Config, pitch
    ) -> None:
        from visionpitch.common.types import Role, TeamId
        from visionpitch.team_classification.classifier import TeamClassifier

        config = config.model_copy(deep=True)
        config.team_classification.method = "color"
        classifier = TeamClassifier(config, pitch)

        referee = Track(
            track_id=9,
            object_class=ObjectClass.REFEREE,
            observations=[
                TrackObservation(i, i / 25.0, BBox(0, 0, 40, 90), 0.9, 0.9, False)
                for i in range(60)
            ],
            class_votes={ObjectClass.REFEREE.value: 54.0},
            class_counts={ObjectClass.REFEREE.value: 60},
        )
        classifier.assign_from_crops({}, {9: referee}, calibration=None)
        assert referee.team_id is TeamId.NONE
        assert referee.role is Role.REFEREE

    def test_isolated_referee_detections_do_not_relabel_a_player(
        self, config: Config, pitch
    ) -> None:
        """One bad referee box must not cost a player their team.

        The detector's referee class has poor precision, and the birth detection
        used to fix a track's class for its whole life. A track that the detector
        called referee twice in sixty frames is a player.
        """
        from visionpitch.common.types import Role, TeamId
        from visionpitch.team_classification.classifier import TeamClassifier

        config = config.model_copy(deep=True)
        config.team_classification.method = "color"
        classifier = TeamClassifier(config, pitch)

        track = Track(
            track_id=11,
            object_class=ObjectClass.REFEREE,  # unlucky birth detection
            observations=[
                TrackObservation(i, i / 25.0, BBox(0, 0, 40, 90), 0.9, 0.9, False)
                for i in range(60)
            ],
            class_votes={
                ObjectClass.REFEREE.value: 1.2,
                ObjectClass.PLAYER.value: 52.0,
            },
            class_counts={
                ObjectClass.REFEREE.value: 2,
                ObjectClass.PLAYER.value: 58,
            },
        )
        classifier.assign_from_crops({}, {11: track}, calibration=None)
        assert track.role is not Role.REFEREE
        assert track.team_id is not TeamId.NONE

    def test_confident_team_vote_vetoes_a_referee_call(self, pitch) -> None:
        """Colour evidence outranks the detector's weakest class."""
        from visionpitch.common.types import Role, TeamId
        from visionpitch.team_classification.roles import resolve_roles

        track = Track(
            track_id=12,
            object_class=ObjectClass.REFEREE,
            observations=[
                TrackObservation(i, i / 25.0, BBox(0, 0, 40, 90), 0.9, 0.9, False)
                for i in range(120)
            ],
            team_id=TeamId.A,
            team_confidence=0.92,
            class_votes={
                ObjectClass.REFEREE.value: 70.0,
                ObjectClass.PLAYER.value: 20.0,
            },
            class_counts={
                ObjectClass.REFEREE.value: 80,
                ObjectClass.PLAYER.value: 40,
            },
        )
        report = resolve_roles({12: track}, None, pitch)
        assert track.role is Role.OUTFIELD
        assert track.team_id is TeamId.A
        assert 12 in report.referee_rejected_team_veto

    def test_an_outlier_kit_beats_a_confident_two_cluster_vote(self, pitch) -> None:
        """A third kit cannot be voted into one of two clusters.

        ``team_confidence`` is relative: with exactly two clusters fitted, an
        official's kit is forced into whichever is nearer and reports near-total
        confidence for a colour matching neither. Measured on the reference
        broadcast, that put 31 officials into a team's dots and passing graph,
        every one of them at confidence 1.00. The absolute kit-fit test has to be
        reachable before the vote is allowed to veto the referee call.
        """
        from visionpitch.common.types import Role, TeamId
        from visionpitch.team_classification.roles import resolve_roles

        def official(track_id: int) -> Track:
            return Track(
                track_id=track_id,
                object_class=ObjectClass.REFEREE,
                observations=[
                    TrackObservation(i, i / 25.0, BBox(0, 0, 40, 90), 0.9, 0.9, False)
                    for i in range(200)
                ],
                team_id=TeamId.B,
                team_confidence=1.0,
                class_votes={
                    ObjectClass.REFEREE.value: 120.0,
                    ObjectClass.PLAYER.value: 60.0,
                },
                class_counts={
                    ObjectClass.REFEREE.value: 140,
                    ObjectClass.PLAYER.value: 60,
                },
            )

        # Kit sits far outside both clusters: an official, whatever the vote says.
        outlier = official(21)
        report = resolve_roles(
            {21: outlier}, None, pitch, kit_distance={21: 3.0, 22: 1.0, 23: 1.0}
        )
        assert outlier.role is Role.REFEREE
        assert outlier.team_id is TeamId.NONE
        assert 21 in report.referee_accepted

        # Kit sits inside a cluster: the confident vote still wins, so a player
        # with an unlucky run of referee boxes keeps their team.
        inside = official(22)
        report = resolve_roles(
            {22: inside}, None, pitch, kit_distance={21: 1.0, 22: 1.0, 23: 1.0}
        )
        assert inside.role is Role.OUTFIELD
        assert inside.team_id is TeamId.B
        assert 22 in report.referee_rejected_team_veto

    def test_a_split_referee_vote_abstains_instead_of_picking_a_team(self, pitch) -> None:
        """Persistent-but-sub-majority referee evidence must not become a player.

        The UCL broadcast's officials wear cyan, which the two-cluster colour
        model reads as closer to Real Madrid's white than to Liverpool's red.
        The detector split its vote on them (0.447 and 0.512 referee share over
        50 and 137 frames), the old majority gate turned them straight into
        players, and they were rendered with team dots and green passing edges.

        Abstention is the only outcome that cannot be visibly wrong: too weak to
        draw as an official, too strong to hand a team's colours to.
        """
        from visionpitch.common.types import Role, TeamId
        from visionpitch.team_classification.roles import resolve_roles

        def split(track_id: int, referee_weight: float, referee_frames: int) -> Track:
            player_frames = 200 - referee_frames
            return Track(
                track_id=track_id,
                object_class=ObjectClass.REFEREE,
                observations=[
                    TrackObservation(i, i / 25.0, BBox(0, 0, 40, 90), 0.9, 0.9, False)
                    for i in range(200)
                ],
                team_id=TeamId.B,
                team_confidence=1.0,
                class_votes={
                    ObjectClass.REFEREE.value: referee_weight,
                    ObjectClass.PLAYER.value: 1.0 - referee_weight,
                },
                class_counts={
                    ObjectClass.REFEREE.value: referee_frames,
                    ObjectClass.PLAYER.value: player_frames,
                },
            )

        official = split(20, 0.512, 137)
        report = resolve_roles({20: official}, None, pitch, kit_distance={20: 1.0})
        assert official.role is Role.UNKNOWN
        assert official.team_id is TeamId.UNKNOWN, "an official kept a team"
        assert 20 in report.referee_abstained_split_vote

        # Well below the abstain floor: an ordinary player who picked up a few
        # spurious referee boxes keeps their team.
        noisy = split(25, 0.049, 20)
        resolve_roles({25: noisy}, None, pitch, kit_distance={25: 1.0})
        assert noisy.role is Role.OUTFIELD
        assert noisy.team_id is TeamId.B

    def test_the_weak_vote_counter_is_not_dead(self, pitch) -> None:
        """It used to be guarded on a role nothing sets, so it always read zero.

        The report therefore claimed no referee call had ever been turned down
        while officials were quietly being handed teams.
        """
        from visionpitch.common.types import TeamId
        from visionpitch.team_classification.roles import resolve_roles

        track = Track(
            track_id=7,
            object_class=ObjectClass.PLAYER,
            observations=[
                TrackObservation(i, i / 25.0, BBox(0, 0, 40, 90), 0.9, 0.9, False)
                for i in range(120)
            ],
            team_id=TeamId.A,
            team_confidence=0.95,
            class_votes={ObjectClass.PLAYER.value: 90.0, ObjectClass.REFEREE.value: 8.0},
            class_counts={ObjectClass.PLAYER.value: 100, ObjectClass.REFEREE.value: 20},
        )
        report = resolve_roles({7: track}, None, pitch, kit_distance={7: 1.0})
        assert 7 in report.referee_rejected_weak_votes
        assert report.to_dict()["referee_rejected_weak_votes"] == 1

    def test_goalkeeper_needs_pitch_evidence_not_only_the_detector(self, pitch) -> None:
        """A detector-only keeper in midfield is demoted, not trusted."""
        import numpy as np

        from visionpitch.common.types import CalibrationResult, Role, TeamId
        from visionpitch.team_classification.roles import resolve_roles

        # Identity homography: image pixels read directly as metres, so the
        # track's box places it at the centre circle rather than a goal line.
        calibration = {
            i: CalibrationResult(
                frame_idx=i,
                homography=np.eye(3, dtype=np.float64),
                confidence=0.9,
                reprojection_error_m=0.2,
                n_keypoints=12,
                n_inliers=12,
            )
            for i in range(60)
        }
        midfielder = Track(
            track_id=13,
            object_class=ObjectClass.GOALKEEPER,
            observations=[
                TrackObservation(i, i / 25.0, BBox(50, 30, 55, 34), 0.9, 0.9, False)
                for i in range(60)
            ],
            team_id=TeamId.B,
            team_confidence=0.8,
            class_votes={ObjectClass.GOALKEEPER.value: 55.0},
            class_counts={ObjectClass.GOALKEEPER.value: 60},
        )
        report = resolve_roles({13: midfielder}, calibration, pitch)
        assert midfielder.role is Role.OUTFIELD
        assert 13 in report.goalkeeper_demoted_position

        keeper = Track(
            track_id=14,
            object_class=ObjectClass.GOALKEEPER,
            observations=[
                TrackObservation(i, i / 25.0, BBox(3, 30, 8, 34), 0.9, 0.9, False)
                for i in range(60)
            ],
            team_id=TeamId.B,
            team_confidence=0.8,
            class_votes={ObjectClass.GOALKEEPER.value: 55.0},
            class_counts={ObjectClass.GOALKEEPER.value: 60},
        )
        report = resolve_roles({14: keeper}, calibration, pitch)
        assert keeper.role is Role.GOALKEEPER
        assert keeper.team_id is TeamId.B

    def test_manual_correction_overrides_with_full_confidence(self, pitch) -> None:
        from visionpitch.common.types import TeamId
        from visionpitch.team_classification.classifier import TeamClassifier

        track = Track(
            track_id=4,
            object_class=ObjectClass.PLAYER,
            observations=[TrackObservation(0, 0.0, BBox(0, 0, 40, 90), 0.9, 0.9, False)],
            team_id=TeamId.A,
            team_confidence=0.51,
        )
        n = TeamClassifier.apply_corrections({4: track}, {"4": {"team_id": "B"}})
        assert n == 1
        assert track.team_id is TeamId.B
        assert track.team_confidence == 1.0
        _ = pitch


class TestJerseyCropExtraction:
    def test_rejects_a_crop_that_is_mostly_grass(self, config: Config) -> None:
        from visionpitch.team_classification.crops import JerseyCropExtractor

        grass = np.zeros((720, 1280, 3), np.uint8)
        grass[:, :] = (45, 135, 50)
        extractor = JerseyCropExtractor(config.team_classification)
        assert extractor.extract(grass, (100.0, 300.0, 140.0, 390.0), 1, 0) is None

    def test_keeps_both_raw_and_masked_crops(self, config: Config) -> None:
        from visionpitch.team_classification.crops import JerseyCropExtractor

        image = np.zeros((720, 1280, 3), np.uint8)
        image[:, :] = (45, 135, 50)
        # A shirt narrower than the person box, so the torso crop genuinely
        # contains some pitch for the grass mask to remove.
        image[315:345, 112:128] = (30, 30, 200)

        crop = JerseyCropExtractor(config.team_classification).extract(
            image, (100.0, 300.0, 140.0, 390.0), 1, 0
        )
        assert crop is not None
        assert crop.image.shape == crop.masked.shape
        # The masked version must have zeros where grass was removed; the raw
        # version must not, since a learned embedder needs an unpunctured image.
        assert (crop.masked == 0).any()
        assert crop.mean_colour[2] > crop.mean_colour[1]  # red channel dominant

    def test_torso_region_excludes_head_and_legs(self, config: Config) -> None:
        from visionpitch.team_classification.crops import JerseyCropExtractor

        extractor = JerseyCropExtractor(config.team_classification)
        x1, y1, x2, y2 = extractor._torso_box((100.0, 300.0, 140.0, 400.0), 1280, 720)
        assert y1 > 300, "the head must be excluded"
        assert y2 < 400, "the legs must be excluded"
        assert x1 > 100 and x2 < 140, "the sides must be trimmed"


def test_cv2_is_available_for_these_tests() -> None:
    assert cv2.__version__
