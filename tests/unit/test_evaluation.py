"""Evaluation metrics.

Metrics are tested against cases with known analytic answers. A metric
implementation that is merely "plausible" is worse than none: it produces
numbers that get quoted.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionpitch.common.types import BBox, Detection, ObjectClass, Track, TrackObservation
from visionpitch.evaluation.detection import _average_precision, evaluate_detection
from visionpitch.evaluation.ground_truth import GroundTruth, GTObject
from visionpitch.evaluation.tracking import evaluate_tracking

Box = tuple[float, float, float, float]


def make_gt(frames: dict[int, list[tuple[int, Box]]]) -> GroundTruth:
    gt = GroundTruth(video_id="test", fps=25.0)
    for frame_idx, entries in frames.items():
        gt.frames[frame_idx] = [
            GTObject(ObjectClass.PLAYER, track_id, BBox(*box)) for track_id, box in entries
        ]
    return gt


def make_track(track_id: int, boxes: dict[int, Box]) -> Track:
    return Track(
        track_id=track_id,
        object_class=ObjectClass.PLAYER,
        observations=[
            TrackObservation(f, f / 25.0, BBox(*b), 0.9, 0.9, False)
            for f, b in sorted(boxes.items())
        ],
    )


class TestAveragePrecision:
    def test_perfect_ranking_gives_one(self) -> None:
        tp = np.array([True] * 10)
        scores = np.linspace(0.9, 0.5, 10)
        assert _average_precision(tp, scores, n_gt=10) == pytest.approx(1.0, abs=1e-6)

    def test_no_true_positives_gives_zero(self) -> None:
        tp = np.array([False] * 5)
        assert _average_precision(tp, np.linspace(0.9, 0.5, 5), n_gt=5) == 0.0

    def test_half_recall_caps_ap(self) -> None:
        """5 of 10 found, all ranked first: precision 1 up to recall 0.5."""
        tp = np.array([True] * 5)
        ap = _average_precision(tp, np.linspace(0.9, 0.5, 5), n_gt=10)
        assert ap == pytest.approx(0.5, abs=0.02)

    def test_no_ground_truth_is_undefined_not_zero(self) -> None:
        assert np.isnan(_average_precision(np.array([True]), np.array([0.9]), n_gt=0))


class TestEvaluateDetection:
    def test_perfect_detector_scores_one(self) -> None:
        gt = make_gt({0: [(1, (100, 100, 140, 200))], 1: [(1, (110, 100, 150, 200))]})
        predictions = {
            0: [Detection(0, ObjectClass.PLAYER, BBox(100, 100, 140, 200), 0.95)],
            1: [Detection(1, ObjectClass.PLAYER, BBox(110, 100, 150, 200), 0.95)],
        }
        result = evaluate_detection(gt, predictions)
        player = result["per_class"]["player"]
        assert player["precision"] == 1.0
        assert player["recall"] == 1.0
        assert player["mAP50"] == pytest.approx(1.0, abs=1e-6)

    def test_missed_object_lowers_recall_only(self) -> None:
        gt = make_gt({0: [(1, (100, 100, 140, 200)), (2, (300, 100, 340, 200))]})
        predictions = {0: [Detection(0, ObjectClass.PLAYER, BBox(100, 100, 140, 200), 0.9)]}
        player = evaluate_detection(gt, predictions)["per_class"]["player"]
        assert player["precision"] == 1.0
        assert player["recall"] == 0.5
        assert player["false_negatives"] == 1

    def test_spurious_detection_lowers_precision_only(self) -> None:
        gt = make_gt({0: [(1, (100, 100, 140, 200))]})
        predictions = {
            0: [
                Detection(0, ObjectClass.PLAYER, BBox(100, 100, 140, 200), 0.9),
                Detection(0, ObjectClass.PLAYER, BBox(600, 400, 640, 500), 0.8),
            ]
        }
        player = evaluate_detection(gt, predictions)["per_class"]["player"]
        assert player["recall"] == 1.0
        assert player["precision"] == 0.5

    def test_unannotated_frames_are_not_counted_as_empty(self) -> None:
        """Predictions on frames nobody annotated must not become false positives."""
        gt = make_gt({0: [(1, (100, 100, 140, 200))]})
        predictions = {
            0: [Detection(0, ObjectClass.PLAYER, BBox(100, 100, 140, 200), 0.9)],
            99: [Detection(99, ObjectClass.PLAYER, BBox(10, 10, 50, 90), 0.9)] * 5,
        }
        player = evaluate_detection(gt, predictions)["per_class"]["player"]
        assert player["precision"] == 1.0
        assert player["false_positives"] == 0

    def test_ball_is_reported_separately(self) -> None:
        gt = GroundTruth(video_id="t", fps=25.0)
        gt.frames[0] = [
            GTObject(ObjectClass.PLAYER, 1, BBox(100, 100, 140, 200)),
            GTObject(ObjectClass.BALL, 0, BBox(300, 300, 310, 310)),
        ]
        predictions = {0: [Detection(0, ObjectClass.PLAYER, BBox(100, 100, 140, 200), 0.9)]}
        result = evaluate_detection(gt, predictions)
        assert result["overall"]["ball_recall"] == 0.0
        assert result["per_class"]["player"]["recall"] == 1.0

    def test_small_object_recall_is_tracked(self) -> None:
        gt = GroundTruth(video_id="t", fps=25.0)
        gt.frames[0] = [GTObject(ObjectClass.BALL, 0, BBox(300, 300, 310, 310))]  # 100 px^2
        predictions = {0: [Detection(0, ObjectClass.BALL, BBox(300, 300, 310, 310), 0.5)]}
        ball = evaluate_detection(gt, predictions)["per_class"]["ball"]
        assert ball["n_small_objects"] == 1
        assert ball["small_object_recall"] == 1.0


class TestEvaluateTracking:
    def test_perfect_tracking_scores_one(self) -> None:
        boxes = {f: (100.0 + 5 * f, 300.0, 140.0 + 5 * f, 400.0) for f in range(15)}
        gt = make_gt({f: [(1, b)] for f, b in boxes.items()})
        tracks = {1: make_track(1, boxes)}

        result = evaluate_tracking(gt, tracks)
        assert result["IDF1"] == pytest.approx(1.0, abs=1e-6)
        assert result["MOTA"] == pytest.approx(1.0, abs=1e-6)
        assert result["id_switches"] == 0
        assert result["HOTA"] > 0.9

    def test_an_id_switch_is_counted(self) -> None:
        boxes = {f: (100.0 + 5 * f, 300.0, 140.0 + 5 * f, 400.0) for f in range(20)}
        gt = make_gt({f: [(1, b)] for f, b in boxes.items()})
        # The tracker splits the same player into two ids halfway through.
        tracks = {
            1: make_track(1, {f: b for f, b in boxes.items() if f < 10}),
            2: make_track(2, {f: b for f, b in boxes.items() if f >= 10}),
        }
        result = evaluate_tracking(gt, tracks)
        assert result["id_switches"] == 1
        assert result["IDF1"] < 0.75
        # MOTA barely notices - which is precisely why IDF1 and HOTA are reported.
        assert result["MOTA"] > 0.9

    def test_association_failure_lowers_hota_assa(self) -> None:
        boxes_a = {f: (100.0, 300.0, 140.0, 400.0) for f in range(20)}
        boxes_b = {f: (500.0, 300.0, 540.0, 400.0) for f in range(20)}
        gt = make_gt({f: [(1, boxes_a[f]), (2, boxes_b[f])] for f in range(20)})

        good = {1: make_track(1, boxes_a), 2: make_track(2, boxes_b)}
        swapped = {
            1: make_track(1, {**{f: boxes_a[f] for f in range(10)},
                              **{f: boxes_b[f] for f in range(10, 20)}}),
            2: make_track(2, {**{f: boxes_b[f] for f in range(10)},
                              **{f: boxes_a[f] for f in range(10, 20)}}),
        }
        assert evaluate_tracking(gt, swapped)["AssA"] < evaluate_tracking(gt, good)["AssA"]

    def test_missing_predictions_lower_deta(self) -> None:
        boxes = {f: (100.0 + 5 * f, 300.0, 140.0 + 5 * f, 400.0) for f in range(20)}
        gt = make_gt({f: [(1, b)] for f, b in boxes.items()})
        partial = {1: make_track(1, {f: b for f, b in boxes.items() if f % 2 == 0})}
        result = evaluate_tracking(gt, partial)
        assert result["DetA"] < 0.75
        assert result["false_negatives"] == 10

    def test_interpolated_boxes_do_not_count_as_detections(self) -> None:
        """A tracker must not be able to inflate recall by coasting."""
        boxes = {f: (100.0 + 5 * f, 300.0, 140.0 + 5 * f, 400.0) for f in range(10)}
        gt = make_gt({f: [(1, b)] for f, b in boxes.items()})

        track = make_track(1, boxes)
        for obs in track.observations[5:]:
            obs.interpolated = True
        result = evaluate_tracking(gt, {1: track})
        assert result["false_negatives"] == 5

    def test_no_predictions_scores_zero_not_an_error(self) -> None:
        gt = make_gt({f: [(1, (100.0, 300.0, 140.0, 400.0))] for f in range(5)})
        result = evaluate_tracking(gt, {})
        assert result["HOTA"] == 0.0
        assert result["IDF1"] == 0.0

    def test_mostly_tracked_buckets(self) -> None:
        boxes = {f: (100.0, 300.0, 140.0, 400.0) for f in range(20)}
        gt = make_gt({f: [(1, boxes[f])] for f in range(20)})
        tracks = {1: make_track(1, {f: boxes[f] for f in range(18)})}
        result = evaluate_tracking(gt, tracks)
        assert result["mostly_tracked"] == 1
        assert result["mostly_lost"] == 0
