"""Phase 2B: dataset splits, event ground truth, event and possession metrics.

No test here touches the network. Fixtures are synthetic or read files the
download scripts already placed on disk, and skip cleanly when absent.
"""

from __future__ import annotations

import json

import pytest

from visionpitch.analytics.types import (
    BallStateKind,
    ClipReference,
    EventType,
    Evidence,
    FootballEvent,
)
from visionpitch.evaluation.event_gt import (
    EVENT_GT_SCHEMA_VERSION,
    EventGroundTruth,
    GTEvent,
    GTEventType,
    IgnoreInterval,
    validate,
)
from visionpitch.evaluation.event_metrics import (
    Rate,
    evaluate_events,
    evaluate_possession,
    wilson_interval,
)
from visionpitch.evaluation.splits import assert_disjoint, make_split


def gt_event(t: float, kind=GTEventType.PASS_START, team="left") -> GTEvent:
    return GTEvent(event_type=kind, start_time_s=t, team=team)


def predicted(t: float, kind=EventType.PASS, team="A") -> FootballEvent:
    return FootballEvent(
        event_id=f"e{t}", event_type=kind, frame_idx=int(t * 25), timestamp_s=t,
        team_id=team, confidence=0.8, ball_state=BallStateKind.OBSERVED,
        evidence=Evidence().add("test"),
        clip=ClipReference(0, 1, 0.0, 1.0),
    )


def make_gt(events, ignore=None, identity=False) -> EventGroundTruth:
    return EventGroundTruth(
        clip_id="t", fps=25.0, source="synthetic", licence="n/a",
        events=events, ignore_intervals=ignore or [], has_player_identity=identity,
    )


class TestSplits:
    def test_split_is_deterministic(self) -> None:
        names = [f"SNGS-{i}" for i in range(40)]
        a, b = make_split(names), make_split(names)
        assert a.assignments == b.assignments
        assert a.fingerprint() == b.fingerprint()

    def test_splits_are_disjoint(self) -> None:
        split = make_split([f"SNGS-{i}" for i in range(40)])
        assert_disjoint(split)
        assert len(split.train) + len(split.val) + len(split.test) == 40

    def test_adding_sequences_does_not_move_existing_ones(self) -> None:
        """A later sequence must never reshuffle an earlier one out of test,
        which would silently invalidate a previously reported number."""
        first = make_split([f"SNGS-{i}" for i in range(20)])
        second = make_split([f"SNGS-{i}" for i in range(40)])
        for name in first.assignments:
            assert first.of(name) == second.of(name)

    def test_ratios_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            make_split(["a"], ratios={"train": 0.5, "val": 0.2, "test": 0.2})

    def test_round_trip_detects_tampering(self, tmp_path) -> None:
        from visionpitch.evaluation.splits import DatasetSplit

        split = make_split([f"S{i}" for i in range(20)])
        path = split.save(tmp_path / "split.json")
        assert DatasetSplit.load(path).fingerprint() == split.fingerprint()

        data = json.loads(path.read_text(encoding="utf-8"))
        moved = next(k for k, v in data["assignments"].items() if v == "test")
        data["assignments"][moved] = "train"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="edited"):
            DatasetSplit.load(path)


class TestEventGroundTruthSchema:
    def test_unknown_ambiguous_ignore_are_not_scorable(self) -> None:
        for kind in (GTEventType.UNKNOWN, GTEventType.AMBIGUOUS, GTEventType.IGNORE):
            assert not kind.is_scorable
        assert GTEventType.PASS_START.is_scorable

    def test_scorable_excludes_non_answers(self) -> None:
        gt = make_gt([
            gt_event(1.0),
            gt_event(2.0, GTEventType.UNKNOWN),
            gt_event(3.0, GTEventType.AMBIGUOUS),
        ])
        assert len(gt.events) == 3
        assert len(gt.scorable()) == 1

    def test_ignore_intervals_remove_events_from_scoring(self) -> None:
        gt = make_gt(
            [gt_event(1.0), gt_event(5.0), gt_event(9.0)],
            ignore=[IgnoreInterval(4.0, 6.0, "replay")],
        )
        assert len(gt.scorable()) == 2
        assert gt.is_ignored(5.0) and not gt.is_ignored(1.0)

    def test_round_trip(self, tmp_path) -> None:
        gt = make_gt([gt_event(1.0), gt_event(2.5, GTEventType.SHOT)])
        path = gt.save(tmp_path / "gt.json")
        loaded = EventGroundTruth.load(path)
        assert loaded.fingerprint() == gt.fingerprint()
        assert loaded.schema_version == EVENT_GT_SCHEMA_VERSION
        assert len(loaded.events) == 2

    def test_validation_flags_malformed_events(self) -> None:
        gt = make_gt([
            GTEvent(GTEventType.PASS_START, start_time_s=5.0, end_time_s=2.0, team="left"),
            GTEvent(GTEventType.PASS_START, start_time_s=1.0, team=None),
            GTEvent(GTEventType.SHOT, start_time_s=2.0, team="left", confidence=1.7),
        ])
        report = validate(gt)
        assert not report["valid"]
        assert "end_before_start" in report["issue_counts"]
        assert "missing_team" in report["issue_counts"]
        assert "confidence_out_of_range" in report["issue_counts"]

    def test_identity_claimed_without_source_is_flagged(self) -> None:
        """A player id on a corpus that has none is fabricated data."""
        gt = make_gt([GTEvent(GTEventType.PASS_START, 1.0, team="left", player_id="7")])
        assert "identity_claimed_without_source" in validate(gt)["issue_counts"]

    def test_clean_annotation_validates(self) -> None:
        gt = make_gt([gt_event(1.0), gt_event(2.0), gt_event(3.0)])
        assert validate(gt)["valid"]


class TestRatesAndIntervals:
    def test_rate_carries_counts(self) -> None:
        payload = Rate(3, 10).to_dict()
        assert payload["value"] == 0.3
        assert payload["numerator"] == 3 and payload["denominator"] == 10
        assert payload["ci95"][0] < 0.3 < payload["ci95"][1]

    def test_zero_denominator_is_none_not_zero(self) -> None:
        assert Rate(0, 0).to_dict()["value"] is None

    def test_wilson_interval_is_wider_for_small_samples(self) -> None:
        small = wilson_interval(3, 5)
        large = wilson_interval(300, 500)
        assert (small[1] - small[0]) > (large[1] - large[0])

    def test_wilson_interval_stays_in_range(self) -> None:
        for successes, total in ((0, 5), (5, 5), (1, 100)):
            lo, hi = wilson_interval(successes, total)
            assert 0.0 <= lo <= hi <= 1.0


class TestEventMatching:
    def test_perfect_prediction_scores_one(self) -> None:
        gt = make_gt([gt_event(1.0), gt_event(2.0), gt_event(3.0)])
        preds = [predicted(1.0), predicted(2.0), predicted(3.0)]
        report = evaluate_events(gt, preds, tolerances_s=(0.4,))
        row = report["per_event_type"]["pass_start"][0]
        assert row["precision"]["value"] == 1.0
        assert row["recall"]["value"] == 1.0
        assert row["f1"] == 1.0

    def test_tolerance_controls_matching(self) -> None:
        gt = make_gt([gt_event(1.0)])
        preds = [predicted(1.3)]
        loose = evaluate_events(gt, preds, tolerances_s=(0.5,))
        tight = evaluate_events(gt, preds, tolerances_s=(0.1,))
        assert loose["per_event_type"]["pass_start"][0]["recall"]["value"] == 1.0
        assert tight["per_event_type"]["pass_start"][0]["recall"]["value"] == 0.0

    def test_matching_is_one_to_one(self) -> None:
        """Two predictions on one ground-truth event is one hit and one false
        positive, not two hits."""
        gt = make_gt([gt_event(1.0)])
        preds = [predicted(1.0), predicted(1.05)]
        row = evaluate_events(gt, preds, tolerances_s=(0.4,))["per_event_type"]["pass_start"][0]
        assert row["recall"]["value"] == 1.0
        assert row["precision"]["value"] == 0.5

    def test_window_restricts_the_denominator(self) -> None:
        """Scoring a short segment against a whole-match annotation file must
        not inflate the recall denominator -- the bug this guard exists for."""
        gt = make_gt([gt_event(float(t)) for t in range(0, 100)])
        preds = [predicted(float(t)) for t in range(10, 20)]

        unwindowed = evaluate_events(gt, preds, tolerances_s=(0.4,))
        windowed = evaluate_events(gt, preds, tolerances_s=(0.4,), window_s=(10.0, 19.0))

        assert unwindowed["per_event_type"]["pass_start"][0]["n_ground_truth"] == 100
        assert windowed["per_event_type"]["pass_start"][0]["n_ground_truth"] == 10
        assert windowed["per_event_type"]["pass_start"][0]["recall"]["value"] == 1.0

    def test_ignored_intervals_are_not_errors(self) -> None:
        gt = make_gt(
            [gt_event(1.0), gt_event(5.0)],
            ignore=[IgnoreInterval(4.0, 6.0, "replay")],
        )
        preds = [predicted(1.0), predicted(5.0)]
        row = evaluate_events(gt, preds, tolerances_s=(0.4,))["per_event_type"]["pass_start"][0]
        # The ignored ground truth and the prediction inside it both vanish.
        assert row["n_ground_truth"] == 1
        assert row["n_predicted"] == 1
        assert row["precision"]["value"] == 1.0

    def test_player_attribution_is_null_without_labels(self) -> None:
        gt = make_gt([gt_event(1.0)], identity=False)
        report = evaluate_events(gt, [predicted(1.0)], tolerances_s=(0.4,))
        assert report["player_attribution_measurable"] is False
        assert report["per_event_type"]["pass_start"][0]["player_accuracy"] is None

    def test_temporal_error_is_reported(self) -> None:
        gt = make_gt([gt_event(1.0), gt_event(2.0)])
        preds = [predicted(1.1), predicted(2.2)]
        row = evaluate_events(gt, preds, tolerances_s=(0.5,))["per_event_type"]["pass_start"][0]
        assert row["temporal_error_s"]["median"] == pytest.approx(0.15, abs=0.02)

    def test_false_positive_categories_are_populated(self) -> None:
        gt = make_gt([gt_event(1.0)])
        preds = [predicted(1.0), predicted(40.0)]
        row = evaluate_events(gt, preds, tolerances_s=(0.4,))["per_event_type"]["pass_start"][0]
        assert sum(row["false_positive_categories"].values()) >= 1


class TestPossessionMetrics:
    def test_identical_tracks_score_one(self) -> None:
        reference = {f: ("controlled", 7) for f in range(50)}
        result = evaluate_possession(reference, dict(reference))
        assert result["frame_accuracy"]["value"] == 1.0
        assert result["player_attribution_accuracy"]["value"] == 1.0

    def test_wrong_player_is_caught_even_when_state_agrees(self) -> None:
        reference = {f: ("controlled", 7) for f in range(50)}
        predicted_track = {f: ("controlled", 9) for f in range(50)}
        result = evaluate_possession(reference, predicted_track)
        assert result["frame_accuracy"]["value"] == 1.0
        assert result["player_attribution_accuracy"]["value"] == 0.0

    def test_unknown_recall_is_reported(self) -> None:
        reference = {f: ("unknown", None) for f in range(10)}
        predicted_track = {f: ("unknown", None) if f < 6 else ("controlled", 1)
                           for f in range(10)}
        result = evaluate_possession(reference, predicted_track)
        assert result["unknown_recall"]["value"] == pytest.approx(0.6)

    def test_ignored_frames_are_excluded(self) -> None:
        reference = {f: ("controlled", 1) for f in range(10)}
        predicted_track = {f: ("loose_ball", None) if f < 5 else ("controlled", 1)
                           for f in range(10)}
        result = evaluate_possession(reference, predicted_track, ignored_frames=set(range(5)))
        assert result["frames"] == 5
        assert result["frame_accuracy"]["value"] == 1.0

    def test_no_overlap_is_reported_not_crashed(self) -> None:
        assert evaluate_possession({1: ("controlled", 1)}, {99: ("controlled", 1)})["frames"] == 0


@pytest.mark.slow
class TestRealCorpora:
    def test_bas_ground_truth_loads_and_validates(self, repo_root) -> None:
        path = repo_root / "data" / "eval" / "bas" / "event_gt_half1.json"
        if not path.exists():
            pytest.skip("SN-BAS ground truth not downloaded")

        gt = EventGroundTruth.load(path)
        assert gt.has_player_identity is False, (
            "SN-BAS has no player labels; claiming otherwise would fabricate metrics"
        )
        assert len(gt.scorable()) > 100
        report = validate(gt)
        # Missing team is acceptable in this corpus; nothing structural may fail.
        assert not report["issue_counts"].get("end_before_start")
        assert not report["issue_counts"].get("identity_claimed_without_source")

    def test_ball_finetune_split_is_leak_free(self, repo_root) -> None:
        from visionpitch.evaluation.splits import DatasetSplit

        path = repo_root / "data" / "ball_finetune" / "split.json"
        if not path.exists():
            pytest.skip("ball fine-tuning dataset not exported")
        split = DatasetSplit.load(path)
        assert_disjoint(split)
        assert split.test, "a held-out test split must exist"
