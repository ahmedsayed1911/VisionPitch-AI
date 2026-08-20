"""Phase 2C: multi-corpus registry, clip-disjoint splits, and the review store.

The properties pinned here are the ones whose failure would be silent: a split
that leaks, a fingerprint that does not notice tampering, and a review that
edits the prediction it was supposed to leave alone.
"""

from __future__ import annotations

import json

import pytest

from visionpitch.analytics.types import is_team
from visionpitch.api.reviews import (
    REVIEW_SCHEMA_VERSION,
    UNKNOWN_PLAYER,
    ReviewAction,
    ReviewStore,
)
from visionpitch.api.store import Store
from visionpitch.evaluation.possession_eval import (
    PossessionResult,
    _engine_label,
    aggregate,
    evaluate_possession_vs_gt,
)
from visionpitch.evaluation.possession_gt import (
    DerivationParams,
    GSRObject,
    PossessionGroundTruth,
    PossessionInterval,
    PossessionLabel,
    _frame_label,
    _to_intervals,
)
from visionpitch.evaluation.registry import (
    CORPORA,
    MultiCorpusSplit,
    assert_no_leakage,
    build_split,
    registry_document,
    roboflow_clip_ids,
)

# --------------------------------------------------------------------------- #
# Registry and splits
# --------------------------------------------------------------------------- #


def test_bas_cannot_train_a_ball_detector():
    """SN-BAS has no ball boxes; eligibility must reflect that, not convenience."""
    bas = CORPORA["bas"]
    assert bas.ball_annotation == "NONE"
    assert not bas.train_eligible and not bas.val_eligible
    assert bas.test_eligible


def test_roboflow_corpora_are_one_domain():
    """Both Roboflow sets share source clips, so they cannot be treated as two."""
    domains = {c.domain for c in CORPORA.values()}
    assert len(domains) == 3, domains
    assert CORPORA["roboflow"].domain == "roboflow"


def test_registry_document_separates_training_from_coverage_only():
    doc = registry_document()
    assert "soccernet_bas" in doc["coverage_only_domains"]
    assert "soccernet_bas" not in doc["ball_training_domains"]
    assert set(doc["ball_training_domains"]) == {"roboflow", "soccernet_gsr"}


def test_split_is_clip_disjoint():
    split = build_split([], [f"seq_{i:03d}" for i in range(60)])
    assert_no_leakage(split)


def test_split_assignment_is_stable_when_clips_are_added():
    """Adding a sequence must not move an existing one between splits.

    Otherwise a later dataset build silently re-partitions history and every
    previously reported held-out number becomes unreproducible.
    """
    before = build_split([], [f"seq_{i:03d}" for i in range(20)])
    after = build_split([], [f"seq_{i:03d}" for i in range(40)])
    for clip, assigned in before.assignments["soccernet_gsr"].items():
        assert after.assignments["soccernet_gsr"][clip] == assigned


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        build_split([], ["a"], ratios={"train": 0.5, "val": 0.2, "test": 0.2})


def test_a_clip_assigned_to_no_split_is_rejected():
    """The live failure mode for this mapping shape.

    A clip cannot literally appear in two splits -- ``assignments`` maps each
    clip to exactly one name -- so leakage here shows up as a clip carrying a
    split name that is not train/val/test, which would silently drop it from
    every set while still counting it as assigned.
    """
    split = MultiCorpusSplit(
        seed="s",
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        assignments={"c": {"a": "train", "b": "test"}},
    )
    assert_no_leakage(split)

    split.assignments["c"]["b"] = "holdout"
    with pytest.raises(AssertionError, match="unassigned"):
        assert_no_leakage(split)


def test_split_file_detects_editing(tmp_path):
    split = build_split([], [f"seq_{i}" for i in range(10)])
    path = split.save(tmp_path / "split.json")
    assert MultiCorpusSplit.load(path).fingerprint() == split.fingerprint()

    data = json.loads(path.read_text(encoding="utf-8"))
    moved = next(iter(data["assignments"]["soccernet_gsr"]))
    # Move it to a split it is definitely not already in, so the edit is real
    # regardless of which split the hash happened to pick.
    others = {"train", "val", "test"} - {data["assignments"]["soccernet_gsr"][moved]}
    data["assignments"]["soccernet_gsr"][moved] = sorted(others)[0]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="was edited"):
        MultiCorpusSplit.load(path)


def test_roboflow_clip_id_is_the_filename_prefix(tmp_path):
    images = tmp_path / "data" / "train" / "images"
    images.mkdir(parents=True)
    for name in ("a1b2c3_frame1.jpg", "a1b2c3_frame2.jpg", "ffeedd_frame1.jpg"):
        (images / name).write_bytes(b"")
    clips = roboflow_clip_ids(tmp_path)
    assert set(clips) == {"a1b2c3", "ffeedd"}
    assert len(clips["a1b2c3"]) == 2


# --------------------------------------------------------------------------- #
# Review store
# --------------------------------------------------------------------------- #


@pytest.fixture
def reviews(tmp_path):
    return ReviewStore(Store(f"sqlite:///{tmp_path / 'r.db'}"))


PREDICTIONS = [
    {"event_id": "e1", "type": "pass", "timestamp_s": 10.0, "track_id": 7,
     "team_id": "left", "confidence": 0.9},
    {"event_id": "e2", "type": "recovery", "timestamp_s": 20.0, "track_id": 3,
     "team_id": "right", "confidence": 0.2},
]


def test_correction_never_mutates_the_prediction(reviews):
    original = json.dumps(PREDICTIONS, sort_keys=True)
    reviews.add("j", ReviewAction.RETYPE, event_id="e2", corrected_type="pass")
    view = reviews.corrected_view("j", PREDICTIONS)

    assert json.dumps(PREDICTIONS, sort_keys=True) == original
    row = next(e for e in view["events"] if e["event_id"] == "e2")
    assert row["type"] == "pass"
    assert row["raw"]["type"] == "recovery"


def test_action_other_than_add_missed_needs_an_event(reviews):
    with pytest.raises(ValueError, match="requires the event_id"):
        reviews.add("j", ReviewAction.CONFIRM)


def test_add_missed_needs_a_time(reviews):
    with pytest.raises(ValueError, match="requires corrected_start_s"):
        reviews.add("j", ReviewAction.ADD_MISSED, corrected_type="shot")


def test_added_event_may_have_an_unknown_actor(reviews):
    """A reviewer who saw a shot but not who took it must be able to say so."""
    reviews.add("j", ReviewAction.ADD_MISSED, corrected_type="shot",
                corrected_start_s=31.0, corrected_track_id=UNKNOWN_PLAYER)
    view = reviews.corrected_view("j", PREDICTIONS)
    added = [e for e in view["events"] if e["review_status"] == "add_missed"]
    assert len(added) == 1
    assert added[0]["track_id"] == UNKNOWN_PLAYER
    assert added[0]["raw"] is None
    assert view["n_predictions"] == 2


def test_history_is_append_only(reviews):
    reviews.add("j", ReviewAction.RETYPE, event_id="e1", corrected_type="cross",
                reviewer="first")
    reviews.add("j", ReviewAction.CONFIRM, event_id="e1", reviewer="second")

    assert len(reviews.list("j")) == 2
    view = reviews.corrected_view("j", PREDICTIONS)
    row = next(e for e in view["events"] if e["event_id"] == "e1")
    assert row["review_status"] == "confirm"
    assert row["reviewer"] == "second"
    # The superseded decision is still on record.
    assert any(c.corrected_type == "cross" for c in reviews.list("j"))


def test_corrections_are_scoped_to_their_job(reviews):
    reviews.add("j1", ReviewAction.REJECT, event_id="e1")
    assert reviews.corrected_view("j2", PREDICTIONS)["n_corrections"] == 0


def test_review_ranking_prefers_uncertainty(reviews):
    ranked = ReviewStore.rank_for_review(
        reviews.corrected_view("j", PREDICTIONS)["events"]
    )
    assert ranked[0]["event_id"] == "e2"  # confidence 0.2 before 0.9


def test_reviewed_events_sink_below_unreviewed(reviews):
    reviews.add("j", ReviewAction.CONFIRM, event_id="e2")
    ranked = ReviewStore.rank_for_review(
        reviews.corrected_view("j", PREDICTIONS)["events"]
    )
    assert ranked[0]["event_id"] == "e1"


def test_export_is_fingerprinted_and_only_covers_reviewed_events(reviews, tmp_path):
    reviews.add("j", ReviewAction.RETYPE, event_id="e1", corrected_type="cross")
    payload = reviews.export_training_examples("j", PREDICTIONS, tmp_path / "x.json")

    assert payload["n_reviewed"] == 1
    assert payload["schema_version"] == REVIEW_SCHEMA_VERSION
    assert len(payload["fingerprint"]) == 16
    assert json.loads((tmp_path / "x.json").read_text(encoding="utf-8")) == payload

    # Same corrections, same fingerprint; a further correction changes it.
    again = reviews.export_training_examples("j", PREDICTIONS, tmp_path / "y.json")
    assert again["fingerprint"] == payload["fingerprint"]
    reviews.add("j", ReviewAction.REJECT, event_id="e2")
    third = reviews.export_training_examples("j", PREDICTIONS, tmp_path / "z.json")
    assert third["fingerprint"] != payload["fingerprint"]


def test_provenance_is_stored_with_the_decision(reviews):
    record = reviews.add(
        "j", ReviewAction.CONFIRM, event_id="e1",
        provenance={"run_fingerprint": "abc123", "models": {"ball_detector": "deadbeef"}},
    )
    assert record.to_dict()["provenance"]["run_fingerprint"] == "abc123"


# --------------------------------------------------------------------------- #
# Possession ground truth
# --------------------------------------------------------------------------- #


def _obj(track_id, role, team, px, py):
    return GSRObject(
        track_id=track_id, role=role, team=team,
        image_x=px * 10, image_y=py * 10, box_height=50.0,
        pitch_x=px, pitch_y=py,
    )


def _frames(spec, fps=25.0):
    """spec: list of (ball_xy | None, [(track, team, x, y), ...]) per frame."""
    out = {}
    for i, (ball, people) in enumerate(spec):
        objects = [_obj(t, "player", team, x, y) for t, team, x, y in people]
        if ball is not None:
            objects.append(_obj(99, "ball", None, ball[0], ball[1]))
        out[i] = objects
    return out, fps


def _derive(spec, params=None, fps=25.0):
    frames, fps = _frames(spec, fps)
    labels, previous = [], None
    params = params or DerivationParams()
    for idx in sorted(frames):
        label, holder, previous = _frame_label(frames[idx], previous, params, 105.0, 68.0)
        labels.append((idx, idx / fps, label, holder))
    return labels


def test_nearest_player_takes_possession():
    labels = _derive([((50.0, 34.0), [(1, "left", 50.5, 34.0), (2, "right", 60.0, 34.0)])])
    assert labels[0][2] is PossessionLabel.LEFT
    assert labels[0][3] == 1


def test_ball_beyond_the_control_radius_is_loose_not_owned():
    labels = _derive([((50.0, 34.0), [(1, "left", 55.0, 34.0), (2, "right", 60.0, 34.0)])])
    assert labels[0][2] is PossessionLabel.LOOSE
    assert labels[0][3] is None


def test_two_close_opponents_are_contested():
    labels = _derive([((50.0, 34.0), [(1, "left", 50.4, 34.0), (2, "right", 50.6, 34.0)])])
    assert labels[0][2] is PossessionLabel.CONTESTED
    assert labels[0][3] is None


def test_two_close_team_mates_are_not_contested():
    """Two players of one team near the ball is that team's possession."""
    labels = _derive([((50.0, 34.0), [(1, "left", 50.4, 34.0), (2, "left", 50.6, 34.0)])])
    assert labels[0][2] is PossessionLabel.LEFT


def test_ball_projected_off_the_pitch_is_unknown_not_loose():
    """An airborne ball's ground projection must not become a possession claim."""
    labels = _derive([((400.0, 34.0), [(1, "left", 50.0, 34.0), (2, "right", 60.0, 34.0)])])
    assert labels[0][2] is PossessionLabel.UNKNOWN


def test_teleporting_ball_is_unknown():
    """A step implying 40 m/s is a broken projection, not a fast ball."""
    labels = _derive([
        ((50.0, 34.0), [(1, "left", 50.2, 34.0), (2, "right", 60.0, 34.0)]),
        ((80.0, 34.0), [(1, "left", 80.2, 34.0), (2, "right", 60.0, 34.0)]),
    ])
    assert labels[0][2] is PossessionLabel.LEFT
    assert labels[1][2] is PossessionLabel.UNKNOWN


def test_missing_ball_is_unknown():
    labels = _derive([(None, [(1, "left", 50.0, 34.0), (2, "right", 60.0, 34.0)])])
    assert labels[0][2] is PossessionLabel.UNKNOWN


def test_flicker_shorter_than_the_minimum_becomes_unknown():
    """Two frames of possession inside a loose passage is not a possession."""
    params = DerivationParams(min_state_duration_s=0.20)  # 5 frames at 25 fps
    fps = 25.0
    labels = []
    for i in range(20):
        near = i in (10, 11)
        label = PossessionLabel.LEFT if near else PossessionLabel.LOOSE
        labels.append((i, i / fps, label, 1 if near else None))
    intervals = _to_intervals(labels, params, fps)
    kinds = {i.label for i in intervals}
    assert PossessionLabel.LEFT not in kinds
    assert PossessionLabel.UNKNOWN in kinds


def test_holder_change_does_not_break_a_team_possession():
    """The regression that halved measured team time.

    Keying the minimum-duration filter on (label, holder) chopped one team's
    possession into per-holder fragments, each below the threshold, and threw
    the whole passage away.
    """
    params = DerivationParams(min_state_duration_s=0.20)
    fps = 25.0
    labels = [
        (i, i / fps, PossessionLabel.LEFT, 1 if i < 3 else (2 if i < 6 else 3))
        for i in range(20)
    ]
    intervals = _to_intervals(labels, params, fps)
    assert all(i.label is PossessionLabel.LEFT for i in intervals)
    assert {i.holder_track_id for i in intervals} == {1, 2, 3}


def test_unknown_is_excluded_from_the_team_share():
    gt = PossessionGroundTruth(
        sequence="s", fps=25.0, duration_s=4.0,
        intervals=[
            PossessionInterval(0.0, 1.0, PossessionLabel.LEFT, 1, 25),
            PossessionInterval(1.0, 2.0, PossessionLabel.RIGHT, 2, 25),
            PossessionInterval(2.0, 4.0, PossessionLabel.UNKNOWN, None, 50),
        ],
        n_frames=100, n_frames_unknown=50,
    )
    share = gt.team_share()
    assert "unknown" not in share
    assert share["left"] == pytest.approx(0.5)
    assert gt.coverage == pytest.approx(0.5)


def test_possession_gt_file_detects_editing(tmp_path):
    gt = PossessionGroundTruth(
        sequence="s", fps=25.0, duration_s=1.0,
        intervals=[PossessionInterval(0.0, 1.0, PossessionLabel.LEFT, 1, 25)],
        n_frames=25,
    )
    path = gt.save(tmp_path / "p.json")
    assert PossessionGroundTruth.load(path).fingerprint() == gt.fingerprint()

    data = json.loads(path.read_text(encoding="utf-8"))
    data["intervals"][0]["label"] = "right"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="was edited"):
        PossessionGroundTruth.load(path)


# --------------------------------------------------------------------------- #
# Possession scoring
# --------------------------------------------------------------------------- #


def test_engine_state_maps_onto_reference_labels():
    assert _engine_label("controlled", "left") is PossessionLabel.LEFT
    assert _engine_label("controlled", "unknown") is PossessionLabel.UNKNOWN
    assert _engine_label("contested", "contested") is PossessionLabel.CONTESTED
    assert _engine_label("loose_ball", "none") is PossessionLabel.LOOSE
    assert _engine_label("out_of_play", "none") is PossessionLabel.LOOSE
    assert _engine_label("unknown", "left") is PossessionLabel.UNKNOWN


def test_unknown_reference_frames_leave_every_denominator_alone():
    gt = PossessionGroundTruth(
        sequence="s", fps=25.0, duration_s=2.0,
        intervals=[
            PossessionInterval(0.0, 1.0, PossessionLabel.LEFT, 1, 25),
            PossessionInterval(1.0, 2.0, PossessionLabel.UNKNOWN, None, 25),
        ],
        n_frames=50, n_frames_unknown=25,
    )
    predicted = {i: ("controlled", "left", 1) for i in range(50)}
    result = evaluate_possession_vs_gt(gt, predicted, 25.0, "test")

    assert result.n_scorable == 25
    assert result.per_label["left"]["tp"] == 25
    # The 25 unknown frames must not appear as wins or as losses anywhere.
    assert sum(
        c["tp"] + c["fp"] + c["fn"] for c in result.per_label.values()
    ) == 25


def test_holder_accuracy_only_counts_frames_with_a_reference_holder():
    gt = PossessionGroundTruth(
        sequence="s", fps=25.0, duration_s=1.0,
        intervals=[
            PossessionInterval(0.0, 0.5, PossessionLabel.LEFT, 7, 12),
            PossessionInterval(0.5, 1.0, PossessionLabel.LOOSE, None, 13),
        ],
        n_frames=25,
    )
    predicted = {i: ("controlled", "left", 7) for i in range(25)}
    result = evaluate_possession_vs_gt(gt, predicted, 25.0, "test")

    # [0.0, 0.5) at 25 fps is frames 0..12 inclusive, so 13 scored frames.
    assert result.holder_total == 13
    assert result.holder_accuracy == pytest.approx(1.0)
    # The loose half contributes nothing: no reference holder there.
    assert result.per_label["loose"]["fn"] == 12


def test_team_f1_is_macro_so_one_sided_clips_cannot_inflate_it():
    gt = PossessionGroundTruth(
        sequence="s", fps=25.0, duration_s=4.0,
        intervals=[
            PossessionInterval(0.0, 3.6, PossessionLabel.LEFT, 1, 90),
            PossessionInterval(3.6, 4.0, PossessionLabel.RIGHT, 2, 10),
        ],
        n_frames=100,
    )
    predicted = {i: ("controlled", "left", 1) for i in range(100)}
    result = evaluate_possession_vs_gt(gt, predicted, 25.0, "test")

    assert result.f1("left") > 0.9
    assert result.f1("right") == 0.0
    # A micro average would report ~0.9 here; macro reports the failure.
    assert result.team_f1 < 0.55


# --------------------------------------------------------------------------- #
# Team vocabulary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("team_id", ["A", "B", "left", "right", "home", "away"])
def test_real_team_labels_are_recognised(team_id):
    """The regression that silently disabled contest and pass/turnover logic.

    Decision rules used to test ``team_id in ("A", "B")``. Against any corpus
    with a different vocabulary that returned False everywhere, with no error:
    contested possession was never detected, passes were never separated from
    turnovers, and team profiles came back empty.
    """
    assert is_team(team_id)


@pytest.mark.parametrize("team_id", ["", "unknown", "none", "contested", None])
def test_sentinels_are_not_teams(team_id):
    assert not is_team(team_id)


def test_contest_needs_two_opponents_whatever_the_teams_are_called():
    """Contest detection must not depend on the classifier's naming."""
    from visionpitch.analytics.possession import PossessionConfig

    config = PossessionConfig()
    assert config.contest_margin_heights < config.control_radius_heights, (
        "a contest margin at or above the control radius makes the test vacuous"
    )


def test_aggregate_sums_counts_rather_than_averaging_rates():
    big = PossessionResult("big", "c", n_frames=1000, n_scorable=1000, n_predicted=1000,
                           per_label={"left": {"tp": 900, "fp": 0, "fn": 100}})
    small = PossessionResult("small", "c", n_frames=4, n_scorable=4, n_predicted=4,
                             per_label={"left": {"tp": 0, "fp": 4, "fn": 4}})
    pooled = aggregate([big, small], "c")

    assert pooled["n_scorable"] == 1004
    # Averaging the two F1 values would give ~0.45; pooling gives ~0.945.
    assert pooled["per_label"]["left"]["f1"] > 0.9
