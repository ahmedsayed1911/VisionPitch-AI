"""Broadcast ball annotation: schema, validation, storage and sampling.

The properties pinned here are the ones whose failure would quietly produce a
dataset that looks fine and measures nothing: a prediction leaking into the
answer key, a contradictory label being stored, a resumed session losing work,
or a "deterministic" sampler that is not.
"""

from __future__ import annotations

import json

import pytest

from visionpitch.annotation.broadcast_audit import (
    AuditResult,
    FrameFeatures,
    Shot,
    ShotType,
    build_shots,
    detect_cuts,
)
from visionpitch.annotation.sampler import (
    FrameSignal,
    SamplingPlan,
    build_samples,
)
from visionpitch.annotation.schema import (
    AnnotationError,
    AnnotationStore,
    BallAnnotation,
    BallVisibility,
    FrameSample,
    IgnoreReason,
    ModelPrediction,
    ReviewStatus,
    SamplingCategory,
    validate,
)


def sample(frame_id="f000010", frame_idx=10, width=1280, height=720, **kwargs):
    defaults = dict(
        timestamp_s=frame_idx / 50.0,
        image_path="",
        shot_index=0,
        shot_type="wide_play",
        sampling_category=SamplingCategory.MIDFIELD_PLAY,
        sampling_reason="test",
        is_live_play_candidate=True,
        likely_slow_motion=False,
        source_content_hash="hash",
        width=width,
        height=height,
    )
    defaults.update(kwargs)
    return FrameSample(frame_id=frame_id, frame_idx=frame_idx, **defaults)


# --------------------------------------------------------------------------- #
# Visibility semantics
# --------------------------------------------------------------------------- #


def test_the_four_kinds_of_absence_are_distinct():
    """Collapsing these would let a detector that never fires score perfectly."""
    assert BallVisibility.VISIBLE.requires_coordinates
    assert BallVisibility.NOT_VISIBLE.forbids_coordinates
    assert BallVisibility.OUTSIDE_FRAME.forbids_coordinates
    assert not BallVisibility.AMBIGUOUS.is_scorable
    for kind in (
        BallVisibility.VISIBLE, BallVisibility.NOT_VISIBLE, BallVisibility.OUTSIDE_FRAME
    ):
        assert kind.is_scorable


def test_ignored_frames_are_never_scorable():
    annotation = BallAnnotation(
        frame_id="f1", visibility=BallVisibility.VISIBLE,
        centre_x=10.0, centre_y=10.0, ignore_reason=IgnoreReason.REPLAY,
    )
    assert not annotation.is_scorable
    assert IgnoreReason.REPLAY.excludes_from_scoring
    assert not IgnoreReason.NONE.excludes_from_scoring


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_visible_without_coordinates_is_rejected():
    with pytest.raises(AnnotationError, match="requires a centre"):
        validate(
            BallAnnotation(frame_id="f000010", visibility=BallVisibility.VISIBLE),
            sample(),
        )


@pytest.mark.parametrize("kind", [BallVisibility.NOT_VISIBLE, BallVisibility.OUTSIDE_FRAME])
def test_absent_ball_with_coordinates_is_rejected(kind):
    """A coordinate here is a contradiction, not extra detail."""
    with pytest.raises(AnnotationError, match="cannot carry a centre"):
        validate(
            BallAnnotation(
                frame_id="f000010", visibility=kind, centre_x=5.0, centre_y=5.0
            ),
            sample(),
        )


@pytest.mark.parametrize("x,y", [(-1.0, 10.0), (1281.0, 10.0), (10.0, -1.0), (10.0, 721.0)])
def test_coordinates_outside_the_image_are_rejected(x, y):
    with pytest.raises(AnnotationError, match="outside"):
        validate(
            BallAnnotation(
                frame_id="f000010", visibility=BallVisibility.VISIBLE,
                centre_x=x, centre_y=y,
            ),
            sample(),
        )


def test_ambiguous_requires_a_reason():
    with pytest.raises(AnnotationError, match="requires a reason"):
        validate(
            BallAnnotation(frame_id="f000010", visibility=BallVisibility.AMBIGUOUS),
            sample(),
        )
    validate(
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.AMBIGUOUS,
            ambiguity_reason="ball hidden in a ruck",
        ),
        sample(),
    )


def test_a_box_implies_visibility():
    with pytest.raises(AnnotationError, match="implies the ball is visible"):
        validate(
            BallAnnotation(
                frame_id="f000010", visibility=BallVisibility.NOT_VISIBLE,
                bbox=[1.0, 1.0, 5.0, 5.0],
            ),
            sample(),
        )


def test_degenerate_box_is_rejected():
    with pytest.raises(AnnotationError, match="non-positive extent"):
        validate(
            BallAnnotation(
                frame_id="f000010", visibility=BallVisibility.VISIBLE,
                centre_x=10.0, centre_y=10.0, bbox=[5.0, 5.0, 5.0, 9.0],
            ),
            sample(),
        )


def test_frame_id_mismatch_is_rejected():
    with pytest.raises(AnnotationError, match="frame id mismatch"):
        validate(
            BallAnnotation(
                frame_id="f999999", visibility=BallVisibility.VISIBLE,
                centre_x=1.0, centre_y=1.0,
            ),
            sample(),
        )


# --------------------------------------------------------------------------- #
# Storage: separation, append safety, resume
# --------------------------------------------------------------------------- #


@pytest.fixture
def store(tmp_path):
    store = AnnotationStore(tmp_path)
    store.write_samples([sample("f000010", 10), sample("f000020", 20)])
    store.write_predictions([
        ModelPrediction("f000010", "box_detector", "abc", 100.0, 200.0, 0.9),
        ModelPrediction("f000010", "heatmap_detector", "def", 640.0, 360.0, 0.5),
    ])
    return store


def test_predictions_and_annotations_live_in_separate_files(store):
    """The rule the whole dataset rests on."""
    assert store.predictions_path != store.annotations_path
    store.append(
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.VISIBLE,
            centre_x=101.0, centre_y=201.0,
        ),
        store.load_samples()["f000010"],
    )
    predictions_text = store.predictions_path.read_text(encoding="utf-8")
    annotations_text = store.annotations_path.read_text(encoding="utf-8")
    assert '"record_type": "prediction"' in predictions_text
    assert '"record_type": "annotation"' not in predictions_text
    assert '"record_type": "annotation"' in annotations_text
    assert '"record_type": "prediction"' not in annotations_text


def test_writing_an_annotation_does_not_touch_predictions(store):
    before = store.predictions_path.read_bytes()
    store.append(
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.NOT_VISIBLE
        ),
        store.load_samples()["f000010"],
    )
    assert store.predictions_path.read_bytes() == before


def test_reannotating_keeps_the_earlier_decision_on_record(store):
    frame = store.load_samples()["f000010"]
    store.append(
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.VISIBLE,
            centre_x=100.0, centre_y=200.0, reviewer="first",
        ), frame,
    )
    store.append(
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.NOT_VISIBLE, reviewer="second",
        ), frame,
    )
    assert len(store.history()) == 2
    latest = store.load_annotations()["f000010"]
    assert latest.visibility is BallVisibility.NOT_VISIBLE
    assert latest.reviewer == "second"


def test_an_invalid_annotation_is_never_written(store):
    frame = store.load_samples()["f000010"]
    with pytest.raises(AnnotationError):
        store.append(
            BallAnnotation(frame_id="f000010", visibility=BallVisibility.VISIBLE),
            frame,
        )
    assert not store.annotations_path.exists() or not store.annotations_path.read_text(
        encoding="utf-8"
    ).strip()


def test_progress_survives_a_reopened_store(tmp_path):
    """Resume safety: a new process must see prior work."""
    first = AnnotationStore(tmp_path)
    first.write_samples([sample("f000010", 10), sample("f000020", 20)])
    first.append(
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.VISIBLE,
            centre_x=5.0, centre_y=5.0,
        ),
        first.load_samples()["f000010"],
    )
    reopened = AnnotationStore(tmp_path)
    progress = reopened.progress()
    assert progress["n_annotated"] == 1
    assert progress["n_remaining"] == 1


def test_duplicate_frame_ids_are_rejected(tmp_path):
    store = AnnotationStore(tmp_path)
    with pytest.raises(AnnotationError, match="duplicate frame id"):
        store.write_samples([sample("f000010", 10), sample("f000010", 11)])


def test_a_package_refuses_a_different_source_video(tmp_path):
    store = AnnotationStore(tmp_path)
    store.write_manifest({"source_content_hash": "aaaa1111"})
    store.assert_source_matches("aaaa1111")
    with pytest.raises(AnnotationError, match="was built from video"):
        store.assert_source_matches("bbbb2222")


def test_annotation_fingerprint_changes_with_content(store):
    frame = store.load_samples()["f000010"]
    before = store.fingerprint()
    store.append(
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.VISIBLE,
            centre_x=1.0, centre_y=1.0,
        ), frame,
    )
    assert store.fingerprint() != before


def test_accepted_proposal_is_recorded_as_such(store):
    """A dataset of accepted proposals is a model grading itself; make it visible."""
    frame = store.load_samples()["f000010"]
    store.append(
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.VISIBLE,
            centre_x=100.0, centre_y=200.0, accepted_proposal_from="box_detector",
        ), frame,
    )
    assert store.load_annotations()["f000010"].accepted_proposal_from == "box_detector"


def test_annotation_round_trips_through_json(store):
    annotation = BallAnnotation(
        frame_id="f000010", visibility=BallVisibility.AMBIGUOUS,
        ambiguity_reason="obscured", review_status=ReviewStatus.NEEDS_SECOND_REVIEW,
    )
    restored = BallAnnotation.from_dict(json.loads(json.dumps(annotation.to_dict())))
    assert restored.visibility is BallVisibility.AMBIGUOUS
    assert restored.review_status is ReviewStatus.NEEDS_SECOND_REVIEW
    assert restored.ambiguity_reason == "obscured"


# --------------------------------------------------------------------------- #
# Audit: cuts and shots
# --------------------------------------------------------------------------- #


def features_with_cuts(n=300, cuts=(100, 200)):
    out = []
    for i in range(n):
        distance = 0.9 if i in cuts else 0.01
        out.append(
            FrameFeatures(
                frame_idx=i, histogram_distance=distance, motion_px=2.0,
                blur_variance=100.0, edge_density=0.1, mean_saturation=80.0,
            )
        )
    return out


def test_cuts_are_found_at_histogram_jumps():
    cuts, threshold = detect_cuts(features_with_cuts())
    assert cuts == [100, 200]
    assert threshold > 0.01


def test_shots_tile_the_video_without_gaps_or_overlap():
    features = features_with_cuts()
    cuts, _ = detect_cuts(features)
    shots = build_shots(cuts, features, fps=50.0)
    assert shots[0].start_frame == 0
    assert shots[-1].end_frame == len(features) - 1
    for a, b in zip(shots, shots[1:], strict=False):
        assert b.start_frame == a.end_frame + 1


def test_close_ups_and_crowds_are_not_live_play():
    assert ShotType.WIDE_PLAY.is_live_play_candidate
    assert ShotType.MEDIUM_PLAY.is_live_play_candidate
    for kind in (ShotType.CLOSE_UP, ShotType.CROWD_OR_BENCH, ShotType.GRAPHIC):
        assert not kind.is_live_play_candidate


def test_audit_file_round_trips(tmp_path):
    audit = AuditResult(
        video_path="v.mp4", content_hash="abc", width=1280, height=720, fps=50.0,
        frame_count=300, duration_s=6.0, codec="h264",
        shots=[Shot(0, 0, 149, 50.0, ShotType.WIDE_PLAY),
               Shot(1, 150, 299, 50.0, ShotType.CROWD_OR_BENCH)],
    )
    path = audit.save(tmp_path / "audit.json")
    restored = AuditResult.load(path)
    assert len(restored.shots) == 2
    assert restored.shots[0].shot_type is ShotType.WIDE_PLAY
    assert restored.live_play_share == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def synthetic_audit(n_frames=6000, fps=50.0):
    shots = []
    frame = 0
    index = 0
    while frame < n_frames:
        length = 400
        kind = (
            ShotType.WIDE_PLAY if index % 3 == 0
            else ShotType.MEDIUM_PLAY if index % 3 == 1
            else ShotType.CROWD_OR_BENCH
        )
        shots.append(
            Shot(index, frame, min(n_frames - 1, frame + length - 1), fps, kind)
        )
        frame += length
        index += 1
    return AuditResult(
        video_path="v.mp4", content_hash="abc", width=1280, height=720, fps=fps,
        frame_count=n_frames, duration_s=n_frames / fps, codec="h264", shots=shots,
    )


def synthetic_signals(audit):
    rng = __import__("random").Random(0)
    signals = []
    for shot in audit.shots:
        for idx in range(shot.start_frame, shot.end_frame + 1):
            signals.append(
                FrameSignal(
                    frame_idx=idx,
                    motion_px=rng.uniform(0, 20),
                    blur_variance=rng.uniform(20, 400),
                    edge_density=rng.uniform(0.02, 0.3),
                    saturation=rng.uniform(40, 140),
                    shot_index=shot.index,
                    shot_type=shot.shot_type,
                    likely_slow_motion=False,
                    n_people=rng.randint(0, 14),
                )
            )
    return signals


def test_sampling_is_deterministic():
    audit = synthetic_audit()
    signals = synthetic_signals(audit)
    plan = SamplingPlan(total_frames=120, seed=7)
    first = build_samples(audit, signals, plan)
    second = build_samples(audit, signals, plan)
    assert [s.frame_idx for s in first.samples] == [s.frame_idx for s in second.samples]
    assert first.fingerprint() == second.fingerprint()


def test_a_different_seed_gives_a_different_sample():
    audit = synthetic_audit()
    signals = synthetic_signals(audit)
    a = build_samples(audit, signals, SamplingPlan(total_frames=120, seed=1))
    b = build_samples(audit, signals, SamplingPlan(total_frames=120, seed=2))
    assert a.fingerprint() != b.fingerprint()


def test_sampling_shares_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SamplingPlan(stratified_live_share=0.9, temporal_window_share=0.9).counts()


def test_temporal_windows_are_consecutive_runs():
    audit = synthetic_audit()
    signals = synthetic_signals(audit)
    result = build_samples(audit, signals, SamplingPlan(total_frames=200, seed=3))
    windows: dict[str, list[int]] = {}
    for s in result.samples:
        if s.window_id:
            windows.setdefault(s.window_id, []).append(s.frame_idx)
    assert windows, "no temporal windows were produced"
    for frames in windows.values():
        frames.sort()
        assert frames == list(range(frames[0], frames[0] + len(frames)))


def test_negatives_are_drawn_from_non_live_shots():
    audit = synthetic_audit()
    signals = synthetic_signals(audit)
    result = build_samples(audit, signals, SamplingPlan(total_frames=200, seed=5))
    negatives = [
        s for s in result.samples
        if s.sampling_category in (
            SamplingCategory.CROWD_NEGATIVE, SamplingCategory.CLOSE_UP_NEGATIVE,
            SamplingCategory.BROADCAST_GRAPHIC,
        )
    ]
    assert negatives, "no negative frames were sampled"
    assert all(not s.is_live_play_candidate for s in negatives)


def test_non_window_frames_are_kept_apart():
    """Guards against hundreds of near-identical consecutive frames."""
    audit = synthetic_audit()
    signals = synthetic_signals(audit)
    plan = SamplingPlan(total_frames=150, seed=11, min_separation_frames=20)
    result = build_samples(audit, signals, plan)
    loose = sorted(s.frame_idx for s in result.samples if not s.window_id)
    for a, b in zip(loose, loose[1:], strict=False):
        assert b - a >= plan.min_separation_frames


# --------------------------------------------------------------------------- #
# Ball radius
# --------------------------------------------------------------------------- #


def test_bbox_is_derived_from_centre_and_radius():
    """One measurement, two views -- two stored fields would eventually disagree."""
    annotation = BallAnnotation(
        frame_id="f000010", visibility=BallVisibility.VISIBLE,
        centre_x=100.0, centre_y=50.0, radius_px=6.0,
    )
    assert annotation.bbox == [94.0, 44.0, 106.0, 56.0]
    assert annotation.diameter_px == pytest.approx(12.0)


def test_an_explicit_bbox_is_not_overwritten_by_the_radius():
    annotation = BallAnnotation(
        frame_id="f000010", visibility=BallVisibility.VISIBLE,
        centre_x=100.0, centre_y=50.0, radius_px=6.0,
        bbox=[1.0, 2.0, 3.0, 4.0],
    )
    assert annotation.bbox == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("radius", [1.0, 4.5, 7.0, 28.0, 60.0])
def test_the_full_radius_range_is_accepted(radius):
    """Very small far-side balls through to large close-shot balls."""
    validate(
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.VISIBLE,
            centre_x=100.0, centre_y=100.0, radius_px=radius,
        ),
        sample(),
    )


def test_a_sub_pixel_radius_is_rejected():
    from visionpitch.annotation.schema import MIN_BALL_RADIUS_PX

    with pytest.raises(AnnotationError, match="below the"):
        validate(
            BallAnnotation(
                frame_id="f000010", visibility=BallVisibility.VISIBLE,
                centre_x=100.0, centre_y=100.0,
                radius_px=MIN_BALL_RADIUS_PX / 2,
            ),
            sample(),
        )


def test_an_absurd_radius_is_rejected_as_a_stray_drag():
    from visionpitch.annotation.schema import MAX_BALL_RADIUS_PX

    with pytest.raises(AnnotationError, match="above the"):
        validate(
            BallAnnotation(
                frame_id="f000010", visibility=BallVisibility.VISIBLE,
                centre_x=100.0, centre_y=100.0,
                radius_px=MAX_BALL_RADIUS_PX + 1,
            ),
            sample(),
        )


@pytest.mark.parametrize("kind", [
    BallVisibility.NOT_VISIBLE, BallVisibility.OUTSIDE_FRAME, BallVisibility.AMBIGUOUS
])
def test_a_radius_without_a_visible_ball_is_rejected(kind):
    annotation = BallAnnotation(
        frame_id="f000010", visibility=kind, radius_px=5.0,
        ambiguity_reason="x" if kind is BallVisibility.AMBIGUOUS else "",
    )
    with pytest.raises(AnnotationError, match="implies the ball is visible"):
        validate(annotation, sample())


def test_radius_survives_a_json_round_trip():
    original = BallAnnotation(
        frame_id="f000010", visibility=BallVisibility.VISIBLE,
        centre_x=100.0, centre_y=50.0, radius_px=4.5,
    )
    restored = BallAnnotation.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.radius_px == pytest.approx(4.5)
    assert restored.bbox == original.bbox


def test_annotations_written_before_the_radius_field_still_load():
    """Schema backward compatibility: 1.0.0 rows carry no radius and must not break."""
    legacy = {
        "frame_id": "f000010", "visibility": "visible", "ignore_reason": "none",
        "centre_x": 10.0, "centre_y": 10.0, "bbox": None,
        "annotation_confidence": 1.0, "ambiguity_reason": "", "reviewer": "old",
        "review_status": "first_pass", "reviewed_at": "2026-01-01T00:00:00+00:00",
        "accepted_proposal_from": None, "schema_version": "1.0.0",
        "record_type": "annotation",
    }
    restored = BallAnnotation.from_dict(legacy)
    assert restored.radius_px is None
    assert restored.bbox is None
    assert restored.schema_version == "1.0.0"
    validate(restored, sample())


def test_default_radius_matches_this_video_s_measured_ball_size():
    """7 px comes from the median 14.15 px proposal on the annotated broadcast."""
    from visionpitch.annotation.schema import (
        DEFAULT_BALL_RADIUS_PX,
        MAX_BALL_RADIUS_PX,
        MIN_BALL_RADIUS_PX,
    )

    assert DEFAULT_BALL_RADIUS_PX == pytest.approx(7.0)
    assert MIN_BALL_RADIUS_PX < DEFAULT_BALL_RADIUS_PX < MAX_BALL_RADIUS_PX


def test_radius_is_stored_through_the_store(tmp_path):
    store = AnnotationStore(tmp_path)
    store.write_samples([sample("f000010", 10)])
    store.append(
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.VISIBLE,
            centre_x=200.0, centre_y=150.0, radius_px=3.5,
        ),
        store.load_samples()["f000010"],
    )
    reloaded = AnnotationStore(tmp_path).load_annotations()["f000010"]
    assert reloaded.radius_px == pytest.approx(3.5)
    assert reloaded.bbox == [196.5, 146.5, 203.5, 153.5]


# --------------------------------------------------------------------------- #
# QC focused review queue
# --------------------------------------------------------------------------- #


def qc_samples(spec):
    """spec: list of (frame_id, category, window_id or None)."""
    return {
        fid: sample(fid, int(fid[1:]), sampling_category=SamplingCategory(cat),
                    window_id=win)
        for fid, cat, win in spec
    }


@pytest.mark.parametrize("visibility,ignore,expected", [
    (BallVisibility.NOT_VISIBLE, IgnoreReason.NONE, True),
    (BallVisibility.OUTSIDE_FRAME, IgnoreReason.NONE, True),
    (BallVisibility.VISIBLE, IgnoreReason.REPLAY, True),
    (BallVisibility.VISIBLE, IgnoreReason.NON_LIVE, True),
    (BallVisibility.VISIBLE, IgnoreReason.NONE, False),
    (BallVisibility.AMBIGUOUS, IgnoreReason.NONE, False),
])
def test_genuine_negative_is_defined_by_the_human_decision(
    visibility, ignore, expected
):
    from visionpitch.annotation.qc import is_genuine_negative

    kwargs = {}
    if visibility is BallVisibility.VISIBLE:
        kwargs = {"centre_x": 10.0, "centre_y": 10.0}
    if visibility is BallVisibility.AMBIGUOUS:
        kwargs = {"ambiguity_reason": "unclear"}
    annotation = BallAnnotation(
        frame_id="f000010", visibility=visibility, ignore_reason=ignore, **kwargs
    )
    assert is_genuine_negative(annotation) is expected


def test_a_sampler_guess_of_crowd_is_not_a_negative():
    """The audit's central lesson: 19 of 19 'crowd negatives' were real play."""
    from visionpitch.annotation.qc import build_qc_queue, is_genuine_negative

    samples = qc_samples([("f000010", "crowd_negative", None)])
    annotations = {
        "f000010": BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.VISIBLE,
            centre_x=5.0, centre_y=5.0,
        )
    }
    assert not is_genuine_negative(annotations["f000010"])
    queue = build_qc_queue(samples, annotations, target_total=10)
    assert queue.negative_quota.achieved == 0


def test_queue_contains_only_unreviewed_frames():
    from visionpitch.annotation.qc import build_qc_queue

    samples = qc_samples([
        ("f000010", "motion_blur", None), ("f000020", "motion_blur", None),
    ])
    annotations = {
        "f000010": BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.VISIBLE,
            centre_x=1.0, centre_y=1.0,
        )
    }
    queue = build_qc_queue(samples, annotations, target_total=10)
    assert "f000010" not in queue.frame_ids
    assert "f000020" in queue.frame_ids


def test_queue_follows_the_declared_priority_order():
    from visionpitch.annotation.qc import PRIORITY, build_qc_queue

    samples = qc_samples([
        ("f000010", "camera_pan", None),
        ("f000020", "crowd_negative", None),
        ("f000030", "motion_blur", None),
        ("f000040", "broadcast_graphic", None),
    ])
    queue = build_qc_queue(samples, {}, target_total=10)
    order = [samples[f].sampling_category.value for f in queue.frame_ids]
    ranks = [PRIORITY.index(c) for c in order if c in PRIORITY]
    assert ranks == sorted(ranks), order


def test_partially_reviewed_windows_are_finished_before_fresh_ones():
    """Completing a 6/7 window costs one frame; a fresh one costs seven."""
    from visionpitch.annotation.qc import build_qc_queue

    spec = []
    for w, prefix in (("wA", 100), ("wB", 200)):
        for i in range(7):
            spec.append((f"f000{prefix + i:03d}", "temporal_window", w))
    samples = qc_samples(spec)
    # wA is 6/7 done, wB untouched.
    annotations = {
        f"f000{100 + i:03d}": BallAnnotation(
            frame_id=f"f000{100 + i:03d}", visibility=BallVisibility.VISIBLE,
            centre_x=1.0, centre_y=1.0,
        )
        for i in range(6)
    }
    queue = build_qc_queue(samples, annotations, target_total=3)
    assert queue.frame_ids[0] == "f000106", queue.frame_ids


def test_quota_reports_when_a_target_cannot_be_reached_from_this_package():
    from visionpitch.annotation.qc import build_qc_queue

    samples = qc_samples([("f000010", "motion_blur", None)])
    queue = build_qc_queue(samples, {}, target_total=5)
    blur = queue.quotas["motion_blur"]
    assert blur.target == 25
    assert blur.achieved == 0
    assert not blur.reachable


def test_already_achieved_counts_reduce_the_deficit():
    from visionpitch.annotation.qc import build_qc_queue

    spec = [(f"f000{i:03d}", "near_goal", None) for i in range(1, 40)]
    samples = qc_samples(spec)
    annotations = {
        f"f000{i:03d}": BallAnnotation(
            frame_id=f"f000{i:03d}", visibility=BallVisibility.VISIBLE,
            centre_x=1.0, centre_y=1.0,
        )
        for i in range(1, 21)
    }
    queue = build_qc_queue(samples, annotations, target_total=200)
    near_goal = queue.quotas["near_goal"]
    assert near_goal.achieved == 20
    assert near_goal.deficit == 5


def test_negative_quota_counts_outcomes_from_any_category():
    from visionpitch.annotation.qc import build_qc_queue

    samples = qc_samples([
        ("f000010", "midfield_play", None), ("f000020", "crowd_negative", None),
    ])
    annotations = {
        # A midfield frame where the reviewer could not see the ball still
        # counts as a negative.
        "f000010": BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.NOT_VISIBLE
        ),
        "f000020": BallAnnotation(
            frame_id="f000020", visibility=BallVisibility.VISIBLE,
            centre_x=1.0, centre_y=1.0,
        ),
    }
    queue = build_qc_queue(samples, annotations, target_total=10)
    assert queue.negative_quota.achieved == 1


def test_qc_mode_serves_only_the_queue(tmp_path):
    from fastapi.testclient import TestClient

    from visionpitch.annotation.server import create_app

    store = AnnotationStore(tmp_path)
    store.write_samples(list(qc_samples([
        ("f000010", "motion_blur", None),
        ("f000020", "midfield_play", None),
        ("f000030", "crowd_negative", None),
    ]).values()))
    store.write_predictions([])
    store.write_manifest({"source_content_hash": "abc"})

    full = TestClient(create_app(tmp_path), raise_server_exceptions=False)
    assert full.get("/api/frames").json()["n"] == 3

    focused = TestClient(
        create_app(tmp_path, qc=True, qc_total=10), raise_server_exceptions=False
    )
    served = [f["frame_id"] for f in focused.get("/api/frames").json()["frames"]]
    # crowd_negative outranks motion_blur, which outranks unlisted midfield.
    assert served[0] == "f000030"
    assert "f000010" in served
    assert focused.get("/api/progress").json()["qc_mode"] is True
    assert full.get("/api/progress").json()["qc_mode"] is False


def test_qc_endpoint_exposes_live_quota_state(tmp_path):
    from fastapi.testclient import TestClient

    from visionpitch.annotation.server import create_app

    store = AnnotationStore(tmp_path)
    store.write_samples(list(qc_samples([
        ("f000010", "motion_blur", None), ("f000020", "crowd_negative", None),
    ]).values()))
    store.write_predictions([])
    client = TestClient(
        create_app(tmp_path, qc=True, qc_total=10), raise_server_exceptions=False
    )
    payload = client.get("/api/qc").json()
    assert payload["negative_quota"]["target"] == 25
    assert "not_visible" in payload["negative_definition"]
    assert payload["temporal_windows"]["target_complete"] == 3


# --------------------------------------------------------------------------- #
# Quality control queue
# --------------------------------------------------------------------------- #


def queue_fixture(annotations, predictions=None, window="w00"):
    from visionpitch.annotation.quality import build_queue

    samples = {
        a.frame_id: sample(
            a.frame_id, int(a.frame_id[1:]), window_id=window
        )
        for a in annotations
    }
    return build_queue(
        samples,
        {a.frame_id: a for a in annotations},
        predictions or {},
    )


def visible(frame_id, x, y, **kwargs):
    return BallAnnotation(
        frame_id=frame_id, visibility=BallVisibility.VISIBLE,
        centre_x=x, centre_y=y, **kwargs
    )


def test_ambiguous_frames_reach_the_queue():
    from visionpitch.annotation.quality import QualityFlag

    queue = queue_fixture([
        BallAnnotation(
            frame_id="f000010", visibility=BallVisibility.AMBIGUOUS,
            ambiguity_reason="obscured",
        )
    ])
    assert queue and QualityFlag.AMBIGUOUS in queue[0].flags


def test_a_physically_impossible_step_is_flagged():
    from visionpitch.annotation.quality import QualityFlag

    queue = queue_fixture([
        visible("f000010", 100.0, 100.0),
        visible("f000011", 900.0, 600.0),
    ])
    flagged = {item.frame_id for item in queue if QualityFlag.TEMPORAL_JUMP in item.flags}
    assert flagged == {"f000010", "f000011"}


def test_smooth_motion_is_not_flagged():
    from visionpitch.annotation.quality import QualityFlag

    queue = queue_fixture([
        visible("f000010", 100.0, 100.0),
        visible("f000011", 112.0, 104.0),
        visible("f000012", 124.0, 108.0),
    ])
    assert not any(QualityFlag.TEMPORAL_JUMP in item.flags for item in queue)


def test_an_isolated_visible_frame_is_flagged():
    from visionpitch.annotation.quality import QualityFlag

    queue = queue_fixture([
        BallAnnotation(frame_id="f000010", visibility=BallVisibility.NOT_VISIBLE),
        visible("f000011", 400.0, 300.0),
        BallAnnotation(frame_id="f000012", visibility=BallVisibility.NOT_VISIBLE),
    ])
    flagged = {i.frame_id for i in queue if QualityFlag.ISOLATED_POSITION in i.flags}
    assert flagged == {"f000011"}


def test_disagreement_needs_every_model_to_disagree():
    from visionpitch.annotation.quality import QualityFlag

    annotation = visible("f000010", 100.0, 100.0)
    # One model far away, one model on the point: not a disagreement.
    queue = queue_fixture(
        [annotation],
        {"f000010": [
            ModelPrediction("f000010", "box_detector", "a", 900.0, 900.0, 0.9),
            ModelPrediction("f000010", "heatmap_detector", "b", 101.0, 101.0, 0.7),
        ]},
    )
    assert not any(
        QualityFlag.DETECTOR_DISAGREEMENT in item.flags for item in queue
    )

    queue = queue_fixture(
        [annotation],
        {"f000010": [
            ModelPrediction("f000010", "box_detector", "a", 900.0, 900.0, 0.9),
            ModelPrediction("f000010", "heatmap_detector", "b", 800.0, 800.0, 0.7),
        ]},
    )
    assert any(QualityFlag.DETECTOR_DISAGREEMENT in item.flags for item in queue)


def test_accepted_proposals_are_flagged_for_audit():
    from visionpitch.annotation.quality import QualityFlag

    queue = queue_fixture([
        visible("f000010", 100.0, 100.0, accepted_proposal_from="box_detector")
    ])
    assert QualityFlag.ACCEPTED_PROPOSAL_ONLY in queue[0].flags


def test_queue_is_ranked_by_priority():
    from visionpitch.annotation.quality import build_queue

    annotations = [
        visible("f000010", 100.0, 100.0),
        visible("f000011", 900.0, 600.0),
        BallAnnotation(
            frame_id="f000012", visibility=BallVisibility.AMBIGUOUS,
            ambiguity_reason="unclear",
        ),
    ]
    samples = {
        a.frame_id: sample(a.frame_id, int(a.frame_id[1:]), window_id="w00")
        for a in annotations
    }
    queue = build_queue(samples, {a.frame_id: a for a in annotations}, {})
    priorities = [item.priority for item in queue]
    assert priorities == sorted(priorities, reverse=True)


def test_every_sample_records_why_it_was_chosen():
    audit = synthetic_audit()
    signals = synthetic_signals(audit)
    result = build_samples(audit, signals, SamplingPlan(total_frames=100, seed=13))
    assert result.samples
    for s in result.samples:
        assert s.sampling_reason
        assert isinstance(s.sampling_category, SamplingCategory)
