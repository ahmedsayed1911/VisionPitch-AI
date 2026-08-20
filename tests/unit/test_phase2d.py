"""Phase 2D: observability, track-before-detect recovery, and the promotion rule.

The properties pinned here are the ones whose failure would silently inflate a
coverage number: a recovery accepted on one frame of noise, a gap longer than
the limit quietly filled, a recovered position counted as a sighting, or a
promotion granted on evidence nobody collected.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionpitch.analytics.types import BallStateKind
from visionpitch.ball_tracking.observability import (
    FrameObservability,
    Observability,
    ObservabilityConfig,
    ObservabilityEstimator,
    ObservabilityReport,
)
from visionpitch.ball_tracking.recovery import (
    RecoveryCandidate,
    RecoveryConfig,
    RecoveryMethod,
    TrackBeforeDetect,
)
from visionpitch.evaluation.ball_failures import (
    FailureCategory,
    FrameEvidence,
    GroundTruthBall,
    classify_miss,
    size_bucket,
)
from visionpitch.evaluation.promotion import (
    CandidateMeasurements,
    PromotionCriteria,
    evaluate_promotion,
)

FRAME_SIZE = (1920, 1080)


# --------------------------------------------------------------------------- #
# Ball state kinds
# --------------------------------------------------------------------------- #


def test_recovered_is_known_but_not_direct():
    """The distinction the whole recovery stage rests on."""
    assert BallStateKind.RECOVERED.is_known
    assert not BallStateKind.RECOVERED.is_direct
    assert BallStateKind.OBSERVED.is_direct


def test_only_observed_and_recovered_have_image_evidence():
    assert BallStateKind.OBSERVED.has_image_evidence
    assert BallStateKind.RECOVERED.has_image_evidence
    # An interpolation is a claim about physics between sightings, not about
    # anything visible in the frame it fills.
    assert not BallStateKind.INTERPOLATED.has_image_evidence
    assert not BallStateKind.UNKNOWN.has_image_evidence


# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #


def estimator(**kwargs):
    return ObservabilityEstimator(ObservabilityConfig(**kwargs))


def label(expected, players=None, motion=0.0, calibration=1.0, keypoints=32):
    return estimator().label_frame(
        frame_idx=1,
        frame_size=FRAME_SIZE,
        expected=expected,
        player_boxes=np.array(players) if players else np.zeros((0, 4)),
        camera_motion_px=motion,
        calibration_confidence=calibration,
        n_pitch_keypoints=keypoints,
        frames_since_observation=1,
    ).state


def test_clear_expected_position_is_visible():
    assert label((960, 540)) is Observability.LIKELY_VISIBLE


def test_expected_position_past_the_frame_edge_is_outside_frame():
    assert label((5, 540)) is Observability.LIKELY_OUTSIDE_FRAME
    assert label((1915, 540)) is Observability.LIKELY_OUTSIDE_FRAME
    assert label((960, 2)) is Observability.LIKELY_OUTSIDE_FRAME


def test_expected_position_inside_a_player_box_is_occluded():
    assert label((960, 540), players=[[900, 480, 1020, 700]]) is (
        Observability.LIKELY_OCCLUDED
    )


def test_a_crowd_hides_the_ball_without_containing_it():
    players = [[960 + dx, 540 + dy, 990 + dx, 640 + dy]
               for dx, dy in [(40, 0), (-60, 10), (30, 40), (-40, -30), (70, 20)]]
    assert label((960, 540), players=players) is Observability.LIKELY_HIDDEN_BY_PLAYERS


def test_no_pitch_in_shot_is_not_a_detector_failure():
    assert label((960, 540), calibration=0.0, keypoints=0) is Observability.NOT_ON_PITCH


def test_fast_camera_motion_is_blur():
    assert label((960, 540), motion=40.0) is Observability.LIKELY_MOTION_BLURRED


def test_without_an_expectation_the_answer_is_uncertain_not_visible():
    """Absence of evidence must not be reported as evidence of visibility."""
    assert label(None) is Observability.UNCERTAIN


@pytest.mark.parametrize("state,fair", [
    (Observability.LIKELY_VISIBLE, True),
    (Observability.LIKELY_MOTION_BLURRED, True),
    (Observability.UNCERTAIN, True),
    (Observability.LIKELY_OCCLUDED, False),
    (Observability.LIKELY_OUTSIDE_FRAME, False),
    (Observability.LIKELY_HIDDEN_BY_PLAYERS, False),
    (Observability.NOT_ON_PITCH, False),
])
def test_only_plausibly_visible_frames_count_against_the_detector(state, fair):
    assert state.is_fair_denominator is fair


def test_out_of_frame_is_not_scored_as_a_miss():
    report = ObservabilityReport()
    report.frames = {
        1: FrameObservability(1, Observability.LIKELY_VISIBLE),
        2: FrameObservability(2, Observability.LIKELY_OUTSIDE_FRAME),
        3: FrameObservability(3, Observability.NOT_ON_PITCH),
        4: FrameObservability(4, Observability.LIKELY_VISIBLE),
    }
    summary = report.summary(observed_frames={1})

    assert summary["n_frames"] == 4
    assert summary["n_observable_frames"] == 2
    # Raw coverage punishes the detector for the replay and the out-of-frame
    # ball; conditioned coverage does not.
    assert summary["raw_ball_coverage"] == pytest.approx(0.25)
    assert summary["observability_conditioned_coverage"] == pytest.approx(0.5)


def test_a_camera_cut_resets_the_expectation():
    """After a cut, the old trajectory says nothing about the new shot."""
    frames = list(range(10))
    observations = {0: (100.0, 100.0), 1: (110.0, 100.0)}
    report = estimator().label_sequence(
        frame_indices=frames,
        frame_size=FRAME_SIZE,
        ball_observations=observations,
        player_boxes_by_frame={},
        cut_frames={5},
    )
    # Frames 2-4 still extrapolate from the pre-cut sighting.
    assert report.state_of(2) is Observability.LIKELY_VISIBLE
    # From the cut onward there is no valid expectation, so the honest answer
    # is uncertain rather than a confident label about a stale position.
    assert report.state_of(5) is Observability.UNCERTAIN
    assert report.state_of(6) is Observability.UNCERTAIN


def test_expectation_expires_rather_than_drifting_forever():
    report = estimator(max_extrapolation_frames=3).label_sequence(
        frame_indices=list(range(12)),
        frame_size=FRAME_SIZE,
        ball_observations={0: (500.0, 500.0), 1: (510.0, 500.0)},
        player_boxes_by_frame={},
    )
    assert report.state_of(3) is Observability.LIKELY_VISIBLE
    assert report.state_of(9) is Observability.UNCERTAIN


def test_observed_frames_are_labelled_visible_with_full_confidence():
    report = estimator().label_sequence(
        frame_indices=[0, 1],
        frame_size=FRAME_SIZE,
        ball_observations={0: (400.0, 400.0)},
        player_boxes_by_frame={},
    )
    assert report.frames[0].state is Observability.LIKELY_VISIBLE
    assert report.frames[0].confidence == 1.0
    assert report.frames[0].frames_since_observation == 0


def test_observability_never_returns_a_ball_position():
    """The module's central safety property.

    Nothing it produces may be mistaken for a ball coordinate; if it could, the
    fabrication the trajectory estimator refuses would re-enter through here.
    """
    report = estimator().label_sequence(
        frame_indices=[0, 1, 2],
        frame_size=FRAME_SIZE,
        ball_observations={0: (400.0, 400.0)},
        player_boxes_by_frame={},
    )
    for entry in report.frames.values():
        payload = entry.to_dict()
        assert "position" not in payload
        assert not any(
            isinstance(v, (tuple, list)) for v in payload.values()
        ), payload


# --------------------------------------------------------------------------- #
# Track-before-detect
# --------------------------------------------------------------------------- #


def candidate(frame_idx, x, y, ratio=9.0, deviation=4.0):
    return RecoveryCandidate(
        frame_idx=frame_idx, position=(x, y), method=RecoveryMethod.FRAME_DIFFERENCE,
        response=60.0, response_ratio=ratio, deviation_px=deviation, blob_area_px2=30.0,
    )


def test_one_frame_of_evidence_is_never_a_recovery():
    """The rule that stops this stage undoing the trajectory search."""
    recoverer = TrackBeforeDetect(RecoveryConfig(min_supporting_frames=3))
    assert recoverer._confirm([candidate(5, 100, 100)]) == []
    assert recoverer._confirm([candidate(5, 100, 100), candidate(6, 105, 100)]) == []


def test_three_consistent_frames_are_accepted():
    recoverer = TrackBeforeDetect(RecoveryConfig(min_supporting_frames=3))
    confirmed = recoverer._confirm([
        candidate(5, 100, 100), candidate(6, 108, 101), candidate(7, 116, 102),
    ])
    assert len(confirmed) == 3
    assert all(c.supporting_frames == [5, 6, 7] for c in confirmed)
    assert all(c.trajectory_consistency > 0.5 for c in confirmed)


def test_a_blob_jumping_around_the_window_is_rejected():
    """Strong per-frame response is not enough; the motion must be plausible."""
    recoverer = TrackBeforeDetect(RecoveryConfig(min_supporting_frames=3, max_step_px=20))
    confirmed = recoverer._confirm([
        candidate(5, 100, 100), candidate(6, 300, 400), candidate(7, 120, 90),
    ])
    assert confirmed == []


def test_non_consecutive_evidence_does_not_form_a_run():
    recoverer = TrackBeforeDetect(RecoveryConfig(min_supporting_frames=3))
    confirmed = recoverer._confirm([
        candidate(5, 100, 100), candidate(7, 108, 101), candidate(9, 116, 102),
    ])
    assert confirmed == []


def test_recovery_confidence_is_capped_below_the_detector():
    recoverer = TrackBeforeDetect(RecoveryConfig(min_supporting_frames=3, max_confidence=0.45))
    confirmed = recoverer._confirm([
        candidate(5, 100, 100, ratio=999.0, deviation=0.0),
        candidate(6, 101, 100, ratio=999.0, deviation=0.0),
        candidate(7, 102, 100, ratio=999.0, deviation=0.0),
    ])
    assert confirmed and all(c.confidence <= 0.45 for c in confirmed)


def test_long_gaps_are_refused_outright():
    """The anti-hallucination guarantee: a long disappearance stays unknown."""
    recoverer = TrackBeforeDetect(RecoveryConfig(max_gap_frames=15))
    gap = list(range(100, 140))  # 40 frames
    called = []

    def accessor(frame_idx):
        called.append(frame_idx)
        return np.zeros((100, 100), dtype=np.uint8)

    assert recoverer.recover(gap, accessor, {i: (50.0, 50.0) for i in gap}) == []
    # It must not even look: searching a long gap wastes time and invites noise.
    assert called == []


def test_unobservable_frames_are_skipped_not_searched():
    """Frames the observability model rules out are never search centres.

    Read access alone does not prove a frame was searched: searching frame *n*
    also reads *n-1* and *n+1* for three-frame differencing. What identifies a
    search centre is that its own neighbours were read -- so if frame 9 was
    never read, frame 10 was never searched.
    """
    recoverer = TrackBeforeDetect(RecoveryConfig())
    gap = [10, 11, 12]
    looked_at = []

    def accessor(frame_idx):
        looked_at.append(frame_idx)
        return np.zeros((200, 200), dtype=np.uint8)

    confirmed = recoverer.recover(
        gap, accessor, {i: (100.0, 100.0) for i in gap},
        observability={
            10: "likely_outside_frame", 11: "not_on_pitch", 12: "likely_visible",
        },
    )
    assert 9 not in looked_at, "frame 10 was searched despite being out of frame"
    assert 10 not in looked_at, "frame 11 was searched despite being off pitch"
    assert all(o.frame_idx == 12 for o in confirmed)


def test_recovery_finds_a_real_moving_blob():
    """End-to-end on synthetic frames with a known moving dot."""
    recoverer = TrackBeforeDetect(
        RecoveryConfig(min_supporting_frames=3, min_response_ratio=2.0)
    )
    size = 240
    rng = np.random.default_rng(0)
    background = (rng.integers(60, 70, (size, size))).astype(np.uint8)

    def frame_for(frame_idx):
        if not 0 <= frame_idx <= 20:
            return None
        image = background.copy()
        x = 60 + 4 * frame_idx
        cv_y = 120
        image[cv_y - 3:cv_y + 3, x - 3:x + 3] = 250
        return image

    gap = [10, 11, 12, 13]
    hints = {i: (60.0 + 4 * i, 120.0) for i in gap}
    confirmed = recoverer.recover(gap, frame_for, hints)

    assert len(confirmed) >= 3
    for observation in confirmed:
        expected_x = 60 + 4 * observation.frame_idx
        assert abs(observation.position[0] - expected_x) < 12
        assert abs(observation.position[1] - 120) < 12


def test_recovery_finds_nothing_in_a_static_scene():
    """No motion means no evidence, and no evidence must mean no output."""
    recoverer = TrackBeforeDetect(RecoveryConfig(min_supporting_frames=3))
    static = np.full((200, 200), 80, dtype=np.uint8)
    gap = [5, 6, 7, 8]
    confirmed = recoverer.recover(
        gap, lambda i: static.copy(), {i: (100.0, 100.0) for i in gap}
    )
    assert confirmed == []


# --------------------------------------------------------------------------- #
# Failure taxonomy
# --------------------------------------------------------------------------- #


def evidence(**kwargs):
    defaults = dict(
        blur_variance=500.0, contrast=0.9, inside_player_box=False,
        nearest_player_px=500.0, line_strength=0.0, touches_frame_edge=False,
        floor_distance_px=9999.0, floor_confidence=0.0, match_px=25.0,
    )
    defaults.update(kwargs)
    return FrameEvidence(**defaults)


def test_occlusion_outranks_tiny_scale():
    """A ball that is not visible cannot be fixed by a better small-object model."""
    tiny = GroundTruthBall(cx=100, cy=100, w=6, h=6)
    assert classify_miss(tiny, evidence(inside_player_box=True)) is (
        FailureCategory.PLAYER_OCCLUSION
    )


def test_threshold_rejection_is_separated_from_blindness():
    ball = GroundTruthBall(cx=100, cy=100, w=20, h=20)
    found_but_dropped = evidence(floor_distance_px=3.0, floor_confidence=0.02)
    assert classify_miss(ball, found_but_dropped) is (
        FailureCategory.DETECTOR_THRESHOLD_REJECTION
    )
    assert classify_miss(ball, evidence()) is FailureCategory.DETECTOR_MISS_UNEXPLAINED


def test_blur_and_low_contrast_together_are_unobservable():
    ball = GroundTruthBall(cx=100, cy=100, w=20, h=20)
    assert classify_miss(ball, evidence(blur_variance=1.0, contrast=0.01)) is (
        FailureCategory.GENUINELY_UNOBSERVABLE
    )


def test_every_miss_lands_in_exactly_one_category():
    ball = GroundTruthBall(cx=500, cy=500, w=10, h=10)
    cases = [
        evidence(inside_player_box=True),
        evidence(touches_frame_edge=True),
        evidence(floor_distance_px=2.0),
        evidence(blur_variance=1.0),
        evidence(contrast=0.01),
        evidence(nearest_player_px=5.0),
        evidence(line_strength=9.0),
        evidence(),
    ]
    categories = [classify_miss(ball, c) for c in cases]
    assert all(isinstance(c, FailureCategory) for c in categories)
    assert FailureCategory.DETECTED not in categories


@pytest.mark.parametrize("area,bucket", [
    (10, "1_tiny_lt150"), (149, "1_tiny_lt150"), (150, "2_small_150_400"),
    (399, "2_small_150_400"), (400, "3_medium_400_2000"), (5000, "4_large_gt2000"),
])
def test_size_buckets_partition_the_range(area, bucket):
    assert size_bucket(area) == bucket


# --------------------------------------------------------------------------- #
# Promotion rule
# --------------------------------------------------------------------------- #


def measurements(label, recall, precision, **kwargs):
    defaults = dict(
        effective_ball_coverage=0.50,
        possession_determinability=0.30,
        pass_recall=0.30,
        n_direct_observations=1000,
        n_inferred_observations=100,
        n_long_gap_fills=0,
        model_fingerprint="abc123",
        config_fingerprint="def456",
    )
    defaults.update(kwargs)
    return CandidateMeasurements(
        label=label, per_domain_recall=recall, per_domain_precision=precision, **defaults
    )


INCUMBENT = measurements(
    "incumbent", {"a": 0.50, "b": 0.40}, {"a": 0.60, "b": 0.58},
)


def test_a_clean_improvement_is_promoted():
    candidate = measurements(
        "candidate", {"a": 0.60, "b": 0.50}, {"a": 0.62, "b": 0.60},
        effective_ball_coverage=0.60, possession_determinability=0.40, pass_recall=0.40,
        n_direct_observations=1200, n_inferred_observations=120,
    )
    verdict = evaluate_promotion(candidate, INCUMBENT)
    assert verdict.promote, verdict.failures


def test_a_worst_domain_regression_blocks_promotion():
    """The Phase 2B failure mode: the mean improves, the weak domain collapses."""
    candidate = measurements(
        "candidate", {"a": 0.80, "b": 0.30}, {"a": 0.62, "b": 0.60},
        effective_ball_coverage=0.60, possession_determinability=0.40, pass_recall=0.40,
    )
    verdict = evaluate_promotion(candidate, INCUMBENT)
    assert not verdict.promote
    assert any("worst-domain" in f for f in verdict.failures)


def test_precision_below_the_floor_blocks_promotion():
    candidate = measurements(
        "candidate", {"a": 0.70, "b": 0.60}, {"a": 0.50, "b": 0.48},
        effective_ball_coverage=0.60, possession_determinability=0.40, pass_recall=0.40,
    )
    verdict = evaluate_promotion(candidate, INCUMBENT)
    assert not verdict.promote
    assert any("precision" in f for f in verdict.failures)


def test_unmeasured_downstream_evidence_is_not_a_pass():
    """A candidate cannot be promoted on evidence nobody collected."""
    candidate = measurements(
        "candidate", {"a": 0.60, "b": 0.50}, {"a": 0.62, "b": 0.60},
        effective_ball_coverage=None, possession_determinability=None, pass_recall=None,
    )
    verdict = evaluate_promotion(candidate, INCUMBENT)
    assert not verdict.promote
    assert sum("not measured" in f for f in verdict.failures) == 3


def test_long_gap_fills_block_promotion_outright():
    candidate = measurements(
        "candidate", {"a": 0.60, "b": 0.50}, {"a": 0.62, "b": 0.60},
        effective_ball_coverage=0.60, possession_determinability=0.40, pass_recall=0.40,
        n_long_gap_fills=1,
    )
    verdict = evaluate_promotion(candidate, INCUMBENT)
    assert not verdict.promote
    assert any("long-gap" in f for f in verdict.failures)


def test_coverage_bought_with_guesses_is_not_coverage():
    """Inferred positions growing faster than direct ones is not an improvement."""
    candidate = measurements(
        "candidate", {"a": 0.60, "b": 0.50}, {"a": 0.62, "b": 0.60},
        effective_ball_coverage=0.60, possession_determinability=0.40, pass_recall=0.40,
        n_direct_observations=1010, n_inferred_observations=900,
    )
    verdict = evaluate_promotion(candidate, INCUMBENT)
    assert not verdict.promote
    assert any("inferred positions" in f for f in verdict.failures)


def test_missing_fingerprints_block_promotion():
    candidate = measurements(
        "candidate", {"a": 0.60, "b": 0.50}, {"a": 0.62, "b": 0.60},
        effective_ball_coverage=0.60, possession_determinability=0.40, pass_recall=0.40,
        model_fingerprint="",
    )
    verdict = evaluate_promotion(candidate, INCUMBENT)
    assert not verdict.promote
    assert any("fingerprint" in f for f in verdict.failures)


def test_comparing_different_domain_sets_is_refused():
    candidate = measurements("candidate", {"a": 0.6, "c": 0.5}, {"a": 0.6, "c": 0.6})
    with pytest.raises(ValueError, match="different domain sets"):
        evaluate_promotion(candidate, INCUMBENT)


def test_criteria_are_frozen_so_they_cannot_drift_mid_run():
    """Thresholds must not be reassignable once a comparison is under way."""
    criteria = PromotionCriteria()
    with pytest.raises(AttributeError):
        criteria.min_precision = 0.1  # type: ignore[misc]
