"""Ball candidate fusion: suppression, provenance, stabilisation, state output.

The properties pinned here are the ones whose failure would corrupt the ball
signal silently: a merge that swallows two genuinely distinct hypotheses, a
fusion layer that invents a position, a stabilised coordinate that overwrites
the image one, or an output that lets a verified observation pass as a direct
sighting.
"""

from __future__ import annotations

import pytest

from visionpitch.ball_tracking.fp_filter import FilterConfig
from visionpitch.ball_tracking.fusion import (
    BallFusion,
    FusionConfig,
    ObservationKind,
    SuppressionMethod,
    stabilise,
    summarise,
    suppress,
)


def permissive() -> FilterConfig:
    return FilterConfig(
        min_support_frames=0, trust_confidence=0.0, max_step_px=1e9,
        camera_motion_px=1e9, max_size_ratio=1e9,
    )


# --------------------------------------------------------------------------- #
# Duplicate suppression
# --------------------------------------------------------------------------- #


def test_iou_fails_to_merge_duplicates_on_a_tiny_ball():
    """The measured defect: two detections 8 px apart on an 11 px ball.

    IoU at 0.5 calls them distinct hypotheses, which is how 26.3% of Candidate
    C's false positives came to be duplicates.
    """
    detections = [(100.0, 100.0, 5.5, 0.5), (108.0, 101.0, 5.5, 0.4)]
    kept = suppress(detections, FusionConfig(suppression=SuppressionMethod.IOU))
    assert len(kept) == 2


def test_centre_distance_merges_them():
    detections = [(100.0, 100.0, 5.5, 0.5), (108.0, 101.0, 5.5, 0.4)]
    kept = suppress(
        detections, FusionConfig(suppression=SuppressionMethod.CENTRE_DISTANCE)
    )
    assert len(kept) == 1
    assert kept[0].n_merged == 2


def test_spatially_distinct_candidates_are_never_merged():
    """Two plausible balls far apart must survive as separate hypotheses."""
    detections = [(100.0, 100.0, 7.0, 0.5), (600.0, 400.0, 7.0, 0.45)]
    for method in (
        SuppressionMethod.CENTRE_DISTANCE, SuppressionMethod.WEIGHTED_CENTRE
    ):
        kept = suppress(detections, FusionConfig(suppression=method))
        assert len(kept) == 2, method


def test_merged_candidate_keeps_provenance():
    detections = [
        (100.0, 100.0, 7.0, 0.5), (110.0, 102.0, 7.0, 0.4), (600.0, 400.0, 7.0, 0.3)
    ]
    kept = suppress(
        detections, FusionConfig(suppression=SuppressionMethod.WEIGHTED_CENTRE)
    )
    winner = kept[0]
    assert winner.source_ids == [0, 1]
    assert winner.merge_method == "weighted_centre"
    assert winner.n_merged == 2
    assert winner.uncertainty_px > 0


def test_weighted_centre_lands_between_its_members():
    detections = [(100.0, 100.0, 7.0, 0.6), (120.0, 100.0, 7.0, 0.4)]
    kept = suppress(
        detections, FusionConfig(suppression=SuppressionMethod.WEIGHTED_CENTRE)
    )
    assert 100.0 < kept[0].x < 120.0
    # Weighted toward the more confident member.
    assert kept[0].x < 110.0


def test_merged_confidence_is_the_strongest_member_not_the_sum():
    """Two weak looks at one blob are one weak detection, not a strong one."""
    detections = [(100.0, 100.0, 7.0, 0.3), (104.0, 100.0, 7.0, 0.3)]
    kept = suppress(
        detections, FusionConfig(suppression=SuppressionMethod.WEIGHTED_CENTRE)
    )
    assert kept[0].confidence == pytest.approx(0.3)


def test_candidate_cap_is_respected():
    detections = [(100.0 * i, 100.0, 7.0, 0.5 - 0.01 * i) for i in range(1, 8)]
    kept = suppress(
        detections,
        FusionConfig(
            suppression=SuppressionMethod.CENTRE_DISTANCE,
            max_candidates_per_frame=3,
        ),
    )
    assert len(kept) == 3
    # Kept in descending confidence.
    assert kept[0].confidence >= kept[1].confidence >= kept[2].confidence


def test_no_detections_yields_no_candidates():
    assert suppress([], FusionConfig()) == []


# --------------------------------------------------------------------------- #
# Camera stabilisation
# --------------------------------------------------------------------------- #


def test_stabilisation_does_not_overwrite_image_coordinates():
    """Every downstream consumer works in image space."""
    fusion = BallFusion(FusionConfig(temporal=permissive()))
    frames = fusion.run(
        {i: [(100.0 + 10 * i, 100.0, 7.0, 0.6)] for i in range(4)},
        list(range(4)),
        camera_shifts={i: (10.0, 0.0) for i in range(4)},
        camera_confidence={i: 1.0 for i in range(4)},
    )
    assert frames[3].x == pytest.approx(130.0)


def test_untrusted_camera_estimates_are_not_applied():
    candidates = {
        i: type("M", (), {"x": 100.0, "y": 50.0})() for i in range(3)
    }
    _, trusted = stabilise(
        candidates, {i: (10.0, 0.0) for i in range(3)},
        camera_confidence={0: 0.9, 1: 0.1, 2: 0.9}, min_confidence=0.35,
    )
    assert trusted == {0: True, 1: False, 2: True}


def test_a_camera_cut_resets_the_accumulated_offset():
    candidates = {i: type("M", (), {"x": 0.0, "y": 0.0})() for i in range(6)}
    stabilised, _ = stabilise(
        candidates, {i: (10.0, 0.0) for i in range(6)},
        camera_confidence={i: 1.0 for i in range(6)}, cut_frames={3},
    )
    # Offset accumulates to frame 2, resets at 3, then accumulates again.
    assert stabilised[2][0] == pytest.approx(30.0)
    assert stabilised[3][0] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Fusion output
# --------------------------------------------------------------------------- #


def test_frames_without_candidates_carry_no_position():
    fusion = BallFusion(FusionConfig(temporal=permissive()))
    frames = fusion.run({0: [(100.0, 100.0, 7.0, 0.6)]}, [0, 1, 2])
    for frame_idx in (1, 2):
        assert frames[frame_idx].kind is ObservationKind.UNKNOWN
        assert frames[frame_idx].x is None and frames[frame_idx].y is None


def test_rejected_candidates_carry_no_position_and_a_reason():
    """The layer's safety property: it removes, it does not relocate."""
    fusion = BallFusion(FusionConfig(temporal=FilterConfig()))
    frames = fusion.run({3: [(100.0, 100.0, 7.0, 0.2)]}, list(range(6)))
    assert frames[3].kind is ObservationKind.UNKNOWN
    assert frames[3].x is None
    assert frames[3].rejection_reasons


def test_a_supported_track_is_marked_temporally_verified_not_direct():
    fusion = BallFusion(FusionConfig())
    frames = fusion.run(
        {i: [(100.0 + 8 * i, 100.0, 7.0, 0.5)] for i in range(6)}, list(range(6))
    )
    kinds = {f.kind for f in frames.values()}
    assert ObservationKind.OBSERVED_TEMPORALLY_VERIFIED in kinds
    for frame in frames.values():
        if frame.kind is ObservationKind.OBSERVED_TEMPORALLY_VERIFIED:
            assert not frame.kind.is_direct


def test_observability_names_the_reason_a_frame_is_empty():
    fusion = BallFusion(FusionConfig(temporal=permissive()))
    frames = fusion.run(
        {}, [0, 1, 2],
        observability={
            0: "likely_outside_frame", 1: "likely_occluded", 2: "likely_visible"
        },
    )
    assert frames[0].kind is ObservationKind.OUTSIDE_FRAME
    assert frames[1].kind is ObservationKind.OCCLUDED
    assert frames[2].kind is ObservationKind.UNKNOWN


def test_fusion_confidence_never_exceeds_detector_confidence():
    """Temporal agreement corroborates evidence; it does not create it."""
    fusion = BallFusion(FusionConfig())
    frames = fusion.run(
        {i: [(100.0 + 8 * i, 100.0, 7.0, 0.5)] for i in range(6)}, list(range(6))
    )
    for frame in frames.values():
        if frame.kind.counts_as_observed:
            assert frame.fusion_confidence <= frame.detector_confidence + 1e-9


def test_observation_kinds_stay_distinguishable():
    assert ObservationKind.OBSERVED_DIRECT.is_direct
    assert not ObservationKind.OBSERVED_TEMPORALLY_VERIFIED.is_direct
    assert not ObservationKind.RECOVERED_WEAK_EVIDENCE.is_direct
    assert not ObservationKind.INTERPOLATED.is_direct
    for kind in (
        ObservationKind.OBSERVED_DIRECT,
        ObservationKind.OBSERVED_TEMPORALLY_VERIFIED,
    ):
        assert kind.counts_as_observed
    for kind in (
        ObservationKind.INTERPOLATED, ObservationKind.OCCLUDED,
        ObservationKind.OUTSIDE_FRAME, ObservationKind.UNKNOWN,
    ):
        assert not kind.counts_as_observed


def test_interpolated_and_occluded_are_never_counted_as_observed():
    assert not ObservationKind.INTERPOLATED.counts_as_observed
    assert not ObservationKind.OCCLUDED.counts_as_observed
    assert ObservationKind.INTERPOLATED.has_position
    assert not ObservationKind.OCCLUDED.has_position


def test_config_fingerprint_changes_with_any_knob():
    base = FusionConfig()
    assert base.fingerprint() != FusionConfig(merge_radius_px=30.0).fingerprint()
    assert base.fingerprint() != FusionConfig(
        suppression=SuppressionMethod.IOU
    ).fingerprint()
    assert base.fingerprint() != FusionConfig(
        temporal=FilterConfig(min_support_frames=5)
    ).fingerprint()
    assert base.fingerprint() == FusionConfig().fingerprint()


def test_summary_separates_direct_from_observed_coverage():
    fusion = BallFusion(FusionConfig())
    frames = fusion.run(
        {i: [(100.0 + 8 * i, 100.0, 7.0, 0.5)] for i in range(6)}, list(range(8))
    )
    report = summarise(frames)
    assert report["n_frames"] == 8
    assert report["observed_coverage"] >= report["direct_coverage"]
    assert "by_kind" in report


def test_ball_state_kind_vocabulary_is_unchanged():
    """Backward compatibility: stored schemas must still parse."""
    from visionpitch.analytics.types import BallStateKind

    assert {k.value for k in BallStateKind} == {
        "observed", "recovered", "interpolated", "unknown"
    }
