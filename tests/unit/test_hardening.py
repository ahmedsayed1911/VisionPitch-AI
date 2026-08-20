"""Precision hardening: FP taxonomy, temporal filter, mining and promotion.

The properties pinned here are the ones whose failure would produce a hardened
model that looks better and is not: a taxonomy that mines fusion failures as
appearance negatives, a filter that invents positions, a hard-negative set built
from the locked test, or a promotion rule that can be reinterpreted.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from visionpitch.ball_tracking.fp_filter import (
    Candidate,
    CandidateState,
    FilterConfig,
    RejectionReason,
    TemporalFalsePositiveFilter,
    summarise,
)
from visionpitch.evaluation.false_positives import (
    FalsePositiveKind,
    classify,
    mark_duplicates,
    measure,
)

# --------------------------------------------------------------------------- #
# False-positive taxonomy
# --------------------------------------------------------------------------- #


def evidence(**kwargs):
    base = {
        "green_fraction": 0.6, "line_responses": 0, "blur_variance": 300.0,
        "roundness": 0.4, "inside_person": False, "person_vertical_position": None,
        "relative_y": 0.6,
    }
    base.update(kwargs)
    return base


def test_containment_in_a_person_outranks_appearance():
    """A white blob inside a player is kit, whatever else it looks like."""
    assert classify(
        evidence(inside_person=True, person_vertical_position=0.85, roundness=0.9),
        10, 10,
    ) is FalsePositiveKind.PLAYER_SOCKS_BOOTS
    assert classify(
        evidence(inside_person=True, person_vertical_position=0.2), 10, 10
    ) is FalsePositiveKind.JERSEY_MARKING


def test_a_line_through_the_patch_beats_roundness():
    assert classify(
        evidence(line_responses=3, roundness=0.9), 10, 10
    ) is FalsePositiveKind.PITCH_LINE


def test_a_round_blob_on_clean_grass_is_a_penalty_spot():
    assert classify(
        evidence(roundness=0.7, line_responses=0), 12, 12
    ) is FalsePositiveKind.PENALTY_SPOT


def test_off_pitch_positions_split_by_frame_region():
    assert classify(
        evidence(green_fraction=0.0, relative_y=0.02), 10, 10
    ) is FalsePositiveKind.BROADCAST_GRAPHIC
    assert classify(
        evidence(green_fraction=0.0, relative_y=0.2, roundness=0.1), 10, 10
    ) is FalsePositiveKind.CROWD_HIGHLIGHT
    assert classify(
        evidence(green_fraction=0.0, relative_y=0.2, roundness=0.8), 10, 10
    ) is FalsePositiveKind.WHITE_SEAT
    assert classify(
        evidence(green_fraction=0.0, relative_y=0.6), 10, 10
    ) is FalsePositiveKind.ADVERTISING_BOARD


def test_blurred_patches_split_by_shape():
    assert classify(
        evidence(blur_variance=5.0, roundness=0.1), 10, 10
    ) is FalsePositiveKind.COMPRESSION_ARTIFACT
    assert classify(
        evidence(blur_variance=5.0, roundness=0.6), 10, 10
    ) is FalsePositiveKind.MOTION_BLUR_ARTIFACT


def test_fusion_failures_are_never_mined_as_appearance_negatives():
    """Cropping a duplicate would teach the detector to suppress real balls."""
    assert not FalsePositiveKind.DUPLICATE_CANDIDATE.minable
    assert not FalsePositiveKind.TRAJECTORY_INCONSISTENT.minable
    assert not FalsePositiveKind.UNKNOWN.minable
    assert FalsePositiveKind.PITCH_LINE.minable
    assert FalsePositiveKind.PLAYER_SOCKS_BOOTS.minable


def test_temporal_categories_are_flagged_as_needing_context():
    assert FalsePositiveKind.DUPLICATE_CANDIDATE.is_temporal
    assert FalsePositiveKind.TRAJECTORY_INCONSISTENT.is_temporal
    assert not FalsePositiveKind.PITCH_LINE.is_temporal


def test_duplicate_marking_keeps_the_most_confident():
    predictions = [(100.0, 100.0, 0.4), (105.0, 102.0, 0.9), (500.0, 500.0, 0.3)]
    flags = mark_duplicates(predictions)
    assert flags == [True, False, False]


def test_measure_returns_evidence_without_naming_it():
    image = np.full((200, 200, 3), 60, dtype=np.uint8)
    image[:, :, 1] = 160  # greenish
    result = measure(image, 100.0, 100.0, 10.0, 10.0, np.zeros((0, 4)))
    assert set(result) == {
        "green_fraction", "line_responses", "blur_variance", "roundness",
        "inside_person", "person_vertical_position", "relative_y",
    }
    assert 0.0 <= result["green_fraction"] <= 1.0


# --------------------------------------------------------------------------- #
# Temporal filter
# --------------------------------------------------------------------------- #


def track(points, confidence=0.4, radius=7.0):
    return {
        i: Candidate(frame_idx=i, x=x, y=y, confidence=confidence, radius_px=radius)
        for i, (x, y) in points.items()
    }


def test_a_smooth_track_is_temporally_verified():
    candidates = track({i: (100.0 + 8 * i, 100.0) for i in range(6)})
    result = TemporalFalsePositiveFilter().filter_sequence(
        candidates, list(range(6))
    )
    assert all(
        r.state is CandidateState.TEMPORALLY_VERIFIED for r in result.values()
    ), {k: v.state for k, v in result.items()}


def test_a_lone_candidate_is_rejected():
    candidates = track({3: (100.0, 100.0)})
    result = TemporalFalsePositiveFilter().filter_sequence(
        candidates, list(range(6))
    )
    assert result[3].state is CandidateState.REJECTED
    assert RejectionReason.SINGLE_FRAME in result[3].reasons


def test_a_very_confident_lone_candidate_survives():
    """A ball reappearing from occlusion has no neighbours yet."""
    candidates = track({3: (100.0, 100.0)}, confidence=0.9)
    result = TemporalFalsePositiveFilter().filter_sequence(
        candidates, list(range(6))
    )
    assert result[3].state is CandidateState.DIRECT


def test_an_implausible_jump_is_rejected():
    candidates = track({0: (100.0, 100.0), 1: (900.0, 700.0), 2: (905.0, 705.0)})
    result = TemporalFalsePositiveFilter().filter_sequence(
        candidates, [0, 1, 2]
    )
    assert RejectionReason.IMPLAUSIBLE_JUMP in result[1].reasons


def test_a_candidate_fixed_in_the_image_during_a_pan_is_an_overlay():
    """A broadcast graphic is painted on the frame and does not move with it."""
    candidates = track({i: (200.0, 150.0) for i in range(5)})
    shifts = {i: (25.0, 0.0) for i in range(5)}
    result = TemporalFalsePositiveFilter().filter_sequence(
        candidates, list(range(5)), camera_shifts=shifts
    )
    rejected = [r for r in result.values() if r.state is CandidateState.REJECTED]
    assert rejected
    assert any(
        RejectionReason.STATIC_WHILE_CAMERA_MOVES in r.reasons for r in rejected
    )


def test_a_candidate_moving_with_the_background_is_world_static():
    """A pitch marking travels by exactly the camera displacement, reversed."""
    shift = (20.0, 0.0)
    candidates = track({i: (300.0 - 20.0 * i, 150.0) for i in range(5)})
    shifts = {i: shift for i in range(5)}
    result = TemporalFalsePositiveFilter().filter_sequence(
        candidates, list(range(5)), camera_shifts=shifts
    )
    assert any(
        RejectionReason.MOVES_WITH_BACKGROUND in r.reasons for r in result.values()
    )


def test_a_still_camera_triggers_neither_camera_test():
    candidates = track({i: (200.0, 150.0) for i in range(5)})
    shifts = {i: (0.5, 0.0) for i in range(5)}
    result = TemporalFalsePositiveFilter().filter_sequence(
        candidates, list(range(5)), camera_shifts=shifts
    )
    for item in result.values():
        assert RejectionReason.STATIC_WHILE_CAMERA_MOVES not in item.reasons
        assert RejectionReason.MOVES_WITH_BACKGROUND not in item.reasons


def test_a_candidate_outside_the_plausible_region_is_rejected():
    candidates = track({i: (10.0, 10.0) for i in range(4)})
    result = TemporalFalsePositiveFilter().filter_sequence(
        candidates, list(range(4)), plausible_region=(100.0, 100.0, 500.0, 500.0)
    )
    assert all(
        RejectionReason.OUTSIDE_PLAUSIBLE_REGION in r.reasons for r in result.values()
    )


def test_a_sudden_size_change_is_rejected():
    candidates = track({0: (100.0, 100.0), 1: (108.0, 100.0), 2: (116.0, 100.0)})
    candidates[1] = Candidate(1, 108.0, 100.0, 0.4, radius_px=40.0)
    result = TemporalFalsePositiveFilter().filter_sequence(candidates, [0, 1, 2])
    assert RejectionReason.SIZE_INCONSISTENT in result[1].reasons


def test_the_filter_never_invents_a_position():
    """Its central safety property: it removes and downgrades, nothing else."""
    candidates = track({0: (100.0, 100.0), 5: (140.0, 100.0)})
    result = TemporalFalsePositiveFilter().filter_sequence(
        candidates, list(range(8))
    )
    for frame_idx, item in result.items():
        if frame_idx not in candidates:
            assert item.state is CandidateState.UNKNOWN
            assert item.x is None and item.y is None
    # Rejected candidates carry no coordinates either.
    for item in result.values():
        if item.state is CandidateState.REJECTED:
            assert item.x is None and item.y is None


def test_states_stay_distinguishable():
    assert CandidateState.DIRECT.is_usable
    assert CandidateState.TEMPORALLY_VERIFIED.is_usable
    assert not CandidateState.REJECTED.is_usable
    assert not CandidateState.UNKNOWN.is_usable


def test_summary_reports_states_and_reasons():
    candidates = track({0: (100.0, 100.0), 4: (900.0, 700.0)})
    result = TemporalFalsePositiveFilter().filter_sequence(
        candidates, list(range(6))
    )
    report = summarise(result)
    assert report["n_frames"] == 6
    assert "never proposes a position" in report["note"]
    assert sum(report["by_state"].values()) == 6


def test_filter_config_is_serialisable_for_the_manifest():
    payload = FilterConfig().to_dict()
    for key in ("max_step_px", "camera_motion_px", "min_support_frames"):
        assert key in payload


# --------------------------------------------------------------------------- #
# Mined dataset provenance and leakage
# --------------------------------------------------------------------------- #


def test_mined_hard_negatives_exclude_the_locked_tests():
    from pathlib import Path

    path = Path("data/hard_negatives/PROVENANCE.json")
    if not path.exists():
        pytest.skip("hard negatives not mined in this checkout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["excluded_splits"]) == {"public_test", "local_test"}
    for record in payload["records"]:
        assert "test" not in record["source_split"], record
    assert len(payload["fingerprint"]) == 16


def test_mined_kinds_are_all_minable():
    from pathlib import Path

    path = Path("data/hard_negatives/PROVENANCE.json")
    if not path.exists():
        pytest.skip("hard negatives not mined in this checkout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for kind in payload["by_kind"]:
        assert FalsePositiveKind(kind).minable, kind


def test_calibration_was_fitted_on_validation_only():
    from pathlib import Path

    path = Path("data/eval/hardening/calibration.json")
    if not path.exists():
        pytest.skip("calibration not run in this checkout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "validation only" in payload["fitted_on"]
    assert "test" not in payload["fitted_on"]
    assert payload["selected_method"] in payload["methods"]


def test_calibration_operating_point_is_reproducible():
    from pathlib import Path

    path = Path("data/eval/hardening/calibration.json")
    if not path.exists():
        pytest.skip("calibration not run in this checkout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chosen = payload["selected_operating_point"]
    method = payload["methods"][payload["selected_method"]]
    # The selected point must be present in the recorded curve, so the choice
    # can be re-derived from the artefact rather than trusted.
    assert any(
        row["operating_point"] == chosen["operating_point"] for row in method["curve"]
    )
