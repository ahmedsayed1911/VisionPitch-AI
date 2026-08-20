"""Tiny-ball representation study: protocol, centre metrics, heatmap targets.

The properties pinned here are the ones that would silently flatter a new
representation: a split that leaks, a metric that counts one prediction twice,
a target that puts no positive anywhere, or a decoder that quantises its output
to the grid and hides it behind a good-looking average.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from visionpitch.detection.heatmap import (
    BallHeatmapNet,
    HeatmapConfig,
    decode,
    focal_loss,
    render_target,
)
from visionpitch.evaluation.tinyball import (
    CENTRE_TOLERANCES_PX,
    POSSESSION_TOLERANCE_PX,
    Partition,
    TinyBallProtocol,
    assert_clip_disjoint,
    clip_of,
    domain_of,
    pool,
    score_centres,
)

# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


def test_test_partition_is_the_only_untunable_one():
    assert not Partition.TEST.is_tunable
    assert Partition.TRAIN.is_tunable
    assert Partition.VAL_IN_DOMAIN.is_tunable
    assert Partition.VAL_CROSS_DOMAIN.is_tunable


@pytest.mark.parametrize("name,expected_domain,expected_clip", [
    ("soccernet_gsr_SNGS-116_000001.jpg", "soccernet_gsr", "SNGS-116"),
    ("roboflow_a1b2c3_frame0007.jpg", "roboflow", "a1b2c3"),
])
def test_domain_and_clip_are_recovered_from_the_filename(
    name, expected_domain, expected_clip
):
    assert domain_of(name) == expected_domain
    assert clip_of(name) == expected_clip


def test_clip_disjointness_is_checked_on_the_files_not_the_record():
    """A dataset rebuild that mixes clips must be caught even if split.json agrees."""
    partitions = {
        "train": [__import__("pathlib").Path("roboflow_aaa_1.jpg")],
        "test": [__import__("pathlib").Path("roboflow_bbb_1.jpg")],
    }
    assert_clip_disjoint(partitions)

    partitions["test"].append(__import__("pathlib").Path("roboflow_aaa_9.jpg"))
    with pytest.raises(AssertionError, match="share 1 source clip"):
        assert_clip_disjoint(partitions)


def test_protocol_file_detects_editing(tmp_path):
    protocol = TinyBallProtocol(
        dataset_root=tmp_path, base_split_fingerprint="abc123",
        cross_domain_holdout="soccernet_gsr", domains=["roboflow", "soccernet_gsr"],
        counts={"train": {"roboflow": 10}},
    )
    path = protocol.save(tmp_path / "protocol.json")
    assert TinyBallProtocol.load(path).fingerprint() == protocol.fingerprint()

    data = json.loads(path.read_text(encoding="utf-8"))
    data["counts"]["train"]["roboflow"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="was edited"):
        TinyBallProtocol.load(path)


def test_protocol_records_that_only_two_domains_carry_labels():
    """The study's central limitation must travel with its own protocol record."""
    protocol = TinyBallProtocol(
        dataset_root=__import__("pathlib").Path("."),
        base_split_fingerprint="x", cross_domain_holdout="soccernet_gsr",
    )
    note = protocol.to_dict()["note"]
    assert "SoccerNet-BAS" in note
    assert "no ball annotations" in note


# --------------------------------------------------------------------------- #
# Centre metrics
# --------------------------------------------------------------------------- #


def test_exact_hit_counts_at_every_tolerance():
    result = score_centres("m", "d", [([(100.0, 100.0)], [(100.0, 100.0)])])
    for tolerance in CENTRE_TOLERANCES_PX:
        assert result.recall_at(tolerance) == 1.0
    assert result.median_error_px == pytest.approx(0.0)


def test_tolerance_ladder_is_monotone():
    """Recall can never fall as the tolerance loosens."""
    frames = [
        ([(100.0, 100.0)], [(103.0, 100.0)]),
        ([(200.0, 200.0)], [(212.0, 200.0)]),
        ([(300.0, 300.0)], [(322.0, 300.0)]),
        ([(400.0, 400.0)], [(480.0, 400.0)]),
    ]
    result = score_centres("m", "d", frames)
    values = [result.recall_at(t) for t in CENTRE_TOLERANCES_PX]
    assert values == sorted(values)
    assert result.recall_at(5.0) == pytest.approx(0.25)
    assert result.recall_at(25.0) == pytest.approx(0.75)


def test_one_prediction_cannot_satisfy_two_ground_truth_balls():
    result = score_centres(
        "m", "d", [([(100.0, 100.0), (102.0, 100.0)], [(101.0, 100.0)])]
    )
    assert result.hits_at.get(5.0, 0) == 1
    assert result.n_truth == 2


def test_duplicate_predictions_count_as_false_positives():
    frames = [([(100.0, 100.0)], [(100.0, 100.0), (101.0, 100.0), (400.0, 400.0)])]
    result = score_centres("m", "d", frames)
    assert result.n_predicted == 3
    assert result.false_positives_per_frame == pytest.approx(2.0)


def test_possession_usability_uses_the_measured_task_tolerance():
    """A 40 px error is a detection failure but not a possession failure."""
    frames = [
        ([(100.0, 100.0)], [(140.0, 100.0)]),   # 40 px  -> usable
        ([(200.0, 200.0)], [(400.0, 200.0)]),   # 200 px -> not usable
    ]
    result = score_centres("m", "d", frames).to_dict()
    assert POSSESSION_TOLERANCE_PX == pytest.approx(64.8)
    assert result["possession_usable_rate"] == pytest.approx(0.5)
    assert result["recall_at_px"]["25.0"] == pytest.approx(0.0)


def test_frames_without_predictions_lower_coverage_not_error():
    frames = [([(1.0, 1.0)], []), ([(2.0, 2.0)], [(2.0, 2.0)])]
    result = score_centres("m", "d", frames)
    assert result.direct_coverage == pytest.approx(0.5)
    assert result.median_error_px == pytest.approx(0.0)


def test_pooling_reports_worst_domain_beside_the_macro():
    strong = score_centres("m", "good", [([(0.0, 0.0)], [(0.0, 0.0)])] * 10)
    weak = score_centres("m", "bad", [([(0.0, 0.0)], [(900.0, 900.0)])] * 10)
    summary = pool([strong, weak], "m")
    assert summary["macro_recall_at_px"]["25.0"] == pytest.approx(0.5)
    assert summary["worst_domain_recall_at_px"]["25.0"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Heatmap targets
# --------------------------------------------------------------------------- #


def test_target_has_an_exact_positive_for_every_ball():
    """Focal loss normalises by the positive count; a target with none is degenerate."""
    config = HeatmapConfig()
    target = render_target(
        [(101.0, 61.0), (300.0, 200.0)], [11.0, 16.0], (160, 160), 2, config
    )
    assert np.isclose(target.max(), 1.0)
    assert (target >= 1.0).sum() == 2


def test_target_peaks_at_the_ball_and_decays_away_from_it():
    config = HeatmapConfig()
    target = render_target([(100.0, 100.0)], [12.0], (160, 160), 2, config)
    peak_y, peak_x = np.unravel_index(np.argmax(target), target.shape)
    assert (peak_x, peak_y) == (50, 50)
    assert target[50, 50] > target[50, 53] > target[50, 60]


def test_overlapping_balls_do_not_sum_above_one():
    """Additive combination would create a target focal loss can never reach."""
    config = HeatmapConfig()
    target = render_target(
        [(100.0, 100.0), (102.0, 100.0)], [12.0, 12.0], (160, 160), 2, config
    )
    assert target.max() <= 1.0 + 1e-6


def test_balls_outside_the_output_grid_are_dropped_not_wrapped():
    config = HeatmapConfig()
    target = render_target([(-40.0, -40.0), (9999.0, 9999.0)], [11.0, 11.0],
                           (160, 160), 2, config)
    assert target.max() == 0.0


def test_sigma_never_collapses_below_the_floor():
    """A one-hot target with no gradient neighbourhood is what min_sigma prevents."""
    config = HeatmapConfig(min_sigma=1.0)
    target = render_target([(100.0, 100.0)], [1.0], (160, 160), 2, config)
    assert (target > 0.1).sum() > 4


def test_focal_loss_is_lower_when_the_prediction_matches():
    config = HeatmapConfig()
    target = torch.from_numpy(
        render_target([(100.0, 100.0)], [12.0], (80, 80), 2, config)
    )[None, None]
    good = target.clone().clamp(1e-3, 1 - 1e-3)
    bad = torch.full_like(target, 0.5)
    assert float(focal_loss(good, target, config)) < float(focal_loss(bad, target, config))


def test_focal_loss_survives_a_frame_with_no_ball():
    config = HeatmapConfig()
    target = torch.zeros(1, 1, 40, 40)
    prediction = torch.full_like(target, 0.1)
    value = float(focal_loss(prediction, target, config))
    assert np.isfinite(value)


# --------------------------------------------------------------------------- #
# Heatmap decoding
# --------------------------------------------------------------------------- #


def gaussian_heatmap(size=64, cx=32.0, cy=32.0, sigma=1.5, peak=1.0):
    ys, xs = np.mgrid[0:size, 0:size]
    return (peak * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))).astype(
        np.float32
    )


def test_decode_recovers_the_centre_in_input_pixels():
    config = HeatmapConfig(output_stride=2)
    detections = decode(gaussian_heatmap(cx=20.0, cy=30.0), config)
    assert detections
    assert detections[0].x == pytest.approx(40.0, abs=1.5)
    assert detections[0].y == pytest.approx(60.0, abs=1.5)


def test_decode_is_sub_pixel_not_grid_quantised():
    """Integer peaks alone would floor centre error at the output stride."""
    config = HeatmapConfig(output_stride=2)
    detections = decode(gaussian_heatmap(cx=20.4, cy=30.0), config)
    assert detections
    # 20.4 cells -> 40.8 input px. A grid-quantised decoder returns exactly 40.0.
    assert detections[0].x != pytest.approx(40.0, abs=1e-6)
    assert detections[0].x == pytest.approx(40.8, abs=1.0)


def test_decode_reports_uncertainty_that_grows_with_a_broader_peak():
    config = HeatmapConfig(output_stride=2)
    sharp = decode(gaussian_heatmap(sigma=1.0), config)[0]
    broad = decode(gaussian_heatmap(sigma=4.0), config)[0]
    assert broad.uncertainty_px > sharp.uncertainty_px
    assert sharp.uncertainty_px > 0


def test_decode_respects_the_peak_threshold():
    config = HeatmapConfig(peak_threshold=0.5)
    assert decode(gaussian_heatmap(peak=0.9), config)
    assert decode(gaussian_heatmap(peak=0.3), config) == []


def test_nms_keeps_one_detection_per_peak():
    config = HeatmapConfig(peak_threshold=0.3)
    heatmap = np.maximum(
        gaussian_heatmap(cx=15.0, cy=15.0), gaussian_heatmap(cx=45.0, cy=45.0)
    )
    detections = decode(heatmap, config)
    assert len(detections) == 2


def test_decode_rejects_a_non_2d_heatmap():
    with pytest.raises(ValueError, match="2-D heatmap"):
        decode(np.zeros((1, 1, 8, 8), dtype=np.float32), HeatmapConfig())


def test_detection_serialises_without_width_or_height():
    """The representation deliberately drops box size; the schema must reflect that."""
    payload = decode(gaussian_heatmap(), HeatmapConfig())[0].to_dict()
    assert set(payload) == {"x", "y", "confidence", "uncertainty_px"}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def test_model_output_is_a_probability_map_at_the_configured_stride():
    config = HeatmapConfig(input_size=128, output_stride=2, base_channels=4)
    model = BallHeatmapNet(config).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 128))
    assert out.shape == (1, 1, 64, 64)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_model_starts_predicting_near_zero_everywhere():
    """Without the negative head bias, background dominates and training collapses."""
    model = BallHeatmapNet(HeatmapConfig(input_size=64, base_channels=4)).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 64, 64))
    assert float(out.mean()) < 0.15


def test_model_is_fully_convolutional_so_it_can_run_off_its_training_size():
    """Needed for the inference-resolution sweep to be meaningful."""
    model = BallHeatmapNet(HeatmapConfig(input_size=128, base_channels=4)).eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 256, 256)).shape == (1, 1, 128, 128)


# --------------------------------------------------------------------------- #
# Pipeline adapter and schema compatibility
# --------------------------------------------------------------------------- #


@pytest.fixture
def heatmap_checkpoint(tmp_path):
    config = HeatmapConfig(input_size=128, base_channels=4)
    model = BallHeatmapNet(config)
    path = tmp_path / "heatmap.pt"
    torch.save({"model": model.state_dict(), "config": config.to_dict(), "epoch": 3}, path)
    return path


def test_representation_defaults_to_box_so_no_existing_run_moves():
    from visionpitch.common.config import Config

    assert Config().ball_detection.representation == "box"


def test_factory_returns_the_configured_representation(heatmap_checkpoint):
    from visionpitch.common.config import Config
    from visionpitch.detection.ball import build_ball_detector
    from visionpitch.detection.heatmap_detector import HeatmapBallDetector

    config = Config()
    config.ball_detection.representation = "heatmap"
    config.ball_detection.model_path = str(heatmap_checkpoint)
    config.runtime.device = "cpu"

    detector = build_ball_detector(config)
    assert isinstance(detector, HeatmapBallDetector)


def test_adapter_emits_valid_detections_with_the_unchanged_schema(heatmap_checkpoint):
    """The pipeline's Detection type must be satisfied exactly as before."""
    from visionpitch.common.config import Config
    from visionpitch.common.types import Detection, ObjectClass
    from visionpitch.detection.heatmap_detector import HeatmapBallDetector

    config = Config()
    config.ball_detection.model_path = str(heatmap_checkpoint)
    config.ball_detection.conf_threshold = 0.0
    config.runtime.device = "cpu"

    detector = HeatmapBallDetector(config)
    detections = detector.detect(np.zeros((256, 256, 3), dtype=np.uint8), frame_idx=7)
    for detection in detections:
        assert isinstance(detection, Detection)
        assert detection.object_class is ObjectClass.BALL
        assert detection.frame_idx == 7
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.bbox.width > 0 and detection.bbox.height > 0


def test_adapter_declares_that_its_box_is_synthesised(heatmap_checkpoint):
    """A synthesised box must never be mistaken for a predicted one."""
    from visionpitch.common.config import Config
    from visionpitch.detection.heatmap_detector import HeatmapBallDetector

    config = Config()
    config.ball_detection.model_path = str(heatmap_checkpoint)
    config.runtime.device = "cpu"

    info = HeatmapBallDetector(config).info.to_dict()
    assert info["representation"] == "centre_heatmap"
    assert info["bbox_is_synthesised"] is True
    assert "not a box" in info["bbox_note"]
    assert info["weights_sha256"]


def test_adapter_ignores_roi_hints_rather_than_quietly_using_them(heatmap_checkpoint):
    """Accepting the argument keeps the pipeline unchanged; using it would make
    the two representations incomparable."""
    from visionpitch.common.config import Config
    from visionpitch.detection.heatmap_detector import HeatmapBallDetector

    config = Config()
    config.ball_detection.model_path = str(heatmap_checkpoint)
    config.ball_detection.conf_threshold = 0.0
    config.runtime.device = "cpu"
    detector = HeatmapBallDetector(config)

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    without = detector.detect(image, 0, None, allow_tiled=True)
    with_hint = detector.detect(image, 0, (10.0, 10.0), allow_tiled=False)
    assert [d.bbox.x1 for d in without] == [d.bbox.x1 for d in with_hint]


# --------------------------------------------------------------------------- #
# Representation promotion rule
# --------------------------------------------------------------------------- #


def representation(label, **overrides):
    from visionpitch.evaluation.representation_promotion import (
        RepresentationMeasurements,
    )

    defaults = dict(
        macro_centre_recall_25px=0.50,
        worst_domain_centre_recall_25px=0.45,
        macro_precision_25px=0.60,
        macro_direct_coverage=0.60,
        worst_domain_direct_coverage=0.55,
        false_positives_per_frame=0.30,
        per_domain_centre_recall_25px={"a": 0.55, "b": 0.45},
        determinability=0.12,
        pass_recall=0.23,
        ball_coverage_direct=0.43,
        model_fingerprint="abc123",
    )
    defaults.update(overrides)
    return RepresentationMeasurements(label=label, **defaults)


def test_a_representation_meeting_every_criterion_is_promoted():
    from visionpitch.evaluation.representation_promotion import evaluate_representation

    incumbent = representation("box")
    candidate = representation(
        "heatmap",
        macro_centre_recall_25px=0.60,
        worst_domain_centre_recall_25px=0.58,
        worst_domain_direct_coverage=0.66,
        per_domain_centre_recall_25px={"a": 0.62, "b": 0.58},
        determinability=0.15,
        pass_recall=0.28,
        false_positives_per_frame=0.33,
    )
    verdict = evaluate_representation(candidate, incumbent)
    assert verdict.promote, verdict.failures


def test_a_small_coverage_gain_is_not_enough():
    """The declared bar is +0.10 absolute on the worst domain, not any gain."""
    from visionpitch.evaluation.representation_promotion import evaluate_representation

    incumbent = representation("box")
    candidate = representation("heatmap", worst_domain_direct_coverage=0.58)
    verdict = evaluate_representation(candidate, incumbent)
    assert not verdict.promote
    assert any("worst-domain direct coverage" in f for f in verdict.failures)


def test_extra_recall_bought_with_false_positives_is_rejected():
    from visionpitch.evaluation.representation_promotion import evaluate_representation

    incumbent = representation("box")
    candidate = representation(
        "heatmap",
        macro_centre_recall_25px=0.62,
        worst_domain_direct_coverage=0.70,
        per_domain_centre_recall_25px={"a": 0.62, "b": 0.62},
        determinability=0.15, pass_recall=0.28,
        false_positives_per_frame=0.90,
    )
    verdict = evaluate_representation(candidate, incumbent)
    assert not verdict.promote
    assert any("false positives" in f for f in verdict.failures)


def test_downstream_evidence_must_exist():
    from visionpitch.evaluation.representation_promotion import evaluate_representation

    incumbent = representation("box")
    candidate = representation(
        "heatmap", worst_domain_direct_coverage=0.70,
        macro_centre_recall_25px=0.60,
        per_domain_centre_recall_25px={"a": 0.62, "b": 0.60},
        determinability=None, pass_recall=None,
    )
    verdict = evaluate_representation(candidate, incumbent)
    assert not verdict.promote
    assert sum("not measured" in f for f in verdict.failures) == 2


def test_a_single_domain_regression_blocks_promotion():
    from visionpitch.evaluation.representation_promotion import evaluate_representation

    incumbent = representation("box")
    candidate = representation(
        "heatmap", worst_domain_direct_coverage=0.70,
        macro_centre_recall_25px=0.60,
        per_domain_centre_recall_25px={"a": 0.30, "b": 0.90},
        determinability=0.15, pass_recall=0.28,
    )
    verdict = evaluate_representation(candidate, incumbent)
    assert not verdict.promote
    assert any(f.startswith("a centre recall") for f in verdict.failures)


def test_representation_criteria_are_frozen():
    from visionpitch.evaluation.representation_promotion import RepresentationCriteria

    criteria = RepresentationCriteria()
    assert criteria.min_worst_domain_coverage_gain == 0.10
    assert criteria.min_precision == 0.55
    with pytest.raises(AttributeError):
        criteria.min_worst_domain_coverage_gain = 0.0  # type: ignore[misc]


def test_ball_state_kinds_are_unchanged_by_this_study():
    """Schema backward compatibility: no consumer's vocabulary may shift."""
    from visionpitch.analytics.types import BallStateKind

    assert {k.value for k in BallStateKind} == {
        "observed", "recovered", "interpolated", "unknown"
    }
    assert BallStateKind.OBSERVED.is_direct
    assert not BallStateKind.RECOVERED.is_direct
