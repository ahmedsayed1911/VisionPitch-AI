"""Phase 2E: the promoted ball operating point must resolve from config alone.

Production now ships `ball_gsrtrain_v2c @ imgsz 1280, conf 0.08, engine=legacy`.

Two defects made this worth pinning down:

* `load_config` merges `configs/modes/<mode>.yaml` **on top of** the base file,
  and `balanced.yaml` pins `ball_detection.imgsz`. A base file could therefore
  not choose its own resolution, and the candidate previously needed
  `--set ball_detection.imgsz=1280` on every invocation. A production setting
  that depends on someone remembering a CLI flag is a defect, so the resolution
  now lives in the overlay where it is actually read, and a self-contained
  config may opt out entirely with `apply_mode_overlay: false`.
* `max_accuracy` used a *coarser* ball resolution than `balanced` after the
  promotion, which would have made the "more accurate" mode less accurate.

These tests fail if either regresses, and they check that the rollback path is
still on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visionpitch.common.config import AnalysisMode, load_config

PRODUCTION = Path("configs/default.yaml")
CANDIDATE = Path("configs/candidate_ball_v2.yaml")
ROLLBACK = [
    Path("configs/default.pre_phase2e_ball.yaml"),
    Path("configs/modes/balanced.pre_phase2e_ball.yaml"),
    Path("configs/modes/max_accuracy.pre_phase2e_ball.yaml"),
]

WINNING_CHECKPOINT = "models/finetune/ball_gsrtrain_v2c/weights/best.pt"
WINNING_IMGSZ = 1280
WINNING_CONF = 0.08
WINNING_ENGINE = "legacy"
PREVIOUS_CHECKPOINT = "models/yolo-football-ball-detection.pt"


@pytest.fixture(scope="module")
def production():
    return load_config(config_path=PRODUCTION, mode=AnalysisMode.BALANCED)


def test_production_resolves_to_the_winning_operating_point(production):
    """The file alone must be enough -- no --set, no mode gymnastics."""
    ball = production.ball_detection
    assert ball.model_path == WINNING_CHECKPOINT
    assert ball.imgsz == WINNING_IMGSZ
    assert ball.conf_threshold == pytest.approx(WINNING_CONF)
    assert ball.enabled is True


def test_production_keeps_the_legacy_fusion_engine(production):
    """Temporal fusion was rejected on VALID evidence; it must stay off."""
    assert production.ball_fusion.engine == WINNING_ENGINE


def test_max_accuracy_is_not_coarser_than_balanced():
    """The higher-accuracy mode must not use a lower ball resolution."""
    balanced = load_config(config_path=PRODUCTION, mode=AnalysisMode.BALANCED)
    accurate = load_config(config_path=PRODUCTION, mode=AnalysisMode.MAX_ACCURACY)
    assert accurate.ball_detection.imgsz >= balanced.ball_detection.imgsz


def test_fast_preview_still_disables_the_ball_pass():
    """Promotion must not silently switch the cheap mode back on."""
    preview = load_config(config_path=PRODUCTION, mode=AnalysisMode.FAST_PREVIEW)
    assert preview.ball_detection.enabled is False


def test_candidate_config_matches_production(production):
    """The frozen candidate record must agree with what actually shipped."""
    if not CANDIDATE.exists():
        pytest.skip(f"{CANDIDATE} not present")
    candidate = load_config(config_path=CANDIDATE, mode=AnalysisMode.BALANCED)
    assert candidate.ball_detection.model_path == production.ball_detection.model_path
    assert candidate.ball_detection.imgsz == production.ball_detection.imgsz
    assert candidate.ball_detection.conf_threshold == pytest.approx(
        production.ball_detection.conf_threshold
    )


def test_apply_mode_overlay_opt_out_works():
    """A config declaring itself resolved must not be overwritten by the overlay.

    The candidate sets the flag, so the overlay must not win there even though a
    mode is passed explicitly -- fast_preview would otherwise disable the ball
    pass and reset the resolution.
    """
    if not CANDIDATE.exists():
        pytest.skip(f"{CANDIDATE} not present")
    raw = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
    assert raw.get("apply_mode_overlay") is False
    resolved = load_config(config_path=CANDIDATE, mode=AnalysisMode.FAST_PREVIEW)
    assert resolved.ball_detection.enabled is True
    assert resolved.ball_detection.imgsz == WINNING_IMGSZ


def test_apply_mode_overlay_flag_is_not_a_config_field(production):
    """It is a resolution directive, and must not leak into the run config."""
    assert not hasattr(production, "apply_mode_overlay")
    assert "apply_mode_overlay" not in production.model_dump()


def test_production_config_declares_no_opt_out():
    """Production must keep using its mode overlays."""
    raw = yaml.safe_load(PRODUCTION.read_text(encoding="utf-8"))
    assert "apply_mode_overlay" not in raw


def test_rollback_configs_are_preserved():
    """The previous production configuration must remain restorable."""
    for path in ROLLBACK:
        assert path.is_file(), f"rollback config missing: {path}"
    previous = yaml.safe_load(ROLLBACK[0].read_text(encoding="utf-8"))
    assert previous["ball_detection"]["model_path"] == PREVIOUS_CHECKPOINT
    assert previous["ball_detection"]["imgsz"] == 640


def test_both_checkpoints_exist_on_disk():
    """The winner must be loadable and the previous one must not be deleted."""
    assert Path(WINNING_CHECKPOINT).is_file()
    assert Path(PREVIOUS_CHECKPOINT).is_file(), (
        "the superseded ball checkpoint must be kept for rollback"
    )
