"""Regression tests for the chunked run that produced zero usable player rows.

Two independent defects combined to make a chunked run emit 11,304 person rows
of which *none* were usable, with no error anywhere:

1. ``PipelineResult`` had no ``support_regions`` field, and the chunked merge
   read it through ``getattr(result, "support_regions", {})``. The default
   silently supplied an empty dict on every run.
2. ``extrapolation_risk`` returned a middling value for an unknown support
   region, which exceeded the caller's threshold — so "no information about the
   support" was treated as "definitely extrapolated".

Together: every row was downgraded to EXTRAPOLATED, and every physical metric
became unavailable. Both halves are pinned here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from visionpitch.calibration.propagation import extrapolation_risk, support_region
from visionpitch.common.types import (
    BBox,
    CalibrationResult,
    ObjectClass,
    Role,
    TeamId,
    Track,
    TrackObservation,
    ValidationStatus,
)
from visionpitch.game_state.assembler import GameStateAssembler
from visionpitch.pipeline.runner import PipelineResult


class TestExtrapolationRisk:
    def test_unknown_region_is_not_treated_as_extrapolation(self) -> None:
        """Absence of evidence must not be evidence of extrapolation."""
        risk = extrapolation_risk((640.0, 400.0), None)
        assert risk == 0.0

    def test_unknown_region_never_exceeds_the_default_threshold(self) -> None:
        from visionpitch.common.config import CalibrationConfig

        threshold = CalibrationConfig().max_extrapolation_risk
        assert extrapolation_risk((0.0, 0.0), None) <= threshold

    def test_inside_the_region_is_zero_risk(self) -> None:
        region = (100.0, 100.0, 500.0, 400.0)
        assert extrapolation_risk((300.0, 250.0), region) == 0.0

    def test_risk_grows_outside_the_region(self) -> None:
        region = (100.0, 100.0, 500.0, 400.0)
        near = extrapolation_risk((520.0, 250.0), region)
        far = extrapolation_risk((900.0, 250.0), region)
        assert 0.0 < near < far <= 1.0

    def test_support_region_needs_three_points(self) -> None:
        assert support_region(np.array([[1.0, 2.0]]), (1280, 720)) is None
        region = support_region(
            np.array([[100.0, 100.0], [300.0, 200.0], [200.0, 400.0]]), (1280, 720)
        )
        assert region is not None and region[2] > region[0]


class TestPipelineResultCarriesSupportRegions:
    def test_field_exists_so_the_chunked_merge_cannot_silently_miss_it(self) -> None:
        """The original bug was a missing attribute hidden by a getattr default."""
        names = {f.name for f in dataclasses.fields(PipelineResult)}
        assert "support_regions" in names

    def test_chunked_merge_uses_direct_attribute_access(self) -> None:
        """A getattr default here would restore the silent-failure mode."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "visionpitch" / "pipeline" / "chunked_runner.py"
        ).read_text(encoding="utf-8")
        assert "support_regions.update(result.support_regions)" in source
        assert 'getattr(result, "support_regions"' not in source


class TestAssemblerStatuses:
    def _inputs(self, homography):
        track = Track(
            track_id=1,
            object_class=ObjectClass.PLAYER,
            observations=[
                TrackObservation(f, f / 25.0, BBox(600, 400, 640, 500), 0.9, 0.8, False)
                for f in range(3)
            ],
            team_id=TeamId.A,
            team_confidence=0.9,
            role=Role.OUTFIELD,
        )
        calibration = {
            f: CalibrationResult(f, homography, 0.8, 0.3, 10, 9) for f in range(3)
        }
        return {1: track}, {}, calibration, {f: f / 25.0 for f in range(3)}, [0, 1, 2]

    def test_no_support_regions_still_yields_valid_rows(
        self, pitch, synthetic_homography
    ) -> None:
        """The exact failure: a merge that lost its support regions must still
        produce usable rows, not downgrade every one of them."""
        assembler = GameStateAssembler(
            "clip", pitch, min_calibration_confidence=0.4, support_regions=None
        )
        rows = assembler.assemble(*self._inputs(synthetic_homography))
        players = [r for r in rows if r.object_class == "player"]
        assert players
        assert all(r.validation_status == ValidationStatus.VALID.value for r in players)

    def test_support_regions_still_mark_far_points_extrapolated(
        self, pitch, synthetic_homography
    ) -> None:
        """The fix must not disable the feature it was protecting."""
        tiny = {f: (0.0, 0.0, 50.0, 50.0) for f in range(3)}
        assembler = GameStateAssembler(
            "clip", pitch, min_calibration_confidence=0.4, support_regions=tiny,
            max_extrapolation_risk=0.35,
        )
        rows = assembler.assemble(*self._inputs(synthetic_homography))
        players = [r for r in rows if r.object_class == "player"]
        assert players
        assert all(
            r.validation_status == ValidationStatus.EXTRAPOLATED.value for r in players
        )


@pytest.mark.slow
class TestChunkedRunMatchesSinglePass:
    def test_a_chunked_run_produces_usable_rows(self, repo_root) -> None:
        """End-to-end guard: no chunked run may report zero usable player rows
        while its calibration table says most frames were solved."""
        import json

        import pandas as pd

        chunked = [
            p for p in repo_root.glob("outputs/*/*")
            if (p / "manifest.json").exists()
            and "chunking" in json.loads(
                (p / "manifest.json").read_text(encoding="utf-8")
            ).get("stages", {})
        ]
        if not chunked:
            pytest.skip("no chunked run available")

        run = max(chunked, key=lambda p: p.stat().st_mtime)
        game_state = pd.read_parquet(run / "game_state.parquet")
        calibration = pd.read_parquet(run / "calibration.parquet")

        solved = calibration.homography.apply(
            lambda v: v is not None and len(v) == 9
        ).mean()
        people = game_state[game_state.object_class != "ball"]
        usable = people[
            (people.validation_status == "valid") & people.pitch_x.notna()
        ]

        if solved > 0.5:
            assert len(usable) > 0, (
                f"{100 * solved:.0f}% of frames calibrated but no usable player rows"
            )
        assert game_state.duplicated(
            subset=["frame_idx", "track_id", "object_class"]
        ).sum() == 0
