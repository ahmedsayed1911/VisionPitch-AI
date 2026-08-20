"""Phase 2 analytics: metrics, kinematics, possession, events."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from visionpitch.analytics.kinematics import (
    KinematicProfile,
    _rts_smooth,
    compute_kinematics,
    extract_segments,
)
from visionpitch.analytics.types import (
    BallStateKind,
    Confidence,
    CoverageProfile,
    Evidence,
    Metric,
    MetricBasis,
    PossessionSpan,
    PossessionState,
)


def track_frame(frames, xs, ys, fps=25.0, track_id=1) -> pd.DataFrame:
    return pd.DataFrame({
        "track_id": track_id,
        "frame_idx": frames,
        "timestamp_s": [f / fps for f in frames],
        "pitch_x": xs,
        "pitch_y": ys,
    })


class TestMetric:
    def test_unavailable_is_not_zero(self) -> None:
        """'No data' and 'zero' are different facts."""
        metric = Metric.unavailable("m")
        assert metric.value is None
        assert not metric.is_reportable

    def test_reportable_requires_samples_and_coverage(self) -> None:
        assert not Metric(value=10.0, coverage=0.0, n_samples=5).is_reportable
        assert not Metric(value=10.0, coverage=0.5, n_samples=0).is_reportable
        assert Metric(value=10.0, coverage=0.5, n_samples=5).is_reportable

    def test_serialisation_always_carries_coverage(self) -> None:
        payload = Metric(value=1.0, coverage=0.4, confidence=0.3, n_samples=9).to_dict()
        for key in ("value", "coverage", "confidence", "n_samples", "basis", "reportable"):
            assert key in payload

    def test_confidence_bands(self) -> None:
        assert Confidence.band(0.9) is Confidence.HIGH
        assert Confidence.band(0.5) is Confidence.PROBABLE
        assert Confidence.band(0.1) is Confidence.UNCERTAIN

    def test_basis_distinguishes_extrapolated_aggregates(self) -> None:
        team = Metric(value=1.0, coverage=0.5, n_samples=3,
                      basis=MetricBasis.INCLUDES_EXTRAPOLATED)
        assert team.basis is not MetricBasis.VALID_ONLY


class TestBallState:
    def test_unknown_is_not_known(self) -> None:
        assert not BallStateKind.UNKNOWN.is_known
        assert BallStateKind.OBSERVED.is_known
        assert BallStateKind.INTERPOLATED.is_known

    def test_coverage_profile_worst(self) -> None:
        profile = CoverageProfile(tracking=0.9, pitch=0.2, ball=0.8, identity=1.0)
        assert profile.worst == pytest.approx(0.2)


class TestSmoother:
    def test_recovers_constant_velocity_from_noise(self) -> None:
        """Why the smoother exists: differencing noisy positions reports many
        times the true speed. Measured on the real clip, the naive estimate was
        21.5 m/s for players moving at walking pace."""
        rng = np.random.default_rng(0)
        times = np.arange(60) / 25.0
        noisy_x = 3.0 * times + rng.normal(0, 0.5, len(times))
        noisy_y = rng.normal(0, 0.5, len(times))

        naive = np.hypot(np.diff(noisy_x), np.diff(noisy_y)) / np.diff(times)
        _, _, vx, vy = _rts_smooth(times, noisy_x, noisy_y)
        smoothed = np.hypot(vx, vy)

        assert np.median(naive) > 8.0, "the naive estimate should be badly inflated"
        assert 2.0 < np.median(smoothed) < 4.5, "the smoother must recover ~3 m/s"

    def test_stationary_player_does_not_drift(self) -> None:
        rng = np.random.default_rng(1)
        times = np.arange(50) / 25.0
        x = 40.0 + rng.normal(0, 0.5, 50)
        y = 30.0 + rng.normal(0, 0.5, 50)
        _, _, vx, vy = _rts_smooth(times, x, y)
        assert np.median(np.hypot(vx, vy)) < 1.5

    def test_short_input_is_returned_unchanged(self) -> None:
        times = np.array([0.0, 0.04])
        x, y = np.array([1.0, 2.0]), np.array([1.0, 1.0])
        sx, sy, _, _ = _rts_smooth(times, x, y)
        assert np.allclose(sx, x) and np.allclose(sy, y)


class TestSegments:
    def test_splits_on_a_gap(self) -> None:
        frames = [0, 1, 2, 3, 40, 41, 42, 43]
        rows = track_frame(frames, [float(f) for f in frames], [0.0] * len(frames))
        assert len(extract_segments(rows, fps=25.0, max_gap_frames=5)) == 2

    def test_tolerates_a_single_dropped_frame(self) -> None:
        frames = [0, 1, 3, 4, 5]
        rows = track_frame(frames, [float(f) for f in frames], [0.0] * len(frames))
        assert len(extract_segments(rows, fps=25.0, max_gap_frames=5)) == 1

    def test_single_sample_segments_are_dropped(self) -> None:
        rows = track_frame([0, 100], [0.0, 1.0], [0.0, 0.0])
        assert extract_segments(rows, fps=25.0, max_gap_frames=5) == []


class TestKinematics:
    def test_distance_of_a_known_walk(self) -> None:
        frames = list(range(50))
        xs = [0.2 * f for f in frames]  # 0.2 m/frame at 25 fps = 5 m/s
        rows = track_frame(frames, xs, [34.0] * len(frames))
        profile = compute_kinematics(1, rows, all_rows_count=50, fps=25.0)
        assert profile.distance_m == pytest.approx(9.8, abs=1.5)
        assert profile.mean_speed_m_s == pytest.approx(5.0, abs=1.2)

    def test_coverage_reflects_unusable_rows(self) -> None:
        frames = list(range(20))
        rows = track_frame(frames, [float(f) for f in frames], [0.0] * 20)
        profile = compute_kinematics(1, rows, all_rows_count=100, fps=25.0)
        assert profile.coverage == pytest.approx(0.2)
        assert profile.metric(1.0, "m").coverage == pytest.approx(0.2)

    def test_no_usable_rows_gives_zero_coverage_not_zero_distance(self) -> None:
        profile = compute_kinematics(1, pd.DataFrame(), all_rows_count=50, fps=25.0)
        assert profile.coverage == 0.0
        assert not profile.metric(profile.distance_m, "m").is_reportable

    def test_impossible_samples_are_discarded(self) -> None:
        """A teleport must not add its metres to the distance total."""
        frames = list(range(20))
        xs = [0.1 * f for f in frames]
        xs[10] = 500.0
        rows = track_frame(frames, xs, [0.0] * 20)
        profile = compute_kinematics(1, rows, all_rows_count=20, fps=25.0)
        assert profile.top_speed_m_s <= 12.0
        assert profile.distance_m < 100.0

    def test_sprint_requires_sustained_speed(self) -> None:
        frames = list(range(40))
        xs = [0.05 * f for f in frames]
        xs[20] += 1.5
        rows = track_frame(frames, xs, [0.0] * 40)
        assert compute_kinematics(1, rows, all_rows_count=40, fps=25.0).n_sprints == 0

    def test_profile_metric_carries_valid_only_basis(self) -> None:
        profile = KinematicProfile(track_id=1, n_rows_total=10, n_rows_usable=10)
        assert profile.metric(1.0, "m").basis is MetricBasis.VALID_ONLY


class TestPossessionTypes:
    def test_span_duration_and_frames(self) -> None:
        span = PossessionSpan(
            start_frame=10, end_frame=20, start_time_s=1.0, end_time_s=2.0,
            state=PossessionState.CONTROLLED, team_id="A", track_id=7,
        )
        assert span.duration_s == pytest.approx(1.0)
        assert span.n_frames == 11
        assert span.to_dict()["state"] == "controlled"

    def test_unknown_state_is_distinct_from_loose(self) -> None:
        assert PossessionState.UNKNOWN is not PossessionState.LOOSE_BALL


class TestEvidence:
    def test_accumulates_reasons_and_measurements(self) -> None:
        evidence = Evidence().add("ball travelled", distance_m=12.4).add("receiver controlled")
        payload = evidence.to_dict()
        assert len(payload["reasons"]) == 2
        assert payload["measurements"]["distance_m"] == pytest.approx(12.4)


# --------------------------------------------------------------------------- #
# Against the real run, when one exists
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_run(repo_root):
    root = repo_root / "outputs"
    if not root.exists():
        return None
    candidates = [
        p for p in root.glob("*/*")
        if (p / "game_state.parquet").exists() and (p / "manifest.json").exists()
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


@pytest.mark.slow
class TestAgainstRealRun:
    def test_context_filters_to_valid_rows_only(self, real_run) -> None:
        if real_run is None:
            pytest.skip("no completed vision run available")
        from visionpitch.analytics.context import load_context

        ctx = load_context(real_run)
        assert ctx.n_frames > 0
        assert len(ctx.valid_players) <= len(ctx.players)
        assert ctx.valid_players.pitch_x.notna().all()
        assert (ctx.valid_players.validation_status == "valid").all()

    def test_possession_never_invents_a_holder(self, real_run) -> None:
        """Phase 1B constraint 1: unknown is first class and never backfilled."""
        if real_run is None:
            pytest.skip("no completed vision run available")
        from visionpitch.analytics.context import load_context
        from visionpitch.analytics.possession import run as run_possession

        ctx = load_context(real_run)
        spans, per_frame, summary = run_possession(ctx)

        for decision in per_frame:
            if not decision.ball_state.is_known:
                assert decision.state is PossessionState.UNKNOWN
        for span in spans:
            if span.state is PossessionState.CONTROLLED:
                assert span.track_id is not None
        assert 0.0 <= summary["unknown_ratio"] <= 1.0

    def test_events_carry_evidence_and_clips(self, real_run) -> None:
        if real_run is None:
            pytest.skip("no completed vision run available")
        from visionpitch.analytics.context import load_context
        from visionpitch.analytics.events import run as run_events
        from visionpitch.analytics.possession import run as run_possession

        ctx = load_context(real_run)
        spans, _, _ = run_possession(ctx)
        events = run_events(ctx, spans)

        assert events, "no events detected on the real run"
        for event in events:
            assert event.clip is not None
            assert event.clip.frame_start <= event.frame_idx <= event.clip.frame_end
            assert 0.0 <= event.confidence <= 1.0
            assert event.evidence.reasons, f"{event.event_type} has no evidence"

    def test_full_analytics_writes_every_artefact(self, real_run, tmp_path) -> None:
        if real_run is None:
            pytest.skip("no completed vision run available")
        import shutil

        from visionpitch.analytics.runner import run_analytics

        work = tmp_path / "run"
        shutil.copytree(real_run, work, ignore=shutil.ignore_patterns("video", "analytics"))
        result = run_analytics(work)

        for name in ("events", "possession", "players", "teams", "timeline",
                     "heatmaps", "networks", "summary", "manifest"):
            assert name in result.outputs

        quality = result.summary["data_quality"]
        for key in ("ball_known_pct", "ball_observed_pct", "valid_player_row_pct",
                    "possession_determinable_pct", "warnings"):
            assert key in quality

    def test_physical_metrics_use_only_valid_rows(self, real_run) -> None:
        """Phase 1B constraint 4, enforced end to end."""
        if real_run is None:
            pytest.skip("no completed vision run available")
        from visionpitch.analytics.context import load_context
        from visionpitch.analytics.kinematics import compute_all

        ctx = load_context(real_run)
        profiles = compute_all(ctx.valid_players, ctx.players, ctx.fps)
        for track_id, profile in profiles.items():
            usable = len(ctx.valid_players[ctx.valid_players.track_id == track_id])
            assert profile.n_rows_usable == usable
            assert profile.coverage <= 1.0
