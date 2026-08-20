"""API integration tests.

The worker is stubbed rather than run: these tests verify the HTTP contract and
the artefact plumbing, not the ninety seconds of GPU work behind them. The
pipeline itself is covered by the Phase 1 integration tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from visionpitch.api.app import Settings, create_app  # noqa: E402
from visionpitch.api.store import JobStatus  # noqa: E402
from visionpitch.api.worker import JobWorker  # noqa: E402


class StubWorker(JobWorker):
    """Records submissions instead of running the GPU pipeline."""

    def __init__(self, store, output_root):
        super().__init__(store, output_root)
        self.submitted: list[str] = []

    def start(self) -> None:  # noqa: D102
        pass

    def submit(self, job_id: str) -> None:  # noqa: D102
        self.submitted.append(job_id)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONPITCH_DATA_ROOT", str(tmp_path))
    settings = Settings()
    from visionpitch.api.store import Store

    store = Store(settings.database_url)
    worker = StubWorker(store, settings.outputs)
    app = create_app(settings, worker=worker)
    app.state.store = store
    with TestClient(app) as test_client:
        yield test_client, app


@pytest.fixture(scope="module")
def completed_run(repo_root: Path):
    root = repo_root / "outputs"
    if not root.exists():
        return None
    candidates = [p for p in root.glob("*/*") if (p / "analytics" / "summary.json").exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


class TestMetaAndProjects:
    def test_health(self, client) -> None:
        test_client, _ = client
        assert test_client.get("/api/health").json()["status"] == "ok"

    def test_project_lifecycle(self, client) -> None:
        test_client, _ = client
        created = test_client.post("/api/projects", json={"name": "Season 24/25"})
        assert created.status_code == 201
        project_id = created.json()["id"]

        assert len(test_client.get("/api/projects").json()) == 1
        assert test_client.get(f"/api/projects/{project_id}").json()["name"] == "Season 24/25"

        assert test_client.delete(f"/api/projects/{project_id}").status_code == 200
        assert test_client.get("/api/projects").json() == []

    def test_missing_project_is_404(self, client) -> None:
        test_client, _ = client
        assert test_client.get("/api/projects/nope").status_code == 404

    def test_project_name_is_required(self, client) -> None:
        test_client, _ = client
        assert test_client.post("/api/projects", json={"name": ""}).status_code == 422


class TestUpload:
    def test_rejects_a_non_video(self, client) -> None:
        test_client, _ = client
        project_id = test_client.post("/api/projects", json={"name": "p"}).json()["id"]
        response = test_client.post(
            f"/api/projects/{project_id}/jobs",
            files={"file": ("notes.txt", b"not a video", "text/plain")},
            data={"mode": "balanced"},
        )
        assert response.status_code == 422
        assert "unsupported file type" in response.text

    def test_rejects_an_unknown_mode(self, client) -> None:
        test_client, _ = client
        project_id = test_client.post("/api/projects", json={"name": "p"}).json()["id"]
        response = test_client.post(
            f"/api/projects/{project_id}/jobs",
            files={"file": ("m.mp4", b"\x00" * 64, "video/mp4")},
            data={"mode": "turbo"},
        )
        assert response.status_code == 422

    def test_accepted_upload_is_queued(self, client) -> None:
        test_client, app = client
        project_id = test_client.post("/api/projects", json={"name": "p"}).json()["id"]
        response = test_client.post(
            f"/api/projects/{project_id}/jobs",
            files={"file": ("match.mp4", b"\x00" * 2048, "video/mp4")},
            data={"mode": "balanced", "name": "Match 1"},
        )
        assert response.status_code == 201
        job = response.json()
        assert job["status"] == "pending"
        assert job["mode"] == "balanced"
        assert app.state.worker.submitted == [job["id"]]

        progress = test_client.get(f"/api/jobs/{job['id']}/progress").json()
        assert progress["status"] == "pending"

    def test_analytics_endpoints_409_before_completion(self, client) -> None:
        """A dashboard must be told the analysis is not ready, not given an empty
        object it would render as a match with no events."""
        test_client, _ = client
        project_id = test_client.post("/api/projects", json={"name": "p"}).json()["id"]
        job = test_client.post(
            f"/api/projects/{project_id}/jobs",
            files={"file": ("m.mp4", b"\x00" * 64, "video/mp4")},
            data={"mode": "balanced"},
        ).json()
        assert test_client.get(f"/api/jobs/{job['id']}/summary").status_code == 409


@pytest.mark.slow
class TestAnalyticsEndpoints:
    def _job(self, client, completed_run):
        test_client, app = client
        project_id = test_client.post("/api/projects", json={"name": "p"}).json()["id"]
        job = app.state.store.create_job(
            project_id=project_id, name="real", mode="balanced",
            status=JobStatus.COMPLETED, run_dir=str(completed_run),
            video_filename="clip.mp4", progress=1.0, stage="done",
        )
        return test_client, job.id

    def test_every_analytics_endpoint_responds(self, client, completed_run) -> None:
        if completed_run is None:
            pytest.skip("no run with analytics available")
        test_client, job_id = self._job(client, completed_run)

        for path in (
            f"/api/jobs/{job_id}/summary",
            f"/api/jobs/{job_id}/players",
            f"/api/jobs/{job_id}/teams",
            f"/api/jobs/{job_id}/goalkeepers",
            f"/api/jobs/{job_id}/timeline",
            f"/api/jobs/{job_id}/heatmaps",
            f"/api/jobs/{job_id}/networks",
            f"/api/jobs/{job_id}/events",
            f"/api/jobs/{job_id}/report",
        ):
            assert test_client.get(path).status_code == 200, path

    def test_every_player_metric_carries_coverage(self, client, completed_run) -> None:
        if completed_run is None:
            pytest.skip("no run with analytics available")
        test_client, job_id = self._job(client, completed_run)
        players = test_client.get(f"/api/jobs/{job_id}/players").json()
        assert players
        for player in players[:5]:
            for key in ("tracking", "pitch", "ball", "identity"):
                assert key in player["coverage"]
            for name, metric in player["metrics"].items():
                assert "coverage" in metric, f"{name} has no coverage"
                assert "basis" in metric
                assert "reportable" in metric

    def test_event_filters(self, client, completed_run) -> None:
        if completed_run is None:
            pytest.skip("no run with analytics available")
        test_client, job_id = self._job(client, completed_run)

        every = test_client.get(f"/api/jobs/{job_id}/events").json()
        passes = test_client.get(f"/api/jobs/{job_id}/events?event_type=pass").json()
        confident = test_client.get(f"/api/jobs/{job_id}/events?min_confidence=0.9").json()

        assert len(passes) <= len(every)
        assert all(e["type"] == "pass" for e in passes)
        assert all(e["confidence"] >= 0.9 for e in confident)

    def test_events_are_seekable(self, client, completed_run) -> None:
        """Every event must carry what a player needs to jump to it."""
        if completed_run is None:
            pytest.skip("no run with analytics available")
        test_client, job_id = self._job(client, completed_run)
        for event in test_client.get(f"/api/jobs/{job_id}/events?limit=25").json():
            assert event["timestamp_s"] >= 0
            assert event["frame_idx"] >= 0
            assert event["clip"] is not None

    def test_csv_export_includes_coverage_columns(self, client, completed_run) -> None:
        if completed_run is None:
            pytest.skip("no run with analytics available")
        test_client, job_id = self._job(client, completed_run)
        response = test_client.get(f"/api/jobs/{job_id}/export/players.csv")
        assert response.status_code == 200
        header = response.text.splitlines()[0]
        assert "coverage_tracking" in header
        assert "distance_m__coverage" in header

    def test_parquet_download(self, client, completed_run) -> None:
        if completed_run is None:
            pytest.skip("no run with analytics available")
        test_client, job_id = self._job(client, completed_run)
        response = test_client.get(f"/api/jobs/{job_id}/download/events")
        assert response.status_code == 200
        assert response.content[:4] == b"PAR1"

    def test_unknown_artefact_is_404(self, client, completed_run) -> None:
        if completed_run is None:
            pytest.skip("no run with analytics available")
        test_client, job_id = self._job(client, completed_run)
        assert test_client.get(f"/api/jobs/{job_id}/download/nonsense").status_code == 404


def test_settings_uses_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONPITCH_DATA_ROOT", str(tmp_path / "custom"))
    settings = Settings()
    assert settings.data_root == (tmp_path / "custom").resolve()
    assert settings.uploads.exists()
    _ = os
