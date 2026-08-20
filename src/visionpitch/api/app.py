"""VisionPitch REST API.

Serves precomputed analytics artefacts. Nothing here recomputes football
analysis: the pipeline writes JSON and Parquet into the run directory, and these
endpoints read them. That keeps responses fast, keeps the dashboard and the
exports in agreement by construction, and means an API restart cannot change a
number.
"""

from __future__ import annotations

import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from visionpitch import __version__
from visionpitch.api.store import JobStatus, Store, load_analytics_json
from visionpitch.api.worker import JobWorker
from visionpitch.common.logging import configure_logging, get_logger

log = get_logger("api")

#: Upload guard. A football match is large but not unbounded, and an unchecked
#: upload endpoint is a denial-of-service vector.
MAX_UPLOAD_BYTES = int(os.environ.get("VISIONPITCH_MAX_UPLOAD_BYTES", 8 * 1024**3))
ALLOWED_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".ts", ".mpg", ".mpeg"}


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class CorrectionRequest(BaseModel):
    """A reviewer decision.

    Defined at module level, not inside ``create_app``: this module uses
    ``from __future__ import annotations``, so FastAPI resolves the handler's
    type hints against the *module* namespace. A locally-scoped model is
    invisible there and the parameter silently degrades to a query field.
    """

    action: str
    event_id: str | None = None
    reviewer: str = "anonymous"
    note: str = ""
    corrected_type: str | None = None
    corrected_track_id: int | None = None
    corrected_related_track_id: int | None = None
    corrected_team_id: str | None = None
    corrected_start_s: float | None = None
    corrected_end_s: float | None = None


class Settings:
    """Runtime paths, overridable by environment for deployment."""

    def __init__(self) -> None:
        root = Path(os.environ.get("VISIONPITCH_DATA_ROOT", "data/api")).resolve()
        self.data_root = root
        self.uploads = root / "uploads"
        self.outputs = Path(
            os.environ.get("VISIONPITCH_OUTPUT_ROOT", str(root / "runs"))
        ).resolve()
        self.database_url = os.environ.get(
            "VISIONPITCH_DATABASE_URL", f"sqlite:///{root / 'visionpitch.db'}"
        )
        for path in (self.data_root, self.uploads, self.outputs):
            path.mkdir(parents=True, exist_ok=True)


def create_app(settings: Settings | None = None, worker: JobWorker | None = None) -> FastAPI:
    configure_logging()
    settings = settings or Settings()
    store = Store(settings.database_url)
    job_worker = worker or JobWorker(store, settings.outputs)
    job_worker.start()

    app = FastAPI(
        title="VisionPitch AI",
        version=__version__,
        description="Football video analysis: projects, jobs and match analytics.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("VISIONPITCH_CORS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store
    app.state.worker = job_worker
    app.state.settings = settings

    def get_store() -> Store:
        return store

    # -- helpers ------------------------------------------------------------- #

    def require_job(job_id: str, need_analytics: bool = True):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, f"job {job_id} not found")
        if need_analytics and job.analytics_dir is None:
            raise HTTPException(
                409,
                f"job {job_id} has no analytics yet (status: {job.status.value})",
            )
        return job

    def artefact(job, name: str) -> Any:
        data = load_analytics_json(job, name)
        if data is None:
            raise HTTPException(404, f"{name} not available for job {job.id}")
        return data

    # -- meta ----------------------------------------------------------------- #

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    # -- projects -------------------------------------------------------------- #

    @app.get("/api/projects", tags=["projects"])
    def list_projects(db: Store = Depends(get_store)) -> list[dict]:
        return [p.to_dict() for p in db.list_projects()]

    @app.post("/api/projects", status_code=201, tags=["projects"])
    def create_project(payload: ProjectCreate, db: Store = Depends(get_store)) -> dict:
        return db.create_project(payload.name, payload.description).to_dict()

    @app.get("/api/projects/{project_id}", tags=["projects"])
    def get_project(project_id: str, db: Store = Depends(get_store)) -> dict:
        project = db.get_project(project_id)
        if project is None:
            raise HTTPException(404, f"project {project_id} not found")
        return project.to_dict()

    @app.delete("/api/projects/{project_id}", tags=["projects"])
    def delete_project(project_id: str, db: Store = Depends(get_store)) -> dict:
        if not db.delete_project(project_id):
            raise HTTPException(404, f"project {project_id} not found")
        return {"deleted": project_id}

    # -- jobs -------------------------------------------------------------------- #

    @app.get("/api/jobs", tags=["jobs"])
    def list_jobs(project_id: str | None = None, db: Store = Depends(get_store)) -> list[dict]:
        return [j.to_summary() for j in db.list_jobs(project_id)]

    @app.post("/api/projects/{project_id}/jobs", status_code=201, tags=["jobs"])
    async def upload_match(
        project_id: str,
        file: UploadFile = File(...),
        mode: str = Form("balanced"),
        name: str = Form(""),
        db: Store = Depends(get_store),
    ) -> dict:
        """Upload a match video and queue it for analysis."""
        if db.get_project(project_id) is None:
            raise HTTPException(404, f"project {project_id} not found")
        if mode not in ("fast_preview", "balanced", "max_accuracy"):
            raise HTTPException(422, f"unknown analysis mode {mode!r}")

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                422,
                f"unsupported file type {suffix!r}; expected one of "
                f"{sorted(ALLOWED_SUFFIXES)}",
            )

        job = db.create_job(
            project_id=project_id,
            name=name or Path(file.filename or "match").stem,
            mode=mode,
            video_filename=file.filename or "match.mp4",
            status=JobStatus.PENDING,
        )

        destination = settings.uploads / f"{job.id}{suffix}"
        written = 0
        try:
            with destination.open("wb") as handle:
                while chunk := await file.read(4 * 1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            413,
                            f"upload exceeds the {MAX_UPLOAD_BYTES // 1024**3} GB limit",
                        )
                    handle.write(chunk)
        except HTTPException:
            destination.unlink(missing_ok=True)
            db.delete_job(job.id)
            raise

        db.update_job(job.id, video_path=str(destination), video_bytes=written)
        job_worker.submit(job.id)
        log.info("queued job %s (%s, %.1f MB)", job.id, mode, written / 1e6)

        return db.get_job(job.id).to_dict()

    @app.get("/api/jobs/{job_id}", tags=["jobs"])
    def get_job(job_id: str) -> dict:
        return require_job(job_id, need_analytics=False).to_dict()

    @app.get("/api/jobs/{job_id}/progress", tags=["jobs"])
    def job_progress(job_id: str) -> dict:
        job = require_job(job_id, need_analytics=False)
        return {
            "id": job.id,
            "status": job.status.value,
            "stage": job.stage,
            "progress": round(job.progress, 3),
            "error": job.error,
        }

    @app.delete("/api/jobs/{job_id}", tags=["jobs"])
    def delete_job(job_id: str, db: Store = Depends(get_store)) -> dict:
        if not db.delete_job(job_id):
            raise HTTPException(404, f"job {job_id} not found")
        return {"deleted": job_id}

    # -- analytics ---------------------------------------------------------------- #

    @app.get("/api/jobs/{job_id}/summary", tags=["analytics"])
    def match_summary(job_id: str) -> dict:
        return artefact(require_job(job_id), "summary.json")

    @app.get("/api/jobs/{job_id}/players", tags=["analytics"])
    def players(job_id: str, team_id: str | None = None) -> list[dict]:
        data = artefact(require_job(job_id), "player_stats.json")
        items = list(data.values())
        if team_id:
            items = [p for p in items if p["team_id"] == team_id]
        # Best-covered players first: a dashboard listing 116 tracks should lead
        # with the ones whose numbers mean something.
        return sorted(items, key=lambda p: -p["coverage"]["tracking"])

    @app.get("/api/jobs/{job_id}/players/{track_id}", tags=["analytics"])
    def player(job_id: str, track_id: int) -> dict:
        data = artefact(require_job(job_id), "player_stats.json")
        entry = data.get(str(track_id))
        if entry is None:
            raise HTTPException(404, f"player {track_id} not found")
        return entry

    @app.get("/api/jobs/{job_id}/goalkeepers", tags=["analytics"])
    def goalkeepers(job_id: str) -> list[dict]:
        return list(artefact(require_job(job_id), "goalkeeper_stats.json").values())

    @app.get("/api/jobs/{job_id}/teams", tags=["analytics"])
    def teams(job_id: str) -> list[dict]:
        return list(artefact(require_job(job_id), "team_stats.json").values())

    @app.get("/api/jobs/{job_id}/events", tags=["analytics"])
    def events(
        job_id: str,
        event_type: str | None = None,
        team_id: str | None = None,
        track_id: int | None = None,
        min_confidence: float = 0.0,
        half: int | None = None,
        limit: int = 2000,
    ) -> list[dict]:
        timeline = artefact(require_job(job_id), "timeline.json")
        items = timeline["events"]
        if event_type:
            wanted = set(event_type.split(","))
            items = [e for e in items if e["type"] in wanted]
        if team_id:
            items = [e for e in items if e["team_id"] == team_id]
        if track_id is not None:
            items = [
                e for e in items
                if e["track_id"] == track_id or e["related_track_id"] == track_id
            ]
        if half is not None:
            items = [e for e in items if e["half"] == half]
        items = [e for e in items if e["confidence"] >= min_confidence]
        return items[:limit]

    @app.get("/api/jobs/{job_id}/timeline", tags=["analytics"])
    def timeline(job_id: str) -> dict:
        return artefact(require_job(job_id), "timeline.json")

    @app.get("/api/jobs/{job_id}/heatmaps", tags=["analytics"])
    def heatmaps(
        job_id: str,
        track_id: int | None = None,
        team_id: str | None = None,
        kind: str | None = None,
    ) -> list[dict]:
        data = artefact(require_job(job_id), "heatmaps.json")
        items = list(data.get("players", [])) + list(data.get("teams", []))
        if track_id is not None:
            items = [h for h in items if h["track_id"] == track_id]
        if team_id:
            items = [h for h in items if h["team_id"] == team_id and h["track_id"] is None]
        if kind:
            items = [h for h in items if h["kind"] == kind]
        return items

    @app.get("/api/jobs/{job_id}/networks", tags=["analytics"])
    def networks(job_id: str, team_id: str | None = None, window: str | None = None):
        items = artefact(require_job(job_id), "networks.json")
        if team_id:
            items = [n for n in items if n["team_id"] == team_id]
        if window:
            items = [n for n in items if n["window"] == window]
        return items

    # -- event review ------------------------------------------------------------- #

    from visionpitch.api.reviews import ReviewAction, ReviewStore

    review_store = ReviewStore(store)
    app.state.reviews = review_store

    def _predictions(job) -> list[dict]:
        return artefact(job, "timeline.json")["events"]

    @app.get("/api/jobs/{job_id}/review", tags=["review"])
    def review_queue(
        job_id: str,
        event_type: str | None = None,
        max_confidence: float = 1.0,
        unreviewed_only: bool = False,
        limit: int = 200,
    ) -> dict:
        """Events ranked by how much a human decision would teach the model."""
        job = require_job(job_id)
        view = review_store.corrected_view(job_id, _predictions(job))

        items = view["events"]
        if event_type:
            wanted = set(event_type.split(","))
            items = [e for e in items if e.get("type") in wanted]
        items = [e for e in items if float(e.get("confidence") or 0) <= max_confidence]
        if unreviewed_only:
            items = [e for e in items if e.get("review_status") == "unreviewed"]

        return {
            "job_id": job_id,
            "n_total": view["n_predictions"],
            "n_corrections": view["n_corrections"],
            "n_matching": len(items),
            "events": ReviewStore.rank_for_review(items, limit=limit),
        }

    @app.post("/api/jobs/{job_id}/review", status_code=201, tags=["review"])
    def submit_correction(job_id: str, payload: CorrectionRequest) -> dict:
        """Record a reviewer decision. Never edits the stored prediction."""
        job = require_job(job_id)
        try:
            action = ReviewAction(payload.action)
        except ValueError:
            raise HTTPException(
                422, f"unknown action {payload.action!r}; expected one of "
                     f"{[a.value for a in ReviewAction]}"
            ) from None

        manifest_path = Path(job.run_dir) / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists() else {}
        )
        provenance = {
            "run_fingerprint": manifest.get("config_fingerprint"),
            "models": {
                k: v.get("weights_sha256")
                for k, v in (manifest.get("models") or {}).items()
            },
            "analytics_schema_version": (
                json.loads(
                    (job.analytics_dir / "manifest.json").read_text(encoding="utf-8")
                ).get("analytics_schema_version")
                if job.analytics_dir and (job.analytics_dir / "manifest.json").exists()
                else None
            ),
        }

        try:
            record = review_store.add(
                job_id=job_id, action=action, event_id=payload.event_id,
                reviewer=payload.reviewer, note=payload.note, provenance=provenance,
                corrected_type=payload.corrected_type,
                corrected_track_id=payload.corrected_track_id,
                corrected_related_track_id=payload.corrected_related_track_id,
                corrected_team_id=payload.corrected_team_id,
                corrected_start_s=payload.corrected_start_s,
                corrected_end_s=payload.corrected_end_s,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return record.to_dict()

    @app.get("/api/jobs/{job_id}/review/corrections", tags=["review"])
    def list_corrections(job_id: str) -> list[dict]:
        require_job(job_id, need_analytics=False)
        return [c.to_dict() for c in review_store.list(job_id)]

    @app.get("/api/jobs/{job_id}/review/corrected", tags=["review"])
    def corrected_events(job_id: str) -> dict:
        """Predictions with corrections applied. Raw values kept per row."""
        job = require_job(job_id)
        return review_store.corrected_view(job_id, _predictions(job))

    @app.post("/api/jobs/{job_id}/review/export", tags=["review"])
    def export_reviewed(job_id: str) -> dict:
        """Build a versioned training dataset from reviewed events."""
        job = require_job(job_id)
        # Written outside the run directory on purpose: a run is the immutable
        # record of what the models produced, and a human-labelled dataset does
        # not belong inside it.
        destination = settings.data_root / "reviews" / f"{job_id}.json"
        return review_store.export_training_examples(
            job_id, _predictions(job), destination
        )

    # -- media and downloads --------------------------------------------------------- #

    @app.get("/api/jobs/{job_id}/video", tags=["media"])
    def annotated_video(job_id: str, kind: str = "annotated"):
        """Stream a rendered video. ``kind`` is annotated | radar | combined."""
        job = require_job(job_id, need_analytics=False)
        if not job.run_dir:
            raise HTTPException(404, "no run directory for this job")
        for suffix in (".mp4", ".avi"):
            path = Path(job.run_dir) / "video" / f"{kind}{suffix}"
            if path.exists():
                return FileResponse(path, media_type="video/mp4", filename=path.name)
        raise HTTPException(404, f"no {kind} video for job {job_id}")

    @app.get("/api/jobs/{job_id}/download/{artefact_name}", tags=["downloads"])
    def download(job_id: str, artefact_name: str):
        """Download a stored artefact by name."""
        job = require_job(job_id, need_analytics=False)
        if not job.run_dir:
            raise HTTPException(404, "no run directory for this job")

        candidates = {
            "game_state": Path(job.run_dir) / "game_state.parquet",
            "frames": Path(job.run_dir) / "frames.parquet",
            "tracks": Path(job.run_dir) / "tracks.parquet",
            "detections": Path(job.run_dir) / "detections.parquet",
            "calibration": Path(job.run_dir) / "calibration.parquet",
            "events": Path(job.run_dir) / "analytics" / "events.parquet",
            "possession": Path(job.run_dir) / "analytics" / "possession.parquet",
            "manifest": Path(job.run_dir) / "manifest.json",
        }
        path = candidates.get(artefact_name)
        if path is None:
            raise HTTPException(
                404, f"unknown artefact {artefact_name!r}; expected one of "
                     f"{sorted(candidates)}"
            )
        if not path.exists():
            raise HTTPException(404, f"{artefact_name} has not been produced for this job")
        return FileResponse(path, filename=f"{job.id}_{path.name}")

    @app.get("/api/jobs/{job_id}/export/{table}.csv", tags=["downloads"])
    def export_csv(job_id: str, table: str):
        """CSV export of an analytics table."""
        import pandas as pd

        job = require_job(job_id)
        sources = {
            "events": job.analytics_dir / "events.parquet",
            "possession": job.analytics_dir / "possession.parquet",
            "players": None,
            "teams": None,
        }
        if table not in sources:
            raise HTTPException(404, f"unknown table {table!r}")

        if table in ("players", "teams"):
            payload = artefact(job, f"{'player' if table == 'players' else 'team'}_stats.json")
            rows = []
            for entry in payload.values():
                flat = {
                    "track_id": entry.get("track_id"),
                    "team_id": entry.get("team_id"),
                    "display_name": entry.get("display_name"),
                    **{f"coverage_{k}": v for k, v in entry["coverage"].items()},
                }
                for name, metric in entry["metrics"].items():
                    flat[name] = metric["value"]
                    # Coverage travels with every exported value: a CSV that
                    # drops it lets a number be quoted without its caveat.
                    flat[f"{name}__coverage"] = metric["coverage"]
                rows.append(flat)
            frame = pd.DataFrame(rows)
        else:
            frame = pd.read_parquet(sources[table])

        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{job.id}_{table}.csv"'},
        )

    @app.get("/api/jobs/{job_id}/report", tags=["downloads"])
    def match_report(job_id: str) -> dict:
        """A structured match report assembled from stored analytics only."""
        job = require_job(job_id)
        summary = artefact(job, "summary.json")
        teams_data = artefact(job, "team_stats.json")
        players_data = artefact(job, "player_stats.json")

        ranked = sorted(
            players_data.values(),
            key=lambda p: -(p["metrics"].get("touches", {}).get("value") or 0),
        )
        return {
            "job": job.to_dict(),
            "data_quality": summary.get("data_quality", {}),
            "possession": summary.get("possession", {}),
            "event_counts": summary.get("event_counts", {}),
            "teams": teams_data,
            "top_players": ranked[:10],
            "caveats": summary.get("data_quality", {}).get("warnings", []),
        }

    return app


def _unused():  # pragma: no cover
    return json, shutil


app = None  # populated by run_server


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    global app
    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload)
