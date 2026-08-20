"""Local annotation server for human ball review.

Broadcast ball annotation workflow, step 4.

A local FastAPI app plus one HTML page. Runs entirely offline against an
extracted review package; nothing is uploaded and no network call is made.

Design rules that matter for data quality
-----------------------------------------
**Neither model is shown as the answer.** The two proposals are drawn in
different colours with neutral labels, both off by default on the first view of
a frame, and the reviewer must press a key to accept one. If one were pre-loaded
as the working point, the dataset would inherit that model's bias wherever the
reviewer was unsure — which is precisely where the measurement matters most.

**Accepting a proposal is recorded as such.** ``accepted_proposal_from`` marks
frames where the human agreed rather than placed a point. A dataset made
entirely of accepted proposals is a model measuring itself, and this field makes
that visible instead of invisible.

**Writes are appends.** Every save adds a line. Re-annotating leaves the earlier
decision in the file, and a session killed mid-review loses at most the frame in
progress.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from visionpitch.annotation.schema import (
    ANNOTATION_SCHEMA_VERSION,
    DEFAULT_BALL_RADIUS_PX,
    MAX_BALL_RADIUS_PX,
    MIN_BALL_RADIUS_PX,
    AnnotationError,
    AnnotationStore,
    BallAnnotation,
    BallVisibility,
    IgnoreReason,
    ReviewStatus,
)
from visionpitch.common.logging import get_logger

log = get_logger("annotation.server")


class AnnotatePayload(BaseModel):
    """Request body for a reviewer decision.

    Module level, not nested inside ``create_app``: this file uses
    ``from __future__ import annotations``, so FastAPI resolves the handler's
    type hints against the *module* namespace. A locally-scoped model is
    invisible there and the parameter silently degrades to a query field, which
    fails as a confusing 422 on every save.
    """

    frame_id: str
    visibility: str
    ignore_reason: str = "none"
    centre_x: float | None = None
    centre_y: float | None = None
    radius_px: float | None = None
    #: Omitted by the UI: the box is derived from centre and radius so the two
    #: cannot drift apart. Accepted here for any caller that wants to send one.
    bbox: list[float] | None = None
    annotation_confidence: float = 1.0
    ambiguity_reason: str = ""
    reviewer: str = "reviewer"
    review_status: str = ReviewStatus.FIRST_PASS.value
    accepted_proposal_from: str | None = None


def create_app(package_root: str | Path, qc: bool = False, qc_total: int = 125):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    store = AnnotationStore(package_root)
    samples = store.load_samples()
    predictions = store.load_predictions()
    if not samples:
        raise RuntimeError(
            f"no samples in {package_root}; build the review package first"
        )

    if qc:
        # QC mode serves only the prioritised unreviewed queue, in quota order.
        # The full sample stays on disk untouched -- this narrows what is shown,
        # it does not resample or discard anything.
        from visionpitch.annotation.qc import build_qc_queue

        queue = build_qc_queue(samples, store.load_annotations(), qc_total)
        order = list(queue.frame_ids)
        log.info("QC mode: serving %d prioritised frame(s)", len(order))
    else:
        order = sorted(samples, key=lambda k: samples[k].frame_idx)
    position = {frame_id: i for i, frame_id in enumerate(order)}

    app = FastAPI(title="VisionPitch ball annotation", version=ANNOTATION_SCHEMA_VERSION)

    # -- pages ---------------------------------------------------------------- #

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "annotate.html").read_text(
            encoding="utf-8"
        )

    @app.get("/api/image/{frame_id}")
    def image(frame_id: str, offset: int = 0):
        sample = samples.get(frame_id)
        if sample is None:
            raise HTTPException(404, f"unknown frame {frame_id}")
        path = Path(sample.image_path)
        if offset:
            candidate = path.parent / "context" / f"{frame_id}{offset:+d}.jpg"
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise HTTPException(404, f"no image on disk for {frame_id}")
        return FileResponse(path, media_type="image/jpeg")

    # -- data ------------------------------------------------------------------ #

    @app.get("/api/frames")
    def list_frames() -> dict:
        annotations = store.load_annotations()
        return {
            "n": len(order),
            "frames": [
                {
                    "frame_id": frame_id,
                    "frame_idx": samples[frame_id].frame_idx,
                    "timestamp_s": samples[frame_id].timestamp_s,
                    "category": samples[frame_id].sampling_category.value,
                    "window_id": samples[frame_id].window_id,
                    "annotated": frame_id in annotations,
                    "visibility": (
                        annotations[frame_id].visibility.value
                        if frame_id in annotations else None
                    ),
                    "review_status": (
                        annotations[frame_id].review_status.value
                        if frame_id in annotations else "unreviewed"
                    ),
                }
                for frame_id in order
            ],
        }

    @app.get("/api/frame/{frame_id}")
    def frame(frame_id: str) -> dict:
        sample = samples.get(frame_id)
        if sample is None:
            raise HTTPException(404, f"unknown frame {frame_id}")
        annotations = store.load_annotations()
        index = position[frame_id]
        existing = annotations.get(frame_id)
        context = sorted(
            int(p.stem.replace(frame_id, ""))
            for p in (Path(sample.image_path).parent / "context").glob(f"{frame_id}[+-]*.jpg")
        ) if Path(sample.image_path).parent.joinpath("context").exists() else []

        return {
            "sample": sample.to_dict(),
            # Proposals, explicitly labelled. The UI must not preselect either.
            "predictions": [p.to_dict() for p in predictions.get(frame_id, [])],
            "annotation": existing.to_dict() if existing else None,
            "position": index,
            "total": len(order),
            "previous_id": order[index - 1] if index > 0 else None,
            "next_id": order[index + 1] if index + 1 < len(order) else None,
            "context_offsets": context,
        }

    @app.post("/api/annotate", status_code=201)
    def annotate(payload: AnnotatePayload) -> dict:
        sample = samples.get(payload.frame_id)
        if sample is None:
            raise HTTPException(404, f"unknown frame {payload.frame_id}")
        try:
            annotation = BallAnnotation(
                frame_id=payload.frame_id,
                visibility=BallVisibility(payload.visibility),
                ignore_reason=IgnoreReason(payload.ignore_reason),
                centre_x=payload.centre_x,
                centre_y=payload.centre_y,
                radius_px=payload.radius_px,
                bbox=payload.bbox,
                annotation_confidence=payload.annotation_confidence,
                ambiguity_reason=payload.ambiguity_reason,
                reviewer=payload.reviewer,
                review_status=ReviewStatus(payload.review_status),
                accepted_proposal_from=payload.accepted_proposal_from,
            )
            store.append(annotation, sample)
        except (ValueError, AnnotationError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return annotation.to_dict()

    @app.get("/api/defaults")
    def defaults() -> dict:
        """Radius bounds, so the UI cannot drift from what validation accepts."""
        return {
            "default_radius_px": DEFAULT_BALL_RADIUS_PX,
            "min_radius_px": MIN_BALL_RADIUS_PX,
            "max_radius_px": MAX_BALL_RADIUS_PX,
            "schema_version": ANNOTATION_SCHEMA_VERSION,
        }

    @app.get("/api/progress")
    def progress() -> dict:
        payload = store.progress()
        payload["qc_mode"] = qc
        if qc:
            payload["n_samples"] = len(order)
            payload["n_annotated"] = sum(
                1 for f in order if f in store.load_annotations()
            )
            payload["n_remaining"] = payload["n_samples"] - payload["n_annotated"]
        return payload

    @app.get("/api/qc")
    def qc_status() -> dict:
        """Live quota state, so the reviewer can see which gap is still open."""
        from visionpitch.annotation.qc import progress_report

        return progress_report(package_root, qc_total)

    @app.get("/api/next-unreviewed")
    def next_unreviewed(after: str | None = None) -> dict:
        annotations = store.load_annotations()
        start = position.get(after, -1) + 1 if after else 0
        for frame_id in order[start:] + order[:start]:
            if frame_id not in annotations:
                return {"frame_id": frame_id}
        return {"frame_id": None}

    @app.get("/api/manifest")
    def manifest() -> dict:
        return store.manifest()

    app.state.store = store
    app.state.samples = samples
    return app


def serve(
    package_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8009,
    qc: bool = False,
    qc_total: int = 125,
) -> None:
    import uvicorn

    app = create_app(package_root, qc=qc, qc_total=qc_total)
    log.info("annotation UI at http://%s:%d%s", host, port, " (QC mode)" if qc else "")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def package_summary(package_root: str | Path) -> dict:
    store = AnnotationStore(package_root)
    return {**store.manifest(), "progress": store.progress()}


__all__ = ["create_app", "package_summary", "serve"]


def _unused() -> None:  # pragma: no cover - keeps json import meaningful
    json.dumps({})
