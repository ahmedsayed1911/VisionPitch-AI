"""Event review and correction store.

Phase 2C, Part 11 -- the workspace missing since Phase 2B.

The invariant this module exists to hold: **a model prediction is immutable**.
A review never edits `events.parquet`. It writes a separate correction record
that references the prediction by id and carries who made it, when, against
which model and which run. The corrected view is produced by *applying*
corrections over predictions at read time, so the raw output and the human
judgement remain independently inspectable forever -- and the difference between
them is exactly the training signal Part 12 wants to export.

A reviewer may also assert that the model produced *nothing* where something
happened (a missed event) or that an interval is genuinely unknowable. Both are
first-class, because a review workflow that can only delete and retype teaches
the model nothing about its blind spots.
"""

from __future__ import annotations

import enum
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column, relationship

from visionpitch.api.store import Base, Store
from visionpitch.common.logging import get_logger

log = get_logger("api.reviews")

REVIEW_SCHEMA_VERSION = "1.0.0"

#: Used when a reviewer is confident an event happened but not who did it.
#: Forcing a name here would fabricate the exact label the model needs to learn.
UNKNOWN_PLAYER = -1


class ReviewAction(str, enum.Enum):
    CONFIRM = "confirm"
    REJECT = "reject"
    RETYPE = "retype"
    REASSIGN = "reassign"
    RETIME = "retime"
    MARK_UNKNOWN = "mark_unknown"
    ADD_MISSED = "add_missed"


def _uuid() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> datetime:
    return datetime.now(UTC)


class EventCorrection(Base):
    """One reviewer decision about one event.

    Append-only. Superseding a decision writes a new row; nothing is updated in
    place, so the review history is reconstructable.
    """

    __tablename__ = "event_corrections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False, index=True)

    #: id of the prediction being reviewed; null for a reviewer-added event
    event_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    action: Mapped[ReviewAction] = mapped_column(Enum(ReviewAction), nullable=False)

    # -- corrected values; null means "unchanged" ---------------------------- #
    corrected_type: Mapped[str | None] = mapped_column(String(48), nullable=True)
    corrected_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    corrected_related_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    corrected_team_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    corrected_start_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    corrected_end_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    note: Mapped[str] = mapped_column(Text, default="")
    reviewer: Mapped[str] = mapped_column(String(120), default="anonymous")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    #: provenance of what was reviewed, so a correction stays interpretable
    #: after the model or config changes
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)

    job = relationship("Job")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "event_id": self.event_id,
            "action": self.action.value,
            "corrected_type": self.corrected_type,
            "corrected_track_id": self.corrected_track_id,
            "corrected_related_track_id": self.corrected_related_track_id,
            "corrected_team_id": self.corrected_team_id,
            "corrected_start_s": self.corrected_start_s,
            "corrected_end_s": self.corrected_end_s,
            "note": self.note,
            "reviewer": self.reviewer,
            "created_at": self.created_at.isoformat(),
            "provenance": self.provenance or {},
        }


class ReviewStore:
    """Correction persistence and the corrected-view projection."""

    def __init__(self, store: Store) -> None:
        self.store = store
        Base.metadata.create_all(store.engine)

    # -- writing -------------------------------------------------------------- #

    def add(
        self,
        job_id: str,
        action: ReviewAction,
        event_id: str | None = None,
        reviewer: str = "anonymous",
        note: str = "",
        provenance: dict | None = None,
        **corrected,
    ) -> EventCorrection:
        if action is not ReviewAction.ADD_MISSED and not event_id:
            raise ValueError(f"{action.value} requires the event_id it refers to")
        if action is ReviewAction.ADD_MISSED and corrected.get("corrected_start_s") is None:
            raise ValueError("add_missed requires corrected_start_s")

        with self.store.session() as session:
            record = EventCorrection(
                job_id=job_id, event_id=event_id, action=action,
                reviewer=reviewer, note=note, provenance=provenance or {},
                **{k: v for k, v in corrected.items() if v is not None},
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def list(self, job_id: str) -> list[EventCorrection]:
        with self.store.session() as session:
            return list(
                session.scalars(
                    select(EventCorrection)
                    .where(EventCorrection.job_id == job_id)
                    .order_by(EventCorrection.created_at)
                )
            )

    def latest_by_event(self, job_id: str) -> dict[str, EventCorrection]:
        """Most recent decision per event. Earlier ones remain on record."""
        latest: dict[str, EventCorrection] = {}
        for record in self.list(job_id):
            if record.event_id:
                latest[record.event_id] = record
        return latest

    # -- reading -------------------------------------------------------------- #

    def corrected_view(self, job_id: str, predictions: list[dict]) -> dict:
        """Predictions with corrections applied, without mutating the source.

        Each row keeps its original values under ``raw`` so a consumer can always
        see what the model actually said.
        """
        latest = self.latest_by_event(job_id)
        added = [
            r for r in self.list(job_id) if r.action is ReviewAction.ADD_MISSED
        ]

        out = []
        for event in predictions:
            record = latest.get(event.get("event_id"))
            entry = dict(event)
            entry["review_status"] = "unreviewed"
            entry["raw"] = {
                "type": event.get("type") or event.get("event_type"),
                "track_id": event.get("track_id"),
                "related_track_id": event.get("related_track_id"),
                "team_id": event.get("team_id"),
                "timestamp_s": event.get("timestamp_s"),
            }
            if record is None:
                out.append(entry)
                continue

            entry["review_status"] = record.action.value
            entry["reviewer"] = record.reviewer
            entry["reviewed_at"] = record.created_at.isoformat()
            entry["review_note"] = record.note

            if record.action is ReviewAction.REJECT:
                entry["rejected"] = True
            if record.corrected_type:
                entry["type"] = record.corrected_type
            if record.corrected_track_id is not None:
                entry["track_id"] = record.corrected_track_id
            if record.corrected_related_track_id is not None:
                entry["related_track_id"] = record.corrected_related_track_id
            if record.corrected_team_id:
                entry["team_id"] = record.corrected_team_id
            if record.corrected_start_s is not None:
                entry["timestamp_s"] = record.corrected_start_s
            out.append(entry)

        for record in added:
            out.append({
                "event_id": f"added:{record.id}",
                "type": record.corrected_type or "unknown",
                "timestamp_s": record.corrected_start_s,
                "track_id": record.corrected_track_id,
                "related_track_id": record.corrected_related_track_id,
                "team_id": record.corrected_team_id,
                "confidence": 1.0,
                "confidence_band": "high",
                "review_status": "add_missed",
                "reviewer": record.reviewer,
                "reviewed_at": record.created_at.isoformat(),
                "review_note": record.note,
                "raw": None,
            })

        out.sort(key=lambda e: e.get("timestamp_s") or 0.0)
        return {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "n_predictions": len(predictions),
            "n_corrections": len(self.list(job_id)),
            "n_added": len(added),
            "events": out,
        }

    # -- active learning ------------------------------------------------------- #

    @staticmethod
    def rank_for_review(events: list[dict], limit: int = 200) -> list[dict]:
        """Order events by how much a human decision would teach the model.

        Highest value is where the model is *uncertain*, not where it is wrong --
        a reviewer cannot know which is which in advance, and confident errors
        are found by sampling, not by ranking on confidence.
        """
        def priority(event: dict) -> tuple:
            confidence = float(event.get("confidence") or 0.0)
            ball_coverage = float(event.get("ball_coverage") or 0.0)
            unknown_actor = event.get("track_id") in (None, UNKNOWN_PLAYER)
            interpolated = event.get("ball_state") == "interpolated"
            unreviewed = event.get("review_status", "unreviewed") == "unreviewed"
            # Sort key: unreviewed first, then genuinely ambiguous cases.
            return (
                not unreviewed,
                -(1.0 - confidence),
                -(1.0 - ball_coverage),
                -float(unknown_actor),
                -float(interpolated),
            )

        return sorted(events, key=priority)[:limit]

    def export_training_examples(
        self, job_id: str, predictions: list[dict], destination: str | Path
    ) -> dict:
        """Write reviewed events as a versioned, fingerprinted dataset.

        Deliberately a separate explicit step, never automatic: a correction is
        an opinion until someone decides to build a dataset from it, and a
        pipeline that silently retrains on review clicks has no reproducible
        training set.
        """
        import hashlib

        view = self.corrected_view(job_id, predictions)
        reviewed = [
            e for e in view["events"]
            if e.get("review_status", "unreviewed") != "unreviewed"
        ]
        payload = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "job_id": job_id,
            "exported_at": _now().isoformat(),
            "n_reviewed": len(reviewed),
            "examples": reviewed,
        }
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(payload["examples"], sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info(
            "exported %d reviewed example(s) -> %s (%s)",
            len(reviewed), destination, payload["fingerprint"],
        )
        return payload
