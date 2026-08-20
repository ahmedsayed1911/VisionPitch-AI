"""Project and job persistence.

SQLite via SQLAlchemy rather than PostgreSQL. For a single-node deployment
serving one analyst's matches this is adequate, needs no operational work, and
keeps the whole product runnable with one command. The ORM layer means moving to
PostgreSQL is a connection-string change rather than a rewrite, and the schema
below avoids anything SQLite-specific.

Heavy artefacts are **not** stored in the database. Videos and analytics live on
disk in the existing run directory layout, and rows here hold paths. A Parquet
table of three million game-state rows does not belong in a relational database,
and the run directory is already the reproducibility unit.
"""

from __future__ import annotations

import enum
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from visionpitch.common.logging import get_logger

log = get_logger("api.store")


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    ANALYSING = "analysing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _uuid() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> datetime:
    return datetime.now(UTC)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    jobs: Mapped[list[Job]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "n_jobs": len(self.jobs),
            "jobs": [job.to_summary() for job in
                     sorted(self.jobs, key=lambda j: j.created_at, reverse=True)],
        }


class Job(Base):
    """One analysis of one uploaded video."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), default="")
    mode: Mapped[str] = mapped_column(String(32), default="balanced")
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.PENDING, nullable=False
    )

    video_path: Mapped[str] = mapped_column(Text, default="")
    video_filename: Mapped[str] = mapped_column(String(300), default="")
    video_bytes: Mapped[int] = mapped_column(Integer, default=0)

    #: run directory produced by the vision pipeline; the analytics live under it
    run_dir: Mapped[str] = mapped_column(Text, default="")

    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str] = mapped_column(String(64), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    #: data-quality header copied from the analytics summary, so the job list can
    #: warn about a poor run without opening the artefacts
    quality: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped[Project] = relationship(back_populates="jobs")

    @property
    def analytics_dir(self) -> Path | None:
        if not self.run_dir:
            return None
        path = Path(self.run_dir) / "analytics"
        return path if path.exists() else None

    def to_summary(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "mode": self.mode,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "stage": self.stage,
            "video_filename": self.video_filename,
            "created_at": self.created_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "has_analytics": self.analytics_dir is not None,
            "error": self.error,
        }

    def to_dict(self) -> dict:
        return {
            **self.to_summary(),
            "run_dir": self.run_dir,
            "video_bytes": self.video_bytes,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "quality": self.quality or {},
        }


class Store:
    """Thin session factory plus the queries the API needs."""

    def __init__(self, database_url: str, echo: bool = False) -> None:
        # check_same_thread is required because the background worker touches the
        # same SQLite file from another thread.
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, echo=echo, connect_args=connect_args)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        log.info("store ready at %s", database_url)

    def session(self):
        return self.session_factory()

    # -- projects ------------------------------------------------------------- #

    def create_project(self, name: str, description: str = "") -> Project:
        with self.session() as session:
            project = Project(name=name, description=description)
            session.add(project)
            session.commit()
            session.refresh(project)
            _ = project.jobs
            return project

    def list_projects(self) -> list[Project]:
        with self.session() as session:
            projects = list(session.scalars(select(Project).order_by(Project.created_at.desc())))
            for project in projects:
                _ = project.jobs
            return projects

    def get_project(self, project_id: str) -> Project | None:
        with self.session() as session:
            project = session.get(Project, project_id)
            if project is not None:
                _ = project.jobs
            return project

    def delete_project(self, project_id: str) -> bool:
        """Remove a project, its jobs, and every artefact they produced.

        Deletes files as well as rows: leaving multi-gigabyte run directories
        behind after the user asked for deletion would be a privacy problem, not
        just a housekeeping one.
        """
        import shutil

        with self.session() as session:
            project = session.get(Project, project_id)
            if project is None:
                return False
            for job in project.jobs:
                for path in (job.run_dir, job.video_path):
                    if not path:
                        continue
                    target = Path(path)
                    try:
                        if target.is_dir():
                            shutil.rmtree(target, ignore_errors=True)
                        elif target.exists():
                            target.unlink()
                    except OSError as exc:
                        log.warning("could not delete %s: %s", target, exc)
            session.delete(project)
            session.commit()
            return True

    # -- jobs ------------------------------------------------------------------ #

    def create_job(self, project_id: str, **kwargs) -> Job:
        with self.session() as session:
            job = Job(project_id=project_id, **kwargs)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def get_job(self, job_id: str) -> Job | None:
        with self.session() as session:
            return session.get(Job, job_id)

    def list_jobs(self, project_id: str | None = None) -> list[Job]:
        with self.session() as session:
            query = select(Job).order_by(Job.created_at.desc())
            if project_id:
                query = query.where(Job.project_id == project_id)
            return list(session.scalars(query))

    def update_job(self, job_id: str, **fields) -> Job | None:
        with self.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                return None
            for key, value in fields.items():
                setattr(job, key, value)
            session.commit()
            session.refresh(job)
            return job

    def delete_job(self, job_id: str) -> bool:
        import shutil

        with self.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                return False
            for path in (job.run_dir, job.video_path):
                if not path:
                    continue
                target = Path(path)
                try:
                    if target.is_dir():
                        shutil.rmtree(target, ignore_errors=True)
                    elif target.exists():
                        target.unlink()
                except OSError as exc:
                    log.warning("could not delete %s: %s", target, exc)
            session.delete(job)
            session.commit()
            return True


def load_analytics_json(job: Job, name: str) -> dict | list | None:
    """Read one analytics artefact for a job, or ``None`` if absent."""
    directory = job.analytics_dir
    if directory is None:
        return None
    path = directory / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
