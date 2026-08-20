"""Background job execution.

Runs the existing Phase 1 pipeline followed by Phase 2 analytics, updating job
progress as it goes. A thread pool rather than Celery or RQ: the work is
GPU-bound and serialised anyway, so a broker would add operational weight
without adding throughput. The queue is explicit and single-worker by default so
two jobs never contend for the GPU.

Failure is recorded on the job rather than raised into the request that started
it, because the request finished long before the work did.
"""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from visionpitch.api.store import JobStatus, Store, _now
from visionpitch.common.config import AnalysisMode, load_config
from visionpitch.common.logging import get_logger

log = get_logger("api.worker")


class JobWorker:
    """Serial background worker for analysis jobs."""

    def __init__(self, store: Store, output_root: Path, max_workers: int = 1) -> None:
        self.store = store
        self.output_root = Path(output_root)
        self.queue: queue.Queue[str] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self.max_workers = max_workers
        #: set by tests to run synchronously instead of spawning threads
        self.inline = False

    # -- lifecycle -------------------------------------------------------------- #

    def start(self) -> None:
        if self._threads:
            return
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._loop, name=f"vp-worker-{i}", daemon=True)
            thread.start()
            self._threads.append(thread)
        log.info("job worker started with %d thread(s)", self.max_workers)

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self.queue.put("")
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()

    def submit(self, job_id: str) -> None:
        if self.inline:
            self.run_job(job_id)
        else:
            self.queue.put(job_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job_id = self.queue.get()
            if not job_id:
                continue
            try:
                self.run_job(job_id)
            except Exception:  # noqa: BLE001 - a worker thread must never die
                log.exception("job %s crashed", job_id)
            finally:
                self.queue.task_done()

    # -- execution -------------------------------------------------------------- #

    def run_job(self, job_id: str) -> None:
        """Run vision then analytics for one job."""
        from visionpitch.analytics.runner import run_analytics
        from visionpitch.pipeline.runner import Phase1Pipeline

        job = self.store.get_job(job_id)
        if job is None:
            log.warning("job %s no longer exists", job_id)
            return

        self.store.update_job(
            job_id, status=JobStatus.RUNNING, started_at=_now(),
            stage="vision", progress=0.02, error="",
        )

        try:
            config = load_config(
                mode=AnalysisMode(job.mode),
                overrides={"storage": {"output_dir": str(self.output_root)}},
            )
            pipeline = Phase1Pipeline(config)
            result = pipeline.run(job.video_path)

            self.store.update_job(
                job_id, run_dir=str(result.run_dir), stage="analytics",
                progress=0.85, status=JobStatus.ANALYSING,
            )

            analytics = run_analytics(result.run_dir)

            self.store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                stage="done",
                progress=1.0,
                finished_at=_now(),
                quality=analytics.summary.get("data_quality", {}),
            )
            log.info("job %s completed: %d events", job_id, analytics.n_events)

        except Exception as exc:  # noqa: BLE001
            log.exception("job %s failed", job_id)
            self.store.update_job(
                job_id,
                status=JobStatus.FAILED,
                stage="failed",
                finished_at=_now(),
                # The whole traceback, not just the message: a failure a user
                # cannot diagnose is a failure they will report as a bug.
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}",
            )


def make_worker(store: Store, output_root: Path, factory: Callable | None = None) -> JobWorker:
    worker = factory(store, output_root) if factory else JobWorker(store, output_root)
    worker.start()
    return worker
