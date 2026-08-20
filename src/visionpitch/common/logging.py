"""Logging and progress reporting.

Stages log through :func:`get_logger`. Failures are never swallowed: a stage
that cannot process a frame records it in a :class:`StageCounters` and the run
manifest reports the totals, so a "successful" run with 12% dropped frames is
visible rather than silent.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

_CONSOLE = Console(stderr=True)
_CONFIGURED = False


def configure_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Install the rich console handler, plus an optional plain file handler."""
    global _CONFIGURED
    root = logging.getLogger("visionpitch")
    root.setLevel(level.upper())
    root.handlers.clear()

    root.addHandler(
        RichHandler(
            console=_CONSOLE,
            rich_tracebacks=True,
            show_path=False,
            omit_repeated_times=False,
        )
    )

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
        )
        root.addHandler(handler)

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Logger for a stage. ``name`` is appended to the ``visionpitch`` root."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"visionpitch.{name}")


def progress_bar(description: str = "processing") -> Progress:
    """A consistent progress bar for long stages."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_CONSOLE,
        transient=False,
    )


@dataclass
class StageCounters:
    """Per-stage tally of successes and every distinct kind of failure.

    Rule 18 of the project brief: do not silently ignore failed frames, missing
    detections or calibration errors. This is the mechanism that enforces it.
    """

    stage: str
    processed: int = 0
    failures: Counter = field(default_factory=Counter)
    warnings: Counter = field(default_factory=Counter)

    def ok(self, n: int = 1) -> None:
        self.processed += n

    def fail(self, reason: str, n: int = 1) -> None:
        self.failures[reason] += n

    def warn(self, reason: str, n: int = 1) -> None:
        self.warnings[reason] += n

    @property
    def total_failures(self) -> int:
        return sum(self.failures.values())

    @property
    def failure_ratio(self) -> float:
        denom = self.processed + self.total_failures
        return self.total_failures / denom if denom else 0.0

    def summary(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "processed": self.processed,
            "failures": dict(self.failures),
            "warnings": dict(self.warnings),
            "failure_ratio": round(self.failure_ratio, 6),
        }

    def log(self, logger: logging.Logger | None = None) -> None:
        log = logger or get_logger(self.stage)
        log.info("%s: %d processed, %d failed", self.stage, self.processed, self.total_failures)
        for reason, count in self.failures.most_common():
            log.warning("  %s: %d frames failed (%s)", self.stage, count, reason)
        for reason, count in self.warnings.most_common():
            log.info("  %s: %d warnings (%s)", self.stage, count, reason)
