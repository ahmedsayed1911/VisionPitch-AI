"""Replay the ball trajectory search over stored detections.

The trajectory estimator consumes detections, not pixels, so it can be re-run
against ``detections.parquet`` in milliseconds. That turns what would be a
90-second experiment per configuration into an instant one, which is the
difference between tuning this stage empirically and guessing at it.

Crucially it also separates the two failure sources that the headline "ball
observed" percentage conflates:

* the detector never produced a candidate  -> a detection problem
* a candidate existed and was not accepted -> a trajectory-search problem
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionpitch.ball_tracking.trajectory import BallTrajectoryEstimator
from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger
from visionpitch.common.types import BBox, Detection, ObjectClass
from visionpitch.storage.tables import read_table

log = get_logger("evaluation.ball_replay")


@dataclass
class ReplayResult:
    label: str
    settings: dict[str, Any]
    quality: dict[str, Any]
    error_analysis: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "settings": self.settings,
            "quality": self.quality,
            "error_analysis": self.error_analysis,
        }


def load_ball_candidates(
    run_dir: str | Path,
) -> tuple[dict[int, list[Detection]], list[int], dict[int, float], int]:
    """Read ball candidates and the frame lattice out of a completed run."""
    run_dir = Path(run_dir)
    detections_path = run_dir / "detections.parquet"
    if not detections_path.exists():
        raise FileNotFoundError(f"no detections.parquet in {run_dir}")

    data = read_table(detections_path).to_pydict()
    by_frame: dict[int, list[Detection]] = {}
    timestamps: dict[int, float] = {}

    for i in range(len(data["frame_idx"])):
        frame_idx = int(data["frame_idx"][i])
        timestamps[frame_idx] = float(data["timestamp_s"][i])
        if data["object_class"][i] != ObjectClass.BALL.value:
            continue
        by_frame.setdefault(frame_idx, []).append(
            Detection(
                frame_idx=frame_idx,
                object_class=ObjectClass.BALL,
                bbox=BBox(
                    float(data["bbox_x1"][i]),
                    float(data["bbox_y1"][i]),
                    float(data["bbox_x2"][i]),
                    float(data["bbox_y2"][i]),
                ),
                confidence=float(data["confidence"][i]),
                source=str(data["source"][i]),
            )
        )

    frame_indices = sorted(timestamps)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    width = int(manifest.get("video", {}).get("width", 1920))
    return by_frame, frame_indices, timestamps, width


def replay(
    run_dir: str | Path,
    config: Config,
    label: str = "baseline",
) -> ReplayResult:
    """Run the trajectory search over a run's stored candidates."""
    candidates, frame_indices, timestamps, width = load_ball_candidates(run_dir)

    estimator = BallTrajectoryEstimator(config)
    states = estimator.estimate(candidates, frame_indices, timestamps, width)

    cfg = config.ball_tracking
    return ReplayResult(
        label=label,
        settings={
            "max_interpolation_gap_frames": cfg.max_interpolation_gap_frames,
            "min_segment_frames": cfg.min_segment_frames,
            "max_speed_px_per_frame": cfg.max_speed_px_per_frame,
            "smoothing_window": cfg.smoothing_window,
        },
        quality=BallTrajectoryEstimator.quality_report(states),
        error_analysis=estimator.error_analysis(),
    )


def sweep(
    run_dir: str | Path,
    base_config: Config,
    variants: dict[str, dict[str, Any]],
) -> list[ReplayResult]:
    """Replay several trajectory-search configurations over the same candidates.

    ``variants`` maps a label to ``ball_tracking`` field overrides.
    """
    results = [replay(run_dir, base_config, label="baseline")]
    for label, overrides in variants.items():
        config = base_config.model_copy(deep=True)
        for key, value in overrides.items():
            setattr(config.ball_tracking, key, value)
        results.append(replay(run_dir, config, label=label))
    return results


def format_sweep(results: list[ReplayResult]) -> list[dict[str, Any]]:
    """Flatten a sweep into comparable rows."""
    rows = []
    for r in results:
        rows.append(
            {
                "label": r.label,
                "observed_%": round(100 * (r.quality.get("observed_ratio") or 0), 2),
                "visible_%": round(100 * (r.quality.get("visible_ratio") or 0), 2),
                "acceptance_rate": r.error_analysis.get("acceptance_rate"),
                "lost_frames": r.error_analysis.get("lost_frames"),
                "top_cause": (
                    next(iter(r.error_analysis.get("causes", {}).items()), ("none", 0))[0]
                ),
                **{f"set:{k}": v for k, v in r.settings.items()},
            }
        )
    return rows
