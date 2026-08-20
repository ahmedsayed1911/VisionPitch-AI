"""Trajectory-shape metrics read back off a completed run.

These are the Part 6 numbers that no existing script produced: how fragmented
the ball trajectory is, how far the estimator bridged, and whether anything was
emitted that no detector ever saw.

Everything here is derived from ``game_state.parquet`` plus the run manifest --
the same artifacts a user gets -- so a metric that cannot be computed from a
stored run is not reported as if it could.

Definitions, because these words are used loosely elsewhere
-----------------------------------------------------------
run
    a maximal set of consecutive processed frames all carrying a ball position,
    of any provenance.
direct observation
    a ball row with ``interpolated == False``. This is what the assembler writes
    when the trajectory search seated a real detection in the frame.
long bridge
    a maximal interpolated run of at least ``LONG_BRIDGE_MIN_RUN_FRAMES``
    frames. This is the mechanism by which emptying frames could manufacture
    trajectory, so it is measured directly rather than inferred.
long-gap hallucination
    a frame carrying a position whose distance, in processed frames, to the
    nearest direct observation exceeds the estimator's own interpolation cap.
    The cap makes this impossible; a non-zero value is a defect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visionpitch.evaluation.fusion_thresholds import LONG_BRIDGE_MIN_RUN_FRAMES

FUSION_METRICS_VERSION = "1.0.0"


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """``(start_index, length)`` for each maximal True run."""
    out: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(flags) - start))
    return out


def trajectory_metrics(
    run_dir: Path,
    max_interpolation_gap_frames: int = 12,
    max_speed_px_per_frame: float = 140.0,
    cut_frames: set[int] | None = None,
) -> dict:
    """Read a completed run and describe the shape of its ball trajectory."""
    import pyarrow.parquet as pq

    table = pq.read_table(run_dir / "game_state.parquet")
    columns = table.to_pydict()

    frames = sorted(set(columns["frame_idx"]))
    index_of = {f: i for i, f in enumerate(frames)}
    n = len(frames)
    if n == 0:
        return {"n_frames": 0, "note": "no rows"}

    position: dict[int, tuple[float, float]] = {}
    interpolated: dict[int, bool] = {}
    confidence: dict[int, float] = {}
    for cls, frame_idx, x, y, interp, conf in zip(
        columns["object_class"], columns["frame_idx"], columns["image_x"],
        columns["image_y"], columns["interpolated"], columns["detection_confidence"],
        strict=True,
    ):
        if cls != "ball":
            continue
        position[frame_idx] = (float(x), float(y))
        interpolated[frame_idx] = bool(interp)
        confidence[frame_idx] = float(conf)

    has_position = [f in position for f in frames]
    is_direct = [f in position and not interpolated[f] for f in frames]
    is_interp = [f in position and interpolated[f] for f in frames]

    n_direct = sum(is_direct)
    n_interp = sum(is_interp)
    n_unknown = n - n_direct - n_interp

    # -- trajectory runs ---------------------------------------------------- #
    runs = _runs(has_position)
    lengths = [length for _, length in runs]
    one_frame = sum(1 for length in lengths if length == 1)

    # -- implausible jumps -------------------------------------------------- #
    jumps = 0
    previous: tuple[int, float, float] | None = None
    for f in frames:
        if f not in position:
            previous = None
            continue
        if previous is not None:
            gap = max(1, f - previous[0])
            step = float(np.hypot(
                position[f][0] - previous[1], position[f][1] - previous[2]
            )) / gap
            if step > max_speed_px_per_frame:
                jumps += 1
        previous = (f, *position[f])

    # -- long bridges -------------------------------------------------------- #
    bridge_runs = [
        (start, length) for start, length in _runs(is_interp)
        if length >= LONG_BRIDGE_MIN_RUN_FRAMES
    ]
    frames_in_long_bridges = sum(length for _, length in bridge_runs)

    # -- long-gap hallucination ---------------------------------------------- #
    direct_indices = [i for i, flag in enumerate(is_direct) if flag]
    hallucinated = 0
    if direct_indices:
        arr = np.array(direct_indices)
        for i, flag in enumerate(has_position):
            if not flag or is_direct[i]:
                continue
            if int(np.abs(arr - i).min()) > max_interpolation_gap_frames:
                hallucinated += 1
    else:
        # No direct observation anywhere: every emitted position is unsupported.
        hallucinated = sum(has_position)

    # -- camera-cut continuity ------------------------------------------------ #
    cuts = cut_frames or set()
    cut_continuity_errors = 0
    for cut in cuts:
        i = index_of.get(cut)
        if i is None or i == 0:
            continue
        before, after = frames[i - 1], frames[i]
        if before in position and after in position:
            step = float(np.hypot(
                position[after][0] - position[before][0],
                position[after][1] - position[before][1],
            ))
            # A position carried across a cut as though it were motion.
            if step <= max_speed_px_per_frame:
                cut_continuity_errors += 1

    observed_conf = [confidence[f] for f in frames if is_direct[index_of[f]]]

    return {
        "metrics_version": FUSION_METRICS_VERSION,
        "n_frames": n,
        "direct_observation_coverage": round(n_direct / n, 4),
        "interpolated_coverage": round(n_interp / n, 4),
        "unknown_coverage": round(n_unknown / n, 4),
        "n_trajectory_runs": len(runs),
        "one_frame_trajectory_rate": round(one_frame / max(1, len(runs)), 4),
        "median_trajectory_frames": float(np.median(lengths)) if lengths else 0.0,
        "mean_trajectory_frames": round(float(np.mean(lengths)), 3) if lengths else 0.0,
        "trajectory_length_percentiles": {
            str(p): float(np.percentile(lengths, p)) for p in (10, 25, 50, 75, 90)
        } if lengths else {},
        # Runs per 1000 frames: more runs over the same coverage is a more
        # broken trajectory, and the raw count is not comparable across clips.
        "fragmentation_per_1000_frames": round(1000 * len(runs) / n, 3),
        "implausible_jumps": jumps,
        "implausible_jump_rate": round(jumps / max(1, sum(has_position)), 5),
        "camera_cut_continuity_errors": cut_continuity_errors,
        "n_camera_cuts": len(cuts),
        "long_bridge_runs": len(bridge_runs),
        "long_bridge_rate": round(frames_in_long_bridges / n, 4),
        "long_gap_hallucination_frames": hallucinated,
        "long_gap_hallucination_rate": round(hallucinated / n, 5),
        "detector_confidence_on_direct": {
            "mean": round(float(np.mean(observed_conf)), 4) if observed_conf else 0.0,
            "median": round(float(np.median(observed_conf)), 4) if observed_conf else 0.0,
            "p10": round(float(np.percentile(observed_conf, 10)), 4) if observed_conf else 0.0,
            "p90": round(float(np.percentile(observed_conf, 90)), 4) if observed_conf else 0.0,
        },
    }


def system_metrics(run_dir: Path) -> dict:
    """Runtime and output size, from the run's own manifest and files."""
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {}
    )
    stages = manifest.get("stages", {}) or {}
    timings = stages.get("timings_s", {}) or {}
    fusion = stages.get("ball_fusion", {}) or {}

    output_bytes = sum(
        p.stat().st_size for p in run_dir.rglob("*") if p.is_file()
    )
    return {
        "total_runtime_s": timings.get("total"),
        "pass1_analysis_s": timings.get("pass1_analysis"),
        "ball_fusion_s": timings.get("ball_fusion", 0.0),
        "ball_trajectory_s": timings.get("ball_trajectory"),
        "fusion_ms_per_frame": fusion.get("fusion_ms_per_frame", 0.0),
        "output_bytes": output_bytes,
        "output_mb": round(output_bytes / 1e6, 2),
        "config_fingerprint": manifest.get("config_fingerprint"),
    }
