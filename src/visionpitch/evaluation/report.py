"""End-to-end evaluation of a completed run.

Reads a run directory's stored tables -- never the video -- and measures them
against ground-truth annotations. Because it works from the Parquet output, a
run can be evaluated long after it finished, and re-evaluated against improved
annotations without reprocessing anything.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger
from visionpitch.common.types import (
    BBox,
    CalibrationResult,
    Detection,
    ObjectClass,
    Role,
    SegmentKind,
    TeamId,
    Track,
    TrackObservation,
)
from visionpitch.evaluation.calibration import evaluate_calibration
from visionpitch.evaluation.detection import evaluate_detection
from visionpitch.evaluation.ground_truth import load_ground_truth
from visionpitch.evaluation.tracking import evaluate_tracking
from visionpitch.pitch.geometry import PitchConfiguration
from visionpitch.storage.tables import read_table

log = get_logger("evaluation.report")


def _load_detections(path: Path) -> dict[int, list[Detection]]:
    data = read_table(path).to_pydict()
    out: dict[int, list[Detection]] = {}
    for i in range(len(data["frame_idx"])):
        frame_idx = int(data["frame_idx"][i])
        out.setdefault(frame_idx, []).append(
            Detection(
                frame_idx=frame_idx,
                object_class=ObjectClass(data["object_class"][i]),
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
    return out


def _load_tracks(path: Path) -> dict[int, Track]:
    """Rebuild per-frame tracks from the game-state table."""
    data = read_table(path).to_pydict()
    tracks: dict[int, Track] = {}

    for i in range(len(data["frame_idx"])):
        track_id = data["track_id"][i]
        if track_id is None:
            continue  # the ball has no track id
        track_id = int(track_id)
        object_class = ObjectClass(data["object_class"][i])

        track = tracks.get(track_id)
        if track is None:
            track = Track(
                track_id=track_id,
                object_class=object_class,
                team_id=TeamId(data["team_id"][i]),
                team_confidence=float(data["team_confidence"][i]),
                role=Role(data["role"][i]),
            )
            tracks[track_id] = track

        track.observations.append(
            TrackObservation(
                frame_idx=int(data["frame_idx"][i]),
                timestamp_s=float(data["timestamp_s"][i]),
                bbox=BBox(
                    float(data["bbox_x1"][i]),
                    float(data["bbox_y1"][i]),
                    float(data["bbox_x2"][i]),
                    float(data["bbox_y2"][i]),
                ),
                det_confidence=float(data["detection_confidence"][i]),
                track_confidence=float(data["tracking_confidence"][i]),
                interpolated=bool(data["interpolated"][i]),
            )
        )

    for track in tracks.values():
        track.observations.sort(key=lambda o: o.frame_idx)
    return tracks


def _load_calibration(path: Path) -> dict[int, CalibrationResult]:
    from visionpitch.storage.tables import homography_from_row

    data = read_table(path).to_pydict()
    out: dict[int, CalibrationResult] = {}
    for i in range(len(data["frame_idx"])):
        H = homography_from_row(data["homography"][i])
        frame_idx = int(data["frame_idx"][i])
        out[frame_idx] = CalibrationResult(
            frame_idx=frame_idx,
            homography=H,
            confidence=float(data["confidence"][i]),
            reprojection_error_m=float(data["reprojection_error_m"][i]),
            n_keypoints=int(data["n_keypoints"][i]),
            n_inliers=int(data["n_inliers"][i]),
            smoothed=bool(data["smoothed"][i]),
            segment_kind=SegmentKind(data["segment_kind"][i]),
        )
    return out


def evaluate_run(
    run_dir: str | Path,
    annotations: str | Path | None,
    output: str | Path | None = None,
) -> dict:
    """Measure a run and write an evaluation report.

    ``annotations`` may be ``None``. Two classes of metric exist and the report
    keeps them strictly apart:

    **Reference-free** -- calibration coverage, temporal stability, ball
    coverage, track fragmentation, team cluster separation, throughput. These
    need no ground truth and are fully rigorous. They cannot tell you the system
    is *right*, but they reliably tell you when it is wrong.

    **Ground-truthed** -- detection precision/recall/mAP, HOTA/IDF1/MOTA, and
    pitch-position error. These require annotation and are simply absent, and
    labelled as absent, when none is supplied. They are never estimated,
    substituted or inferred from the system's own output, because a system
    scored against itself always scores perfectly.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in {run_dir}; is this a run directory?")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = Config.model_validate(
        __import__("yaml").safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    )
    pitch = PitchConfiguration(length=config.pitch.length_m, width=config.pitch.width_m)

    gt = load_ground_truth(annotations) if annotations is not None else None
    video = manifest.get("video", {})
    run_video_id = video.get("video_id")
    if gt is not None and gt.video_id and run_video_id and gt.video_id != run_video_id:
        log.warning(
            "annotation video_id %r does not match the run's %r; evaluating anyway, "
            "but confirm these are the same clip",
            gt.video_id,
            video["video_id"],
        )

    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "mode": config.mode.value,
        "config_fingerprint": manifest.get("config_fingerprint"),
        "video": video,
        "ground_truth": gt.summary() if gt is not None else None,
        "models": manifest.get("models", {}),
    }

    detections_path = run_dir / "detections.parquet"
    game_state_path = run_dir / "game_state.parquet"

    # -- reference-free measurements ----------------------------------------- #
    report["reference_free"] = _reference_free(manifest, game_state_path)

    # -- detection ----------------------------------------------------------- #
    if gt is None:
        report["detection"] = {
            "status": "not measured",
            "reason": "no ground-truth annotation supplied",
            "how_to_measure": (
                "python scripts/annotate.py boxes <video> --frames <list>, then "
                "visionpitch evaluate <run_dir> --annotations <json>"
            ),
        }
    elif detections_path.exists():
        detections = _load_detections(detections_path)
        report["detection"] = evaluate_detection(
            gt,
            detections,
            iou_thresholds=config.evaluation.map_iou_thresholds,
            small_object_area_px=config.evaluation.small_object_area_px,
        )
    else:
        report["detection"] = {"skipped": "detections.parquet not found"}

    # -- tracking ------------------------------------------------------------ #
    if gt is None:
        report["tracking"] = {
            "status": "not measured",
            "reason": "HOTA, IDF1 and MOTA all require annotated identities",
        }
    elif game_state_path.exists():
        tracks = _load_tracks(game_state_path)
        report["tracking"] = evaluate_tracking(
            gt,
            tracks,
            iou_threshold=config.evaluation.tracking_iou_threshold,
            hota_alphas=config.evaluation.hota_alphas,
        )
    else:
        report["tracking"] = {"skipped": "game_state.parquet not found"}

    # -- calibration ----------------------------------------------------------- #
    # Runs with or without ground truth: without it, coverage and stability are
    # still measured, only the pitch-position error is missing.
    calibration_path = run_dir / "calibration.parquet"
    if calibration_path.exists():
        calibration = _load_calibration(calibration_path)
        frame_indices = sorted(calibration)
        report["calibration"] = evaluate_calibration(
            calibration,
            frame_indices,
            pitch,
            (int(video.get("width", 1920)), int(video.get("height", 1080))),
            ground_truth=gt,
            min_confidence=config.calibration.min_confidence,
        )
    else:
        report["calibration"] = {"skipped": "calibration.parquet not found"}

    report["data_quality"] = manifest.get("data_quality", {})
    report["warnings"] = manifest.get("warnings", [])
    report["interpretation"] = _interpret(report)

    output = Path(output) if output else run_dir / "evaluation" / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote evaluation report -> %s", output)
    return report


def _reference_free(manifest: dict, game_state_path: Path) -> dict:
    """Metrics that need no annotation and are therefore always available.

    These cannot prove the system is accurate. What they can do -- and what
    makes them worth reporting on every run -- is prove it is *not*: a
    calibration that jitters by 30 m between frames, or a ball seen in 5% of
    frames, is broken regardless of what any ground truth would say.
    """
    stages = manifest.get("stages", {})
    calibration = stages.get("calibration", {})
    ball = stages.get("ball_tracking", {})
    tracking = stages.get("tracking", {})
    teams = stages.get("team_classification", {})
    game_state = stages.get("game_state", {})
    timings = stages.get("timings_s", {})

    out = {
        "calibration": {
            "valid_frame_percentage": round(100 * (calibration.get("valid_ratio") or 0), 2),
            "confident_frame_percentage": round(
                100 * (calibration.get("confident_ratio") or 0), 2
            ),
            "self_reported_reprojection_error_m": calibration.get(
                "mean_reprojection_error_m"
            ),
            "temporal_stability_m": calibration.get("temporal_stability"),
            "rejections": calibration.get("rejections"),
        },
        "ball": {
            "observed_percentage": round(100 * (ball.get("observed_ratio") or 0), 2),
            "visible_percentage": round(100 * (ball.get("visible_ratio") or 0), 2),
            "interpolated_frames": ball.get("interpolated"),
            "unknown_frames": ball.get("unknown"),
        },
        "tracks": {
            "raw": tracking.get("tracks_in"),
            "after_cleaning": tracking.get("tracks_out"),
            "stitched": tracking.get("stitched"),
            "dropped_short": tracking.get("dropped_short"),
            "mean_length_frames": tracking.get("mean_track_length"),
            "median_length_frames": tracking.get("median_track_length"),
        },
        "teams": {
            "cluster_separation_silhouette": teams.get("cluster_separation"),
            "counts": teams.get("team_counts"),
            "discovered_colours_bgr": teams.get("team_colours"),
        },
        "game_state": {
            "rows": game_state.get("rows"),
            "pitch_coordinate_percentage": round(
                100 * (game_state.get("pitch_coordinate_ratio") or 0), 2
            ),
            "interpolated_percentage": round(
                100 * (game_state.get("interpolated_ratio") or 0), 2
            ),
            "mean_people_per_frame": game_state.get("mean_people_per_frame"),
        },
        "throughput_s": timings,
    }

    frames = manifest.get("data_quality", {}).get("frames_processed")
    total = timings.get("total")
    if frames and total:
        out["throughput_fps"] = round(frames / total, 2)
        out["realtime_factor"] = round(
            frames / total / max(1e-9, manifest.get("video", {}).get("fps", 25.0)), 3
        )
    _ = game_state_path
    return out


def _interpret(report: dict) -> list[str]:
    """Plain-language notes so a reader cannot mistake a weak result for a good one."""
    notes: list[str] = []

    detection = report.get("detection", {}).get("overall", {})
    ball_recall = detection.get("ball_recall")
    if ball_recall is not None and ball_recall < 0.6:
        notes.append(
            f"ball recall is {ball_recall:.2f}. Ball-dependent analytics in later "
            f"phases (possession, passes) will inherit this as an upper bound."
        )

    tracking = report.get("tracking", {})
    hota = tracking.get("HOTA")
    if hota is not None:
        if hota < 0.4:
            notes.append(f"HOTA {hota:.3f} is weak; identities are not reliable enough "
                         f"for per-player statistics.")
        elif hota < 0.6:
            notes.append(f"HOTA {hota:.3f} is moderate; aggregate team statistics are "
                         f"usable but individual player totals will contain errors.")

    calibration = report.get("calibration", {})
    if calibration.get("ground_truth_available"):
        error = (calibration.get("pitch_position_error_m") or {}).get("median")
        if error is not None and error > 2.0:
            notes.append(
                f"median pitch-position error is {error:.2f}m. Distance and speed "
                f"metrics derived from these coordinates carry at least this error."
            )
    else:
        notes.append(
            "calibration was not measured against manually marked landmarks, so a "
            "systematic homography bias would not have been detected."
        )

    valid = calibration.get("valid_frame_percentage")
    if valid is not None and valid < 80:
        notes.append(
            f"only {valid:.0f}% of frames were calibrated; the rest have no pitch "
            f"coordinates at all."
        )

    gt_summary = report.get("ground_truth") or {}
    if not gt_summary:
        notes.append(
            "No ground-truth annotation was supplied, so detection precision/recall "
            "and the tracking identity metrics (HOTA, IDF1, MOTA) are NOT MEASURED "
            "for this clip. Only reference-free diagnostics are reported. Do not "
            "quote an accuracy figure for this run."
        )
    elif gt_summary.get("annotated_frames", 0) < 30:
        notes.append(
            f"only {gt_summary.get('annotated_frames')} frames are annotated. These "
            f"numbers indicate direction, not a reliable performance estimate."
        )

    reference_free = report.get("reference_free", {})
    stability = (reference_free.get("calibration") or {}).get("temporal_stability_m") or {}
    median = stability.get("median_delta_m")
    mean = stability.get("mean_delta_m")
    if median is not None and mean is not None and mean > 3 * max(median, 0.01):
        notes.append(
            f"calibration stability is heavy-tailed: median {median:.2f} m but mean "
            f"{mean:.2f} m. Most frames are stable; a minority fit badly. Quote the "
            f"median, and treat low-confidence frames as unusable."
        )

    return notes
