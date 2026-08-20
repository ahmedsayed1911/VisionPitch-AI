"""Benchmarking the vision stages against annotated corpora.

Separate from ``report.py``, which evaluates a *completed pipeline run* on a
video. This module runs a stage directly against a labelled dataset, which is
what makes A/B comparison of configurations practical: swap a setting, re-run,
compare, with no video decoding in the loop.

Every result records whether the corpus is in-distribution for the shipped
checkpoints. Mixing the two would let a strong in-domain score hide weak
generalisation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger, progress_bar
from visionpitch.common.types import Detection
from visionpitch.evaluation.datasets import (
    YoloDetectionDataset,
    bootstrap_interval,
    validate_ground_truth,
)
from visionpitch.evaluation.detection import evaluate_detection
from visionpitch.evaluation.ground_truth import GroundTruth

log = get_logger("evaluation.benchmark")


@dataclass
class BenchmarkResult:
    label: str
    dataset: dict
    config_summary: dict
    metrics: dict
    runtime_s: float
    generated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def run_detection_benchmark(
    config: Config,
    dataset: YoloDetectionDataset,
    label: str = "baseline",
    use_ball_detector: bool | None = None,
) -> BenchmarkResult:
    """Score the configured detector(s) on a labelled still-image corpus.

    When the corpus contains ball annotations and the specialist ball detector
    is enabled, both detectors run and their outputs are fused exactly as the
    pipeline fuses them — otherwise the benchmark would measure a configuration
    that never actually runs.
    """
    import cv2

    from visionpitch.detection.ball import BallDetector
    from visionpitch.detection.fusion import fuse_detections
    from visionpitch.detection.yolo import build_detector

    gt = dataset.to_ground_truth()
    info = dataset.info()

    detector = build_detector(config)
    want_ball = (
        config.ball_detection.enabled if use_ball_detector is None else use_ball_detector
    )
    ball_detector = BallDetector(config) if want_ball else None

    predictions: dict[int, list[Detection]] = {}
    start = time.perf_counter()

    batch_size = max(1, config.runtime.batch_size)
    frames = dataset.frames
    with progress_bar() as progress:
        task = progress.add_task(f"detection benchmark [{label}]", total=len(frames))
        for offset in range(0, len(frames), batch_size):
            chunk = frames[offset : offset + batch_size]
            images = [cv2.imread(str(f.image_path)) for f in chunk]
            indices = [f.frame_idx for f in chunk]

            multiclass = detector.detect_batch(images, indices)
            for frame, image, mc in zip(chunk, images, multiclass, strict=True):
                specialist: list[Detection] = []
                if ball_detector is not None:
                    # No motion history on stills, so the specialist must sweep.
                    specialist = ball_detector.detect_tiled(image, frame.frame_idx)
                predictions[frame.frame_idx] = fuse_detections(mc, specialist)
            progress.update(task, advance=len(chunk))

    runtime = time.perf_counter() - start

    metrics = evaluate_detection(
        gt,
        predictions,
        iou_thresholds=config.evaluation.map_iou_thresholds,
        small_object_area_px=config.evaluation.small_object_area_px,
    )
    metrics["confidence_intervals"] = _detection_intervals(gt, predictions)
    metrics["validation"] = validate_ground_truth(gt, require_identity=False)

    return BenchmarkResult(
        label=label,
        dataset=info.to_dict(),
        config_summary=_config_summary(config, ball_detector is not None),
        metrics=metrics,
        runtime_s=round(runtime, 2),
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _detection_intervals(
    gt: GroundTruth, predictions: dict[int, list[Detection]], iou: float = 0.5
) -> dict:
    """Bootstrap intervals for per-class recall and precision, resampled by frame."""
    from visionpitch.evaluation.detection import _match_frame

    per_class_recall: dict[str, list[float]] = {}
    per_class_precision: dict[str, list[float]] = {}

    for frame_idx in gt.annotated_frames:
        objects = gt.frames[frame_idx]
        preds = predictions.get(frame_idx, [])
        classes = {o.object_class for o in objects} | {p.object_class for p in preds}
        for object_class in classes:
            gt_boxes = np.array(
                [o.bbox.to_array() for o in objects if o.object_class is object_class]
            ) if any(o.object_class is object_class for o in objects) else np.zeros((0, 4))
            frame_preds = [p for p in preds if p.object_class is object_class]
            pred_boxes = (
                np.array([p.bbox.to_array() for p in frame_preds])
                if frame_preds
                else np.zeros((0, 4))
            )
            scores = (
                np.array([p.confidence for p in frame_preds])
                if frame_preds
                else np.zeros(0)
            )
            tp, matched = _match_frame(gt_boxes, pred_boxes, scores, iou)

            if gt_boxes.shape[0]:
                per_class_recall.setdefault(object_class.value, []).append(
                    float(matched.sum() / gt_boxes.shape[0])
                )
            if pred_boxes.shape[0]:
                per_class_precision.setdefault(object_class.value, []).append(
                    float(tp.sum() / pred_boxes.shape[0])
                )

    out: dict[str, dict] = {}
    for name, values in per_class_recall.items():
        entry = out.setdefault(name, {})
        entry["recall_ci95"] = bootstrap_interval(values)
        entry["n_frames_with_gt"] = len(values)
    for name, values in per_class_precision.items():
        entry = out.setdefault(name, {})
        entry["precision_ci95"] = bootstrap_interval(values)
    return out


def _config_summary(config: Config, ball_enabled: bool) -> dict:
    """The settings that actually move detection numbers."""
    return {
        "mode": config.mode.value,
        "detection.imgsz": config.detection.imgsz,
        "detection.augment": config.detection.augment,
        "detection.conf_threshold": config.detection.conf_threshold,
        "detection.class_conf_overrides": dict(config.detection.class_conf_overrides),
        "ball_detection.enabled": ball_enabled,
        "ball_detection.imgsz": config.ball_detection.imgsz,
        "ball_detection.conf_threshold": config.ball_detection.conf_threshold,
        "ball_detection.tiles": (
            f"{config.ball_detection.tile_rows}x{config.ball_detection.tile_cols}"
        ),
        "runtime.half_precision": config.runtime.half_precision,
        "config_fingerprint": config.fingerprint(),
    }


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #


def run_tracking_benchmark(
    config: Config,
    sequences: list,
    label: str = "baseline",
    max_frames_per_sequence: int | None = None,
) -> BenchmarkResult:
    """Score detection + tracking on identity-annotated image sequences.

    Runs the same detector, tracker and offline association the video pipeline
    runs, but reading frames from a directory. Calibration is *not* run: these
    sequences are scored in image space, so a calibration failure cannot
    contaminate the tracking numbers.

    Metrics are computed per sequence and reported both individually and pooled,
    because a single pooled figure hides the fact that most tracking failures
    concentrate in a few hard clips.
    """
    import cv2

    from visionpitch.common.types import ObjectClass
    from visionpitch.detection.yolo import build_detector
    from visionpitch.evaluation.tracking import evaluate_tracking
    from visionpitch.tracking.postprocess import clean_tracks
    from visionpitch.tracking.tracker import MultiObjectTracker

    detector = build_detector(config)
    per_sequence: list[dict] = []
    start = time.perf_counter()

    with progress_bar() as progress:
        task = progress.add_task(f"tracking benchmark [{label}]", total=len(sequences))
        for sequence in sequences:
            tracker = MultiObjectTracker(config)
            frames = sorted(sequence.image_paths)
            if max_frames_per_sequence:
                frames = frames[:max_frames_per_sequence]

            batch_size = max(1, config.runtime.batch_size)
            for offset in range(0, len(frames), batch_size):
                chunk = frames[offset : offset + batch_size]
                images = [cv2.imread(str(sequence.image_paths[f])) for f in chunk]
                usable = [(f, im) for f, im in zip(chunk, images, strict=True) if im is not None]
                if not usable:
                    continue
                indices = [f for f, _ in usable]
                batch_images = [im for _, im in usable]
                detections = detector.detect_batch(batch_images, indices)
                for frame_idx, image, dets in zip(
                    indices, batch_images, detections, strict=True
                ):
                    tracker.update(dets, frame_idx, frame_idx / 25.0, image)

            raw = tracker.finalise()
            cleaned, _, clean_report = clean_tracks(raw, config.tracking)

            gt = sequence.ground_truth
            # Restrict ground truth to the frames actually processed, otherwise
            # unprocessed frames count as total detection failure.
            processed = set(frames)
            gt.frames = {f: objs for f, objs in gt.frames.items() if f in processed}
            if sequence.ignored_frames:
                gt.frames = {
                    f: objs for f, objs in gt.frames.items()
                    if f not in sequence.ignored_frames
                }
            if not gt.frames:
                progress.update(task, advance=1)
                continue

            metrics = evaluate_tracking(
                gt,
                cleaned,
                iou_threshold=config.evaluation.tracking_iou_threshold,
                hota_alphas=config.evaluation.hota_alphas,
                classes=(
                    ObjectClass.PLAYER,
                    ObjectClass.GOALKEEPER,
                    ObjectClass.REFEREE,
                ),
            )
            metrics["sequence"] = sequence.name
            metrics["n_frames_processed"] = len(frames)
            metrics["median_track_length"] = clean_report.get("median_track_length")
            per_sequence.append(metrics)
            progress.update(task, advance=1)

    runtime = time.perf_counter() - start
    pooled = _pool_tracking(per_sequence)

    return BenchmarkResult(
        label=label,
        dataset={
            "name": "SoccerNet SN-GSR-2025",
            "kind": "out_of_distribution",
            "task": "tracking",
            "n_frames": sum(m["n_frames_processed"] for m in per_sequence),
            "n_objects": sum(m.get("ground_truth_detections", 0) for m in per_sequence),
            "n_sequences": len(per_sequence),
        },
        config_summary={
            "mode": config.mode.value,
            "tracker": config.tracking.tracker,
            "association": config.tracking.association,
            "gmc_enabled": config.tracking.gmc_enabled,
            "reid_enabled": config.tracking.reid_enabled,
            "detection.imgsz": config.detection.imgsz,
            "config_fingerprint": config.fingerprint(),
        },
        metrics={"pooled": pooled, "per_sequence": per_sequence},
        runtime_s=round(runtime, 2),
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _pool_tracking(per_sequence: list[dict]) -> dict:
    """Pool per-sequence tracking metrics.

    Identity metrics are averaged across sequences rather than over pooled
    detections: track ids are only meaningful within a sequence, so pooling the
    raw counts across clips would compare identities that were never comparable.
    Count metrics are summed, since those are genuinely additive.
    """
    if not per_sequence:
        return {}

    def mean(key: str) -> float | None:
        values = [m[key] for m in per_sequence if isinstance(m.get(key), (int, float))]
        return round(float(np.mean(values)), 4) if values else None

    def total(key: str) -> int:
        return int(sum(m.get(key, 0) or 0 for m in per_sequence))

    pooled = {
        "n_sequences": len(per_sequence),
        "HOTA": mean("HOTA"),
        "DetA": mean("DetA"),
        "AssA": mean("AssA"),
        "IDF1": mean("IDF1"),
        "MOTA": mean("MOTA"),
        "MOTP_iou": mean("MOTP_iou"),
        "id_switches": total("id_switches"),
        "fragmentations": total("fragmentations"),
        "mostly_tracked": total("mostly_tracked"),
        "partially_tracked": total("partially_tracked"),
        "mostly_lost": total("mostly_lost"),
        "false_positives": total("false_positives"),
        "false_negatives": total("false_negatives"),
        "ground_truth_detections": total("ground_truth_detections"),
        "median_track_length": mean("median_track_length"),
    }
    for key in ("HOTA", "IDF1", "MOTA"):
        values = [m[key] for m in per_sequence if isinstance(m.get(key), (int, float))]
        pooled[f"{key}_ci95"] = bootstrap_interval(values)
    return pooled


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def write_benchmark(result: BenchmarkResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    log.info("wrote benchmark -> %s", path)
    return path


def compare_benchmarks(results: list[BenchmarkResult]) -> dict:
    """Side-by-side comparison of runs over the same corpus.

    Refuses to compare across corpora: a number from an in-distribution split
    and one from an out-of-distribution split are not comparable, and lining
    them up in a table is how that mistake gets made.
    """
    if not results:
        return {}
    keys = {r.dataset["key"] for r in results}
    if len(keys) > 1:
        raise ValueError(
            f"cannot compare across different corpora: {sorted(keys)}. "
            f"Run one comparison per dataset."
        )

    rows = []
    for r in results:
        overall = r.metrics.get("overall", {})
        per_class = r.metrics.get("per_class", {})
        rows.append(
            {
                "label": r.label,
                "mAP50_all": overall.get("mAP50_all_classes"),
                "mAP50_95_all": overall.get("mAP50_95_all_classes"),
                "ball_recall": overall.get("ball_recall"),
                "ball_precision": per_class.get("ball", {}).get("precision"),
                "ball_mAP50": overall.get("ball_mAP50"),
                "player_recall": per_class.get("player", {}).get("recall"),
                "referee_recall": per_class.get("referee", {}).get("recall"),
                "goalkeeper_recall": per_class.get("goalkeeper", {}).get("recall"),
                "runtime_s": r.runtime_s,
            }
        )

    baseline = rows[0]
    for row in rows[1:]:
        row["delta_vs_baseline"] = {
            k: (
                round(row[k] - baseline[k], 4)
                if isinstance(row.get(k), (int, float))
                and isinstance(baseline.get(k), (int, float))
                else None
            )
            for k in ("mAP50_all", "ball_recall", "ball_precision", "player_recall")
        }

    return {
        "dataset": results[0].dataset,
        "rows": rows,
        "note": (
            "in_distribution results measure the detector on its own training "
            "domain and are not evidence of generalisation"
        ),
    }


def summarise_for_console(result: BenchmarkResult) -> list[tuple[str, str]]:
    """Rows for the CLI table."""
    overall = result.metrics.get("overall", {})
    per_class = result.metrics.get("per_class", {})
    intervals = result.metrics.get("confidence_intervals", {})

    rows: list[tuple[str, str]] = [
        ("dataset", f"{result.dataset['name']} ({result.dataset['kind']})"),
        ("frames / objects", f"{result.dataset['n_frames']} / {result.dataset['n_objects']}"),
    ]
    for name in ("player", "goalkeeper", "referee", "ball"):
        stats = per_class.get(name)
        if not stats:
            continue
        ci = (intervals.get(name) or {}).get("recall_ci95")
        ci_text = f"  [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        rows.append(
            (
                f"{name} P / R",
                f"{stats['precision']:.3f} / {stats['recall']:.3f}{ci_text}",
            )
        )
        rows.append((f"{name} mAP50 / 50-95", f"{stats['mAP50']:.3f} / {stats['mAP50_95']}"))
    rows.append(("mAP50 (all classes)", str(overall.get("mAP50_all_classes"))))
    rows.append(("runtime", f"{result.runtime_s:.1f}s"))
    return rows
