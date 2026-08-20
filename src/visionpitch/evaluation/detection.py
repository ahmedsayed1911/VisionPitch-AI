"""Detection metrics: precision, recall and COCO-style mAP.

Implemented directly rather than delegated to a COCO wrapper, for two reasons:
the annotation format here is deliberately simpler than COCO's, and the ball
needs a size-stratified breakdown that the standard tooling does not give
per-class out of the box.

The ball is reported separately throughout. Averaged with three person classes
that each score above 0.95, ball performance disappears into the mean -- and
ball performance is the binding constraint on every downstream football metric.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from visionpitch.common.geometry import iou_matrix
from visionpitch.common.logging import get_logger
from visionpitch.common.types import Detection, ObjectClass
from visionpitch.evaluation.ground_truth import GroundTruth

log = get_logger("evaluation.detection")


def _match_frame(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy score-ordered matching, the COCO convention.

    Returns ``(tp_flags, matched_gt_mask)``. Greedy-by-score rather than
    Hungarian is used deliberately: it is what mAP is defined against, and it
    rewards a detector for ranking its true positives above its false ones.
    """
    n_pred = pred_boxes.shape[0]
    tp = np.zeros(n_pred, dtype=bool)
    matched_gt = np.zeros(gt_boxes.shape[0], dtype=bool)
    if n_pred == 0 or gt_boxes.shape[0] == 0:
        return tp, matched_gt

    ious = iou_matrix(pred_boxes, gt_boxes)
    for pred_idx in np.argsort(-scores):
        candidates = np.where(~matched_gt, ious[pred_idx], -1.0)
        best = int(np.argmax(candidates))
        if candidates[best] >= iou_threshold:
            tp[pred_idx] = True
            matched_gt[best] = True
    return tp, matched_gt


def _average_precision(
    tp: np.ndarray, scores: np.ndarray, n_gt: int, n_points: int = 101
) -> float:
    """101-point interpolated AP over the score-sorted detections."""
    if n_gt == 0:
        return float("nan")
    if tp.size == 0:
        return 0.0

    order = np.argsort(-scores)
    tp_sorted = tp[order]
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(~tp_sorted)

    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Make precision monotonically decreasing, then sample at fixed recalls.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    thresholds = np.linspace(0, 1, n_points)
    sampled = np.zeros(n_points)
    indices = np.searchsorted(recall, thresholds, side="left")
    valid = indices < precision.size
    sampled[valid] = precision[indices[valid]]
    return float(sampled.mean())


def evaluate_detection(
    ground_truth: GroundTruth,
    predictions: dict[int, list[Detection]],
    iou_thresholds: list[float] | None = None,
    small_object_area_px: float = 1024.0,
    primary_iou: float = 0.5,
) -> dict:
    """Per-class and overall detection metrics on the annotated frames only."""
    iou_thresholds = iou_thresholds or [round(0.5 + 0.05 * i, 2) for i in range(10)]
    annotated = set(ground_truth.annotated_frames)
    if not annotated:
        raise ValueError("ground truth contains no annotated frames")

    classes = sorted({o.object_class for objs in ground_truth.frames.values() for o in objs},
                     key=lambda c: c.value)
    per_class: dict[str, dict] = {}

    for object_class in classes:
        # Accumulate across frames so AP is computed over the whole clip, not
        # averaged per frame (which would over-weight sparse frames).
        all_tp: dict[float, list[np.ndarray]] = defaultdict(list)
        all_scores: list[np.ndarray] = []
        n_gt = 0
        n_gt_small = 0
        small_recall_hits = 0
        tp_primary = fp_primary = fn_primary = 0

        for frame_idx in sorted(annotated):
            gt_objs = [o for o in ground_truth.frames[frame_idx] if o.object_class is object_class]
            preds = [
                d for d in predictions.get(frame_idx, []) if d.object_class is object_class
            ]

            gt_boxes = (
                np.array([o.bbox.to_array() for o in gt_objs])
                if gt_objs
                else np.zeros((0, 4))
            )
            pred_boxes = (
                np.array([d.bbox.to_array() for d in preds]) if preds else np.zeros((0, 4))
            )
            scores = np.array([d.confidence for d in preds]) if preds else np.zeros(0)

            n_gt += len(gt_objs)
            small_flags = np.array(
                [o.bbox.area < small_object_area_px for o in gt_objs], dtype=bool
            )
            n_gt_small += int(small_flags.sum())

            for threshold in iou_thresholds:
                tp, matched = _match_frame(gt_boxes, pred_boxes, scores, threshold)
                all_tp[threshold].append(tp)
                if abs(threshold - primary_iou) < 1e-9:
                    tp_primary += int(tp.sum())
                    fp_primary += int((~tp).sum())
                    fn_primary += int((~matched).sum())
                    if small_flags.size:
                        small_recall_hits += int((matched & small_flags).sum())

            all_scores.append(scores)

        scores_concat = np.concatenate(all_scores) if all_scores else np.zeros(0)
        aps = {}
        for threshold in iou_thresholds:
            tp_concat = (
                np.concatenate(all_tp[threshold]) if all_tp[threshold] else np.zeros(0, bool)
            )
            aps[threshold] = _average_precision(tp_concat, scores_concat, n_gt)

        finite_aps = [v for v in aps.values() if np.isfinite(v)]
        precision = tp_primary / max(1, tp_primary + fp_primary)
        recall = tp_primary / max(1, tp_primary + fn_primary)

        per_class[object_class.value] = {
            "n_ground_truth": n_gt,
            "n_predictions": int(scores_concat.size),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(
                float(2 * precision * recall / max(1e-9, precision + recall)), 4
            ),
            "mAP50": round(float(aps.get(0.5, float("nan"))), 4),
            "mAP50_95": round(float(np.mean(finite_aps)), 4) if finite_aps else None,
            "true_positives": tp_primary,
            "false_positives": fp_primary,
            "false_negatives": fn_primary,
            "n_small_objects": n_gt_small,
            "small_object_recall": (
                round(small_recall_hits / n_gt_small, 4) if n_gt_small else None
            ),
        }

    person_classes = [c.value for c in classes if c.is_person]
    overall = {
        "iou_thresholds": iou_thresholds,
        "primary_iou": primary_iou,
        "annotated_frames": len(annotated),
        "mAP50_all_classes": _safe_mean(
            [per_class[c]["mAP50"] for c in per_class]
        ),
        "mAP50_95_all_classes": _safe_mean(
            [per_class[c]["mAP50_95"] for c in per_class]
        ),
        "mAP50_person_classes": _safe_mean(
            [per_class[c]["mAP50"] for c in person_classes if c in per_class]
        ),
        "ball_recall": per_class.get(ObjectClass.BALL.value, {}).get("recall"),
        "ball_mAP50": per_class.get(ObjectClass.BALL.value, {}).get("mAP50"),
        "ball_mAP50_95": per_class.get(ObjectClass.BALL.value, {}).get("mAP50_95"),
    }

    return {"per_class": per_class, "overall": overall}


def _safe_mean(values: list) -> float | None:
    usable = [v for v in values if v is not None and np.isfinite(v)]
    return round(float(np.mean(usable)), 4) if usable else None
