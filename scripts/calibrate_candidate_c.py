"""Calibrate Candidate C's confidence on validation data only.

Part 3 of precision hardening.

Compares a fixed threshold against Platt scaling, isotonic regression and a
size-aware logistic model, then picks one operating point under a rule declared
before any curve was drawn:

    among operating points keeping validation centre-25 recall at or above 95%
    of the uncalibrated model's, choose the one with the fewest false positives
    per frame.

That encodes the objective exactly: cut false positives, do not spend recall
doing it. The locked tests are not touched here.

Calibrators are fitted with cross-validation *within* validation, so the
reported curves are not the curves the fit optimised. Without that, isotonic
regression in particular would look perfect and generalise badly.

Usage::

    python scripts/calibrate_candidate_c.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.annotation.schema import AnnotationStore, BallVisibility  # noqa: E402
from visionpitch.annotation.splits import LocalSplit  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("hardening.calibrate")

CANDIDATE_C = "models/finetune/bcast_adapt/weights/best.pt"
MULTICORPUS = Path("data/ball_multicorpus")
LOCAL_PACKAGE = Path("data/annotation/package")
LOCAL_SPLIT = Path("data/annotation/local_split.json")
MATCH_PX = 25.0
#: Declared before fitting: recall may fall by at most this fraction.
RECALL_FLOOR_RATIO = 0.95


def validation_frames():
    for path in sorted((MULTICORPUS / "val" / "images").glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        label = path.parent.parent / "labels" / f"{path.stem}.txt"
        balls = []
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    balls.append((cx * w, cy * h, bw * w, bh * h))
        domain = (
            "soccernet_gsr" if path.name.startswith("soccernet_gsr_") else "roboflow"
        )
        yield domain, image, balls

    store = AnnotationStore(LOCAL_PACKAGE)
    samples = store.load_samples()
    annotations = store.load_annotations()
    split = LocalSplit.load(LOCAL_SPLIT)
    for frame_id in sorted(split.frames.get("val", [])):
        annotation = annotations[frame_id]
        image = cv2.imread(samples[frame_id].image_path)
        if image is None:
            continue
        balls = []
        if annotation.visibility is BallVisibility.VISIBLE and annotation.radius_px:
            d = annotation.radius_px * 2
            balls.append((annotation.centre_x, annotation.centre_y, d, d))
        yield "local_broadcast", image, balls


def collect(model, imgsz: int, floor: float):
    """Per-prediction features and TP/FP labels, plus the per-frame truth count."""
    features, labels, domains, frame_ids = [], [], [], []
    n_truth = 0
    n_frames = 0
    for index, (domain, image, balls) in enumerate(validation_frames()):
        n_frames += 1
        n_truth += len(balls)
        result = model.predict(image, imgsz=imgsz, conf=floor, verbose=False)[0]
        if result.boxes is None or not len(result.boxes):
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]

        matched: set[int] = set()
        assignment = [0] * len(boxes)
        for cx, cy, _, _ in balls:
            best, slot = MATCH_PX, None
            for j, box in enumerate(boxes):
                if j in matched:
                    continue
                px, py = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                d = float(np.hypot(px - cx, py - cy))
                if d <= best:
                    best, slot = d, j
            if slot is not None:
                matched.add(slot)
                assignment[slot] = 1

        for j, box in enumerate(boxes):
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            features.append([float(scores[j]), float(np.log1p(area))])
            labels.append(assignment[j])
            domains.append(domain)
            frame_ids.append(index)
    return (
        np.array(features), np.array(labels), domains, np.array(frame_ids),
        n_truth, n_frames,
    )


def curve(probabilities, labels, frame_ids, n_truth, n_frames, grid):
    """Recall / precision / FP-per-frame across an operating-point grid."""
    rows = []
    for point in grid:
        keep = probabilities >= point
        tp = int(labels[keep].sum())
        fp = int(keep.sum() - tp)
        rows.append({
            "operating_point": round(float(point), 4),
            "recall": round(tp / n_truth, 4) if n_truth else 0.0,
            "precision": round(tp / max(1, keep.sum()), 4),
            "fp_per_frame": round(fp / max(1, n_frames), 4),
            "n_kept": int(keep.sum()),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=CANDIDATE_C)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--floor", type=float, default=0.02)
    parser.add_argument("--baseline-threshold", type=float, default=0.12)
    parser.add_argument("--out", type=Path, default=Path("data/eval/hardening"))
    args = parser.parse_args()

    configure_logging("INFO")
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from ultralytics import YOLO

    model = YOLO(args.weights)
    log.info("collecting validation predictions")
    features, labels, domains, frame_ids, n_truth, n_frames = collect(
        model, args.imgsz, args.floor
    )
    log.info(
        "%d prediction(s) over %d frame(s); %d true, %d truth balls",
        len(labels), n_frames, int(labels.sum()), n_truth,
    )

    confidence = features[:, 0]
    grid = np.unique(np.round(np.linspace(0.02, 0.95, 60), 4))

    methods: dict[str, np.ndarray] = {"fixed_threshold": confidence}

    # Platt: logistic on the raw confidence, cross-validated so the reported
    # curve is not the curve the fit saw.
    methods["platt"] = cross_val_predict(
        LogisticRegression(max_iter=1000), confidence.reshape(-1, 1), labels,
        cv=5, method="predict_proba",
    )[:, 1]

    # Isotonic: flexible and prone to memorising; cross-validation matters most
    # here.
    methods["isotonic"] = cross_val_predict(
        CalibratedClassifierCV(
            LogisticRegression(max_iter=1000), method="isotonic", cv=3
        ),
        confidence.reshape(-1, 1), labels, cv=5, method="predict_proba",
    )[:, 1]

    # Size-aware: confidence plus log box area. A small candidate at 0.3 is a
    # different proposition from a large one at 0.3 when the ball is 11 px.
    methods["size_aware"] = cross_val_predict(
        LogisticRegression(max_iter=1000), features, labels,
        cv=5, method="predict_proba",
    )[:, 1]

    baseline_keep = confidence >= args.baseline_threshold
    baseline_tp = int(labels[baseline_keep].sum())
    baseline_recall = baseline_tp / n_truth if n_truth else 0.0
    baseline_fp = (int(baseline_keep.sum()) - baseline_tp) / max(1, n_frames)
    recall_floor = RECALL_FLOOR_RATIO * baseline_recall
    log.info(
        "uncalibrated at %.2f: val recall %.4f, FP/frame %.4f -> recall floor %.4f",
        args.baseline_threshold, baseline_recall, baseline_fp, recall_floor,
    )

    results: dict[str, dict] = {}
    for name, probabilities in methods.items():
        rows = curve(probabilities, labels, frame_ids, n_truth, n_frames, grid)
        eligible = [r for r in rows if r["recall"] >= recall_floor]
        chosen = (
            min(eligible, key=lambda r: r["fp_per_frame"]) if eligible
            else max(rows, key=lambda r: r["recall"])
        )
        results[name] = {
            "curve": rows,
            "selected": chosen,
            "meets_recall_floor": bool(eligible),
            "fp_reduction_vs_uncalibrated": round(
                1 - chosen["fp_per_frame"] / baseline_fp, 4
            ) if baseline_fp else None,
        }
        log.info(
            "%-16s point %.3f  recall %.4f  precision %.4f  FP/frame %.4f",
            name, chosen["operating_point"], chosen["recall"],
            chosen["precision"], chosen["fp_per_frame"],
        )

    # Selection: fewest false positives among methods that hold the recall floor.
    viable = {k: v for k, v in results.items() if v["meets_recall_floor"]}
    winner = min(
        viable or results, key=lambda k: results[k]["selected"]["fp_per_frame"]
    )

    # Refit the winner on all of validation for deployment. The *reported*
    # numbers stay the cross-validated ones.
    fitted = None
    if winner == "platt":
        fitted = LogisticRegression(max_iter=1000).fit(confidence.reshape(-1, 1), labels)
    elif winner == "size_aware":
        fitted = LogisticRegression(max_iter=1000).fit(features, labels)
    elif winner == "isotonic":
        fitted = IsotonicRegression(out_of_bounds="clip").fit(confidence, labels)

    args.out.mkdir(parents=True, exist_ok=True)
    if fitted is not None:
        with (args.out / "calibrator.pkl").open("wb") as handle:
            pickle.dump({"method": winner, "model": fitted}, handle)

    payload = {
        "weights": args.weights,
        "fitted_on": "validation only (public val + local val)",
        "n_predictions": int(len(labels)),
        "n_frames": n_frames,
        "n_truth": n_truth,
        "uncalibrated_baseline": {
            "threshold": args.baseline_threshold,
            "recall": round(baseline_recall, 4),
            "fp_per_frame": round(baseline_fp, 4),
        },
        "selection_rule": (
            f"among operating points with validation centre-25 recall >= "
            f"{RECALL_FLOOR_RATIO} x uncalibrated recall, minimise validation "
            f"false positives per frame; declared before fitting"
        ),
        "recall_floor": round(recall_floor, 4),
        "methods": results,
        "selected_method": winner,
        "selected_operating_point": results[winner]["selected"],
        "cross_validation": "5-fold within validation; reported curves are out-of-fold",
    }
    (args.out / "calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(f"\nuncalibrated @ {args.baseline_threshold}: recall {baseline_recall:.4f}, "
          f"FP/frame {baseline_fp:.4f}")
    print(f"recall floor: {recall_floor:.4f}\n")
    print(f"{'method':<18}{'point':>8}{'recall':>9}{'prec':>9}{'FP/frame':>10}{'FPcut':>8}")
    for name, block in results.items():
        s = block["selected"]
        print(f"  {name:<16}{s['operating_point']:>8.3f}{s['recall']:>9.4f}"
              f"{s['precision']:>9.4f}{s['fp_per_frame']:>10.4f}"
              f"{str(block['fp_reduction_vs_uncalibrated']):>8}")
    print(f"\nselected: {winner}")
    print(f"wrote {args.out / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
