"""Score every ball candidate identically on the locked public and local tests.

Same architecture, same input size, same matcher, same metrics, same splits for
every candidate. Per-image ground truth and the image-level metadata used for
the breakdowns (ball scale, blur, occlusion) are computed **once** and reused,
so no candidate is scored against a slightly different reference.

Order of operations, which is the point of the script:

1. sweep the confidence threshold on **validation only**
2. freeze it
3. score the locked public test and the locked local test exactly once

Two honesty rules are enforced in the output rather than left to the write-up:

* **No local precision is reported.** The local test contains 23 positive frames
  and zero negatives, so there is no denominator for a false-positive rate. The
  field is emitted as ``null`` with the reason attached.
* **False-positive behaviour comes from the public negatives** and is labelled
  cross-domain evidence, because that is what it is.

Usage::

    python scripts/evaluate_broadcast_candidates.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.annotation.schema import AnnotationStore, BallVisibility  # noqa: E402
from visionpitch.annotation.splits import LocalSplit  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("broadcast.evaluate")

MULTICORPUS = Path("data/ball_multicorpus")
LOCAL_PACKAGE = Path("data/annotation/package")
LOCAL_SPLIT = Path("data/annotation/local_split.json")

CENTRE_TOLERANCES = (5.0, 10.0, 15.0, 20.0, 25.0)
THRESHOLD_GRID = [0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.60]
#: Declared before the sweep: maximise centre-25 recall subject to this floor.
MIN_VAL_PRECISION = 0.55

TINY_AREA, SMALL_AREA = 150.0, 400.0
BLUR_CUT = 40.0
OCCLUSION_MARGIN = 6.0

CANDIDATES = {
    "A_default": "models/yolo-football-ball-detection.pt",
    "B_public": "models/finetune/bcast_public/weights/best.pt",
    "C_adapt": "models/finetune/bcast_adapt/weights/best.pt",
    "D_adapt_aug": "models/finetune/bcast_adapt_aug/weights/best.pt",
    "C_hardened": "models/finetune/bcast_hardened/weights/best.pt",
}


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    spread = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def domain_of(path: Path) -> str:
    return "soccernet_gsr" if path.name.startswith("soccernet_gsr_") else "roboflow"


def patch_blur(image, cx, cy, side) -> float:
    h, w = image.shape[:2]
    r = max(4, int(round(side)))
    patch = image[
        int(max(0, cy - r)): int(min(h, cy + r)),
        int(max(0, cx - r)): int(min(w, cx + r)),
    ]
    if patch.size == 0:
        return 1e9
    return float(cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


# --------------------------------------------------------------------------- #
# Reference data, computed once
# --------------------------------------------------------------------------- #


def build_public_reference(split: str, person_detector) -> list[dict]:
    from visionpitch.common.types import ObjectClass

    records = []
    images = sorted((MULTICORPUS / split / "images").glob("*.jpg"))
    for index, path in enumerate(images):
        if index and index % 300 == 0:
            log.info("  reference %s %d/%d", split, index, len(images))
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        label = path.parent.parent / "labels" / f"{path.stem}.txt"
        balls = []
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                balls.append((cx * w, cy * h, bw * w, bh * h))

        occluded = []
        if balls and person_detector is not None:
            detections = person_detector.detect_batch([image], [index])[0]
            people = [
                d.bbox for d in detections
                if d.object_class in (
                    ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE
                )
            ]
            for cx, cy, _, _ in balls:
                near = False
                for box in people:
                    if box.x1 <= cx <= box.x2 and box.y1 <= cy <= box.y2:
                        near = True
                        break
                    dx = max(box.x1 - cx, 0.0, cx - box.x2)
                    dy = max(box.y1 - cy, 0.0, cy - box.y2)
                    if np.hypot(dx, dy) <= OCCLUSION_MARGIN:
                        near = True
                        break
                occluded.append(near)
        else:
            occluded = [False] * len(balls)

        records.append({
            "path": str(path),
            "domain": domain_of(path),
            "size": (w, h),
            "balls": balls,
            "blurs": [patch_blur(image, cx, cy, max(bw, bh)) for cx, cy, bw, bh in balls],
            "occluded": occluded,
            "is_negative": not balls,
        })
    return records


def build_local_reference(split_name: str) -> list[dict]:
    store = AnnotationStore(LOCAL_PACKAGE)
    samples = store.load_samples()
    annotations = store.load_annotations()
    split = LocalSplit.load(LOCAL_SPLIT)
    records = []
    for frame_id in sorted(split.frames.get(split_name, [])):
        annotation = annotations[frame_id]
        sample = samples[frame_id]
        image = cv2.imread(sample.image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        balls = []
        if annotation.visibility is BallVisibility.VISIBLE and annotation.radius_px:
            d = annotation.radius_px * 2
            balls.append((annotation.centre_x, annotation.centre_y, d, d))
        records.append({
            "path": sample.image_path,
            "frame_id": frame_id,
            "domain": "local_broadcast",
            "size": (w, h),
            "balls": balls,
            "blurs": [patch_blur(image, b[0], b[1], max(b[2], b[3])) for b in balls],
            "occluded": [False] * len(balls),
            "is_negative": not balls,
            "category": sample.sampling_category.value,
        })
    return records


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def predict_all(model, records, imgsz: int, floor: float):
    """One inference pass per candidate; thresholds are applied offline."""
    out = []
    started = time.perf_counter()
    for record in records:
        image = cv2.imread(record["path"])
        if image is None:
            out.append(([], []))
            continue
        result = model.predict(image, imgsz=imgsz, conf=floor, verbose=False)[0]
        if result.boxes is None or not len(result.boxes):
            out.append(([], []))
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        order = np.argsort(-scores)
        out.append((boxes[order], scores[order]))
    elapsed = time.perf_counter() - started
    return out, 1000 * elapsed / max(1, len(records))


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    )


def score(records, predictions, threshold: float) -> dict:
    """Greedy one-to-one matching under IoU50 and centre distance."""
    stats = {
        "iou_tp": 0, "iou_fp": 0, "iou_fn": 0,
        "n_truth": 0, "n_pred": 0, "n_frames": 0,
        "hits": dict.fromkeys(CENTRE_TOLERANCES, 0),
        "errors": [],
        "fp_on_negative_frames": 0, "n_negative_frames": 0,
        "by_size": defaultdict(lambda: {"n": 0, "hit": 0}),
        "by_blur": defaultdict(lambda: {"n": 0, "hit": 0}),
        "by_occlusion": defaultdict(lambda: {"n": 0, "hit": 0}),
        "by_domain": defaultdict(lambda: {"n": 0, "hit": 0, "pred": 0}),
    }

    for record, (boxes, scores_) in zip(records, predictions, strict=True):
        stats["n_frames"] += 1
        keep = [b for b, s in zip(boxes, scores_, strict=True) if s >= threshold] \
            if len(boxes) else []
        stats["n_pred"] += len(keep)
        truths = record["balls"]
        stats["n_truth"] += len(truths)
        domain = record["domain"]
        stats["by_domain"][domain]["n"] += len(truths)
        stats["by_domain"][domain]["pred"] += len(keep)

        if record["is_negative"]:
            stats["n_negative_frames"] += 1
            stats["fp_on_negative_frames"] += len(keep)
            stats["iou_fp"] += len(keep)
            continue

        centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in keep]
        used_centre: set[int] = set()
        for slot, (cx, cy, bw, bh) in enumerate(truths):
            area = bw * bh
            bucket = ("1_tiny" if area < TINY_AREA
                      else "2_small" if area < SMALL_AREA else "3_medium_plus")
            blurred = record["blurs"][slot] < BLUR_CUT
            occluded = record["occluded"][slot] if slot < len(record["occluded"]) else False

            best, best_slot = float("inf"), None
            for j, (px, py) in enumerate(centres):
                if j in used_centre:
                    continue
                d = float(np.hypot(px - cx, py - cy))
                if d < best:
                    best, best_slot = d, j
            if best_slot is not None:
                used_centre.add(best_slot)
                stats["errors"].append(best)
            for tolerance in CENTRE_TOLERANCES:
                if best <= tolerance:
                    stats["hits"][tolerance] += 1

            hit25 = best <= 25.0
            stats["by_size"][bucket]["n"] += 1
            stats["by_size"][bucket]["hit"] += int(hit25)
            stats["by_blur"]["blurred" if blurred else "sharp"]["n"] += 1
            stats["by_blur"]["blurred" if blurred else "sharp"]["hit"] += int(hit25)
            stats["by_occlusion"]["occluded" if occluded else "clear"]["n"] += 1
            stats["by_occlusion"]["occluded" if occluded else "clear"]["hit"] += int(hit25)
            stats["by_domain"][domain]["hit"] += int(hit25)

        # IoU50 accounting, from the same kept predictions
        used_box: set[int] = set()
        hits = 0
        for cx, cy, bw, bh in truths:
            truth_box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            best, best_slot = 0.5, None
            for j, box in enumerate(keep):
                if j in used_box:
                    continue
                value = iou(box, truth_box)
                if value >= best:
                    best, best_slot = value, j
            if best_slot is not None:
                used_box.add(best_slot)
                hits += 1
        stats["iou_tp"] += hits
        stats["iou_fn"] += len(truths) - hits
        stats["iou_fp"] += len(keep) - hits

    return stats


def rates(stats: dict, local: bool) -> dict:
    tp, fp, fn = stats["iou_tp"], stats["iou_fp"], stats["iou_fn"]
    iou_recall = tp / (tp + fn) if tp + fn else 0.0
    iou_precision = tp / (tp + fp) if tp + fp else 0.0
    n_truth = stats["n_truth"]

    centre = {}
    for tolerance in CENTRE_TOLERANCES:
        hits = stats["hits"][tolerance]
        low, high = wilson(hits, n_truth)
        centre[str(tolerance)] = {
            "recall": round(hits / n_truth, 4) if n_truth else 0.0,
            "ci95": [round(low, 4), round(high, 4)],
        }

    out = {
        "n_frames": stats["n_frames"],
        "n_truth": n_truth,
        "n_predictions": stats["n_pred"],
        "iou50": {
            "recall": round(iou_recall, 4),
            "precision": None if local else round(iou_precision, 4),
            "f1": None if local else round(
                2 * iou_precision * iou_recall / (iou_precision + iou_recall), 4
            ) if iou_precision + iou_recall else 0.0,
        },
        "centre_recall": centre,
        "median_centre_error_px": (
            round(float(np.median(stats["errors"])), 3) if stats["errors"] else None
        ),
        "by_ball_size": {
            k: {
                "n": v["n"],
                "recall_25px": round(v["hit"] / v["n"], 4) if v["n"] else None,
            }
            for k, v in sorted(stats["by_size"].items())
        },
        "by_blur": {
            k: {
                "n": v["n"],
                "recall_25px": round(v["hit"] / v["n"], 4) if v["n"] else None,
            }
            for k, v in sorted(stats["by_blur"].items())
        },
        "by_occlusion": {
            k: {
                "n": v["n"],
                "recall_25px": round(v["hit"] / v["n"], 4) if v["n"] else None,
            }
            for k, v in sorted(stats["by_occlusion"].items())
        },
        "by_domain": {
            k: {
                "n_truth": v["n"],
                "recall_25px": round(v["hit"] / v["n"], 4) if v["n"] else None,
            }
            for k, v in sorted(stats["by_domain"].items())
        },
    }
    if local:
        out["precision"] = None
        out["precision_unavailable_reason"] = (
            f"the local test contains {stats['n_negative_frames']} negative frames; "
            f"with no no-ball denominator a precision or false-positive rate "
            f"cannot be computed. See the public negatives for cross-domain "
            f"false-positive evidence."
        )
    else:
        negatives = stats["n_negative_frames"]
        out["false_positives"] = {
            "per_frame_all": round(stats["iou_fp"] / max(1, stats["n_frames"]), 4),
            "n_negative_frames": negatives,
            "fp_on_negative_frames": stats["fp_on_negative_frames"],
            "fp_per_negative_frame": round(
                stats["fp_on_negative_frames"] / negatives, 4
            ) if negatives else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--floor", type=float, default=0.02)
    parser.add_argument("--out", type=Path,
                        default=Path("data/eval/broadcast/candidates.json"))
    args = parser.parse_args()

    configure_logging("INFO")
    import torch
    from ultralytics import YOLO

    from visionpitch.common.config import load_config
    from visionpitch.detection.yolo import build_detector

    log.info("building shared reference data (computed once, reused by all candidates)")
    person_detector = build_detector(load_config())
    public_val = build_public_reference("val", person_detector)
    public_test = build_public_reference("test", person_detector)
    local_val = build_local_reference("val")
    local_test = build_local_reference("test")
    log.info(
        "reference: public val %d, public test %d, local val %d, local test %d",
        len(public_val), len(public_test), len(local_val), len(local_test),
    )
    del person_detector
    torch.cuda.empty_cache()

    results: dict[str, dict] = {}
    for label, weights in CANDIDATES.items():
        path = Path(weights)
        if not path.exists():
            log.warning("%s missing at %s; skipped", label, path)
            continue
        log.info("evaluating %s", label)
        model = YOLO(str(path))
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        # -- 1. threshold sweep on VALIDATION only --------------------------- #
        val_records = public_val + local_val
        val_predictions, _ = predict_all(model, val_records, args.imgsz, args.floor)
        sweep = []
        for threshold in THRESHOLD_GRID:
            stats = score(val_records, val_predictions, threshold)
            n_truth = stats["n_truth"]
            recall = stats["hits"][25.0] / n_truth if n_truth else 0.0
            precision = (
                stats["hits"][25.0] / stats["n_pred"] if stats["n_pred"] else 0.0
            )
            sweep.append({
                "threshold": threshold,
                "val_centre_recall_25": round(recall, 4),
                "val_centre_precision_25": round(precision, 4),
            })
        eligible = [
            row for row in sweep
            if row["val_centre_precision_25"] >= MIN_VAL_PRECISION
        ]
        chosen = (
            max(eligible, key=lambda r: r["val_centre_recall_25"]) if eligible
            else max(sweep, key=lambda r: r["val_centre_precision_25"])
        )
        threshold = chosen["threshold"]
        log.info("  chosen threshold %.2f (val recall %.4f, precision %.4f)",
                 threshold, chosen["val_centre_recall_25"],
                 chosen["val_centre_precision_25"])

        # -- 2. locked tests, scored once at the frozen threshold ------------- #
        public_predictions, public_ms = predict_all(
            model, public_test, args.imgsz, args.floor
        )
        local_predictions, local_ms = predict_all(
            model, local_test, args.imgsz, args.floor
        )
        peak_mb = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available() else None
        )

        results[label] = {
            "weights": str(path),
            "checkpoint_fingerprint": fingerprint(path),
            "imgsz": args.imgsz,
            "threshold_sweep_on_validation": sweep,
            "threshold_selection_rule": (
                f"maximise centre-25 recall on validation subject to centre-25 "
                f"precision >= {MIN_VAL_PRECISION}; declared before the sweep"
            ),
            "selected_threshold": threshold,
            "public_test": rates(score(public_test, public_predictions, threshold), False),
            "local_test": rates(score(local_test, local_predictions, threshold), True),
            "runtime_ms_per_frame": round(public_ms, 2),
            "local_runtime_ms_per_frame": round(local_ms, 2),
            "peak_gpu_mb": round(peak_mb, 1) if peak_mb else None,
        }
        del model
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'candidate':<14}{'thr':>6}{'pubR':>8}{'pubP':>8}{'pubF1':>8}"
          f"{'locR@25':>9}{'locErr':>8}{'FP/neg':>8}{'ms':>7}{'GPUmb':>8}")
    for label, block in results.items():
        pub, loc = block["public_test"], block["local_test"]
        fp = pub["false_positives"]["fp_per_negative_frame"]
        print(f"{label:<14}{block['selected_threshold']:>6.2f}"
              f"{pub['iou50']['recall']:>8.4f}{pub['iou50']['precision']:>8.4f}"
              f"{pub['iou50']['f1']:>8.4f}"
              f"{loc['centre_recall']['25.0']['recall']:>9.4f}"
              f"{str(loc['median_centre_error_px']):>8}"
              f"{str(fp):>8}{block['runtime_ms_per_frame']:>7.1f}"
              f"{str(block['peak_gpu_mb']):>8}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
