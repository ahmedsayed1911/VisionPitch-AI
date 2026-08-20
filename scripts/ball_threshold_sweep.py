"""Calibrate the ball detector's operating confidence, and report two matchers.

Phase 2D.

Two things are measured here, and conflating them is what made Phase 2C's
headline number misleading.

**Matching criterion.** A ball on broadcast footage is about 11 px across. At
that size IoU >= 0.5 requires the predicted box to sit within roughly two
pixels of truth, so the standard detection metric is dominated by *localisation
precision*. Nothing downstream needs that: possession compares the ball centre
against player boxes at a radius of tens of pixels. Both are reported --
``iou50`` for continuity with every previously published figure, and
``centre25`` for what the pipeline actually consumes. Neither replaces the
other, and the readiness threshold is judged on the one it was declared against.

**Operating threshold.** The failure audit found 24.3% of missed balls had a
candidate at confidence 0.001 that was discarded at the 0.08 operating point.
That is recall available without touching the model, if precision holds. The
threshold is chosen on train+val and measured once on test.

Usage::

    python scripts/ball_threshold_sweep.py --model models/finetune/ball_multicorpus/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("ball.threshold")

DATASET = Path("data/ball_multicorpus")
GRID = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50]


def domain_of(path: Path) -> str:
    return "soccernet_gsr" if path.name.startswith("soccernet_gsr_") else "roboflow"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def score_split(model, images: list[Path], imgsz: int, floor: float, match_px: float):
    """Collect per-image predictions once, then score every threshold offline."""
    records = []
    for index, path in enumerate(images):
        if index and index % 250 == 0:
            log.info("  %d/%d", index, len(images))
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        height, width = frame.shape[:2]

        truths = []
        label = DATASET / path.parent.parent.name / "labels" / f"{path.stem}.txt"
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                cx, cy, bw, bh = cx * width, cy * height, bw * width, bh * height
                truths.append(np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))

        result = model.predict(frame, imgsz=imgsz, conf=floor, verbose=False)[0]
        boxes = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None else np.zeros((0, 4))
        )
        scores = (
            result.boxes.conf.cpu().numpy()
            if result.boxes is not None else np.zeros(0)
        )
        records.append((domain_of(path), truths, boxes, scores))
    return records


def evaluate_at(records, threshold: float, match_px: float):
    """Greedy one-to-one matching under both criteria, per domain."""
    stats = defaultdict(lambda: {
        "tp_iou": 0, "fp_iou": 0, "fn_iou": 0,
        "tp_ctr": 0, "fp_ctr": 0, "fn_ctr": 0,
    })
    for domain, truths, boxes, scores in records:
        keep = scores >= threshold
        predictions = boxes[keep]
        order = np.argsort(-scores[keep])
        predictions = predictions[order]

        for criterion in ("iou", "ctr"):
            used = set()
            hits = 0
            for truth in truths:
                # -inf, not -1: the centre criterion scores candidates as
                # NEGATIVE distance, so a -1 floor silently rejected every match
                # further than one pixel and made centre25 recall come out below
                # iou50 recall, which cannot happen.
                best, best_slot = float("-inf"), None
                for slot, prediction in enumerate(predictions):
                    if slot in used:
                        continue
                    if criterion == "iou":
                        value = iou(prediction, truth)
                        ok = value >= 0.5
                    else:
                        pcx = (prediction[0] + prediction[2]) / 2
                        pcy = (prediction[1] + prediction[3]) / 2
                        tcx = (truth[0] + truth[2]) / 2
                        tcy = (truth[1] + truth[3]) / 2
                        distance = float(np.hypot(pcx - tcx, pcy - tcy))
                        value = -distance
                        ok = distance <= match_px
                    if ok and value > best:
                        best, best_slot = value, slot
                if best_slot is not None:
                    used.add(best_slot)
                    hits += 1
            suffix = criterion
            stats[domain][f"tp_{suffix}"] += hits
            stats[domain][f"fn_{suffix}"] += len(truths) - hits
            stats[domain][f"fp_{suffix}"] += len(predictions) - hits
    return stats


def rates(counts: dict, suffix: str) -> dict:
    tp, fp, fn = counts[f"tp_{suffix}"], counts[f"fp_{suffix}"], counts[f"fn_{suffix}"]
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"recall": round(recall, 4), "precision": round(precision, 4), "f1": round(f1, 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/finetune/ball_multicorpus/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--match-px", type=float, default=25.0)
    parser.add_argument("--floor", type=float, default=0.005)
    parser.add_argument("--min-precision", type=float, default=0.55,
                        help="fixed precision floor from the Phase 2D thresholds")
    parser.add_argument("--out", type=Path, default=Path("data/eval/ball_threshold"))
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    model = YOLO(args.model)
    selection_images = (
        sorted((DATASET / "train" / "images").glob("*.jpg"))
        + sorted((DATASET / "val" / "images").glob("*.jpg"))
    )
    test_images = sorted((DATASET / "test" / "images").glob("*.jpg"))
    log.info("selection set %d image(s), test %d", len(selection_images), len(test_images))

    log.info("scoring selection set")
    selection = score_split(model, selection_images, args.imgsz, args.floor, args.match_px)
    log.info("scoring test set")
    test = score_split(model, test_images, args.imgsz, args.floor, args.match_px)

    def summarise(records, threshold):
        stats = evaluate_at(records, threshold, args.match_px)
        pooled = defaultdict(int)
        for counts in stats.values():
            for key, value in counts.items():
                pooled[key] += value
        return {
            "per_domain": {
                d: {"iou50": rates(c, "iou"), "centre25": rates(c, "ctr")}
                for d, c in sorted(stats.items())
            },
            "macro_iou50": {
                metric: round(
                    sum(rates(c, "iou")[metric] for c in stats.values()) / len(stats), 4
                )
                for metric in ("recall", "precision", "f1")
            },
            "macro_centre25": {
                metric: round(
                    sum(rates(c, "ctr")[metric] for c in stats.values()) / len(stats), 4
                )
                for metric in ("recall", "precision", "f1")
            },
            "worst_domain_recall_iou50": round(
                min(rates(c, "iou")["recall"] for c in stats.values()), 4
            ),
            "worst_domain_recall_centre25": round(
                min(rates(c, "ctr")["recall"] for c in stats.values()), 4
            ),
        }

    sweep = []
    for threshold in GRID:
        entry = {"threshold": threshold, **summarise(selection, threshold)}
        sweep.append(entry)
        log.info(
            "conf %.3f  iou50 R %.4f P %.4f | centre25 R %.4f P %.4f | worst-domain R %.4f",
            threshold, entry["macro_iou50"]["recall"], entry["macro_iou50"]["precision"],
            entry["macro_centre25"]["recall"], entry["macro_centre25"]["precision"],
            entry["worst_domain_recall_centre25"],
        )

    # Selection rule, fixed before the numbers: maximise worst-domain
    # centre-distance recall subject to macro precision staying at or above the
    # declared floor. Worst-domain rather than mean, because the whole point of
    # multi-corpus work is that the weakest domain is the product's real quality.
    eligible = [
        e for e in sweep if e["macro_centre25"]["precision"] >= args.min_precision
    ]
    chosen = (
        max(eligible, key=lambda e: e["worst_domain_recall_centre25"])
        if eligible else max(sweep, key=lambda e: e["macro_centre25"]["precision"])
    )

    held_out = summarise(test, chosen["threshold"])
    baseline_test = summarise(test, 0.08)

    payload = {
        "model": args.model,
        "imgsz": args.imgsz,
        "match_px": args.match_px,
        "min_precision": args.min_precision,
        "selection_set": "train + val (test never used for selection)",
        "n_selection_images": len(selection_images),
        "n_test_images": len(test_images),
        "selection_rule": (
            "maximise worst-domain centre25 recall subject to macro centre25 "
            "precision >= min_precision; declared before the sweep was run"
        ),
        "grid": GRID,
        "sweep_on_selection_set": sweep,
        "chosen_threshold": chosen["threshold"],
        "eligible_thresholds": [e["threshold"] for e in eligible],
        "test_at_chosen": held_out,
        "test_at_previous_default_0.08": baseline_test,
        "matcher_note": (
            "iou50 is the standard criterion and is what every figure before "
            "Phase 2D used. centre25 accepts a prediction whose centre is within "
            "25 px of truth. On an ~11 px ball the two differ enormously, and "
            "centre25 is the one the possession engine consumes."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"threshold_{Path(args.model).stem}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nchosen threshold: {chosen['threshold']} (previous default 0.08)")
    for label, block in (("test @ 0.08", baseline_test), ("test @ chosen", held_out)):
        print(f"\n  {label}")
        print(f"    iou50    macro R {block['macro_iou50']['recall']:.4f} "
              f"P {block['macro_iou50']['precision']:.4f} "
              f"worst-domain R {block['worst_domain_recall_iou50']:.4f}")
        print(f"    centre25 macro R {block['macro_centre25']['recall']:.4f} "
              f"P {block['macro_centre25']['precision']:.4f} "
              f"worst-domain R {block['worst_domain_recall_centre25']:.4f}")
        for d, r in block["per_domain"].items():
            iou, ctr = r["iou50"], r["centre25"]
            print(
                f"      {d:<15} iou50 R {iou['recall']:.4f} P {iou['precision']:.4f}"
                f"  | centre25 R {ctr['recall']:.4f} P {ctr['precision']:.4f}"
            )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
