"""Stratify ball detection failures by cause, domain and ball size.

Phase 2D, Part 1.

Two questions this answers that a single recall number cannot:

* **Did the detector see it at all?** Every image is run twice, once at the
  operating threshold and once at ``--floor-conf`` (0.001). A ground-truth ball
  with a candidate at the floor but not at the operating point was *found and
  rejected by thresholding*; one with no candidate anywhere was *not seen*.
  Those two failures have completely different remedies and Phase 2B's taxonomy
  could not tell them apart.
* **What did the misses have in common?** Each miss is characterised by
  measurable image evidence at the ball's location -- scale, local blur,
  contrast against surroundings, distance to the nearest player box, line
  structure nearby -- and assigned to the single most specific matching
  category.

Categories are assigned by a fixed priority order, so every miss lands in
exactly one bucket and the percentages sum to 100. The order encodes which
explanation is more actionable when several apply: a 40 px^2 ball behind a
player is reported as an occlusion, because making the detector better at tiny
balls will not reveal a ball that is not visible.

Player boxes come from the shipped multiclass detector, not from ground truth:
the ball corpora do not all carry player labels. They are *evidence*, used only
to characterise a miss, never to score one.

Usage::

    python scripts/ball_failure_audit.py --model models/finetune/ball_multicorpus/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.ball_failures import (  # noqa: E402
    FailureCategory,
    FrameEvidence,
    GroundTruthBall,
    classify_miss,
    size_bucket,
)

log = get_logger("ball.failures")

DATASET = Path("data/ball_multicorpus")


def domain_of(path: Path) -> str:
    return "soccernet_gsr" if path.name.startswith("soccernet_gsr_") else "roboflow"


def load_gt(label_path: Path, width: int, height: int) -> list[GroundTruthBall]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append(
            GroundTruthBall(
                cx=cx * width, cy=cy * height,
                w=bw * width, h=bh * height,
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="models/yolo-football-ball-detection.pt",
        help="ball checkpoint to audit",
    )
    parser.add_argument(
        "--player-model", default="models/yolo-football-player-detection.pt"
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.08,
                        help="operating threshold, matching BallDetectionConfig")
    parser.add_argument("--floor-conf", type=float, default=0.001,
                        help="floor used to detect threshold-rejected balls")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--match-px", type=float, default=25.0,
                        help="centre distance counting as a match; Phase 2B measured 25 px")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/eval/ball_failures"))
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    images = sorted((DATASET / args.split / "images").glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        log.error("no images in %s/%s", DATASET, args.split)
        return 1
    log.info("auditing %s on %d image(s)", args.model, len(images))

    ball_model = YOLO(args.model)
    player_model = YOLO(args.player_model)

    counts: Counter = Counter()
    by_domain: dict[str, Counter] = defaultdict(Counter)
    by_size: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    coverage_loss: Counter = Counter()
    per_domain_hits: dict[str, list[int]] = defaultdict(list)
    false_positives: Counter = Counter()
    n_pred: Counter = Counter()

    for index, image_path in enumerate(images):
        if index and index % 200 == 0:
            log.info("  %d/%d", index, len(images))

        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        domain = domain_of(image_path)
        truths = load_gt(
            DATASET / args.split / "labels" / f"{image_path.stem}.txt", width, height
        )
        totals[domain] += len(truths)

        result = ball_model.predict(
            frame, imgsz=args.imgsz, conf=args.floor_conf, verbose=False
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
        scores = (
            result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros(0)
        )
        operating = [
            ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, s)
            for b, s in zip(boxes, scores, strict=True) if s >= args.conf
        ]
        floor = [
            ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, s)
            for b, s in zip(boxes, scores, strict=True) if s >= args.floor_conf
        ]
        n_pred[domain] += len(operating)

        players = player_model.predict(
            frame, imgsz=args.imgsz, conf=0.25, verbose=False
        )[0]
        player_boxes = (
            players.boxes.xyxy.cpu().numpy()
            if players.boxes is not None else np.zeros((0, 4))
        )

        matched_predictions: set[int] = set()
        for truth in truths:
            hit = None
            for slot, (px, py, _score) in enumerate(operating):
                if np.hypot(px - truth.cx, py - truth.cy) <= args.match_px:
                    hit = slot
                    break
            bucket = size_bucket(truth.area)
            if hit is not None:
                matched_predictions.add(hit)
                counts[FailureCategory.DETECTED.value] += 1
                by_domain[domain][FailureCategory.DETECTED.value] += 1
                by_size[bucket][FailureCategory.DETECTED.value] += 1
                per_domain_hits[domain].append(1)
                continue

            per_domain_hits[domain].append(0)
            best_floor = min(
                (np.hypot(px - truth.cx, py - truth.cy), s) for px, py, s in floor
            ) if floor else (float("inf"), 0.0)
            evidence = FrameEvidence.measure(
                frame, truth, player_boxes,
                floor_distance_px=best_floor[0],
                floor_confidence=best_floor[1],
                match_px=args.match_px,
            )
            category = classify_miss(truth, evidence)
            counts[category.value] += 1
            by_domain[domain][category.value] += 1
            by_size[bucket][category.value] += 1
            coverage_loss[category.value] += 1

        false_positives[domain] += len(operating) - len(matched_predictions)

    # -- report ---------------------------------------------------------------- #
    total_gt = sum(totals.values())
    detected = counts[FailureCategory.DETECTED.value]
    misses = total_gt - detected

    rows = []
    for category, count in counts.most_common():
        if category == FailureCategory.DETECTED.value:
            continue
        rows.append({
            "category": category,
            "count": count,
            "pct_of_misses": round(count / misses, 4) if misses else 0.0,
            "pct_of_all_gt": round(count / total_gt, 4) if total_gt else 0.0,
            "coverage_loss_pct": round(count / total_gt, 4) if total_gt else 0.0,
            "by_domain": {d: by_domain[d][category] for d in sorted(by_domain)},
            "by_size": {s: by_size[s][category] for s in sorted(by_size)},
        })

    payload = {
        "model": args.model,
        "split": args.split,
        "operating_conf": args.conf,
        "floor_conf": args.floor_conf,
        "match_px": args.match_px,
        "imgsz": args.imgsz,
        "n_images": len(images),
        "n_gt_balls": total_gt,
        "n_detected": detected,
        "recall": round(detected / total_gt, 4) if total_gt else 0.0,
        "per_domain": {
            d: {
                "n_gt": totals[d],
                "n_detected": by_domain[d][FailureCategory.DETECTED.value],
                "recall": round(
                    by_domain[d][FailureCategory.DETECTED.value] / totals[d], 4
                ) if totals[d] else 0.0,
                "n_predictions": n_pred[d],
                "n_false_positives": false_positives[d],
                "precision": round(
                    by_domain[d][FailureCategory.DETECTED.value] / n_pred[d], 4
                ) if n_pred[d] else 0.0,
            }
            for d in sorted(totals)
        },
        "by_size_bucket": {
            bucket: {
                "n_gt": sum(by_size[bucket].values()),
                "n_detected": by_size[bucket][FailureCategory.DETECTED.value],
                "recall": round(
                    by_size[bucket][FailureCategory.DETECTED.value]
                    / max(1, sum(by_size[bucket].values())), 4
                ),
            }
            for bucket in sorted(by_size)
        },
        "failure_categories": rows,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    name = Path(args.model).stem
    destination = args.out / f"failures_{name}_{args.split}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{args.model}  split={args.split}  recall={payload['recall']:.4f}  "
          f"({detected}/{total_gt})")
    print(f"\n{'category':<30}{'count':>7}{'% miss':>9}{'% all GT':>10}")
    for row in rows:
        print(f"  {row['category']:<28}{row['count']:>7}{row['pct_of_misses']:>9.3f}"
              f"{row['pct_of_all_gt']:>10.3f}")
    print(f"\n{'domain':<16}{'n_gt':>6}{'recall':>9}{'precision':>11}")
    for d, s in payload["per_domain"].items():
        print(f"  {d:<14}{s['n_gt']:>6}{s['recall']:>9.4f}{s['precision']:>11.4f}")
    print(f"\n{'size bucket':<16}{'n_gt':>6}{'recall':>9}")
    for b, s in payload["by_size_bucket"].items():
        print(f"  {b:<14}{s['n_gt']:>6}{s['recall']:>9.4f}")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
