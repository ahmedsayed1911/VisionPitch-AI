"""Structured error analysis for out-of-distribution ball detection.

Phase 2B, Part 5. The rule this script exists to serve: *do not begin
fine-tuning until the failure distribution is known.* An 0.300 recall could be
one dominant cause or twelve small ones, and the right intervention is
completely different in each case.

Every miss and every false positive is attributed to a measurable cause, not a
guessed one. The distinction that matters most is between:

* **detector blind** — no candidate produced anywhere near the true ball even at
  a floor-level confidence. A model problem; fine-tuning may help.
* **confidence rejected** — a candidate existed at the right place but scored
  below the operating threshold. A calibration problem; a threshold change fixes
  it at some precision cost.
* **suppressed** — a candidate existed and was discarded by NMS or the
  per-frame candidate cap. A plumbing problem; free to fix.

Only the first justifies retraining. Reporting them together as "recall is low"
hides which lever to pull.

Usage::

    python scripts/ball_error_analysis.py --sequences 8 --max-frames 150
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.config import AnalysisMode, load_config  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.common.types import BBox, ObjectClass  # noqa: E402
from visionpitch.evaluation.datasets import GSRDataset  # noqa: E402

log = get_logger("ball.error_analysis")

#: A prediction this close to the truth counts as a hit. Deliberately looser
#: than the IoU 0.5 used for scoring: here we are asking "did the detector see
#: something at the ball", not "did it localise it precisely".
HIT_IOU = 0.3
HIT_CENTRE_PX = 12.0


def centre_distance(a: BBox, b: BBox) -> float:
    ax, ay = a.center
    bx, by = b.center
    return float(np.hypot(ax - bx, ay - by))


def is_hit(pred: BBox, truth: BBox) -> bool:
    return pred.iou(truth) >= HIT_IOU or centre_distance(pred, truth) <= HIT_CENTRE_PX


def sharpness(patch: np.ndarray) -> float:
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def local_contrast(patch: np.ndarray) -> float:
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return float(gray.std())


def whiteness(patch: np.ndarray) -> float:
    """Fraction of near-white pixels. High around pitch lines, which is where a
    white ball loses contrast against the background."""
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return float((gray > 190).mean())


def classify_miss(
    truth: BBox,
    image: np.ndarray,
    players: list[BBox],
    low_conf_candidates: list[tuple[BBox, float]],
    operating_threshold: float,
) -> str:
    """Attribute one missed ball to a single dominant cause.

    Order matters: the pipeline-side explanations are checked first, because if
    a candidate existed at the right place the model was *not* blind, whatever
    the image looked like.
    """
    near = [(b, c) for b, c in low_conf_candidates if is_hit(b, truth)]
    if near:
        best_conf = max(c for _, c in near)
        if best_conf >= operating_threshold:
            # Present, above threshold, and still not in the final output.
            return "suppressed_or_capped"
        return "confidence_rejected"

    # No candidate anywhere near the ball. Now ask what the image looked like.
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in truth.to_xyxy())
    pad = 12
    patch = image[max(0, y1 - pad): min(h, y2 + pad), max(0, x1 - pad): min(w, x2 + pad)]

    area = truth.area
    if area < 60:
        return "tiny_scale"

    occluding = [p for p in players if p.iou(truth) > 0.02 or centre_distance(p, truth) < 45]
    if occluding:
        overlapped = any(p.iou(truth) > 0.15 for p in occluding)
        return "occluded_by_player" if overlapped else "near_player_body"

    if whiteness(patch) > 0.35:
        return "low_contrast_on_lines"
    if sharpness(patch) < 25.0:
        return "motion_blur"
    if local_contrast(patch) < 18.0:
        return "low_contrast"
    return "detector_blind_other"


#: A prediction further than this from the truth is a different object, not a
#: poorly-localised ball.
#:
#: Measured, not chosen: the distance from a ground-truth ball to the nearest
#: prediction is sharply bimodal. 48.3% of frames have a prediction within 12 px
#: and 49.3% within 60 px -- a single percentage point in between. There is no
#: population of "found but mislocalised" detections. A 60 px threshold, used
#: initially, therefore labelled hundreds of unrelated false positives as near
#: misses and made detector blindness look like a localisation problem.
NEAR_MISS_PX = 25.0


def classify_false_positive(
    pred: BBox, image: np.ndarray, players: list[BBox], truth: BBox | None
) -> str:
    if truth is not None and centre_distance(pred, truth) < NEAR_MISS_PX:
        return "near_miss_localisation"
    on_player = [p for p in players if p.iou(pred) > 0.05 or centre_distance(p, pred) < 40]
    if on_player:
        return "false_on_player"

    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in pred.to_xyxy())
    patch = image[max(0, y1): min(h, y2), max(0, x1): min(w, x2)]
    if whiteness(patch) > 0.4:
        return "false_on_line_or_marking"
    if pred.center[1] < 0.25 * h:
        return "false_in_crowd_or_stands"
    return "false_other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/eval/gsr"))
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--mode", default="balanced")
    parser.add_argument("--floor-conf", type=float, default=0.01,
                        help="floor threshold used to ask whether the model saw anything")
    parser.add_argument("--out", type=Path,
                        default=Path("data/eval/gsr/benchmarks/ball_error_analysis.json"))
    args = parser.parse_args()

    configure_logging("WARNING")
    config = load_config(mode=AnalysisMode(args.mode))
    operating = config.ball_detection.conf_threshold

    from visionpitch.detection.ball import BallDetector
    from visionpitch.detection.yolo import build_detector

    detector = build_detector(config)
    ball_detector = BallDetector(config)

    dataset = GSRDataset(args.root, max_sequences=args.sequences)
    print(f"corpus  : {dataset.info().name} ({dataset.info().kind})")
    print(f"seqs    : {len(dataset.sequences)}, {args.max_frames} frames each max")
    print(f"operating ball threshold = {operating}, floor = {args.floor_conf}\n")

    miss_causes: Counter = Counter()
    fp_causes: Counter = Counter()
    size_hits: dict[str, list[int]] = {"tiny(<150)": [0, 0], "small(150-400)": [0, 0],
                                       "medium(400-900)": [0, 0], "large(>900)": [0, 0]}
    n_gt = n_hit = n_pred = 0
    frames_seen = 0

    for sequence in dataset.sequences:
        frames = sorted(sequence.image_paths)[: args.max_frames]
        for frame_idx in frames:
            objects = sequence.ground_truth.frames.get(frame_idx, [])
            truths = [o.bbox for o in objects if o.object_class is ObjectClass.BALL]
            players = [o.bbox for o in objects if o.object_class.is_person]

            image = cv2.imread(str(sequence.image_paths[frame_idx]))
            if image is None:
                continue
            frames_seen += 1

            # Operating-point output: exactly what the pipeline would use.
            multiclass = detector.detect_batch([image], [frame_idx])[0]
            specialist = ball_detector.detect_tiled(image, frame_idx)
            predicted = [
                d.bbox for d in (multiclass + specialist)
                if d.object_class is ObjectClass.BALL and d.confidence >= operating
            ]

            # Floor-level output: did the model see *anything* there?
            floor = ball_detector.model.predict(
                image, imgsz=config.ball_detection.imgsz, conf=args.floor_conf,
                iou=0.5, max_det=50, device=ball_detector.device,
                quantize="fp16" if ball_detector.half else None, verbose=False,
            )[0]
            low_conf: list[tuple[BBox, float]] = []
            if floor.boxes is not None and len(floor.boxes):
                for box, conf in zip(
                    floor.boxes.xyxy.cpu().numpy(), floor.boxes.conf.cpu().numpy(), strict=True
                ):
                    low_conf.append((BBox.from_xyxy(box), float(conf)))

            n_gt += len(truths)
            n_pred += len(predicted)
            matched_preds: set[int] = set()

            for truth in truths:
                bucket = ("tiny(<150)" if truth.area < 150
                          else "small(150-400)" if truth.area < 400
                          else "medium(400-900)" if truth.area < 900 else "large(>900)")
                size_hits[bucket][1] += 1

                hit_index = next(
                    (i for i, p in enumerate(predicted)
                     if i not in matched_preds and is_hit(p, truth)), None
                )
                if hit_index is not None:
                    matched_preds.add(hit_index)
                    n_hit += 1
                    size_hits[bucket][0] += 1
                else:
                    miss_causes[
                        classify_miss(truth, image, players, low_conf, operating)
                    ] += 1

            for i, pred in enumerate(predicted):
                if i in matched_preds:
                    continue
                nearest = min(truths, key=lambda t: centre_distance(pred, t)) if truths else None
                fp_causes[classify_false_positive(pred, image, players, nearest)] += 1

        print(f"  {sequence.name}: cumulative recall "
              f"{n_hit}/{n_gt} = {n_hit / max(1, n_gt):.3f}")

    recall = n_hit / max(1, n_gt)
    n_fp = sum(fp_causes.values())
    report = {
        "corpus": dataset.info().to_dict(),
        "operating_threshold": operating,
        "floor_threshold": args.floor_conf,
        "frames": frames_seen,
        "ground_truth_balls": n_gt,
        "predictions": n_pred,
        "hits": n_hit,
        "recall": round(recall, 4),
        "precision": round(n_hit / max(1, n_pred), 4),
        "false_positives_per_frame": round(n_fp / max(1, frames_seen), 3),
        "miss_causes": dict(miss_causes.most_common()),
        "miss_causes_pct": {
            k: round(100 * v / max(1, sum(miss_causes.values())), 1)
            for k, v in miss_causes.most_common()
        },
        "false_positive_causes": dict(fp_causes.most_common()),
        "recall_by_size": {
            k: {"hits": v[0], "total": v[1],
                "recall": round(v[0] / v[1], 4) if v[1] else None}
            for k, v in size_hits.items()
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"BALL RECALL (OOD) {n_hit}/{n_gt} = {recall:.3f}   "
          f"precision {report['precision']:.3f}   FP/frame {report['false_positives_per_frame']}")
    print("\nMISS CAUSES")
    for cause, count in miss_causes.most_common():
        print(f"  {cause:26s} {count:6d}  {report['miss_causes_pct'][cause]:5.1f}%")
    print("\nFALSE POSITIVE CAUSES")
    for cause, count in fp_causes.most_common():
        print(f"  {cause:26s} {count:6d}")
    print("\nRECALL BY BALL SIZE")
    for bucket, stats in report["recall_by_size"].items():
        print(f"  {bucket:18s} {stats['hits']:5d}/{stats['total']:<6d} {stats['recall']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
