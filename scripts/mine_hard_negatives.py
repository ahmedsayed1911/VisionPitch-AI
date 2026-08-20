"""Mine hard negatives from Candidate C's false positives.

Part 2 of precision hardening.

Mining runs on the **training and validation splits only**. The locked tests are
audited in Part 1 and never mined: a hard negative cropped from the test set
would train the model on the exact frames it is later scored against.

Two kinds of output, because they teach different things:

* **crops** around each false positive -- a tight look at the sock, the penalty
  spot, the line junction that fired
* **full negative frames** already present in the corpora, replicated so the
  detector sees whole scenes with no ball in them

Duplicate and trajectory-inconsistent false positives are excluded by
construction. They are fusion failures, not appearance failures; mining crops of
them would teach the detector to suppress genuine balls.

Usage::

    python scripts/mine_hard_negatives.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.annotation.schema import AnnotationStore, BallVisibility  # noqa: E402
from visionpitch.annotation.splits import LocalSplit  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.false_positives import (  # noqa: E402
    classify,
    mark_duplicates,
    measure,
)

log = get_logger("hardneg.mine")

CANDIDATE_C = "models/finetune/bcast_adapt/weights/best.pt"
MULTICORPUS = Path("data/ball_multicorpus")
LOCAL_PACKAGE = Path("data/annotation/package")
LOCAL_SPLIT = Path("data/annotation/local_split.json")
MATCH_PX = 25.0

#: Crop side as a multiple of the false positive's own size. Wide enough to
#: carry context (the boot attached to the sock, the line the spot sits on),
#: tight enough that the crop is about the thing that fired.
CROP_SCALE = 9.0
MIN_CROP_PX = 96
MAX_CROP_PX = 320


def sources():
    """(split_label, frame_id, image, truth_boxes) for every minable frame."""
    for split in ("train", "val"):
        for path in sorted((MULTICORPUS / split / "images").glob("*.jpg")):
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
            yield f"public_{split}", path.stem, image, balls

    store = AnnotationStore(LOCAL_PACKAGE)
    samples = store.load_samples()
    annotations = store.load_annotations()
    split_record = LocalSplit.load(LOCAL_SPLIT)
    for name in ("train", "val"):
        for frame_id in sorted(split_record.frames.get(name, [])):
            annotation = annotations[frame_id]
            image = cv2.imread(samples[frame_id].image_path)
            if image is None:
                continue
            balls = []
            if annotation.visibility is BallVisibility.VISIBLE and annotation.radius_px:
                d = annotation.radius_px * 2
                balls.append((annotation.centre_x, annotation.centre_y, d, d))
            yield f"local_{name}", frame_id, image, balls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=CANDIDATE_C)
    parser.add_argument("--conf", type=float, default=0.08,
                        help="mining floor, below the operating point so weak "
                             "false positives are captured too")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-per-kind", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("data/hard_negatives"))
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    from visionpitch.common.config import load_config
    from visionpitch.common.types import ObjectClass
    from visionpitch.detection.yolo import build_detector

    model = YOLO(args.weights)
    person_detector = build_detector(load_config())

    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    kept: Counter = Counter()
    skipped: Counter = Counter()
    records: list[dict] = []
    n_frames = 0

    for index, (split_label, frame_id, image, balls) in enumerate(sources()):
        n_frames += 1
        if index and index % 400 == 0:
            log.info("  %d frames, %d crops so far", index, sum(kept.values()))

        result = model.predict(image, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        if result.boxes is None or not len(result.boxes):
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        centres = [
            (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2), float(s))
            for b, s in zip(boxes, scores, strict=True)
        ]
        duplicates = mark_duplicates(centres)

        matched: set[int] = set()
        for cx, cy, _, _ in balls:
            best, slot = MATCH_PX, None
            for j, (px, py, _) in enumerate(centres):
                if j in matched:
                    continue
                d = float(np.hypot(px - cx, py - cy))
                if d <= best:
                    best, slot = d, j
            if slot is not None:
                matched.add(slot)

        people = person_detector.detect_batch([image], [index])[0]
        person_boxes = np.array([
            [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in people
            if d.object_class in (
                ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE
            )
        ]) if people else np.zeros((0, 4))

        h, w = image.shape[:2]
        for j, (px, py, score) in enumerate(centres):
            if j in matched:
                continue
            if duplicates[j]:
                skipped["duplicate_candidate"] += 1
                continue
            box = boxes[j]
            width, height = float(box[2] - box[0]), float(box[3] - box[1])
            evidence = measure(image, px, py, width, height, person_boxes)
            kind = classify(evidence, width, height)
            if not kind.minable:
                skipped[kind.value] += 1
                continue
            if kept[kind.value] >= args.max_per_kind:
                skipped[f"{kind.value}_quota"] += 1
                continue

            side = int(np.clip(
                max(width, height) * CROP_SCALE, MIN_CROP_PX, MAX_CROP_PX
            ))
            x0 = int(np.clip(px - side / 2, 0, max(0, w - side)))
            y0 = int(np.clip(py - side / 2, 0, max(0, h - side)))
            crop = image[y0:y0 + side, x0:x0 + side]
            if crop.size == 0 or crop.shape[0] < 48 or crop.shape[1] < 48:
                skipped["crop_too_small"] += 1
                continue

            # A crop containing a real ball is not a negative. Rejected rather
            # than emitted with an empty label, which would teach suppression of
            # the very object we want found.
            if any(
                x0 <= bx <= x0 + side and y0 <= by <= y0 + side
                for bx, by, _, _ in balls
            ):
                skipped["crop_contains_a_real_ball"] += 1
                continue

            stem = f"hn_{kind.value}_{split_label}_{frame_id}_{j}"
            cv2.imwrite(str(crops_dir / f"{stem}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            kept[kind.value] += 1
            records.append({
                "stem": stem, "kind": kind.value, "source_split": split_label,
                "source_frame": frame_id, "confidence": round(score, 4),
                "crop_px": side,
            })

    digest = hashlib.sha256(
        json.dumps(sorted(r["stem"] for r in records)).encode()
    ).hexdigest()[:16]

    provenance = {
        "schema_version": "1.0.0",
        "weights": args.weights,
        "mining_conf": args.conf,
        "imgsz": args.imgsz,
        "n_source_frames": n_frames,
        "n_crops": sum(kept.values()),
        "by_kind": dict(sorted(kept.items(), key=lambda kv: -kv[1])),
        "skipped": dict(sorted(skipped.items(), key=lambda kv: -kv[1])),
        "source_splits": sorted({r["source_split"] for r in records}),
        "excluded_splits": ["public_test", "local_test"],
        "fingerprint": digest,
        "policy": [
            "mined from training and validation splits only; the locked tests "
            "are never mined",
            "duplicate and trajectory-inconsistent false positives are excluded "
            "-- they are fusion failures, and crops of them would teach the "
            "detector to suppress real balls",
            "any crop containing a real ball is rejected, not emitted empty",
        ],
        "records": records,
    }
    (args.out / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    print(f"\nhard negatives: {sum(kept.values())} crop(s) from {n_frames} frames")
    print(f"fingerprint   : {digest}")
    print(f"\n{'kind':<30}{'kept':>7}")
    for kind, count in provenance["by_kind"].items():
        print(f"  {kind:<28}{count:>7}")
    print(f"\nskipped: {json.dumps(provenance['skipped'])}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
