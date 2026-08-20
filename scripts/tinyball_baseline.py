"""Lock the box-detector baseline under the fixed tiny-ball protocol.

Part 3.

Reports the official IoU50 figures unchanged, and adds the centre metrics that
Part 1 showed are the task-relevant ones. Both come from the same predictions in
the same pass, so no comparison between them is confounded by a different run.

Also emits the protocol record with its fingerprint, and asserts clip-disjointness
from the actual file listing rather than trusting the stored assignment.

Usage::

    python scripts/tinyball_baseline.py \
        --model models/yolo-football-ball-detection.pt --label box_baseline
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

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.tinyball import (  # noqa: E402
    CENTRE_TOLERANCES_PX,
    Partition,
    TinyBallProtocol,
    assert_clip_disjoint,
    domain_of,
    pool,
    score_centres,
)

log = get_logger("tinyball.baseline")

DATASET = Path("data/ball_multicorpus")


def partition_images(partition: Partition) -> list[Path]:
    """Images for a partition, mapped onto the frozen base split."""
    folder = {
        Partition.TRAIN: "train",
        Partition.VAL_IN_DOMAIN: "val",
        Partition.TEST: "test",
    }.get(partition)
    if folder is None:
        raise ValueError(f"{partition} is derived, not a folder")
    return sorted((DATASET / folder / "images").glob("*.jpg"))


def load_truth_centres(image_path: Path, width: int, height: int) -> list[tuple[float, float]]:
    label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label.exists():
        return []
    out = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append((float(parts[1]) * width, float(parts[2]) * height))
    return out


def load_truth_boxes(image_path: Path, width: int, height: int) -> list[np.ndarray]:
    label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label.exists():
        return []
    out = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        cx, cy, bw, bh = cx * width, cy * height, bw * width, bh * height
        out.append(np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
    return out


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def write_protocol(images: list[Path], out: Path) -> TinyBallProtocol:
    listings = {
        p.value: partition_images(p)
        for p in (Partition.TRAIN, Partition.VAL_IN_DOMAIN, Partition.TEST)
    }
    assert_clip_disjoint(listings)
    base = json.loads((DATASET / "split.json").read_text(encoding="utf-8"))
    counts = {
        name: {
            d: sum(1 for p in paths if domain_of(p) == d)
            for d in sorted({domain_of(p) for p in paths})
        }
        for name, paths in listings.items()
    }
    protocol = TinyBallProtocol(
        dataset_root=DATASET,
        base_split_fingerprint=base["fingerprint"],
        cross_domain_holdout="soccernet_gsr",
        domains=sorted({domain_of(p) for p in images}),
        counts=counts,
        augmentation={"inference_only": True},
    )
    protocol.save(out / "protocol.json")
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolo-football-ball-detection.pt")
    parser.add_argument("--label", default="box_baseline")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--partition", default="test",
        choices=[p.value for p in Partition if p is not Partition.VAL_CROSS_DOMAIN],
    )
    parser.add_argument("--out", type=Path, default=Path("data/eval/tinyball"))
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    partition = Partition(args.partition)
    images = partition_images(partition)
    if not images:
        log.error("no images for partition %s", partition.value)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    protocol = write_protocol(images, args.out)
    log.info(
        "protocol %s (base split %s); clip-disjointness verified from file listing",
        protocol.fingerprint(), protocol.base_split_fingerprint,
    )

    model = YOLO(args.model)
    per_domain_frames: dict[str, list] = defaultdict(list)
    iou_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    started = time.perf_counter()
    n_frames = 0

    for index, image_path in enumerate(images):
        if index and index % 250 == 0:
            log.info("  %d/%d", index, len(images))
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        domain = domain_of(image_path)
        n_frames += 1

        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        boxes = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None else np.zeros((0, 4))
        )
        scores = (
            result.boxes.conf.cpu().numpy()
            if result.boxes is not None else np.zeros(0)
        )
        boxes = boxes[np.argsort(-scores)]

        per_domain_frames[domain].append((
            load_truth_centres(image_path, width, height),
            [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes],
        ))

        # Official IoU50 scoring, unchanged, from the very same predictions.
        truth_boxes = load_truth_boxes(image_path, width, height)
        used: set[int] = set()
        hits = 0
        for truth in truth_boxes:
            best, best_slot = 0.5, None
            for slot, prediction in enumerate(boxes):
                if slot in used:
                    continue
                value = iou(prediction, truth)
                if value >= best:
                    best, best_slot = value, slot
            if best_slot is not None:
                used.add(best_slot)
                hits += 1
        iou_counts[domain]["tp"] += hits
        iou_counts[domain]["fn"] += len(truth_boxes) - hits
        iou_counts[domain]["fp"] += len(boxes) - hits

    elapsed = time.perf_counter() - started
    results = [
        score_centres(args.label, domain, frames)
        for domain, frames in sorted(per_domain_frames.items())
    ]

    iou_block = {}
    for domain, counts in sorted(iou_counts.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        iou_block[domain] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4)
            if precision + recall else 0.0,
        }

    payload = {
        "label": args.label,
        "representation": "bounding_box",
        "model": args.model,
        "model_fingerprint": file_fingerprint(Path(args.model)),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "partition": partition.value,
        "partition_is_tunable": partition.is_tunable,
        "protocol_fingerprint": protocol.fingerprint(),
        "base_split_fingerprint": protocol.base_split_fingerprint,
        "n_images": n_frames,
        "runtime_s": round(elapsed, 2),
        "runtime_ms_per_frame": round(1000 * elapsed / max(1, n_frames), 2),
        "iou50": iou_block,
        "iou50_macro": {
            metric: round(
                sum(v[metric] for v in iou_block.values()) / max(1, len(iou_block)), 4
            )
            for metric in ("recall", "precision", "f1")
        },
        "centre": pool(results, args.label),
        "centre_tolerances_px": list(CENTRE_TOLERANCES_PX),
    }

    destination = args.out / f"{args.label}_{partition.value}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{args.label}  partition={partition.value}  "
          f"protocol={protocol.fingerprint()}  model={payload['model_fingerprint']}")
    print(f"\nIoU50 macro: R {payload['iou50_macro']['recall']:.4f}  "
          f"P {payload['iou50_macro']['precision']:.4f}")
    for domain, block in iou_block.items():
        print(f"  {domain:<15} R {block['recall']:.4f}  P {block['precision']:.4f}")
    print("\ncentre recall by tolerance (macro / worst-domain):")
    for tolerance in CENTRE_TOLERANCES_PX:
        macro = payload["centre"]["macro_recall_at_px"][str(tolerance)]
        worst = payload["centre"]["worst_domain_recall_at_px"][str(tolerance)]
        print(f"  <= {tolerance:>5.1f} px   {macro:.4f} / {worst:.4f}")
    print(f"\nmedian centre error  : {payload['centre']['median_error_px']} px")
    print(f"macro direct coverage: {payload['centre']['macro_direct_coverage']:.4f}")
    print(f"worst-domain coverage: {payload['centre']['worst_domain_direct_coverage']:.4f}")
    print(f"false positives/frame: {payload['centre']['macro_false_positives_per_frame']:.4f}")
    print(f"runtime              : {payload['runtime_ms_per_frame']} ms/frame")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
