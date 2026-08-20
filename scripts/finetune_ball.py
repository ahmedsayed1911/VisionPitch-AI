"""Fine-tune the dedicated ball detector on out-of-distribution footage.

Phase 2B, Part 6. Justified by the Part 5 error analysis, which found the
dominant failure is **genuine detector blindness** rather than localisation
error or threshold calibration: the distance from a ground-truth ball to the
nearest prediction is bimodal, with 48.3% of frames within 12 px and only 49.3%
within 60 px. When the model finds the ball it localises it well; the problem is
that half the time it does not find it at all.

Method
------
Export ball crops from SoccerNet SN-GSR in YOLO format, split **by sequence**
so no frame of a test clip can appear in training, and fine-tune the shipped
checkpoint. Augmentation is aimed at the measured failure modes rather than
applied indiscriminately: scale (tiny balls), motion blur, and compression.

The held-out test split is never used for training or for choosing a threshold.

Usage::

    python scripts/finetune_ball.py export
    python scripts/finetune_ball.py train --epochs 30
    python scripts/finetune_ball.py evaluate
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.common.types import ObjectClass  # noqa: E402
from visionpitch.evaluation.datasets import GSRDataset  # noqa: E402
from visionpitch.evaluation.splits import assert_disjoint, make_split  # noqa: E402
from visionpitch.training.data_policy import (  # noqa: E402
    DataBoundaryError,
    TrainingDataPolicy,
    find_project_root,
)

log = get_logger("ball.finetune")

DATA_ROOT = Path("data/ball_finetune")
SPLIT_PATH = DATA_ROOT / "split.json"


def export(args) -> int:
    """Write a YOLO-format ball dataset with a leak-free sequence split."""
    raise DataBoundaryError(
        "Legacy ball export is disabled: its default source is LEGACY_CONTAMINATED_TEST. "
        "Use scripts/prepare_yolo_dataset.py with official TRAIN/VALID instead."
    )
    dataset = GSRDataset(args.root, max_sequences=None)
    names = [s.name for s in dataset.sequences]
    split = make_split(names, name="sn_gsr_ball")
    assert_disjoint(split)

    if DATA_ROOT.exists() and args.clean:
        shutil.rmtree(DATA_ROOT)
    for part in ("train", "val", "test"):
        (DATA_ROOT / part / "images").mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / part / "labels").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    empty = {"train": 0, "val": 0, "test": 0}

    for sequence in dataset.sequences:
        part = split.of(sequence.name)
        if part == "unassigned":
            continue
        frames = sorted(sequence.image_paths)
        stride = args.stride if part == "train" else max(1, args.stride * 2)
        for frame_idx in frames[:: stride]:
            objects = sequence.ground_truth.frames.get(frame_idx, [])
            balls = [o.bbox for o in objects if o.object_class is ObjectClass.BALL]

            source = sequence.image_paths[frame_idx]
            image = cv2.imread(str(source))
            if image is None:
                continue
            h, w = image.shape[:2]

            # Frames with no ball are kept, at a reduced rate, as negatives.
            # Training only on frames containing a ball teaches the model that
            # something ball-like is always present, which is exactly the
            # failure the error analysis found: confident detections on other
            # objects.
            if not balls:
                if empty[part] > counts[part] * args.negative_ratio:
                    continue
                empty[part] += 1

            stem = f"{sequence.name}_{frame_idx:06d}"
            cv2.imwrite(str(DATA_ROOT / part / "images" / f"{stem}.jpg"), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            lines = []
            for box in balls:
                cx = (box.x1 + box.x2) / 2 / w
                cy = (box.y1 + box.y2) / 2 / h
                bw = box.width / w
                bh = box.height / h
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (DATA_ROOT / part / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            counts[part] += 1

        log.info("%s -> %s (%d frames so far)", sequence.name, part, counts[part])

    yaml_text = (
        f"path: {DATA_ROOT.resolve().as_posix()}\n"
        f"train: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: 1\nnames: ['ball']\n"
    )
    (DATA_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    split.save(SPLIT_PATH)

    provenance = {
        "source": "SoccerNet/SN-GSR-2025 (test.zip)",
        "licence": "SoccerNet terms; non-commercial research",
        "split_fingerprint": split.fingerprint(),
        "split_unit": "sequence (never frame)",
        "frames": counts,
        "empty_frames_kept": empty,
        "stride_train": args.stride,
        "negative_ratio": args.negative_ratio,
    }
    (DATA_ROOT / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    print(json.dumps(provenance, indent=2))
    print(f"\nsplit: {len(split.train)} train / {len(split.val)} val / "
          f"{len(split.test)} test sequences")
    print(f"held-out test sequences: {split.test}")
    return 0


def train(args) -> int:
    """Fine-tune the shipped ball checkpoint on a ball dataset.

    ``--data`` points at any dataset with this script's layout; the Phase 2C
    multi-corpus set built by ``build_ball_dataset.py`` is one such. Without it
    the single-corpus GSR export is used, which is what Phase 2B trained on and
    what Phase 2B showed does not transfer.
    """
    from ultralytics import YOLO

    data_yaml = Path(args.data) if args.data else DATA_ROOT / "data.yaml"
    project_root = find_project_root(Path(__file__).parent)
    try:
        TrainingDataPolicy(project_root).validate_dataset_yaml(data_yaml)
    except (DataBoundaryError, OSError, ValueError) as exc:
        print(f"TRAINING DATA BOUNDARY ABORT: {exc}")
        return 2
    if not data_yaml.exists():
        print(f"no dataset at {data_yaml}; run `export` or build_ball_dataset.py first")
        return 1

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        # Windows spawns dataloader workers rather than forking; this machine
        # has a known failure there, so the count is explicit and overridable.
        workers=args.workers,
        project=str(Path("models/finetune").resolve()),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        # Augmentation aimed at the measured failure distribution rather than
        # applied blindly. Scale dominates: the tiny-ball bucket scored 0.00.
        scale=0.6,
        translate=0.15,
        degrees=0.0,      # broadcast cameras are level; rotation is off-domain
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.8,       # forces multi-scale context and small-object crops
        mixup=0.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,   # kit and floodlight variation
        erasing=0.2,                          # partial occlusion by players
        verbose=True,
    )
    best = Path("models/finetune") / args.name / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    return 0 if best.exists() else 1


def evaluate(args) -> int:
    """Score baseline and fine-tuned checkpoints on the held-out test split."""
    from ultralytics import YOLO

    if not SPLIT_PATH.exists():
        print("run `export` first")
        return 1

    results = {}
    for label, weights in (("baseline", args.weights), ("finetuned", args.finetuned)):
        if not Path(weights).exists():
            print(f"  {label}: {weights} not found, skipped")
            continue
        model = YOLO(weights)
        metrics = model.val(
            data=str((DATA_ROOT / "data.yaml").resolve()),
            split="test", imgsz=args.imgsz, conf=0.001, iou=0.5,
            device=0, verbose=False,
        )
        box = metrics.box
        results[label] = {
            "weights": str(weights),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "f1": round(
                float(2 * box.mp * box.mr / max(1e-9, box.mp + box.mr)), 4
            ),
        }
        print(f"  {label:10s} P {results[label]['precision']}  R {results[label]['recall']}  "
              f"mAP50 {results[label]['mAP50']}  mAP50-95 {results[label]['mAP50_95']}")

    out = Path("data/eval/gsr/benchmarks/ball_finetune.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    out.write_text(json.dumps({
        "split_fingerprint": split["fingerprint"],
        "test_sequences": [k for k, v in split["assignments"].items() if v == "test"],
        "results": results,
        "note": "held-out test split; never used for training or threshold selection",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export")
    p.add_argument("--root", type=Path, default=Path("data/eval/gsr"))
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--negative-ratio", type=float, default=0.15)
    p.add_argument("--clean", action="store_true")
    p.set_defaults(func=export)

    p = sub.add_parser("train")
    p.add_argument("--weights", default="models/yolo-football-ball-detection.pt")
    p.add_argument(
        "--data", default=None,
        help="dataset data.yaml; defaults to the single-corpus GSR export",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default="ball_gsr")
    p.set_defaults(func=train)

    p = sub.add_parser("evaluate")
    p.add_argument("--weights", default="models/yolo-football-ball-detection.pt")
    p.add_argument("--finetuned", default="models/finetune/ball_gsr/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.set_defaults(func=evaluate)

    args = parser.parse_args()
    configure_logging("INFO")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
