"""Fine-tune the ball specialist on the clean SN-GSR *train* dataset.

Phase 2E, CASE D.

Why a separate entry point rather than ``finetune_ball.py train``
----------------------------------------------------------------
``finetune_ball.py`` routes through ``TrainingDataPolicy``, which was written for
the four-class person dataset. It requires the canonical
``{player, goalkeeper, referee, ball}`` class mapping and a manifest whose
``source_splits`` are exactly ``{"train", "valid"}`` -- that is, it *requires the
official VALID split to be part of training*. For a one-class ball dataset whose
whole purpose is to keep VALID untouched so it can still be used for selection,
satisfying that policy would mean doing the very thing this run must not do.

No existing ball dataset carries such a manifest either, so that path is already
closed for every ball checkpoint on disk; this is a pre-existing mismatch, not
one introduced here.

So the guard is not bypassed -- it is replaced with a stricter one for this
dataset shape. This script asserts, against the published manifest and against
the exported filenames themselves, that:

* every source sequence is in the canonical **train** split
* zero validation, test or challenge sequences are present
* none of the eight sequences used by every Phase-2E measurement is present

and it refuses to run otherwise. Validation during training uses sequences held
out from the *train* split, never the official VALID split.

Experimental discipline
-----------------------
The augmentation profile is copied unchanged from the recipe that produced the
existing checkpoints. The only variable that moves is the training data, so a
difference in the result is attributable to the corpus change and not to a
simultaneous augmentation tweak. (The measured failure distribution says
occlusion dominates, which argues for raising ``erasing`` -- that is a separate
experiment, deliberately not folded into this one.)

Usage::

    python scripts/train_ball_gsrtrain.py --epochs 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("ball.train.gsrtrain")

SPLIT_MANIFEST = Path("data/eval/gsr/sequences_info.json")
EVAL_SEQUENCES = {f"SNGS-{i:03d}" for i in range(21, 29)}
SEQUENCE_PATTERN = re.compile(r"(SNGS-\d{3})")


def guard(dataset: Path) -> dict:
    """Prove the dataset is train-only before a single gradient step is taken."""
    provenance_path = dataset / "provenance.json"
    if not provenance_path.is_file():
        raise SystemExit(f"ABORT - no provenance at {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    canonical = {
        key: {s["name"] for s in value}
        for key, value in manifest.items()
        if isinstance(value, list)
    }

    # Trust the exported filenames, not the provenance file's own summary: the
    # filenames are what the trainer will actually read.
    found: set[str] = set()
    for split in ("train", "val"):
        images = dataset / split / "images"
        if not images.is_dir():
            raise SystemExit(f"ABORT - missing split directory {images}")
        for image in images.iterdir():
            match = SEQUENCE_PATTERN.search(image.name)
            if match:
                found.add(match.group(1))

    forbidden = sorted(
        found & (
            canonical.get("validation", set())
            | canonical.get("test", set())
            | canonical.get("challenge", set())
        )
    )
    if forbidden:
        raise SystemExit(f"ABORT - non-train sequences in dataset: {forbidden[:10]}")
    overlap = sorted(found & EVAL_SEQUENCES)
    if overlap:
        raise SystemExit(f"ABORT - evaluation sequences in dataset: {overlap}")
    outside = sorted(s for s in found if s not in canonical.get("train", set()))
    if outside:
        raise SystemExit(f"ABORT - sequences outside canonical train: {outside[:10]}")

    log.info(
        "leakage guard passed: %d SN-GSR sequences, all canonical train; "
        "0 validation, 0 test, 0 challenge, 0 evaluation",
        len(found),
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/ball_gsrtrain_v1"))
    parser.add_argument("--weights", default="models/yolo-football-ball-detection.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 on Windows: this machine has a known "
                             "multiprocessing dataloader failure")
    parser.add_argument("--name", default="ball_gsrtrain_v1")
    args = parser.parse_args()

    configure_logging("INFO")

    provenance = guard(args.dataset)
    log.info(
        "dataset: %d train / %d val images, GSR stride %s, domain share %s",
        sum(provenance["counts"]["train"].values()),
        sum(provenance["counts"]["val"].values()),
        provenance["gsr_stride"], provenance["max_domain_share"],
    )

    data_yaml = args.dataset / "data.yaml"
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if config.get("names") != ["ball"]:
        raise SystemExit(f"ABORT - expected a one-class ball dataset: {config}")

    destination = Path("models/finetune") / args.name
    if (destination / "weights" / "best.pt").exists():
        raise SystemExit(
            f"ABORT - {destination} already holds weights; refusing to overwrite. "
            "Choose a new --name."
        )

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(Path("models/finetune").resolve()),
        name=args.name,
        exist_ok=False,
        patience=args.patience,
        workers=args.workers,
        # Copied verbatim from the existing recipe so the corpus is the only
        # variable that moves between this run and the checkpoints it is
        # compared against.
        scale=0.6,
        translate=0.15,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.8,
        mixup=0.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        erasing=0.2,
        verbose=True,
    )

    best = destination / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    return 0 if best.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
