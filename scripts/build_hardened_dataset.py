"""Assemble Candidate C's training set plus mined hard negatives.

Part 5 of precision hardening.

The recipe is deliberately conservative: Candidate C's exact training data, plus
the mined hard-negative crops as empty-label images. **No new augmentation.**
Candidate D already measured what heavy augmentation does here -- it lost recall
on both domains and raised false positives -- so repeating it would be
re-running a failed experiment.

Hard-negative crops carry an empty label file, which is how YOLO encodes "this
image contains no object of the class". They are the false positives the
detector actually produced, so this is targeted rather than generic negative
mining.

Usage::

    python scripts/build_hardened_dataset.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("hardening.dataset")

SOURCE = Path("data/ball_broadcast_adapt")
HARD_NEGATIVES = Path("data/hard_negatives")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ball_hardened"))
    args = parser.parse_args()
    configure_logging("INFO")

    if not SOURCE.exists():
        log.error("Candidate C's dataset is missing at %s", SOURCE)
        return 1
    provenance_path = HARD_NEGATIVES / "PROVENANCE.json"
    if not provenance_path.exists():
        log.error("no mined hard negatives at %s", HARD_NEGATIVES)
        return 1
    mined = json.loads(provenance_path.read_text(encoding="utf-8"))

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {"copied_train": 0, "copied_val": 0, "hard_negatives": 0}
    for split, key in (("train", "copied_train"), ("val", "copied_val")):
        for image in sorted((SOURCE / split / "images").glob("*.jpg")):
            shutil.copyfile(image, out / split / "images" / image.name)
            label = SOURCE / split / "labels" / f"{image.stem}.txt"
            target = out / split / "labels" / f"{image.stem}.txt"
            if label.exists():
                shutil.copyfile(label, target)
            else:
                target.write_text("", encoding="utf-8")
            counts[key] += 1

    # Hard negatives go to TRAIN only. A validation set stuffed with mined
    # negatives would make every threshold look good on data chosen to be hard
    # for this exact model.
    for record in mined["records"]:
        source = HARD_NEGATIVES / "crops" / f"{record['stem']}.jpg"
        if not source.exists():
            continue
        shutil.copyfile(source, out / "train" / "images" / source.name)
        (out / "train" / "labels" / f"{source.stem}.txt").write_text(
            "", encoding="utf-8"
        )
        counts["hard_negatives"] += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: train/images\nval: val/images\n\nnc: 1\nnames: ['ball']\n",
        encoding="utf-8",
    )

    n_train = len(list((out / "train" / "images").glob("*.jpg")))
    n_val = len(list((out / "val" / "images").glob("*.jpg")))
    (out / "PROVENANCE.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "base_dataset": str(SOURCE),
        "hard_negative_fingerprint": mined["fingerprint"],
        "hard_negative_kinds": mined["by_kind"],
        "counts": counts,
        "n_train_images": n_train,
        "n_val_images": n_val,
        "augmentation": (
            "none beyond the training recipe already validated for Candidate C; "
            "Candidate D's heavy offline augmentation is deliberately not repeated"
        ),
        "hard_negatives_placed_in": "train only",
        "excluded": ["public_test", "local_test"],
    }, indent=2), encoding="utf-8")

    print(f"\nout   : {out}")
    print(f"train : {n_train} ({counts['copied_train']} base + "
          f"{counts['hard_negatives']} hard negatives)")
    print(f"val   : {n_val}")
    print(f"hard-negative fingerprint: {mined['fingerprint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
