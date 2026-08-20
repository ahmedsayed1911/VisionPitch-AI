"""Build a ball dataset from the SN-GSR *train* split, with clean provenance.

Phase 2E, CASE D. Two problems are fixed at once.

**Contamination.** Every fine-tuned ball checkpoint on disk -- ``multicorpus``,
``C_adapt``, ``C_hardened``, ``D_adapt_aug``, ``B_public`` and ``ball_gsr`` --
was trained on SN-GSR frames drawn from ``data/eval/gsr``, which the canonical
manifest labels **test** (SNGS-116..200). None of them ever saw the validation
sequences, so the VALID measurements are sound, but the SN-GSR test split is
burned for the entire ball subsystem: no honest held-out number can ever be
produced for those checkpoints. This dataset uses ``data/SoccerNetGS/train``
only, so a model trained on it can still be evaluated on test one day.

**Accuracy.** The pure-GSR checkpoint reaches 0.5616 pooled possession team F1
on VALID against 0.4770 for the best multi-domain checkpoint -- direct evidence
that in-domain SN-GSR data is worth roughly +0.085 team F1 on this corpus. That
checkpoint is unusable (it is the test-trained one), but the *data* that made it
good has a legitimate counterpart in the train split, which is larger: 57
sequences against the 49 in the test root.

Domain balance is kept for the reason the original multi-corpus build documented:
SN-GSR contributes an order of magnitude more frames than the other corpora, and
an unbalanced mix becomes a GSR-only run wearing a multi-corpus label. The
roboflow and local-broadcast frames are reused from the existing extracted
datasets, which contain no SN-GSR frames at all.

Leakage guard
-------------
Aborts if any source sequence is in the manifest's validation, test or challenge
lists. The eight sequences used by every Phase-2E measurement are checked by
name as well, so a manifest edit cannot silently admit them.

Usage::

    python scripts/build_ball_gsrtrain_dataset.py --out data/ball_gsrtrain_v1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("ball.gsrtrain")

GSR_TRAIN_ROOT = Path("data/SoccerNetGS/train")
SPLIT_MANIFEST = Path("data/eval/gsr/sequences_info.json")
BALL_CATEGORY_ID = 4

#: Already-extracted frames with no SN-GSR content, reused rather than rebuilt.
REUSE_SOURCES = {
    "roboflow": [Path("data/ball_multicorpus")],
    "local": [Path("data/ball_broadcast_adapt")],
}

#: The eight sequences every Phase-2E measurement scores. Named explicitly so a
#: manifest edit cannot quietly let them into training.
EVAL_SEQUENCES = {f"SNGS-{i:03d}" for i in range(21, 29)}


def manifest_splits() -> dict[str, set[str]]:
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    return {
        key: {s["name"] for s in value}
        for key, value in manifest.items()
        if isinstance(value, list)
    }


def gsr_frames(sequence_dir: Path, stride: int) -> list[tuple[Path, str]]:
    """(image path, YOLO label text) for frames of this sequence with a ball."""
    labels_path = sequence_dir / "Labels-GameState.json"
    if not labels_path.exists():
        return []
    data = json.loads(labels_path.read_text(encoding="utf-8"))

    images = {}
    for image in data.get("images", []):
        images[str(image["image_id"])] = (
            image["file_name"], float(image["width"]), float(image["height"])
        )

    rows: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for index, annotation in enumerate(data.get("annotations", [])):
        if annotation.get("category_id") != BALL_CATEGORY_ID:
            continue
        image_id = str(annotation.get("image_id"))
        if image_id not in images or image_id in seen:
            continue
        box = annotation.get("bbox_image") or {}
        if not box or not box.get("w") or not box.get("h"):
            continue
        file_name, width, height = images[image_id]
        image_path = sequence_dir / "img1" / file_name
        if not image_path.exists():
            continue
        seen.add(image_id)
        if len(seen) % stride:
            continue
        cx = float(box["x_center"]) / width
        cy = float(box["y_center"]) / height
        bw = float(box["w"]) / width
        bh = float(box["h"]) / height
        if not (0 < cx < 1 and 0 < cy < 1 and 0 < bw < 1 and 0 < bh < 1):
            continue
        rows.append((image_path, f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"))
        _ = index
    return rows


def reuse_frames(prefix: str, roots: list[Path]) -> list[tuple[Path, Path]]:
    """(image, label) pairs for already-extracted non-GSR frames."""
    pairs: list[tuple[Path, Path]] = []
    for root in roots:
        for split in ("train", "val", "test"):
            images = root / split / "images"
            labels = root / split / "labels"
            if not images.is_dir():
                continue
            for image in sorted(images.glob(f"{prefix}*")):
                label = labels / (image.stem + ".txt")
                if label.exists():
                    pairs.append((image, label))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ball_gsrtrain_v1"))
    parser.add_argument("--gsr-stride", type=int, default=4)
    parser.add_argument("--val-sequences", type=int, default=9)
    parser.add_argument("--max-domain-share", type=float, default=0.62)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    configure_logging("INFO")
    random.seed(args.seed)

    # -- leakage guard, before any file is written ---------------------------- #
    splits = manifest_splits()
    sequences = sorted(
        p.name for p in GSR_TRAIN_ROOT.iterdir()
        if p.is_dir() and (p / "Labels-GameState.json").exists()
    )
    if not sequences:
        raise SystemExit(f"no sequences under {GSR_TRAIN_ROOT}")

    forbidden = sorted(
        s for s in sequences
        if s in splits.get("validation", set()) | splits.get("test", set())
        | splits.get("challenge", set())
    )
    if forbidden:
        raise SystemExit(f"ABORT - non-train sequences present: {forbidden[:8]}")
    overlap = sorted(set(sequences) & EVAL_SEQUENCES)
    if overlap:
        raise SystemExit(f"ABORT - evaluation sequences present: {overlap}")
    outside = sorted(s for s in sequences if s not in splits.get("train", set()))
    if outside:
        raise SystemExit(f"ABORT - sequences not in canonical train: {outside[:8]}")
    log.info("leakage guard passed: %d canonical train sequences", len(sequences))

    # -- clip-disjoint split over sequences ---------------------------------- #
    shuffled = sequences[:]
    random.shuffle(shuffled)
    val_sequences = set(shuffled[: args.val_sequences])
    train_sequences = [s for s in sequences if s not in val_sequences]
    log.info("train %d sequences, val %d sequences",
             len(train_sequences), len(val_sequences))

    gsr: dict[str, list[tuple[Path, str]]] = {"train": [], "val": []}
    for sequence in sequences:
        split = "val" if sequence in val_sequences else "train"
        rows = gsr_frames(GSR_TRAIN_ROOT / sequence, args.gsr_stride)
        gsr[split].extend(rows)
        log.info("  %s (%s): %d ball frames", sequence, split, len(rows))

    # -- reused non-GSR frames ------------------------------------------------ #
    reused: dict[str, list[tuple[Path, Path]]] = {}
    for domain, roots in REUSE_SOURCES.items():
        reused[domain] = reuse_frames(domain, roots)
        log.info("reused %s: %d frames", domain, len(reused[domain]))

    other_train = sum(len(v) for v in reused.values())
    # Domain balance: cap GSR so it cannot exceed max_domain_share of train.
    if other_train:
        cap = int(args.max_domain_share / (1 - args.max_domain_share) * other_train)
        if len(gsr["train"]) > cap:
            random.shuffle(gsr["train"])
            log.info("capping GSR train frames %d -> %d for domain balance",
                     len(gsr["train"]), cap)
            gsr["train"] = gsr["train"][:cap]

    # -- write ---------------------------------------------------------------- #
    if args.out.exists():
        shutil.rmtree(args.out)
    counts: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    for split in ("train", "val"):
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)

    for split, rows in gsr.items():
        for image_path, label_text in rows:
            sequence = image_path.parent.parent.name
            stem = f"soccernet_gsrtrain_{sequence}_{image_path.stem}"
            shutil.copy2(image_path, args.out / split / "images" / f"{stem}.jpg")
            (args.out / split / "labels" / f"{stem}.txt").write_text(
                label_text, encoding="utf-8"
            )
            counts[split]["gsr_train"] += 1

    for domain, pairs in reused.items():
        random.shuffle(pairs)
        cut = max(1, int(0.85 * len(pairs))) if pairs else 0
        for index, (image, label) in enumerate(pairs):
            split = "train" if index < cut else "val"
            stem = image.stem
            shutil.copy2(image, args.out / split / "images" / f"{stem}.jpg")
            shutil.copy2(label, args.out / split / "labels" / f"{stem}.txt")
            counts[split][domain] += 1

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        "train: train/images\nval: val/images\n\nnc: 1\nnames: ['ball']\n",
        encoding="utf-8",
    )

    provenance = {
        "schema_version": "1.0.0",
        "built_by": "scripts/build_ball_gsrtrain_dataset.py",
        "gsr_root": str(GSR_TRAIN_ROOT),
        "gsr_split": "canonical train only",
        "leakage_guard": {
            "validation_sequences_used": 0,
            "test_sequences_used": 0,
            "challenge_sequences_used": 0,
            "evaluation_sequences_used": 0,
            "asserted_against": str(SPLIT_MANIFEST),
        },
        "gsr_train_sequences": sorted(train_sequences),
        "gsr_val_sequences": sorted(val_sequences),
        "gsr_stride": args.gsr_stride,
        "max_domain_share": args.max_domain_share,
        "seed": args.seed,
        "counts": {k: dict(v) for k, v in counts.items()},
        "reused_from": {k: [str(p) for p in v] for k, v in REUSE_SOURCES.items()},
        "why": (
            "every existing ball checkpoint was fine-tuned on SN-GSR test-split "
            "frames; this dataset uses the train split only so a model built on "
            "it keeps the test split available for honest held-out evaluation"
        ),
    }
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    for split in ("train", "val"):
        log.info("%s: %d images %s", split, sum(counts[split].values()),
                 dict(counts[split]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
