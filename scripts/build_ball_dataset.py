"""Build a versioned multi-corpus ball dataset with clip-disjoint splits.

Phase 2C, Parts 1 and 2.

Two decisions that change the numbers, both forced by the audit:

**Clip-disjoint splitting.** The Roboflow corpus ships a random *frame* split in
which all 14 test clips also appear in training. Using it measures memorisation.
Every corpus here is re-split by source clip, so a frame of a test match can
never be seen in training.

**Domain balancing.** SoccerNet-GSR contributes an order of magnitude more
frames than Roboflow. Left unbalanced, training becomes a GSR-only run wearing a
multi-corpus label -- which is exactly what Phase 2B showed does not transfer.
Frames are sampled so no domain exceeds ``--max-domain-share`` of the training
set.

Usage::

    python scripts/build_ball_dataset.py --out data/ball_multicorpus
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.common.types import ObjectClass  # noqa: E402
from visionpitch.evaluation.datasets import GSRDataset  # noqa: E402
from visionpitch.evaluation.registry import (  # noqa: E402
    assert_no_leakage,
    build_split,
    registry_document,
    roboflow_clip_ids,
)

log = get_logger("ball.dataset")

ROBOFLOW_ROOTS = [Path("data/eval/ball_det"), Path("data/eval/player_det")]
GSR_ROOT = Path("data/eval/gsr")


def _label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def _ball_lines(image: Path, ball_class_ids: set[int]) -> list[str] | None:
    """YOLO lines for the ball class only, renumbered to class 0.

    Returns ``None`` when the label file is missing entirely -- an unlabelled
    image is not a negative, it is unknown, and treating it as "no ball here"
    teaches the model to suppress real balls.
    """
    label = _label_path(image)
    if not label.exists():
        return None
    out = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if int(float(parts[0])) in ball_class_ids:
            out.append("0 " + " ".join(parts[1:5]))
    return out


def collect_roboflow(split_map, ball_class_ids_by_root) -> dict[str, list[tuple]]:
    """(image, lines) per split, from every Roboflow root, by source clip."""
    buckets: dict[str, list[tuple]] = defaultdict(list)
    for root in ROBOFLOW_ROOTS:
        ball_ids = ball_class_ids_by_root[root]
        for clip, images in roboflow_clip_ids(root).items():
            split = split_map.split_of("roboflow", clip)
            if split == "unassigned":
                continue
            for image in images:
                lines = _ball_lines(image, ball_ids)
                if lines is None:
                    continue
                buckets[split].append((image, lines, "roboflow", clip))
    return buckets


def collect_gsr(split_map, stride: int) -> dict[str, list[tuple]]:
    dataset = GSRDataset(GSR_ROOT, max_sequences=None)
    buckets: dict[str, list[tuple]] = defaultdict(list)
    for sequence in dataset.sequences:
        split = split_map.split_of("soccernet_gsr", sequence.name)
        if split == "unassigned":
            continue
        step = stride if split == "train" else stride * 2
        for frame_idx in sorted(sequence.image_paths)[::step]:
            objects = sequence.ground_truth.frames.get(frame_idx, [])
            balls = [o.bbox for o in objects if o.object_class is ObjectClass.BALL]
            buckets[split].append(
                (sequence.image_paths[frame_idx], balls, "soccernet_gsr", sequence.name)
            )
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ball_multicorpus"))
    parser.add_argument("--gsr-stride", type=int, default=6)
    parser.add_argument("--max-domain-share", type=float, default=0.62,
                        help="cap on any one domain's share of the training set")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")
    rng = random.Random(args.seed)

    # -- class ids ---------------------------------------------------------- #
    import yaml

    ball_ids: dict[Path, set[int]] = {}
    for root in ROBOFLOW_ROOTS:
        data_yaml = root / "data" / "data.yaml"
        names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
        names = names if isinstance(names, list) else [names[i] for i in sorted(names)]
        ball_ids[root] = {
            i for i, n in enumerate(names) if str(n).strip().lower() in ("ball", "football")
        }

    # -- clip-disjoint split ------------------------------------------------ #
    gsr = GSRDataset(GSR_ROOT, max_sequences=None)
    split_map = build_split(ROBOFLOW_ROOTS, [s.name for s in gsr.sequences])
    assert_no_leakage(split_map)

    roboflow = collect_roboflow(split_map, ball_ids)
    soccernet = collect_gsr(split_map, args.gsr_stride)

    # -- domain balancing --------------------------------------------------- #
    # Applied to training only. Validation and test keep every frame, because
    # capping them would change what the metric is measuring.
    train = list(roboflow["train"]) + list(soccernet["train"])
    counts = Counter(item[2] for item in train)
    if counts:
        smallest = min(counts.values())
        cap = int(smallest / max(1e-9, 1 - args.max_domain_share) * args.max_domain_share)
        balanced: list[tuple] = []
        per_domain: Counter = Counter()
        rng.shuffle(train)
        for item in train:
            if per_domain[item[2]] >= cap:
                continue
            balanced.append(item)
            per_domain[item[2]] += 1
        log.info("domain balance: %s -> %s (cap %d)", dict(counts), dict(per_domain), cap)
        train = balanced

    buckets = {
        "train": train,
        "val": list(roboflow["val"]) + list(soccernet["val"]),
        "test": list(roboflow["test"]) + list(soccernet["test"]),
    }

    # -- write --------------------------------------------------------------- #
    if args.out.exists() and args.clean:
        shutil.rmtree(args.out)
    for part in ("train", "val", "test"):
        (args.out / part / "images").mkdir(parents=True, exist_ok=True)
        (args.out / part / "labels").mkdir(parents=True, exist_ok=True)

    stats: dict[str, Counter] = {k: Counter() for k in buckets}
    instances: dict[str, int] = dict.fromkeys(buckets, 0)

    for part, items in buckets.items():
        for source, payload, domain, clip in items:
            stem = f"{domain}_{clip}_{source.stem}"[:120]
            target = args.out / part / "images" / f"{stem}.jpg"

            if domain == "roboflow":
                if not target.exists():
                    shutil.copyfile(source, target)
                lines = payload
            else:
                image = cv2.imread(str(source))
                if image is None:
                    continue
                h, w = image.shape[:2]
                if not target.exists():
                    cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                lines = []
                for box in payload:
                    cx = (box.x1 + box.x2) / 2 / w
                    cy = (box.y1 + box.y2) / 2 / h
                    bw, bh = box.width / w, box.height / h
                    if bw > 0 and bh > 0:
                        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (args.out / part / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            stats[part][domain] += 1
            instances[part] += len(lines)

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        f"train: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: 1\nnames: ['ball']\n",
        encoding="utf-8",
    )
    split_map.save(args.out / "split.json")

    provenance = {
        "registry": registry_document(),
        "split_fingerprint": split_map.fingerprint(),
        "split_unit": "source clip (Roboflow) / sequence (GSR) -- never frame",
        "discarded_official_split": (
            "The Roboflow published train/test split shares all 14 test clips with "
            "train and is not used."
        ),
        "max_domain_share": args.max_domain_share,
        "gsr_stride": args.gsr_stride,
        "frames_per_split_per_domain": {k: dict(v) for k, v in stats.items()},
        "ball_instances": instances,
    }
    (args.out / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    print(json.dumps(provenance["frames_per_split_per_domain"], indent=2))
    print(f"ball instances: {instances}")
    print(f"split fingerprint: {split_map.fingerprint()}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
