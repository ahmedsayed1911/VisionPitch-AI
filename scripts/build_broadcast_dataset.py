"""Assemble the training sets for the A/B/C/D broadcast-adaptation comparison.

Variants:

* ``public``      public multi-corpus train only
* ``adapt``       public train + the local broadcast *train* split
* ``adapt_aug``   the above plus offline augmented copies of training frames

Rules this script enforces rather than trusts:

* **Augmentation touches the training split only.** Validation and test are
  copied verbatim. An augmented validation frame would make model selection
  measure the augmentation.
* **No transform may invalidate a label.** Geometric transforms recompute the
  box; photometric ones cannot move it. Crops that would push the ball out of
  frame are rejected and retried, never emitted with a stale label.
* **Local test frames are never written into any training set.** Asserted
  against the shot-disjoint split, not assumed.

Usage::

    python scripts/build_broadcast_dataset.py --variant adapt_aug
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.annotation.schema import AnnotationStore, BallVisibility  # noqa: E402
from visionpitch.annotation.splits import LocalSplit  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("broadcast.dataset")

MULTICORPUS = Path("data/ball_multicorpus")
LOCAL_PACKAGE = Path("data/annotation/package")
LOCAL_SPLIT = Path("data/annotation/local_split.json")

#: Augmented copies produced per local training frame. The local set is 69
#: frames against 2465 public ones, so without replication the broadcast domain
#: is a rounding error in the gradient.
LOCAL_AUG_COPIES = 6
#: Augmented copies per public training frame.
PUBLIC_AUG_COPIES = 1
#: Pitch-line hard negatives cropped from training frames, away from the ball.
LINE_NEGATIVES = 120


def yolo_line(cx, cy, w, h, width, height) -> str:
    return (
        f"0 {cx / width:.6f} {cy / height:.6f} {w / width:.6f} {h / height:.6f}"
    )


def read_boxes(label_path: Path, width: int, height: int):
    """(cx, cy, w, h) in pixels."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append((cx * width, cy * height, bw * width, bh * height))
    return out


# --------------------------------------------------------------------------- #
# Augmentation -- training frames only
# --------------------------------------------------------------------------- #


def motion_blur(image, rng):
    """Directional blur, the failure mode public data almost never contains.

    Measured in the corpus audit: under 1.2% of public training balls read as
    blurred, against a broadcast where fast pans are routine.
    """
    k = rng.choice([5, 7, 9, 11])
    kernel = np.zeros((k, k), np.float32)
    angle = rng.uniform(0, np.pi)
    cx = cy = k // 2
    for i in range(k):
        x = int(round(cx + (i - cx) * np.cos(angle)))
        y = int(round(cy + (i - cy) * np.sin(angle)))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1
    total = kernel.sum()
    if total == 0:
        return image
    return cv2.filter2D(image, -1, kernel / total)


def jpeg(image, rng):
    ok, buf = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(28, 72)]
    )
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else image


def low_resolution(image, rng):
    h, w = image.shape[:2]
    f = rng.uniform(0.45, 0.8)
    small = cv2.resize(image, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def photometric(image, rng):
    out = image.astype(np.float32)
    out = out * rng.uniform(0.72, 1.32) + rng.uniform(-28, 28)          # brightness
    out = np.clip(out, 0, 255)
    gamma = rng.uniform(0.7, 1.45)                                       # gamma
    out = 255.0 * np.power(out / 255.0, gamma)
    out = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)          # mild colour
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-6, 6)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.2), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def noise(image, rng):
    sigma = rng.uniform(2.0, 8.0)
    out = image.astype(np.float32) + np.random.normal(0, sigma, image.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def scale_and_crop(image, boxes, rng):
    """Scale, then take a crop that keeps every ball fully inside.

    Retries rather than emitting a frame whose label no longer matches. A crop
    that clips the ball is discarded: a half-ball labelled as a whole one is
    exactly the silent corruption this pipeline must not create.
    """
    h, w = image.shape[:2]
    for _ in range(12):
        scale = rng.uniform(0.7, 1.45)
        nw, nh = int(w * scale), int(h * scale)
        if nw < 64 or nh < 64:
            continue
        scaled = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        moved = [(cx * scale, cy * scale, bw * scale, bh * scale)
                 for cx, cy, bw, bh in boxes]

        cw, ch = min(nw, w), min(nh, h)
        if not moved:
            x0 = rng.randint(0, max(0, nw - cw))
            y0 = rng.randint(0, max(0, nh - ch))
            return scaled[y0:y0 + ch, x0:x0 + cw], []

        # Bias the crop toward a ball so tiny objects survive cropping.
        anchor = moved[rng.randrange(len(moved))]
        x0 = int(np.clip(anchor[0] - cw / 2 + rng.uniform(-cw / 6, cw / 6), 0, max(0, nw - cw)))
        y0 = int(np.clip(anchor[1] - ch / 2 + rng.uniform(-ch / 6, ch / 6), 0, max(0, nh - ch)))
        crop = scaled[y0:y0 + ch, x0:x0 + cw]

        kept = []
        ok = True
        for cx, cy, bw, bh in moved:
            nx, ny = cx - x0, cy - y0
            if (nx - bw / 2 < 0 or ny - bh / 2 < 0
                    or nx + bw / 2 > crop.shape[1] or ny + bh / 2 > crop.shape[0]):
                ok = False
                break
            kept.append((nx, ny, bw, bh))
        if ok and kept:
            return crop, kept
    return None, None


def augment(image, boxes, rng):
    """One augmented copy, or (None, None) if a valid crop could not be made."""
    out, kept = scale_and_crop(image, boxes, rng)
    if out is None:
        return None, None
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
        kept = [(out.shape[1] - cx, cy, bw, bh) for cx, cy, bw, bh in kept]
    if rng.random() < 0.45:
        out = motion_blur(out, rng)
    elif rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (rng.choice([3, 5]), rng.choice([3, 5])), 0)
    if rng.random() < 0.35:
        out = low_resolution(out, rng)
    if rng.random() < 0.7:
        out = photometric(out, rng)
    if rng.random() < 0.45:
        out = jpeg(out, rng)
    if rng.random() < 0.3:
        out = noise(out, rng)
    return out, kept


def line_negative(image, boxes, rng):
    """A pitch-line crop containing no ball.

    The Phase 2D taxonomy put pitch-line confusion among the detector's false
    positives, and the public negatives are whole frames rather than the line
    close-ups a detector actually trips on.
    """
    h, w = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 70, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=70, maxLineGap=8)
    if lines is None:
        return None
    # HoughLinesP returns (N, 1, 4) on some OpenCV builds and (N, 4) on others,
    # so the shape is normalised rather than indexed by assumption.
    segments = np.asarray(lines).reshape(-1, 4)
    if not len(segments):
        return None
    for _ in range(10):
        x1, y1, x2, y2 = segments[rng.randrange(len(segments))]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        size = rng.randint(96, 192)
        x0 = int(np.clip(cx - size // 2, 0, max(0, w - size)))
        y0 = int(np.clip(cy - size // 2, 0, max(0, h - size)))
        crop_box = (x0, y0, x0 + size, y0 + size)
        # Reject any crop containing a ball -- a negative with a ball in it is
        # a mislabel, not a hard negative.
        if any(
            crop_box[0] <= bx <= crop_box[2] and crop_box[1] <= by <= crop_box[3]
            for bx, by, _, _ in boxes
        ):
            continue
        crop = image[y0:y0 + size, x0:x0 + size]
        if crop.size and crop.shape[0] >= 64 and crop.shape[1] >= 64:
            return crop
    return None


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True,
                        choices=["public", "adapt", "adapt_aug"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    configure_logging("INFO")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out = args.out or Path(f"data/ball_broadcast_{args.variant}")

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {
        "public_train": 0, "public_val": 0, "local_train": 0, "local_val": 0,
        "augmented": 0, "line_negatives": 0, "aug_rejected": 0,
    }

    # -- public data ----------------------------------------------------------- #
    for source_split, target_split, key in (
        ("train", "train", "public_train"), ("val", "val", "public_val")
    ):
        for image_path in sorted((MULTICORPUS / source_split / "images").glob("*.jpg")):
            label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            shutil.copyfile(image_path, out / target_split / "images" / image_path.name)
            shutil.copyfile(
                label, out / target_split / "labels" / f"{image_path.stem}.txt"
            ) if label.exists() else (
                out / target_split / "labels" / f"{image_path.stem}.txt"
            ).write_text("", encoding="utf-8")
            counts[key] += 1

    # -- local broadcast data -------------------------------------------------- #
    local_split = None
    if args.variant in ("adapt", "adapt_aug"):
        store = AnnotationStore(LOCAL_PACKAGE)
        samples = store.load_samples()
        annotations = store.load_annotations()
        local_split = LocalSplit.load(LOCAL_SPLIT)

        test_ids = set(local_split.frames.get("test", []))
        for name, target_split, key in (
            ("train", "train", "local_train"), ("val", "val", "local_val")
        ):
            for frame_id in local_split.frames.get(name, []):
                if frame_id in test_ids:
                    raise AssertionError(
                        f"{frame_id} is in the locked test split and must not be "
                        f"written into {target_split}"
                    )
                annotation = annotations[frame_id]
                sample = samples[frame_id]
                image = cv2.imread(sample.image_path)
                if image is None:
                    continue
                h, w = image.shape[:2]
                lines = []
                if (annotation.visibility is BallVisibility.VISIBLE
                        and annotation.radius_px):
                    d = annotation.radius_px * 2
                    lines.append(
                        yolo_line(annotation.centre_x, annotation.centre_y, d, d, w, h)
                    )
                stem = f"local_{frame_id}"
                cv2.imwrite(str(out / target_split / "images" / f"{stem}.jpg"),
                            image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out / target_split / "labels" / f"{stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
                counts[key] += 1

    # -- augmentation: training split only ------------------------------------- #
    if args.variant == "adapt_aug":
        train_images = sorted((out / "train" / "images").glob("*.jpg"))
        for image_path in train_images:
            is_local = image_path.stem.startswith("local_")
            copies = LOCAL_AUG_COPIES if is_local else PUBLIC_AUG_COPIES
            if not is_local and rng.random() > 0.45:
                continue  # augment a subset of the public frames, all local ones
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            boxes = read_boxes(
                image_path.parent.parent / "labels" / f"{image_path.stem}.txt", w, h
            )
            if not boxes:
                continue
            for copy_index in range(copies):
                aug, kept = augment(image, boxes, rng)
                if aug is None:
                    counts["aug_rejected"] += 1
                    continue
                stem = f"aug{copy_index}_{image_path.stem}"
                cv2.imwrite(str(out / "train" / "images" / f"{stem}.jpg"), aug,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                ah, aw = aug.shape[:2]
                (out / "train" / "labels" / f"{stem}.txt").write_text(
                    "\n".join(
                        yolo_line(cx, cy, bw, bh, aw, ah) for cx, cy, bw, bh in kept
                    ),
                    encoding="utf-8",
                )
                counts["augmented"] += 1

        # pitch-line hard negatives
        pool = [p for p in train_images if not p.stem.startswith("aug")]
        rng.shuffle(pool)
        for image_path in pool:
            if counts["line_negatives"] >= LINE_NEGATIVES:
                break
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            boxes = read_boxes(
                image_path.parent.parent / "labels" / f"{image_path.stem}.txt", w, h
            )
            crop = line_negative(image, boxes, rng)
            if crop is None:
                continue
            stem = f"lineneg_{counts['line_negatives']:04d}"
            cv2.imwrite(str(out / "train" / "images" / f"{stem}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "train" / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")
            counts["line_negatives"] += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: train/images\nval: val/images\n\nnc: 1\nnames: ['ball']\n",
        encoding="utf-8",
    )

    n_train = len(list((out / "train" / "images").glob("*.jpg")))
    n_val = len(list((out / "val" / "images").glob("*.jpg")))
    provenance = {
        "variant": args.variant,
        "seed": args.seed,
        "public_source": str(MULTICORPUS),
        "public_split_fingerprint": json.loads(
            (MULTICORPUS / "split.json").read_text(encoding="utf-8")
        )["fingerprint"],
        "local_split_fingerprint": (
            local_split.fingerprint() if local_split else None
        ),
        "counts": counts,
        "n_train_images": n_train,
        "n_val_images": n_val,
        "augmentation": {
            "applied_to": "train split only" if args.variant == "adapt_aug" else "none",
            "local_copies_per_frame": LOCAL_AUG_COPIES,
            "public_copies_per_frame": PUBLIC_AUG_COPIES,
            "transforms": [
                "scale 0.70-1.45 with ball-preserving crop",
                "horizontal flip", "directional motion blur", "mild Gaussian blur",
                "low-resolution simulation", "brightness/contrast", "gamma",
                "mild hue/saturation", "JPEG quality 28-72", "Gaussian noise",
            ],
            "pitch_line_hard_negatives": counts["line_negatives"],
            "rejected_crops": counts["aug_rejected"],
            "guarantees": [
                "no transform erases the ball",
                "geometric transforms recompute the label; crops clipping a ball "
                "are rejected and retried",
                "hard-negative crops are checked to contain no ball",
                "validation frames are copied verbatim, never augmented",
                "the locked local test split is never written into any training set",
            ],
        },
    }
    (out / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    print(f"\nvariant : {args.variant}")
    print(f"out     : {out}")
    print(f"train   : {n_train} images")
    print(f"val     : {n_val} images")
    print(f"counts  : {json.dumps(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
