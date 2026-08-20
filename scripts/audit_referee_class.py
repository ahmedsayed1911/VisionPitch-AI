"""Referee-class audit: separability, label sanity, and the HSV hypothesis.

Answers three measurable questions without training anything:

1. **Is referee identity carried by colour at all?** Torso-region hue/saturation
   statistics per class, sampled across many sequences.
2. **How separable are referee and player?** A cheap logistic probe on the
   colour histogram, scored by cross-validated accuracy. This is a *lower*
   bound on what a CNN could learn, but if colour alone separates them well,
   the class is learnable and the failure is elsewhere.
3. **Does the training augmentation destroy that cue?** The exact Ultralytics
   HSV transform is applied at the run's settings and the probe re-scored.
   The drop is the measured cost of the augmentation.

Sampling is stratified across sequences, never consecutive frames, because
adjacent frames are near-duplicates and would inflate every estimate.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
TORSO = (0.15, 0.55)  # vertical band of the box treated as shirt


def hsv_augment(img: np.ndarray, hgain: float, sgain: float, vgain: float,
                rng: np.random.Generator) -> np.ndarray:
    """Ultralytics' HSV augmentation, reproduced exactly."""
    r = rng.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    merged = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def torso_descriptor(crop: np.ndarray, bins: tuple[int, int] = (12, 6)) -> np.ndarray:
    """Normalised hue-saturation histogram of the torso band."""
    h, w = crop.shape[:2]
    if h < 8 or w < 4:
        return np.zeros(bins[0] * bins[1], dtype=np.float32)
    y0, y1 = int(h * TORSO[0]), max(int(h * TORSO[1]), int(h * TORSO[0]) + 1)
    x0, x1 = int(w * 0.2), max(int(w * 0.8), int(w * 0.2) + 1)
    patch = crop[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros(bins[0] * bins[1], dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # Mask grass so the descriptor describes kit, not pitch.
    grass = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90)) & (hsv[:, :, 1] >= 60)
    mask = (~grass).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        mask = None
    hist = cv2.calcHist([hsv], [0, 1], mask, list(bins), [0, 180, 0, 256]).ravel()
    total = hist.sum()
    return (hist / total).astype(np.float32) if total > 0 else hist.astype(np.float32)


def probe_accuracy(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """5-fold CV accuracy of a logistic probe on colour descriptors."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    if len(np.unique(y)) < 2 or len(y) < 50:
        return {"error": "insufficient samples"}
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    scores = cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy")
    return {"balanced_accuracy": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4), "n": int(len(y))}


def collect(images_dir: Path, labels_dir: Path, per_class: int, seed: int):
    """Stratified sample of boxes, spread across sequences."""
    rng = random.Random(seed)
    by_seq = defaultdict(list)
    for p in images_dir.glob("*.jpg"):
        m = re.search(r"(SNGS-\d+)", p.name)
        by_seq[m.group(1) if m else "unknown"].append(p)

    # Take a bounded number of well-separated frames from every sequence.
    frames = []
    for _seq, paths in by_seq.items():
        paths.sort()
        step = max(1, len(paths) // 25)
        frames.extend(paths[::step][:25])
    rng.shuffle(frames)

    wanted = {0: per_class, 1: per_class, 2: per_class}
    got = Counter()
    crops: dict[int, list[np.ndarray]] = defaultdict(list)
    seqs_used: dict[int, set] = defaultdict(set)
    boxes_seen = 0
    degenerate = 0

    for path in frames:
        if all(got[c] >= wanted[c] for c in wanted):
            break
        lp = labels_dir / (path.stem + ".txt")
        if not lp.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        H, W = img.shape[:2]
        seq = re.search(r"(SNGS-\d+)", path.name)
        seq = seq.group(1) if seq else "unknown"

        for line in lp.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            cid = int(f[0])
            boxes_seen += 1
            if cid not in wanted or got[cid] >= wanted[cid]:
                continue
            cx, cy, bw, bh = (float(v) for v in f[1:5])
            x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 6 or y2 - y1 < 12:
                degenerate += 1
                continue
            crops[cid].append(img[y1:y2, x1:x2].copy())
            seqs_used[cid].add(seq)
            got[cid] += 1
    return crops, dict(got), {k: len(v) for k, v in seqs_used.items()}, boxes_seen, degenerate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/yolo_gsr_detect"))
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument("--per-class", type=int, default=900)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--hsv-s", type=float, default=0.7)
    p.add_argument("--hsv-v", type=float, default=0.4)
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--out", type=Path, default=Path("runs/diagnosis/referee_audit"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    images = args.root / "images" / args.split
    labels = args.root / "labels" / args.split

    crops, got, seqs, boxes_seen, degenerate = collect(
        images, labels, args.per_class, args.seed
    )
    print(f"sampled: {got}  across sequences: {seqs}  (scanned {boxes_seen} boxes)")

    rng = np.random.default_rng(args.seed)

    # --- descriptors, clean vs augmented ---------------------------------
    clean: dict[int, np.ndarray] = {}
    aug: dict[int, np.ndarray] = {}
    for cid, items in crops.items():
        clean[cid] = np.stack([torso_descriptor(c) for c in items])
        aug[cid] = np.stack(
            [torso_descriptor(hsv_augment(c, args.hsv_h, args.hsv_s, args.hsv_v, rng))
             for c in items]
        )

    def pair(a: int, b: int, table: dict) -> dict:
        if a not in table or b not in table:
            return {"error": "missing class"}
        X = np.vstack([table[a], table[b]])
        y = np.array([0] * len(table[a]) + [1] * len(table[b]))
        return probe_accuracy(X, y, args.seed)

    report = {
        "split": args.split,
        "sampled_per_class": {CLASS_NAMES[k]: v for k, v in got.items()},
        "sequences_covered": {CLASS_NAMES[k]: v for k, v in seqs.items()},
        "boxes_scanned": boxes_seen,
        "degenerate_boxes_skipped": degenerate,
        "augmentation": {"hsv_h": args.hsv_h, "hsv_s": args.hsv_s, "hsv_v": args.hsv_v},
        "colour_separability": {
            "referee_vs_player": {
                "clean": pair(2, 0, clean),
                "augmented": pair(2, 0, aug),
            },
            "goalkeeper_vs_player": {
                "clean": pair(1, 0, clean),
                "augmented": pair(1, 0, aug),
            },
            "referee_vs_goalkeeper": {
                "clean": pair(2, 1, clean),
                "augmented": pair(2, 1, aug),
            },
        },
    }

    # --- montage: real crops, before and after augmentation ---------------
    tiles = []
    for cid in (0, 1, 2):
        row = []
        for c in crops.get(cid, [])[:8]:
            t = cv2.resize(c, (48, 96))
            a = cv2.resize(hsv_augment(c, args.hsv_h, args.hsv_s, args.hsv_v, rng), (48, 96))
            row.append(np.vstack([t, a]))
        if row:
            tiles.append(np.hstack(row))
    if tiles:
        width = max(t.shape[1] for t in tiles)
        padded = [np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0))) for t in tiles]
        cv2.imwrite(str(args.out / f"crops_{args.split}.png"), np.vstack(padded))

    (args.out / f"referee_audit_{args.split}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["colour_separability"], indent=2))
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
