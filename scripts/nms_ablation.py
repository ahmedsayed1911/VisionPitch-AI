"""Zero-training NMS diagnostic: are referee false positives duplicate hypotheses?

Two parts:

A/B evaluation
    Identical VALID evaluations of the same checkpoint, differing only in
    ``agnostic_nms``. Class-wise NMS (the default) suppresses overlapping boxes
    only *within* a class, so one person can legitimately receive both a
    ``player`` box and a ``referee`` box. Agnostic NMS suppresses across
    classes, keeping only the highest-scoring hypothesis per location.

Duplicate-overlap census
    Runs raw inference and counts, directly, how often two different person
    classes claim the same location at IoU >= 0.5 / 0.7 / 0.9, and what
    fraction of referee predictions are (a) overlapped by a higher-scoring
    player prediction and (b) sitting on a ground-truth *player*. Together
    those separate "duplicate hypothesis" from "genuinely wrong detection".

VALID only. TEST and holdout are never referenced.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]
PLAYER, GOALKEEPER, REFEREE, BALL = 0, 1, 2, 3


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def evaluate(model, data: Path, agnostic: bool, name: str) -> dict:
    metrics = model.val(
        data=str(data), split="val", imgsz=1280, batch=8, device="0", workers=0,
        agnostic_nms=agnostic,          # the only variable
        plots=False, save_json=False,
        project="runs/diagnosis", name=name, exist_ok=True, verbose=False,
    )
    box = metrics.box
    per_class = {
        CLASS_NAMES[int(cid)]: {
            "precision": round(float(box.p[i]), 5),
            "recall": round(float(box.r[i]), 5),
            "AP50": round(float(box.ap50[i]), 5),
            "AP50-95": round(float(box.ap[i]), 5),
        }
        for i, cid in enumerate(box.ap_class_index)
    }
    out = {
        "agnostic_nms": agnostic,
        "aggregate": {
            "precision": round(float(box.mp), 5),
            "recall": round(float(box.mr), 5),
            "mAP50": round(float(box.map50), 5),
            "mAP50-95": round(float(box.map), 5),
        },
        "per_class": per_class,
    }
    try:
        cm = metrics.confusion_matrix.matrix
        labels = CLASS_NAMES + ["background"]
        idx = {n: i for i, n in enumerate(labels)}
        out["confusion"] = {
            "player_to_referee": int(cm[idx["referee"], idx["player"]]),
            "referee_to_player": int(cm[idx["player"], idx["referee"]]),
            "correct_referee": int(cm[idx["referee"], idx["referee"]]),
            "correct_player": int(cm[idx["player"], idx["player"]]),
        }
    except Exception as exc:  # pragma: no cover
        out["confusion_error"] = str(exc)
    return out


def duplicate_census(model, root: Path, sample: int, conf: float, seed: int) -> dict:
    """Count cross-class overlapping predictions on a stratified VALID sample."""
    images = sorted((root / "images" / "val").glob("*.jpg"))
    by_seq = defaultdict(list)
    for p in images:
        m = re.search(r"(SNGS-\d+)", p.name)
        by_seq[m.group(1) if m else "?"].append(p)
    picked: list[Path] = []
    per_seq = max(1, sample // max(1, len(by_seq)))
    for _seq, paths in sorted(by_seq.items()):
        paths.sort()
        step = max(1, len(paths) // per_seq)
        picked.extend(paths[::step][:per_seq])
    picked = picked[:sample]

    thresholds = (0.5, 0.7, 0.9)
    pair_counts = {f"{a}|{b}": dict.fromkeys(thresholds, 0)
                   for a, b in (("player", "referee"), ("player", "goalkeeper"),
                                ("referee", "goalkeeper"))}
    totals = Counter()
    ref_overlapped_by_player = dict.fromkeys(thresholds, 0)
    ref_on_gt_player = 0
    ref_total = 0
    ref_matched_gt_referee = 0

    labels_dir = root / "labels" / "val"

    for path in picked:
        res = model.predict(str(path), imgsz=1280, conf=conf, iou=0.7,
                            device="0", verbose=False, max_det=300)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()
        for c in cls:
            totals[CLASS_NAMES[c]] += 1

        groups = {c: xyxy[cls == c] for c in (PLAYER, GOALKEEPER, REFEREE)}
        score_groups = {c: scores[cls == c] for c in (PLAYER, GOALKEEPER, REFEREE)}

        for (ca, cb), key in (((PLAYER, REFEREE), "player|referee"),
                              ((PLAYER, GOALKEEPER), "player|goalkeeper"),
                              ((REFEREE, GOALKEEPER), "referee|goalkeeper")):
            M = iou_matrix(groups[ca], groups[cb])
            for t in thresholds:
                pair_counts[key][t] += int((M >= t).sum())

        # Referee predictions overlapped by a *higher-scoring* player prediction
        ref_boxes, ref_scores = groups[REFEREE], score_groups[REFEREE]
        ref_total += len(ref_boxes)
        if len(ref_boxes) and len(groups[PLAYER]):
            M = iou_matrix(ref_boxes, groups[PLAYER])
            for t in thresholds:
                for i in range(len(ref_boxes)):
                    over = np.where(M[i] >= t)[0]
                    if len(over) and score_groups[PLAYER][over].max() > ref_scores[i]:
                        ref_overlapped_by_player[t] += 1

        # Referee predictions sitting on ground-truth players
        lp = labels_dir / (path.stem + ".txt")
        if lp.exists() and len(ref_boxes):
            H, W = res.orig_shape
            gt_p, gt_r = [], []
            for line in lp.read_text(encoding="utf-8").splitlines():
                f = line.split()
                if len(f) < 5:
                    continue
                cid = int(f[0])
                cx, cy, bw, bh = (float(v) for v in f[1:5])
                box = [(cx - bw / 2) * W, (cy - bh / 2) * H,
                       (cx + bw / 2) * W, (cy + bh / 2) * H]
                if cid == PLAYER:
                    gt_p.append(box)
                elif cid == REFEREE:
                    gt_r.append(box)
            gt_p, gt_r = np.array(gt_p), np.array(gt_r)
            if len(gt_p):
                ref_on_gt_player += int((iou_matrix(ref_boxes, gt_p) >= 0.5).any(axis=1).sum())
            if len(gt_r):
                ref_matched_gt_referee += int(
                    (iou_matrix(ref_boxes, gt_r) >= 0.5).any(axis=1).sum()
                )

    return {
        "images_sampled": len(picked),
        "confidence": conf,
        "predictions_by_class": dict(totals),
        "cross_class_overlaps": {
            k: {str(t): v[t] for t in thresholds} for k, v in pair_counts.items()
        },
        "referee_predictions": ref_total,
        "referee_overlapped_by_higher_scoring_player": {
            str(t): ref_overlapped_by_player[t] for t in thresholds
        },
        "referee_pred_on_gt_player_iou50": ref_on_gt_player,
        "referee_pred_on_gt_referee_iou50": ref_matched_gt_referee,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path,
                   default=Path("runs/detect/runs/vp_yolo11x_gsr_1280/weights/best.pt"))
    p.add_argument("--data", type=Path, default=Path("data/yolo_gsr_detect/dataset.yaml"))
    p.add_argument("--root", type=Path, default=Path("data/yolo_gsr_detect"))
    p.add_argument("--sample", type=int, default=1200)
    p.add_argument("--census-conf", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path,
                   default=Path("runs/diagnosis/nms_ablation.json"))
    args = p.parse_args()

    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if "test" in cfg:
        sys.exit("REFUSING: dataset.yaml declares a 'test' split.")
    if not args.weights.exists():
        sys.exit(f"REFUSING: missing {args.weights}")

    from ultralytics import YOLO

    report = {"weights": str(args.weights), "split": "val", "imgsz": 1280}

    print("=== A: default class-wise NMS ===", flush=True)
    report["A_default_nms"] = evaluate(YOLO(str(args.weights)), args.data, False, "nms_A")

    print("=== B: agnostic_nms=True ===", flush=True)
    report["B_agnostic_nms"] = evaluate(YOLO(str(args.weights)), args.data, True, "nms_B")

    print("=== duplicate-overlap census ===", flush=True)
    report["duplicate_census"] = duplicate_census(
        YOLO(str(args.weights)), args.root, args.sample, args.census_conf, args.seed
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
