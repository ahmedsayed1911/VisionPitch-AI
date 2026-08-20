"""Phase 3: is semantic role better decided per-track than per-frame?

Runs the canonical detector + Phase-2D tracker over VALID sequences, records the
per-observation predicted class (which the production pipeline currently
discards), and compares frame-level role assignment against track-level
aggregation.

Why this matters architecturally
--------------------------------
``_ActiveTrack.__init__`` fixes ``object_class`` from the *first* detection and
never revises it; ``team_classification/classifier.py`` then derives the role
straight from that single value. So today one bad first detection determines a
whole track's role. This experiment measures what per-observation voting would
buy, without changing production behaviour.

Sequence selection is deliberately adversarial rather than convenient: the ten
sequences span the full measured referee/player colour-separability range from
the earlier audit, so the hardest cases are included by construction.

VALID only. Nothing is trained; nothing in the trunk is modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]
PERSON_CLASSES = ("player", "goalkeeper", "referee")
LENGTH_BANDS = [(1, 4, "<5"), (5, 14, "5-14"), (15, 29, "15-29"),
                (30, 49, "30-49"), (50, 10**9, "50+")]


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


def select_sequences(root: Path, n: int) -> list[str]:
    """Stratified spread across measured referee/player separability."""
    sep_path = Path("runs/diagnosis/referee_audit/per_sequence_separability.json")
    available = sorted({
        m.group(1)
        for p in (root / "images" / "val").glob("*.jpg")
        if (m := re.search(r"(SNGS-\d+)", p.name))
    })
    if not sep_path.exists():
        return available[:n]
    sep = json.loads(sep_path.read_text(encoding="utf-8"))
    ranked = [s for s, _ in sorted(sep.items(), key=lambda kv: kv[1]) if s in available]
    if len(ranked) <= n:
        return ranked
    idx = np.linspace(0, len(ranked) - 1, n).round().astype(int)
    return [ranked[i] for i in dict.fromkeys(idx)]


def load_gt(labels_dir: Path, stem: str, w: int, h: int):
    """Ground-truth boxes and class ids for one frame."""
    lp = labels_dir / f"{stem}.txt"
    boxes, classes = [], []
    if not lp.exists():
        return np.zeros((0, 4)), []
    for line in lp.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        cid = int(f[0])
        cx, cy, bw, bh = (float(v) for v in f[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
        classes.append(cid)
    return np.array(boxes) if boxes else np.zeros((0, 4)), classes


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def score(pairs: list[tuple[str, str]]) -> dict:
    """pairs = (gt_class, pred_class); pred may be 'unknown' (abstention)."""
    out = {}
    for c in PERSON_CLASSES:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        out[c] = prf(tp, fp, fn)
    decided = [(g, p) for g, p in pairs if p != "unknown"]
    out["_overall"] = {
        "n": len(pairs),
        "n_decided": len(decided),
        "coverage": round(len(decided) / len(pairs), 4) if pairs else 0.0,
        "accuracy_on_decided": round(
            sum(1 for g, p in decided if g == p) / len(decided), 4) if decided else 0.0,
    }
    conf = defaultdict(Counter)
    for g, p in pairs:
        conf[g][p] += 1
    out["_confusion_gt_to_pred"] = {g: dict(c) for g, c in conf.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path,
                    default=Path("runs/detect/runs/vp_yolo11x_gsr_1280/weights/best.pt"))
    ap.add_argument("--root", type=Path, default=Path("data/yolo_gsr_detect"))
    ap.add_argument("--sequences", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-margin", type=float, default=0.10,
                    help="vote margin below which a track abstains (UNKNOWN)")
    ap.add_argument("--out", type=Path,
                    default=Path("runs/diagnosis/phase3_track_roles.json"))
    args = ap.parse_args()

    if not args.weights.exists():
        sys.exit(f"REFUSING: missing {args.weights}")

    from ultralytics import YOLO

    from visionpitch.common.config import Config
    from visionpitch.common.types import BBox, Detection, ObjectClass
    from visionpitch.tracking.tracker import MultiObjectTracker

    config = Config()
    config.tracking.reid_enabled = False  # image embeddings not needed for role voting
    model = YOLO(str(args.weights))

    images_dir, labels_dir = args.root / "images" / "val", args.root / "labels" / "val"
    seqs = select_sequences(args.root, args.sequences)
    print(f"sequences: {seqs}")

    frame_pairs: list[tuple[str, str]] = []          # per-frame role baseline
    track_records: list[dict] = []                   # one per predicted track

    for seq in seqs:
        frames = sorted(images_dir.glob(f"*{seq}*.jpg"))[: args.max_frames]
        if not frames:
            continue
        tracker = MultiObjectTracker(config)
        # track_id -> list of (frame_idx, pred_class, confidence, bbox)
        obs: dict[int, list] = defaultdict(list)
        # track_id -> Counter of matched GT class
        gt_votes: dict[int, Counter] = defaultdict(Counter)

        for fi, path in enumerate(frames):
            res = model.predict(str(path), imgsz=1280, device="0", verbose=False)[0]
            h, w = res.orig_shape
            gt_boxes, gt_classes = load_gt(labels_dir, path.stem, w, h)

            dets, det_meta = [], []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                cf = res.boxes.conf.cpu().numpy()
                # per-frame baseline: match every person detection to GT
                person = [i for i in range(len(cls)) if CLASS_NAMES[cls[i]] in PERSON_CLASSES]
                if person and len(gt_boxes):
                    M = iou_matrix(xyxy[person], gt_boxes)
                    for k, i in enumerate(person):
                        j = int(M[k].argmax())
                        if M[k, j] >= args.iou and CLASS_NAMES[gt_classes[j]] in PERSON_CLASSES:
                            frame_pairs.append(
                                (CLASS_NAMES[gt_classes[j]], CLASS_NAMES[cls[i]])
                            )
                for i in range(len(cls)):
                    name = CLASS_NAMES[cls[i]]
                    if name not in PERSON_CLASSES:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                    dets.append(Detection(
                        frame_idx=fi, object_class=ObjectClass(name),
                        bbox=BBox.from_xyxy(np.array([x1, y1, x2, y2])),
                        confidence=float(cf[i])))
                    det_meta.append((name, float(cf[i]), np.array([x1, y1, x2, y2])))

            active = tracker.update(dets, fi, fi / 25.0)

            # Recover which detection backed each track this frame by bbox match.
            if det_meta:
                dboxes = np.stack([m[2] for m in det_meta])
                for t in active:
                    if t.time_since_update != 0 or not t.observations:
                        continue
                    ob = t.observations[-1]
                    if ob.frame_idx != fi or ob.interpolated:
                        continue
                    tb = ob.bbox.to_array().reshape(1, 4)
                    M = iou_matrix(tb, dboxes)[0]
                    j = int(M.argmax())
                    if M[j] < 0.9:
                        continue
                    obs[t.track_id].append((fi, det_meta[j][0], det_meta[j][1]))
                    if len(gt_boxes):
                        G = iou_matrix(tb, gt_boxes)[0]
                        gj = int(G.argmax())
                        if G[gj] >= args.iou and CLASS_NAMES[gt_classes[gj]] in PERSON_CLASSES:
                            gt_votes[t.track_id][CLASS_NAMES[gt_classes[gj]]] += 1

        for tid, items in obs.items():
            if not gt_votes[tid]:
                continue  # no GT identity: excluded rather than guessed at
            gt_class = gt_votes[tid].most_common(1)[0][0]
            classes = [c for _, c, _ in items]
            confs = [s for _, _, s in items]
            counts = Counter(classes)
            weighted: Counter = Counter()
            for c, s in zip(classes, confs, strict=False):
                weighted[c] += s
            flips = sum(1 for a, b in zip(classes, classes[1:], strict=False) if a != b)
            track_records.append({
                "sequence": seq, "track_id": tid, "n_obs": len(items),
                "gt_class": gt_class,
                "first_class": classes[0],
                "majority": counts.most_common(1)[0][0],
                "majority_share": round(counts.most_common(1)[0][1] / len(classes), 4),
                "weighted": weighted.most_common(1)[0][0],
                "weighted_share": round(
                    weighted.most_common(1)[0][1] / max(sum(weighted.values()), 1e-9), 4),
                "flips": flips,
                "class_counts": dict(counts),
            })

    # ---- scoring ---------------------------------------------------------
    def abstain(rec: dict, key: str, share_key: str) -> str:
        return rec[key] if rec[share_key] - (1 - rec[share_key]) >= args.min_margin \
            else ("unknown" if rec[share_key] < 0.5 + args.min_margin / 2 else rec[key])

    report = {
        "sequences": seqs,
        "iou_match": args.iou,
        "n_tracks": len(track_records),
        "tracks_by_gt_class": dict(Counter(r["gt_class"] for r in track_records)),
        "per_frame": score(frame_pairs),
        "track_first_detection_current_behaviour": score(
            [(r["gt_class"], r["first_class"]) for r in track_records]),
        "track_majority_vote": score(
            [(r["gt_class"], r["majority"]) for r in track_records]),
        "track_confidence_weighted": score(
            [(r["gt_class"], r["weighted"]) for r in track_records]),
        "track_majority_with_abstention": score(
            [(r["gt_class"], abstain(r, "majority", "majority_share")) for r in track_records]),
    }

    # referee by track length
    bands = {}
    for lo, hi, name in LENGTH_BANDS:
        sub = [r for r in track_records if lo <= r["n_obs"] <= hi]
        if not sub:
            bands[name] = {"n_tracks": 0}
            continue
        s = score([(r["gt_class"], r["majority"]) for r in sub])
        bands[name] = {
            "n_tracks": len(sub),
            "n_gt_referee": sum(1 for r in sub if r["gt_class"] == "referee"),
            "referee": s["referee"],
            "accuracy": s["_overall"]["accuracy_on_decided"],
        }
    report["referee_by_track_length"] = bands

    ref_tracks = [r for r in track_records if r["gt_class"] == "referee"]
    player_tracks = [r for r in track_records if r["gt_class"] == "player"]
    report["error_stability"] = {
        "mean_obs_per_referee_track": round(
            float(np.mean([r["n_obs"] for r in ref_tracks])), 2) if ref_tracks else None,
        "mean_flips_per_track": round(
            float(np.mean([r["flips"] for r in track_records])), 3) if track_records else None,
        "tracks_with_zero_flips": sum(1 for r in track_records if r["flips"] == 0),
        "gt_player_tracks_where_referee_wins": sum(
            1 for r in player_tracks if r["majority"] == "referee"),
        "gt_player_tracks": len(player_tracks),
        "gt_referee_tracks_where_player_wins": sum(
            1 for r in ref_tracks if r["majority"] == "player"),
        "gt_referee_tracks": len(ref_tracks),
        "mean_majority_share": round(
            float(np.mean([r["majority_share"] for r in track_records])), 4)
        if track_records else None,
    }
    report["failure_examples"] = sorted(
        [r for r in track_records if r["gt_class"] != r["majority"]],
        key=lambda r: -r["n_obs"])[:10]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("n_tracks", "tracks_by_gt_class", "error_stability")}, indent=2))
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
