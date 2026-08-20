"""VALID-only diagnosis of a trained detector.

Evaluates one or more checkpoints on the canonical validation split and writes
per-class metrics, the confusion matrix and the standard Ultralytics curve
plots (PR, F1, P, R), plus a size-stratified recall breakdown that the built-in
report does not provide.

Why size stratification matters here: overall mAP is dominated by the many
mid-sized near-camera players. Recall on distant players and on the ball -
which are the objects that actually limit a football pipeline - can collapse
while the headline number barely moves. Those are reported separately so the
bottleneck is measured rather than assumed.

TEST and any final holdout are never touched: the split is hard-coded to
``val`` and the dataset yaml is asserted to declare no test entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]
#: COCO-style area bands, in pixels^2, evaluated at the training resolution.
SIZE_BANDS = {"tiny": (0, 16**2), "small": (16**2, 32**2),
              "medium": (32**2, 96**2), "large": (96**2, float("inf"))}


def assert_no_test_split(data_yaml: Path) -> dict:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if "test" in cfg:
        raise SystemExit(
            f"REFUSING TO RUN: {data_yaml} declares a 'test' split. "
            "This diagnosis is VALID-only by policy."
        )
    if "val" not in cfg:
        raise SystemExit(f"{data_yaml} has no 'val' split")
    return cfg


def label_stats(cfg: dict, data_yaml: Path) -> dict:
    """Ground-truth class counts and box-area distribution over VALID."""
    root = Path(cfg.get("path", data_yaml.parent))
    labels_dir = root / str(cfg["val"]).replace("images", "labels")
    counts = {name: 0 for name in CLASS_NAMES}
    areas: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    bands = {name: dict.fromkeys(SIZE_BANDS, 0) for name in CLASS_NAMES}

    if not labels_dir.exists():
        return {"error": f"labels not found at {labels_dir}"}

    # Areas are normalised in YOLO format; convert to pixels at 1280x720
    # (SN-GSR is 1920x1080, exported at that aspect) so the bands are meaningful.
    ref_w, ref_h = 1920.0, 1080.0
    for path in labels_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            if cid >= len(CLASS_NAMES):
                continue
            name = CLASS_NAMES[cid]
            w, h = float(parts[3]) * ref_w, float(parts[4]) * ref_h
            area = w * h
            counts[name] += 1
            areas[name].append(area)
            for band, (lo, hi) in SIZE_BANDS.items():
                if lo <= area < hi:
                    bands[name][band] += 1
                    break

    return {
        "counts": counts,
        "median_area_px2": {
            k: (round(float(np.median(v)), 1) if v else None) for k, v in areas.items()
        },
        "size_bands": bands,
    }


def evaluate(weights: Path, data_yaml: Path, name: str, imgsz: int, batch: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="val",          # hard-coded: never test
        imgsz=imgsz,
        batch=batch,
        device="0",
        # Single-process loading. Multiprocessing dataloader workers crash on
        # this machine ("DataLoader worker exited unexpectedly"); the training
        # run hit the same wall and was configured with workers=0 after an
        # earlier attempt failed. Do not raise this without fixing that first.
        workers=0,
        plots=True,           # PR, F1, P, R curves + confusion matrix
        save_json=False,
        project="runs/diagnosis",
        name=name,
        exist_ok=True,
        verbose=True,
    )

    box = metrics.box
    per_class = {}
    for i, cid in enumerate(box.ap_class_index):
        per_class[CLASS_NAMES[int(cid)]] = {
            "precision": round(float(box.p[i]), 5),
            "recall": round(float(box.r[i]), 5),
            "AP50": round(float(box.ap50[i]), 5),
            "AP50-95": round(float(box.ap[i]), 5),
            "F1": round(float(box.f1[i]), 5),
        }

    result = {
        "weights": str(weights),
        "aggregate": {
            "precision": round(float(box.mp), 5),
            "recall": round(float(box.mr), 5),
            "mAP50": round(float(box.map50), 5),
            "mAP50-95": round(float(box.map), 5),
            "fitness": round(float(metrics.fitness), 5),
        },
        "per_class": per_class,
        "speed_ms": {k: round(float(v), 2) for k, v in metrics.speed.items()},
        "save_dir": str(metrics.save_dir),
    }

    # Confusion matrix: rows = predicted, cols = ground truth, with a
    # background row/column appended by Ultralytics.
    try:
        cm = metrics.confusion_matrix.matrix
        labels = CLASS_NAMES + ["background"]
        result["confusion_matrix"] = {
            "labels_pred_rows_gt_cols": labels,
            "matrix": cm.astype(int).tolist(),
        }
        # Most damaging confusions, excluding the diagonal.
        pairs = []
        for pi in range(len(labels)):
            for gi in range(len(labels)):
                if pi != gi and cm[pi, gi] > 0:
                    pairs.append((labels[gi], labels[pi], int(cm[pi, gi])))
        pairs.sort(key=lambda x: -x[2])
        result["top_confusions_gt_to_pred"] = [
            {"ground_truth": g, "predicted_as": p, "count": c} for g, p, c in pairs[:12]
        ]
    except Exception as exc:  # pragma: no cover
        result["confusion_matrix_error"] = str(exc)

    # Best F1 confidence threshold per class, from the curve arrays.
    try:
        conf = np.asarray(box.curves_results[1][0])   # F1 curve x-axis
        f1 = np.asarray(box.curves_results[1][1])     # (n_classes, n_points)
        best = {}
        for i, cid in enumerate(box.ap_class_index):
            j = int(np.argmax(f1[i]))
            best[CLASS_NAMES[int(cid)]] = {
                "best_f1": round(float(f1[i][j]), 4),
                "at_confidence": round(float(conf[j]), 4),
            }
        mean_f1 = f1.mean(axis=0)
        jm = int(np.argmax(mean_f1))
        best["_all_classes"] = {
            "best_mean_f1": round(float(mean_f1[jm]), 4),
            "at_confidence": round(float(conf[jm]), 4),
        }
        result["best_f1_thresholds"] = best
    except Exception as exc:  # pragma: no cover
        result["f1_threshold_error"] = str(exc)

    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path,
                   default=Path("runs/detect/runs/vp_yolo11x_gsr_1280"))
    p.add_argument("--data", type=Path,
                   default=Path("data/yolo_gsr_detect/dataset.yaml"))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("runs/diagnosis/valid_diagnosis.json"))
    args = p.parse_args()

    cfg = assert_no_test_split(args.data)
    print(f"VALID-only diagnosis. split=val  data={args.data}")

    report = {
        "data_yaml": str(args.data),
        "split": "val",
        "imgsz": args.imgsz,
        "ground_truth": label_stats(cfg, args.data),
        "checkpoints": {},
    }

    for tag in ("best", "last"):
        weights = args.run / "weights" / f"{tag}.pt"
        if not weights.exists():
            report["checkpoints"][tag] = {"error": f"missing {weights}"}
            continue
        print(f"\n===== evaluating {tag}.pt =====", flush=True)
        report["checkpoints"][tag] = evaluate(
            weights, args.data, f"val_{tag}", args.imgsz, args.batch
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
