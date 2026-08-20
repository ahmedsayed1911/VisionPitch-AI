"""Frame-by-frame calibration failure attribution on a real broadcast segment.

Answers, per frame, which specific gate rejected calibration - rather than
collapsing everything into `no_calibration`. Measures the keypoint stage and the
homography stage separately, because they fail for different reasons and only
one of them is fixable by coordinate handling.

Read-only: no config is modified, no threshold changed, nothing trained.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.calibration.homography import estimate_homography  # noqa: E402
from visionpitch.calibration.keypoints import PitchKeypointDetector  # noqa: E402
from visionpitch.common.config import load_config  # noqa: E402
from visionpitch.pitch.geometry import PitchConfiguration  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=225.0)
    ap.add_argument("--end", type=float, default=265.0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/diagnosis/calib_diag.json"))
    a = ap.parse_args()

    cfg = load_config(mode="balanced")
    if a.imgsz:
        cfg.calibration.imgsz = a.imgsz
    det = PitchKeypointDetector(cfg)
    pitch_cfg = PitchConfiguration(length=cfg.pitch.length_m, width=cfg.pitch.width_m)

    video = glob.glob("*.mp4")[0]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f0, f1 = int(a.start * fps), int(a.end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    conf_floor = cfg.calibration.keypoint_conf_threshold
    reasons: Counter = Counter()
    rows, n_kp, confs = [], [], []
    idx = f0
    while idx < f1:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - f0) % a.stride == 0:
            kp = det.detect_batch([frame], [idx])[0]
            if kp is None:
                reasons["model_no_output"] += 1
            else:
                indices, pts, cs = kp.confident(conf_floor)
                n_kp.append(int(indices.size))
                confs.extend(cs.tolist())
                if indices.size < cfg.calibration.min_keypoints:
                    reasons[f"insufficient_keypoints(<{cfg.calibration.min_keypoints})"] += 1
                    rows.append({"frame": idx, "n_conf": int(indices.size),
                                 "max_conf": float(kp.confidences.max()),
                                 "reason": "insufficient_keypoints"})
                else:
                    fit = estimate_homography(
                        pts, indices, pitch_cfg,
                        (frame.shape[1], frame.shape[0]),
                        keypoint_confidences=cs,
                        ransac_threshold_m=cfg.calibration.ransac_threshold_m,
                        max_reprojection_error_m=cfg.calibration.max_reprojection_error_m,
                        min_keypoints=cfg.calibration.min_keypoints,
                    )
                    r = getattr(fit, "rejection_reason", None) or (
                        "ok" if getattr(fit, "ok", False) else "unknown"
                    )
                    reasons[r] += 1
                    rows.append({"frame": idx,
                                 "n_conf": int(indices.size),
                                 "reason": r,
                                 "reproj": float(
                                     getattr(fit, "reprojection_error_m", float("nan"))
                                 ),
                                 "conf": float(getattr(fit, "confidence", float("nan")))})
        idx += 1
    cap.release()

    rep = {
        "video_size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280, 720],
        "calib_imgsz": cfg.calibration.imgsz,
        "keypoint_conf_threshold": conf_floor,
        "min_keypoints": cfg.calibration.min_keypoints,
        "max_reprojection_error_m": cfg.calibration.max_reprojection_error_m,
        "min_confidence": cfg.calibration.min_confidence,
        "frames_sampled": len(rows),
        "reason_distribution": dict(reasons),
        "keypoints_above_threshold": {
            "mean": round(float(np.mean(n_kp)), 2) if n_kp else 0,
            "median": float(np.median(n_kp)) if n_kp else 0,
            "min": int(min(n_kp)) if n_kp else 0, "max": int(max(n_kp)) if n_kp else 0,
        },
        "confidence_percentiles": {
            p: round(float(np.percentile(confs, p)), 4) for p in (10, 50, 90)
        } if confs else {},
        "rows": rows[:40],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
