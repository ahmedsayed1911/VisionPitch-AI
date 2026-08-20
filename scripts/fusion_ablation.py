"""Ablate the fusion stack on real sequences, detector held fixed.

Parts 9 and 10, fusion-only. The detector runs **once per checkpoint** and its
raw candidates are cached; every ablation row then re-scores the same candidates
offline. That is what makes the rows comparable -- any difference is the fusion
configuration and nothing else.

Ground truth comes from SN-GSR sequences, which carry a per-frame annotated ball,
so candidate-level precision and recall are real numbers here rather than
coverage proxies.

Usage::

    python scripts/fusion_ablation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.ball_tracking.fp_filter import FilterConfig  # noqa: E402
from visionpitch.ball_tracking.fusion import (  # noqa: E402
    BallFusion,
    FusionConfig,
    ObservationKind,
    SuppressionMethod,
)
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.possession_gt import load_gsr_gamestate  # noqa: E402

log = get_logger("fusion.ablation")

# Canonical VALID split. The historical default was ``data/eval/gsr``, which is
# the SN-GSR *test* split - those results are not eligible for model selection
# under the current protocol. Both roots use the identical
# ``SNGS-xxx/{Labels-GameState.json, img1/}`` layout, so only the root changes.
GSR_ROOT = Path("data/SoccerNetGS/valid")

#: Manifest used to prove no evaluated sequence belongs to test or challenge.
SPLIT_MANIFEST = Path("data/eval/gsr/sequences_info.json")
MATCH_PX = 25.0

DETECTORS = {
    "A_default": "models/yolo-football-ball-detection.pt",
    "C_adapt": "models/finetune/bcast_adapt/weights/best.pt",
}


def ablations() -> dict[str, FusionConfig]:
    """Each row isolates one component. Declared before any row was scored."""
    strict = FilterConfig()
    permissive = FilterConfig(
        min_support_frames=0, trust_confidence=0.0, max_step_px=1e9,
        camera_motion_px=1e9, max_size_ratio=1e9,
    )
    return {
        "1_no_fusion": FusionConfig(
            suppression=SuppressionMethod.NONE, temporal=permissive
        ),
        "2_iou_suppression_only": FusionConfig(
            suppression=SuppressionMethod.IOU, temporal=permissive
        ),
        "3_centre_suppression_only": FusionConfig(
            suppression=SuppressionMethod.CENTRE_DISTANCE, temporal=permissive
        ),
        "4_persistence_only": FusionConfig(
            suppression=SuppressionMethod.NONE,
            temporal=FilterConfig(camera_motion_px=1e9, max_step_px=1e9,
                                  max_size_ratio=1e9),
        ),
        "5_camera_motion_only": FusionConfig(
            suppression=SuppressionMethod.NONE,
            temporal=FilterConfig(min_support_frames=0, trust_confidence=0.0,
                                  max_step_px=1e9, max_size_ratio=1e9),
        ),
        "6_trajectory_only": FusionConfig(
            suppression=SuppressionMethod.NONE,
            temporal=FilterConfig(min_support_frames=0, trust_confidence=0.0,
                                  camera_motion_px=1e9),
        ),
        "7_suppression_plus_temporal": FusionConfig(
            suppression=SuppressionMethod.CENTRE_DISTANCE, temporal=strict
        ),
        "8_full_stack": FusionConfig(
            suppression=SuppressionMethod.WEIGHTED_CENTRE, temporal=strict
        ),
    }


def collect(model, sequences, conf: float, imgsz: int, max_frames: int | None):
    """Raw detector candidates plus ground truth and camera shifts, cached."""
    data = []
    for labels_path in sequences:
        frames_meta, fps = load_gsr_gamestate(labels_path)
        image_dir = labels_path.parent / "img1"
        if not image_dir.exists():
            continue
        frame_indices = sorted(frames_meta)
        if max_frames:
            frame_indices = frame_indices[:max_frames]

        truth: dict[int, tuple[float, float]] = {}
        detections: dict[int, list] = {}
        shifts: dict[int, tuple[float, float]] = {}
        previous = None
        kept: list[int] = []

        for frame_idx in frame_indices:
            path = image_dir / f"{frame_idx:06d}.jpg"
            if not path.exists():
                continue
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            kept.append(frame_idx)

            ball = next((o for o in frames_meta[frame_idx] if o.role == "ball"), None)
            if ball is not None:
                truth[frame_idx] = (
                    ball.image_x, ball.image_y - ball.box_height / 2
                )

            result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            found = []
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                for box, score in zip(boxes, scores, strict=True):
                    cx = float((box[0] + box[2]) / 2)
                    cy = float((box[1] + box[3]) / 2)
                    radius = float(max(box[2] - box[0], box[3] - box[1]) / 2)
                    found.append((cx, cy, radius, float(score)))
            detections[frame_idx] = found

            grey = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            if previous is not None:
                shift = cv2.phaseCorrelate(
                    previous.astype(np.float32), grey.astype(np.float32)
                )[0]
                scale = frame.shape[1] / 320.0
                shifts[frame_idx] = (
                    float(shift[0] * scale), float(shift[1] * scale)
                )
            previous = grey

        data.append({
            "sequence": labels_path.parent.name,
            "frame_indices": kept,
            "truth": truth,
            "detections": detections,
            "shifts": shifts,
        })
        log.info("  cached %s: %d frames, %d with truth",
                 labels_path.parent.name, len(kept), len(truth))
    return data


def score(data, config: FusionConfig) -> dict:
    fusion = BallFusion(config)
    tp = fp = fn = 0
    n_frames = 0
    kinds: dict[str, int] = {}
    merged_counts: list[int] = []
    single_frame_survivors = 0
    track_lengths: list[int] = []
    jumps = 0
    started = time.perf_counter()

    for entry in data:
        frames = fusion.run(
            entry["detections"], entry["frame_indices"],
            camera_shifts=entry["shifts"],
            camera_confidence={i: 1.0 for i in entry["shifts"]},
        )
        n_frames += len(frames)
        run = 0
        previous_position = None
        previous_idx = None
        for frame_idx in entry["frame_indices"]:
            frame = frames[frame_idx]
            kinds[frame.kind.value] = kinds.get(frame.kind.value, 0) + 1
            if frame.n_merged:
                merged_counts.append(frame.n_merged)

            truth = entry["truth"].get(frame_idx)
            if frame.kind.counts_as_observed:
                run += 1
                if truth is not None:
                    distance = float(np.hypot(frame.x - truth[0], frame.y - truth[1]))
                    if distance <= MATCH_PX:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                else:
                    fp += 1
                if previous_position is not None and previous_idx is not None:
                    gap = max(1, frame_idx - previous_idx)
                    step = float(np.hypot(
                        frame.x - previous_position[0], frame.y - previous_position[1]
                    )) / gap
                    if step > config.temporal.max_step_px:
                        jumps += 1
                previous_position = (frame.x, frame.y)
                previous_idx = frame_idx
            else:
                if run:
                    track_lengths.append(run)
                    if run == 1:
                        single_frame_survivors += 1
                run = 0
                previous_position = None
                previous_idx = None
                if truth is not None:
                    fn += 1
        if run:
            track_lengths.append(run)
            if run == 1:
                single_frame_survivors += 1

    elapsed = time.perf_counter() - started
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    lengths = np.array(track_lengths) if track_lengths else np.array([0])
    return {
        "config_fingerprint": config.fingerprint(),
        "suppression": config.suppression.value,
        "n_frames": n_frames,
        "detections_merged_into_selected": round(
            float(np.mean(merged_counts)) if merged_counts else 0.0, 4
        ),
        "frames_with_a_merge": round(
            float(np.mean([c > 1 for c in merged_counts])) if merged_counts else 0.0, 4
        ),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(
            2 * precision * recall / (precision + recall), 4
        ) if precision + recall else 0.0,
        "false_positives_per_frame": round(fp / max(1, n_frames), 4),
        "observed_coverage": round(
            sum(
                v for k, v in kinds.items()
                if ObservationKind(k).counts_as_observed
            ) / max(1, n_frames), 4
        ),
        "by_kind": dict(sorted(kinds.items())),
        "n_tracks": len(track_lengths),
        "single_frame_tracks": single_frame_survivors,
        "single_frame_share": round(
            single_frame_survivors / max(1, len(track_lengths)), 4
        ),
        "median_track_frames": float(np.median(lengths)),
        "implausible_jumps": jumps,
        "fusion_ms_per_frame": round(1000 * elapsed / max(1, n_frames), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--out", type=Path,
                        default=Path("data/eval/fusion/ablation.json"))
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    labels = {
        p.parent.name: p for p in sorted(GSR_ROOT.glob("**/Labels-GameState.json"))
    }
    if not labels:
        raise SystemExit(f"No SN-GSR sequences found under {GSR_ROOT}")

    # Leakage guard: prove every evaluated sequence is canonical VALID. This is
    # an assertion against the published manifest, not a filter over a mixed
    # pool - if anything from test/challenge appears here we abort rather than
    # silently drop it, because a contaminated selection is a silent result.
    if SPLIT_MANIFEST.exists():
        manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
        canonical = {
            key: {s["name"] for s in value}
            for key, value in manifest.items()
            if isinstance(value, list)
        }
        forbidden = sorted(
            s for s in labels
            if s in canonical.get("test", set()) | canonical.get("challenge", set())
        )
        if forbidden:
            raise SystemExit(
                f"ABORT - prohibited split detected under {GSR_ROOT}: {forbidden[:5]}"
            )
        outside = sorted(s for s in labels if s not in canonical.get("validation", set()))
        if outside:
            raise SystemExit(
                f"ABORT - sequences not in canonical validation split: {outside[:5]}"
            )

    chosen = [labels[s] for s in sorted(labels)][: args.sequences]
    log.info("ablating on %d canonical VALID sequence(s)", len(chosen))

    rows = ablations()
    results: dict[str, dict] = {}
    for label, weights in DETECTORS.items():
        path = Path(weights)
        if not path.exists():
            log.warning("%s missing; skipped", label)
            continue
        log.info("caching candidates for %s", label)
        model = YOLO(str(path))
        data = collect(model, chosen, args.conf, args.imgsz, args.max_frames)
        del model

        results[label] = {}
        for name, config in rows.items():
            results[label][name] = score(data, config)
            r = results[label][name]
            log.info(
                "  %-28s merged %.3f  mergeFr %.3f  R %.4f  P %.4f  FP/frame %.3f  "
                "coverage %.4f",
                name, r["detections_merged_into_selected"], r["frames_with_a_merge"],
                r["recall"], r["precision"], r["false_positives_per_frame"],
                r["observed_coverage"],
            )

    payload = {
        "schema_version": "1.0.0",
        "conf": args.conf,
        "imgsz": args.imgsz,
        "split": "soccernet_gsr validation (canonical, leakage-guarded)",
        # Fingerprint the exact evaluated sequence set, so a later run can be
        # proven to have scored the same footage.
        "split_fingerprint": hashlib.sha256(
            "|".join(p.parent.name for p in chosen).encode()
        ).hexdigest()[:16],
        "sequences_evaluated": [p.parent.name for p in chosen],
        "n_sequences": len(chosen),
        "max_frames_per_sequence": args.max_frames,
        "match_px": MATCH_PX,
        "note": (
            "the detector runs once per checkpoint and its raw candidates are "
            "cached; every ablation row re-scores the same candidates, so any "
            "difference is the fusion configuration alone"
        ),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for detector, rows_out in results.items():
        print(f"\n{detector}")
        print(f"{'ablation':<30}{'merged':>9}{'mergeFr':>8}{'recall':>8}{'prec':>8}"
              f"{'FP/fr':>8}{'1-frm':>7}{'cover':>8}")
        for name, r in rows_out.items():
            print(f"  {name:<28}{r['detections_merged_into_selected']:>9.3f}"
                  f"{r['frames_with_a_merge']:>7.3f}{r['recall']:>8.4f}"
                  f"{r['precision']:>8.4f}{r['false_positives_per_frame']:>8.3f}"
                  f"{r['single_frame_share']:>7.3f}{r['observed_coverage']:>8.4f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
