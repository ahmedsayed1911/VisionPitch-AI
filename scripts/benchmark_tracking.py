"""Measure detection and tracking against SoccerNet Game State Reconstruction.

This closes the Phase 1B item that could not be closed before: real HOTA, IDF1
and MOTA on identity-consistent ground truth. SN-GSR is **out of distribution**
for the shipped checkpoints -- a different corpus from the one they were
fine-tuned on -- so these numbers describe generalisation rather than in-domain
performance.

The sequences are image folders rather than videos, so the pipeline's ingestion
stage is bypassed and the detector and tracker are driven directly. Everything
downstream of tracking is untouched.

Usage::

    python scripts/benchmark_tracking.py --sequences 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.config import AnalysisMode, load_config  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.common.types import ObjectClass  # noqa: E402
from visionpitch.detection.yolo import build_detector  # noqa: E402
from visionpitch.evaluation.datasets import GSRDataset, validate_ground_truth  # noqa: E402
from visionpitch.evaluation.detection import evaluate_detection  # noqa: E402
from visionpitch.evaluation.tracking import evaluate_tracking  # noqa: E402
from visionpitch.tracking.postprocess import clean_tracks  # noqa: E402
from visionpitch.tracking.tracker import MultiObjectTracker  # noqa: E402

log = get_logger("benchmark.tracking")


def run_sequence(sequence, config, max_frames: int | None):
    """Detect and track one GSR sequence. Returns (detections, tracks)."""
    import cv2

    detector = build_detector(config)
    tracker = MultiObjectTracker(config)

    frames = sorted(sequence.image_paths)
    if max_frames:
        frames = frames[:max_frames]

    detections: dict[int, list] = {}
    fps = sequence.ground_truth.fps or 25.0
    batch_size = max(1, config.runtime.batch_size)

    for start in range(0, len(frames), batch_size):
        chunk = frames[start : start + batch_size]
        images = [cv2.imread(str(sequence.image_paths[f])) for f in chunk]
        usable = [(f, im) for f, im in zip(chunk, images, strict=True) if im is not None]
        if not usable:
            continue
        indices = [f for f, _ in usable]
        batch = [im for _, im in usable]

        for frame_idx, image, dets in zip(
            indices, batch, detector.detect_batch(batch, indices), strict=True
        ):
            detections[frame_idx] = dets
            tracker.update(dets, frame_idx, frame_idx / fps, image)

    raw = tracker.finalise()
    cleaned, _, report = clean_tracks(raw, config.tracking)
    return detections, cleaned, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/eval/gsr"))
    parser.add_argument("--sequences", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--mode", default="balanced")
    parser.add_argument("--out", type=Path, default=Path("data/eval/gsr/benchmarks"))
    args = parser.parse_args()

    configure_logging("WARNING")
    config = load_config(mode=AnalysisMode(args.mode))
    dataset = GSRDataset(args.root, max_sequences=args.sequences)
    info = dataset.info()

    print(f"corpus : {info.name}  ({info.kind})")
    print(f"seqs   : {len(dataset.sequences)}  frames/seq capped at {args.max_frames}\n")

    per_sequence = []
    for sequence in dataset.sequences:
        started = time.perf_counter()
        detections, tracks, clean_report = run_sequence(sequence, config, args.max_frames)

        gt = sequence.ground_truth
        # Restrict ground truth to the frames actually processed, otherwise
        # every unprocessed frame counts as a total miss.
        processed = set(detections)
        gt.frames = {f: objs for f, objs in gt.frames.items() if f in processed}
        if not gt.frames:
            print(f"  {sequence.name}: no overlapping annotated frames, skipped")
            continue

        validation = validate_ground_truth(gt, require_identity=True)
        detection_metrics = evaluate_detection(gt, detections)
        tracking_metrics = evaluate_tracking(
            gt, tracks,
            classes=(ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE),
        )

        elapsed = time.perf_counter() - started
        entry = {
            "sequence": sequence.name,
            "frames": len(processed),
            "gt_objects": gt.n_objects,
            "validation": validation["issue_counts"],
            "detection": detection_metrics["per_class"],
            "detection_overall": detection_metrics["overall"],
            "tracking": tracking_metrics,
            "cleaning": clean_report,
            "seconds": round(elapsed, 1),
        }
        per_sequence.append(entry)

        print(f"  {sequence.name}: {len(processed)} frames in {elapsed:.0f}s")
        print(f"    HOTA {tracking_metrics['HOTA']}  DetA {tracking_metrics['DetA']}  "
              f"AssA {tracking_metrics['AssA']}")
        print(f"    IDF1 {tracking_metrics['IDF1']}  MOTA {tracking_metrics['MOTA']}  "
              f"IDsw {tracking_metrics['id_switches']}  frag {tracking_metrics['fragmentations']}")
        player = detection_metrics["per_class"].get("player", {})
        print(f"    player P/R {player.get('precision')}/{player.get('recall')}  "
              f"mAP50 {player.get('mAP50')}\n")

    if not per_sequence:
        print("no sequences produced metrics")
        return 1

    def mean(path):
        values = []
        for entry in per_sequence:
            node = entry
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, (int, float)):
                values.append(float(node))
        return round(float(np.mean(values)), 4) if values else None

    aggregate = {
        "corpus": info.to_dict(),
        "mode": args.mode,
        "n_sequences": len(per_sequence),
        "total_frames": sum(e["frames"] for e in per_sequence),
        "tracking": {
            "HOTA": mean(["tracking", "HOTA"]),
            "DetA": mean(["tracking", "DetA"]),
            "AssA": mean(["tracking", "AssA"]),
            "IDF1": mean(["tracking", "IDF1"]),
            "MOTA": mean(["tracking", "MOTA"]),
            "id_switches": sum(e["tracking"]["id_switches"] for e in per_sequence),
            "fragmentations": sum(e["tracking"]["fragmentations"] for e in per_sequence),
            "mostly_tracked": sum(e["tracking"]["mostly_tracked"] for e in per_sequence),
            "mostly_lost": sum(e["tracking"]["mostly_lost"] for e in per_sequence),
        },
        "detection": {
            "player_precision": mean(["detection", "player", "precision"]),
            "player_recall": mean(["detection", "player", "recall"]),
            "player_mAP50": mean(["detection", "player", "mAP50"]),
            "ball_recall": mean(["detection", "ball", "recall"]),
            "mAP50_all": mean(["detection_overall", "mAP50_all_classes"]),
        },
        "per_sequence": per_sequence,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"tracking_{args.mode}.json"
    path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("=" * 62)
    print("AGGREGATE (out-of-distribution)")
    for key, value in aggregate["tracking"].items():
        print(f"  {key:16s} {value}")
    for key, value in aggregate["detection"].items():
        print(f"  {key:16s} {value}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
