"""Audit Candidate C's false positives on the locked data.

Part 1 of precision hardening.

Two passes, kept apart because they answer different questions:

* **still splits** (public val/test, local val/test) give the spatial taxonomy --
  what the detector fires on
* **a video sequence** gives temporal persistence -- whether a false positive is
  a one-frame flicker or a stable false track, and whether the trajectory
  estimator absorbed it

Auditing the locked test is allowed and necessary; *training* on it is not. This
script only reads, and records which split every false positive came from so the
mining step can exclude test-derived ones.

Usage::

    python scripts/audit_false_positives.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.annotation.schema import AnnotationStore, BallVisibility  # noqa: E402
from visionpitch.annotation.splits import LocalSplit  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.false_positives import (  # noqa: E402
    FalsePositive,
    FalsePositiveKind,
    classify,
    mark_duplicates,
    measure,
    summarise,
)

log = get_logger("fp.audit")

CANDIDATE_C = "models/finetune/bcast_adapt/weights/best.pt"
MULTICORPUS = Path("data/ball_multicorpus")
LOCAL_PACKAGE = Path("data/annotation/package")
LOCAL_SPLIT = Path("data/annotation/local_split.json")
MATCH_PX = 25.0


def public_frames(split: str):
    for path in sorted((MULTICORPUS / split / "images").glob("*.jpg")):
        label = path.parent.parent / "labels" / f"{path.stem}.txt"
        balls = []
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    balls.append((cx * w, cy * h, bw * w, bh * h))
        domain = (
            "soccernet_gsr" if path.name.startswith("soccernet_gsr_") else "roboflow"
        )
        yield path.stem, domain, image, balls


def local_frames(split_name: str):
    store = AnnotationStore(LOCAL_PACKAGE)
    samples = store.load_samples()
    annotations = store.load_annotations()
    split = LocalSplit.load(LOCAL_SPLIT)
    for frame_id in sorted(split.frames.get(split_name, [])):
        annotation = annotations[frame_id]
        image = cv2.imread(samples[frame_id].image_path)
        if image is None:
            continue
        balls = []
        if annotation.visibility is BallVisibility.VISIBLE and annotation.radius_px:
            d = annotation.radius_px * 2
            balls.append((annotation.centre_x, annotation.centre_y, d, d))
        yield frame_id, "local_broadcast", image, balls


def audit_stills(model, person_detector, frames, split_label: str, conf: float,
                 imgsz: int) -> tuple[list[FalsePositive], int]:
    from visionpitch.common.types import ObjectClass

    items: list[FalsePositive] = []
    n_frames = 0
    for index, (frame_id, domain, image, balls) in enumerate(frames):
        n_frames += 1
        if index and index % 250 == 0:
            log.info("  %s %d frames", split_label, index)
        result = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
        if result.boxes is None or not len(result.boxes):
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        centres = [
            (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2), float(s))
            for b, s in zip(boxes, scores, strict=True)
        ]
        duplicates = mark_duplicates(centres)

        # Greedy match against truth; whatever is left over is a false positive.
        matched: set[int] = set()
        for cx, cy, _, _ in balls:
            best, slot = MATCH_PX, None
            for j, (px, py, _) in enumerate(centres):
                if j in matched:
                    continue
                d = float(np.hypot(px - cx, py - cy))
                if d <= best:
                    best, slot = d, j
            if slot is not None:
                matched.add(slot)

        people = person_detector.detect_batch([image], [index])[0]
        person_boxes = np.array([
            [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in people
            if d.object_class in (
                ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE
            )
        ]) if people else np.zeros((0, 4))

        for j, (px, py, score) in enumerate(centres):
            if j in matched:
                continue
            box = boxes[j]
            width, height = float(box[2] - box[0]), float(box[3] - box[1])
            evidence = measure(image, px, py, width, height, person_boxes)
            kind = (
                FalsePositiveKind.DUPLICATE_CANDIDATE if duplicates[j]
                else classify(evidence, width, height)
            )
            items.append(FalsePositive(
                frame_id=f"{split_label}:{frame_id}", domain=domain,
                x=px, y=py, width=width, height=height, confidence=score,
                kind=kind, **evidence,
            ))
    return items, n_frames


def audit_sequence(model, person_detector, video: Path, start: float, end: float,
                   conf: float, imgsz: int, stride: int) -> dict:
    """Temporal persistence of false positives on real footage.

    There is no ball ground truth here, so a *candidate* is anything the
    detector emits. What this measures is how long a candidate at a given
    location survives -- one frame, or a stable false track. That distinction
    decides whether a temporal filter can help at all.
    """
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 50.0
    first, last = int(start * fps), int(end * fps)

    tracks: list[dict] = []
    frame_idx = 0
    n_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx > last:
            break
        if frame_idx < first or (frame_idx - first) % stride:
            frame_idx += 1
            continue
        n_frames += 1
        result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        centres = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            centres = [
                (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2), float(s))
                for b, s in zip(boxes, scores, strict=True)
            ]
        # A candidate continues a track if it lands near the previous sighting.
        for cx, cy, score in centres:
            joined = False
            for track in tracks:
                if track["last_frame"] >= frame_idx - stride * 2 and float(
                    np.hypot(cx - track["x"], cy - track["y"])
                ) <= 40.0:
                    track["x"], track["y"] = cx, cy
                    track["last_frame"] = frame_idx
                    track["length"] += 1
                    track["confidences"].append(score)
                    joined = True
                    break
            if not joined:
                tracks.append({
                    "x": cx, "y": cy, "first_frame": frame_idx,
                    "last_frame": frame_idx, "length": 1, "confidences": [score],
                })
        frame_idx += 1
    capture.release()

    lengths = np.array([t["length"] for t in tracks]) if tracks else np.array([])
    return {
        "video": str(video),
        "segment_s": [start, end],
        "stride": stride,
        "n_frames_sampled": n_frames,
        "n_candidate_tracks": len(tracks),
        "candidates_per_frame": round(
            float(sum(t["length"] for t in tracks) / max(1, n_frames)), 4
        ),
        "track_length_frames": {
            "median": float(np.median(lengths)) if lengths.size else None,
            "share_single_frame": (
                round(float((lengths <= 1).mean()), 4) if lengths.size else None
            ),
            "share_ge_5": round(float((lengths >= 5).mean()), 4) if lengths.size else None,
        },
        "note": (
            "no ball ground truth on this clip, so these are candidate tracks, "
            "not confirmed false tracks. The split between one-frame flickers "
            "and persistent tracks is what a temporal filter can act on."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=CANDIDATE_C)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--sequence-stride", type=int, default=5)
    parser.add_argument("--skip-sequence", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=Path("data/eval/hardening/fp_audit.json"))
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    from visionpitch.common.config import load_config
    from visionpitch.detection.yolo import build_detector

    model = YOLO(args.weights)
    person_detector = build_detector(load_config())

    per_split: dict[str, dict] = {}
    all_items: list[FalsePositive] = []
    for label, frames in (
        ("public_val", public_frames("val")),
        ("public_test", public_frames("test")),
        ("local_val", local_frames("val")),
        ("local_test", local_frames("test")),
    ):
        log.info("auditing %s", label)
        items, n_frames = audit_stills(
            model, person_detector, frames, label, args.conf, args.imgsz
        )
        per_split[label] = summarise(items, n_frames)
        per_split[label]["minable"] = label not in ("public_test", "local_test")
        all_items.extend(items)
        log.info("  %s: %d false positive(s) over %d frame(s)",
                 label, len(items), n_frames)

    sequence = None
    if not args.skip_sequence:
        video = Path(
            "ملخص مباراة نيوزيلندا ومصر _ دور المجموعات - كأس العالم FIFA 2026™.mp4"
        )
        if video.exists():
            log.info("auditing temporal persistence on the local clip")
            sequence = audit_sequence(
                model, person_detector, video, 0.0, 120.0,
                args.conf, args.imgsz, args.sequence_stride,
            )

    payload = {
        "weights": args.weights,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "match_px": MATCH_PX,
        "per_split": per_split,
        "pooled": summarise(
            all_items, sum(v["n_frames"] for v in per_split.values())
        ),
        "temporal": sequence,
        "mining_policy": (
            "only public_val, local_val and the training splits may contribute "
            "hard negatives; public_test and local_test are audited but never "
            "mined"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Per-item records, so mining can filter by split without re-running.
    items_path = args.out.parent / "fp_items.json"
    items_path.write_text(
        json.dumps([item.to_dict() for item in all_items], indent=2), encoding="utf-8"
    )

    pooled = payload["pooled"]
    print(f"\npooled: {pooled['n_false_positives']} false positive(s) over "
          f"{pooled['n_frames']} frames ({pooled['false_positives_per_frame']}/frame)")
    print(f"\n{'kind':<32}{'n':>6}{'%':>8}{'/frame':>9}{'medConf':>9}{'minable':>9}")
    for kind, block in pooled["by_kind"].items():
        print(f"  {kind:<30}{block['count']:>6}{block['pct_of_false_positives']:>8.3f}"
              f"{block['per_frame']:>9.4f}{block['confidence']['median']:>9.3f}"
              f"{str(block['minable_as_hard_negative']):>9}")
    print(f"\n{'split':<14}{'frames':>8}{'FPs':>7}{'/frame':>9}  minable")
    for label, block in per_split.items():
        print(f"  {label:<12}{block['n_frames']:>8}{block['n_false_positives']:>7}"
              f"{block.get('false_positives_per_frame', 0):>9.4f}  {block['minable']}")
    if sequence:
        t = sequence["track_length_frames"]
        print(f"\ntemporal: {sequence['n_candidate_tracks']} candidate track(s) over "
              f"{sequence['n_frames_sampled']} sampled frames")
        print(f"  single-frame share {t['share_single_frame']}, "
              f">=5 frames share {t['share_ge_5']}, median {t['median']}")
    print(f"\nwrote {args.out} and {items_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
