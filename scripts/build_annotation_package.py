"""Build the human review package: sampled frames, images, and model proposals.

Broadcast ball annotation workflow, steps 2 and 3.

Three passes over the video:

1. **Scan** every frame for the cheap signals the strata are built from.
2. **Probe** a strided candidate pool with the person detector and both ball
   detectors, to measure detector disagreement and crowd density. Running both
   detectors on all 26k frames would cost far more than it is worth; the pool is
   what the model-steered stratum is drawn from.
3. **Extract** the selected frames plus a short context window each, and record
   both models' proposals for every selected frame.

Proposals are written to a separate file from annotations and are never merged.
They exist to save clicks, not to answer the question.

Usage::

    python scripts/build_annotation_package.py --video "<path>.mp4" --frames 400
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.annotation.broadcast_audit import (  # noqa: E402
    AuditResult,
    ShotType,
    scan_frames,
)
from visionpitch.annotation.sampler import (  # noqa: E402
    FrameSignal,
    SamplingPlan,
    build_samples,
)
from visionpitch.annotation.schema import (  # noqa: E402
    ANNOTATION_SCHEMA_VERSION,
    AnnotationStore,
    ModelPrediction,
)
from visionpitch.common.config import load_config  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.common.types import ObjectClass  # noqa: E402

log = get_logger("annotation.package")

BOX_LABEL = "box_detector"
HEATMAP_LABEL = "heatmap_detector"
#: Centre distance beyond which the two detectors are treated as disagreeing.
#: 25 px is the operating tolerance used throughout this project.
DISAGREEMENT_PX = 25.0


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def box_predict(model, frame, conf: float, imgsz: int):
    result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    if result.boxes is None or not len(result.boxes):
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    best = int(np.argmax(scores))
    b = boxes[best]
    return {
        "centre": (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)),
        "bbox": [float(v) for v in b[:4]],
        "confidence": float(scores[best]),
    }


def heatmap_predict(detector, frame, frame_idx: int):
    detections = detector.detect(frame, frame_idx)
    if not detections:
        return None
    best = max(detections, key=lambda d: d.confidence)
    return {
        "centre": (
            (best.bbox.x1 + best.bbox.x2) / 2, (best.bbox.y1 + best.bbox.y2) / 2
        ),
        "bbox": [best.bbox.x1, best.bbox.y1, best.bbox.x2, best.bbox.y2],
        "confidence": float(best.confidence),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--audit", type=Path, default=Path("data/annotation/broadcast_audit.json"))
    parser.add_argument("--out", type=Path, default=Path("data/annotation/package"))
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--candidate-stride", type=int, default=10)
    parser.add_argument("--context", type=int, default=2,
                        help="context frames extracted either side of each sample")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--heatmap-checkpoint",
                        default="models/finetune/heatmap/best.pt")
    parser.add_argument("--box-conf", type=float, default=0.08)
    parser.add_argument("--heatmap-conf", type=float, default=0.40)
    args = parser.parse_args()

    configure_logging("INFO")
    video = Path(args.video)
    if not video.exists():
        log.error("no video at %s", video)
        return 1
    if not args.audit.exists():
        log.error("no audit at %s; run scripts/broadcast_audit.py first", args.audit)
        return 1

    audit = AuditResult.load(args.audit)
    log.info(
        "audit: %d shots, live-play share %.3f", len(audit.shots), audit.live_play_share
    )

    # -- pass 1: cheap signals ------------------------------------------------- #
    log.info("pass 1/3: scanning frames")
    features = scan_frames(video)
    shot_of: dict[int, tuple[int, ShotType, bool]] = {}
    for shot in audit.shots:
        for idx in range(shot.start_frame, shot.end_frame + 1):
            shot_of[idx] = (shot.index, shot.shot_type, shot.likely_slow_motion)

    signals: dict[int, FrameSignal] = {}
    for feature in features:
        shot_index, shot_type, slow = shot_of.get(
            feature.frame_idx, (-1, ShotType.UNKNOWN, False)
        )
        signals[feature.frame_idx] = FrameSignal(
            frame_idx=feature.frame_idx,
            motion_px=feature.motion_px,
            blur_variance=feature.blur_variance,
            edge_density=feature.edge_density,
            saturation=feature.mean_saturation,
            shot_index=shot_index,
            shot_type=shot_type,
            likely_slow_motion=slow,
        )

    # -- pass 2: probe a candidate pool with the models ------------------------ #
    pool = sorted(i for i in signals if i % args.candidate_stride == 0)
    log.info("pass 2/3: probing %d candidate frame(s) with both detectors", len(pool))

    config = load_config()
    config.ball_detection.conf_threshold = args.heatmap_conf
    from ultralytics import YOLO

    from visionpitch.detection.heatmap_detector import HeatmapBallDetector
    from visionpitch.detection.yolo import build_detector

    person_detector = build_detector(config)
    box_model = YOLO(config.ball_detection.model_path)
    box_fingerprint = file_fingerprint(Path(config.ball_detection.model_path))

    heatmap_detector = None
    heatmap_fingerprint = ""
    checkpoint = Path(args.heatmap_checkpoint)
    if checkpoint.exists():
        heatmap_detector = HeatmapBallDetector(config, checkpoint)
        heatmap_fingerprint = file_fingerprint(checkpoint)
    else:
        log.warning("no heatmap checkpoint at %s; only one proposal per frame", checkpoint)

    pool_set = set(pool)
    disagreements: dict[int, float] = {}
    goal_area: set[int] = set()
    probe: dict[int, dict] = {}

    capture = cv2.VideoCapture(str(video))
    frame_idx = 0
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx in pool_set:
            detections = person_detector.detect_batch([frame], [frame_idx])[0]
            people = [
                d for d in detections
                if d.object_class in (
                    ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE
                )
            ]
            signals[frame_idx].n_people = len(people)

            box = box_predict(box_model, frame, args.box_conf, config.ball_detection.imgsz)
            heat = (
                heatmap_predict(heatmap_detector, frame, frame_idx)
                if heatmap_detector else None
            )
            probe[frame_idx] = {"box": box, "heatmap": heat}
            if box and heat:
                distance = float(np.hypot(
                    box["centre"][0] - heat["centre"][0],
                    box["centre"][1] - heat["centre"][1],
                ))
                if distance > DISAGREEMENT_PX:
                    disagreements[frame_idx] = distance
            elif bool(box) != bool(heat):
                # One found a ball and the other did not: maximal disagreement.
                disagreements[frame_idx] = 1e4

            # Goal-area proxy: a goalkeeper in shot is the cheapest reliable
            # signal that play is near a penalty area.
            if any(d.object_class is ObjectClass.GOALKEEPER for d in detections):
                goal_area.add(frame_idx)

            if len(probe) % 400 == 0:
                log.info("  probed %d/%d", len(probe), len(pool))
        frame_idx += 1
    capture.release()
    log.info(
        "probe complete in %.1fs: %d disagreement(s), %d goal-area frame(s)",
        time.perf_counter() - started, len(disagreements), len(goal_area),
    )

    # -- sample ---------------------------------------------------------------- #
    plan = SamplingPlan(total_frames=args.frames, seed=args.seed)
    result = build_samples(
        audit=audit, signals=list(signals.values()), plan=plan,
        disagreements=disagreements, goal_area_frames=goal_area,
    )
    selected = {s.frame_idx: s for s in result.samples}
    log.info("selected %d frame(s)", len(selected))

    # -- pass 3: extract images and record proposals --------------------------- #
    out = args.out
    images_dir = out / "images"
    context_dir = images_dir / "context"
    images_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    context_wanted: dict[int, list[tuple[str, int]]] = {}
    for sample in result.samples:
        for offset in range(-args.context, args.context + 1):
            if offset == 0:
                continue
            neighbour = sample.frame_idx + offset
            if 0 <= neighbour < audit.frame_count:
                context_wanted.setdefault(neighbour, []).append(
                    (sample.frame_id, offset)
                )

    log.info("pass 3/3: extracting %d frame(s) + %d context frame(s)",
             len(selected), len(context_wanted))
    predictions: list[ModelPrediction] = []
    capture = cv2.VideoCapture(str(video))
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_idx in selected:
            sample = selected[frame_idx]
            path = images_dir / f"{sample.frame_id}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            sample.image_path = str(path.resolve())

            cached = probe.get(frame_idx)
            box = cached["box"] if cached else box_predict(
                box_model, frame, args.box_conf, config.ball_detection.imgsz
            )
            heat = (
                cached["heatmap"] if cached
                else (heatmap_predict(heatmap_detector, frame, frame_idx)
                      if heatmap_detector else None)
            )
            predictions.append(ModelPrediction(
                frame_id=sample.frame_id, model_label=BOX_LABEL,
                model_fingerprint=box_fingerprint,
                centre_x=box["centre"][0] if box else None,
                centre_y=box["centre"][1] if box else None,
                confidence=box["confidence"] if box else 0.0,
                bbox=box["bbox"] if box else None,
            ))
            if heatmap_detector is not None:
                predictions.append(ModelPrediction(
                    frame_id=sample.frame_id, model_label=HEATMAP_LABEL,
                    model_fingerprint=heatmap_fingerprint,
                    centre_x=heat["centre"][0] if heat else None,
                    centre_y=heat["centre"][1] if heat else None,
                    confidence=heat["confidence"] if heat else 0.0,
                    bbox=heat["bbox"] if heat else None,
                ))

        for frame_id, offset in context_wanted.get(frame_idx, []):
            cv2.imwrite(
                str(context_dir / f"{frame_id}{offset:+d}.jpg"), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 82],
            )
        frame_idx += 1
    capture.release()

    # -- write the package ----------------------------------------------------- #
    store = AnnotationStore(out)
    store.write_samples(result.samples)
    store.write_predictions(predictions)

    n_box = sum(
        1 for p in predictions
        if p.model_label == BOX_LABEL and p.centre_x is not None
    )
    n_heat = sum(
        1 for p in predictions
        if p.model_label == HEATMAP_LABEL and p.centre_x is not None
    )
    selected_disagreements = sum(
        1 for s in result.samples if s.frame_idx in disagreements
    )

    manifest = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "source_video": str(video.resolve()),
        "source_content_hash": audit.content_hash,
        "video": {
            "width": audit.width, "height": audit.height, "fps": audit.fps,
            "frame_count": audit.frame_count, "duration_s": audit.duration_s,
            "codec": audit.codec,
        },
        "audit_summary": {
            "n_shots": len(audit.shots),
            "shot_type_breakdown": audit.by_type(),
            "live_play_share": round(audit.live_play_share, 4),
            "likely_slow_motion_share": round(audit.slow_motion_share, 4),
        },
        "sampling": result.to_dict(),
        "context_frames_each_side": args.context,
        "candidate_stride": args.candidate_stride,
        "models": {
            BOX_LABEL: {
                "weights": config.ball_detection.model_path,
                "fingerprint": box_fingerprint,
                "conf": args.box_conf,
                "n_frames_with_proposal": n_box,
            },
            HEATMAP_LABEL: {
                "weights": str(checkpoint),
                "fingerprint": heatmap_fingerprint,
                "conf": args.heatmap_conf,
                "n_frames_with_proposal": n_heat,
            },
        },
        "disagreements": {
            "threshold_px": DISAGREEMENT_PX,
            "n_in_candidate_pool": len(disagreements),
            "n_in_selected_sample": selected_disagreements,
        },
        "note": (
            "predictions.jsonl holds MODEL PROPOSALS ONLY. Ground truth lives in "
            "annotations.jsonl and is written exclusively by a human reviewer. "
            "Neither file is ever merged into the other."
        ),
    }
    store.write_manifest(manifest)

    print(f"\npackage: {out.resolve()}")
    print(f"frames sampled      : {len(result.samples)}")
    print(f"sampling fingerprint: {result.fingerprint()}")
    print(f"\n{'category':<24}{'n':>6}")
    for name, count in result.to_dict()["by_category"].items():
        print(f"  {name:<22}{count:>6}")
    print(f"\nproposals: {BOX_LABEL} {n_box}/{len(result.samples)}, "
          f"{HEATMAP_LABEL} {n_heat}/{len(result.samples)}")
    print(f"disagreements in sample: {selected_disagreements}")
    print(f"context frames: {len(list(context_dir.glob('*.jpg')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
