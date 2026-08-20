"""Candidate-level perception under legacy vs temporal fusion, real GMC.

Feeds gates G1 (local recall), G2 (false-positive reduction) and G9 (public
recall) of the pinned promotion specification. Nothing here is a pipeline run --
this measures what fusion does to the *candidates*, which the downstream
evaluation cannot separate out.

Two things that had to change from the isolated ablation
--------------------------------------------------------
**Real GMC.** The isolated ablation estimated camera motion with phase
correlation on a 320x180 downscale. Part 2 forbids that substitute in the final
comparison, so this runs the production
:class:`~visionpitch.tracking.gmc.GlobalMotionCompensator` -- masked by real
person detections, exactly as the tracker does.

**The local test needs sequences.** The locked local test is 23 *isolated*
frames, 200+ frames apart. Temporal verification cannot be applied to isolated
frames at all, so each test frame is scored inside a window of +-``--window``
frames decoded from the source video. The annotation, the split and the frame
identities are untouched; only the temporal context around them is
reconstructed.

That reconstruction has a consequence worth stating rather than burying: a
production run sees an unbounded history, so a 13-frame window gives temporal
verification *less* evidence than production has. Local recall measured this way
is a **lower bound** on the production figure, not an estimate of it.

Usage::

    python scripts/fusion_perception_eval.py
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
from visionpitch.ball_tracking.fp_filter import FilterConfig  # noqa: E402
from visionpitch.ball_tracking.fusion import (  # noqa: E402
    BallFusion,
    FusionConfig,
    SuppressionMethod,
)
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.fusion_rows import CONF_THRESHOLD, DETECTORS  # noqa: E402
from visionpitch.evaluation.possession_gt import load_gsr_gamestate  # noqa: E402
from visionpitch.evaluation.registry import build_split  # noqa: E402
from visionpitch.tracking.gmc import GlobalMotionCompensator  # noqa: E402

log = get_logger("fusion.perception")

GSR_ROOT = Path("data/eval/gsr")
LOCAL_VIDEO = Path(
    "ملخص مباراة نيوزيلندا ومصر _ دور المجموعات - كأس العالم FIFA 2026™.mp4"
)
LOCAL_PACKAGE = Path("data/annotation/package")
LOCAL_SPLIT = Path("data/annotation/local_split.json")
OUT = Path("data/eval/fusion/perception.json")

MATCH_PX = 25.0
IMGSZ = 960
#: Ball bbox area below this is "tiny". Same cut as the broadcast comparison.
TINY_AREA = 150.0
#: A ball whose centre is inside a person box, or within this of one.
OCCLUSION_MARGIN = 6.0

PERSON_MODEL = "models/yolo-football-player-detection.pt"


def engines() -> dict[str, FusionConfig]:
    """Exactly two configurations, matching the frozen four-way rows.

    ``legacy`` reproduces production's IoU de-duplication at 0.5 with no
    temporal verification, so the comparison isolates the change under test
    rather than the difference between two code paths.
    """
    permissive = FilterConfig(
        min_support_frames=0, trust_confidence=0.0, max_step_px=1e9,
        camera_motion_px=1e9, max_size_ratio=1e9,
    )
    return {
        "legacy": FusionConfig(
            suppression=SuppressionMethod.IOU, iou_threshold=0.5,
            temporal=permissive, max_candidates_per_frame=4,
        ),
        "temporal": FusionConfig(
            suppression=SuppressionMethod.WEIGHTED_CENTRE, merge_radius_px=22.0,
            temporal=FilterConfig(
                min_support_frames=2, trust_confidence=0.75, camera_motion_px=6.0
            ),
            min_camera_confidence=0.35,
        ),
    }


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    spread = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4)]


def detect_balls(model, image) -> list[tuple[float, float, float, float]]:
    result = model.predict(image, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)[0]
    found = []
    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for box, score in zip(boxes, scores, strict=True):
            found.append((
                float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2),
                float(max(box[2] - box[0], box[3] - box[1]) / 2), float(score),
            ))
    return found


def person_boxes(model, image) -> np.ndarray:
    if model is None:
        return np.zeros((0, 4), dtype=np.float32)
    result = model.predict(image, imgsz=IMGSZ, conf=0.25, verbose=False)[0]
    if result.boxes is None or not len(result.boxes):
        return np.zeros((0, 4), dtype=np.float32)
    return result.boxes.xyxy.cpu().numpy().astype(np.float32)


def real_gmc(images: list[np.ndarray], boxes: list[np.ndarray],
             frame_ids: list[int]) -> tuple[dict, dict]:
    """Production GMC over a decoded sequence, masked by real detections."""
    compensator = GlobalMotionCompensator(method="sparseOptFlow", downscale=2)
    shifts: dict[int, tuple[float, float]] = {}
    confidence: dict[int, float] = {}
    identity = np.eye(2, 3, dtype=np.float64)
    for image, person, frame_id in zip(images, boxes, frame_ids, strict=True):
        warp = compensator.apply(image, person if len(person) else None)
        h, w = image.shape[:2]
        centre = np.array([w / 2.0, h / 2.0, 1.0])
        moved = np.asarray(warp, dtype=np.float64) @ centre
        shifts[frame_id] = (float(moved[0] - centre[0]), float(moved[1] - centre[1]))
        confidence[frame_id] = (
            0.0 if np.allclose(warp, identity, atol=1e-9) else 1.0
        )
    return shifts, confidence


# --------------------------------------------------------------------------- #
# Public: SN-GSR held-out sequences
# --------------------------------------------------------------------------- #


def collect_public(ball_model, person_model, sequences, max_frames: int) -> list[dict]:
    data = []
    for labels_path in sequences:
        frames_meta, _ = load_gsr_gamestate(labels_path)
        image_dir = labels_path.parent / "img1"
        if not image_dir.exists():
            continue
        indices = sorted(frames_meta)[:max_frames]

        kept, images, boxes = [], [], []
        truth: dict[int, tuple[float, float]] = {}
        tiny: set[int] = set()
        occluded: set[int] = set()
        detections: dict[int, list] = {}

        for frame_idx in indices:
            path = image_dir / f"{frame_idx:06d}.jpg"
            image = cv2.imread(str(path)) if path.exists() else None
            if image is None:
                continue
            kept.append(frame_idx)
            images.append(image)

            ball = next(
                (o for o in frames_meta[frame_idx] if o.role == "ball"), None
            )
            if ball is not None:
                bx, by = ball.image_x, ball.image_y - ball.box_height / 2
                truth[frame_idx] = (bx, by)
                side = max(1.0, ball.box_height)
                if side * side < TINY_AREA:
                    tiny.add(frame_idx)

            people = person_boxes(person_model, image)
            boxes.append(people)
            if ball is not None and len(people):
                bx, by = truth[frame_idx]
                inside = (
                    (people[:, 0] - OCCLUSION_MARGIN <= bx)
                    & (bx <= people[:, 2] + OCCLUSION_MARGIN)
                    & (people[:, 1] - OCCLUSION_MARGIN <= by)
                    & (by <= people[:, 3] + OCCLUSION_MARGIN)
                )
                if bool(inside.any()):
                    occluded.add(frame_idx)

            detections[frame_idx] = detect_balls(ball_model, image)

        shifts, confidence = real_gmc(images, boxes, kept)
        data.append({
            "sequence": labels_path.parent.name, "frame_indices": kept,
            "truth": truth, "detections": detections, "shifts": shifts,
            "camera_confidence": confidence, "tiny": tiny, "occluded": occluded,
        })
        log.info("  %s: %d frames, %d with ball (%d tiny, %d occluded)",
                 labels_path.parent.name, len(kept), len(truth),
                 len(tiny), len(occluded))
    return data


def score_public(data, config: FusionConfig) -> dict:
    fusion = BallFusion(config)
    tp = fp = fn = 0
    n_frames = n_negative = fp_on_negative = 0
    tiny_tp = tiny_n = occ_tp = occ_n = 0

    for entry in data:
        frames = fusion.run(
            entry["detections"], entry["frame_indices"],
            camera_shifts=entry["shifts"],
            camera_confidence=entry["camera_confidence"],
        )
        for frame_idx in entry["frame_indices"]:
            frame = frames[frame_idx]
            truth = entry["truth"].get(frame_idx)
            n_frames += 1
            if truth is None:
                n_negative += 1
            if frame.kind.counts_as_observed:
                if truth is None:
                    fp += 1
                    fp_on_negative += 1
                elif float(np.hypot(frame.x - truth[0], frame.y - truth[1])) <= MATCH_PX:
                    tp += 1
                    if frame_idx in entry["tiny"]:
                        tiny_tp += 1
                    if frame_idx in entry["occluded"]:
                        occ_tp += 1
                else:
                    fp += 1
                    fn += 1
            elif truth is not None:
                fn += 1
            if truth is not None:
                tiny_n += int(frame_idx in entry["tiny"])
                occ_n += int(frame_idx in entry["occluded"])

    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    return {
        "config_fingerprint": config.fingerprint(),
        "n_frames": n_frames,
        "n_positive_frames": tp + fn,
        "n_negative_frames": n_negative,
        "public_centre_recall_25px": round(recall, 4),
        "public_centre_recall_ci95": wilson(tp, tp + fn),
        "public_precision": round(precision, 4),
        "public_fp_total": fp,
        "public_fp_per_frame": round(fp / max(1, n_frames), 4),
        "public_fp_per_negative_frame": round(
            fp_on_negative / max(1, n_negative), 4
        ),
        "tiny_ball_recall": round(tiny_tp / max(1, tiny_n), 4) if tiny_n else None,
        "tiny_ball_n": tiny_n,
        "occluded_ball_recall": round(occ_tp / max(1, occ_n), 4) if occ_n else None,
        "occluded_ball_n": occ_n,
    }


# --------------------------------------------------------------------------- #
# Local: locked test frames, temporal window decoded from the source video
# --------------------------------------------------------------------------- #


def collect_local(ball_model, person_model, window: int) -> list[dict]:
    split = json.loads(LOCAL_SPLIT.read_text(encoding="utf-8"))
    test_ids = set(split["frames"]["test"])
    store = AnnotationStore(LOCAL_PACKAGE / "annotations.jsonl")
    samples = {
        json.loads(line)["frame_id"]: json.loads(line)
        for line in (LOCAL_PACKAGE / "samples.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    }
    annotations = {a.frame_id: a for a in store.load()}

    capture = cv2.VideoCapture(str(LOCAL_VIDEO))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {LOCAL_VIDEO}")

    entries = []
    for frame_id in sorted(test_ids):
        annotation = annotations.get(frame_id)
        sample = samples.get(frame_id)
        if annotation is None or sample is None or not annotation.is_scorable:
            continue
        centre = sample["frame_idx"]
        first = max(0, centre - window)
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)

        kept, images, boxes = [], [], []
        detections: dict[int, list] = {}
        for offset in range(first, centre + window + 1):
            ok, image = capture.read()
            if not ok:
                break
            kept.append(offset)
            images.append(image)
            boxes.append(person_boxes(person_model, image))
            detections[offset] = detect_balls(ball_model, image)

        if centre not in kept:
            log.warning("%s: centre frame %d not decoded", frame_id, centre)
            continue
        shifts, confidence = real_gmc(images, boxes, kept)

        visible = annotation.visibility is BallVisibility.VISIBLE
        entries.append({
            "frame_id": frame_id, "centre_frame": centre, "frame_indices": kept,
            "detections": detections, "shifts": shifts,
            "camera_confidence": confidence,
            "truth": (
                (annotation.centre_x, annotation.centre_y) if visible else None
            ),
            "radius_px": annotation.radius_px,
        })
    capture.release()
    log.info("local: %d scorable test frames, window +-%d", len(entries), window)
    return entries


def score_local(entries, config: FusionConfig) -> dict:
    fusion = BallFusion(config)
    tp = fn = 0
    tiny_tp = tiny_n = 0

    for entry in entries:
        frames = fusion.run(
            entry["detections"], entry["frame_indices"],
            camera_shifts=entry["shifts"],
            camera_confidence=entry["camera_confidence"],
        )
        frame = frames[entry["centre_frame"]]
        truth = entry["truth"]
        if truth is None:
            continue
        radius = entry["radius_px"] or 7.0
        is_tiny = (2 * radius) ** 2 < TINY_AREA
        tiny_n += int(is_tiny)
        if (
            frame.kind.counts_as_observed
            and float(np.hypot(frame.x - truth[0], frame.y - truth[1])) <= MATCH_PX
        ):
            tp += 1
            tiny_tp += int(is_tiny)
        else:
            fn += 1

    total = tp + fn
    return {
        "config_fingerprint": config.fingerprint(),
        "n_positive_frames": total,
        "local_centre_recall_25px": round(tp / max(1, total), 4),
        "local_centre_recall_ci95": wilson(tp, total),
        "local_tiny_ball_recall": (
            round(tiny_tp / max(1, tiny_n), 4) if tiny_n else None
        ),
        "local_tiny_ball_n": tiny_n,
        "local_precision": None,
        "local_precision_reason": (
            "the locked local test contains zero negative frames, so there is "
            "no denominator for a false-positive rate; use the public figures "
            "and label them cross-domain"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    labels = {
        p.parent.name: p for p in sorted(GSR_ROOT.glob("**/Labels-GameState.json"))
    }
    split = build_split([], sorted(labels))
    chosen = [
        labels[s] for s in sorted(labels)
        if split.split_of("soccernet_gsr", s) == "test"
    ][: args.sequences]

    person_model = YOLO(PERSON_MODEL) if Path(PERSON_MODEL).exists() else None
    if person_model is None:
        log.warning("%s missing -- GMC will run unmasked, which is NOT what the "
                    "tracker does; this is recorded in the output", PERSON_MODEL)

    configs = engines()
    results: dict[str, dict] = {}
    for name, spec in DETECTORS.items():
        weights = Path(spec["weights"])
        if not weights.exists():
            log.warning("%s missing; skipped", name)
            continue
        log.info("=== detector %s (%s) ===", name, spec["fingerprint"])
        ball_model = YOLO(str(weights))

        public = collect_public(ball_model, person_model, chosen, args.max_frames)
        local = (
            [] if args.skip_local
            else collect_local(ball_model, person_model, args.window)
        )
        del ball_model

        results[name] = {"detector_fingerprint": spec["fingerprint"]}
        for engine, config in configs.items():
            block = score_public(public, config)
            if local:
                block.update(score_local(local, config))
            results[name][engine] = block
            log.info("  %-8s public R %.4f P %.4f FP/frame %.4f",
                     engine, block["public_centre_recall_25px"],
                     block["public_precision"], block["public_fp_per_frame"])

    payload = {
        "schema_version": "1.0.0",
        "conf": CONF_THRESHOLD,
        "imgsz": IMGSZ,
        "match_px": MATCH_PX,
        "public_split": "soccernet_gsr test (clip-disjoint)",
        "public_split_fingerprint": split.fingerprint(),
        "n_sequences": len(chosen),
        "max_frames_per_sequence": args.max_frames,
        "local_split": str(LOCAL_SPLIT),
        "local_window_frames": args.window,
        "gmc": {
            "source": "tracking.gmc.GlobalMotionCompensator",
            "method": "sparseOptFlow",
            "downscale": 2,
            "detection_masked": person_model is not None,
        },
        "caveats": [
            "the locked local test is 23 isolated frames; each is scored inside "
            f"a +-{args.window}-frame window decoded from the source video, "
            "because temporal verification cannot be applied to isolated frames",
            "a production run sees unbounded history, so local recall measured "
            "this way is a lower bound on the production figure",
            "no local precision is reported: the local test has zero negatives",
        ],
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
