"""Why the ball is missed, and why one checkpoint beats another.

Phase 2E, follow-up to the operating-point sweep. Two questions the sweep table
cannot answer:

1. The zero-candidate rate is 40-57% of frames. How much of that is the detector
   failing on a ball that is plainly there, and how much is a ball that is not
   usefully visible? Only the first kind is fixable by training.
2. ``C_adapt`` beats ``A_default`` by a large margin downstream. Recall,
   coverage, localisation, miss-streak length and confidence calibration are
   different mechanisms with different remedies, so "it wins" is not an answer.

Conditioning, not counting
--------------------------
Reporting the *distribution of factors among misses* is misleading on its own:
if 80% of missed frames have a small ball but 80% of hit frames do too, ball
size explains nothing. Every factor here is therefore reported as a **miss rate
conditioned on the factor**, against the base rate, so a factor only looks
important when it actually discriminates.

Factors are measured from data that exists. Ball size, crowding, occlusion,
camera motion, local blur and the aerial proxy are all computable from the
annotation plus the image. Compression artefacts and field-line confusion are
not separable with what is available here and are not invented -- frames that
match no measured factor are reported as ``unattributed`` rather than assigned
to a plausible-sounding bucket.

Usage::

    python scripts/ball_failure_diagnosis.py --detector C_adapt --imgsz 1280 --conf 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ball_operating_point_sweep import (  # noqa: E402
    ALL_DETECTORS,
    CONF_FLOOR,
    GSR_ROOT,
    PERCEPTION_RECORD,
    detect_sequence,
    sequence_footage,
)

#: every ball checkpoint on disk, so any of them can be diagnosed
DETECTORS = ALL_DETECTORS

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.possession_gt import load_gsr_gamestate  # noqa: E402

log = get_logger("ball.diagnosis")

RECORD = Path("data/eval/fusion/failure_diagnosis.json")

#: Thresholds for the factor tests. Each is a stated cut, not a fitted one.
TINY_BALL_PX = 10.0          # annotated ball diameter below this, pixels
BLUR_VARIANCE = 60.0         # Laplacian variance in the ball patch below this
FAST_CAMERA_PX = 12.0        # frame-to-frame background displacement above this
CROWD_RADIUS_PX = 60.0       # players whose box is within this of the ball
CROWDED_COUNT = 2            # this many or more is a crowded neighbourhood
AERIAL_HEIGHT_RATIO = 1.2    # ball this far above the nearest player's feet,
                             # in units of that player's box height
PATCH = 24                   # half-size of the ball patch, pixels


def player_geometry(labels_path: Path, window: list[int]) -> dict[int, list[tuple]]:
    """Per-frame player boxes: (cx, foot_y, height, x1, y1, x2, y2)."""
    frames_meta, _ = load_gsr_gamestate(labels_path)
    out: dict[int, list[tuple]] = {}
    keep = set(window)
    for frame_idx, objects in frames_meta.items():
        if frame_idx not in keep:
            continue
        rows = []
        for obj in objects:
            if obj.role not in ("player", "goalkeeper"):
                continue
            height = obj.box_height
            if height <= 1.0:
                continue
            rows.append((
                obj.image_x, obj.image_y, height,
                obj.image_x - height * 0.2, obj.image_y - height,
                obj.image_x + height * 0.2, obj.image_y,
            ))
        out[frame_idx] = rows
    return out


def ball_sizes(labels_path: Path, window: list[int]) -> dict[int, float]:
    frames_meta, _ = load_gsr_gamestate(labels_path)
    keep = set(window)
    out: dict[int, float] = {}
    for frame_idx, objects in frames_meta.items():
        if frame_idx not in keep:
            continue
        ball = next((o for o in objects if o.role == "ball"), None)
        if ball is not None:
            # GSRObject carries only box_height; the ball is round, so height is
            # its diameter and no width is needed.
            out[frame_idx] = float(ball.box_height)
    return out


def factors_for_frame(image, position, size_px, players, shift) -> dict[str, bool]:
    """Which measurable conditions hold for this frame's annotated ball."""
    x, y = position
    height, width = image.shape[:2]

    x1 = max(0, int(x) - PATCH)
    x2 = min(width, int(x) + PATCH)
    y1 = max(0, int(y) - PATCH)
    y2 = min(height, int(y) + PATCH)
    patch = image[y1:y2, x1:x2]
    if patch.size:
        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    else:
        blur = float("inf")

    near = 0
    occluded = False
    aerial = False
    nearest_height = None
    nearest_distance = float("inf")
    for cx, foot_y, box_height, bx1, by1, bx2, by2 in players:
        distance = float(np.hypot(cx - x, foot_y - y))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_height = box_height
        if abs(cx - x) <= CROWD_RADIUS_PX and abs(foot_y - y) <= CROWD_RADIUS_PX:
            near += 1
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            occluded = True
    if nearest_height:
        aerial = (nearest_distance / nearest_height) > AERIAL_HEIGHT_RATIO and (
            y < (min((p[1] for p in players), default=y) - nearest_height * 0.5)
        )

    magnitude = float(np.hypot(*shift)) if shift else 0.0

    return {
        "tiny_ball": size_px is not None and size_px < TINY_BALL_PX,
        "motion_blur": blur < BLUR_VARIANCE,
        "fast_camera": magnitude > FAST_CAMERA_PX,
        "crowded": near >= CROWDED_COUNT,
        "occluded_by_player": occluded,
        "aerial": aerial,
        "near_frame_edge": x < 40 or x > width - 40 or y < 40 or y > height - 40,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", default="multicorpus", choices=sorted(DETECTORS))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--compare-detector", default="A_default")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--out", type=Path, default=RECORD)
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    perception = json.loads(PERCEPTION_RECORD.read_text(encoding="utf-8"))
    expected = list(perception["sequences_evaluated"])
    labels = {
        p.parent.name: p for p in sorted(GSR_ROOT.glob("**/Labels-GameState.json"))
    }

    log.info("loading footage")
    footages = [sequence_footage(labels[s], args.max_frames) for s in expected]

    detections: dict[str, dict] = {}
    for name in {args.detector, args.compare_detector}:
        model = YOLO(DETECTORS[name])
        detections[name] = {
            f["sequence"]: detect_sequence(model, f, args.imgsz) for f in footages
        }
        del model
        log.info("cached candidates for %s @ %d", name, args.imgsz)

    # -- zero-candidate anatomy for the primary detector --------------------- #
    totals = Counter()
    factor_frames = Counter()
    factor_misses = Counter()
    unattributed = 0
    per_sequence = []

    for footage in footages:
        name = footage["sequence"]
        window = footage["frame_indices"]
        players_by_frame = player_geometry(footage["labels_path"], window)
        sizes = ball_sizes(footage["labels_path"], window)
        seq = Counter()

        for frame_idx in window:
            candidates = [
                c for c in detections[args.detector][name].get(frame_idx, [])
                if c[3] >= args.conf
            ]
            has_ball = frame_idx in footage["truth"]
            zero = not candidates

            totals["frames"] += 1
            seq["frames"] += 1
            totals["zero_candidate"] += int(zero)
            seq["zero_candidate"] += int(zero)
            if not has_ball:
                totals["no_annotated_ball"] += 1
                seq["no_annotated_ball"] += 1
                if zero:
                    totals["zero_and_no_ball"] += 1
                    seq["zero_and_no_ball"] += 1
                continue

            totals["annotated_ball"] += 1
            seq["annotated_ball"] += 1
            if zero:
                totals["true_miss"] += 1
                seq["true_miss"] += 1

            image = cv2.imread(str(footage["image_dir"] / f"{frame_idx:06d}.jpg"))
            if image is None:
                continue
            factors = factors_for_frame(
                image, footage["truth"][frame_idx], sizes.get(frame_idx),
                players_by_frame.get(frame_idx, []),
                footage["shifts"].get(frame_idx),
            )
            for key, active in factors.items():
                if active:
                    factor_frames[key] += 1
                    if zero:
                        factor_misses[key] += 1
            if zero and not any(factors.values()):
                unattributed += 1

        per_sequence.append({
            "sequence": name,
            "frames": seq["frames"],
            "zero_candidate_rate": round(seq["zero_candidate"] / max(1, seq["frames"]), 4),
            "annotated_ball_frames": seq["annotated_ball"],
            "no_annotated_ball": seq["no_annotated_ball"],
            "true_miss_rate": round(seq["true_miss"] / max(1, seq["annotated_ball"]), 4),
        })

    base_rate = totals["true_miss"] / max(1, totals["annotated_ball"])
    conditioned = {}
    for key, n in sorted(factor_frames.items()):
        misses = factor_misses[key]
        conditioned[key] = {
            "frames_with_factor": n,
            "share_of_annotated_frames": round(n / max(1, totals["annotated_ball"]), 4),
            "miss_rate_given_factor": round(misses / max(1, n), 4),
            "lift_vs_base_rate": round(
                (misses / max(1, n)) / base_rate, 3
            ) if base_rate else None,
            "share_of_all_misses": round(misses / max(1, totals["true_miss"]), 4),
        }

    # -- detector comparison -------------------------------------------------- #
    comparison = {}
    for name in (args.detector, args.compare_detector):
        hit_scores: list[float] = []
        false_scores: list[float] = []
        errors: list[float] = []
        streaks: list[int] = []
        found = Counter()
        for footage in footages:
            seq = footage["sequence"]
            streak = 0
            for frame_idx in footage["frame_indices"]:
                candidates = [
                    c for c in detections[name][seq].get(frame_idx, [])
                    if c[3] >= args.conf
                ]
                gt = footage["truth"].get(frame_idx)
                found["frames"] += 1
                if not candidates:
                    streak += 1
                    continue
                if streak:
                    streaks.append(streak)
                    streak = 0
                best = max(candidates, key=lambda c: c[3])
                found["observed"] += 1
                if gt is None:
                    false_scores.append(best[3])
                    continue
                distance = float(np.hypot(best[0] - gt[0], best[1] - gt[1]))
                errors.append(distance)
                if distance <= 25.0:
                    hit_scores.append(best[3])
                    found["hit"] += 1
                else:
                    false_scores.append(best[3])
            if streak:
                streaks.append(streak)
        comparison[name] = {
            "coverage": round(found["observed"] / max(1, found["frames"]), 4),
            "hit_rate": round(found["hit"] / max(1, found["frames"]), 4),
            "localisation_error_px": {
                "median": round(float(np.median(errors)), 3) if errors else None,
                "p90": round(float(np.percentile(errors, 90)), 3) if errors else None,
                "p95": round(float(np.percentile(errors, 95)), 3) if errors else None,
            },
            "confidence_on_hits": {
                "median": round(float(np.median(hit_scores)), 4) if hit_scores else None,
                "p10": round(float(np.percentile(hit_scores, 10)), 4) if hit_scores else None,
                "n": len(hit_scores),
            },
            "confidence_on_false": {
                "median": round(float(np.median(false_scores)), 4) if false_scores else None,
                "p90": round(float(np.percentile(false_scores, 90)), 4) if false_scores else None,
                "n": len(false_scores),
            },
            "miss_streaks": {
                "median": float(np.median(streaks)) if streaks else 0.0,
                "p90": round(float(np.percentile(streaks, 90)), 2) if streaks else 0.0,
                "max": int(max(streaks)) if streaks else 0,
                "n": len(streaks),
                "frames_in_streaks_over_6": sum(s for s in streaks if s > 6),
            },
        }

    payload = {
        "schema_version": "1.0.0",
        "split_fingerprint": perception["split_fingerprint"],
        "sequences_evaluated": expected,
        "operating_point": {
            "detector": args.detector, "imgsz": args.imgsz, "conf": args.conf,
            "conf_floor_for_cache": CONF_FLOOR,
        },
        "thresholds": {
            "tiny_ball_px": TINY_BALL_PX, "blur_variance": BLUR_VARIANCE,
            "fast_camera_px": FAST_CAMERA_PX, "crowd_radius_px": CROWD_RADIUS_PX,
            "crowded_count": CROWDED_COUNT,
            "aerial_height_ratio": AERIAL_HEIGHT_RATIO,
        },
        "zero_candidate_anatomy": {
            "frames": totals["frames"],
            "zero_candidate_frames": totals["zero_candidate"],
            "zero_candidate_rate": round(
                totals["zero_candidate"] / max(1, totals["frames"]), 4
            ),
            "frames_without_annotated_ball": totals["no_annotated_ball"],
            "zero_candidate_and_no_annotated_ball": totals["zero_and_no_ball"],
            "frames_with_annotated_ball": totals["annotated_ball"],
            "true_misses": totals["true_miss"],
            "true_miss_rate": round(base_rate, 4),
            "unattributed_misses": unattributed,
            "unattributed_share_of_misses": round(
                unattributed / max(1, totals["true_miss"]), 4
            ),
        },
        "factors_conditioned": conditioned,
        "detector_comparison": comparison,
        "per_sequence": per_sequence,
        "caveats": [
            "compression artefacts and field-line confusion are not separable "
            "with the available data and are not assigned a bucket",
            "factors overlap; shares of misses do not sum to 1",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
