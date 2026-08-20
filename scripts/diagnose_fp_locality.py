"""Do a detector's wrong ball detections land on players?

Phase 2E, Part 7b. The G4 diagnosis showed the controlled-possession regression
is driven by one quantity: the candidate's ball sits further from the nearest
player (median 0.646 -> 0.744 player-heights; share inside the control radius
0.4734 -> 0.4052). Two explanations fit that equally well from broadcast alone,
where there is no ball ground truth:

1. the candidate misplaces the ball, so it drifts away from the true holder; or
2. the *baseline* was frequently detecting something player-shaped -- a boot, a
   sock, a shadow -- and the possession engine then dutifully concluded that the
   player standing there was in control.

If (2) is what happens, controlled-possession share is partly manufactured by
false positives, and G4 rewards the detector that makes them.

SN-GSR VALID has an annotated ball, so the two can be separated. For every frame
where a detector reports a ball, the report is scored against the annotation and
split into hits (<= 25 px) and misses. Then the distance from the *reported*
position to the nearest annotated player is measured, in the same player-height
units the possession engine uses.

The prediction that distinguishes the two stories: under (2), the baseline's
**wrong** detections sit inside the control radius far more often than the
candidate's do.

Usage::

    python scripts/diagnose_fp_locality.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ball_operating_point_sweep import (  # noqa: E402
    ALL_DETECTORS,
    GSR_ROOT,
    PERCEPTION_RECORD,
    detect_sequence,
    sequence_footage,
)

from visionpitch.analytics.possession import PossessionConfig  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.possession_gt import load_gsr_gamestate  # noqa: E402

log = get_logger("phase2e.fp_locality")

RECORD = Path("data/eval/fusion/fp_locality.json")
MATCH_PX = 25.0

#: (label, checkpoint key, imgsz, conf) -- the two shipped/candidate points.
ARMS = [
    ("production", "A_default", 640, 0.08),
    ("candidate", "gsrtrain_v2c_CLEAN", 1280, 0.08),
]


def players_by_frame(labels_path: Path, window: list[int]) -> dict[int, list[tuple]]:
    """(foot_x, foot_y, box_height) per tracked person, per frame."""
    frames_meta, _ = load_gsr_gamestate(labels_path)
    keep = set(window)
    out: dict[int, list[tuple]] = {}
    for frame_idx, objects in frames_meta.items():
        if frame_idx not in keep:
            continue
        rows = []
        for obj in objects:
            if obj.role not in ("player", "goalkeeper"):
                continue
            if obj.box_height <= 1.0:
                continue
            rows.append((obj.image_x, obj.image_y, obj.box_height))
        out[frame_idx] = rows
    return out


def nearest_player_heights(position, players) -> float | None:
    """Distance to the nearest player, in that player's own box heights.

    The same normalisation the possession engine uses, so the numbers are
    directly comparable to ``control_radius_heights``.
    """
    best = None
    for px, py, height in players:
        distance = float(np.hypot(px - position[0], py - position[1])) / height
        if best is None or distance < best:
            best = distance
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--out", type=Path, default=RECORD)
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    control_radius = PossessionConfig().control_radius_heights
    perception = json.loads(PERCEPTION_RECORD.read_text(encoding="utf-8"))
    expected = list(perception["sequences_evaluated"])
    labels = {
        p.parent.name: p for p in sorted(GSR_ROOT.glob("**/Labels-GameState.json"))
    }

    log.info("loading footage")
    footages = [sequence_footage(labels[s], args.max_frames) for s in expected]
    players = {
        f["sequence"]: players_by_frame(f["labels_path"], f["frame_indices"])
        for f in footages
    }

    results: dict[str, dict] = {}
    for label, key, imgsz, conf in ARMS:
        weights = ALL_DETECTORS[key]
        if not Path(weights).is_file():
            log.warning("%s missing; skipped", weights)
            continue
        model = YOLO(weights)
        log.info("%s: %s @ %d conf %.2f", label, key, imgsz, conf)

        hits: list[float] = []
        misses: list[float] = []
        counts = Counter()
        for footage in footages:
            name = footage["sequence"]
            detections = detect_sequence(model, footage, imgsz)
            for frame_idx in footage["frame_indices"]:
                candidates = [c for c in detections.get(frame_idx, []) if c[3] >= conf]
                if not candidates:
                    continue
                best = max(candidates, key=lambda c: c[3])
                position = (best[0], best[1])
                truth = footage["truth"].get(frame_idx)
                near = nearest_player_heights(position, players[name].get(frame_idx, []))
                if near is None:
                    continue
                counts["observed"] += 1
                if truth is not None and float(
                    np.hypot(position[0] - truth[0], position[1] - truth[1])
                ) <= MATCH_PX:
                    counts["hit"] += 1
                    hits.append(near)
                else:
                    counts["miss"] += 1
                    misses.append(near)
        del model

        def summarise(values: list[float]) -> dict:
            if not values:
                return {"n": 0}
            array = np.array(values)
            return {
                "n": len(values),
                "median_heights": round(float(np.median(array)), 4),
                "p25_heights": round(float(np.percentile(array, 25)), 4),
                "share_inside_control_radius": round(
                    float(np.mean(array <= control_radius)), 4
                ),
            }

        results[label] = {
            "checkpoint": key,
            "weights": weights,
            "imgsz": imgsz,
            "conf": conf,
            "frames_with_a_detection": counts["observed"],
            "correct": counts["hit"],
            "wrong": counts["miss"],
            "wrong_rate": round(counts["miss"] / max(1, counts["observed"]), 4),
            "distance_to_nearest_player": {
                "correct_detections": summarise(hits),
                "wrong_detections": summarise(misses),
            },
            "wrong_detections_inside_control_radius": int(
                sum(1 for v in misses if v <= control_radius)
            ),
            "wrong_inside_radius_per_1000_frames": round(
                1000 * sum(1 for v in misses if v <= control_radius)
                / max(1, sum(len(f["frame_indices"]) for f in footages)), 2
            ),
        }

    payload = {
        "schema_version": "1.0.0",
        "question": (
            "do wrong ball detections land close enough to a player that the "
            "possession engine would call that player in control?"
        ),
        "split": "soccernet_gsr validation (canonical)",
        "split_fingerprint": perception["split_fingerprint"],
        "sequences_evaluated": expected,
        "match_px": MATCH_PX,
        "control_radius_heights": control_radius,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\ncontrol radius = {control_radius} player-heights\n")
    print(f"{'arm':<12}{'obs':>7}{'wrong':>7}{'wrong%':>8}"
          f"{'wrongNear':>10}{'wrongInRadius%':>15}{'per1000fr':>11}")
    for label, r in results.items():
        wrong = r["distance_to_nearest_player"]["wrong_detections"]
        print(f"{label:<12}{r['frames_with_a_detection']:>7}{r['wrong']:>7}"
              f"{r['wrong_rate']:>8.4f}{wrong.get('median_heights', 0):>10.3f}"
              f"{wrong.get('share_inside_control_radius', 0):>15.4f}"
              f"{r['wrong_inside_radius_per_1000_frames']:>11.2f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
