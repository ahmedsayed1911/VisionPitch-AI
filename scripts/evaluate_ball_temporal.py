"""Measure observability, recovery and effective coverage on real sequences.

Phase 2D, Parts 2, 4, 7 and the coverage half of Part 8.

The multi-corpus test split is a bag of still frames, so it can measure a
detector but says nothing about *temporal* coverage -- which is what possession
actually consumes. SN-GSR sequences are contiguous 30-second clips with a
per-frame annotated ball, so they can.

For each held-out sequence this runs the real per-frame detector, labels every
frame with the observability model, attempts track-before-detect recovery over
each gap, and then scores four things that are usually conflated:

* **detector recall** -- did the detector see the ball
* **recovery yield** -- how many gap frames were recovered
* **recovery accuracy** -- were those recoveries *right*, measured against the
  annotated ball. A recovery stage that fills gaps with plausible nonsense
  improves every coverage number and corrupts everything downstream, so this is
  the number that decides whether the stage ships.
* **hallucination rate** -- recoveries produced where the ground truth says the
  ball is somewhere else entirely, or is not annotated at all

Nothing here tunes anything. Parameters come from the module defaults, and the
sequences are the test split of the Phase 2C clip-disjoint partition.

Usage::

    python scripts/evaluate_ball_temporal.py --max-sequences 9
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

from visionpitch.ball_tracking.observability import (  # noqa: E402
    ObservabilityEstimator,
)
from visionpitch.ball_tracking.recovery import (  # noqa: E402
    RecoveryConfig,
    TrackBeforeDetect,
)
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.possession_gt import load_gsr_gamestate  # noqa: E402
from visionpitch.evaluation.registry import build_split  # noqa: E402

log = get_logger("ball.temporal")

GSR_ROOT = Path("data/eval/gsr")
MATCH_PX = 25.0


def sequence_dirs() -> dict[str, Path]:
    return {
        p.parent.name: p for p in sorted(GSR_ROOT.glob("**/Labels-GameState.json"))
    }


def predicted_positions_for_gap(
    before: tuple[int, float, float] | None,
    after: tuple[int, float, float] | None,
    gap: list[int],
) -> dict[int, tuple[float, float]]:
    """Where to look during a gap.

    Interpolate between the bracketing sightings when both exist; extrapolate
    from one when only one does. These positions are search hints only -- they
    are never emitted, and a recovery is accepted on image evidence, not on
    agreement with this guess.
    """
    out: dict[int, tuple[float, float]] = {}
    if before is not None and after is not None:
        span = after[0] - before[0]
        if span <= 0:
            return out
        for frame_idx in gap:
            t = (frame_idx - before[0]) / span
            out[frame_idx] = (
                before[1] + (after[1] - before[1]) * t,
                before[2] + (after[2] - before[2]) * t,
            )
    elif before is not None:
        for frame_idx in gap:
            out[frame_idx] = (before[1], before[2])
    elif after is not None:
        for frame_idx in gap:
            out[frame_idx] = (after[1], after[2])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/finetune/ball_multicorpus/weights/best.pt")
    parser.add_argument("--player-model", default="models/yolo-football-player-detection.pt")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/eval/ball_temporal"))
    parser.add_argument("--label", default="multicorpus")
    # Recovery parameters are exposed so they can be swept on the TRAIN
    # sequences. The test split must be scored once, with whatever the sweep
    # chose, and never used to choose anything.
    parser.add_argument("--min-supporting-frames", type=int, default=None)
    parser.add_argument("--min-response-ratio", type=float, default=None)
    parser.add_argument("--max-deviation-px", type=float, default=None)
    parser.add_argument("--search-radius-px", type=float, default=None)
    parser.add_argument("--max-gap-frames", type=int, default=None)
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    sequences = sequence_dirs()
    split = build_split([], sorted(sequences))
    chosen = [
        s for s in sorted(sequences)
        if split.split_of("soccernet_gsr", s) == args.split
    ]
    if args.max_sequences:
        chosen = chosen[: args.max_sequences]
    log.info("%s split: %d sequence(s), fingerprint %s",
             args.split, len(chosen), split.fingerprint())

    ball_model = YOLO(args.model)
    player_model = YOLO(args.player_model)
    estimator = ObservabilityEstimator()
    recovery_config = RecoveryConfig()
    for attribute, value in (
        ("min_supporting_frames", args.min_supporting_frames),
        ("min_response_ratio", args.min_response_ratio),
        ("max_deviation_px", args.max_deviation_px),
        ("search_radius_px", args.search_radius_px),
        ("max_gap_frames", args.max_gap_frames),
    ):
        if value is not None:
            setattr(recovery_config, attribute, value)
    recoverer = TrackBeforeDetect(recovery_config)

    totals = Counter()
    per_sequence = []

    for name in chosen:
        labels_path = sequences[name]
        frames_meta, fps = load_gsr_gamestate(labels_path)
        image_dir = labels_path.parent / "img1"
        if not image_dir.exists():
            log.warning("%s has no img1 directory; skipped", name)
            continue

        frame_indices = sorted(frames_meta)
        if args.max_frames:
            frame_indices = frame_indices[: args.max_frames]

        # -- ground truth ball, image coordinates ---------------------------- #
        truth: dict[int, tuple[float, float]] = {}
        for frame_idx in frame_indices:
            ball = next(
                (o for o in frames_meta[frame_idx] if o.role == "ball"), None
            )
            if ball is not None:
                truth[frame_idx] = (ball.image_x, ball.image_y - ball.box_height / 2)

        images = {
            frame_idx: image_dir / f"{frame_idx:06d}.jpg" for frame_idx in frame_indices
        }
        images = {k: v for k, v in images.items() if v.exists()}
        if not images:
            log.warning("%s: no readable frames; skipped", name)
            continue
        frame_indices = sorted(images)

        grey_cache: dict[int, np.ndarray] = {}

        # Loop variables are bound as defaults rather than captured. The closure
        # only outlives one iteration by accident today, and a late-bound cache
        # would silently serve one sequence's frames while scoring another's.
        def grey(frame_idx: int, _images=images, _cache=grey_cache):
            if frame_idx not in _images:
                return None
            if frame_idx not in _cache:
                _cache[frame_idx] = cv2.imread(
                    str(_images[frame_idx]), cv2.IMREAD_GRAYSCALE
                )
            return _cache[frame_idx]

        # -- detector pass ---------------------------------------------------- #
        observed: dict[int, tuple[float, float]] = {}
        players_by_frame: dict[int, np.ndarray] = {}
        motion_by_frame: dict[int, float] = {}
        previous_grey = None

        for frame_idx in frame_indices:
            frame = cv2.imread(str(images[frame_idx]))
            if frame is None:
                continue
            result = ball_model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, verbose=False
            )[0]
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                best = int(np.argmax(scores))
                observed[frame_idx] = (
                    float((boxes[best][0] + boxes[best][2]) / 2),
                    float((boxes[best][1] + boxes[best][3]) / 2),
                )
            people = player_model.predict(
                frame, imgsz=args.imgsz, conf=0.25, verbose=False
            )[0]
            players_by_frame[frame_idx] = (
                people.boxes.xyxy.cpu().numpy()
                if people.boxes is not None else np.zeros((0, 4))
            )

            current_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey_cache[frame_idx] = current_grey
            if previous_grey is not None and previous_grey.shape == current_grey.shape:
                small_a = cv2.resize(previous_grey, (160, 90))
                small_b = cv2.resize(current_grey, (160, 90))
                shift = cv2.phaseCorrelate(
                    small_a.astype(np.float32), small_b.astype(np.float32)
                )[0]
                scale = current_grey.shape[1] / 160.0
                motion_by_frame[frame_idx] = float(
                    np.hypot(shift[0], shift[1]) * scale
                )
            previous_grey = current_grey

        height, width = cv2.imread(str(images[frame_indices[0]])).shape[:2]

        # -- observability ---------------------------------------------------- #
        report = estimator.label_sequence(
            frame_indices=frame_indices,
            frame_size=(width, height),
            ball_observations=observed,
            player_boxes_by_frame=players_by_frame,
            camera_motion_by_frame=motion_by_frame,
        )
        observability_states = {
            idx: entry.state.value for idx, entry in report.frames.items()
        }

        # -- recovery over gaps ------------------------------------------------ #
        recovered: dict[int, tuple[float, float]] = {}
        n_gap_frames = 0
        if not args.no_recovery:
            gaps: list[list[int]] = []
            current_gap: list[int] = []
            for frame_idx in frame_indices:
                if frame_idx in observed:
                    if current_gap:
                        gaps.append(current_gap)
                        current_gap = []
                else:
                    current_gap.append(frame_idx)
            if current_gap:
                gaps.append(current_gap)

            for gap in gaps:
                n_gap_frames += len(gap)
                before_candidates = [i for i in observed if i < gap[0]]
                after_candidates = [i for i in observed if i > gap[-1]]
                before = (
                    (max(before_candidates), *observed[max(before_candidates)])
                    if before_candidates else None
                )
                after = (
                    (min(after_candidates), *observed[min(after_candidates)])
                    if after_candidates else None
                )
                hints = predicted_positions_for_gap(before, after, gap)
                for observation in recoverer.recover(
                    gap, grey, hints, observability_states
                ):
                    recovered[observation.frame_idx] = observation.position

        # -- scoring ------------------------------------------------------------ #
        scorable = [i for i in frame_indices if i in truth]
        detector_hits = sum(
            1 for i in scorable
            if i in observed
            and np.hypot(*np.subtract(observed[i], truth[i])) <= MATCH_PX
        )
        recovery_scorable = [i for i in recovered if i in truth]
        recovery_hits = sum(
            1 for i in recovery_scorable
            if np.hypot(*np.subtract(recovered[i], truth[i])) <= MATCH_PX
        )
        # A recovery on a frame with no annotated ball cannot be verified, and a
        # recovery far from the annotated ball is a wrong answer. Both are
        # counted as hallucination risk rather than quietly ignored.
        recovery_wrong = len(recovery_scorable) - recovery_hits
        recovery_unverifiable = len(recovered) - len(recovery_scorable)

        fair = report.fair_frames
        fair_scorable = [i for i in scorable if i in fair]
        fair_hits = sum(
            1 for i in fair_scorable
            if i in observed
            and np.hypot(*np.subtract(observed[i], truth[i])) <= MATCH_PX
        )

        entry = {
            "sequence": name,
            "n_frames": len(frame_indices),
            "n_gt_ball": len(scorable),
            "detector_recall": round(detector_hits / max(1, len(scorable)), 4),
            "observability_counts": report.counts(),
            "n_observable_frames": len(fair),
            "observable_fraction": round(len(fair) / max(1, len(frame_indices)), 4),
            "detector_recall_on_observable": round(
                fair_hits / max(1, len(fair_scorable)), 4
            ),
            "n_gap_frames": n_gap_frames,
            "n_recovered": len(recovered),
            "recovery_yield": round(len(recovered) / max(1, n_gap_frames), 4),
            "recovery_accuracy": round(
                recovery_hits / max(1, len(recovery_scorable)), 4
            ) if recovery_scorable else None,
            "n_recovery_wrong": recovery_wrong,
            "n_recovery_unverifiable": recovery_unverifiable,
            "coverage_direct": round(len(observed) / max(1, len(frame_indices)), 4),
            "coverage_direct_plus_recovered": round(
                (len(observed) + len(recovered)) / max(1, len(frame_indices)), 4
            ),
        }
        per_sequence.append(entry)
        for key in ("n_frames", "n_gt_ball", "n_gap_frames", "n_recovered",
                    "n_recovery_wrong", "n_recovery_unverifiable"):
            totals[key] += entry[key]
        totals["detector_hits"] += detector_hits
        totals["recovery_hits"] += recovery_hits
        totals["recovery_scorable"] += len(recovery_scorable)
        totals["observable"] += len(fair)
        totals["fair_scorable"] += len(fair_scorable)
        totals["fair_hits"] += fair_hits
        totals["observed"] += len(observed)
        for state, count in report.counts().items():
            totals[f"obs_{state}"] += count

        log.info(
            "%s: det R %.3f | observable %.3f | recovered %d/%d (acc %s)",
            name, entry["detector_recall"], entry["observable_fraction"],
            entry["n_recovered"], entry["n_gap_frames"],
            entry["recovery_accuracy"],
        )

    pooled = {
        "n_sequences": len(per_sequence),
        "n_frames": totals["n_frames"],
        "detector_recall": round(
            totals["detector_hits"] / max(1, totals["n_gt_ball"]), 4
        ),
        "detector_recall_on_observable": round(
            totals["fair_hits"] / max(1, totals["fair_scorable"]), 4
        ),
        "observable_fraction": round(
            totals["observable"] / max(1, totals["n_frames"]), 4
        ),
        "coverage_direct": round(totals["observed"] / max(1, totals["n_frames"]), 4),
        "coverage_direct_plus_recovered": round(
            (totals["observed"] + totals["n_recovered"]) / max(1, totals["n_frames"]), 4
        ),
        "n_gap_frames": totals["n_gap_frames"],
        "n_recovered": totals["n_recovered"],
        "recovery_yield": round(
            totals["n_recovered"] / max(1, totals["n_gap_frames"]), 4
        ),
        "recovery_accuracy": round(
            totals["recovery_hits"] / max(1, totals["recovery_scorable"]), 4
        ) if totals["recovery_scorable"] else None,
        "n_recovery_wrong": totals["n_recovery_wrong"],
        "n_recovery_unverifiable": totals["n_recovery_unverifiable"],
        "observability_counts": {
            k[4:]: v for k, v in totals.items() if k.startswith("obs_")
        },
    }

    payload = {
        "label": args.label,
        "model": args.model,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "split": args.split,
        "split_fingerprint": split.fingerprint(),
        "match_px": MATCH_PX,
        "recovery_enabled": not args.no_recovery,
        "recovery_config": recovery_config.to_dict(),
        "observability_config": ObservabilityEstimator().cfg.to_dict(),
        "pooled": pooled,
        "per_sequence": per_sequence,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "norecovery" if args.no_recovery else "recovery"
    destination = args.out / f"temporal_{args.label}_{suffix}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(pooled, indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
