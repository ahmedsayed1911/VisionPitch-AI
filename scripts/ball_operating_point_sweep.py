"""Detector x confidence x resolution sweep, scored on downstream possession.

Phase 2E. The temporal fusion path was rejected on VALID evidence, and the
decisive number that survived was a *detector* difference measured at
``conf=0.12, imgsz=960`` while production ships roughly ``conf=0.08, imgsz=640``.
Every existing ball claim therefore rests on an operating point nobody ships.
This removes that confound before any retraining is considered.

``engine="legacy"`` throughout: temporal fusion failed gates G3/G4 and is not
under test here.

Why one detector pass per (detector, resolution) covers seven confidences
------------------------------------------------------------------------
Ultralytics applies ``conf`` as a filter *before* NMS, and NMS only ever lets a
higher-scoring box suppress a lower-scoring one -- a 0.05 candidate can never
remove a 0.25 candidate. So the set of surviving boxes at score >= c is
identical whether the model was asked for ``conf=c`` or ``conf=0.02`` and then
filtered offline. The sweep runs the detector at a floor of 0.02 and derives
every confidence level from the cache: 8 GPU passes instead of 56.

Camera shifts are a property of the footage, not of the detector, so they are
computed once per sequence and shared across the whole grid.

What is measured, and what is not
---------------------------------
Possession determinability is the primary metric, but it is never read alone: a
detector can raise it by accepting large numbers of false balls, so holder
accuracy, team F1, false positives and localisation error are reported for every
cell and the selection rule weighs them together.

SN-GSR has **no** pass ground truth. No event metric is computed here; the
SN-BAS event evidence stands separately and is not mixed in.

Usage::

    python scripts/ball_operating_point_sweep.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from fusion_downstream_valid import (  # noqa: E402
    DETERMINABLE,
    _homographies,
    _project,
    _restrict,
    arms,
    clip_reference,
    switch_rate,
)

from visionpitch.analytics.possession import (  # noqa: E402
    PossessionConfig,
    PossessionEngine,
)
from visionpitch.analytics.types import BallStateKind  # noqa: E402
from visionpitch.ball_tracking.fusion import BallFusion  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.possession_eval import (  # noqa: E402
    aggregate,
    context_from_gsr,
    evaluate_possession_vs_gt,
)
from visionpitch.evaluation.possession_gt import (  # noqa: E402
    DerivationParams,
    derive_from_gsr,
    load_gsr_gamestate,
)

log = get_logger("ball.sweep")

GSR_ROOT = Path("data/SoccerNetGS/valid")
SPLIT_MANIFEST = Path("data/eval/gsr/sequences_info.json")
PERCEPTION_RECORD = Path("data/eval/fusion/ablation_valid.json")
RECORD = Path("data/eval/fusion/operating_point_sweep.json")

DETECTORS = {
    "A_default": "models/yolo-football-ball-detection.pt",
    "C_adapt": "models/finetune/bcast_adapt/weights/best.pt",
}

#: Every ball checkpoint on disk, for the wider comparison. ``ball_gsr`` was
#: trained on SNGS-116..200 -- the SN-GSR *test* split -- so it is usable for
#: VALID-only selection but is permanently burned for any test-set claim, and
#: that is recorded next to it rather than left for someone to rediscover.
ALL_DETECTORS = {
    "A_default": "models/yolo-football-ball-detection.pt",
    "C_adapt": "models/finetune/bcast_adapt/weights/best.pt",
    "D_adapt_aug": "models/finetune/bcast_adapt_aug/weights/best.pt",
    "B_public": "models/finetune/bcast_public/weights/best.pt",
    "C_hardened": "models/finetune/bcast_hardened/weights/best.pt",
    "multicorpus": "models/finetune/ball_multicorpus/weights/best.pt",
    "ball_gsr_TESTBURNED": "models/finetune/ball_gsr/weights/best.pt",
    "gsr_shipped": "models/yolo-football-ball-detection-gsr.pt",
    # Phase 2E CASE D: the first ball checkpoint trained only on the canonical
    # SN-GSR *train* split, so it leaves the test split available for a real
    # held-out evaluation. Skipped automatically until the run finishes.
    "gsrtrain_v1_CLEAN": "models/finetune/ball_gsrtrain_v1/weights/best.pt",
    # v2 repeats v1 on the same data with the reference recipe's schedule
    # (45 epochs, patience 12); v1 early-stopped at 21 and was undertrained.
    "gsrtrain_v2_CLEAN": "models/finetune/ball_gsrtrain_v2/weights/best.pt",
    # v2c continues v2 to genuine convergence. A true resume was impossible --
    # Ultralytics strips the optimizer/EMA state when a run finishes normally --
    # so this is a warm restart from v2's last.pt with identical hyperparameters.
    "gsrtrain_v2c_CLEAN": "models/finetune/ball_gsrtrain_v2c/weights/best.pt",
}

CONFIDENCES = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
RESOLUTIONS = [640, 960, 1280, 1536]
CONF_FLOOR = 0.02
MATCH_THRESHOLDS = [10.0, 15.0, 25.0]
CANONICAL_MATCH_PX = 25.0


# --------------------------------------------------------------------------- #
# Detector cache
# --------------------------------------------------------------------------- #


def camera_shifts(image_dir: Path, frame_indices: list[int]) -> dict[int, tuple[float, float]]:
    """Frame-to-frame background displacement, independent of the detector."""
    shifts: dict[int, tuple[float, float]] = {}
    previous = None
    for frame_idx in frame_indices:
        frame = cv2.imread(str(image_dir / f"{frame_idx:06d}.jpg"))
        if frame is None:
            continue
        grey = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        if previous is not None:
            shift = cv2.phaseCorrelate(
                previous.astype(np.float32), grey.astype(np.float32)
            )[0]
            scale = frame.shape[1] / 320.0
            shifts[frame_idx] = (float(shift[0] * scale), float(shift[1] * scale))
        previous = grey
    return shifts


def sequence_footage(labels_path: Path, max_frames: int) -> dict:
    """Frame window, annotated ball, and camera shifts for one sequence."""
    frames_meta, fps = load_gsr_gamestate(labels_path)
    image_dir = labels_path.parent / "img1"
    frame_indices = sorted(frames_meta)[:max_frames]
    kept = [i for i in frame_indices if (image_dir / f"{i:06d}.jpg").exists()]

    truth: dict[int, tuple[float, float]] = {}
    for frame_idx in kept:
        ball = next((o for o in frames_meta[frame_idx] if o.role == "ball"), None)
        if ball is not None:
            truth[frame_idx] = (ball.image_x, ball.image_y - ball.box_height / 2)

    return {
        "sequence": labels_path.parent.name,
        "labels_path": labels_path,
        "image_dir": image_dir,
        "frame_indices": kept,
        "truth": truth,
        "fps": fps,
        "shifts": camera_shifts(image_dir, kept),
    }


def detect_sequence(
    model, footage: dict, imgsz: int, conf_floor: float = CONF_FLOOR
) -> dict[int, list]:
    """All candidates at the confidence floor, for later offline filtering."""
    out: dict[int, list] = {}
    for frame_idx in footage["frame_indices"]:
        frame = cv2.imread(str(footage["image_dir"] / f"{frame_idx:06d}.jpg"))
        if frame is None:
            out[frame_idx] = []
            continue
        result = model.predict(frame, imgsz=imgsz, conf=conf_floor, verbose=False)[0]
        found = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores, strict=True):
                found.append((
                    float((box[0] + box[2]) / 2),
                    float((box[1] + box[3]) / 2),
                    float(max(box[2] - box[0], box[3] - box[1]) / 2),
                    float(score),
                ))
        out[frame_idx] = found
    return out


# --------------------------------------------------------------------------- #
# Cached ground-truth side
# --------------------------------------------------------------------------- #


def build_reference(footage: dict) -> dict:
    """Context, homographies and derived possession reference, built once."""
    labels_path = footage["labels_path"]
    window = footage["frame_indices"]
    context = _restrict(context_from_gsr(labels_path), window)
    fps = context.fps
    reference = clip_reference(
        derive_from_gsr(labels_path, DerivationParams()),
        min(window) / fps, (max(window) + 1) / fps,
    )
    return {
        "context": context,
        "homographies": _homographies(context),
        "reference": reference,
        "fps": fps,
    }


def apply_ball(cached: dict, positions: dict[int, tuple[float, float]], window: list[int]):
    """Swap a fused ball into the cached context without rebuilding it."""
    context = cached["context"]
    homographies = cached["homographies"]
    rows = []
    ball_by_frame: dict[int, tuple] = {}
    for idx in window:
        position = positions.get(idx)
        if position is None:
            continue
        matrix = homographies.get(idx)
        projected = _project(matrix, *position) if matrix is not None else None
        pitch_x, pitch_y = projected if projected is not None else (None, None)
        rows.append({
            "frame_idx": idx,
            "timestamp_s": context.timestamps.get(idx, idx / context.fps),
            "image_x": position[0], "image_y": position[1],
            "pitch_x": pitch_x, "pitch_y": pitch_y,
            "ball_state": BallStateKind.OBSERVED.value,
        })
        ball_by_frame[idx] = (pitch_x, pitch_y, BallStateKind.OBSERVED, 1.0)
    context.ball = pd.DataFrame(
        rows,
        columns=["frame_idx", "timestamp_s", "image_x", "image_y",
                 "pitch_x", "pitch_y", "ball_state"],
    )
    context.ball_by_frame = ball_by_frame
    return context


# --------------------------------------------------------------------------- #
# Scoring one cell of the grid
# --------------------------------------------------------------------------- #


def score_cell(footages, caches, detections, references, conf: float) -> dict:
    """Perception and possession for one (detector, resolution, confidence)."""
    config = arms()["A_legacy"]

    counts = Counter()
    errors: list[float] = []
    miss_streaks: list[int] = []
    results = []
    per_sequence = []
    fusion_seconds = 0.0

    for footage in footages:
        name = footage["sequence"]
        window = footage["frame_indices"]
        truth = footage["truth"]

        filtered = {
            idx: [c for c in detections[name].get(idx, []) if c[3] >= conf]
            for idx in window
        }

        fusion = BallFusion(config)
        started = time.perf_counter()
        frames = fusion.run(
            filtered, window,
            camera_shifts=footage["shifts"],
            camera_confidence={i: 1.0 for i in footage["shifts"]},
        )
        fusion_seconds += time.perf_counter() - started

        positions: dict[int, tuple[float, float]] = {}
        streak = 0
        seq_counts = Counter()
        for idx in window:
            frame = frames[idx]
            observed = frame.kind.counts_as_observed
            gt = truth.get(idx)
            counts["frames"] += 1
            seq_counts["frames"] += 1
            if observed:
                positions[idx] = (frame.x, frame.y)
                counts["observed"] += 1
                seq_counts["observed"] += 1
                if streak:
                    miss_streaks.append(streak)
                    streak = 0
                if gt is not None:
                    distance = float(np.hypot(frame.x - gt[0], frame.y - gt[1]))
                    errors.append(distance)
                    for threshold in MATCH_THRESHOLDS:
                        if distance <= threshold:
                            counts[f"tp@{threshold}"] += 1
                            seq_counts[f"tp@{threshold}"] += 1
                    if distance > CANONICAL_MATCH_PX:
                        counts["fp"] += 1
                else:
                    counts["fp"] += 1
            else:
                streak += 1
            if gt is not None:
                counts["truth"] += 1
                seq_counts["truth"] += 1
        if streak:
            miss_streaks.append(streak)

        # -- downstream ------------------------------------------------------ #
        cached = caches[name]
        context = apply_ball(cached, positions, window)
        engine = PossessionEngine(context, PossessionConfig())
        decisions = engine.per_frame()
        determinable = sum(1 for d in decisions if d.state in DETERMINABLE)
        counts["determinable"] += determinable
        counts["unknown"] += len(decisions) - determinable

        predicted = {
            d.frame_idx: (d.state.value, d.team_id,
                          None if d.track_id is None else int(d.track_id))
            for d in decisions
        }
        result = evaluate_possession_vs_gt(
            references[name], predicted, cached["fps"], configuration=f"conf={conf}"
        )
        results.append(result)
        stability = switch_rate(decisions, cached["fps"])
        per_sequence.append({
            "sequence": name,
            "recall@25": round(seq_counts["tp@25.0"] / max(1, seq_counts["truth"]), 4),
            "coverage": round(seq_counts["observed"] / max(1, seq_counts["frames"]), 4),
            "determinability": round(determinable / max(1, len(decisions)), 4),
            "team_f1": round(result.team_f1, 4),
            "holder_accuracy": round(result.holder_accuracy, 4),
            "prediction_coverage": round(result.prediction_coverage, 4),
            **stability,
        })

    pooled = aggregate(results, "sweep")
    recall25 = counts["tp@25.0"] / max(1, counts["truth"])
    precision = counts["tp@25.0"] / max(1, counts["tp@25.0"] + counts["fp"])
    compared = sum(p["controlled_transitions_compared"] for p in per_sequence)
    flicker = sum(
        p["holder_switches_per_controlled_s"] * p["controlled_transitions_compared"]
        for p in per_sequence
    ) / max(1, compared)

    determinabilities = [p["determinability"] for p in per_sequence]
    return {
        "conf": conf,
        "recall@10": round(counts["tp@10.0"] / max(1, counts["truth"]), 4),
        "recall@15": round(counts["tp@15.0"] / max(1, counts["truth"]), 4),
        "recall@25": round(recall25, 4),
        "precision": round(precision, 4),
        "f1": round(
            2 * precision * recall25 / (precision + recall25), 4
        ) if precision + recall25 else 0.0,
        "coverage": round(counts["observed"] / max(1, counts["frames"]), 4),
        "false_positives_per_frame": round(counts["fp"] / max(1, counts["frames"]), 4),
        "localisation_error_px": {
            "median": round(float(np.median(errors)), 3) if errors else None,
            "p90": round(float(np.percentile(errors, 90)), 3) if errors else None,
            "p95": round(float(np.percentile(errors, 95)), 3) if errors else None,
            "n": len(errors),
        },
        "miss_streaks": {
            "median": float(np.median(miss_streaks)) if miss_streaks else 0.0,
            "p90": round(float(np.percentile(miss_streaks, 90)), 2) if miss_streaks else 0.0,
            "max": int(max(miss_streaks)) if miss_streaks else 0,
            "n": len(miss_streaks),
        },
        "possession_determinability": round(
            counts["determinable"] / max(1, counts["frames"]), 4
        ),
        "unknown_rate": round(counts["unknown"] / max(1, counts["frames"]), 4),
        "holder_accuracy": round(pooled["holder_accuracy"], 4),
        "team_f1_macro": round(pooled["team_f1_macro"], 4),
        "prediction_coverage": round(pooled["prediction_coverage"], 4),
        "holder_flicker_per_controlled_s": round(flicker, 4),
        "determinability_by_sequence": {
            "mean": round(float(np.mean(determinabilities)), 4),
            "median": round(float(np.median(determinabilities)), 4),
            "worst": round(float(np.min(determinabilities)), 4),
            "best": round(float(np.max(determinabilities)), 4),
            "stdev": round(float(np.std(determinabilities)), 4),
        },
        "fusion_ms_per_frame": round(
            1000 * fusion_seconds / max(1, counts["frames"]), 4
        ),
        "per_sequence": per_sequence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--resolutions", type=int, nargs="*", default=RESOLUTIONS)
    parser.add_argument("--confidences", type=float, nargs="*", default=CONFIDENCES)
    parser.add_argument("--conf-floor", type=float, default=CONF_FLOOR)
    parser.add_argument("--all-detectors", action="store_true",
                        help="evaluate every ball checkpoint on disk")
    parser.add_argument("--only", nargs="*", default=None,
                        help="evaluate just these checkpoint keys")
    parser.add_argument("--out", type=Path, default=RECORD)
    args = parser.parse_args()

    configure_logging("INFO")
    import torch
    from ultralytics import YOLO

    # -- provenance ---------------------------------------------------------- #
    perception = json.loads(PERCEPTION_RECORD.read_text(encoding="utf-8"))
    expected = list(perception["sequences_evaluated"])
    labels = {
        p.parent.name: p for p in sorted(GSR_ROOT.glob("**/Labels-GameState.json"))
    }
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    canonical = {
        k: {s["name"] for s in v} for k, v in manifest.items() if isinstance(v, list)
    }
    forbidden = sorted(
        s for s in expected
        if s in canonical.get("test", set()) | canonical.get("challenge", set())
    )
    if forbidden:
        raise SystemExit(f"ABORT - test/challenge contamination: {forbidden}")
    outside = sorted(s for s in expected if s not in canonical.get("validation", set()))
    if outside:
        raise SystemExit(f"ABORT - not in canonical validation: {outside}")
    fingerprint = hashlib.sha256("|".join(expected).encode()).hexdigest()[:16]
    if fingerprint != perception["split_fingerprint"]:
        raise SystemExit(
            f"ABORT - sequence set drift: {fingerprint} != {perception['split_fingerprint']}"
        )
    log.info("provenance ok: %d VALID sequences, fingerprint %s", len(expected), fingerprint)

    # -- footage and ground truth, built once -------------------------------- #
    log.info("loading footage and deriving references")
    footages = [sequence_footage(labels[s], args.max_frames) for s in expected]
    caches = {f["sequence"]: build_reference(f) for f in footages}
    references = {name: c["reference"] for name, c in caches.items()}

    grid: dict = {}
    runtime: dict = {}
    pool = ALL_DETECTORS if args.all_detectors else DETECTORS
    if args.only:
        missing = [k for k in args.only if k not in ALL_DETECTORS]
        if missing:
            raise SystemExit(f"unknown checkpoint key(s): {missing}")
        pool = {k: ALL_DETECTORS[k] for k in args.only}
    for detector, weights in pool.items():
        if not Path(weights).exists():
            log.warning("%s missing; skipped", detector)
            continue
        model = YOLO(weights)
        grid[detector] = {}
        runtime[detector] = {}
        for imgsz in args.resolutions:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            detections = {
                f["sequence"]: detect_sequence(model, f, imgsz, args.conf_floor)
                for f in footages
            }
            detect_seconds = time.perf_counter() - started
            n_frames = sum(len(f["frame_indices"]) for f in footages)
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            runtime[detector][str(imgsz)] = {
                "detector_ms_per_frame": round(1000 * detect_seconds / n_frames, 3),
                "peak_vram_gb": round(peak_gb, 3),
                "n_frames": n_frames,
            }
            log.info(
                "%s @ %d: %.1f ms/frame, peak VRAM %.2f GB",
                detector, imgsz, 1000 * detect_seconds / n_frames, peak_gb,
            )

            grid[detector][str(imgsz)] = {}
            for conf in args.confidences:
                cell = score_cell(footages, caches, detections, references, conf)
                grid[detector][str(imgsz)][str(conf)] = cell
                log.info(
                    "  conf %.2f: R@25 %.4f  P %.4f  cover %.4f  FP/fr %.3f  "
                    "determ %.4f  holder %.4f  teamF1 %.4f",
                    conf, cell["recall@25"], cell["precision"], cell["coverage"],
                    cell["false_positives_per_frame"],
                    cell["possession_determinability"], cell["holder_accuracy"],
                    cell["team_f1_macro"],
                )
        del model
        torch.cuda.empty_cache()

    # -- annotated-ball ceiling, same window --------------------------------- #
    log.info("scoring annotated-ball ceiling")
    ceiling_counts = Counter()
    ceiling_results = []
    ceiling_per_sequence = []
    for footage in footages:
        name = footage["sequence"]
        cached = caches[name]
        context = _restrict(
            context_from_gsr(footage["labels_path"]), footage["frame_indices"]
        )
        keep = set(footage["frame_indices"])
        context.ball = context.ball[
            context.ball.frame_idx.astype(int).isin(keep)
        ].reset_index(drop=True)
        context.ball_by_frame = {
            i: v for i, v in context.ball_by_frame.items() if i in keep
        }
        engine = PossessionEngine(context, PossessionConfig())
        decisions = engine.per_frame()
        determinable = sum(1 for d in decisions if d.state in DETERMINABLE)
        ceiling_counts["frames"] += len(decisions)
        ceiling_counts["determinable"] += determinable
        predicted = {
            d.frame_idx: (d.state.value, d.team_id,
                          None if d.track_id is None else int(d.track_id))
            for d in decisions
        }
        result = evaluate_possession_vs_gt(
            references[name], predicted, cached["fps"], configuration="ceiling"
        )
        ceiling_results.append(result)
        ceiling_per_sequence.append({
            "sequence": name,
            "determinability": round(determinable / max(1, len(decisions)), 4),
            "team_f1": round(result.team_f1, 4),
            "holder_accuracy": round(result.holder_accuracy, 4),
        })
    ceiling_pooled = aggregate(ceiling_results, "ceiling")
    ceiling = {
        "possession_determinability": round(
            ceiling_counts["determinable"] / max(1, ceiling_counts["frames"]), 4
        ),
        "holder_accuracy": round(ceiling_pooled["holder_accuracy"], 4),
        "team_f1_macro": round(ceiling_pooled["team_f1_macro"], 4),
        "prediction_coverage": round(ceiling_pooled["prediction_coverage"], 4),
        "per_sequence": ceiling_per_sequence,
    }

    payload = {
        "schema_version": "1.0.0",
        "what_this_is": (
            "detector x confidence x resolution sweep on canonical VALID, "
            "engine=legacy, scored on downstream possession determinability"
        ),
        "engine": "legacy",
        "split": "soccernet_gsr validation (canonical, leakage-guarded)",
        "split_fingerprint": fingerprint,
        "sequences_evaluated": expected,
        "max_frames_per_sequence": args.max_frames,
        "confidences": args.confidences,
        "resolutions": args.resolutions,
        "conf_floor_for_cache": args.conf_floor,
        "match_thresholds_px": MATCH_THRESHOLDS,
        "note": (
            "SN-GSR has no pass ground truth; no event metric is computed here. "
            "Detection ran once per (detector, resolution) at the confidence "
            "floor and every confidence level was derived offline, which is "
            "exact because NMS never lets a lower-scoring box suppress a "
            "higher-scoring one."
        ),
        "annotated_ball_ceiling": ceiling,
        "runtime": runtime,
        "grid": grid,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
