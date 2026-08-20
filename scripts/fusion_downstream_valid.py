"""Downstream VALID-only A/B: legacy vs temporal ball engine.

The perception ablation (``scripts/fusion_ablation.py``) answers "does temporal
fusion find the ball more often". This answers the question that actually
decides promotion: **what happens to the product when it does**.

Design
------
The same 8 canonical VALID sequences, the same frame window, the same detector
candidates. The detector runs once per checkpoint and its candidates are cached;
each arm then re-scores those identical candidates through a different fusion
configuration. Everything downstream of fusion -- the Phase-2D possession engine
and the event engine -- is imported unchanged and given identical player, team
and calibration inputs taken from the GSR annotation.

So the only thing that differs between arm A and arm B is the ball. That is the
point: any downstream movement is attributable to the ball engine and to nothing
else.

Arms
----
``A_legacy``
    ``BallFusionConfig(engine="legacy")``. The production runner skips the
    temporal layer entirely for this engine, so the faithful offline analogue is
    the permissive configuration -- no suppression, no temporal verification.
    This is the same configuration the perception ablation scores as
    ``1_no_fusion``.

``B_temporal``
    ``build_fusion_config(BallFusionConfig(engine="temporal"))`` with the shipped
    defaults, read from the production config object rather than restated here,
    so this cannot drift from what would actually ship.

``reference_annotated_ball``
    The annotated ball, as an upper bound. Not an arm -- it is the ceiling both
    arms are trying to reach, and it makes the size of the perception gap legible.

What this harness deliberately does not do
------------------------------------------
It does not run the full Phase-1 pipeline. The pipeline ingests video via
``probe_video``/``VideoReader``; the GSR VALID sequences are ``img1/`` JPEG
directories, so there is no canonical pipeline path to them. Re-encoding them to
video would put a lossy transcode inside the measurement, and writing an image
ingestion path would add a second production code path -- both are exactly the
kind of divergence that silently corrupts a decisive measurement. Holding
players/teams/calibration at annotation quality is the stricter test anyway: it
removes every confound except the ball.

Ball pitch coordinates
----------------------
There is no stored homography for these sequences, and the ball-dependent event
family (pass, carry, turnover, interception, recovery) is gated on
``ctx.ball_position``, which is a *pitch* position. Leaving it ``None`` makes
every one of those events structurally impossible in both arms -- a harness
artifact that would masquerade as a finding.

So the ball is projected instead. GSR annotates every player with both an image
foot point and a pitch position, which is an image->pitch correspondence set on
every frame; a RANSAC homography is fitted from those and the **fused** ball is
projected through it. No ball ground truth enters: the annotation supplies the
mapping, exactly as it already supplies players, teams and calibration, and the
ball position being mapped is whichever arm's fusion produced it.

``ball_state`` stays ``OBSERVED`` whenever fusion observed the ball, regardless
of where the projection lands, so determinability continues to measure the ball
engine rather than the quality of the projection.

Usage::

    python scripts/fusion_downstream_valid.py
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion_ablation import GSR_ROOT, SPLIT_MANIFEST, collect  # noqa: E402

from visionpitch.analytics import events as events_module  # noqa: E402
from visionpitch.analytics.possession import (  # noqa: E402
    PossessionConfig,
    PossessionEngine,
)
from visionpitch.analytics.types import BallStateKind, PossessionState  # noqa: E402
from visionpitch.ball_tracking.fp_filter import FilterConfig  # noqa: E402
from visionpitch.ball_tracking.fusion import (  # noqa: E402
    BallFusion,
    FusionConfig,
    SuppressionMethod,
)
from visionpitch.common.config import BallFusionConfig  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.possession_eval import (  # noqa: E402
    aggregate,
    context_from_gsr,
    evaluate_possession_vs_gt,
)
from visionpitch.evaluation.possession_gt import (  # noqa: E402
    DerivationParams,
    derive_from_gsr,
)
from visionpitch.pipeline.ball_fusion import build_fusion_config  # noqa: E402

log = get_logger("fusion.downstream")

PERCEPTION_RECORD = Path("data/eval/fusion/ablation_valid.json")
RECORD = Path("data/eval/fusion/downstream_valid.json")

DETECTORS = {
    "A_default": "models/yolo-football-ball-detection.pt",
    "C_adapt": "models/finetune/bcast_adapt/weights/best.pt",
}

#: Possession states on which the engine has committed to an answer. Taken from
#: scripts/possession_determinability.py unchanged, so "determinability" means
#: here exactly what it means in the Phase-2D report.
DETERMINABLE = {
    PossessionState.CONTROLLED,
    PossessionState.CONTESTED,
    PossessionState.LOOSE_BALL,
    PossessionState.OUT_OF_PLAY,
}


def arms() -> dict[str, FusionConfig]:
    """The two engines, both derived from the production config object."""
    temporal_block = BallFusionConfig(engine="temporal")
    return {
        # engine="legacy": the runner bypasses the temporal layer entirely.
        "A_legacy": FusionConfig(
            suppression=SuppressionMethod.NONE,
            temporal=FilterConfig(
                min_support_frames=0,
                trust_confidence=0.0,
                max_step_px=1e9,
                camera_motion_px=1e9,
                max_size_ratio=1e9,
            ),
        ),
        "B_temporal": build_fusion_config(temporal_block),
    }


def fused_ball(
    entry: dict, config: FusionConfig
) -> tuple[dict[int, tuple[float, float]], float]:
    """Ball image positions per frame for one sequence, plus fusion seconds."""
    fusion = BallFusion(config)
    started = time.perf_counter()
    frames = fusion.run(
        entry["detections"],
        entry["frame_indices"],
        camera_shifts=entry["shifts"],
        camera_confidence={i: 1.0 for i in entry["shifts"]},
    )
    elapsed = time.perf_counter() - started
    return (
        {
            idx: (frames[idx].x, frames[idx].y)
            for idx in entry["frame_indices"]
            if frames[idx].kind.counts_as_observed
        },
        elapsed,
    )


def _restrict(context, window: list[int]):
    """Restrict a context to the evaluated frame window, in place."""
    keep = set(window)
    context.players = context.players[
        context.players.frame_idx.astype(int).isin(keep)
    ].reset_index(drop=True)
    context.valid_players = context.players
    context.frames = context.frames[
        context.frames.frame_idx.astype(int).isin(keep)
    ].reset_index(drop=True)
    context.timestamps = {i: t for i, t in context.timestamps.items() if i in keep}
    return context


def _homographies(context) -> dict[int, np.ndarray]:
    """Per-frame image->pitch homography fitted from annotated player feet.

    The annotation gives both sides of the correspondence for every player, so
    this needs no ball information at all. RANSAC because a few players carry
    noisy pitch coordinates near the frame edges.
    """
    players = context.players
    if not len(players) or "pitch_x" not in players.columns:
        return {}
    usable = players.dropna(subset=["pitch_x", "pitch_y"])
    out: dict[int, np.ndarray] = {}
    for frame_idx, group in usable.groupby(usable.frame_idx.astype(int)):
        if len(group) < 4:
            continue
        source = np.column_stack([
            group.image_x.to_numpy(dtype=np.float64),
            group.image_y.to_numpy(dtype=np.float64),
        ])
        target = np.column_stack([
            group.pitch_x.to_numpy(dtype=np.float64),
            group.pitch_y.to_numpy(dtype=np.float64),
        ])
        matrix, _ = cv2.findHomography(source, target, cv2.RANSAC, 1.0)
        if matrix is not None:
            out[int(frame_idx)] = matrix
    return out


def _project(matrix: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    point = matrix @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(point[2]) < 1e-9:
        return None
    return float(point[0] / point[2]), float(point[1] / point[2])


def context_with_ball(
    labels_path: Path, positions: dict[int, tuple[float, float]], window: list[int]
):
    """The GSR context with its ball replaced by a fused one.

    Players, teams and calibration stay at annotation quality; only the ball
    moves between arms.
    """
    context = _restrict(context_from_gsr(labels_path), window)
    homographies = _homographies(context)

    rows = []
    ball_by_frame: dict[int, tuple] = {}
    for idx in window:
        position = positions.get(idx)
        if position is None:
            continue
        matrix = homographies.get(idx)
        projected = _project(matrix, *position) if matrix is not None else None
        pitch_x, pitch_y = projected if projected is not None else (None, None)
        rows.append(
            {
                "frame_idx": idx,
                "timestamp_s": context.timestamps.get(idx, idx / context.fps),
                "image_x": position[0],
                "image_y": position[1],
                "pitch_x": pitch_x,
                "pitch_y": pitch_y,
                "ball_state": BallStateKind.OBSERVED.value,
            }
        )
        ball_by_frame[idx] = (pitch_x, pitch_y, BallStateKind.OBSERVED, 1.0)

    context.ball = pd.DataFrame(
        rows,
        columns=[
            "frame_idx", "timestamp_s", "image_x", "image_y",
            "pitch_x", "pitch_y", "ball_state",
        ],
    )
    context.ball_by_frame = ball_by_frame
    return context


def annotated_context(labels_path: Path, window: list[int]):
    """The annotated-ball ceiling, restricted to the same frame window."""
    context = _restrict(context_from_gsr(labels_path), window)
    keep = set(window)
    context.ball = context.ball[
        context.ball.frame_idx.astype(int).isin(keep)
    ].reset_index(drop=True)
    context.ball_by_frame = {
        i: v for i, v in context.ball_by_frame.items() if i in keep
    }
    return context


def clip_reference(gt, start_s: float, end_s: float):
    """Restrict the derived reference to the evaluated window."""
    kept = []
    for interval in gt.intervals:
        if interval.end_s <= start_s or interval.start_s >= end_s:
            continue
        interval.start_s = max(interval.start_s, start_s)
        interval.end_s = min(interval.end_s, end_s)
        kept.append(interval)
    gt.intervals = kept
    return gt


def switch_rate(decisions, fps: float) -> dict:
    """Frame-to-frame instability of the committed possession answer.

    Not a pre-existing Phase-2D metric -- the shipped engine absorbs flicker in
    ``spans()`` via ``min_span_s``, so no per-frame stability figure existed.
    Computed identically for both arms and reported as derived.
    """
    team_switches = holder_switches = comparable = 0
    previous_team = previous_holder = None
    for decision in decisions:
        if decision.state is not PossessionState.CONTROLLED:
            previous_team = previous_holder = None
            continue
        if previous_team is not None:
            comparable += 1
            team_switches += int(decision.team_id != previous_team)
            holder_switches += int(decision.track_id != previous_holder)
        previous_team = decision.team_id
        previous_holder = decision.track_id
    seconds = comparable / fps if comparable else 0.0
    return {
        "controlled_transitions_compared": comparable,
        "team_switches_per_controlled_s": (
            round(team_switches / seconds, 4) if seconds else 0.0
        ),
        "holder_switches_per_controlled_s": (
            round(holder_switches / seconds, 4) if seconds else 0.0
        ),
    }


def score_arm(
    labels: dict[str, Path],
    data: list[dict],
    config: FusionConfig | None,
    arm: str,
    annotated: bool = False,
) -> dict:
    params = DerivationParams()
    results = []
    per_sequence = []
    totals: Counter = Counter()
    fusion_seconds = 0.0
    analytics_started = time.perf_counter()
    event_counts: Counter = Counter()
    n_spans = 0

    for entry in data:
        name = entry["sequence"]
        window = entry["frame_indices"]
        labels_path = labels[name]

        if annotated:
            context = annotated_context(labels_path, window)
        else:
            positions, fusion_s = fused_ball(entry, config)
            fusion_seconds += fusion_s
            context = context_with_ball(labels_path, positions, window)

        engine = PossessionEngine(context, PossessionConfig())
        decisions = engine.per_frame()
        spans = engine.spans(decisions)
        n_spans += len(spans)

        determinable = sum(1 for d in decisions if d.state in DETERMINABLE)
        unknown = sum(1 for d in decisions if d.state is PossessionState.UNKNOWN)
        totals["frames"] += len(decisions)
        totals["determinable"] += determinable
        totals["unknown"] += unknown
        totals["controlled"] += sum(
            1 for d in decisions if d.state is PossessionState.CONTROLLED
        )
        totals["ball_known"] += sum(1 for d in decisions if d.ball_state.is_known)

        for event in events_module.run(context, spans):
            event_counts[event.event_type.value] += 1

        fps = context.fps
        gt = clip_reference(
            derive_from_gsr(labels_path, params),
            min(window) / fps,
            (max(window) + 1) / fps,
        )
        predicted = {
            d.frame_idx: (
                d.state.value,
                d.team_id,
                None if d.track_id is None else int(d.track_id),
            )
            for d in decisions
        }
        result = evaluate_possession_vs_gt(gt, predicted, fps, configuration=arm)
        results.append(result)
        stability = switch_rate(decisions, fps)
        per_sequence.append(
            {
                **result.to_dict(),
                "frames": len(decisions),
                "determinable": determinable,
                "unknown": unknown,
                **stability,
            }
        )
        log.info(
            "  %-26s %s: determinability %.4f  unknown %.4f  teamF1 %.4f  holder %.4f",
            arm, name,
            determinable / max(1, len(decisions)),
            unknown / max(1, len(decisions)),
            result.team_f1, result.holder_accuracy,
        )

    analytics_seconds = time.perf_counter() - analytics_started
    pooled = aggregate(results, arm)
    compared = sum(p["controlled_transitions_compared"] for p in per_sequence)
    stability = {
        key: round(
            sum(p[key] * p["controlled_transitions_compared"] for p in per_sequence)
            / max(1, compared),
            4,
        )
        for key in (
            "team_switches_per_controlled_s",
            "holder_switches_per_controlled_s",
        )
    }
    return {
        "arm": arm,
        "fusion_config_fingerprint": config.fingerprint() if config else None,
        "suppression": config.suppression.value if config else "annotated",
        "n_frames": totals["frames"],
        "possession_determinability": round(
            totals["determinable"] / max(1, totals["frames"]), 4
        ),
        "unknown_rate": round(totals["unknown"] / max(1, totals["frames"]), 4),
        "controlled_rate": round(totals["controlled"] / max(1, totals["frames"]), 4),
        "ball_known_rate": round(totals["ball_known"] / max(1, totals["frames"]), 4),
        "possession_vs_reference": pooled,
        "stability": stability,
        "n_possession_spans": n_spans,
        "events": dict(sorted(event_counts.items())),
        "fusion_seconds": round(fusion_seconds, 4),
        "analytics_seconds": round(analytics_seconds, 4),
        "fusion_ms_per_frame": round(
            1000 * fusion_seconds / max(1, totals["frames"]), 4
        ),
        "per_sequence": per_sequence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--out", type=Path, default=RECORD)
    args = parser.parse_args()

    configure_logging("INFO")
    from ultralytics import YOLO

    # -- provenance, asserted before anything is measured -------------------- #
    if not PERCEPTION_RECORD.exists():
        raise SystemExit(f"ABORT - perception record missing: {PERCEPTION_RECORD}")
    perception = json.loads(PERCEPTION_RECORD.read_text(encoding="utf-8"))
    expected = list(perception["sequences_evaluated"])

    labels = {
        p.parent.name: p for p in sorted(GSR_ROOT.glob("**/Labels-GameState.json"))
    }
    if not labels:
        raise SystemExit(f"No SN-GSR sequences found under {GSR_ROOT}")

    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    canonical = {
        key: {s["name"] for s in value}
        for key, value in manifest.items()
        if isinstance(value, list)
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
    missing = sorted(s for s in expected if s not in labels)
    if missing:
        raise SystemExit(f"ABORT - perception sequences absent from VALID root: {missing}")

    fingerprint = hashlib.sha256("|".join(expected).encode()).hexdigest()[:16]
    if fingerprint != perception["split_fingerprint"]:
        raise SystemExit(
            "ABORT - sequence set differs from the perception ablation: "
            f"{fingerprint} != {perception['split_fingerprint']}"
        )
    log.info(
        "provenance ok: %d canonical VALID sequences, fingerprint %s",
        len(expected), fingerprint,
    )

    chosen = [labels[s] for s in expected]
    configs = arms()
    results: dict[str, dict] = {}

    for detector, weights in DETECTORS.items():
        path = Path(weights)
        if not path.exists():
            log.warning("%s missing; skipped", detector)
            continue
        log.info("caching candidates for %s", detector)
        model = YOLO(str(path))
        started = time.perf_counter()
        data = collect(model, chosen, args.conf, args.imgsz, args.max_frames)
        detect_seconds = time.perf_counter() - started
        del model

        results[detector] = {"detector_seconds": round(detect_seconds, 2)}
        for arm, config in configs.items():
            results[detector][arm] = score_arm(labels, data, config, arm)
        results[detector]["reference_annotated_ball"] = score_arm(
            labels, data, None, "reference_annotated_ball", annotated=True
        )

    payload = {
        "schema_version": "1.0.0",
        "what_this_is": (
            "downstream VALID-only A/B of the ball engine; players, teams and "
            "calibration are held at annotation quality so the only variable is "
            "the ball"
        ),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "split": "soccernet_gsr validation (canonical, leakage-guarded)",
        "split_fingerprint": fingerprint,
        "sequences_evaluated": expected,
        "n_sequences": len(expected),
        "max_frames_per_sequence": args.max_frames,
        "matches_perception_record": str(PERCEPTION_RECORD),
        "limitations": [
            "ball pitch coordinates come from a per-frame homography fitted to "
            "annotated player feet, not from a stored calibration; no ball "
            "ground truth is used and both arms are projected identically",
            "no pass/event ground truth exists for SN-GSR, so event counts are "
            "reported without precision/recall",
        ],
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
