"""Possession determinability, broken down by what the ball evidence was.

Phase 2D, Part 8.

"Possession is determinable on 12% of frames" is the number that caps every
event metric, and on its own it explains nothing. This report splits it by the
kind of ball evidence behind each frame and by whether the ball could have been
seen at all, so the 88% of frames where possession is *not* determinable can be
attributed rather than lamented:

* the detector saw the ball and possession was still undecidable
* the ball position was interpolated, and the engine declined to trust it
* the ball was genuinely unobservable -- occluded, out of frame, off pitch
* no ball evidence of any kind

Runs on a completed run directory, so it needs no GPU and no video.

Usage::

    python scripts/possession_determinability.py --run outputs_bas/<video>/<fingerprint>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.analytics.context import load_context  # noqa: E402
from visionpitch.analytics.possession import PossessionEngine  # noqa: E402
from visionpitch.analytics.types import BallStateKind, PossessionState  # noqa: E402
from visionpitch.ball_tracking.observability import (  # noqa: E402
    ObservabilityEstimator,
)
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("possession.determinability")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", type=Path, default=Path("data/eval/determinability"))
    args = parser.parse_args()

    configure_logging("INFO")
    context = load_context(args.run)
    engine = PossessionEngine(context)
    decisions = engine.per_frame()
    # The pre-existing metric, and the one the Phase 2D threshold was declared
    # against: the share of match time a team was in *controlled* possession,
    # measured on smoothed spans. A per-frame "the engine committed to some
    # state" figure is a different, much larger number, and quoting that one
    # against a threshold set for this one would be moving the goalposts.
    spans = engine.spans(decisions)
    engine_summary = engine.summary(spans)

    # -- observability from stored tables ------------------------------------- #
    players = context.players
    player_boxes: dict[int, np.ndarray] = {}
    for frame_idx, group in players.groupby(players.frame_idx.astype(int)):
        player_boxes[int(frame_idx)] = np.column_stack([
            group.bbox_x1.to_numpy(), group.bbox_y1.to_numpy(),
            group.bbox_x2.to_numpy(), group.bbox_y2.to_numpy(),
        ])

    observed_positions = {
        int(row.frame_idx): (float(row.image_x), float(row.image_y))
        for row in context.ball.itertuples(index=False)
        if not bool(getattr(row, "interpolated", False))
    }

    frames_table = context.frames
    calibration_confidence = (
        dict(zip(
            frames_table.frame_idx.astype(int),
            frames_table.calibration_confidence.astype(float),
            strict=True,
        ))
        if "calibration_confidence" in frames_table.columns else {}
    )

    width = height = None
    if {"bbox_x2", "bbox_y2"}.issubset(players.columns) and len(players):
        width = int(players.bbox_x2.max() * 1.02)
        height = int(players.bbox_y2.max() * 1.02)
    frame_size = (width or 1280, height or 720)

    report = ObservabilityEstimator().label_sequence(
        frame_indices=context.frame_indices,
        frame_size=frame_size,
        ball_observations=observed_positions,
        player_boxes_by_frame=player_boxes,
        calibration_confidence_by_frame=calibration_confidence,
    )

    # -- cross-tabulate -------------------------------------------------------- #
    determinable_states = {
        PossessionState.CONTROLLED, PossessionState.CONTESTED,
        PossessionState.LOOSE_BALL, PossessionState.OUT_OF_PLAY,
    }
    by_ball_state: dict[str, Counter] = {}
    by_observability: dict[str, Counter] = {}
    totals = Counter()

    for decision in decisions:
        determinable = decision.state in determinable_states
        controlled = decision.state is PossessionState.CONTROLLED

        ball_kind = decision.ball_state.value
        bucket = by_ball_state.setdefault(ball_kind, Counter())
        bucket["frames"] += 1
        bucket["determinable"] += int(determinable)
        bucket["controlled"] += int(controlled)

        state = report.state_of(decision.frame_idx).value
        observability_bucket = by_observability.setdefault(state, Counter())
        observability_bucket["frames"] += 1
        observability_bucket["determinable"] += int(determinable)
        observability_bucket["controlled"] += int(controlled)

        totals["frames"] += 1
        totals["determinable"] += int(determinable)
        totals["controlled"] += int(controlled)

    fair = report.fair_frames
    fair_determinable = sum(
        1 for d in decisions
        if d.frame_idx in fair and d.state in determinable_states
    )

    payload = {
        "run": str(args.run),
        "label": args.label or args.run.name,
        "n_frames": totals["frames"],
        # PRIMARY: the pre-existing definition, unchanged, comparable to every
        # figure published in Phase 2B and 2C.
        "determinability": engine_summary["determinable_ratio"],
        "determinability_definition": (
            "controlled_s / total_s on smoothed spans -- the share of match time "
            "a team was in controlled possession"
        ),
        "unknown_ratio": engine_summary["unknown_ratio"],
        # SECONDARY, and deliberately named differently so it can never be
        # mistaken for the metric above.
        "state_committed_fraction": round(
            totals["determinable"] / max(1, totals["frames"]), 4
        ),
        "state_committed_definition": (
            "per-frame share where the engine committed to ANY state including "
            "loose and out-of-play; always much larger, not comparable"
        ),
        "controlled_fraction": round(totals["controlled"] / max(1, totals["frames"]), 4),
        "n_observable_frames": len(fair),
        "observable_fraction": round(len(fair) / max(1, totals["frames"]), 4),
        "state_committed_on_observable_frames": round(
            fair_determinable / max(1, len(fair)), 4
        ),
        "ball_coverage_direct": round(
            sum(
                1 for d in decisions if d.ball_state is BallStateKind.OBSERVED
            ) / max(1, totals["frames"]), 4
        ),
        "by_ball_evidence": {
            kind: {
                "frames": counts["frames"],
                "share_of_frames": round(counts["frames"] / max(1, totals["frames"]), 4),
                "determinability": round(
                    counts["determinable"] / max(1, counts["frames"]), 4
                ),
                "controlled_rate": round(
                    counts["controlled"] / max(1, counts["frames"]), 4
                ),
            }
            for kind, counts in sorted(by_ball_state.items())
        },
        "by_observability": {
            state: {
                "frames": counts["frames"],
                "share_of_frames": round(counts["frames"] / max(1, totals["frames"]), 4),
                "determinability": round(
                    counts["determinable"] / max(1, counts["frames"]), 4
                ),
            }
            for state, counts in sorted(by_observability.items())
        },
        "note": (
            "determinability counts frames where the engine committed to any "
            "possession state. Frames the observability model rules unobservable "
            "are reported separately rather than folded into the failure."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"determinability_{payload['label']}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({
        k: payload[k] for k in (
            "label", "n_frames", "ball_coverage_direct", "determinability",
            "unknown_ratio", "state_committed_fraction", "observable_fraction",
            "state_committed_on_observable_frames",
        )
    }, indent=2))
    print("\nby ball evidence:")
    for kind, block in payload["by_ball_evidence"].items():
        print(f"  {kind:<14} {block['frames']:>6} frames "
              f"({block['share_of_frames']:.3f})  determinable {block['determinability']:.3f}")
    print("\nby observability:")
    for state, block in payload["by_observability"].items():
        print(f"  {state:<26} {block['frames']:>6} frames "
              f"({block['share_of_frames']:.3f})  determinable {block['determinability']:.3f}")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
