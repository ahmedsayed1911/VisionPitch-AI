"""Evaluate possession and events against SoccerNet Ball Action Spotting.

Phase 2B, Parts 3, 4 and 8. Runs the analytics stage over a completed vision
run and scores the resulting events against expert ground truth.

The ground-truth timestamps are absolute within the source video, while the run
covers a segment of it, so ``--offset`` shifts predicted timestamps back into
the source clock. Getting this wrong silently produces a recall of zero, so the
script reports the alignment it used and the ground-truth events inside the
window before it scores anything.

Usage::

    python scripts/evaluate_events.py --run outputs_bas/<video>/<fp> \\
        --gt data/eval/bas/event_gt_half1.json --offset 600 --label baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.analytics import events as event_module  # noqa: E402
from visionpitch.analytics import possession as possession_module  # noqa: E402
from visionpitch.analytics.context import load_context  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.event_gt import EventGroundTruth  # noqa: E402
from visionpitch.evaluation.event_metrics import (  # noqa: E402
    DEFAULT_TOLERANCES_S,
    evaluate_events,
)

log = get_logger("evaluation.events")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--offset", type=float, default=0.0,
                        help="seconds from the source video start to the run start")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    configure_logging("WARNING")

    ground_truth = EventGroundTruth.load(args.gt)
    context = load_context(args.run)
    spans, _, possession_summary = possession_module.run(context)
    predictions = event_module.run(context, spans)

    times = sorted(context.timestamps.values())
    window = (args.offset + times[0], args.offset + times[-1])
    in_window = [
        e for e in ground_truth.scorable()
        if window[0] <= e.start_time_s <= window[1]
    ]

    print(f"run      : {args.run.name}")
    print(f"window   : {window[0]:.1f}s .. {window[1]:.1f}s of the source video")
    print(f"gt events inside window: {len(in_window)} of {len(ground_truth.scorable())}")
    print(f"predicted events       : {len(predictions)}")
    if not in_window:
        print("\nNo ground-truth events inside this window -- check --offset.")
        return 1

    from collections import Counter
    print(f"gt in window by type   : {dict(Counter(e.event_type.value for e in in_window))}")
    print(f"predicted by type      : "
          f"{dict(Counter(p.event_type.value for p in predictions))}\n")

    report = evaluate_events(
        ground_truth, predictions,
        tolerances_s=DEFAULT_TOLERANCES_S, time_offset_s=args.offset,
        window_s=window,
    )
    report["label"] = args.label
    report["run_dir"] = str(args.run)
    report["window_s"] = list(window)
    report["possession_summary"] = possession_summary
    report["ball_quality"] = {
        "observed_pct": round(100 * context.ball_observed_coverage, 2),
        "known_pct": round(100 * context.ball_coverage, 2),
        "unknown_pct": round(100 * (1 - context.ball_coverage), 2),
    }
    report["vision_manifest"] = {
        "config_fingerprint": context.manifest.get("config_fingerprint"),
        "models": {
            k: v.get("weights_sha256")
            for k, v in (context.manifest.get("models") or {}).items()
        },
    }

    # Headline table at the tolerance most relevant to possession chains.
    headline_tolerance = 0.40
    print(f"{'event':16s} {'P':>7s} {'R':>7s} {'F1':>7s} {'nGT':>5s} {'nPred':>6s} "
          f"{'medErr':>7s}   (tolerance {headline_tolerance}s)")
    for event_type, entries in sorted(report["per_event_type"].items()):
        row = next((e for e in entries if e["tolerance_s"] == headline_tolerance), None)
        if row is None:
            continue
        def fmt(value) -> str:
            return f"{value:.3f}" if isinstance(value, (int, float)) else "-"

        print(f"{event_type:16s} {fmt(row['precision']['value']):>7} "
              f"{fmt(row['recall']['value']):>7} {fmt(row['f1']):>7} "
              f"{row['n_ground_truth']:>5d} {row['n_predicted']:>6d} "
              f"{fmt(row['temporal_error_s']['median']):>7}")

    print(f"\nball observed {report['ball_quality']['observed_pct']}%  "
          f"unknown {report['ball_quality']['unknown_pct']}%  "
          f"possession determinable "
          f"{100 * possession_summary.get('determinable_ratio', 0):.1f}%")
    if not ground_truth.has_player_identity:
        print("\nplayer attribution: NOT MEASURABLE against this corpus (no player labels)")

    out = args.out or (
        Path("data/eval/bas/benchmarks") / f"events_{args.label}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
