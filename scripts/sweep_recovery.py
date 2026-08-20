"""Sweep track-before-detect parameters on the TRAIN sequences.

Phase 2D, Part 4.

The first held-out measurement of the default recovery configuration returned
248 recoveries at **57.4% accuracy** -- 87 of them demonstrably in the wrong
place. A stage that fills gaps with positions that are wrong half the time
raises every coverage number and corrupts possession, which is the precise
failure mode this milestone forbids.

So the parameters are swept here, on training sequences, against a target that
is *accuracy first*: yield is worthless if the recovered positions are wrong.
The selection rule is declared before the sweep runs and is applied
mechanically:

    among configurations reaching at least ``--min-accuracy`` recovery accuracy,
    take the one with the highest yield; if none reaches it, recommend that the
    stage stay disabled.

That last clause matters. "No configuration is good enough" is a permitted and
expected outcome, not a failure of the sweep.

Usage::

    python scripts/sweep_recovery.py --sequences 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("ball.recovery.sweep")

#: Deliberately small. Each combination re-runs the detector over every frame,
#: so a wide grid costs hours of GPU for a stage whose first held-out accuracy
#: was 0.57 -- a long way from the 0.85 bar. The three axes here are the ones
#: that plausibly separate a real ball from a distractor: how much temporal
#: support is demanded, how far the evidence must stand out from its
#: surroundings, and how close to the predicted path it must land.
GRID = {
    "min_supporting_frames": [3, 5],
    "min_response_ratio": [3.0, 8.0],
    "max_deviation_px": [36.0, 12.0],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--model", default="models/finetune/ball_multicorpus/weights/best.pt")
    parser.add_argument(
        "--min-accuracy", type=float, default=0.85,
        help="recovery accuracy required before the stage may be enabled",
    )
    parser.add_argument("--out", type=Path, default=Path("data/eval/ball_temporal"))
    args = parser.parse_args()

    configure_logging("INFO")
    scratch = args.out / "sweep"
    scratch.mkdir(parents=True, exist_ok=True)

    rows = []
    combinations = list(itertools.product(*GRID.values()))
    for index, values in enumerate(combinations):
        params = dict(zip(GRID, values, strict=True))
        label = f"sweep{index:02d}"
        command = [
            sys.executable, "scripts/evaluate_ball_temporal.py",
            "--split", "train",
            "--max-sequences", str(args.sequences),
            "--max-frames", str(args.max_frames),
            "--model", args.model,
            "--label", label,
            "--out", str(scratch),
            "--min-supporting-frames", str(params["min_supporting_frames"]),
            "--min-response-ratio", str(params["min_response_ratio"]),
            "--max-deviation-px", str(params["max_deviation_px"]),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(
            (scratch / f"temporal_{label}_recovery.json").read_text(encoding="utf-8")
        )
        pooled = payload["pooled"]
        row = {
            **params,
            "n_recovered": pooled["n_recovered"],
            "n_gap_frames": pooled["n_gap_frames"],
            "recovery_yield": pooled["recovery_yield"],
            "recovery_accuracy": pooled["recovery_accuracy"],
            "n_recovery_wrong": pooled["n_recovery_wrong"],
            "n_recovery_unverifiable": pooled["n_recovery_unverifiable"],
            "coverage_direct": pooled["coverage_direct"],
            "coverage_direct_plus_recovered": pooled["coverage_direct_plus_recovered"],
        }
        rows.append(row)
        log.info(
            "%2d/%d  frames>=%d ratio>=%.1f dev<=%.0f  ->  recovered %3d  "
            "yield %.3f  accuracy %s",
            index + 1, len(combinations), params["min_supporting_frames"],
            params["min_response_ratio"], params["max_deviation_px"],
            row["n_recovered"], row["recovery_yield"], row["recovery_accuracy"],
        )

    eligible = [
        r for r in rows
        if r["recovery_accuracy"] is not None
        and r["recovery_accuracy"] >= args.min_accuracy
        and r["n_recovered"] >= 20  # too few recoveries to estimate accuracy from
    ]
    best = max(eligible, key=lambda r: r["recovery_yield"]) if eligible else None

    payload = {
        "split": "train",
        "n_sequences": args.sequences,
        "max_frames_per_sequence": args.max_frames,
        "model": args.model,
        "min_accuracy_required": args.min_accuracy,
        "selection_rule": (
            "highest yield among configurations with recovery accuracy >= "
            "min_accuracy and at least 20 recoveries; declared before the sweep"
        ),
        "grid": GRID,
        "rows": rows,
        "selected": best,
        "recommendation": (
            "enable recovery with the selected parameters"
            if best else
            "keep track-before-detect DISABLED: no configuration reached the "
            "required recovery accuracy on the training sequences"
        ),
    }
    destination = args.out / "recovery_sweep_train.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'frames':>7}{'ratio':>7}{'dev':>6}{'recovered':>11}{'yield':>8}{'accuracy':>10}")
    for row in rows:
        accuracy = "n/a" if row["recovery_accuracy"] is None else f"{row['recovery_accuracy']:.3f}"
        print(f"{row['min_supporting_frames']:>7}{row['min_response_ratio']:>7.1f}"
              f"{row['max_deviation_px']:>6.0f}{row['n_recovered']:>11}"
              f"{row['recovery_yield']:>8.3f}{accuracy:>10}")
    print(f"\n{payload['recommendation']}")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
