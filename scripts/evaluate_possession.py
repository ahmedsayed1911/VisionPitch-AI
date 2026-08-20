"""Measure the possession engine against the derived GSR reference.

Phase 2C, Parts 4 and 5.

Runs on the **test** sequences of the Phase 2C clip-disjoint split only. The
train and val sequences were used to choose the derivation thresholds, so
scoring on them would report how well the thresholds fit the data they came
from.

Usage::

    python scripts/evaluate_possession.py --out data/eval/possession
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.analytics.possession import (  # noqa: E402
    PossessionConfig,
    PossessionEngine,
)
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
from visionpitch.evaluation.registry import build_split  # noqa: E402

log = get_logger("possession.eval")

GSR_ROOT = Path("data/eval/gsr")


def sequence_labels() -> dict[str, Path]:
    return {
        path.parent.name: path
        for path in sorted(GSR_ROOT.glob("**/Labels-GameState.json"))
    }


def run_sweep(labels, chosen, params, args) -> int:
    """Sweep the engine's control radius against the reference.

    Run on the training sequences only. The chosen value is then measured once
    on the held-out test sequences, which is the only number that may be quoted.

    What this calibrates and what it does not: the engine's control radius was
    never fitted to anything, and this fits it to annotated geometry. It aligns
    the engine's image-space rule with the reference's metric rule. It does not
    make either of them a better description of football -- both still assume
    the nearest player owns the ball.
    """
    grid = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]
    references = {name: derive_from_gsr(labels[name], params) for name in chosen}
    contexts = {name: context_from_gsr(labels[name]) for name in chosen}

    rows = []
    for radius in grid:
        config = PossessionConfig(control_radius_heights=radius)
        config.loose_radius_heights = max(config.loose_radius_heights, radius * 2.5)
        results = []
        for name in chosen:
            engine = PossessionEngine(contexts[name], config)
            predicted = {
                frame.frame_idx: (
                    frame.state.value, frame.team_id,
                    None if frame.track_id is None else int(frame.track_id),
                )
                for frame in engine.per_frame()
            }
            results.append(
                evaluate_possession_vs_gt(
                    references[name], predicted, references[name].fps,
                    configuration="ground_truth_boxes",
                )
            )
        pooled = aggregate(results, "ground_truth_boxes")
        rows.append({"control_radius_heights": radius, **pooled})
        log.info(
            "radius %.2f heights: team F1 %.4f, loose F1 %.4f, holder acc %.4f, "
            "prediction coverage %.4f",
            radius, pooled["team_f1_macro"], pooled["per_label"].get("loose", {}).get("f1", 0.0),
            pooled["holder_accuracy"], pooled["prediction_coverage"],
        )

    best = max(rows, key=lambda r: r["team_f1_macro"])
    destination = args.out / "possession_radius_sweep_train.json"
    destination.write_text(
        json.dumps(
            {
                "split": args.split,
                "n_sequences": len(chosen),
                "selected_on": "train split only; test never used for selection",
                "grid": grid,
                "best_control_radius_heights": best["control_radius_heights"],
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"best control_radius_heights = {best['control_radius_heights']} "
        f"(team F1 {best['team_f1_macro']:.4f} on {len(chosen)} train sequences)"
    )
    print(f"wrote {destination}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/eval/possession"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--control-radius-m", type=float, default=None)
    parser.add_argument(
        "--sweep", action="store_true",
        help="sweep the engine's control radius; only meaningful on --split train",
    )
    args = parser.parse_args()

    configure_logging("INFO")
    labels = sequence_labels()
    if not labels:
        log.error("no GSR sequences under %s; run scripts/download_eval_data.py gsr", GSR_ROOT)
        return 1

    split = build_split([], sorted(labels))
    chosen = [s for s in sorted(labels) if split.split_of("soccernet_gsr", s) == args.split]
    if args.max_sequences:
        chosen = chosen[: args.max_sequences]
    log.info(
        "%s split: %d of %d sequence(s); split fingerprint %s",
        args.split, len(chosen), len(labels), split.fingerprint(),
    )

    params = DerivationParams()
    if args.control_radius_m is not None:
        params.control_radius_m = args.control_radius_m

    args.out.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        return run_sweep(labels, chosen, params, args)

    results = []
    per_sequence = []

    for name in chosen:
        gt = derive_from_gsr(labels[name], params)
        gt.save(args.out / f"{name}.json")

        context = context_from_gsr(labels[name])
        engine = PossessionEngine(context, PossessionConfig())
        predicted = {
            frame.frame_idx: (
                frame.state.value,
                frame.team_id,
                None if frame.track_id is None else int(frame.track_id),
            )
            for frame in engine.per_frame()
        }
        result = evaluate_possession_vs_gt(
            gt, predicted, gt.fps, configuration="ground_truth_boxes"
        )
        results.append(result)
        per_sequence.append({**result.to_dict(), "reference_fingerprint": gt.fingerprint()})
        log.info(
            "%s: scorable %d, team F1 %.3f, holder acc %.3f (n=%d)",
            name, result.n_scorable, result.team_f1,
            result.holder_accuracy, result.holder_total,
        )

    summary = {
        "split": args.split,
        "split_fingerprint": split.fingerprint(),
        "derivation_params": params.to_dict(),
        "configuration": "ground_truth_boxes",
        "what_this_measures": (
            "Possession logic with perfect perception. Detection, tracking and "
            "team classification are annotated, so every error here belongs to "
            "the state machine. It is an upper bound on the pipeline, not the "
            "number a user would see."
        ),
        "pooled": aggregate(results, "ground_truth_boxes"),
        "per_sequence": per_sequence,
    }
    destination = args.out / f"possession_{args.split}_gtboxes.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pooled = summary["pooled"]
    print(json.dumps({k: pooled[k] for k in (
        "n_frames", "n_scorable", "reference_coverage", "prediction_coverage",
        "team_f1_macro", "holder_accuracy", "holder_accuracy_ci95", "holder_n",
    )}, indent=2))
    print("per label:", json.dumps(pooled["per_label"], indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
