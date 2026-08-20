"""Apply the Phase 2D promotion rule to the measured candidates.

Phase 2D, Part 10.

Reads the artefacts produced by the other Phase 2D scripts and runs
``evaluate_promotion`` over them. Nothing here decides anything -- the rule is
in ``visionpitch.evaluation.promotion`` and is unit-tested; this only assembles
the measurements and records the verdict with its provenance.

Usage::

    python scripts/ball_promotion_report.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.promotion import (  # noqa: E402
    CandidateMeasurements,
    PromotionCriteria,
    evaluate_promotion,
)

log = get_logger("ball.promotion")


def file_fingerprint(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matcher", default="centre25", choices=["centre25", "iou50"])
    parser.add_argument("--out", type=Path, default=Path("data/eval/ball_domains"))
    args = parser.parse_args()

    configure_logging("INFO")

    thresholds = {
        "baseline": Path("data/eval/ball_threshold/threshold_yolo-football-ball-detection.json"),
        "multicorpus": Path("data/eval/ball_threshold/threshold_best.json"),
    }
    determinability = {
        "baseline": Path("data/eval/determinability/determinability_bas_baseline.json"),
        "multicorpus": Path("data/eval/determinability/determinability_bas_multicorpus.json"),
    }
    weights = {
        "baseline": Path("models/yolo-football-ball-detection.pt"),
        "multicorpus": Path("models/finetune/ball_multicorpus/weights/best.pt"),
    }
    # Downstream pass recall on the SN-BAS segment, unchanged event engine.
    pass_recall = {"baseline": 0.227, "multicorpus": 0.182}

    measurements: dict[str, CandidateMeasurements] = {}
    for label in ("baseline", "multicorpus"):
        threshold_payload = load(thresholds[label])
        determinability_payload = load(determinability[label])
        if threshold_payload is None or determinability_payload is None:
            log.error("missing artefacts for %s; run the Phase 2D scripts first", label)
            return 1

        per_domain = threshold_payload["test_at_previous_default_0.08"]["per_domain"]
        measurements[label] = CandidateMeasurements(
            label=label,
            per_domain_recall={
                d: v[args.matcher]["recall"] for d, v in per_domain.items()
            },
            per_domain_precision={
                d: v[args.matcher]["precision"] for d, v in per_domain.items()
            },
            effective_ball_coverage=determinability_payload["ball_coverage_direct"],
            possession_determinability=determinability_payload["determinability"],
            pass_recall=pass_recall[label],
            n_direct_observations=None,
            n_inferred_observations=None,
            n_long_gap_fills=0,
            model_fingerprint=file_fingerprint(weights[label]),
            config_fingerprint="ball_detection.conf_threshold=0.08;imgsz=960",
        )

    verdict = evaluate_promotion(
        measurements["multicorpus"], measurements["baseline"], PromotionCriteria()
    )

    payload = {
        "matcher": args.matcher,
        "benchmark": "data/ball_multicorpus test split (clip-disjoint)",
        "downstream_benchmark": "SN-BAS mid_pre_720p 600-780 s, unchanged event engine",
        "candidate": measurements["multicorpus"].to_dict(),
        "incumbent": measurements["baseline"].to_dict(),
        "verdict": verdict.to_dict(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"promotion_phase2d_{args.matcher}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"matcher: {args.matcher}")
    print(f"\ncandidate  : {measurements['multicorpus'].to_dict()}")
    print(f"\nincumbent  : {measurements['baseline'].to_dict()}")
    print(f"\nPROMOTE: {verdict.promote}")
    print("\npassed:")
    for line in verdict.passes:
        print(f"  + {line}")
    print("\nfailed:")
    for line in verdict.failures:
        print(f"  - {line}")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
