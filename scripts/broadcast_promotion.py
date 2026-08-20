"""Apply the eight predeclared promotion criteria to the broadcast candidates.

The criteria were fixed before any candidate was trained. This reads the locked
evaluation artefacts and applies them mechanically; nothing here interprets,
weights or waives a criterion.

One criterion could not be evaluated as written, and that is reported rather
than quietly substituted: the local test contains zero negative frames, so
"local test precision >= 0.55" has no denominator. Cross-domain false-positive
evidence from the locked public negatives is reported in its place and labelled
as such.

Usage::

    python scripts/broadcast_promotion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("broadcast.promotion")

CANDIDATES = Path("data/eval/broadcast/candidates.json")
PIPELINE = Path("data/eval/broadcast/pipeline.json")
BASELINE = "A_default"

# -- the eight criteria, as declared before training ------------------------- #
MIN_LOCAL_RECALL_GAIN = 0.05
MIN_LOCAL_PRECISION = 0.55
MAX_PUBLIC_DOMAIN_REGRESSION = 0.05
MAX_FP_GROWTH = 0.25
MIN_COVERAGE_DELTA = 0.0
MIN_DETERMINABILITY_DELTA = 0.0
MIN_EVENT_DELTA = 0.0
MAX_RUNTIME_RATIO = 1.5


@dataclass
class Verdict:
    candidate: str
    passes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    unevaluable: list[str] = field(default_factory=list)

    @property
    def promote(self) -> bool:
        return not self.failures and not self.unevaluable

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "promote": self.promote,
            "n_pass": len(self.passes),
            "n_fail": len(self.failures),
            "n_unevaluable": len(self.unevaluable),
            "passes": self.passes,
            "failures": self.failures,
            "unevaluable": self.unevaluable,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("data/eval/broadcast/promotion.json"))
    args = parser.parse_args()
    configure_logging("INFO")

    detection = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    pipeline = json.loads(PIPELINE.read_text(encoding="utf-8"))
    base_d, base_p = detection[BASELINE], pipeline[BASELINE]

    base_local = base_d["local_test"]["centre_recall"]["25.0"]["recall"]
    base_fp_neg = base_d["public_test"]["false_positives"]["fp_per_negative_frame"]
    base_fp_all = base_d["public_test"]["false_positives"]["per_frame_all"]
    base_runtime = base_d["runtime_ms_per_frame"]

    verdicts: dict[str, dict] = {}
    for label in detection:
        if label == BASELINE:
            continue
        d, p = detection[label], pipeline.get(label, {})
        verdict = Verdict(candidate=label)

        # ``verdict`` bound as a default: the closure is consumed inside this
        # iteration, but a late-bound one would silently write every candidate's
        # results into the last verdict.
        def check(ok: bool, text: str, _verdict: Verdict = verdict) -> None:
            (_verdict.passes if ok else _verdict.failures).append(text)

        # 1. local held-out recall
        local = d["local_test"]["centre_recall"]["25.0"]
        gain = local["recall"] - base_local
        check(
            gain >= MIN_LOCAL_RECALL_GAIN,
            f"1. local test centre recall@25 {base_local:.4f} -> {local['recall']:.4f} "
            f"({gain:+.4f}, need >= +{MIN_LOCAL_RECALL_GAIN}); "
            f"95% CI [{local['ci95'][0]:.2f}, {local['ci95'][1]:.2f}] on 23 frames",
        )

        # 2. local precision -- no denominator exists
        verdict.unevaluable.append(
            f"2. local test precision >= {MIN_LOCAL_PRECISION}: NOT EVALUABLE. The "
            f"locked local test has 0 negative frames, so no local false-positive "
            f"rate exists. Cross-domain evidence from the locked public negatives "
            f"is reported under criterion 4 and must not be read as local precision."
        )

        # 3. public per-domain regression
        for domain, block in d["public_test"]["by_domain"].items():
            baseline_recall = base_d["public_test"]["by_domain"][domain]["recall_25px"]
            delta = block["recall_25px"] - baseline_recall
            check(
                delta >= -MAX_PUBLIC_DOMAIN_REGRESSION,
                f"3. public {domain} centre recall@25 {baseline_recall:.4f} -> "
                f"{block['recall_25px']:.4f} ({delta:+.4f}, tolerance "
                f"-{MAX_PUBLIC_DOMAIN_REGRESSION})",
            )

        # 4. false positives -- both readings, both must hold
        fp = d["public_test"]["false_positives"]
        for name, value, baseline in (
            ("on public negative frames", fp["fp_per_negative_frame"], base_fp_neg),
            ("over all public frames", fp["per_frame_all"], base_fp_all),
        ):
            growth = (value / baseline - 1.0) if baseline else float("inf")
            check(
                growth <= MAX_FP_GROWTH,
                f"4. false positives {name} {baseline:.4f} -> {value:.4f} "
                f"({growth:+.1%}, cap +{MAX_FP_GROWTH:.0%})",
            )

        # 5-7. pipeline, identical segments
        if not p:
            verdict.unevaluable.append("5-7. pipeline results missing")
        else:
            for number, key, source, minimum, name in (
                (5, "ball_coverage_direct", "local", MIN_COVERAGE_DELTA,
                 "local video direct ball coverage"),
                (6, "determinability", "local", MIN_DETERMINABILITY_DELTA,
                 "local possession determinability"),
                (7, "pass_f1", "bas", MIN_EVENT_DELTA, "unchanged-engine pass F1"),
                (7, "carry_f1", "bas", MIN_EVENT_DELTA, "unchanged-engine carry F1"),
            ):
                value = p.get(source, {}).get(key)
                baseline = base_p.get(source, {}).get(key)
                if value is None or baseline is None:
                    verdict.unevaluable.append(f"{number}. {name} not measured")
                    continue
                delta = value - baseline
                check(
                    delta >= minimum,
                    f"{number}. {name} {baseline:.4f} -> {value:.4f} "
                    f"({delta:+.4f}, need >= {minimum})",
                )

        # 8. runtime
        ratio = d["runtime_ms_per_frame"] / base_runtime
        check(
            ratio <= MAX_RUNTIME_RATIO,
            f"8. runtime {base_runtime:.1f} -> {d['runtime_ms_per_frame']:.1f} "
            f"ms/frame ({ratio:.2f}x, cap {MAX_RUNTIME_RATIO}x)",
        )

        verdicts[label] = verdict.to_dict()
        log.info(
            "%s: %s (%d pass, %d fail, %d unevaluable)",
            label, "PROMOTE" if verdict.promote else "REJECT",
            len(verdict.passes), len(verdict.failures), len(verdict.unevaluable),
        )

    promoted = [k for k, v in verdicts.items() if v["promote"]]
    decision = f"PROMOTE {promoted[0].split('_')[0]}" if promoted else "KEEP CURRENT DEFAULT"

    payload = {
        "baseline": BASELINE,
        "criteria_declared_before_training": {
            "1_min_local_recall_gain": MIN_LOCAL_RECALL_GAIN,
            "2_min_local_precision": MIN_LOCAL_PRECISION,
            "3_max_public_domain_regression": MAX_PUBLIC_DOMAIN_REGRESSION,
            "4_max_fp_growth": MAX_FP_GROWTH,
            "5_min_coverage_delta": MIN_COVERAGE_DELTA,
            "6_min_determinability_delta": MIN_DETERMINABILITY_DELTA,
            "7_min_event_delta": MIN_EVENT_DELTA,
            "8_max_runtime_ratio": MAX_RUNTIME_RATIO,
        },
        "verdicts": verdicts,
        "decision": decision,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for label, verdict in verdicts.items():
        print(f"\n{'=' * 72}\n{label}: "
              f"{'PROMOTE' if verdict['promote'] else 'REJECT'}\n{'=' * 72}")
        for line in verdict["passes"]:
            print(f"  PASS  {line}")
        for line in verdict["failures"]:
            print(f"  FAIL  {line}")
        for line in verdict["unevaluable"]:
            print(f"  N/A   {line}")
    print(f"\n\nDECISION: {decision}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
