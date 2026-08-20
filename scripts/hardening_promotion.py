"""Apply the eleven predeclared criteria to C-Hardened.

Part 7 of precision hardening. The criteria were fixed before the hardened model
was trained; this reads the locked artefacts and applies them mechanically.

Two of them say "materially", which is not a number. They are pinned here to
values chosen before the results were read, and both are stated in the output so
the reader can see the interpretation rather than infer it.

Usage::

    python scripts/hardening_promotion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("hardening.promotion")

CANDIDATES = Path("data/eval/broadcast/candidates.json")
PIPELINE = Path("data/eval/broadcast/pipeline.json")

DEFAULT = "A_default"
BASE_C = "C_adapt"
HARDENED = "C_hardened"

# -- the eleven criteria, declared before C-Hardened was trained -------------- #
MIN_LOCAL_RECALL = 0.82
MIN_PUBLIC_RECALL = 0.68
MIN_TINY_RECALL = 0.62
MIN_OCCLUDED_RECALL = 0.47
#: "materially" for criterion 5: at least a 10% relative cut.
MIN_FP_NEG_REDUCTION = 0.10
MAX_FP_ALL_VS_DEFAULT = 1.25
MIN_LOCAL_COVERAGE = 0.34
#: "materially" for criterion 10: no more than a 5% relative drop.
MAX_CARRY_REGRESSION = 0.05
MAX_RUNTIME_RATIO = 1.5


def wilson_half_width(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    d = 1 + z * z / total
    return float(
        z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("data/eval/hardening/promotion.json"))
    args = parser.parse_args()
    configure_logging("INFO")

    detection = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    pipeline = json.loads(PIPELINE.read_text(encoding="utf-8"))

    default_d, base_d, hard_d = (
        detection[DEFAULT], detection[BASE_C], detection[HARDENED]
    )
    default_p, base_p, hard_p = (
        pipeline[DEFAULT], pipeline[BASE_C], pipeline[HARDENED]
    )

    passes: list[str] = []
    failures: list[str] = []

    def check(ok: bool, text: str) -> None:
        (passes if ok else failures).append(text)

    # 1
    local = hard_d["local_test"]["centre_recall"]["25.0"]
    check(
        local["recall"] >= MIN_LOCAL_RECALL,
        f"1. local centre recall@25 {local['recall']:.4f} "
        f"(need >= {MIN_LOCAL_RECALL}); 95% CI "
        f"[{local['ci95'][0]:.2f}, {local['ci95'][1]:.2f}] on 23 frames",
    )
    # 2
    public = hard_d["public_test"]["centre_recall"]["25.0"]["recall"]
    check(
        public >= MIN_PUBLIC_RECALL,
        f"2. public centre recall@25 {public:.4f} (need >= {MIN_PUBLIC_RECALL})",
    )
    # 3
    tiny = hard_d["public_test"]["by_ball_size"]["1_tiny"]["recall_25px"]
    check(tiny >= MIN_TINY_RECALL,
          f"3. tiny-ball recall {tiny:.4f} (need >= {MIN_TINY_RECALL})")
    # 4
    occluded = hard_d["public_test"]["by_occlusion"]["occluded"]["recall_25px"]
    check(occluded >= MIN_OCCLUDED_RECALL,
          f"4. occluded recall {occluded:.4f} (need >= {MIN_OCCLUDED_RECALL})")
    # 5
    base_fp_neg = base_d["public_test"]["false_positives"]["fp_per_negative_frame"]
    hard_fp_neg = hard_d["public_test"]["false_positives"]["fp_per_negative_frame"]
    reduction = 1 - hard_fp_neg / base_fp_neg if base_fp_neg else 0.0
    check(
        reduction >= MIN_FP_NEG_REDUCTION,
        f"5. FP per negative frame {base_fp_neg:.4f} -> {hard_fp_neg:.4f} "
        f"({reduction:+.1%} reduction vs Candidate C; 'materially' pinned at "
        f">= {MIN_FP_NEG_REDUCTION:.0%})",
    )
    # 6
    default_fp_all = default_d["public_test"]["false_positives"]["per_frame_all"]
    hard_fp_all = hard_d["public_test"]["false_positives"]["per_frame_all"]
    ratio = hard_fp_all / default_fp_all if default_fp_all else float("inf")
    check(
        ratio <= MAX_FP_ALL_VS_DEFAULT,
        f"6. FP per all frames {default_fp_all:.4f} (default) -> {hard_fp_all:.4f} "
        f"({ratio:.2f}x, cap {MAX_FP_ALL_VS_DEFAULT}x)",
    )
    # 7
    coverage = hard_p["local"]["ball_coverage_direct"]
    check(coverage >= MIN_LOCAL_COVERAGE,
          f"7. local direct coverage {coverage:.4f} (need >= {MIN_LOCAL_COVERAGE})")
    # 8
    default_det = default_p["local"]["determinability"]
    hard_det = hard_p["local"]["determinability"]
    n_frames = 6000  # 0-120 s at 50 fps, identical for every candidate
    tolerance = wilson_half_width(int(default_det * n_frames), n_frames)
    check(
        hard_det >= default_det - tolerance,
        f"8. local determinability {default_det:.4f} (default) -> {hard_det:.4f} "
        f"(delta {hard_det - default_det:+.4f}, 95% tolerance +-{tolerance:.4f} "
        f"on {n_frames} frames)",
    )
    # 9
    base_pass = base_p["bas"]["pass_f1"]
    hard_pass = hard_p["bas"]["pass_f1"]
    check(
        hard_pass >= base_pass,
        f"9. pass F1 {base_pass:.4f} (Candidate C) -> {hard_pass:.4f} "
        f"({hard_pass - base_pass:+.4f}, need >= 0)",
    )
    # 10
    base_carry = base_p["bas"]["carry_f1"]
    hard_carry = hard_p["bas"]["carry_f1"]
    drop = 1 - hard_carry / base_carry if base_carry else 0.0
    check(
        drop <= MAX_CARRY_REGRESSION,
        f"10. carry F1 {base_carry:.4f} -> {hard_carry:.4f} "
        f"({-drop:+.1%}; 'materially' pinned at a {MAX_CARRY_REGRESSION:.0%} drop)",
    )
    # 11
    runtime_ratio = (
        hard_d["runtime_ms_per_frame"] / default_d["runtime_ms_per_frame"]
    )
    check(
        runtime_ratio <= MAX_RUNTIME_RATIO,
        f"11. runtime {default_d['runtime_ms_per_frame']:.1f} -> "
        f"{hard_d['runtime_ms_per_frame']:.1f} ms/frame ({runtime_ratio:.2f}x, "
        f"cap {MAX_RUNTIME_RATIO}x)",
    )

    promote = not failures
    payload = {
        "candidate": HARDENED,
        "checkpoint_fingerprint": hard_d["checkpoint_fingerprint"],
        "criteria_declared_before_training": {
            "1_min_local_recall": MIN_LOCAL_RECALL,
            "2_min_public_recall": MIN_PUBLIC_RECALL,
            "3_min_tiny_recall": MIN_TINY_RECALL,
            "4_min_occluded_recall": MIN_OCCLUDED_RECALL,
            "5_min_fp_neg_reduction": MIN_FP_NEG_REDUCTION,
            "6_max_fp_all_vs_default": MAX_FP_ALL_VS_DEFAULT,
            "7_min_local_coverage": MIN_LOCAL_COVERAGE,
            "8_determinability": "at least default minus the 95% Wilson half-width",
            "9_pass_f1": "at least Candidate C",
            "10_max_carry_regression": MAX_CARRY_REGRESSION,
            "11_max_runtime_ratio": MAX_RUNTIME_RATIO,
        },
        "passes": passes,
        "failures": failures,
        "decision": "PROMOTE C-HARDENED" if promote else "KEEP CURRENT DEFAULT",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}\nC-Hardened: "
          f"{'PROMOTE' if promote else 'REJECT'}\n{'=' * 72}")
    for line in passes:
        print(f"  PASS  {line}")
    for line in failures:
        print(f"  FAIL  {line}")
    print(f"\n\nDECISION: {payload['decision']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
