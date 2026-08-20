"""Promotion thresholds for ball temporal fusion, pinned before evaluation.

Every number here was written down and committed **before** any four-way row was
scored. That ordering is the whole point: this project has four times now
promoted an intervention on a benchmark win that then regressed downstream, and
the only defence is to declare the bar before seeing whether it was cleared.

Each threshold states the measurement that set it. None of them says "material"
or "acceptable".

One honesty note, stated here rather than buried
------------------------------------------------
``MAX_PUBLIC_RECALL_DROP`` is **not a blind gate**. The isolated ablation already
measured the public-domain recall cost (Candidate C 0.7043 -> 0.6629, default
0.2336 -> 0.2089), so a threshold set above those values is one I already know
will pass. It is retained as a regression guard, not as evidence. The gates that
are genuinely unmeasured at the time of writing -- and therefore the ones that
decide this -- are coverage (G3), determinability (G4), pass F1 (G5) and carry
F1 (G6).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

THRESHOLD_SPEC_VERSION = "1.0.0"

#: Where the versioned artifact is written. Written once, before evaluation.
THRESHOLD_ARTIFACT = Path("data/eval/fusion/promotion_thresholds.json")


# --------------------------------------------------------------------------- #
# The numbers
# --------------------------------------------------------------------------- #

#: G1. Local centre recall @25 px, locked local broadcast test.
#:
#: The locked local test carries **23 positive frames**. One frame is 0.0435 of
#: recall and the Wilson 95% interval at n=23 spans roughly +-0.19, so a drop of
#: one or two frames is indistinguishable from noise. The gate is therefore set
#: at two frames and is deliberately weak: passing it is not evidence of no
#: harm, it only rules out a collapse.
MAX_LOCAL_RECALL_DROP = 0.087

#: G2. False positives per frame on the locked public held-out set, relative.
#:
#: Candidate C hardening achieved a 37.5% reduction in false positives per
#: negative frame and still regressed determinability from 0.0841 to 0.0471.
#: A precision gain smaller than that has already been shown insufficient to
#: pay for a coverage loss, so 30% is the floor below which the trade cannot
#: possibly be worth making.
MIN_FP_REDUCTION_RELATIVE = 0.30

#: G3. Direct ball coverage in the production run, absolute, on **both** videos.
#:
#: Every coverage loss this project has measured downstream was harmful, and the
#: smallest of them was 0.034 (multi-corpus box, 0.434 -> 0.400). The others were
#: 0.051 (centre heatmap), 0.062 (SN-GSR fine-tune) and 0.066 (C-hardened,
#: local). 0.020 sits below the smallest loss ever observed to regress
#: downstream here.
MAX_COVERAGE_DROP = 0.020

#: G4. Possession determinability, expressed as a ratio because the absolute
#: values are small (0.047 to 0.121) and absolute deltas at that scale mislead.
MIN_DETERMINABILITY_RATIO = 0.95

#: G5/G6. Event F1 on the locked SN-BAS window, absolute change.
#:
#: The window holds 22 scorable passes, so one event is 0.045 of recall. The
#: gate is set inside a single event: fusion may not cost a whole event.
MIN_PASS_F1_CHANGE = -0.020
MIN_CARRY_F1_CHANGE = -0.020

#: G7a. A position emitted further than ``max_interpolation_gap_frames`` (12)
#: from the nearest direct observation. The estimator's own cap makes this
#: impossible, so any non-zero value is a defect, not a tuning question.
MAX_LONG_GAP_HALLUCINATION_RATE = 0.000

#: G7b. The mechanism by which this change *could* invent trajectory: emptying
#: frames lengthens gaps, which lets the estimator bridge further. Measured as
#: the share of all frames sitting inside an interpolated run of >= 6 frames
#: (0.25 s at the 24 fps effective rate), against the legacy row.
MAX_LONG_BRIDGE_RATE_INCREASE = 0.020
LONG_BRIDGE_MIN_RUN_FRAMES = 6

#: G8. Fusion is an offline pass over cached candidates; it decodes nothing and
#: runs no model. A measurable share of end-to-end runtime would mean it is
#: doing something other than what it claims.
MAX_RUNTIME_MULTIPLIER = 1.05
MAX_FUSION_MS_PER_FRAME = 2.0

#: G9. Public centre recall @25 px on the locked SN-GSR test split. See the
#: module docstring: this one is not blind.
MAX_PUBLIC_RECALL_DROP = 0.050

# -- benefit: gates alone only establish "did no harm" ----------------------- #

#: B1. Determinability must actually improve somewhere, by more than the 5%
#: tolerance the gate allows in the other direction.
MIN_BENEFIT_DETERMINABILITY_RATIO = 1.05
#: B2/B3. Or an event F1 must gain at least one event's worth.
MIN_BENEFIT_PASS_F1_CHANGE = 0.020
MIN_BENEFIT_CARRY_F1_CHANGE = 0.020

# -- distinctness, for the KEEP MULTIPLE MODES verdict ----------------------- #

#: Two modes are distinct enough to both be kept only if they differ this much
#: in direct coverage on at least one video **and** trade in opposite directions
#: on at least one downstream metric. Without both, keeping two modes is
#: shipping a choice the user cannot make correctly.
MIN_MODE_COVERAGE_SEPARATION = 0.050


# --------------------------------------------------------------------------- #
# Mechanical decision rule
# --------------------------------------------------------------------------- #


@dataclass
class GateResult:
    name: str
    passed: bool
    observed: float | None
    threshold: float
    comparison: str
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RowVerdict:
    """One temporal row judged against its own legacy reference."""

    row: str
    reference: str
    gates: list[GateResult] = field(default_factory=list)
    benefits: list[GateResult] = field(default_factory=list)

    @property
    def gates_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def has_benefit(self) -> bool:
        return any(b.passed for b in self.benefits)

    @property
    def promotable(self) -> bool:
        return self.gates_passed and self.has_benefit

    @property
    def failures(self) -> list[str]:
        return [g.name for g in self.gates if not g.passed]

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "reference": self.reference,
            "gates": [g.to_dict() for g in self.gates],
            "benefits": [b.to_dict() for b in self.benefits],
            "gates_passed": self.gates_passed,
            "has_benefit": self.has_benefit,
            "promotable": self.promotable,
            "failed_gates": self.failures,
        }


def _get(block: dict, *path: str) -> float | None:
    node: Any = block
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return None if node is None else float(node)


def _gate(
    name: str, observed: float | None, threshold: float, comparison: str, detail: str = ""
) -> GateResult:
    """A missing measurement fails its gate. It is never treated as a pass."""
    if observed is None:
        return GateResult(name, False, None, threshold, comparison,
                          detail or "not measured")
    passed = observed >= threshold if comparison == ">=" else observed <= threshold
    return GateResult(name, bool(passed), round(observed, 6), threshold,
                      comparison, detail)


def evaluate_row(row: str, new: dict, reference_name: str, ref: dict) -> RowVerdict:
    """Apply every pinned gate to one temporal row against its legacy reference.

    ``new`` and ``ref`` are the per-row metric blocks assembled by
    ``scripts/fusion_production_eval.py``. Nothing is interpreted here that the
    thresholds above do not name.
    """
    verdict = RowVerdict(row=row, reference=reference_name)

    # -- G1 local candidate recall ------------------------------------------ #
    local_new = _get(new, "perception", "local_centre_recall_25px")
    local_ref = _get(ref, "perception", "local_centre_recall_25px")
    verdict.gates.append(_gate(
        "G1_local_recall_drop",
        None if local_new is None or local_ref is None else local_ref - local_new,
        MAX_LOCAL_RECALL_DROP, "<=",
        "locked local test, 23 positive frames",
    ))

    # -- G2 false-positive reduction ---------------------------------------- #
    fp_new = _get(new, "perception", "public_fp_per_frame")
    fp_ref = _get(ref, "perception", "public_fp_per_frame")
    reduction = (
        None if fp_new is None or fp_ref is None or fp_ref <= 0
        else (fp_ref - fp_new) / fp_ref
    )
    verdict.gates.append(_gate(
        "G2_fp_reduction_relative", reduction, MIN_FP_REDUCTION_RELATIVE, ">=",
        "locked public held-out (SN-GSR test)",
    ))

    # -- G3 coverage, both videos ------------------------------------------- #
    for video in ("local", "bas"):
        cov_new = _get(new, video, "ball_coverage_direct")
        cov_ref = _get(ref, video, "ball_coverage_direct")
        verdict.gates.append(_gate(
            f"G3_coverage_drop_{video}",
            None if cov_new is None or cov_ref is None else cov_ref - cov_new,
            MAX_COVERAGE_DROP, "<=",
        ))

    # -- G4 determinability, both videos ------------------------------------ #
    for video in ("local", "bas"):
        det_new = _get(new, video, "determinability")
        det_ref = _get(ref, video, "determinability")
        verdict.gates.append(_gate(
            f"G4_determinability_ratio_{video}",
            None if det_new is None or not det_ref else det_new / det_ref,
            MIN_DETERMINABILITY_RATIO, ">=",
        ))

    # -- G5/G6 events -------------------------------------------------------- #
    for key, floor, name in (
        ("pass_f1", MIN_PASS_F1_CHANGE, "G5_pass_f1_change"),
        ("carry_f1", MIN_CARRY_F1_CHANGE, "G6_carry_f1_change"),
    ):
        new_v = _get(new, "bas", key)
        ref_v = _get(ref, "bas", key)
        verdict.gates.append(_gate(
            name, None if new_v is None or ref_v is None else new_v - ref_v,
            floor, ">=", "locked SN-BAS window, 22 scorable passes",
        ))

    # -- G7 hallucination ---------------------------------------------------- #
    verdict.gates.append(_gate(
        "G7a_long_gap_hallucination_rate",
        _get(new, "trajectory", "long_gap_hallucination_rate"),
        MAX_LONG_GAP_HALLUCINATION_RATE, "<=",
        "position further than 12 frames from any direct observation",
    ))
    bridge_new = _get(new, "trajectory", "long_bridge_rate")
    bridge_ref = _get(ref, "trajectory", "long_bridge_rate")
    verdict.gates.append(_gate(
        "G7b_long_bridge_rate_increase",
        None if bridge_new is None or bridge_ref is None else bridge_new - bridge_ref,
        MAX_LONG_BRIDGE_RATE_INCREASE, "<=",
        f"frames inside an interpolated run of >= {LONG_BRIDGE_MIN_RUN_FRAMES}",
    ))

    # -- G8 runtime ---------------------------------------------------------- #
    rt_new = _get(new, "system", "total_runtime_s")
    rt_ref = _get(ref, "system", "total_runtime_s")
    verdict.gates.append(_gate(
        "G8a_runtime_multiplier",
        None if rt_new is None or not rt_ref else rt_new / rt_ref,
        MAX_RUNTIME_MULTIPLIER, "<=",
    ))
    verdict.gates.append(_gate(
        "G8b_fusion_ms_per_frame",
        _get(new, "system", "fusion_ms_per_frame"),
        MAX_FUSION_MS_PER_FRAME, "<=",
    ))

    # -- G9 public regression ------------------------------------------------ #
    pub_new = _get(new, "perception", "public_centre_recall_25px")
    pub_ref = _get(ref, "perception", "public_centre_recall_25px")
    verdict.gates.append(_gate(
        "G9_public_recall_drop",
        None if pub_new is None or pub_ref is None else pub_ref - pub_new,
        MAX_PUBLIC_RECALL_DROP, "<=",
        "NOT a blind gate -- see module docstring",
    ))

    # -- benefit ------------------------------------------------------------- #
    for video in ("local", "bas"):
        det_new = _get(new, video, "determinability")
        det_ref = _get(ref, video, "determinability")
        verdict.benefits.append(_gate(
            f"B1_determinability_gain_{video}",
            None if det_new is None or not det_ref else det_new / det_ref,
            MIN_BENEFIT_DETERMINABILITY_RATIO, ">=",
        ))
    for key, floor, name in (
        ("pass_f1", MIN_BENEFIT_PASS_F1_CHANGE, "B2_pass_f1_gain"),
        ("carry_f1", MIN_BENEFIT_CARRY_F1_CHANGE, "B3_carry_f1_gain"),
    ):
        new_v = _get(new, "bas", key)
        ref_v = _get(ref, "bas", key)
        verdict.benefits.append(_gate(
            name, None if new_v is None or ref_v is None else new_v - ref_v,
            floor, ">=",
        ))
    return verdict


def modes_are_distinct(b: dict, d: dict) -> tuple[bool, str]:
    """Whether B and D are different enough to justify shipping both."""
    separated = False
    for video in ("local", "bas"):
        cov_b = _get(b, video, "ball_coverage_direct")
        cov_d = _get(d, video, "ball_coverage_direct")
        if cov_b is not None and cov_d is not None:
            if abs(cov_b - cov_d) >= MIN_MODE_COVERAGE_SEPARATION:
                separated = True
    opposed = False
    for path in (("local", "determinability"), ("bas", "determinability"),
                 ("bas", "pass_f1"), ("bas", "carry_f1")):
        vb, vd = _get(b, *path), _get(d, *path)
        if vb is None or vd is None:
            continue
        if (vb > vd and not opposed) or (vd > vb and not opposed):
            opposed = opposed or abs(vb - vd) > 0
    if separated and opposed:
        return True, "coverage separated and downstream metrics trade in opposite directions"
    if not separated:
        return False, (
            f"direct coverage differs by less than {MIN_MODE_COVERAGE_SEPARATION} "
            "on every video"
        )
    return False, "no downstream metric trades in the opposite direction"


def decide(verdict_b: RowVerdict, verdict_d: RowVerdict, b: dict, d: dict) -> dict:
    """The four allowed verdicts, chosen mechanically. No judgement applied."""
    if verdict_b.promotable and verdict_d.promotable:
        distinct, why = modes_are_distinct(b, d)
        if distinct:
            return {"verdict": "KEEP MULTIPLE MODES", "reason": why}
        det_b = _get(b, "local", "determinability") or 0.0
        det_d = _get(d, "local", "determinability") or 0.0
        winner = "D" if det_d > det_b else "B"
        return {
            "verdict": (
                "PROMOTE C HIGH-RECALL FUSION" if winner == "D"
                else "PROMOTE NEW BALANCED FUSION"
            ),
            "reason": (
                f"both rows promotable but not distinct ({why}); "
                f"row {winner} has the higher local determinability"
            ),
        }
    if verdict_b.promotable:
        return {"verdict": "PROMOTE NEW BALANCED FUSION",
                "reason": "row B cleared every gate and showed a measured benefit"}
    if verdict_d.promotable:
        return {"verdict": "PROMOTE C HIGH-RECALL FUSION",
                "reason": "row D cleared every gate and showed a measured benefit"}
    return {
        "verdict": "KEEP CURRENT FUSION",
        "reason": (
            "B failed: " + (", ".join(verdict_b.failures) or "no measured benefit")
            + "; D failed: "
            + (", ".join(verdict_d.failures) or "no measured benefit")
        ),
    }


# --------------------------------------------------------------------------- #
# The artifact
# --------------------------------------------------------------------------- #


def specification() -> dict:
    return {
        "spec_version": THRESHOLD_SPEC_VERSION,
        "declared_before_evaluation": True,
        "gates": {
            "G1_local_recall_drop": {
                "max": MAX_LOCAL_RECALL_DROP, "unit": "absolute recall",
                "measured_on": "locked local broadcast test, 23 positive frames",
                "basis": "two frames; Wilson 95% at n=23 spans about +-0.19",
            },
            "G2_fp_reduction_relative": {
                "min": MIN_FP_REDUCTION_RELATIVE, "unit": "relative reduction",
                "measured_on": "locked public held-out (SN-GSR test)",
                "basis": "C-hardening cut FP/negative 37.5% and still regressed "
                         "determinability 0.0841 -> 0.0471",
            },
            "G3_coverage_drop": {
                "max": MAX_COVERAGE_DROP, "unit": "absolute direct coverage",
                "measured_on": "both locked pipeline videos",
                "basis": "smallest coverage loss ever observed to regress "
                         "downstream in this project was 0.034",
            },
            "G4_determinability_ratio": {
                "min": MIN_DETERMINABILITY_RATIO, "unit": "ratio to legacy",
                "measured_on": "both locked pipeline videos",
                "basis": "absolute values span 0.047-0.121; ratios are stable "
                         "where absolute deltas are not",
            },
            "G5_pass_f1_change": {
                "min": MIN_PASS_F1_CHANGE, "unit": "absolute F1",
                "measured_on": "locked SN-BAS window",
                "basis": "22 scorable passes; one event is 0.045 of recall",
            },
            "G6_carry_f1_change": {
                "min": MIN_CARRY_F1_CHANGE, "unit": "absolute F1",
                "measured_on": "locked SN-BAS window",
            },
            "G7a_long_gap_hallucination_rate": {
                "max": MAX_LONG_GAP_HALLUCINATION_RATE, "unit": "share of frames",
                "basis": "the estimator caps interpolation at 12 frames, so any "
                         "non-zero value is a defect rather than a trade",
            },
            "G7b_long_bridge_rate_increase": {
                "max": MAX_LONG_BRIDGE_RATE_INCREASE, "unit": "share of frames",
                "min_run_frames": LONG_BRIDGE_MIN_RUN_FRAMES,
                "basis": "emptying frames lengthens gaps, which is the specific "
                         "mechanism by which this change could invent trajectory",
            },
            "G8a_runtime_multiplier": {"max": MAX_RUNTIME_MULTIPLIER},
            "G8b_fusion_ms_per_frame": {"max": MAX_FUSION_MS_PER_FRAME},
            "G9_public_recall_drop": {
                "max": MAX_PUBLIC_RECALL_DROP, "unit": "absolute recall",
                "measured_on": "locked SN-GSR test split",
                "blind": False,
                "basis": "the isolated ablation already measured this cost "
                         "(C 0.7043 -> 0.6629); retained as a regression guard, "
                         "not offered as evidence",
            },
        },
        "benefits_any_one_required": {
            "B1_determinability_gain": {"min_ratio": MIN_BENEFIT_DETERMINABILITY_RATIO},
            "B2_pass_f1_gain": {"min": MIN_BENEFIT_PASS_F1_CHANGE},
            "B3_carry_f1_gain": {"min": MIN_BENEFIT_CARRY_F1_CHANGE},
        },
        "mode_distinctness": {
            "min_coverage_separation": MIN_MODE_COVERAGE_SEPARATION,
            "and_requires": "opposite-direction trade on a downstream metric",
        },
        "decision_rule": [
            "a row is promotable only if it passes every gate AND shows >= 1 benefit",
            "B promotable only -> PROMOTE NEW BALANCED FUSION",
            "D promotable only -> PROMOTE C HIGH-RECALL FUSION",
            "both promotable and distinct -> KEEP MULTIPLE MODES",
            "both promotable and not distinct -> promote the higher local "
            "determinability",
            "neither promotable -> KEEP CURRENT FUSION",
        ],
        "missing_measurement_policy": "a gate with no measurement fails",
    }


def spec_fingerprint() -> str:
    return hashlib.sha256(
        json.dumps(specification(), sort_keys=True).encode()
    ).hexdigest()[:16]


def write_artifact(path: Path | None = None) -> Path:
    """Write the versioned artifact. Refuses to overwrite a differing spec."""
    destination = path or THRESHOLD_ARTIFACT
    payload = {**specification(), "fingerprint": spec_fingerprint()}
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != payload["fingerprint"]:
            raise SystemExit(
                f"{destination} already pins a different specification "
                f"({existing.get('fingerprint')} != {payload['fingerprint']}). "
                "Thresholds are pinned before evaluation and are not revised "
                "after results are seen."
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination
