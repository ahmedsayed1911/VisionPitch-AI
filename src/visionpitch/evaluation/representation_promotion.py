"""The tiny-ball study's promotion rule, as executable code.

Part 9.

The criteria were declared before any representation was trained. Putting them
in code rather than in prose means they cannot be softened while writing up, and
a unit test fails if anyone edits them. Phase 2D established the habit for a
reason: a candidate there failed by 0.0011 and was rejected, and the only way
that stays credible is if the rule is mechanical.

Every criterion is conjunctive. There is no weighting and no partial credit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from visionpitch.common.logging import get_logger

log = get_logger("evaluation.representation_promotion")

REPRESENTATION_PROMOTION_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class RepresentationCriteria:
    """Declared before results existed. Frozen so they cannot drift."""

    #: worst-domain direct coverage must improve by this much, absolute
    min_worst_domain_coverage_gain: float = 0.10
    #: cross-domain centre recall must improve by at least this much
    min_centre_recall_gain: float = 0.02
    #: precision floor the candidate must clear outright
    min_precision: float = 0.55
    #: largest tolerated relative growth in false positives per frame
    max_false_positive_growth: float = 0.25
    #: downstream metrics must not regress at all
    min_determinability_gain: float = 0.0
    min_pass_recall_gain: float = 0.0
    #: largest tolerated drop in any single domain's centre recall
    max_domain_regression: float = 0.02

    def to_dict(self) -> dict:
        return {
            "schema_version": REPRESENTATION_PROMOTION_SCHEMA_VERSION,
            "min_worst_domain_coverage_gain": self.min_worst_domain_coverage_gain,
            "min_centre_recall_gain": self.min_centre_recall_gain,
            "min_precision": self.min_precision,
            "max_false_positive_growth": self.max_false_positive_growth,
            "min_determinability_gain": self.min_determinability_gain,
            "min_pass_recall_gain": self.min_pass_recall_gain,
            "max_domain_regression": self.max_domain_regression,
        }


@dataclass
class RepresentationMeasurements:
    """Everything the rule reads about one representation."""

    label: str
    macro_centre_recall_25px: float
    worst_domain_centre_recall_25px: float
    macro_precision_25px: float
    macro_direct_coverage: float
    worst_domain_direct_coverage: float
    false_positives_per_frame: float
    per_domain_centre_recall_25px: dict[str, float] = field(default_factory=dict)
    #: downstream, from the unchanged pipeline and event engine
    determinability: float | None = None
    pass_recall: float | None = None
    ball_coverage_direct: float | None = None
    runtime_ms_per_frame: float | None = None
    model_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "macro_centre_recall_25px": self.macro_centre_recall_25px,
            "worst_domain_centre_recall_25px": self.worst_domain_centre_recall_25px,
            "macro_precision_25px": self.macro_precision_25px,
            "macro_direct_coverage": self.macro_direct_coverage,
            "worst_domain_direct_coverage": self.worst_domain_direct_coverage,
            "false_positives_per_frame": self.false_positives_per_frame,
            "per_domain_centre_recall_25px": self.per_domain_centre_recall_25px,
            "determinability": self.determinability,
            "pass_recall": self.pass_recall,
            "ball_coverage_direct": self.ball_coverage_direct,
            "runtime_ms_per_frame": self.runtime_ms_per_frame,
            "model_fingerprint": self.model_fingerprint,
        }


@dataclass
class RepresentationVerdict:
    promote: bool
    candidate: str
    incumbent: str
    passes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    criteria: RepresentationCriteria = field(default_factory=RepresentationCriteria)

    def to_dict(self) -> dict:
        return {
            "schema_version": REPRESENTATION_PROMOTION_SCHEMA_VERSION,
            "promote_to_default": self.promote,
            "candidate": self.candidate,
            "incumbent": self.incumbent,
            "criteria": self.criteria.to_dict(),
            "passes": self.passes,
            "failures": self.failures,
        }


def evaluate_representation(
    candidate: RepresentationMeasurements,
    incumbent: RepresentationMeasurements,
    criteria: RepresentationCriteria | None = None,
) -> RepresentationVerdict:
    """Apply the declared criteria. All must pass."""
    criteria = criteria or RepresentationCriteria()
    passes: list[str] = []
    failures: list[str] = []

    def check(ok: bool, description: str) -> None:
        (passes if ok else failures).append(description)

    coverage_gain = (
        candidate.worst_domain_direct_coverage - incumbent.worst_domain_direct_coverage
    )
    check(
        coverage_gain >= criteria.min_worst_domain_coverage_gain,
        f"worst-domain direct coverage {incumbent.worst_domain_direct_coverage:.4f} "
        f"-> {candidate.worst_domain_direct_coverage:.4f} ({coverage_gain:+.4f}, "
        f"need >= +{criteria.min_worst_domain_coverage_gain})",
    )

    recall_gain = (
        candidate.macro_centre_recall_25px - incumbent.macro_centre_recall_25px
    )
    check(
        recall_gain >= criteria.min_centre_recall_gain,
        f"cross-domain centre recall @25px {incumbent.macro_centre_recall_25px:.4f} "
        f"-> {candidate.macro_centre_recall_25px:.4f} ({recall_gain:+.4f}, "
        f"need >= +{criteria.min_centre_recall_gain})",
    )

    check(
        candidate.macro_precision_25px >= criteria.min_precision,
        f"precision @25px {candidate.macro_precision_25px:.4f} "
        f"(need >= {criteria.min_precision})",
    )

    if incumbent.false_positives_per_frame > 0:
        growth = (
            candidate.false_positives_per_frame / incumbent.false_positives_per_frame
        ) - 1.0
        check(
            growth <= criteria.max_false_positive_growth,
            f"false positives/frame {incumbent.false_positives_per_frame:.4f} -> "
            f"{candidate.false_positives_per_frame:.4f} ({growth:+.1%}, "
            f"cap +{criteria.max_false_positive_growth:.0%})",
        )

    for domain, value in sorted(candidate.per_domain_centre_recall_25px.items()):
        baseline = incumbent.per_domain_centre_recall_25px.get(domain)
        if baseline is None:
            failures.append(f"{domain} not measured for the incumbent")
            continue
        check(
            value - baseline >= -criteria.max_domain_regression,
            f"{domain} centre recall {value - baseline:+.4f} "
            f"(tolerance -{criteria.max_domain_regression})",
        )

    for name, got, base, minimum in (
        ("possession determinability", candidate.determinability,
         incumbent.determinability, criteria.min_determinability_gain),
        ("downstream pass recall", candidate.pass_recall,
         incumbent.pass_recall, criteria.min_pass_recall_gain),
    ):
        if got is None or base is None:
            # Not measured is not a pass. A representation cannot be promoted on
            # evidence nobody collected.
            failures.append(f"{name} not measured for both representations")
            continue
        check(
            got - base >= minimum,
            f"{name} {base:.4f} -> {got:.4f} ({got - base:+.4f}, need >= {minimum})",
        )

    if not candidate.model_fingerprint:
        failures.append("candidate model fingerprint must be recorded before promotion")

    verdict = RepresentationVerdict(
        promote=not failures,
        candidate=candidate.label,
        incumbent=incumbent.label,
        passes=passes,
        failures=failures,
        criteria=criteria,
    )
    log.info(
        "representation %s -> %s: %s (%d pass, %d fail)",
        incumbent.label, candidate.label,
        "PROMOTE" if verdict.promote else "REJECT", len(passes), len(failures),
    )
    return verdict
