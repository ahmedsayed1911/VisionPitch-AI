"""The Phase 2D ball-model promotion rule, as executable code.

Phase 2D, Part 10.

The rule lives here rather than in a script so it can be unit-tested and cannot
be quietly reinterpreted while writing up results. Every criterion is a field
with a fixed default; changing one is a visible code change with a visible test
failure, not a sentence in a report.

Phase 2C rejected a checkpoint that failed a single criterion by 0.0011. That is
the behaviour this module is meant to preserve: the rule decides, and the
write-up explains, never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from visionpitch.common.logging import get_logger

log = get_logger("evaluation.promotion")

PROMOTION_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True)
class PromotionCriteria:
    """Thresholds, declared before any candidate is measured.

    ``max_domain_regression`` applies per domain and per metric. It exists
    because a mean can improve while the weakest domain collapses, which is
    exactly what the Phase 2B single-corpus fine-tune did.
    """

    #: minimum improvement in cross-domain recall to count as material
    min_recall_gain: float = 0.02
    #: precision floor the candidate must clear outright
    min_precision: float = 0.55
    #: largest tolerated per-domain drop in any headline metric
    max_domain_regression: float = 0.02
    #: worst-domain recall must not fall by more than this
    max_worst_domain_regression: float = 0.02
    #: effective ball coverage must improve by at least this
    min_coverage_gain: float = 0.0
    #: possession determinability must improve by at least this
    min_determinability_gain: float = 0.0
    #: downstream pass recall must improve by at least this
    min_pass_recall_gain: float = 0.0
    #: recovered/interpolated positions may not grow faster than this multiple
    #: of the gain in direct observations -- the anti-hallucination guard
    max_inferred_to_direct_ratio: float = 1.0

    def to_dict(self) -> dict:
        return {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "min_recall_gain": self.min_recall_gain,
            "min_precision": self.min_precision,
            "max_domain_regression": self.max_domain_regression,
            "max_worst_domain_regression": self.max_worst_domain_regression,
            "min_coverage_gain": self.min_coverage_gain,
            "min_determinability_gain": self.min_determinability_gain,
            "min_pass_recall_gain": self.min_pass_recall_gain,
            "max_inferred_to_direct_ratio": self.max_inferred_to_direct_ratio,
        }


@dataclass
class CandidateMeasurements:
    """Everything the rule needs about one configuration.

    ``per_domain_recall`` and ``per_domain_precision`` must cover the same
    domains for candidate and incumbent, or the comparison is meaningless and
    :func:`evaluate_promotion` refuses it.
    """

    label: str
    per_domain_recall: dict[str, float]
    per_domain_precision: dict[str, float]
    effective_ball_coverage: float | None = None
    possession_determinability: float | None = None
    pass_recall: float | None = None
    #: frames whose ball position came from the detector itself
    n_direct_observations: int | None = None
    #: frames filled by recovery or interpolation
    n_inferred_observations: int | None = None
    #: gaps longer than the fusion limit that were filled anyway -- must be 0
    n_long_gap_fills: int = 0
    runtime_s_per_1k_frames: float | None = None
    model_fingerprint: str = ""
    config_fingerprint: str = ""

    @property
    def mean_recall(self) -> float:
        values = list(self.per_domain_recall.values())
        return sum(values) / len(values) if values else 0.0

    @property
    def mean_precision(self) -> float:
        values = list(self.per_domain_precision.values())
        return sum(values) / len(values) if values else 0.0

    @property
    def worst_domain_recall(self) -> float:
        return min(self.per_domain_recall.values()) if self.per_domain_recall else 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "per_domain_recall": self.per_domain_recall,
            "per_domain_precision": self.per_domain_precision,
            "mean_recall": round(self.mean_recall, 4),
            "mean_precision": round(self.mean_precision, 4),
            "worst_domain_recall": round(self.worst_domain_recall, 4),
            "effective_ball_coverage": self.effective_ball_coverage,
            "possession_determinability": self.possession_determinability,
            "pass_recall": self.pass_recall,
            "n_direct_observations": self.n_direct_observations,
            "n_inferred_observations": self.n_inferred_observations,
            "n_long_gap_fills": self.n_long_gap_fills,
            "runtime_s_per_1k_frames": self.runtime_s_per_1k_frames,
            "model_fingerprint": self.model_fingerprint,
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass
class PromotionVerdict:
    promote: bool
    candidate: str
    incumbent: str
    failures: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)
    criteria: PromotionCriteria = field(default_factory=PromotionCriteria)

    def to_dict(self) -> dict:
        return {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "promote_to_default": self.promote,
            "candidate": self.candidate,
            "incumbent": self.incumbent,
            "criteria": self.criteria.to_dict(),
            "passes": self.passes,
            "failures": self.failures,
            "reason": (
                "every declared criterion met"
                if self.promote else "; ".join(self.failures)
            ),
        }


def evaluate_promotion(
    candidate: CandidateMeasurements,
    incumbent: CandidateMeasurements,
    criteria: PromotionCriteria | None = None,
) -> PromotionVerdict:
    """Apply the rule. Every criterion must pass; there is no weighting."""
    criteria = criteria or PromotionCriteria()
    failures: list[str] = []
    passes: list[str] = []

    if set(candidate.per_domain_recall) != set(incumbent.per_domain_recall):
        raise ValueError(
            "candidate and incumbent were measured on different domain sets "
            f"({sorted(candidate.per_domain_recall)} vs "
            f"{sorted(incumbent.per_domain_recall)}); the comparison would be "
            "meaningless"
        )

    def check(condition: bool, description: str) -> None:
        (passes if condition else failures).append(description)

    recall_gain = candidate.mean_recall - incumbent.mean_recall
    check(
        recall_gain >= criteria.min_recall_gain,
        f"cross-domain recall {incumbent.mean_recall:.4f} -> "
        f"{candidate.mean_recall:.4f} (gain {recall_gain:+.4f}, "
        f"need >= {criteria.min_recall_gain})",
    )

    check(
        candidate.mean_precision >= criteria.min_precision,
        f"cross-domain precision {candidate.mean_precision:.4f} "
        f"(need >= {criteria.min_precision})",
    )

    worst_delta = candidate.worst_domain_recall - incumbent.worst_domain_recall
    check(
        worst_delta >= -criteria.max_worst_domain_regression,
        f"worst-domain recall {incumbent.worst_domain_recall:.4f} -> "
        f"{candidate.worst_domain_recall:.4f} ({worst_delta:+.4f}, "
        f"tolerance {-criteria.max_worst_domain_regression})",
    )

    for domain in sorted(candidate.per_domain_recall):
        for name, table in (
            ("recall", (candidate.per_domain_recall, incumbent.per_domain_recall)),
            ("precision", (candidate.per_domain_precision, incumbent.per_domain_precision)),
        ):
            delta = table[0].get(domain, 0.0) - table[1].get(domain, 0.0)
            check(
                delta >= -criteria.max_domain_regression,
                f"{domain} {name} {delta:+.4f} "
                f"(tolerance {-criteria.max_domain_regression})",
            )

    for label, value, baseline, minimum in (
        ("effective ball coverage", candidate.effective_ball_coverage,
         incumbent.effective_ball_coverage, criteria.min_coverage_gain),
        ("possession determinability", candidate.possession_determinability,
         incumbent.possession_determinability, criteria.min_determinability_gain),
        ("downstream pass recall", candidate.pass_recall,
         incumbent.pass_recall, criteria.min_pass_recall_gain),
    ):
        if value is None or baseline is None:
            # Not measured is not a pass. A candidate cannot be promoted on the
            # strength of evidence nobody collected.
            failures.append(f"{label} not measured for both configurations")
            continue
        check(
            value - baseline >= minimum,
            f"{label} {baseline:.4f} -> {value:.4f} "
            f"({value - baseline:+.4f}, need >= {minimum})",
        )

    check(
        candidate.n_long_gap_fills == 0,
        f"long-gap fills {candidate.n_long_gap_fills} (must be 0)",
    )

    if (
        candidate.n_direct_observations is not None
        and incumbent.n_direct_observations is not None
        and candidate.n_inferred_observations is not None
        and incumbent.n_inferred_observations is not None
    ):
        direct_gain = candidate.n_direct_observations - incumbent.n_direct_observations
        inferred_gain = (
            candidate.n_inferred_observations - incumbent.n_inferred_observations
        )
        # Coverage bought mostly with inference is not coverage. If the inferred
        # count grows faster than the direct count, the configuration is
        # guessing more, not seeing more.
        acceptable = inferred_gain <= max(0, direct_gain) * criteria.max_inferred_to_direct_ratio
        check(
            acceptable or inferred_gain <= 0,
            f"inferred positions {inferred_gain:+d} against direct "
            f"{direct_gain:+d} (ratio cap {criteria.max_inferred_to_direct_ratio})",
        )

    if not candidate.model_fingerprint or not candidate.config_fingerprint:
        failures.append(
            "model and config fingerprints must both be recorded before promotion"
        )

    verdict = PromotionVerdict(
        promote=not failures,
        candidate=candidate.label,
        incumbent=incumbent.label,
        failures=failures,
        passes=passes,
        criteria=criteria,
    )
    log.info(
        "promotion %s -> %s: %s (%d pass, %d fail)",
        incumbent.label, candidate.label,
        "PROMOTE" if verdict.promote else "REJECT",
        len(passes), len(failures),
    )
    return verdict
