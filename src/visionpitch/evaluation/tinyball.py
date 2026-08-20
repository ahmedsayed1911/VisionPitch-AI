"""Fixed data protocol and centre metrics for the tiny-ball representation study.

Parts 2 and 3.

Two decisions frame everything else.

**The existing clip assignment is reused, not replaced.** Phase 2C's
clip-disjoint split is already fingerprinted, already leak-checked, and every
baseline number in the repository was measured on it. Re-partitioning now would
make the box baseline incomparable with anything a new representation produces,
which is the one thing this study cannot afford. The study adds a *fourth*
partition -- cross-domain validation -- derived by leave-one-domain-out inside
train and val only. The final test clips are untouched.

**Centre metrics are added, not substituted.** IoU50 stays, because every prior
figure uses it and dropping it would quietly relabel a measurement change as
progress. But it is reported alongside centre-distance recall at fixed
tolerances, because Part 1 measured that IoU50 on a median 11.5 px ball demands
3.86 px of centre accuracy while the possession engine tolerates 64.8 px -- a
factor of roughly seventeen. A metric seventeen times stricter than the task is
not measuring the product.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from visionpitch.common.logging import get_logger

log = get_logger("evaluation.tinyball")

TINYBALL_SCHEMA_VERSION = "1.0.0"

#: Centre-distance tolerances reported for every representation.
#: 25 px is the Phase 2B operating tolerance; the tighter values expose whether
#: a representation is merely *finding* the ball or actually localising it.
CENTRE_TOLERANCES_PX = (5.0, 10.0, 15.0, 20.0, 25.0)

#: Measured in Part 1: 0.6 player-heights at a median player box of 108 px.
#: A centre error below this changes no possession decision.
POSSESSION_TOLERANCE_PX = 64.8


class Partition(str, Enum):
    """The four fixed partitions."""

    TRAIN = "train"
    VAL_IN_DOMAIN = "val_in_domain"
    VAL_CROSS_DOMAIN = "val_cross_domain"
    TEST = "test"

    @property
    def is_tunable(self) -> bool:
        """Whether anything may be selected on this partition.

        ``TEST`` is scored once per representation and never used to choose a
        hyper-parameter, a checkpoint, a threshold or a stopping point.
        """
        return self is not Partition.TEST


@dataclass
class TinyBallProtocol:
    """The fixed protocol, with every input fingerprinted.

    ``cross_domain_holdout`` names the domain withheld from training when
    measuring transfer. With only two labelled corpora this is the entire
    available cross-domain design, and the study says so rather than implying a
    richer one.
    """

    dataset_root: Path
    base_split_fingerprint: str
    cross_domain_holdout: str
    domains: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    augmentation: dict = field(default_factory=dict)
    schema_version: str = TINYBALL_SCHEMA_VERSION

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "base_split": self.base_split_fingerprint,
                "holdout": self.cross_domain_holdout,
                "domains": sorted(self.domains),
                "counts": self.counts,
                "augmentation": self.augmentation,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dataset_root": str(self.dataset_root),
            "base_split_fingerprint": self.base_split_fingerprint,
            "cross_domain_holdout": self.cross_domain_holdout,
            "domains": sorted(self.domains),
            "counts": self.counts,
            "augmentation": self.augmentation,
            "protocol_fingerprint": self.fingerprint(),
            "partitions": {
                "train": "base split train clips, all domains",
                "val_in_domain": "base split val clips, all domains",
                "val_cross_domain": (
                    f"base split train+val clips of {self.cross_domain_holdout} "
                    "only, withheld from training when measuring transfer"
                ),
                "test": "base split test clips, scored once, never tuned on",
            },
            "note": (
                "Only two corpora carry ball boxes, so leave-one-domain-out over "
                "two domains is the whole cross-domain design available. "
                "SoccerNet-BAS -- the domain where coverage actually fails -- has "
                "no ball annotations at all and can only be probed for coverage, "
                "never for recall or precision."
            ),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> TinyBallProtocol:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        protocol = TinyBallProtocol(
            dataset_root=Path(data["dataset_root"]),
            base_split_fingerprint=data["base_split_fingerprint"],
            cross_domain_holdout=data["cross_domain_holdout"],
            domains=data["domains"],
            counts=data["counts"],
            augmentation=data.get("augmentation", {}),
            schema_version=data.get("schema_version", TINYBALL_SCHEMA_VERSION),
        )
        if protocol.fingerprint() != data["protocol_fingerprint"]:
            raise ValueError(
                f"{path} was edited: stored {data['protocol_fingerprint']} != "
                f"recomputed {protocol.fingerprint()}"
            )
        return protocol


def domain_of(path: Path | str) -> str:
    name = Path(path).name
    return "soccernet_gsr" if name.startswith("soccernet_gsr_") else "roboflow"


def clip_of(path: Path | str) -> str:
    """Source clip id encoded in the dataset builder's filename."""
    stem = Path(path).stem
    parts = stem.split("_")
    if stem.startswith("soccernet_gsr_"):
        return parts[2] if len(parts) > 2 else "unknown"
    return parts[1] if len(parts) > 1 else "unknown"


def assert_clip_disjoint(partitions: dict[str, list[Path]]) -> None:
    """No source clip may appear in two partitions.

    Checked on the actual file listing rather than on the split record, so a
    dataset rebuild that silently mixes clips is caught even if the recorded
    assignment still looks right.
    """
    clips = {
        name: {(domain_of(p), clip_of(p)) for p in paths}
        for name, paths in partitions.items()
    }
    names = sorted(clips)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = clips[a] & clips[b]
            if shared:
                raise AssertionError(
                    f"{a} and {b} share {len(shared)} source clip(s): "
                    f"{sorted(shared)[:5]}"
                )


# --------------------------------------------------------------------------- #
# Centre metrics
# --------------------------------------------------------------------------- #


@dataclass
class CentreResult:
    """Centre-localisation performance for one representation on one domain."""

    label: str
    domain: str
    n_truth: int = 0
    n_predicted: int = 0
    #: tolerance px -> number of ground-truth balls matched within it
    hits_at: dict[float, int] = field(default_factory=dict)
    errors_px: list[float] = field(default_factory=list)
    n_frames: int = 0
    n_frames_with_prediction: int = 0

    def recall_at(self, tolerance: float) -> float:
        return self.hits_at.get(tolerance, 0) / self.n_truth if self.n_truth else 0.0

    def precision_at(self, tolerance: float) -> float:
        return (
            self.hits_at.get(tolerance, 0) / self.n_predicted
            if self.n_predicted else 0.0
        )

    @property
    def median_error_px(self) -> float | None:
        return float(np.median(self.errors_px)) if self.errors_px else None

    @property
    def mean_error_px(self) -> float | None:
        return float(np.mean(self.errors_px)) if self.errors_px else None

    @property
    def false_positives_per_frame(self) -> float:
        if not self.n_frames:
            return 0.0
        matched = self.hits_at.get(max(CENTRE_TOLERANCES_PX), 0)
        return max(0, self.n_predicted - matched) / self.n_frames

    @property
    def direct_coverage(self) -> float:
        """Share of frames in which the model emitted any ball position."""
        return (
            self.n_frames_with_prediction / self.n_frames if self.n_frames else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "domain": self.domain,
            "n_truth": self.n_truth,
            "n_predicted": self.n_predicted,
            "n_frames": self.n_frames,
            "recall_at_px": {
                str(t): round(self.recall_at(t), 4) for t in CENTRE_TOLERANCES_PX
            },
            "precision_at_px": {
                str(t): round(self.precision_at(t), 4) for t in CENTRE_TOLERANCES_PX
            },
            "median_error_px": (
                round(self.median_error_px, 3) if self.median_error_px is not None else None
            ),
            "mean_error_px": (
                round(self.mean_error_px, 3) if self.mean_error_px is not None else None
            ),
            "false_positives_per_frame": round(self.false_positives_per_frame, 4),
            "direct_coverage": round(self.direct_coverage, 4),
            "possession_usable_rate": round(
                sum(1 for e in self.errors_px if e <= POSSESSION_TOLERANCE_PX)
                / self.n_truth, 4
            ) if self.n_truth else 0.0,
        }


def score_centres(
    label: str,
    domain: str,
    per_frame: list[tuple[list[tuple[float, float]], list[tuple[float, float]]]],
) -> CentreResult:
    """Greedy nearest-first centre matching.

    ``per_frame`` is a list of ``(truth_centres, predicted_centres)``. Matching
    is one-to-one and nearest-first, so a single prediction cannot satisfy two
    ground-truth balls and duplicate predictions count as false positives.
    """
    result = CentreResult(label=label, domain=domain)
    for truths, predictions in per_frame:
        result.n_frames += 1
        result.n_truth += len(truths)
        result.n_predicted += len(predictions)
        if predictions:
            result.n_frames_with_prediction += 1
        if not truths or not predictions:
            continue

        pairs = sorted(
            (
                float(np.hypot(p[0] - t[0], p[1] - t[1])), ti, pi
            )
            for ti, t in enumerate(truths)
            for pi, p in enumerate(predictions)
        )
        used_truth: set[int] = set()
        used_prediction: set[int] = set()
        for distance, ti, pi in pairs:
            if ti in used_truth or pi in used_prediction:
                continue
            used_truth.add(ti)
            used_prediction.add(pi)
            result.errors_px.append(distance)
            for tolerance in CENTRE_TOLERANCES_PX:
                if distance <= tolerance:
                    result.hits_at[tolerance] = result.hits_at.get(tolerance, 0) + 1
    return result


def pool(results: list[CentreResult], label: str) -> dict:
    """Pool per-domain results by summing counts, and report the worst domain.

    Worst-domain is reported beside the macro average throughout this study
    because the whole cross-domain question is about the weakest corpus, and a
    mean hides exactly that.
    """
    if not results:
        return {"label": label, "n_domains": 0}
    per_domain = {r.domain: r.to_dict() for r in results}
    macro = {
        str(t): round(
            sum(r.recall_at(t) for r in results) / len(results), 4
        )
        for t in CENTRE_TOLERANCES_PX
    }
    worst = {
        str(t): round(min(r.recall_at(t) for r in results), 4)
        for t in CENTRE_TOLERANCES_PX
    }
    all_errors = [e for r in results for e in r.errors_px]
    return {
        "label": label,
        "n_domains": len(results),
        "per_domain": per_domain,
        "macro_recall_at_px": macro,
        "worst_domain_recall_at_px": worst,
        "macro_direct_coverage": round(
            sum(r.direct_coverage for r in results) / len(results), 4
        ),
        "worst_domain_direct_coverage": round(
            min(r.direct_coverage for r in results), 4
        ),
        "median_error_px": (
            round(float(np.median(all_errors)), 3) if all_errors else None
        ),
        "macro_false_positives_per_frame": round(
            sum(r.false_positives_per_frame for r in results) / len(results), 4
        ),
    }
