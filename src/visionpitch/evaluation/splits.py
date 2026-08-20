"""Deterministic, leak-free dataset splits.

Phase 2B requires that the clips used for detector adaptation, for threshold
selection, and for the final held-out evaluation are disjoint. The unit of
splitting is the **sequence**, never the frame: consecutive frames of the same
clip are almost identical, so a frame-level split would put near-duplicates of
the test data into training and report a number that means nothing.

The assignment is a hash of the sequence name, so it is stable across machines
and runs without storing a manifest, and adding sequences later does not
reshuffle the existing ones.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from visionpitch.common.logging import get_logger

log = get_logger("evaluation.splits")

SPLIT_SCHEMA_VERSION = "1.0.0"

#: Proportions. Test is deliberately generous: the final number matters more
#: than squeezing the last few sequences into training.
DEFAULT_RATIOS = {"train": 0.55, "val": 0.20, "test": 0.25}


@dataclass
class DatasetSplit:
    """Which sequence belongs to which split, and why it is reproducible."""

    name: str
    seed: str
    ratios: dict[str, float]
    assignments: dict[str, str] = field(default_factory=dict)

    @property
    def train(self) -> list[str]:
        return sorted(k for k, v in self.assignments.items() if v == "train")

    @property
    def val(self) -> list[str]:
        return sorted(k for k, v in self.assignments.items() if v == "val")

    @property
    def test(self) -> list[str]:
        return sorted(k for k, v in self.assignments.items() if v == "test")

    def of(self, sequence: str) -> str:
        return self.assignments.get(sequence, "unassigned")

    def fingerprint(self) -> str:
        """Hash of the exact membership, for the run manifest."""
        payload = json.dumps(
            {"name": self.name, "seed": self.seed, "assignments": self.assignments},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "ratios": self.ratios,
            "counts": {
                "train": len(self.train),
                "val": len(self.val),
                "test": len(self.test),
            },
            "fingerprint": self.fingerprint(),
            "assignments": dict(sorted(self.assignments.items())),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> DatasetSplit:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        split = DatasetSplit(
            name=data["name"], seed=data["seed"], ratios=data["ratios"],
            assignments=data["assignments"],
        )
        if split.fingerprint() != data["fingerprint"]:
            raise ValueError(
                f"split file {path} has been edited: stored fingerprint "
                f"{data['fingerprint']} != recomputed {split.fingerprint()}"
            )
        return split


def make_split(
    sequences: list[str],
    name: str = "sn_gsr_ball",
    seed: str = "visionpitch-phase2b",
    ratios: dict[str, float] | None = None,
) -> DatasetSplit:
    """Assign sequences to train/val/test by a stable hash of their names.

    Hashing rather than shuffling: the assignment for a given sequence never
    changes, so adding new sequences cannot silently move an old one from test
    into train and invalidate a previously reported number.
    """
    ratios = ratios or dict(DEFAULT_RATIOS)
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    train_cut = ratios["train"]
    val_cut = train_cut + ratios["val"]

    assignments: dict[str, str] = {}
    for sequence in sequences:
        digest = hashlib.sha256(f"{seed}/{sequence}".encode()).digest()
        # 32 bits of the digest as a uniform value in [0, 1).
        position = int.from_bytes(digest[:4], "big") / 2**32
        if position < train_cut:
            assignments[sequence] = "train"
        elif position < val_cut:
            assignments[sequence] = "val"
        else:
            assignments[sequence] = "test"

    split = DatasetSplit(name=name, seed=seed, ratios=ratios, assignments=assignments)
    log.info(
        "%s split: %d train / %d val / %d test (fingerprint %s)",
        name, len(split.train), len(split.val), len(split.test), split.fingerprint(),
    )
    return split


def assert_disjoint(split: DatasetSplit) -> None:
    """Guard against the mistake this module exists to prevent."""
    train, val, test = set(split.train), set(split.val), set(split.test)
    overlaps = {
        "train/val": train & val,
        "train/test": train & test,
        "val/test": val & test,
    }
    for label, shared in overlaps.items():
        if shared:
            raise AssertionError(f"{label} splits share sequences: {sorted(shared)}")
