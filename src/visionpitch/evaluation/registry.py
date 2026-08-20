"""Multi-corpus dataset registry and clip-disjoint splits.

Phase 2C, Part 1.

Why this module exists
----------------------
An audit of the corpora already on disk found that the two Roboflow datasets
(`martinjolif/football-ball-detection` and `.../football-player-detection`) are
drawn from the **same 17 source match clips**, and that their published
train/test split is a random *frame* split: all 14 clips in `ball_det/test`
also appear in `ball_det/train`.

Two consequences, both of which change how results must be read:

* the previously reported in-distribution ball recall of 0.912 was measured on
  frames from matches the model trained on, and is therefore optimistic
* the two Roboflow sets are one domain, not two

This module re-splits every corpus **by source clip**, so a frame of a test
match can never appear in training, and records the provenance needed to
reproduce that decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from visionpitch.common.logging import get_logger

log = get_logger("evaluation.registry")

REGISTRY_SCHEMA_VERSION = "1.0.0"

#: Roboflow filenames encode the source match as a 6-hex prefix.
_ROBOFLOW_CLIP = re.compile(r"^([0-9a-f]{6})_")


@dataclass
class CorpusRecord:
    """One dataset, described well enough to decide what it may be used for."""

    key: str
    name: str
    source: str
    licence: str
    competition: str
    broadcast_style: str
    resolution: str
    frame_rate: str
    camera: str
    ball_annotation: str
    event_annotation: str
    domain: str
    n_clips: int = 0
    n_frames: int = 0
    n_ball_instances: int = 0
    train_eligible: bool = False
    val_eligible: bool = False
    test_eligible: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


#: The corpora available to this project. Eligibility is a property of the
#: dataset, not of convenience: a corpus with no ball boxes cannot train a ball
#: detector however useful it is for evaluating coverage.
CORPORA: dict[str, CorpusRecord] = {
    "roboflow": CorpusRecord(
        key="roboflow",
        name="martinjolif football-ball-detection + football-player-detection",
        source="HuggingFace (Roboflow workspace football-project-pifbc)",
        licence="CC BY 4.0",
        competition="mixed European club football",
        broadcast_style="tactical/broadcast crops, mixed",
        resolution="1920x1080 (variable crops)",
        frame_rate="unknown (still frames)",
        camera="broadcast main camera",
        ball_annotation="bounding box",
        event_annotation="none",
        domain="roboflow",
        train_eligible=True,
        val_eligible=True,
        test_eligible=True,
        notes=(
            "The two published datasets share source clips and are treated as ONE "
            "domain. The official train/test split is a random frame split with "
            "14/14 test clips also present in train; it is discarded here in "
            "favour of a clip-disjoint split."
        ),
    ),
    "gsr": CorpusRecord(
        key="gsr",
        name="SoccerNet SN-GSR-2025 (Game State Reconstruction)",
        source="HuggingFace SoccerNet/SN-GSR-2025 (test.zip)",
        licence="SoccerNet terms; non-commercial research",
        competition="mixed European leagues",
        broadcast_style="broadcast main camera, 30s sequences",
        resolution="1920x1080",
        frame_rate="25",
        camera="panning broadcast",
        ball_annotation="bounding box, per frame",
        event_annotation="none (tracking + identity + role)",
        domain="soccernet_gsr",
        train_eligible=True,
        val_eligible=True,
        test_eligible=True,
        notes="Split by sequence; sequences are distinct matches or distinct phases.",
    ),
    "bas": CorpusRecord(
        key="bas",
        name="SoccerNet SN-BAS-2025 (Ball Action Spotting)",
        source="HuggingFace SoccerNet/SN-BAS-2025 (valid.zip)",
        licence="SoccerNet terms; non-commercial research",
        competition="England EFL",
        broadcast_style="full-match broadcast",
        resolution="1280x720",
        frame_rate="25",
        camera="broadcast main camera with cuts and replays",
        ball_annotation="NONE",
        event_annotation="timestamped ball actions, no player identity",
        domain="soccernet_bas",
        train_eligible=False,
        val_eligible=False,
        test_eligible=True,
        notes=(
            "No ball boxes, so it cannot train or validate a ball detector. It is "
            "the cross-domain *coverage and event* test set: the only corpus here "
            "measuring what the pipeline does on footage it has never seen in any "
            "form."
        ),
    ),
}


# --------------------------------------------------------------------------- #
# Clip-disjoint splits
# --------------------------------------------------------------------------- #


@dataclass
class MultiCorpusSplit:
    """Clip-level split assignments across every corpus."""

    seed: str
    ratios: dict[str, float]
    #: corpus key -> clip id -> split name
    assignments: dict[str, dict[str, str]] = field(default_factory=dict)
    schema_version: str = REGISTRY_SCHEMA_VERSION

    def clips(self, corpus: str, split: str) -> list[str]:
        return sorted(
            c for c, s in self.assignments.get(corpus, {}).items() if s == split
        )

    def split_of(self, corpus: str, clip: str) -> str:
        return self.assignments.get(corpus, {}).get(clip, "unassigned")

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"seed": self.seed, "assignments": self.assignments}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "ratios": self.ratios,
            "fingerprint": self.fingerprint(),
            "counts": {
                corpus: {
                    split: len(self.clips(corpus, split))
                    for split in ("train", "val", "test")
                }
                for corpus in self.assignments
            },
            "assignments": {
                corpus: dict(sorted(clips.items()))
                for corpus, clips in sorted(self.assignments.items())
            },
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> MultiCorpusSplit:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        split = MultiCorpusSplit(
            seed=data["seed"], ratios=data["ratios"], assignments=data["assignments"],
            schema_version=data.get("schema_version", REGISTRY_SCHEMA_VERSION),
        )
        if split.fingerprint() != data["fingerprint"]:
            raise ValueError(
                f"{path} was edited: stored fingerprint {data['fingerprint']} "
                f"!= recomputed {split.fingerprint()}"
            )
        return split


def _assign(clip: str, corpus: str, seed: str, ratios: dict[str, float]) -> str:
    digest = hashlib.sha256(f"{seed}/{corpus}/{clip}".encode()).digest()
    position = int.from_bytes(digest[:4], "big") / 2**32
    if position < ratios["train"]:
        return "train"
    if position < ratios["train"] + ratios["val"]:
        return "val"
    return "test"


def roboflow_clip_ids(root: Path) -> dict[str, list[Path]]:
    """Map source-clip id -> every image belonging to it, across all splits.

    The published split is deliberately ignored. Its test set shares all 14
    source clips with its training set, so honouring it would measure memorised
    frames.
    """
    clips: dict[str, list[Path]] = {}
    for split in ("train", "valid", "test"):
        images = root / "data" / split / "images"
        if not images.exists():
            continue
        for path in images.glob("*.*"):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            match = _ROBOFLOW_CLIP.match(path.name)
            clip = match.group(1) if match else "unknown"
            clips.setdefault(clip, []).append(path)
    return clips


def build_split(
    roboflow_roots: list[Path],
    gsr_sequences: list[str],
    seed: str = "visionpitch-phase2c",
    ratios: dict[str, float] | None = None,
) -> MultiCorpusSplit:
    """Assign every source clip and sequence to exactly one split."""
    ratios = ratios or {"train": 0.55, "val": 0.20, "test": 0.25}
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    assignments: dict[str, dict[str, str]] = {"roboflow": {}, "soccernet_gsr": {}}

    seen: set[str] = set()
    for root in roboflow_roots:
        for clip in roboflow_clip_ids(Path(root)):
            seen.add(clip)
    for clip in sorted(seen):
        assignments["roboflow"][clip] = _assign(clip, "roboflow", seed, ratios)

    for sequence in sorted(gsr_sequences):
        assignments["soccernet_gsr"][sequence] = _assign(
            sequence, "soccernet_gsr", seed, ratios
        )

    split = MultiCorpusSplit(seed=seed, ratios=ratios, assignments=assignments)
    for corpus in assignments:
        log.info(
            "%s: %d train / %d val / %d test clips",
            corpus, len(split.clips(corpus, "train")),
            len(split.clips(corpus, "val")), len(split.clips(corpus, "test")),
        )
    log.info("multi-corpus split fingerprint %s", split.fingerprint())
    return split


def assert_no_leakage(split: MultiCorpusSplit) -> None:
    """Every clip belongs to exactly one split, within every corpus."""
    for corpus, clips in split.assignments.items():
        train = set(split.clips(corpus, "train"))
        val = set(split.clips(corpus, "val"))
        test = set(split.clips(corpus, "test"))
        for a, b, label in (
            (train, val, "train/val"), (train, test, "train/test"), (val, test, "val/test")
        ):
            if a & b:
                raise AssertionError(f"{corpus} {label} share clips: {sorted(a & b)}")
        if set(clips) != train | val | test:
            raise AssertionError(f"{corpus}: some clips are unassigned")


def registry_document() -> dict:
    """The registry, for the docs and for the run manifest."""
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "corpora": {k: v.to_dict() for k, v in CORPORA.items()},
        "domains": sorted({c.domain for c in CORPORA.values()}),
        "ball_training_domains": sorted(
            {c.domain for c in CORPORA.values() if c.train_eligible}
        ),
        "coverage_only_domains": sorted(
            {c.domain for c in CORPORA.values() if not c.train_eligible and c.test_eligible}
        ),
    }
