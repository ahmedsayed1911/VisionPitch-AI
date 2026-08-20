"""Shot-disjoint splitting of the local broadcast annotations.

Splitting 115 frames drawn from 26 shots by *frame* would put near-identical
neighbours on both sides of the boundary and report a memorisation score. The
unit here is the **broadcast shot**, so every frame from one continuous camera
take lands in exactly one split.

Temporal windows are a second hazard: a window is seven consecutive frames and
must never straddle a split. Windows live inside a single shot by construction,
so splitting by shot handles them, and the check below asserts it rather than
assuming it.

Allocation is by shot, greedily balanced on frame count, because shots vary from
1 to 12 reviewed frames and a naive 60/20/20 over shot *ids* would produce wildly
uneven frame counts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from visionpitch.annotation.schema import (
    AnnotationStore,
    BallAnnotation,
    BallVisibility,
    FrameSample,
)
from visionpitch.common.logging import get_logger

log = get_logger("annotation.splits")

LOCAL_SPLIT_SCHEMA_VERSION = "1.0.0"

DEFAULT_RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}


@dataclass
class LocalSplit:
    """Shot-level assignment for the local broadcast set."""

    ratios: dict[str, float]
    seed: str
    #: shot index -> split name
    shot_assignment: dict[int, str] = field(default_factory=dict)
    #: split name -> frame ids
    frames: dict[str, list[str]] = field(default_factory=dict)
    schema_version: str = LOCAL_SPLIT_SCHEMA_VERSION

    def split_of_frame(self, frame_id: str) -> str | None:
        for name, ids in self.frames.items():
            if frame_id in ids:
                return name
        return None

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "seed": self.seed,
                "ratios": self.ratios,
                "shots": {str(k): v for k, v in sorted(self.shot_assignment.items())},
                "frames": {k: sorted(v) for k, v in sorted(self.frames.items())},
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "ratios": self.ratios,
            "unit": "broadcast shot",
            "counts": {k: len(v) for k, v in sorted(self.frames.items())},
            "shots_per_split": {
                name: sorted(s for s, n in self.shot_assignment.items() if n == name)
                for name in sorted(self.frames)
            },
            "frames": {k: sorted(v) for k, v in sorted(self.frames.items())},
            "fingerprint": self.fingerprint(),
            "locked_test_note": (
                "the test split must not be used for training, checkpoint "
                "selection, threshold tuning, augmentation choice or early "
                "stopping; it is scored once per candidate"
            ),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> LocalSplit:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        split = LocalSplit(
            ratios=data["ratios"], seed=data["seed"],
            shot_assignment={
                int(s): name
                for name, shots in data["shots_per_split"].items()
                for s in shots
            },
            frames={k: list(v) for k, v in data["frames"].items()},
            schema_version=data.get("schema_version", LOCAL_SPLIT_SCHEMA_VERSION),
        )
        if split.fingerprint() != data["fingerprint"]:
            raise ValueError(
                f"{path} was edited: stored {data['fingerprint']} != recomputed "
                f"{split.fingerprint()}"
            )
        return split


def build_local_split(
    samples: dict[str, FrameSample],
    annotations: dict[str, BallAnnotation],
    ratios: dict[str, float] | None = None,
    seed: str = "visionpitch-broadcast-2026",
) -> LocalSplit:
    """Assign whole shots to splits, balancing on frame count."""
    ratios = ratios or DEFAULT_RATIOS
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    reviewed = [f for f in annotations if f in samples]
    by_shot: dict[int, list[str]] = {}
    for frame_id in reviewed:
        by_shot.setdefault(samples[frame_id].shot_index, []).append(frame_id)

    n_frames = len(reviewed)
    targets = {name: ratio * n_frames for name, ratio in ratios.items()}
    assigned: dict[str, list[str]] = {name: [] for name in ratios}
    shot_assignment: dict[int, str] = {}

    # Largest shots first, each to whichever split is furthest below its target.
    # Greedy rather than random: with 26 shots and a 1-to-12 frame spread, a
    # hash assignment routinely lands 40% of the frames in one split.
    order = sorted(
        by_shot.items(),
        key=lambda kv: (-len(kv[1]), hashlib.sha256(f"{seed}/{kv[0]}".encode()).digest()),
    )
    for shot_index, frames in order:
        name = max(targets, key=lambda k: targets[k] - len(assigned[k]))
        assigned[name].extend(frames)
        shot_assignment[shot_index] = name

    split = LocalSplit(
        ratios=ratios, seed=seed, shot_assignment=shot_assignment, frames=assigned
    )
    log.info(
        "local split by shot: %s (fingerprint %s)",
        {k: len(v) for k, v in assigned.items()}, split.fingerprint(),
    )
    return split


def assert_no_leakage(
    split: LocalSplit, samples: dict[str, FrameSample]
) -> dict:
    """No shot, window or adjacent frame pair may cross a split boundary."""
    names = sorted(split.frames)

    # 1. shots
    shots_per_split = {
        name: {samples[f].shot_index for f in split.frames[name]} for name in names
    }
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = shots_per_split[a] & shots_per_split[b]
            if shared:
                raise AssertionError(
                    f"shots {sorted(shared)} appear in both {a} and {b}"
                )

    # 2. temporal windows
    windows_per_split = {
        name: {
            samples[f].window_id for f in split.frames[name] if samples[f].window_id
        }
        for name in names
    }
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = windows_per_split[a] & windows_per_split[b]
            if shared:
                raise AssertionError(
                    f"temporal windows {sorted(shared)} cross {a}/{b}"
                )

    # 3. adjacency: frames closer than this in the source video are visually
    #    related even when the shot classifier disagrees.
    adjacency_gap = 50
    placed = [
        (samples[f].frame_idx, split.split_of_frame(f))
        for name in names for f in split.frames[name]
    ]
    placed.sort()
    violations = [
        (a_idx, b_idx, a_split, b_split)
        for (a_idx, a_split), (b_idx, b_split) in zip(placed, placed[1:], strict=False)
        if b_idx - a_idx < adjacency_gap and a_split != b_split
    ]

    return {
        "shot_disjoint": True,
        "window_disjoint": True,
        "adjacency_gap_frames": adjacency_gap,
        "n_close_pairs_across_splits": len(violations),
        "close_pairs": [
            {"a": a, "b": b, "gap": b - a, "splits": [sa, sb]}
            for a, b, sa, sb in violations[:10]
        ],
        "note": (
            "close pairs across splits are reported, not fatal: two frames 40 "
            "frames apart in different shots are a scene cut, not a duplicate"
        ),
    }


def split_summary(
    split: LocalSplit,
    samples: dict[str, FrameSample],
    annotations: dict[str, BallAnnotation],
) -> dict:
    out: dict = {}
    for name, frame_ids in sorted(split.frames.items()):
        visible = [
            f for f in frame_ids
            if annotations[f].visibility is BallVisibility.VISIBLE
        ]
        negatives = [
            f for f in frame_ids
            if annotations[f].visibility in (
                BallVisibility.NOT_VISIBLE, BallVisibility.OUTSIDE_FRAME
            ) or annotations[f].ignore_reason.excludes_from_scoring
        ]
        categories: dict[str, int] = {}
        for f in frame_ids:
            key = samples[f].sampling_category.value
            categories[key] = categories.get(key, 0) + 1
        out[name] = {
            "n_frames": len(frame_ids),
            "n_shots": len({samples[f].shot_index for f in frame_ids}),
            "n_positive": len(visible),
            "n_negative": len(negatives),
            "categories": dict(sorted(categories.items())),
        }
    return out


def load_local(package_root: str | Path):
    store = AnnotationStore(package_root)
    return store.load_samples(), store.load_annotations()
