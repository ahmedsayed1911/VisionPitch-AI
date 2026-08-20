"""Focused QC queue for the remaining broadcast-ball annotations.

Built after the 115-frame coverage audit, which found every reviewed frame
labelled ``visible`` and several categories far below the floor at which a
per-category rate means anything.

Two kinds of target, and they are not the same thing
----------------------------------------------------
**Category quotas** count *frames reviewed* in a sampling category — motion
blur, low contrast, and so on. The sampler assigned those before review, so the
queue can guarantee how many get seen.

**The negative quota counts an outcome**, not a category: a frame is a genuine
negative only if the reviewer marks it ``not_visible``, ``outside_frame``,
``ignore_replay`` or ``ignore_non_live``. No queue can promise those, because the
human decides. The audit proved why this distinction matters — all 19 reviewed
frames from the "crowd negative" stratum turned out to be real play with a
visible ball, because the shot classifier mistook sparse pitch markings for
absence of a pitch. **The sampler's guess is not a label.**

So the queue front-loads the strata where negatives are *plausible* and reports
the yield as it is measured, rather than assuming it.

Window economics
----------------
Completing a temporal window needs all 7 consecutive frames. Windows already
part-reviewed are far cheaper to finish than fresh ones, so they are ordered by
how few frames remain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from visionpitch.annotation.schema import (
    AnnotationStore,
    BallAnnotation,
    BallVisibility,
    FrameSample,
)
from visionpitch.common.logging import get_logger

log = get_logger("annotation.qc")

QC_SCHEMA_VERSION = "1.0.0"

#: Review order. Negatives first because they are the binding gap and the
#: scarcest resource; temporal windows last because they are cheap to finish
#: and least urgent.
PRIORITY: tuple[str, ...] = (
    "crowd_negative",
    "broadcast_graphic",
    "motion_blur",
    "low_contrast",
    "fast_transition",
    "near_goal",
    "camera_pan",
    "temporal_window",
)

#: Frames needed per category before a per-category test rate has a 95%
#: interval narrower than roughly +-0.20.
CATEGORY_TARGET = 25
#: Genuine negatives needed to measure a hallucination rate at all.
NEGATIVE_TARGET = 25
#: Complete 7-frame runs needed to measure temporal consistency.
WINDOW_TARGET = 3
WINDOW_LENGTH = 7

#: Secondary source of negatives. A ball lost in a ruck of players is the most
#: common genuine ``not_visible``, and crowded scenes are where that happens.
#: Included in the queue for that reason, not to hit a category quota.
NEGATIVE_FALLBACK_CATEGORY = "crowded_scene"
NEGATIVE_FALLBACK_FRAMES = 10


def is_genuine_negative(annotation: BallAnnotation) -> bool:
    """Whether a reviewed frame counts toward the negative quota.

    Determined entirely by what the human recorded. A frame the sampler guessed
    was crowd or graphics but which the reviewer marked ``visible`` is not a
    negative, and must not be counted as one.
    """
    if annotation.ignore_reason.excludes_from_scoring:
        return True
    return annotation.visibility in (
        BallVisibility.NOT_VISIBLE, BallVisibility.OUTSIDE_FRAME
    )


@dataclass
class Quota:
    name: str
    target: int
    achieved: int = 0
    queued: int = 0
    available: int = 0

    @property
    def deficit(self) -> int:
        return max(0, self.target - self.achieved)

    @property
    def reachable(self) -> bool:
        return self.achieved + self.available >= self.target

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target": self.target,
            "achieved": self.achieved,
            "deficit": self.deficit,
            "queued": self.queued,
            "available_unreviewed": self.available,
            "reachable_from_this_package": self.reachable,
        }


@dataclass
class WindowState:
    window_id: str
    reviewed: int
    total: int

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.reviewed)

    @property
    def complete(self) -> bool:
        return self.reviewed >= min(self.total, WINDOW_LENGTH)


@dataclass
class QCQueue:
    frame_ids: list[str] = field(default_factory=list)
    quotas: dict[str, Quota] = field(default_factory=dict)
    negative_quota: Quota | None = None
    window_target: int = WINDOW_TARGET
    windows: list[WindowState] = field(default_factory=list)
    schema_version: str = QC_SCHEMA_VERSION

    @property
    def n_complete_windows(self) -> int:
        return sum(1 for w in self.windows if w.complete)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "n_queued": len(self.frame_ids),
            "priority_order": list(PRIORITY),
            "category_quotas": {
                name: q.to_dict() for name, q in self.quotas.items()
            },
            "negative_quota": (
                self.negative_quota.to_dict() if self.negative_quota else None
            ),
            "negative_definition": (
                "not_visible | outside_frame | ignore_replay | ignore_non_live -- "
                "decided by the reviewer, never inferred from the sampling category"
            ),
            "temporal_windows": {
                "target_complete": self.window_target,
                "complete_now": self.n_complete_windows,
                "states": [
                    {
                        "window_id": w.window_id, "reviewed": w.reviewed,
                        "total": w.total, "remaining": w.remaining,
                        "complete": w.complete,
                    }
                    for w in self.windows
                ],
            },
        }


def build_qc_queue(
    samples: dict[str, FrameSample],
    annotations: dict[str, BallAnnotation],
    target_total: int = 125,
) -> QCQueue:
    """Order the unreviewed frames so the scarcest quota is served first."""
    unreviewed = [f for f in samples if f not in annotations]
    by_category: dict[str, list[str]] = {}
    for frame_id in unreviewed:
        by_category.setdefault(
            samples[frame_id].sampling_category.value, []
        ).append(frame_id)
    for frames in by_category.values():
        frames.sort(key=lambda f: samples[f].frame_idx)

    reviewed_by_category: dict[str, int] = {}
    for frame_id in annotations:
        if frame_id in samples:
            key = samples[frame_id].sampling_category.value
            reviewed_by_category[key] = reviewed_by_category.get(key, 0) + 1

    queue: list[str] = []
    taken: set[str] = set()
    quotas: dict[str, Quota] = {}

    def take(frame_ids: list[str], count: int) -> int:
        added = 0
        for frame_id in frame_ids:
            if added >= count:
                break
            if frame_id in taken:
                continue
            taken.add(frame_id)
            queue.append(frame_id)
            added += 1
        return added

    # -- 1. negative strata: take everything, they are the binding gap -------- #
    negatives_available = 0
    for category in ("crowd_negative", "broadcast_graphic"):
        pool = by_category.get(category, [])
        negatives_available += len(pool)
        taken_now = take(pool, len(pool))
        quotas[category] = Quota(
            name=category,
            target=len(pool),  # the whole remaining stratum
            achieved=reviewed_by_category.get(category, 0),
            queued=taken_now,
            available=len(pool),
        )

    # -- 2. secondary negative source ----------------------------------------- #
    fallback = by_category.get(NEGATIVE_FALLBACK_CATEGORY, [])
    fallback_taken = take(fallback, NEGATIVE_FALLBACK_FRAMES)
    negatives_available += fallback_taken
    quotas[NEGATIVE_FALLBACK_CATEGORY] = Quota(
        name=NEGATIVE_FALLBACK_CATEGORY,
        target=CATEGORY_TARGET,
        achieved=reviewed_by_category.get(NEGATIVE_FALLBACK_CATEGORY, 0),
        queued=fallback_taken,
        available=len(fallback),
    )

    # -- 3. under-covered categories, in the declared priority order ---------- #
    for category in ("motion_blur", "low_contrast", "fast_transition",
                     "near_goal", "camera_pan"):
        pool = by_category.get(category, [])
        achieved = reviewed_by_category.get(category, 0)
        quota = Quota(
            name=category, target=CATEGORY_TARGET, achieved=achieved,
            available=len(pool),
        )
        quota.queued = take(pool, quota.deficit)
        quotas[category] = quota

    # -- 4. temporal windows, cheapest to finish first ------------------------ #
    window_frames: dict[str, list[str]] = {}
    for frame_id, sample in samples.items():
        if sample.window_id:
            window_frames.setdefault(sample.window_id, []).append(frame_id)

    states: list[WindowState] = []
    for window_id, frames in window_frames.items():
        reviewed = sum(1 for f in frames if f in annotations)
        states.append(WindowState(window_id, reviewed, len(frames)))
    # Already-complete windows first (so they are reported), then the ones
    # needing fewest frames: finishing a 4/7 window costs 3, a fresh one costs 7.
    states.sort(key=lambda w: (not w.complete, w.remaining))

    completed = sum(1 for w in states if w.complete)
    for state in states:
        if completed >= WINDOW_TARGET:
            break
        if state.complete:
            continue
        frames = sorted(window_frames[state.window_id], key=lambda f: samples[f].frame_idx)
        take([f for f in frames if f not in annotations], state.remaining)
        completed += 1

    quotas["temporal_window"] = Quota(
        name="temporal_window",
        target=CATEGORY_TARGET,
        achieved=reviewed_by_category.get("temporal_window", 0),
        queued=sum(
            1 for f in queue if samples[f].sampling_category.value == "temporal_window"
        ),
        available=len(by_category.get("temporal_window", [])),
    )

    # -- 5. top up toward the requested total, in priority order -------------- #
    if len(queue) < target_total:
        for category in PRIORITY:
            if len(queue) >= target_total:
                break
            take(by_category.get(category, []), target_total - len(queue))

    achieved_negatives = sum(
        1 for a in annotations.values() if is_genuine_negative(a)
    )
    negative_quota = Quota(
        name="genuine_negatives", target=NEGATIVE_TARGET,
        achieved=achieved_negatives,
        queued=sum(
            1 for f in queue
            if samples[f].sampling_category.value in (
                "crowd_negative", "broadcast_graphic", NEGATIVE_FALLBACK_CATEGORY
            )
        ),
        available=negatives_available,
    )

    # Queue order follows the declared priority, then frame index inside each
    # group, so a reviewer working top-down serves the scarcest quota first.
    rank = {name: i for i, name in enumerate(PRIORITY)}
    queue.sort(
        key=lambda f: (
            rank.get(samples[f].sampling_category.value, len(PRIORITY)),
            samples[f].frame_idx,
        )
    )

    result = QCQueue(
        frame_ids=queue, quotas=quotas, negative_quota=negative_quota,
        windows=states,
    )
    log.info(
        "QC queue: %d frame(s); %d complete window(s) after review; "
        "negatives achieved %d of %d",
        len(queue), min(completed, len(states)), achieved_negatives, NEGATIVE_TARGET,
    )
    return result


def progress_report(package_root: str | Path, target_total: int = 125) -> dict:
    """Current QC state: what is done, what is queued, what is still missing."""
    store = AnnotationStore(package_root)
    samples = store.load_samples()
    annotations = store.load_annotations()
    queue = build_qc_queue(samples, annotations, target_total)

    negatives = [
        f for f, a in annotations.items() if is_genuine_negative(a)
    ]
    by_kind: dict[str, int] = {}
    for frame_id in negatives:
        annotation = annotations[frame_id]
        key = (
            annotation.ignore_reason.value
            if annotation.ignore_reason.excludes_from_scoring
            else annotation.visibility.value
        )
        by_kind[key] = by_kind.get(key, 0) + 1

    return {
        **queue.to_dict(),
        "n_samples": len(samples),
        "n_reviewed": len(annotations),
        "n_unreviewed": len(samples) - len(annotations),
        "genuine_negatives_found": len(negatives),
        "genuine_negatives_by_kind": dict(sorted(by_kind.items())),
        "annotation_fingerprint": store.fingerprint(),
    }
