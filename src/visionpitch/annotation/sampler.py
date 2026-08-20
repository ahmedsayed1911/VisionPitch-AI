"""Deterministic stratified frame sampling for annotation.

Broadcast ball annotation workflow, step 2.

Sampling uniformly from a highlights package would spend most of the annotation
budget on replays and close-ups. Sampling only the easiest live wide shots would
produce a dataset the detector already passes. This module allocates a fixed
budget across strata chosen so the result measures the things that are actually
uncertain.

Which categories can be assigned before annotation
--------------------------------------------------
Some can and some cannot, and pretending otherwise would put fiction in the
metadata. Frame-level properties -- camera motion, blur, player density, shot
type, pitch region -- are measurable now. Ball-relative properties -- "ball near
a line", "ball against a player's body" -- are **not**, because nobody knows
where the ball is until a human says so. Those are derived after annotation, and
this module does not guess at them.

Model disagreement as a sampling signal
---------------------------------------
A share of the budget goes to frames where the two detectors disagree. That is
legitimate -- predictions steer *where to look*, humans decide *what is true* --
but it biases the sample toward model-hard cases, so those frames are marked and
the stratified base is drawn independently of any model.

Determinism
-----------
Given the same audit and the same seed, this returns the same frames in the same
order. The selection is recorded with a fingerprint so a rebuilt package can be
proven identical.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

import numpy as np

from visionpitch.annotation.broadcast_audit import AuditResult, Shot, ShotType
from visionpitch.annotation.schema import FrameSample, SamplingCategory
from visionpitch.common.logging import get_logger

log = get_logger("annotation.sampler")

SAMPLER_SCHEMA_VERSION = "1.0.0"


@dataclass
class SamplingPlan:
    """Budget allocation. Shares are of the total target, and must sum to 1."""

    total_frames: int = 400
    #: independent of any model: the backbone of the dataset
    stratified_live_share: float = 0.50
    #: short consecutive runs, so temporal consistency can be measured
    temporal_window_share: float = 0.16
    #: frames where the two detectors disagree -- model-steered, marked as such
    disagreement_share: float = 0.16
    #: close-ups, crowd and graphics, where a detector must find nothing
    negative_share: float = 0.10
    #: goal-mouth and penalty-area play
    goal_area_share: float = 0.08

    #: consecutive frames per temporal window
    window_length: int = 7
    #: frames closer than this to an already-picked frame are skipped, so the
    #: stratified sample cannot collapse onto one passage of play
    min_separation_frames: int = 20
    seed: int = 20260803

    def counts(self) -> dict[str, int]:
        shares = {
            "stratified_live": self.stratified_live_share,
            "temporal_window": self.temporal_window_share,
            "disagreement": self.disagreement_share,
            "negative": self.negative_share,
            "goal_area": self.goal_area_share,
        }
        total = sum(shares.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"sampling shares must sum to 1.0, got {total}")
        return {k: int(round(v * self.total_frames)) for k, v in shares.items()}

    def to_dict(self) -> dict:
        return {
            "schema_version": SAMPLER_SCHEMA_VERSION,
            "total_frames": self.total_frames,
            "shares": {
                "stratified_live": self.stratified_live_share,
                "temporal_window": self.temporal_window_share,
                "disagreement": self.disagreement_share,
                "negative": self.negative_share,
                "goal_area": self.goal_area_share,
            },
            "counts": self.counts(),
            "window_length": self.window_length,
            "min_separation_frames": self.min_separation_frames,
            "seed": self.seed,
        }


@dataclass
class FrameSignal:
    """Per-frame measurements the sampler strata are built from."""

    frame_idx: int
    motion_px: float
    blur_variance: float
    edge_density: float
    saturation: float
    shot_index: int
    shot_type: ShotType
    likely_slow_motion: bool
    #: filled when predictions are available; None before that
    n_people: int | None = None
    disagreement_px: float | None = None


@dataclass
class SamplingResult:
    samples: list[FrameSample] = field(default_factory=list)
    plan: SamplingPlan = field(default_factory=SamplingPlan)
    strata_counts: dict[str, int] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "plan": self.plan.to_dict(),
                "frames": [
                    [s.frame_idx, s.sampling_category.value] for s in self.samples
                ],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        by_category: dict[str, int] = {}
        for sample in self.samples:
            key = sample.sampling_category.value
            by_category[key] = by_category.get(key, 0) + 1
        return {
            "schema_version": SAMPLER_SCHEMA_VERSION,
            "n_samples": len(self.samples),
            "plan": self.plan.to_dict(),
            "strata_counts": self.strata_counts,
            "by_category": dict(sorted(by_category.items())),
            "sampling_fingerprint": self.fingerprint(),
        }


def _category_for(
    signal: FrameSignal, thresholds: dict[str, float]
) -> tuple[SamplingCategory, str]:
    """The most specific frame-level category the evidence supports.

    Order matters: a blurred frame during a fast pan is filed as motion blur,
    because that is the harder property for a detector and the one worth being
    able to slice results by.
    """
    if signal.shot_type is ShotType.GRAPHIC:
        return SamplingCategory.BROADCAST_GRAPHIC, "full-screen graphic"
    if signal.shot_type is ShotType.CLOSE_UP:
        return SamplingCategory.CLOSE_UP_NEGATIVE, "close-up; ball usually absent"
    if signal.shot_type is ShotType.CROWD_OR_BENCH:
        return SamplingCategory.CROWD_NEGATIVE, "crowd or bench; no pitch"
    if signal.blur_variance <= thresholds["blur_low"]:
        return SamplingCategory.MOTION_BLUR, (
            f"blur variance {signal.blur_variance:.0f} in lowest decile"
        )
    if signal.motion_px >= thresholds["motion_high"]:
        return SamplingCategory.CAMERA_PAN, (
            f"camera motion {signal.motion_px:.1f} px/frame in top decile"
        )
    if signal.edge_density <= thresholds["edge_low"]:
        return SamplingCategory.LOW_CONTRAST, (
            f"edge density {signal.edge_density:.3f} in lowest decile"
        )
    if signal.n_people is not None and signal.n_people >= thresholds["people_high"]:
        return SamplingCategory.CROWDED_SCENE, f"{signal.n_people} people detected"
    if signal.motion_px >= thresholds["motion_mid"]:
        return SamplingCategory.FAST_TRANSITION, (
            f"camera motion {signal.motion_px:.1f} px/frame above median"
        )
    return SamplingCategory.MIDFIELD_PLAY, "steady live play"


def _thresholds(signals: list[FrameSignal]) -> dict[str, float]:
    """Strata cut points from this video's own distributions, not constants."""
    live = [s for s in signals if s.shot_type.is_live_play_candidate] or signals
    blur = np.array([s.blur_variance for s in live])
    motion = np.array([s.motion_px for s in live])
    edges = np.array([s.edge_density for s in live])
    people = np.array([s.n_people for s in live if s.n_people is not None])
    return {
        "blur_low": float(np.percentile(blur, 10)),
        "motion_high": float(np.percentile(motion, 90)),
        "motion_mid": float(np.percentile(motion, 60)),
        "edge_low": float(np.percentile(edges, 10)),
        "people_high": float(np.percentile(people, 85)) if people.size else 1e9,
    }


def _pick_spread(
    candidates: list[FrameSignal], count: int, minimum_gap: int, taken: set[int],
    rng: random.Random,
) -> list[FrameSignal]:
    """Choose ``count`` frames spread across the video, never adjacent.

    Shuffled then filtered by separation rather than evenly spaced, because even
    spacing on an edited highlights package lands repeatedly on the same
    positions within each shot.
    """
    pool = sorted(candidates, key=lambda s: s.frame_idx)
    rng.shuffle(pool)
    chosen: list[FrameSignal] = []
    for signal in pool:
        if len(chosen) >= count:
            break
        if any(abs(signal.frame_idx - t) < minimum_gap for t in taken):
            continue
        chosen.append(signal)
        taken.add(signal.frame_idx)
    return sorted(chosen, key=lambda s: s.frame_idx)


def build_samples(
    audit: AuditResult,
    signals: list[FrameSignal],
    plan: SamplingPlan | None = None,
    disagreements: dict[int, float] | None = None,
    goal_area_frames: set[int] | None = None,
) -> SamplingResult:
    """Select frames for review. Deterministic given the same inputs and seed."""
    plan = plan or SamplingPlan()
    rng = random.Random(plan.seed)
    counts = plan.counts()
    thresholds = _thresholds(signals)
    by_index = {s.frame_idx: s for s in signals}
    shots = {s.index: s for s in audit.shots}

    taken: set[int] = set()
    picked: list[tuple[FrameSignal, SamplingCategory, str, str | None]] = []
    strata_counts: dict[str, int] = {}

    def record(signal: FrameSignal, category: SamplingCategory, reason: str,
               window_id: str | None = None) -> None:
        picked.append((signal, category, reason, window_id))

    live = [
        s for s in signals
        if s.shot_type.is_live_play_candidate and not s.likely_slow_motion
    ]

    # -- 1. temporal windows: consecutive runs inside live shots --------------- #
    window_target = counts["temporal_window"]
    n_windows = max(1, window_target // plan.window_length)
    eligible = [
        shot for shot in audit.shots
        if shot.shot_type.is_live_play_candidate
        and shot.n_frames >= plan.window_length * 3
        and not shot.likely_slow_motion
    ]
    rng.shuffle(eligible)
    windows_made = 0
    for shot in eligible:
        if windows_made >= n_windows:
            break
        start = rng.randint(
            shot.start_frame + plan.window_length,
            max(shot.start_frame + plan.window_length, shot.end_frame - plan.window_length),
        )
        run = [by_index[i] for i in range(start, start + plan.window_length) if i in by_index]
        if len(run) < plan.window_length:
            continue
        window_id = f"w{windows_made:02d}_shot{shot.index}"
        for signal in run:
            taken.add(signal.frame_idx)
            record(
                signal, SamplingCategory.TEMPORAL_WINDOW,
                f"consecutive run of {plan.window_length} for temporal consistency",
                window_id,
            )
        windows_made += 1
    strata_counts["temporal_window"] = sum(
        1 for p in picked if p[1] is SamplingCategory.TEMPORAL_WINDOW
    )

    # -- 2. goal-area play ------------------------------------------------------ #
    if goal_area_frames:
        pool = [s for s in live if s.frame_idx in goal_area_frames]
        for signal in _pick_spread(
            pool, counts["goal_area"], plan.min_separation_frames, taken, rng
        ):
            record(signal, SamplingCategory.NEAR_GOAL, "goal or penalty area in shot")
    strata_counts["goal_area"] = sum(
        1 for p in picked if p[1] is SamplingCategory.NEAR_GOAL
    )

    # -- 3. detector disagreement (model-steered, marked) ---------------------- #
    if disagreements:
        ranked = sorted(
            (
                by_index[i] for i in disagreements
                if i in by_index and by_index[i].shot_type.is_live_play_candidate
            ),
            key=lambda s: -disagreements[s.frame_idx],
        )
        added = 0
        for signal in ranked:
            if added >= counts["disagreement"]:
                break
            if any(abs(signal.frame_idx - t) < plan.min_separation_frames for t in taken):
                continue
            taken.add(signal.frame_idx)
            category, _ = _category_for(signal, thresholds)
            record(
                signal, category,
                f"detectors disagree by {disagreements[signal.frame_idx]:.0f} px "
                f"(model-steered selection)",
            )
            added += 1
        strata_counts["disagreement"] = added

    # -- 4. negatives: close-ups, crowd, graphics ------------------------------ #
    negatives = [s for s in signals if not s.shot_type.is_live_play_candidate]
    for signal in _pick_spread(
        negatives, counts["negative"], plan.min_separation_frames, taken, rng
    ):
        category, reason = _category_for(signal, thresholds)
        record(signal, category, reason)
    strata_counts["negative"] = sum(
        1 for p in picked
        if p[1] in (
            SamplingCategory.CLOSE_UP_NEGATIVE, SamplingCategory.CROWD_NEGATIVE,
            SamplingCategory.BROADCAST_GRAPHIC,
        )
    )

    # -- 5. stratified live play, independent of any model --------------------- #
    remaining = plan.total_frames - len(picked)
    buckets: dict[SamplingCategory, list[FrameSignal]] = {}
    for signal in live:
        if signal.frame_idx in taken:
            continue
        category, _ = _category_for(signal, thresholds)
        buckets.setdefault(category, []).append(signal)

    if buckets:
        per_bucket = max(1, remaining // len(buckets))
        for category, pool in sorted(buckets.items(), key=lambda kv: kv[0].value):
            for signal in _pick_spread(
                pool, per_bucket, plan.min_separation_frames, taken, rng
            ):
                _, reason = _category_for(signal, thresholds)
                record(signal, category, reason)

        # Top up from whatever is left if rounding left the budget short.
        shortfall = plan.total_frames - len(picked)
        if shortfall > 0:
            leftovers = [s for s in live if s.frame_idx not in taken]
            for signal in _pick_spread(
                leftovers, shortfall, plan.min_separation_frames, taken, rng
            ):
                category, reason = _category_for(signal, thresholds)
                record(signal, category, reason)
    strata_counts["stratified_live"] = (
        len(picked) - sum(v for k, v in strata_counts.items())
    )

    picked.sort(key=lambda item: item[0].frame_idx)
    samples: list[FrameSample] = []
    for signal, category, reason, window_id in picked:
        shot = shots.get(signal.shot_index, Shot(0, 0, 0, audit.fps))
        samples.append(
            FrameSample(
                frame_id=f"f{signal.frame_idx:06d}",
                frame_idx=signal.frame_idx,
                timestamp_s=round(signal.frame_idx / audit.fps, 4) if audit.fps else 0.0,
                image_path="",
                shot_index=signal.shot_index,
                shot_type=signal.shot_type.value,
                sampling_category=category,
                sampling_reason=reason,
                is_live_play_candidate=signal.shot_type.is_live_play_candidate,
                likely_slow_motion=signal.likely_slow_motion,
                window_id=window_id,
                source_content_hash=audit.content_hash,
                width=audit.width,
                height=audit.height,
            )
        )
        _ = shot

    result = SamplingResult(samples=samples, plan=plan, strata_counts=strata_counts)
    log.info(
        "selected %d frame(s) across %d category/ies; fingerprint %s",
        len(samples), len({s.sampling_category for s in samples}), result.fingerprint(),
    )
    return result
