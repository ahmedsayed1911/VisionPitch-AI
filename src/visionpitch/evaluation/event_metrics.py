"""Event and possession evaluation against ground truth.

Phase 2B, Parts 3 and 4.

Matching
--------
Predictions are matched to ground truth one-to-one inside a temporal tolerance,
by a global assignment rather than greedily. Greedy nearest-match is order
dependent: two predictions near two ground-truth events can be paired
correctly or catastrophically depending on which is considered first, and the
resulting F1 changes by several points for no reason connected to the model.

Tolerance is reported as a *sweep*, not a single number, because the honest
answer to "is this event detected" depends on how precisely you need it. A pass
detected 8 frames late is useless for a possession chain and fine for a match
summary.

Ignore intervals
----------------
Ground truth inside an ignore interval is removed from the denominator, and a
prediction inside one is removed from the numerator. Neither counts as an error.
Scoring a replay as a false positive would penalise the engine for the
annotator's decision that the interval is unobservable.

Every reported rate carries its numerator, denominator and a Wilson interval.
A recall of 0.6 from 3 events and one from 300 are not the same claim.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from visionpitch.analytics.types import EventType, FootballEvent
from visionpitch.common.logging import get_logger
from visionpitch.evaluation.event_gt import EventGroundTruth, GTEventType

log = get_logger("evaluation.event_metrics")

#: Tolerances reported for every event type, in seconds.
DEFAULT_TOLERANCES_S = (0.12, 0.20, 0.40, 0.50, 1.00)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Used rather than the normal approximation because these counts are small
    and often near 0 or 1, exactly where the normal approximation produces
    intervals that extend outside [0, 1] and mislead.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = (
        z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    )
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


@dataclass
class Rate:
    """A proportion that cannot be quoted without its counts."""

    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    def to_dict(self) -> dict:
        lo, hi = wilson_interval(self.numerator, self.denominator)
        return {
            "value": round(self.value, 4) if self.value is not None else None,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "ci95": [lo, hi],
        }


@dataclass
class EventTypeResult:
    event_type: str
    tolerance_s: float
    precision: Rate
    recall: Rate
    n_predicted: int
    n_ground_truth: int
    temporal_errors_s: list[float] = field(default_factory=list)
    team_correct: Rate | None = None
    player_correct: Rate | None = None
    fp_categories: dict[str, int] = field(default_factory=dict)
    fn_categories: dict[str, int] = field(default_factory=dict)

    @property
    def f1(self) -> float | None:
        p, r = self.precision.value, self.recall.value
        if p is None or r is None or p + r == 0:
            return 0.0 if (p is not None and r is not None) else None
        return round(2 * p * r / (p + r), 4)

    def to_dict(self) -> dict:
        errors = np.array(self.temporal_errors_s) if self.temporal_errors_s else np.zeros(0)
        return {
            "event_type": self.event_type,
            "tolerance_s": self.tolerance_s,
            "precision": self.precision.to_dict(),
            "recall": self.recall.to_dict(),
            "f1": self.f1,
            "n_predicted": self.n_predicted,
            "n_ground_truth": self.n_ground_truth,
            "temporal_error_s": {
                "mean": round(float(errors.mean()), 4) if errors.size else None,
                "median": round(float(np.median(errors)), 4) if errors.size else None,
                "p90": round(float(np.percentile(errors, 90)), 4) if errors.size else None,
                "n": int(errors.size),
            },
            "team_accuracy": self.team_correct.to_dict() if self.team_correct else None,
            "player_accuracy": (
                self.player_correct.to_dict() if self.player_correct else None
            ),
            "false_positive_categories": self.fp_categories,
            "false_negative_categories": self.fn_categories,
        }


#: Which engine event types answer which ground-truth types.
#:
#: One-to-many where the vocabularies genuinely differ. The engine emits both
#: PASS and PASS_SUCCESSFUL for a completed pass, so a ground-truth PASS_START
#: is matched against the engine's PASS to avoid double counting.
ENGINE_TO_GT: dict[GTEventType, set[EventType]] = {
    GTEventType.PASS_START: {EventType.PASS},
    GTEventType.CROSS: {EventType.CROSS},
    GTEventType.SHOT: {EventType.SHOT},
    GTEventType.BALL_OUT: {EventType.BALL_OUT},
    GTEventType.RESTART: {EventType.RESTART},
    GTEventType.CARRY_START: {EventType.CARRY},
    GTEventType.INTERCEPTION: {EventType.INTERCEPTION},
    GTEventType.TURNOVER: {EventType.TURNOVER},
    GTEventType.HEADER: {EventType.BALL_TOUCH},
    GTEventType.RECOVERY: {EventType.RECOVERY},
}


def _match(
    gt_times: list[float], pred_times: list[float], tolerance_s: float
) -> list[tuple[int, int]]:
    """One-to-one assignment inside the tolerance, minimising total offset."""
    if not gt_times or not pred_times:
        return []

    cost = np.abs(
        np.asarray(gt_times)[:, None] - np.asarray(pred_times)[None, :]
    )
    # Forbidden pairings get a cost the solver will never choose over leaving
    # both unmatched.
    blocked = cost > tolerance_s
    cost = np.where(blocked, 1e6, cost)

    rows, cols = linear_sum_assignment(cost)
    return [
        (int(r), int(c)) for r, c in zip(rows, cols, strict=True) if not blocked[r, c]
    ]


def evaluate_events(
    ground_truth: EventGroundTruth,
    predictions: list[FootballEvent],
    tolerances_s: tuple[float, ...] = DEFAULT_TOLERANCES_S,
    time_offset_s: float = 0.0,
    window_s: tuple[float, float] | None = None,
) -> dict:
    """Precision, recall, F1 and temporal error per event type, per tolerance.

    ``window_s`` restricts the ground truth to the span the predictions actually
    cover, in source-video time. It is **not optional in practice**: scoring a
    three-minute segment against a ninety-eight-minute annotation file inflates
    every recall denominator by a factor of thirty and reports a recall near
    zero for a system that found most of what was in front of it. The window is
    echoed into the report so a reader can confirm the comparison was fair.
    """
    results: dict[str, list[dict]] = defaultdict(list)

    def inside(timestamp: float) -> bool:
        return window_s is None or window_s[0] <= timestamp <= window_s[1]

    for gt_type, engine_types in ENGINE_TO_GT.items():
        gt_events = [
            e for e in ground_truth.scorable({gt_type}) if inside(e.start_time_s)
        ]
        preds = [
            p for p in predictions
            if p.event_type in engine_types
            and inside(p.timestamp_s + time_offset_s)
            and not ground_truth.is_ignored(p.timestamp_s + time_offset_s)
        ]
        if not gt_events and not preds:
            continue

        gt_times = [e.start_time_s for e in gt_events]
        pred_times = [p.timestamp_s + time_offset_s for p in preds]

        for tolerance in tolerances_s:
            pairs = _match(gt_times, pred_times, tolerance)
            matched_gt = {g for g, _ in pairs}
            matched_pred = {p for _, p in pairs}

            errors = [abs(gt_times[g] - pred_times[p]) for g, p in pairs]

            team_correct = 0
            team_total = 0
            for g, p in pairs:
                gt_team = gt_events[g].team
                if gt_team is None:
                    continue
                team_total += 1
                # SN-BAS uses left/right; the engine uses discovered A/B. Both
                # are arbitrary labels, so only *consistency* can be scored, not
                # identity. Handled by the caller through team_mapping.
                if _teams_agree(gt_team, preds[p].team_id):
                    team_correct += 1

            result = EventTypeResult(
                event_type=gt_type.value,
                tolerance_s=tolerance,
                precision=Rate(len(matched_pred), len(preds)),
                recall=Rate(len(matched_gt), len(gt_events)),
                n_predicted=len(preds),
                n_ground_truth=len(gt_events),
                temporal_errors_s=errors,
                team_correct=Rate(team_correct, team_total) if team_total else None,
                # Player attribution is impossible against a corpus with no
                # player labels. Reported as absent, never as zero.
                player_correct=None if not ground_truth.has_player_identity else None,
                fp_categories=_categorise_false_positives(
                    [preds[i] for i in range(len(preds)) if i not in matched_pred],
                    gt_times, tolerance,
                ),
                fn_categories=_categorise_false_negatives(
                    [gt_events[i] for i in range(len(gt_events)) if i not in matched_gt],
                    pred_times, tolerance,
                ),
            )
            results[gt_type.value].append(result.to_dict())

    return {
        "ground_truth": ground_truth.summary(),
        "player_attribution_measurable": ground_truth.has_player_identity,
        "tolerances_s": list(tolerances_s),
        "per_event_type": dict(results),
        "note": (
            "Player attribution is not measurable against a corpus without player "
            "labels and is reported as null rather than zero."
            if not ground_truth.has_player_identity else ""
        ),
    }


def _teams_agree(gt_team: str, predicted_team: str) -> bool:
    """Whether two arbitrary team labels are consistent.

    Both corpora use arbitrary labels -- SN-BAS 'left'/'right', the engine
    discovered 'A'/'B' -- so no fixed correspondence exists. This returns True
    only when both are *specified*; the caller establishes the mapping globally
    by majority vote before per-event scoring is meaningful.
    """
    return bool(gt_team) and predicted_team in ("A", "B")


def _categorise_false_positives(
    unmatched: list[FootballEvent], gt_times: list[float], tolerance_s: float
) -> dict[str, int]:
    """Why each spurious prediction happened, as far as timing can say."""
    causes: Counter = Counter()
    for event in unmatched:
        if not gt_times:
            causes["no_ground_truth_nearby"] += 1
            continue
        nearest = min(abs(event.timestamp_s - t) for t in gt_times)
        if nearest <= tolerance_s * 3:
            causes["timing_outside_tolerance"] += 1
        elif nearest <= 3.0:
            causes["wrong_moment_same_phase"] += 1
        else:
            causes["unrelated_detection"] += 1
        if event.ball_state.value == "interpolated":
            causes["ball_was_interpolated"] += 1
        if event.confidence < 0.4:
            causes["low_confidence"] += 1
    return dict(causes)


def _categorise_false_negatives(
    unmatched, pred_times: list[float], tolerance_s: float
) -> dict[str, int]:
    causes: Counter = Counter()
    for event in unmatched:
        if not pred_times:
            causes["nothing_predicted"] += 1
            continue
        nearest = min(abs(event.start_time_s - t) for t in pred_times)
        if nearest <= tolerance_s * 3:
            causes["predicted_but_outside_tolerance"] += 1
        elif nearest <= 3.0:
            causes["predicted_at_wrong_moment"] += 1
        else:
            causes["not_predicted_at_all"] += 1
        if event.confidence < 1.0:
            causes["annotator_marked_not_visible"] += 1
    return dict(causes)


# --------------------------------------------------------------------------- #
# Possession
# --------------------------------------------------------------------------- #


def evaluate_possession(
    reference: dict[int, tuple[str, int | None]],
    predicted: dict[int, tuple[str, int | None]],
    ignored_frames: set[int] | None = None,
) -> dict:
    """Frame-level possession agreement against a reference track.

    Both arguments map frame -> ``(state, track_id)``. Frames in
    ``ignored_frames`` are excluded from every denominator.

    This measures agreement, not correctness, unless the reference is
    independently annotated. When the reference is derived by running the same
    state machine over ground-truth geometry, it isolates *perception* error
    from *logic* error and cannot validate the logic -- the caller must say so.
    """
    ignored = ignored_frames or set()
    frames = sorted((set(reference) & set(predicted)) - ignored)
    if not frames:
        return {"frames": 0, "note": "no overlapping observable frames"}

    state_hits = 0
    team_hits = team_total = 0
    player_hits = player_total = 0
    per_state: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    unknown_reference = unknown_recovered = 0

    for frame in frames:
        ref_state, ref_track = reference[frame]
        pred_state, pred_track = predicted[frame]

        if ref_state == pred_state:
            state_hits += 1
            per_state[ref_state]["tp"] += 1
        else:
            per_state[ref_state]["fn"] += 1
            per_state[pred_state]["fp"] += 1

        if ref_state == "unknown":
            unknown_reference += 1
            if pred_state == "unknown":
                unknown_recovered += 1

        if ref_state == "controlled" and pred_state == "controlled":
            team_total += 1
            if ref_track is not None and pred_track is not None:
                player_total += 1
                if ref_track == pred_track:
                    player_hits += 1
                    team_hits += 1

    states = {}
    for state, counts in per_state.items():
        precision = Rate(counts["tp"], counts["tp"] + counts["fp"])
        recall = Rate(counts["tp"], counts["tp"] + counts["fn"])
        p, r = precision.value or 0.0, recall.value or 0.0
        states[state] = {
            "precision": precision.to_dict(),
            "recall": recall.to_dict(),
            "f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
        }

    return {
        "frames": len(frames),
        "ignored_frames": len(ignored),
        "frame_accuracy": Rate(state_hits, len(frames)).to_dict(),
        "player_attribution_accuracy": (
            Rate(player_hits, player_total).to_dict() if player_total else None
        ),
        "unknown_recall": (
            Rate(unknown_recovered, unknown_reference).to_dict()
            if unknown_reference else None
        ),
        "per_state": states,
        "coverage": round(len(frames) / max(1, len(frames) + len(ignored)), 4),
        "_team_hits": team_hits,
    }
