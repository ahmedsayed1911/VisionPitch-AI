"""Validate the possession engine against the derived GSR reference.

Phase 2C, Part 5.

The measurement is run in two configurations, and the gap between them is the
point of the exercise:

``ground_truth_boxes``
    The engine consumes annotated boxes. Detection, tracking and team
    classification are all perfect. Whatever it gets wrong here is **logic**:
    the radius rule, the hysteresis, the contest handling.

``pipeline``
    The engine consumes what the pipeline actually produced. The difference
    against the first configuration is **perception loss**.

Reporting only the second would blame the state machine for the detector's
misses; reporting only the first would claim an accuracy no user will ever see.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from visionpitch.analytics.context import AnalysisContext
from visionpitch.analytics.types import BallStateKind, PossessionState
from visionpitch.common.logging import get_logger
from visionpitch.evaluation.event_metrics import wilson_interval
from visionpitch.evaluation.possession_gt import (
    PossessionGroundTruth,
    PossessionLabel,
    load_gsr_gamestate,
)
from visionpitch.pitch.geometry import PitchConfiguration

log = get_logger("evaluation.possession_eval")


# --------------------------------------------------------------------------- #
# A context built from annotated boxes
# --------------------------------------------------------------------------- #


def context_from_gsr(
    labels_path: str | Path, pitch: PitchConfiguration | None = None
) -> AnalysisContext:
    """An ``AnalysisContext`` in which perception is perfect.

    Every field the possession engine reads is filled from the GSR annotation:
    boxes and team labels from the annotated players, ball position from the
    annotated ball. Calibration confidence is set to 1.0 because the pitch
    coordinates are annotated rather than estimated -- there is no homography to
    be unsure about.

    Ball rows are written only for frames where the annotated ball's ground
    projection is inside the pitch. An airborne ball's projection is not the
    ball (see ``possession_gt``), and feeding it in would ask the engine to
    reason about a position that does not exist.
    """
    pitch = pitch or PitchConfiguration()
    frames, fps = load_gsr_gamestate(labels_path)

    player_rows: list[dict] = []
    ball_rows: list[dict] = []
    frame_rows: list[dict] = []
    timestamps: dict[int, float] = {}
    ball_by_frame: dict[int, tuple] = {}
    track_teams: dict[int, str] = {}
    track_roles: dict[int, str] = {}

    for frame_idx in sorted(frames):
        timestamp = frame_idx / fps
        timestamps[frame_idx] = timestamp
        frame_rows.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp,
                "calibration_confidence": 1.0,
            }
        )

        for obj in frames[frame_idx]:
            if obj.role in ("player", "goalkeeper"):
                if obj.box_height <= 1.0:
                    continue
                team = obj.team or "unknown"
                track_teams[obj.track_id] = team
                track_roles[obj.track_id] = obj.role
                player_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_s": timestamp,
                        "track_id": obj.track_id,
                        "image_x": obj.image_x,
                        "image_y": obj.image_y,
                        "bbox_y1": obj.image_y - obj.box_height,
                        "bbox_y2": obj.image_y,
                        "team_id": team,
                        "pitch_x": obj.pitch_x,
                        "pitch_y": obj.pitch_y,
                        "role": obj.role,
                    }
                )
            elif obj.role == "ball":
                inside = (
                    obj.pitch_x is not None
                    and 0.0 <= obj.pitch_x <= pitch.length
                    and 0.0 <= obj.pitch_y <= pitch.width
                )
                ball_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_s": timestamp,
                        "image_x": obj.image_x,
                        "image_y": obj.image_y - obj.box_height / 2.0,
                        "pitch_x": obj.pitch_x if inside else None,
                        "pitch_y": obj.pitch_y if inside else None,
                        "ball_state": (
                            BallStateKind.OBSERVED.value if inside
                            else BallStateKind.UNKNOWN.value
                        ),
                    }
                )
                ball_by_frame[frame_idx] = (
                    obj.pitch_x if inside else None,
                    obj.pitch_y if inside else None,
                    BallStateKind.OBSERVED if inside else BallStateKind.UNKNOWN,
                    1.0,
                )

    players = pd.DataFrame(player_rows)
    return AnalysisContext(
        run_dir=Path(labels_path).parent,
        video_id=Path(labels_path).parent.name,
        fps=fps,
        pitch=pitch,
        players=players,
        valid_players=players,
        ball=pd.DataFrame(ball_rows),
        frames=pd.DataFrame(frame_rows),
        tracks=pd.DataFrame(
            [
                {"track_id": t, "team_id": team, "role": track_roles.get(t, "player")}
                for t, team in sorted(track_teams.items())
            ]
        ),
        ball_by_frame=ball_by_frame,
        timestamps=timestamps,
        track_teams=track_teams,
        track_roles=track_roles,
        manifest={"source": "soccernet_gsr_annotations", "perception": "ground truth"},
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


#: The engine's vocabulary mapped onto the reference's. The engine names a state
#: and a team separately; the reference names the team directly.
def _engine_label(state: str, team_id: str) -> PossessionLabel:
    if state == PossessionState.CONTESTED.value:
        return PossessionLabel.CONTESTED
    if state == PossessionState.CONTROLLED.value:
        if team_id in ("left", "right"):
            return PossessionLabel(team_id)
        # Controlled by a player whose team is unknown. Not a team prediction,
        # and not a claim that the ball was loose either.
        return PossessionLabel.UNKNOWN
    if state in (PossessionState.LOOSE_BALL.value, PossessionState.OUT_OF_PLAY.value):
        return PossessionLabel.LOOSE
    return PossessionLabel.UNKNOWN


@dataclass
class PossessionResult:
    """Frame-level agreement, with every denominator stated."""

    sequence: str
    configuration: str
    n_frames: int = 0
    #: frames the reference can score
    n_scorable: int = 0
    #: scorable frames on which the engine committed to a label
    n_predicted: int = 0
    per_label: dict[str, dict[str, int]] = field(default_factory=dict)
    holder_correct: int = 0
    holder_total: int = 0

    # -- rates ---------------------------------------------------------------- #

    @property
    def reference_coverage(self) -> float:
        return self.n_scorable / self.n_frames if self.n_frames else 0.0

    @property
    def prediction_coverage(self) -> float:
        """Share of scorable frames the engine was willing to label at all."""
        return self.n_predicted / self.n_scorable if self.n_scorable else 0.0

    def f1(self, label: str) -> float:
        counts = self.per_label.get(label)
        if not counts:
            return 0.0
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        denominator = 2 * tp + fp + fn
        return (2 * tp / denominator) if denominator else 0.0

    @property
    def team_f1(self) -> float:
        """Macro F1 over the two teams -- the Phase 2C readiness criterion.

        Macro, not micro: a clip in which one team dominates would let a model
        that always guesses that team score well on a micro average.
        """
        return (self.f1("left") + self.f1("right")) / 2.0

    @property
    def holder_accuracy(self) -> float:
        return self.holder_correct / self.holder_total if self.holder_total else 0.0

    def to_dict(self) -> dict:
        low, high = (
            wilson_interval(self.holder_correct, self.holder_total)
            if self.holder_total else (0.0, 0.0)
        )
        return {
            "sequence": self.sequence,
            "configuration": self.configuration,
            "n_frames": self.n_frames,
            "n_scorable": self.n_scorable,
            "n_predicted_on_scorable": self.n_predicted,
            "reference_coverage": round(self.reference_coverage, 4),
            "prediction_coverage": round(self.prediction_coverage, 4),
            "team_f1_macro": round(self.team_f1, 4),
            "per_label": {
                label: {**counts, "f1": round(self.f1(label), 4)}
                for label, counts in sorted(self.per_label.items())
            },
            "holder_accuracy": round(self.holder_accuracy, 4),
            "holder_accuracy_ci95": [round(low, 4), round(high, 4)],
            "holder_n": self.holder_total,
        }


def evaluate_possession_vs_gt(
    gt: PossessionGroundTruth,
    predicted: dict[int, tuple[str, str, int | None]],
    fps: float,
    configuration: str,
) -> PossessionResult:
    """Score per-frame engine output against the derived reference.

    ``predicted`` maps frame index -> ``(state, team_id, track_id)``.

    Frames the reference marks ``UNKNOWN`` are excluded from every denominator.
    The engine is not rewarded or punished for what the reference cannot see.
    """
    result = PossessionResult(sequence=gt.sequence, configuration=configuration)
    per_label: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for frame_idx in sorted(set(predicted) | {int(round(i.start_s * fps)) for i in gt.intervals}):
        timestamp = frame_idx / fps
        reference = gt.label_at(timestamp)
        result.n_frames += 1
        if not reference.is_scorable:
            continue
        result.n_scorable += 1

        entry = predicted.get(frame_idx)
        prediction = (
            _engine_label(entry[0], entry[1]) if entry else PossessionLabel.UNKNOWN
        )
        if prediction is not PossessionLabel.UNKNOWN:
            result.n_predicted += 1

        if prediction is reference:
            per_label[reference.value]["tp"] += 1
        else:
            per_label[reference.value]["fn"] += 1
            if prediction is not PossessionLabel.UNKNOWN:
                per_label[prediction.value]["fp"] += 1

        # Holder attribution is only scored where the reference names a holder
        # and the engine claims a team. Anything else is not a wrong answer, it
        # is no answer, and counting it as wrong would conflate the two.
        holder = gt.holder_at(timestamp)
        if holder is not None and reference.is_team and prediction is reference:
            result.holder_total += 1
            if entry and entry[2] == holder:
                result.holder_correct += 1

    result.per_label = {k: dict(v) for k, v in per_label.items()}
    return result


def aggregate(results: list[PossessionResult], configuration: str) -> dict:
    """Pool results across sequences, summing counts rather than averaging rates.

    Averaging per-sequence F1 would give a 30-second clip with four scorable
    frames the same weight as one with seven hundred.
    """
    pooled = PossessionResult(sequence="ALL", configuration=configuration)
    per_label: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    for result in results:
        pooled.n_frames += result.n_frames
        pooled.n_scorable += result.n_scorable
        pooled.n_predicted += result.n_predicted
        pooled.holder_correct += result.holder_correct
        pooled.holder_total += result.holder_total
        for label, counts in result.per_label.items():
            for key in ("tp", "fp", "fn"):
                per_label[label][key] += counts[key]
    pooled.per_label = {k: dict(v) for k, v in per_label.items()}
    return {
        **pooled.to_dict(),
        "n_sequences": len(results),
        "sequences": [r.sequence for r in results],
    }
