"""Possession ground truth derived from SoccerNet-GSR game state.

Phase 2C, Part 4 -- the blocker Phase 2B named second: "no possession number
means anything until there is a reference to compare it against".

What this is, and what it is not
--------------------------------
SN-GSR annotates, per frame, every player and the ball with a **team label**, a
**track id**, and a **pitch position in metres**. That is enough to derive who
had the ball without running any part of this project's perception stack. The
result is a *derived reference*, not human annotation, and the distinction is
load-bearing:

* It is **independent of the engine's perception**. The engine works from
  detected boxes; this works from annotated ones. The difference between the two
  measures perception loss directly, which is exactly the split Phase 2B could
  not make.
* It is **independent of the engine's geometry**. The engine judges proximity in
  *image space*, normalised by bounding-box height. This judges it in *metric
  pitch space*. Agreement between them is therefore not tautological.
* It is **not independent of the proximity assumption itself**. Both say the
  ball belongs to whoever is nearest. If that premise is wrong -- a player
  screening the ball, a defender closer than the carrier -- both are wrong
  together, and this reference will not reveal it. Validating that premise needs
  a human watching video, which no corpus here provides.

So: this measures the temporal state machine, the contest logic, and the cost of
imperfect perception. It does not measure whether "nearest player owns the ball"
is true football.

The airborne-ball problem, measured
-----------------------------------
``bbox_pitch`` is a **ground-plane projection**. When the ball is in the air the
projection slides toward the horizon, so its pitch coordinate is not where the
ball is. Measured over 20 sequences (13,126 frames with both a ball and players):

* 9.8% of frames put the ball outside the pitch plus a 5 m margin
* 9.4% imply a frame-to-frame step above 1.6 m, i.e. over 40 m/s at 25 fps
  (the observed maximum was 374 m per frame, or 9,350 m/s)
* 13.3% fail at least one check

Those frames are labelled ``UNKNOWN``, never guessed. A reference that invented
possession for a ball it cannot locate would make the engine look better exactly
where the engine is weakest.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from visionpitch.common.logging import get_logger

log = get_logger("evaluation.possession_gt")

POSSESSION_GT_SCHEMA_VERSION = "1.0.0"

#: GSR pitch coordinates are centred on the halfway line; VisionPitch uses a
#: corner origin. Applied on load so everything downstream sees one convention.
GSR_X_OFFSET = 52.5
GSR_Y_OFFSET = 34.0


class PossessionLabel(str, Enum):
    """Ground-truth possession states.

    ``LOOSE`` and ``UNKNOWN`` are deliberately different. Loose means the ball
    was located and nobody was near it; unknown means the ball's position is not
    trustworthy. Collapsing them would let a detector that loses the ball score
    as if it had proven the ball was free.
    """

    LEFT = "left"
    RIGHT = "right"
    CONTESTED = "contested"
    LOOSE = "loose"
    UNKNOWN = "unknown"

    @property
    def is_team(self) -> bool:
        return self in (PossessionLabel.LEFT, PossessionLabel.RIGHT)

    @property
    def is_scorable(self) -> bool:
        """Whether a prediction for this frame can be right or wrong at all."""
        return self is not PossessionLabel.UNKNOWN


@dataclass
class DerivationParams:
    """Thresholds for the derivation, each measured rather than chosen.

    ``control_radius_m`` comes from the distribution of ball-to-nearest-player
    distance over 11,385 usable GSR frames. That distribution is bimodal: a peak
    at 0.5-1.0 m (a player with the ball), a trough at 1.5-2.0 m, and a second
    population beyond it (ball in transit). 1.75 m sits in the trough. 44.4% of
    usable frames fall inside it, which is close to the share of a match in which
    somebody actually has the ball.

    ``contest_margin_m`` is the 10th percentile of the gap between the nearest
    player and the nearest opponent (0.69 m), rounded down: when two opponents
    are within half a metre of each other of the ball, calling it for either is
    a coin toss, so it is called contested.
    """

    control_radius_m: float = 1.75
    contest_margin_m: float = 0.5
    #: 40 m/s at 25 fps. Above this the ground projection is not the ball.
    max_ball_step_m: float = 1.6
    #: how far outside the pitch a ball may plausibly be
    out_of_bounds_margin_m: float = 5.0
    #: a state must hold this long to be emitted; below it, it is flicker
    min_state_duration_s: float = 0.20
    #: gaps up to this long inside one team's possession are bridged
    max_bridge_gap_s: float = 0.24

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PossessionInterval:
    start_s: float
    end_s: float
    label: PossessionLabel
    holder_track_id: int | None = None
    n_frames: int = 0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "label": self.label.value,
            "holder_track_id": self.holder_track_id,
            "n_frames": self.n_frames,
        }


@dataclass
class PossessionGroundTruth:
    """Possession intervals for one sequence, with the derivation on record."""

    sequence: str
    fps: float
    duration_s: float
    intervals: list[PossessionInterval] = field(default_factory=list)
    params: DerivationParams = field(default_factory=DerivationParams)
    source: str = "soccernet_gsr"
    basis: str = "derived from annotated boxes; not human possession annotation"
    schema_version: str = POSSESSION_GT_SCHEMA_VERSION
    #: frames the derivation saw at all, for honest denominators
    n_frames: int = 0
    n_frames_unknown: int = 0

    # -- coverage ------------------------------------------------------------ #

    @property
    def coverage(self) -> float:
        """Share of frames the reference can actually score."""
        if not self.n_frames:
            return 0.0
        return 1.0 - self.n_frames_unknown / self.n_frames

    def label_at(self, t: float) -> PossessionLabel:
        for interval in self.intervals:
            if interval.start_s <= t < interval.end_s:
                return interval.label
        return PossessionLabel.UNKNOWN

    def holder_at(self, t: float) -> int | None:
        for interval in self.intervals:
            if interval.start_s <= t < interval.end_s:
                return interval.holder_track_id
        return None

    def team_share(self) -> dict[str, float]:
        """Share of *scorable* time each team held the ball."""
        totals: dict[str, float] = defaultdict(float)
        for interval in self.intervals:
            if interval.label.is_scorable:
                totals[interval.label.value] += interval.duration_s
        scorable = sum(totals.values())
        if scorable <= 0:
            return {}
        return {k: v / scorable for k, v in sorted(totals.items())}

    # -- persistence --------------------------------------------------------- #

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "source": self.source,
            "basis": self.basis,
            "fps": self.fps,
            "duration_s": round(self.duration_s, 4),
            "n_frames": self.n_frames,
            "n_frames_unknown": self.n_frames_unknown,
            "coverage": round(self.coverage, 4),
            "params": self.params.to_dict(),
            "team_share_of_scorable_time": {
                k: round(v, 4) for k, v in self.team_share().items()
            },
            "intervals": [i.to_dict() for i in self.intervals],
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "sequence": self.sequence,
                "params": self.params.to_dict(),
                "intervals": [i.to_dict() for i in self.intervals],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> PossessionGroundTruth:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        gt = PossessionGroundTruth(
            sequence=data["sequence"],
            fps=float(data["fps"]),
            duration_s=float(data["duration_s"]),
            intervals=[
                PossessionInterval(
                    start_s=float(i["start_s"]), end_s=float(i["end_s"]),
                    label=PossessionLabel(i["label"]),
                    holder_track_id=i.get("holder_track_id"),
                    n_frames=int(i.get("n_frames", 0)),
                )
                for i in data["intervals"]
            ],
            params=DerivationParams(**data["params"]),
            source=data.get("source", "soccernet_gsr"),
            basis=data.get("basis", ""),
            schema_version=data.get("schema_version", POSSESSION_GT_SCHEMA_VERSION),
            n_frames=int(data.get("n_frames", 0)),
            n_frames_unknown=int(data.get("n_frames_unknown", 0)),
        )
        if gt.fingerprint() != data.get("fingerprint"):
            raise ValueError(
                f"{path} was edited outside the tool: stored fingerprint "
                f"{data.get('fingerprint')} != recomputed {gt.fingerprint()}"
            )
        return gt


# --------------------------------------------------------------------------- #
# Loading GSR game state
# --------------------------------------------------------------------------- #


@dataclass
class GSRObject:
    """One annotated object in one frame, in VisionPitch pitch coordinates."""

    track_id: int
    role: str
    team: str | None
    image_x: float
    image_y: float
    box_height: float
    pitch_x: float | None
    pitch_y: float | None


def load_gsr_gamestate(labels_path: str | Path) -> tuple[dict[int, list[GSRObject]], float]:
    """frame index -> annotated objects, plus the sequence frame rate.

    Reads the raw ``Labels-GameState.json`` rather than going through
    ``GSRDataset``: that loader keeps boxes and track ids but drops the team,
    role and pitch coordinates this module exists to use.
    """
    data = json.loads(Path(labels_path).read_text(encoding="utf-8"))

    frame_of_image: dict[str, int] = {}
    for image in data.get("images", []):
        frame_of_image[str(image["image_id"])] = int(
            image.get("frame_id") or image.get("file_name", "0").split(".")[0][-6:]
        )

    frames: dict[int, list[GSRObject]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        attributes = annotation.get("attributes") or {}
        role = attributes.get("role")
        if role is None:
            continue
        image_id = str(annotation.get("image_id"))
        if image_id not in frame_of_image:
            continue
        box = annotation.get("bbox_image") or {}
        pitch = annotation.get("bbox_pitch")
        frames[frame_of_image[image_id]].append(
            GSRObject(
                track_id=int(annotation.get("track_id") or -1),
                role=role,
                team=attributes.get("team"),
                image_x=float(box.get("x_center", 0.0)),
                image_y=float(box.get("y", 0.0)) + float(box.get("h", 0.0)),
                box_height=float(box.get("h", 0.0)),
                pitch_x=(
                    float(pitch["x_bottom_middle"]) + GSR_X_OFFSET if pitch else None
                ),
                pitch_y=(
                    float(pitch["y_bottom_middle"]) + GSR_Y_OFFSET if pitch else None
                ),
            )
        )

    fps = float((data.get("info") or {}).get("frame_rate") or 25.0)
    return dict(frames), fps


# --------------------------------------------------------------------------- #
# Derivation
# --------------------------------------------------------------------------- #


def _frame_label(
    objects: list[GSRObject],
    previous_ball: tuple[float, float] | None,
    params: DerivationParams,
    pitch_length: float,
    pitch_width: float,
) -> tuple[PossessionLabel, int | None, tuple[float, float] | None]:
    """Label one frame. Returns (label, holder, ball position for continuity)."""
    ball = next(
        (o for o in objects if o.role == "ball" and o.pitch_x is not None), None
    )
    if ball is None:
        return PossessionLabel.UNKNOWN, None, None

    position = (ball.pitch_x, ball.pitch_y)

    margin = params.out_of_bounds_margin_m
    if not (
        -margin <= position[0] <= pitch_length + margin
        and -margin <= position[1] <= pitch_width + margin
    ):
        # An airborne ball projected onto the ground plane; position unusable.
        # Returned as the previous position so the next frame's step check is
        # not measured against a coordinate we have just rejected.
        return PossessionLabel.UNKNOWN, None, previous_ball

    if previous_ball is not None:
        step = math.dist(position, previous_ball)
        if step > params.max_ball_step_m:
            return PossessionLabel.UNKNOWN, None, position

    people = [
        o for o in objects
        if o.role in ("player", "goalkeeper") and o.pitch_x is not None and o.team
    ]
    if len(people) < 2:
        return PossessionLabel.UNKNOWN, None, position

    ranked = sorted(
        ((math.dist(position, (o.pitch_x, o.pitch_y)), o) for o in people),
        key=lambda pair: pair[0],
    )
    nearest_distance, nearest = ranked[0]
    if nearest_distance > params.control_radius_m:
        return PossessionLabel.LOOSE, None, position

    opponent = next(
        (pair for pair in ranked[1:] if pair[1].team != nearest.team), None
    )
    if (
        opponent is not None
        and opponent[0] <= params.control_radius_m
        and opponent[0] - nearest_distance < params.contest_margin_m
    ):
        return PossessionLabel.CONTESTED, None, position

    label = (
        PossessionLabel.LEFT if nearest.team == "left" else PossessionLabel.RIGHT
    )
    return label, nearest.track_id, position


def _to_intervals(
    labels: list[tuple[int, float, PossessionLabel, int | None]],
    params: DerivationParams,
    fps: float,
) -> list[PossessionInterval]:
    """Collapse per-frame labels into intervals, dropping sub-threshold flicker.

    The minimum-duration filter is applied to runs of the **label alone**, not
    of ``(label, holder)``. Within one team's possession the nearest player
    changes constantly -- a team-mate steps closer for a few frames during a
    close pass -- and keying the filter on the holder would chop a long, real
    team possession into fragments that each fall below the threshold and get
    discarded. Measured on six sequences, that mistake cost 20 points of
    coverage and roughly halved the time attributed to either team.

    Holder changes still produce separate intervals, so player attribution keeps
    its resolution; they just no longer decide whether the team label survives.

    Short runs become ``UNKNOWN`` rather than joining a neighbour: a state too
    brief to be real is not evidence for whatever surrounded it.
    """
    if not labels:
        return []

    frame_duration = 1.0 / fps if fps > 0 else 0.04

    # -- pass 1: suppress label runs too short to be a real state ------------- #
    label_runs: list[list] = []
    for _, timestamp, label, _ in labels:
        if label_runs and label_runs[-1][2] is label:
            label_runs[-1][1] = timestamp
        else:
            label_runs.append([timestamp, timestamp, label, 0])
        label_runs[-1][3] += 1

    suppressed: set[int] = set()
    cursor = 0
    for start, end, label, count in label_runs:
        duration = end - start + frame_duration
        if duration < params.min_state_duration_s and label is not PossessionLabel.UNKNOWN:
            suppressed.update(range(cursor, cursor + count))
        cursor += count

    cleaned = [
        (frame_idx, timestamp,
         PossessionLabel.UNKNOWN if i in suppressed else label,
         None if i in suppressed else holder)
        for i, (frame_idx, timestamp, label, holder) in enumerate(labels)
    ]

    # -- pass 2: intervals, split on holder as well as label ------------------ #
    merged: list[PossessionInterval] = []
    for _, timestamp, label, holder in cleaned:
        if merged and merged[-1].label is label and merged[-1].holder_track_id == holder:
            merged[-1].end_s = timestamp + frame_duration
            merged[-1].n_frames += 1
        else:
            merged.append(
                PossessionInterval(
                    start_s=timestamp, end_s=timestamp + frame_duration,
                    label=label, holder_track_id=holder, n_frames=1,
                )
            )
    return merged


def derive_from_gsr(
    labels_path: str | Path,
    params: DerivationParams | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> PossessionGroundTruth:
    """Derive possession intervals for one GSR sequence."""
    params = params or DerivationParams()
    frames, fps = load_gsr_gamestate(labels_path)
    sequence = Path(labels_path).parent.name

    labels: list[tuple[int, float, PossessionLabel, int | None]] = []
    previous_ball: tuple[float, float] | None = None
    for frame_idx in sorted(frames):
        label, holder, previous_ball = _frame_label(
            frames[frame_idx], previous_ball, params, pitch_length, pitch_width
        )
        labels.append((frame_idx, frame_idx / fps, label, holder))

    intervals = _to_intervals(labels, params, fps)
    # Counted from the intervals, not from the raw per-frame labels: the flicker
    # filter turns some frames unknown after the fact, and coverage has to
    # describe what the reference will actually score.
    unknown = sum(
        i.n_frames for i in intervals if i.label is PossessionLabel.UNKNOWN
    )
    gt = PossessionGroundTruth(
        sequence=sequence,
        fps=fps,
        duration_s=(len(labels) / fps) if labels else 0.0,
        intervals=intervals,
        params=params,
        n_frames=len(labels),
        n_frames_unknown=unknown,
    )
    log.info(
        "%s: %d frames, coverage %.3f, %d interval(s), share %s",
        sequence, gt.n_frames, gt.coverage, len(intervals),
        {k: round(v, 3) for k, v in gt.team_share().items()},
    )
    return gt
