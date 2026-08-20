"""Global tracklet association.

Why the greedy stitcher was not enough
--------------------------------------
The Phase 1 stitcher walked tracks in start order and joined each to the
best-scoring earlier candidate available at that moment. That is locally optimal
and globally arbitrary: an early, mediocre join consumes a tracklet that a later,
much better join needed, and there is no mechanism to undo it. Measured on the
validation clip it rejoined 96 fragments out of 262 raw tracks and still left a
median track of 46 frames.

What this does instead
----------------------
Tracklet association is posed as a **min-cost matching** over the bipartite graph
of (tracklet tail -> tracklet head) and solved exactly with the Hungarian
algorithm. Every admissible join competes simultaneously, so the solver can
decline a good-looking local join in favour of a globally cheaper assignment.
Chains form by iterating the solve to a fixed point: A->B and B->C are found in
successive rounds and collapsed.

Costs combine four independent kinds of evidence, each of which alone is
insufficient in football:

* **motion** - where the earlier tracklet's velocity says it should be
* **appearance** - grass-suppressed torso colour, which separates *teams* well
  and individuals poorly, so it is a weak term deliberately
* **pitch geometry** - metres, not pixels, when calibration is confident. A
  camera pan moves a stationary player 60 px and 0 m, so this is by far the
  strongest cue when it is available
* **size consistency** - a proxy for depth

Safety
------
Wrong merges are worse than fragments: a fragment costs recall on one player,
a wrong merge fabricates a trajectory that teleports between two people and
corrupts every physical statistic derived from it. Every admissibility check is
therefore a hard gate rather than a cost term, and the module reports how many
joins it refused and why.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from visionpitch.common.config import TrackingConfig
from visionpitch.common.geometry import apply_homography
from visionpitch.common.logging import get_logger
from visionpitch.common.types import CalibrationResult, ObjectClass, Track
from visionpitch.tracking.postprocess import merge_class_votes

log = get_logger("tracking.association")

#: Cost assigned to a forbidden pairing. Large enough that the solver will
#: always prefer leaving a tracklet unmatched.
FORBIDDEN = 1e6


@dataclass
class TrackletFeatures:
    """Everything needed to score a tracklet's endpoints, computed once."""

    track_id: int
    object_class: ObjectClass
    first_frame: int
    last_frame: int
    n_observations: int

    head_centre: np.ndarray
    tail_centre: np.ndarray
    head_velocity: np.ndarray
    tail_velocity: np.ndarray
    head_height: float
    tail_height: float

    #: pitch coordinates at the endpoints, when calibration allowed it
    #: pitch coordinates near the endpoints, with the frame they came from.
    #: Not necessarily the endpoint frame itself: calibration is confident on
    #: only ~60% of frames, so insisting on the exact endpoint made pitch
    #: evidence available for 6 merges out of 150. Searching a few observations
    #: inward finds a usable homography far more often, and carrying the frame
    #: index keeps the speed gate honest about the real elapsed time.
    head_pitch: np.ndarray | None = None
    head_pitch_frame: int | None = None
    tail_pitch: np.ndarray | None = None
    tail_pitch_frame: int | None = None

    appearance: np.ndarray | None = None


@dataclass
class AssociationReport:
    rounds: int = 0
    merges: int = 0
    tracks_in: int = 0
    tracks_out: int = 0
    refusals: Counter = field(default_factory=Counter)
    #: joins accepted per evidence type actually used
    pitch_based_merges: int = 0
    image_based_merges: int = 0

    def to_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "merges": self.merges,
            "tracks_in": self.tracks_in,
            "tracks_out": self.tracks_out,
            "refusals": dict(self.refusals),
            "pitch_based_merges": self.pitch_based_merges,
            "image_based_merges": self.image_based_merges,
        }


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


def _endpoint_velocity(track: Track, at_end: bool, n: int = 6) -> np.ndarray:
    observations = [o for o in track.observations if not o.interpolated]
    if len(observations) < 2:
        return np.zeros(2)
    window = observations[-n:] if at_end else observations[:n]
    if len(window) < 2:
        return np.zeros(2)
    p0 = np.array(window[0].bbox.center, dtype=np.float64)
    p1 = np.array(window[-1].bbox.center, dtype=np.float64)
    dt = max(1, window[-1].frame_idx - window[0].frame_idx)
    return (p1 - p0) / dt


def _project_near_endpoint(
    track: Track,
    calibration: dict[int, CalibrationResult] | None,
    min_confidence: float,
    at_end: bool,
    max_probe: int = 12,
) -> tuple[np.ndarray | None, int | None]:
    """Pitch position from the observation nearest an endpoint that calibrates.

    Walks inward from the endpoint for up to ``max_probe`` real observations,
    taking the first one whose frame has a confident homography. Returns the
    position and the frame it came from, or ``(None, None)``.

    Still strict about *quality*: a weak homography yields a plausible-looking
    metre value that is wrong by tens of metres, and that is worse in an
    association cost than having no pitch term at all. What is relaxed is
    *which frame* supplies it, not how good it has to be.
    """
    if calibration is None:
        return None, None

    observations = [o for o in track.observations if not o.interpolated]
    if not observations:
        return None, None
    ordered = observations[::-1] if at_end else observations

    for observation in ordered[:max_probe]:
        result = calibration.get(observation.frame_idx)
        if result is None or not result.is_valid or result.confidence < min_confidence:
            continue
        projected = apply_homography(
            result.homography, np.array([observation.bbox.ground_contact])
        )[0]
        if np.isfinite(projected).all():
            return projected, observation.frame_idx
    return None, None


def extract_features(
    tracks: dict[int, Track],
    calibration: dict[int, CalibrationResult] | None,
    min_calibration_confidence: float,
    appearance: dict[int, np.ndarray] | None = None,
) -> dict[int, TrackletFeatures]:
    features: dict[int, TrackletFeatures] = {}
    for track_id, track in tracks.items():
        real = [o for o in track.observations if not o.interpolated]
        if not real:
            continue
        head, tail = real[0], real[-1]
        head_pitch, head_pitch_frame = _project_near_endpoint(
            track, calibration, min_calibration_confidence, at_end=False
        )
        tail_pitch, tail_pitch_frame = _project_near_endpoint(
            track, calibration, min_calibration_confidence, at_end=True
        )
        features[track_id] = TrackletFeatures(
            track_id=track_id,
            object_class=track.object_class,
            first_frame=head.frame_idx,
            last_frame=tail.frame_idx,
            n_observations=len(real),
            head_centre=np.array(head.bbox.center, dtype=np.float64),
            tail_centre=np.array(tail.bbox.center, dtype=np.float64),
            head_velocity=_endpoint_velocity(track, at_end=False),
            tail_velocity=_endpoint_velocity(track, at_end=True),
            head_height=float(head.bbox.height),
            tail_height=float(tail.bbox.height),
            head_pitch=head_pitch,
            head_pitch_frame=head_pitch_frame,
            tail_pitch=tail_pitch,
            tail_pitch_frame=tail_pitch_frame,
            appearance=(appearance or {}).get(track_id),
        )
    return features


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def pair_cost(
    earlier: TrackletFeatures,
    later: TrackletFeatures,
    config: TrackingConfig,
    refusals: Counter,
) -> tuple[float, str]:
    """Cost of appending ``later`` to ``earlier``.

    Returns ``(cost, evidence)``. ``FORBIDDEN`` means the join is inadmissible;
    ``evidence`` names the strongest cue used, for reporting.
    """
    gap = later.first_frame - earlier.last_frame

    if gap <= 0:
        refusals["temporal_overlap"] += 1
        return FORBIDDEN, "none"
    if gap > config.assoc_max_gap_frames:
        refusals["gap_too_long"] += 1
        return FORBIDDEN, "none"
    if earlier.object_class is not later.object_class:
        refusals["class_mismatch"] += 1
        return FORBIDDEN, "none"

    # Size consistency. A player does not change apparent height by more than
    # about half across a short occlusion; a bigger change means different depth,
    # therefore a different person.
    ratio = earlier.tail_height / max(1e-6, later.head_height)
    if not 0.55 <= ratio <= 1.8:
        refusals["size_inconsistent"] += 1
        return FORBIDDEN, "none"
    size_cost = abs(np.log(ratio))

    # -- geometry: pitch metres if trustworthy, else image pixels ------------ #
    if (
        earlier.tail_pitch is not None
        and later.head_pitch is not None
        and earlier.tail_pitch_frame is not None
        and later.head_pitch_frame is not None
    ):
        # Use the real elapsed time between the two *projected* frames, which is
        # not the tracklet gap when the projection came from a frame inward of
        # the endpoint.
        elapsed_frames = max(1, later.head_pitch_frame - earlier.tail_pitch_frame)
        distance_m = float(np.linalg.norm(later.head_pitch - earlier.tail_pitch))
        # A player covers at most ~9.5 m/s; the constant absorbs calibration
        # error, which dominates over a short gap.
        limit_m = (
            config.assoc_max_pitch_speed_m_s
            * (elapsed_frames / max(1e-6, config.assoc_fps))
            + config.assoc_pitch_slack_m
        )
        if distance_m > limit_m:
            refusals["pitch_distance_exceeded"] += 1
            return FORBIDDEN, "none"
        geometry_cost = distance_m / max(1e-6, limit_m)
        evidence = "pitch"
    else:
        # Beyond a short gap, pixels are not enough evidence to authorise a
        # join. The allowance has to grow with the gap to survive camera pan
        # (see below), but a budget that permits half a frame's width of drift
        # will happily merge two different players. So past this horizon a join
        # must be justified in pitch space or not at all.
        #
        # This is the deliberate trade: refusing these leaves fragments, and a
        # fragment costs recall on one player, whereas a wrong merge fabricates
        # a trajectory that teleports between two people and corrupts every
        # physical statistic computed from it.
        if gap > config.assoc_max_image_only_gap_frames:
            refusals["long_gap_without_pitch_evidence"] += 1
            return FORBIDDEN, "none"

        predicted_px = earlier.tail_centre + earlier.tail_velocity * gap
        distance_px = float(np.linalg.norm(later.head_centre - predicted_px))
        # The allowance must grow with the gap. A flat pixel budget is a
        # statement about a static camera: over a 90-frame gap a panning
        # broadcast camera translates the whole scene by hundreds of pixels, so
        # a fixed 120 px ceiling refuses essentially every long-gap join
        # regardless of whether it is correct. Measured: 7532 refusals on the
        # validation clip, and only 6 of 150 merges could fall back to pitch
        # evidence instead.
        limit_px = (
            config.stitch_max_distance_px + config.assoc_px_per_frame_allowance * gap
        )
        if distance_px > limit_px:
            refusals["image_distance_exceeded"] += 1
            return FORBIDDEN, "none"
        geometry_cost = distance_px / max(1e-6, limit_px)
        evidence = "image"

    # -- appearance: a weak, gated term -------------------------------------- #
    appearance_cost = 0.5
    if earlier.appearance is not None and later.appearance is not None:
        a, b = earlier.appearance, later.appearance
        if np.any(a) and np.any(b):
            similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            appearance_cost = float(np.clip(1.0 - similarity, 0.0, 1.0))
            # Identical kit makes this near-zero for *every* pair of teammates,
            # so it may refine a ranking but must never authorise a join on its
            # own. It is capped at a third of the total weight below.
            if appearance_cost > config.assoc_appearance_reject:
                refusals["appearance_mismatch"] += 1
                return FORBIDDEN, "none"

    gap_cost = gap / max(1, config.assoc_max_gap_frames)

    cost = (
        config.assoc_w_geometry * geometry_cost
        + config.assoc_w_appearance * appearance_cost
        + config.assoc_w_gap * gap_cost
        + config.assoc_w_size * size_cost
    )
    return float(cost), evidence


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


def associate_once(
    tracks: dict[int, Track],
    features: dict[int, TrackletFeatures],
    config: TrackingConfig,
    report: AssociationReport,
) -> dict[int, int]:
    """One global matching round. Returns ``{later_id: earlier_id}``."""
    ids = sorted(features)
    if len(ids) < 2:
        return {}

    index = {track_id: i for i, track_id in enumerate(ids)}
    cost = np.full((len(ids), len(ids)), FORBIDDEN, dtype=np.float64)
    evidence = np.empty((len(ids), len(ids)), dtype=object)

    for earlier_id in ids:
        for later_id in ids:
            if earlier_id == later_id:
                continue
            value, kind = pair_cost(
                features[earlier_id], features[later_id], config, report.refusals
            )
            cost[index[earlier_id], index[later_id]] = value
            evidence[index[earlier_id], index[later_id]] = kind

    rows, cols = linear_sum_assignment(cost)
    merges: dict[int, int] = {}
    for r, c in zip(rows, cols, strict=True):
        if cost[r, c] >= config.assoc_max_cost:
            continue
        earlier_id, later_id = ids[r], ids[c]
        merges[later_id] = earlier_id
        if evidence[r, c] == "pitch":
            report.pitch_based_merges += 1
        else:
            report.image_based_merges += 1

    # A tracklet may be claimed as a successor only once, and may not be both
    # someone's successor and its own predecessor in the same round.
    cleaned = {}
    claimed_predecessors: set[int] = set()
    for later_id, earlier_id in sorted(merges.items(), key=lambda kv: kv[0]):
        if earlier_id in claimed_predecessors or earlier_id in merges:
            continue
        cleaned[later_id] = earlier_id
        claimed_predecessors.add(earlier_id)
    _ = tracks
    return cleaned


def associate_tracklets(
    tracks: dict[int, Track],
    config: TrackingConfig,
    calibration: dict[int, CalibrationResult] | None = None,
    min_calibration_confidence: float = 0.5,
    appearance: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, Track], dict[int, int], AssociationReport]:
    """Globally associate tracklets into longer tracks.

    Iterates the matching to a fixed point so that chains of three or more
    fragments collapse. Returns the merged tracks, a map from every absorbed id
    to its survivor, and a report.
    """
    report = AssociationReport(tracks_in=len(tracks))
    working = {tid: t for tid, t in tracks.items()}
    id_map: dict[int, int] = {}

    for _ in range(config.assoc_max_rounds):
        features = extract_features(
            working, calibration, min_calibration_confidence, appearance
        )
        merges = associate_once(working, features, config, report)
        if not merges:
            break

        report.rounds += 1
        for later_id, earlier_id in merges.items():
            target = working.get(earlier_id)
            source = working.pop(later_id, None)
            if target is None or source is None:
                continue
            target.observations.extend(source.observations)
            target.observations.sort(key=lambda o: o.frame_idx)
            merge_class_votes(target, source)
            id_map[later_id] = earlier_id
            report.merges += 1

    # Collapse chains so a caller's old id resolves to the final survivor.
    resolved: dict[int, int] = {}
    for source in id_map:
        target = source
        guard = set()
        while target in id_map and target not in guard:
            guard.add(target)
            target = id_map[target]
        resolved[source] = target

    report.tracks_out = len(working)
    log.info(
        "global association: %d -> %d tracks in %d round(s) (%d merges: "
        "%d pitch-based, %d image-based)",
        report.tracks_in,
        report.tracks_out,
        report.rounds,
        report.merges,
        report.pitch_based_merges,
        report.image_based_merges,
    )
    return working, resolved, report
