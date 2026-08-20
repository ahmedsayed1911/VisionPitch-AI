"""Conservative track-level role resolution.

Why this exists
---------------
The multiclass detector emits four classes, and two of them are unreliable at
broadcast scale for opposite reasons:

* ``referee`` has poor precision. A player in a dark or unusual kit, a player
  half-occluded at the touchline, or a coach in frame all draw referee boxes.
  Because a track's ``object_class`` used to be frozen at the birth detection,
  a *single* spurious referee box was enough to paint a genuine player as an
  official for the rest of the clip -- and, worse, referees skip team
  classification entirely, so the mistake also erased the player's team.
* ``goalkeeper`` has poor recall *and* poor stability: keepers are small, far
  from the camera and frequently mistaken for outfielders in a different kit.

Neither is fixable by retraining here, so this module treats the detector's
class as **one piece of evidence among several** and requires agreement before
overriding the far more reliable team-colour assignment.

Evidence used
-------------
================  =========================================================
signal            source
================  =========================================================
class votes       confidence-weighted detector votes over the whole track
team vote         colour-cluster vote share from team classification
kit outlierness   distance from both team centroids in colour space
pitch position    median longitudinal position through the homography
================  =========================================================

Bias
----
Every threshold here is set so that the *failure mode is abstention*. A track we
cannot resolve becomes ``Role.UNKNOWN``, which the showcase renderer draws as
nothing at all. Rendering a midfielder as an official is a visible, obviously
wrong error; drawing nothing is merely a gap.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from visionpitch.common.logging import get_logger
from visionpitch.common.types import (
    CalibrationResult,
    ObjectClass,
    Role,
    TeamId,
    Track,
)
from visionpitch.pitch.geometry import PitchConfiguration

log = get_logger("team_classification.roles")


@dataclass
class RolePolicy:
    """Thresholds governing role assignment. All deliberately conservative."""

    # -- referee ---------------------------------------------------------- #
    #: fraction of a track's weighted class evidence that must say "referee"
    referee_min_vote_share: float = 0.55
    #: and at least this many frames the detector actually called referee
    referee_min_frames: int = 15
    #: a team assignment at or above this vote share is not overridden by a
    #: referee prediction, however persistent
    referee_team_veto_confidence: float = 0.70
    #: kit colour must be this many times further from the nearest team centroid
    #: than the median track, unless the class evidence is overwhelming
    referee_min_kit_outlier_ratio: float = 1.15
    #: class evidence above this share bypasses the kit-outlier requirement
    referee_dominant_vote_share: float = 0.85
    #: A referee call that is persistent but never reaches a majority lands
    #: here: at or above this share the track is abstained on rather than handed
    #: a team. Measured across two broadcasts, this is where officials and
    #: players separate cleanly -- officials at 0.447/0.512 (UCL) and >=0.624
    #: (FIFA), against players at <=0.103 (UCL) and <=0.415 (FIFA).
    referee_abstain_vote_share: float = 0.40

    # -- goalkeeper -------------------------------------------------------- #
    goalkeeper_min_vote_share: float = 0.40
    goalkeeper_min_frames: int = 12
    #: a keeper must spend the clip inside this many metres of a goal line
    goalkeeper_max_goal_distance_m: float = 30.0
    #: below this many calibrated samples the positional test cannot run, and a
    #: detector-only keeper is demoted rather than trusted
    goalkeeper_min_calibrated_samples: int = 8
    #: at most this many keeper tracks survive per team (longest first)
    goalkeeper_max_per_team: int = 2


@dataclass
class RoleReport:
    """What the resolver decided, and why. Surfaced in the run summary."""

    referee_accepted: list[int] = field(default_factory=list)
    referee_rejected_team_veto: list[int] = field(default_factory=list)
    referee_rejected_weak_votes: list[int] = field(default_factory=list)
    referee_rejected_kit: list[int] = field(default_factory=list)
    referee_abstained_split_vote: list[int] = field(default_factory=list)
    #: every track the detector called referee at least once, with the evidence
    #: the decision actually used. Without this the report could only say what
    #: was accepted, not why anything was turned down.
    referee_evidence: list[dict] = field(default_factory=list)
    goalkeeper_accepted: list[int] = field(default_factory=list)
    goalkeeper_demoted_position: list[int] = field(default_factory=list)
    goalkeeper_demoted_weak_votes: list[int] = field(default_factory=list)
    goalkeeper_demoted_surplus: list[int] = field(default_factory=list)
    promoted_to_unknown: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "referee_accepted": len(self.referee_accepted),
            "referee_rejected_team_veto": len(self.referee_rejected_team_veto),
            "referee_rejected_weak_votes": len(self.referee_rejected_weak_votes),
            "referee_rejected_kit": len(self.referee_rejected_kit),
            "referee_abstained_split_vote": len(self.referee_abstained_split_vote),
            "goalkeeper_accepted": len(self.goalkeeper_accepted),
            "goalkeeper_demoted_position": len(self.goalkeeper_demoted_position),
            "goalkeeper_demoted_weak_votes": len(self.goalkeeper_demoted_weak_votes),
            "goalkeeper_demoted_surplus": len(self.goalkeeper_demoted_surplus),
            "unresolved_to_unknown": len(self.promoted_to_unknown),
            "referee_track_ids": sorted(self.referee_accepted),
            "goalkeeper_track_ids": sorted(self.goalkeeper_accepted),
            "referee_evidence": sorted(
                self.referee_evidence, key=lambda e: -e["referee_frames"]
            )[:40],
        }


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _project(result: CalibrationResult, point: tuple[float, float]) -> np.ndarray | None:
    if result.homography is None:
        return None
    src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float64)
    dst = np.matmul(
        result.homography, np.array([src[0, 0, 0], src[0, 0, 1], 1.0], dtype=np.float64)
    )
    if abs(dst[2]) < 1e-9:
        return None
    return np.array([dst[0] / dst[2], dst[1] / dst[2]], dtype=np.float64)


def _pitch_samples(
    track: Track,
    calibration: dict[int, CalibrationResult] | None,
    min_confidence: float,
) -> np.ndarray:
    """Longitudinal pitch positions for a track, in metres. May be empty."""
    if not calibration:
        return np.zeros(0)
    out: list[float] = []
    for obs in track.observations:
        result = calibration.get(obs.frame_idx)
        if result is None or not result.is_valid or result.confidence < min_confidence:
            continue
        point = _project(result, obs.bbox.ground_contact)
        if point is not None and np.all(np.isfinite(point)):
            out.append(float(point[0]))
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #


def resolve_roles(
    tracks: dict[int, Track],
    calibration: dict[int, CalibrationResult] | None,
    pitch: PitchConfiguration,
    *,
    kit_distance: dict[int, float] | None = None,
    policy: RolePolicy | None = None,
    min_calibration_confidence: float = 0.4,
) -> RoleReport:
    """Assign :class:`Role` to every person track in place.

    ``kit_distance`` maps a track id to its median distance from the nearest
    team colour centroid. Referees wear kits chosen to clash with both teams, so
    a track sitting *close* to a team centroid is evidence against the detector's
    referee call regardless of how confident that call was.
    """
    policy = policy or RolePolicy()
    report = RoleReport()
    kit_distance = kit_distance or {}

    # Normalise kit distance against the population: the absolute scale depends
    # on the feature space and the fitted clusters, the relative one does not.
    finite = [v for v in kit_distance.values() if np.isfinite(v)]
    kit_reference = float(np.median(finite)) if finite else 0.0

    def kit_ratio(track_id: int) -> float | None:
        if kit_reference <= 0:
            return None
        value = kit_distance.get(track_id)
        if value is None or not np.isfinite(value):
            return None
        return float(value / kit_reference)

    for track in tracks.values():
        if track.object_class is ObjectClass.BALL:
            track.role = Role.BALL
            track.role_confidence = 1.0
            track.team_id = TeamId.NONE
            continue

        _resolve_person(track, policy, report, kit_ratio(track.track_id))

    _resolve_goalkeepers(
        tracks, calibration, pitch, policy, report, min_calibration_confidence
    )

    log.info(
        "roles resolved: %d referee, %d goalkeeper, %d referee call(s) overruled",
        len(report.referee_accepted),
        len(report.goalkeeper_accepted),
        len(report.referee_rejected_team_veto)
        + len(report.referee_rejected_weak_votes)
        + len(report.referee_rejected_kit),
    )
    return report


def _resolve_person(
    track: Track,
    policy: RolePolicy,
    report: RoleReport,
    kit_ratio: float | None,
) -> None:
    """Referee-vs-player decision for one track. Goalkeepers are settled later."""
    referee_share = track.class_share(ObjectClass.REFEREE)
    referee_frames = track.class_frames(ObjectClass.REFEREE)
    has_team = track.team_id in (TeamId.A, TeamId.B)

    if referee_frames > 0:
        report.referee_evidence.append(
            {
                "track_id": track.track_id,
                "referee_share": round(float(referee_share), 3),
                "referee_frames": int(referee_frames),
                "kit_ratio": None if kit_ratio is None else round(float(kit_ratio), 3),
                "team_id": track.team_id.value,
                "team_confidence": round(float(track.team_confidence), 3),
            }
        )

    if referee_share < policy.referee_min_vote_share or (
        referee_frames < policy.referee_min_frames
    ):
        # Counted unconditionally. Guarding this on ``track.role is
        # Role.REFEREE`` made it dead: no stage sets that before the resolver
        # runs, so the report claimed nothing had ever been turned down while
        # officials were being silently handed a team.
        if referee_frames > 0:
            report.referee_rejected_weak_votes.append(track.track_id)

        if (
            referee_frames >= policy.referee_min_frames
            and referee_share >= policy.referee_abstain_vote_share
        ):
            # Persistent referee evidence that never reached a majority: too weak
            # to render as an official, too strong to hand a team's colours and
            # passing graph to. Abstaining is the only option that cannot produce
            # a visibly wrong frame either way.
            #
            # The kit-outlier signal deliberately does *not* rescue these. It is
            # trustworthy when the two team kits bracket the official's, and
            # useless when they do not: on the UCL broadcast the cyan officials
            # scored 1.189 while genuine players scored 1.406-1.978, so promoting
            # on kit would have created false referees while still missing these.
            report.referee_abstained_split_vote.append(track.track_id)
            track.role = Role.UNKNOWN
            track.role_confidence = 0.0
            track.team_id = TeamId.UNKNOWN
            track.team_confidence = 0.0
            report.promoted_to_unknown.append(track.track_id)
            return

        _as_player(track, has_team, report)
        return

    # Does the kit actually fit the team it was assigned to?
    #
    # This has to be asked *before* the team vote is allowed to veto anything.
    # ``team_confidence`` is a **relative** measure: the classifier fits exactly
    # two clusters, so a third kit is forced into whichever of the two is nearer
    # and reports near-total confidence for a colour that matches neither. On the
    # reference broadcast that put 31 officials -- 9,604 person-frames, every one
    # of them team_confidence 1.00 -- into a team's dots and passing graph, at
    # kit distances of 95-152 against a 67.3 outlier cut for genuine players.
    # Consulting the vote first made the absolute evidence unreachable.
    kit_is_outlier = kit_ratio is not None and kit_ratio >= policy.referee_min_kit_outlier_ratio
    class_evidence_dominant = referee_share >= policy.referee_dominant_vote_share

    if not (kit_is_outlier or class_evidence_dominant):
        # The kit sits inside a team's colour cluster, so the referee call is not
        # corroborated by appearance.
        if has_team and track.team_confidence >= policy.referee_team_veto_confidence:
            # A confidently team-assigned player is not reclassified as an
            # official: the colour vote is measured over dozens of crops and the
            # referee class is the detector's weakest.
            report.referee_rejected_team_veto.append(track.track_id)
            _as_player(track, True, report)
            return
        # Persistent referee votes, an unconvinced colour vote, and a kit that
        # says nothing. Abstain rather than pick a side.
        report.referee_rejected_kit.append(track.track_id)
        track.role = Role.UNKNOWN
        track.role_confidence = 0.0
        if not has_team:
            track.team_id = TeamId.UNKNOWN
        report.promoted_to_unknown.append(track.track_id)
        return

    track.role = Role.REFEREE
    track.role_confidence = float(referee_share)
    track.team_id = TeamId.NONE
    track.team_confidence = 1.0
    report.referee_accepted.append(track.track_id)


def _as_player(track: Track, has_team: bool, report: RoleReport) -> None:
    """Mark a track as a field player, provisionally outfield."""
    if track.team_id is TeamId.NONE:
        # Was forced to NONE by an earlier referee call that no longer stands.
        track.team_id = TeamId.UNKNOWN
        track.team_confidence = 0.0
        has_team = False
    track.role = Role.OUTFIELD
    track.role_confidence = 0.8 if has_team else 0.4
    if not has_team:
        report.promoted_to_unknown.append(track.track_id)


def _resolve_goalkeepers(
    tracks: dict[int, Track],
    calibration: dict[int, CalibrationResult] | None,
    pitch: PitchConfiguration,
    policy: RolePolicy,
    report: RoleReport,
    min_calibration_confidence: float,
) -> None:
    """Promote outfield tracks to keeper only with class *and* pitch evidence.

    A goalkeeper is defined by where they stand, not only by what the detector
    calls them. Requiring the track to live near a goal line removes the common
    failure where a defender in a distinctive kit is called a keeper in midfield,
    and it does so without needing the keeper's own colour to be separable.
    """
    candidates = [
        t
        for t in tracks.values()
        if t.role is Role.OUTFIELD
        and t.class_frames(ObjectClass.GOALKEEPER) > 0
    ]

    accepted: list[tuple[Track, float]] = []
    for track in candidates:
        share = track.class_share(ObjectClass.GOALKEEPER)
        frames = track.class_frames(ObjectClass.GOALKEEPER)
        if share < policy.goalkeeper_min_vote_share or frames < policy.goalkeeper_min_frames:
            report.goalkeeper_demoted_weak_votes.append(track.track_id)
            continue

        samples = _pitch_samples(track, calibration, min_calibration_confidence)
        if samples.size < policy.goalkeeper_min_calibrated_samples:
            # No usable geometry. The detector alone is not enough.
            report.goalkeeper_demoted_position.append(track.track_id)
            continue

        median_x = float(np.median(samples))
        goal_distance = min(median_x, pitch.length - median_x)
        if goal_distance > policy.goalkeeper_max_goal_distance_m:
            report.goalkeeper_demoted_position.append(track.track_id)
            continue

        accepted.append((track, goal_distance))

    # At most a couple of keeper tracks per team survive; football supplies one
    # keeper per side, and fragmentation supplies the rest.
    by_team: dict[str, list[tuple[Track, float]]] = defaultdict(list)
    for track, distance in accepted:
        by_team[track.team_id.value].append((track, distance))

    for _team, entries in by_team.items():
        entries.sort(key=lambda e: (-e[0].length, e[1]))
        for rank, (track, distance) in enumerate(entries):
            if rank >= policy.goalkeeper_max_per_team:
                report.goalkeeper_demoted_surplus.append(track.track_id)
                continue
            track.role = Role.GOALKEEPER
            track.role_confidence = float(
                min(1.0, track.class_share(ObjectClass.GOALKEEPER) + 0.2)
            )
            report.goalkeeper_accepted.append(track.track_id)
            _ = distance
