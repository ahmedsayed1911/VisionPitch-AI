"""Automatic match setup discovery.

The product requirement is that the user uploads a video and picks a mode --
nothing else. Team names, colours, keepers, attack direction and squad
composition are all inferred. This module derives the parts that need
*pitch-space* evidence, which is why it runs after calibration rather than
inside team classification.

Everything here reports a confidence and is allowed to answer "unknown". A short
clip genuinely does not contain enough evidence to establish, say, a halftime
direction switch, and inventing one would corrupt every directional metric in
Phase 3.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from visionpitch.common.geometry import apply_homography
from visionpitch.common.logging import get_logger
from visionpitch.common.types import CalibrationResult, Role, TeamId, Track
from visionpitch.pitch.geometry import PitchConfiguration

log = get_logger("game_state.discovery")


@dataclass
class MatchSetup:
    """What the system inferred about the match, with confidences."""

    #: team -> "left_to_right" | "right_to_left" | "unknown"
    attack_directions: dict[str, str] = field(default_factory=dict)
    attack_direction_confidence: dict[str, float] = field(default_factory=dict)
    #: team -> side of the pitch that team's goalkeeper defends
    defended_sides: dict[str, str] = field(default_factory=dict)
    #: estimated simultaneously-active outfield players per team
    active_players: dict[str, int] = field(default_factory=dict)
    #: candidate direction-switch frames (halftime), empty when undetectable
    direction_switch_frames: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "attack_directions": self.attack_directions,
            "attack_direction_confidence": self.attack_direction_confidence,
            "defended_sides": self.defended_sides,
            "active_players": self.active_players,
            "direction_switch_frames": self.direction_switch_frames,
            "warnings": self.warnings,
        }


def _pitch_positions(
    tracks: dict[int, Track],
    calibration: dict[int, CalibrationResult],
    pitch: PitchConfiguration,
) -> dict[TeamId, dict[int, list[np.ndarray]]]:
    """Team -> frame -> list of pitch positions of that team's outfield players."""
    out: dict[TeamId, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for track in tracks.values():
        if track.team_id not in (TeamId.A, TeamId.B):
            continue
        if track.role not in (Role.OUTFIELD, Role.GOALKEEPER):
            continue
        for obs in track.observations:
            if obs.interpolated:
                continue
            result = calibration.get(obs.frame_idx)
            if result is None or not result.is_valid:
                continue
            projected = apply_homography(
                result.homography, np.array([obs.bbox.ground_contact])
            )[0]
            if not np.isfinite(projected).all():
                continue
            if not pitch.contains(float(projected[0]), float(projected[1]), margin=10.0):
                continue
            out[track.team_id][obs.frame_idx].append(projected)
    return out


def discover_match_setup(
    tracks: dict[int, Track],
    calibration: dict[int, CalibrationResult],
    pitch: PitchConfiguration,
    frame_indices: list[int],
) -> MatchSetup:
    """Infer attack direction, defended ends and squad size from pitch geometry."""
    setup = MatchSetup()
    positions = _pitch_positions(tracks, calibration, pitch)

    if not positions:
        setup.warnings.append(
            "no calibrated team positions available; match setup could not be inferred"
        )
        return setup

    # -- which end does each keeper defend ---------------------------------- #
    keeper_sides: dict[TeamId, str] = {}
    for track in tracks.values():
        if track.role is not Role.GOALKEEPER or track.team_id not in (TeamId.A, TeamId.B):
            continue
        xs = []
        for obs in track.observations:
            result = calibration.get(obs.frame_idx)
            if result is None or not result.is_valid:
                continue
            projected = apply_homography(
                result.homography, np.array([obs.bbox.ground_contact])
            )[0]
            if np.isfinite(projected).all():
                xs.append(float(projected[0]))
        if xs:
            keeper_sides[track.team_id] = (
                "left" if float(np.median(xs)) < pitch.length / 2 else "right"
            )

    # -- direction of attack -------------------------------------------------#
    # A team attacks the goal its keeper does *not* defend. When no keeper was
    # identified, fall back on the team's own centroid: over a clip of live play
    # a team spends more time in the opponent's half than its own only weakly,
    # so this fallback is reported with much lower confidence.
    for team in (TeamId.A, TeamId.B):
        key = team.value
        if team in keeper_sides:
            side = keeper_sides[team]
            setup.defended_sides[key] = side
            setup.attack_directions[key] = (
                "left_to_right" if side == "left" else "right_to_left"
            )
            setup.attack_direction_confidence[key] = 0.85
            continue

        frames = positions.get(team, {})
        if not frames:
            setup.attack_directions[key] = "unknown"
            setup.attack_direction_confidence[key] = 0.0
            continue

        centroids = [np.mean(np.array(pts), axis=0)[0] for pts in frames.values() if pts]
        if not centroids:
            setup.attack_directions[key] = "unknown"
            setup.attack_direction_confidence[key] = 0.0
            continue

        mean_x = float(np.mean(centroids))
        setup.attack_directions[key] = (
            "left_to_right" if mean_x < pitch.length / 2 else "right_to_left"
        )
        # Deliberately low: the centroid heuristic is weak evidence.
        setup.attack_direction_confidence[key] = 0.35
        setup.warnings.append(
            f"team {key}: no goalkeeper identified, attack direction inferred from "
            f"team centroid only and should be treated as provisional"
        )

    # Both teams cannot attack the same way. If they do, the evidence conflicts
    # and both are downgraded rather than one being silently flipped.
    directions = [setup.attack_directions.get(t.value) for t in (TeamId.A, TeamId.B)]
    if directions[0] == directions[1] and directions[0] not in (None, "unknown"):
        setup.warnings.append(
            "both teams were inferred to attack the same direction; the inference "
            "is inconsistent and both directions are marked unknown"
        )
        for team in (TeamId.A, TeamId.B):
            setup.attack_directions[team.value] = "unknown"
            setup.attack_direction_confidence[team.value] = 0.0

    # -- squad size ---------------------------------------------------------- #
    for team, frames in positions.items():
        counts = [len(pts) for pts in frames.values()]
        if counts:
            # The 90th percentile rather than the max: the max is set by the one
            # frame where a duplicate track briefly existed.
            setup.active_players[team.value] = int(round(float(np.percentile(counts, 90))))

    for team, n in setup.active_players.items():
        if n > 11:
            setup.warnings.append(
                f"team {team}: {n} simultaneous players detected, which exceeds 11 "
                f"and indicates duplicate tracks or team misclassification"
            )

    setup.direction_switch_frames = _detect_direction_switches(
        keeper_sides, tracks, calibration, pitch, frame_indices
    )

    log.info(
        "match setup: directions=%s, defended=%s, active=%s",
        setup.attack_directions,
        setup.defended_sides,
        setup.active_players,
    )
    return setup


def _detect_direction_switches(
    keeper_sides: dict[TeamId, str],
    tracks: dict[int, Track],
    calibration: dict[int, CalibrationResult],
    pitch: PitchConfiguration,
    frame_indices: list[int],
    window_frames: int = 900,
) -> list[int]:
    """Find frames where the keepers appear to have swapped ends.

    Requires at least two full windows of evidence on each side of a candidate,
    so on a short clip it correctly returns nothing rather than reporting a
    spurious halftime.
    """
    if len(frame_indices) < 4 * window_frames:
        return []

    keeper_tracks = [
        t for t in tracks.values()
        if t.role is Role.GOALKEEPER and t.team_id in (TeamId.A, TeamId.B)
    ]
    if not keeper_tracks:
        return []

    side_by_window: dict[int, dict[TeamId, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for track in keeper_tracks:
        for obs in track.observations:
            result = calibration.get(obs.frame_idx)
            if result is None or not result.is_valid:
                continue
            projected = apply_homography(
                result.homography, np.array([obs.bbox.ground_contact])
            )[0]
            if np.isfinite(projected).all():
                window = obs.frame_idx // window_frames
                side_by_window[window][track.team_id].append(float(projected[0]))

    switches: list[int] = []
    previous: dict[TeamId, str] = {}
    for window in sorted(side_by_window):
        current = {
            team: ("left" if float(np.median(xs)) < pitch.length / 2 else "right")
            for team, xs in side_by_window[window].items()
            if len(xs) >= 20
        }
        if previous and current:
            flipped = [
                team for team in current
                if team in previous and current[team] != previous[team]
            ]
            if len(flipped) == len(current) and flipped:
                switches.append(window * window_frames)
        if current:
            previous = current

    if switches:
        log.info("detected %d candidate direction switch(es) at frames %s", len(switches), switches)
    return switches
