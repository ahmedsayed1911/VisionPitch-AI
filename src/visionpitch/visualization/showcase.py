"""Reference-style tactical overlay.

This is the *presentation* renderer, deliberately separate from
:mod:`visionpitch.visualization.annotate`, which stays as the debug view. The two
have opposite goals: the debug renderer exists to expose every quantity the
pipeline inferred, and this one exists to hide all of them and show only tactical
structure over untouched broadcast footage.

What it draws
-------------
1. a small filled dot at each player's feet, in their team's colour
2. a tactical graph joining *nearby* teammates
3. a distinct dot for match officials, connected to nothing

and nothing at all on a frame where fewer than ``min_team_players_per_frame``
players could be resolved -- see that field for why.

What it never draws
-------------------
Boxes, ids, labels, confidences, panels, minimaps, HUD text, pitch lanes, and
nothing whatsoever on the ball -- the ball must read exactly as it does in the
source broadcast. Ball detection and tracking still run upstream because other
stages consume them; this renderer simply never receives them.

The graph is local
------------------
Measured on the reference clip (12 frames, 129 recovered edges, dots and lines
read back out of the pixels): mean degree 2.1, two to three connected groups per
team per frame, edge length median 0.175 of frame width and p90 0.30. A raw
Delaunay triangulation of ten players averages degree ~4.2 and is connected *by
construction*, so the reference is a pruned triangulation. Three gates reproduce
it: an absolute tactical reach in metres, a relative one in units of the team's
own nearest-neighbour spacing (what keeps the graph local when a team spreads
out), and a ceiling on how long an edge may look on screen.

Delaunay rather than kNN, deliberately. Both hit the reference's degree in
testing, but kNN *forces* every player to keep their k nearest teammates however
far away they are, which is precisely the "distant teammates connected across the
pitch" failure this design has to avoid. A pruned triangulation lets a genuinely
isolated player stay isolated, and multiple disconnected groups per team are the
expected output, not a defect.

Eligibility
-----------
:func:`eligibility` is the single boundary deciding what a track may become on
screen, and **role outranks team assignment there**. A track resolved as an
official is drawn as an official even if team classification also handed it a
team, because the two-cluster colour model cannot express "belongs to neither
side" and will always name one. Officials never reach a team's point set, so
they cannot appear in any edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from visionpitch.common.config import ShowcaseConfig
from visionpitch.common.logging import get_logger
from visionpitch.common.types import Role, TeamId
from visionpitch.pitch.geometry import PitchConfiguration

log = get_logger("visualization.showcase")

#: roles that may carry a team colour and enter that team's graph
_TEAM_ROLES = frozenset({Role.OUTFIELD.value, Role.GOALKEEPER.value})
_TEAMS = frozenset({TeamId.A.value, TeamId.B.value})


class Eligibility(str, Enum):
    """What a track is allowed to become on screen."""

    #: team dot, and a node in that team's graph
    TEAM = "team"
    #: independent official's dot, in no graph
    REFEREE = "referee"
    #: drawn as nothing
    NONE = "none"


@dataclass(slots=True)
class ShowcasePlayer:
    """One tracked person, as the showcase renderer needs them."""

    track_id: int
    team_id: str
    role: str
    image_xy: tuple[float, float]
    pitch_xy: tuple[float, float] | None


def eligibility(player: ShowcasePlayer) -> Eligibility:
    """Classify one track. Role first, team second -- never the other way round.

    Testing the team first is what let officials into team graphs: they arrive
    here carrying a team id and a confidence near 1.0, because ``team_confidence``
    measures which of two clusters is *nearer*, not whether either one fits.
    """
    if player.role == Role.REFEREE.value:
        return Eligibility.REFEREE
    if player.role in _TEAM_ROLES and player.team_id in _TEAMS:
        return Eligibility.TEAM
    return Eligibility.NONE


@dataclass
class _EdgeMemory:
    """Hysteresis for the tactical graph.

    Delaunay neighbourhoods flip whenever four players approach co-circularity,
    which on real tracking data happens several times a second and reads as
    flicker. An edge must therefore be *proposed* for several consecutive frames
    before it is drawn, and *absent* for several before it is dropped, so the
    graph changes at the rate play changes rather than at the rate the
    triangulation is numerically unstable.

    Hysteresis smooths topology only. It is never allowed to keep an edge whose
    players have actually separated: the caller re-checks distance on every frame
    after :meth:`step` returns, so distance validity always wins.
    """

    on_frames: int = 3
    off_frames: int = 8
    score: dict[tuple[int, int], float] = field(default_factory=dict)
    drawn: set[tuple[int, int]] = field(default_factory=set)

    def step(self, proposed: set[tuple[int, int]], alive: set[int]) -> set[tuple[int, int]]:
        rise = 1.0 / max(1, self.on_frames)
        fall = 1.0 / max(1, self.off_frames)

        for edge in proposed:
            self.score[edge] = min(1.0, self.score.get(edge, 0.0) + rise)
        for edge in list(self.score):
            if edge in proposed:
                continue
            value = self.score[edge] - fall
            if value <= 0.0:
                del self.score[edge]
                self.drawn.discard(edge)
            else:
                self.score[edge] = value

        # Asymmetric: admitted only at full score (so it must survive on_frames
        # consecutive proposals) but kept while any score remains (so it takes
        # off_frames absences to disappear).
        for edge, value in self.score.items():
            if value >= 1.0:
                self.drawn.add(edge)

        # A pair whose track left the frame is not "briefly missing", it is over.
        for edge in list(self.score):
            if edge[0] not in alive or edge[1] not in alive:
                del self.score[edge]
                self.drawn.discard(edge)
        self.drawn = {e for e in self.drawn if e in self.score}
        return set(self.drawn)

    def reset(self) -> None:
        self.score.clear()
        self.drawn.clear()


class ShowcaseRenderer:
    """Draws the reference-style overlay onto a broadcast frame."""

    def __init__(self, config: ShowcaseConfig, pitch: PitchConfiguration) -> None:
        self.cfg = config
        self.pitch = pitch
        self._memory: dict[str, _EdgeMemory] = {
            team: _EdgeMemory(
                on_frames=config.graph_hysteresis_on_frames,
                off_frames=config.graph_hysteresis_off_frames,
            )
            for team in _TEAMS
        }
        #: refreshed per frame by render(); the default keeps the image-space
        #: gate meaningful when _team_edges is exercised on its own
        self._frame_width = 1280.0
        #: edges drawn on the most recent frame, as ``(team, a, b)``. Kept so the
        #: graph invariants can be measured on real footage rather than only in a
        #: synthetic test.
        self.last_edges: list[tuple[str, ShowcasePlayer, ShowcasePlayer]] = []
        #: markers actually painted on the most recent frame. Counting intent
        #: rather than output would over-report every abstained frame.
        self.last_team_dots = 0
        self.last_referee_dots = 0

    # -- colours ------------------------------------------------------------- #

    @staticmethod
    def _bgr(rgb) -> tuple[int, int, int]:
        return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    def _team_bgr(self, team_id: str) -> tuple[int, int, int]:
        rgb = self.cfg.team_colors.get(team_id)
        return self._bgr(rgb) if rgb is not None else (200, 200, 200)

    def _edge_bgr(self, team_id: str) -> tuple[int, int, int]:
        rgb = self.cfg.graph_colors.get(team_id, self.cfg.team_colors.get(team_id))
        return self._bgr(rgb) if rgb is not None else (0, 0, 0)

    # -- graph ---------------------------------------------------------------- #

    def _propose(
        self, points: np.ndarray, limit: float
    ) -> list[tuple[int, int]]:
        """Local adjacency over one team's points, pruned to nearby pairs.

        Delaunay supplies the candidate neighbourhood -- it is local by
        construction and produces the small triangles the reference shows. Two
        gates then prune it:

        * ``limit``, an absolute tactical reach (metres, or pixels in the
          uncalibrated fallback);
        * ``graph_local_scale`` times the point set's own median
          nearest-neighbour distance, which is scale-free and is what stops a
          spread-out team from becoming one large network.
        """
        n = points.shape[0]
        if n < 2:
            return []
        candidates = _delaunay_edges(points) if n >= 4 else None
        if candidates is None:
            candidates = _knn_edges(points, self.cfg.graph_knn)

        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        nearest = nearest[np.isfinite(nearest)]
        local = float(np.median(nearest)) if nearest.size else float("inf")
        relative_limit = self.cfg.graph_local_scale * local

        out = []
        for i, j in candidates:
            d = float(distances[i, j])
            if d <= limit and d <= relative_limit:
                out.append((i, j))
        return out

    def _team_edges(
        self, team_id: str, players: list[ShowcasePlayer], calibrated: bool
    ) -> list[tuple[ShowcasePlayer, ShowcasePlayer]]:
        """Local teammate adjacency, hysteresis-smoothed.

        ``players`` must already be one team's eligible members: this never
        filters by role, because by the time a point set arrives here the
        officials are supposed to be gone.
        """
        memory = self._memory[team_id]
        alive = {p.track_id for p in players}
        if len(players) < 2:
            memory.step(set(), alive)
            return []

        usable = [p for p in players if p.pitch_xy is not None] if calibrated else []
        if len(usable) >= 2:
            points = np.array([p.pitch_xy for p in usable], dtype=np.float64)
            limit = self.cfg.graph_max_edge_m
        else:
            # No usable pitch geometry this frame. Image space keeps the graph
            # alive through uncalibrated footage; the gate scales with frame
            # width so it means roughly the same thing at any zoom.
            usable = players
            points = np.array([p.image_xy for p in usable], dtype=np.float64)
            limit = self.cfg.graph_max_edge_image_frac * self._frame_width

        proposed = {
            (min(usable[i].track_id, usable[j].track_id),
             max(usable[i].track_id, usable[j].track_id))
            for i, j in self._propose(points, limit)
        }
        kept = memory.step(proposed, alive)

        # Re-check every surviving edge against the current geometry. Hysteresis
        # may smooth topology; it may not outlive the distance rule.
        by_id = {p.track_id: p for p in usable}
        index = {p.track_id: i for i, p in enumerate(usable)}
        image_limit = self.cfg.graph_max_edge_image_frac * self._frame_width
        out = []
        for a, b in kept:
            pa, pb = by_id.get(a), by_id.get(b)
            if pa is None or pb is None:
                continue
            if float(np.linalg.norm(points[index[a]] - points[index[b]])) > limit:
                continue
            # An offset homography can make two players 20 pitch-metres apart
            # span the whole frame. Both the tactical and the visual gate hold.
            span = float(
                np.hypot(pa.image_xy[0] - pb.image_xy[0], pa.image_xy[1] - pb.image_xy[1])
            )
            if span > image_limit:
                continue
            out.append((pa, pb))
        return out

    # -- public --------------------------------------------------------------- #

    def reset(self) -> None:
        for memory in self._memory.values():
            memory.reset()
        self.last_edges = []

    def render(
        self,
        image: np.ndarray,
        players: list[ShowcasePlayer],
        calibration_confidence: float,
        n_inliers: int = 0,
    ) -> np.ndarray:
        """Return ``image`` with the tactical overlay composited on top.

        There is deliberately no ball parameter and no homography parameter: the
        ball is never drawn, and with pitch lanes gone the homography is only
        needed upstream, to put ``pitch_xy`` on each player.
        """
        canvas = image if self.cfg.in_place else image.copy()
        self._frame_width = float(canvas.shape[1])

        by_team: dict[str, list[ShowcasePlayer]] = {team: [] for team in _TEAMS}
        officials: list[ShowcasePlayer] = []
        for player in players:
            verdict = eligibility(player)
            if verdict is Eligibility.TEAM:
                by_team[player.team_id].append(player)
            elif verdict is Eligibility.REFEREE:
                officials.append(player)

        # A frame the pipeline could barely resolve has no tactical structure to
        # show. Drawing the one dot it did resolve reads as an assertion about
        # the frame rather than an abstention on it, which is how a team dot
        # ended up on a promotional end card.
        if sum(len(m) for m in by_team.values()) < self.cfg.min_team_players_per_frame:
            self.last_edges = []
            self.last_team_dots = self.last_referee_dots = 0
            for memory in self._memory.values():
                memory.step(set(), set())
            return canvas

        calibrated = (
            calibration_confidence >= self.cfg.graph_min_calibration_confidence
            and n_inliers >= self.cfg.graph_min_inliers
        )

        overlay = canvas.copy()
        self.last_edges = []
        self.last_team_dots = self.last_referee_dots = 0
        for team_id, members in by_team.items():
            colour = self._edge_bgr(team_id)
            for pa, pb in self._team_edges(team_id, members, calibrated):
                self.last_edges.append((team_id, pa, pb))
                cv2.line(
                    overlay,
                    _as_point(pa.image_xy),
                    _as_point(pb.image_xy),
                    colour,
                    self.cfg.graph_thickness_px,
                    cv2.LINE_AA,
                )
        alpha = self.cfg.graph_opacity
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        for members in by_team.values():
            for player in members:
                self._draw_dot(canvas, player.image_xy, self._team_bgr(player.team_id))
        for official in officials:
            self._draw_dot(canvas, official.image_xy, self._bgr(self.cfg.referee_color))
        self.last_team_dots = sum(len(m) for m in by_team.values())
        self.last_referee_dots = len(officials)
        return canvas

    def _draw_dot(
        self, canvas: np.ndarray, xy: tuple[float, float], colour: tuple[int, int, int]
    ) -> None:
        centre = _as_point(xy)
        radius = self.cfg.dot_radius_px
        if self.cfg.dot_outline_px > 0:
            cv2.circle(
                canvas,
                centre,
                radius + self.cfg.dot_outline_px,
                self.cfg.dot_outline_bgr,
                -1,
                cv2.LINE_AA,
            )
        cv2.circle(canvas, centre, radius, colour, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _as_point(xy) -> tuple[int, int]:
    return (int(round(float(xy[0]))), int(round(float(xy[1]))))


def _delaunay_edges(points: np.ndarray) -> list[tuple[int, int]] | None:
    """Delaunay adjacency, or ``None`` when the point set is degenerate.

    Degenerate means collinear or near-duplicate, which happens in real football
    whenever a defensive line is flat. The caller falls back to kNN.
    """
    try:
        from scipy.spatial import Delaunay, QhullError
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        return None
    try:
        tri = Delaunay(points)
    except (QhullError, ValueError):
        return None
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = int(simplex[a]), int(simplex[b])
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def _knn_edges(points: np.ndarray, k: int) -> list[tuple[int, int]]:
    """kNN adjacency: each node keeps its ``k`` nearest neighbours."""
    n = points.shape[0]
    if n < 2:
        return []
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in np.argsort(distances[i])[: max(1, k)]:
            if np.isfinite(distances[i, int(j)]):
                edges.add((min(i, int(j)), max(i, int(j))))
    return sorted(edges)
