"""Canonical football pitch model.

Coordinate system
-----------------
Origin at the bottom-left corner of the pitch as drawn, ``x`` along the length
(0 -> ``length``), ``y`` along the width (0 -> ``width``), both in **metres**.
This is a right-handed 2D world frame on the ground plane; it is independent of
camera, resolution and frame rate, which is exactly why all analytics downstream
consume pitch coordinates rather than pixels.

Nothing here is hard-coded to a single stadium: dimensions are configurable and
IFAB permits 90-120 m x 45-90 m. The default is the FIFA-recommended 105 x 68 m.

Keypoint ordering
-----------------
:meth:`PitchConfiguration.vertices` returns 32 landmarks in the ordering used by
the pitch keypoint model this project ships with. The ordering is a *contract*
between the model and the homography solver -- if it is wrong, homography
reprojection error explodes, which is precisely the signal
``calibration.validate`` checks. See ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class PitchConfiguration:
    """Physical pitch dimensions, in metres."""

    length: float = 105.0
    width: float = 68.0
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0
    goal_width: float = 7.32

    def __post_init__(self) -> None:
        if not 90.0 <= self.length <= 120.0:
            raise ValueError(f"pitch length {self.length}m outside IFAB range 90-120m")
        if not 45.0 <= self.width <= 90.0:
            raise ValueError(f"pitch width {self.width}m outside IFAB range 45-90m")

    # -- landmark set ------------------------------------------------------- #

    @property
    def vertices(self) -> np.ndarray:
        """The 32 pitch landmarks, shape ``(32, 2)`` in metres.

        Index order (0-based) follows the keypoint model's output order:
        left goal line top->bottom, left goal box, left penalty spot, left
        penalty box, halfway line and centre circle poles, then the mirrored
        right-hand side, finishing with the two horizontal centre-circle poles.
        """
        w, ln = self.width, self.length
        pbw, pbl = self.penalty_box_width, self.penalty_box_length
        gbw, gbl = self.goal_box_width, self.goal_box_length
        r, ps = self.centre_circle_radius, self.penalty_spot_distance

        pts = [
            # --- left goal line (x = 0) ---
            (0.0, 0.0),  # 0  bottom-left corner
            (0.0, (w - pbw) / 2),  # 1  penalty box lower edge
            (0.0, (w - gbw) / 2),  # 2  goal box lower edge
            (0.0, (w + gbw) / 2),  # 3  goal box upper edge
            (0.0, (w + pbw) / 2),  # 4  penalty box upper edge
            (0.0, w),  # 5  top-left corner
            # --- left goal box ---
            (gbl, (w - gbw) / 2),  # 6
            (gbl, (w + gbw) / 2),  # 7
            # --- left penalty spot ---
            (ps, w / 2),  # 8
            # --- left penalty box ---
            (pbl, (w - pbw) / 2),  # 9
            (pbl, (w - gbw) / 2),  # 10  arc/box intersection (lower)
            (pbl, (w + gbw) / 2),  # 11  arc/box intersection (upper)
            (pbl, (w + pbw) / 2),  # 12
            # --- halfway line & centre circle vertical poles ---
            (ln / 2, 0.0),  # 13  halfway line, bottom touchline
            (ln / 2, w / 2 - r),  # 14  centre circle, lower pole
            (ln / 2, w / 2 + r),  # 15  centre circle, upper pole
            (ln / 2, w),  # 16  halfway line, top touchline
            # --- right penalty box ---
            (ln - pbl, (w - pbw) / 2),  # 17
            (ln - pbl, (w - gbw) / 2),  # 18
            (ln - pbl, (w + gbw) / 2),  # 19
            (ln - pbl, (w + pbw) / 2),  # 20
            # --- right penalty spot ---
            (ln - ps, w / 2),  # 21
            # --- right goal box ---
            (ln - gbl, (w - gbw) / 2),  # 22
            (ln - gbl, (w + gbw) / 2),  # 23
            # --- right goal line (x = length) ---
            (ln, 0.0),  # 24  bottom-right corner
            (ln, (w - pbw) / 2),  # 25
            (ln, (w - gbw) / 2),  # 26
            (ln, (w + gbw) / 2),  # 27
            (ln, (w + pbw) / 2),  # 28
            (ln, w),  # 29  top-right corner
            # --- centre circle horizontal poles ---
            (ln / 2 - r, w / 2),  # 30
            (ln / 2 + r, w / 2),  # 31
        ]
        return np.asarray(pts, dtype=np.float64)

    @property
    def n_vertices(self) -> int:
        return 32

    # -- derived geometry --------------------------------------------------- #

    @property
    def centre(self) -> tuple[float, float]:
        return (self.length / 2, self.width / 2)

    @property
    def area(self) -> float:
        return self.length * self.width

    def goal_centre(self, side: str) -> tuple[float, float]:
        """Centre of the goal mouth. ``side`` is ``"left"`` or ``"right"``."""
        if side == "left":
            return (0.0, self.width / 2)
        if side == "right":
            return (self.length, self.width / 2)
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Whether a pitch coordinate lies on the field of play."""
        return -margin <= x <= self.length + margin and -margin <= y <= self.width + margin

    def in_penalty_area(self, x: float, y: float, side: str) -> bool:
        """Whether a pitch coordinate lies inside a penalty area."""
        lo_y = (self.width - self.penalty_box_width) / 2
        hi_y = (self.width + self.penalty_box_width) / 2
        if not lo_y <= y <= hi_y:
            return False
        if side == "left":
            return 0.0 <= x <= self.penalty_box_length
        if side == "right":
            return self.length - self.penalty_box_length <= x <= self.length
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def normalise(self, x: float, y: float) -> tuple[float, float]:
        """Map metres to a resolution-independent ``[0, 1]`` square."""
        return (x / self.length, y / self.width)


class PitchZone(str, Enum):
    """Coarse thirds x channels grid, used for zonal aggregation in later phases."""

    DEF_LEFT = "def_left"
    DEF_CENTRE = "def_centre"
    DEF_RIGHT = "def_right"
    MID_LEFT = "mid_left"
    MID_CENTRE = "mid_centre"
    MID_RIGHT = "mid_right"
    ATT_LEFT = "att_left"
    ATT_CENTRE = "att_centre"
    ATT_RIGHT = "att_right"
    OFF_PITCH = "off_pitch"


_ZONE_GRID = [
    [PitchZone.DEF_LEFT, PitchZone.MID_LEFT, PitchZone.ATT_LEFT],
    [PitchZone.DEF_CENTRE, PitchZone.MID_CENTRE, PitchZone.ATT_CENTRE],
    [PitchZone.DEF_RIGHT, PitchZone.MID_RIGHT, PitchZone.ATT_RIGHT],
]


def zone_of(x: float, y: float, pitch: PitchConfiguration) -> PitchZone:
    """Classify a pitch coordinate into a thirds x channels zone.

    Zones are expressed left-to-right in the direction of increasing ``x``;
    which third is "attacking" depends on the team's direction of play and is
    resolved by the caller, not here.
    """
    if not pitch.contains(x, y, margin=2.0):
        return PitchZone.OFF_PITCH
    col = min(2, max(0, int(3 * x / pitch.length)))
    row = min(2, max(0, int(3 * y / pitch.width)))
    return _ZONE_GRID[row][col]
