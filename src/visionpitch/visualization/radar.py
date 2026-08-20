"""2D tactical map ("radar").

Renders the reconstructed game state onto a scale drawing of the pitch. This is
the view that makes calibration quality immediately obvious to a human: if the
homography is wrong, players drift, cluster off the touchline, or swim during a
camera pan, and no amount of good detection hides it.

Everything is drawn from *pitch* coordinates, so the radar is independent of
camera, resolution and frame rate.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionpitch.common.config import VisualizationConfig  # noqa: I001
from visionpitch.common.types import Role, TeamId
from visionpitch.pitch.geometry import PitchConfiguration

_GRASS = (58, 122, 63)
_GRASS_DARK = (50, 108, 55)
_LINE = (235, 240, 235)
_BALL = (250, 250, 250)
_BALL_EDGE = (25, 25, 25)


class PitchRenderer:
    """Draws a scale pitch and plots objects on it."""

    def __init__(
        self, pitch: PitchConfiguration, config: VisualizationConfig
    ) -> None:
        self.pitch = pitch
        self.cfg = config
        self.padding = config.radar_padding_px
        self.scale = (config.radar_width_px - 2 * self.padding) / pitch.length
        self.width = config.radar_width_px
        self.height = int(round(pitch.width * self.scale + 2 * self.padding))
        #: kit colours discovered at run time, BGR, keyed by team id
        self._discovered: dict[str, tuple[int, int, int]] = {}
        self._background = self._draw_pitch()

    # -- coordinate transform ----------------------------------------------- #

    def to_canvas(self, x: float, y: float) -> tuple[int, int]:
        """Pitch metres -> radar pixels.

        The y axis is flipped: pitch ``y`` increases upward in the world frame,
        image rows increase downward.
        """
        px = self.padding + x * self.scale
        py = self.padding + (self.pitch.width - y) * self.scale
        return int(round(px)), int(round(py))

    # -- static background --------------------------------------------------- #

    def _draw_pitch(self) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = _GRASS

        # Mowing stripes: purely cosmetic, but they make the scale legible.
        stripe_width = self.pitch.length / 12
        for i in range(12):
            if i % 2:
                continue
            x1, _ = self.to_canvas(i * stripe_width, self.pitch.width)
            x2, _ = self.to_canvas((i + 1) * stripe_width, self.pitch.width)
            cv2.rectangle(
                canvas,
                (x1, self.padding),
                (x2, self.height - self.padding),
                _GRASS_DARK,
                -1,
            )

        p = self.pitch
        line = 2

        def rect(x1, y1, x2, y2):
            cv2.rectangle(canvas, self.to_canvas(x1, y1), self.to_canvas(x2, y2), _LINE, line)

        # Touchlines and goal lines
        rect(0, 0, p.length, p.width)
        # Halfway line
        cv2.line(
            canvas,
            self.to_canvas(p.length / 2, 0),
            self.to_canvas(p.length / 2, p.width),
            _LINE,
            line,
        )
        # Centre circle and spot
        cv2.circle(
            canvas,
            self.to_canvas(p.length / 2, p.width / 2),
            int(round(p.centre_circle_radius * self.scale)),
            _LINE,
            line,
        )
        cv2.circle(canvas, self.to_canvas(p.length / 2, p.width / 2), 3, _LINE, -1)

        pb_lo = (p.width - p.penalty_box_width) / 2
        pb_hi = (p.width + p.penalty_box_width) / 2
        gb_lo = (p.width - p.goal_box_width) / 2
        gb_hi = (p.width + p.goal_box_width) / 2
        goal_lo = (p.width - p.goal_width) / 2
        goal_hi = (p.width + p.goal_width) / 2

        for side in ("left", "right"):
            sign = 1 if side == "left" else -1
            base = 0.0 if side == "left" else p.length
            rect(base, pb_lo, base + sign * p.penalty_box_length, pb_hi)
            rect(base, gb_lo, base + sign * p.goal_box_length, gb_hi)
            spot = self.to_canvas(base + sign * p.penalty_spot_distance, p.width / 2)
            cv2.circle(canvas, spot, 3, _LINE, -1)
            # Goal mouth, drawn outside the goal line
            rect(base, goal_lo, base - sign * 2.0, goal_hi)

        return canvas

    # -- dynamic layer ------------------------------------------------------- #

    def render(
        self,
        objects: list[dict],
        frame_idx: int,
        timestamp_s: float,
        ball_trail: list[tuple[float, float]] | None = None,
        calibration_confidence: float = 0.0,
    ) -> np.ndarray:
        """Draw one frame of the tactical map.

        ``objects`` are dicts with ``pitch_x``, ``pitch_y``, ``team_id``,
        ``role``, ``track_id``, ``label`` and ``confidence``.
        """
        canvas = self._background.copy()

        if ball_trail:
            # Break the trail wherever the ball "moves" further than it physically
            # could between frames. Those jumps are estimation error, and joining
            # them draws a line straight across the pitch that a viewer reads as a
            # 60-metre pass that never happened.
            max_step_m = 6.0
            for i in range(1, len(ball_trail)):
                previous, current = ball_trail[i - 1], ball_trail[i]
                if float(np.hypot(current[0] - previous[0], current[1] - previous[1])) > max_step_m:
                    continue
                alpha = i / len(ball_trail)
                cv2.line(
                    canvas,
                    self.to_canvas(*previous),
                    self.to_canvas(*current),
                    tuple(int(c * alpha) for c in _BALL),
                    max(1, int(2 * alpha)),
                )

        for obj in objects:
            if obj.get("pitch_x") is None or obj.get("pitch_y") is None:
                continue
            centre = self.to_canvas(obj["pitch_x"], obj["pitch_y"])
            role = obj.get("role")

            if role == Role.BALL.value:
                cv2.circle(canvas, centre, 5, _BALL, -1)
                cv2.circle(canvas, centre, 5, _BALL_EDGE, 1)
                continue

            colour = self._colour(obj.get("team_id", TeamId.UNKNOWN.value))
            radius = 9 if role == Role.GOALKEEPER.value else 8

            cv2.circle(canvas, centre, radius, colour, -1)
            # Goalkeepers get a thick white ring, referees a black one, so the
            # roles stay distinguishable when both teams' colours are similar.
            if role == Role.GOALKEEPER.value:
                cv2.circle(canvas, centre, radius, (255, 255, 255), 2)
            elif role == Role.REFEREE.value:
                cv2.circle(canvas, centre, radius, (20, 20, 20), 2)
            else:
                cv2.circle(canvas, centre, radius, (20, 20, 20), 1)

            # A low-confidence team assignment gets a dashed-looking outer ring
            # so uncertainty is visible on the map itself, not only in the table.
            if obj.get("team_confidence", 1.0) < 0.5 and role != Role.REFEREE.value:
                cv2.circle(canvas, centre, radius + 4, (255, 255, 255), 1)

            if self.cfg.show_track_ids and obj.get("track_id") is not None:
                cv2.putText(
                    canvas,
                    str(obj["track_id"]),
                    (centre[0] - 6, centre[1] - radius - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        n_plotted = sum(
            1 for o in objects if o.get("pitch_x") is not None and o.get("pitch_y") is not None
        )
        self._draw_header(
            canvas, frame_idx, timestamp_s, calibration_confidence, len(objects), n_plotted
        )
        return canvas

    def _draw_header(
        self,
        canvas: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        calibration_confidence: float,
        n_objects: int,
        n_plotted: int | None = None,
    ) -> None:
        minutes, seconds = divmod(timestamp_s, 60)
        text = f"frame {frame_idx}  |  {int(minutes):02d}:{seconds:05.2f}  |  {n_objects} objects"
        cv2.putText(
            canvas, text, (self.padding, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        # An object detected in the video but absent from the map has not been
        # lost -- its projection was rejected as untrustworthy. Saying so is the
        # difference between a map that is incomplete and a map that is wrong.
        if n_plotted is not None and n_plotted < n_objects:
            missing = n_objects - n_plotted
            cv2.putText(
                canvas,
                f"{missing} object(s) not plotted: no reliable pitch position",
                (self.padding, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (90, 200, 250),
                1,
                cv2.LINE_AA,
            )

        # Calibration confidence is drawn on the radar because the radar is only
        # as trustworthy as the homography that produced it.
        label = f"calib {calibration_confidence:.2f}"
        colour = (
            (80, 220, 80) if calibration_confidence >= 0.6
            else (60, 200, 240) if calibration_confidence >= 0.4
            else (60, 60, 240)
        )
        cv2.putText(
            canvas, label, (self.width - self.padding - 110, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
        )

    def set_discovered_colours(self, team_colours: dict[str, list[int]]) -> None:
        """Draw each team in the kit colour that was actually discovered.

        Using fixed palette colours makes the map actively misleading: the white
        team rendered in red forces the viewer to translate between two colour
        schemes while reading a tactical picture.
        """
        for team_id, bgr in team_colours.items():
            if len(bgr) == 3:
                self._discovered[team_id] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def _colour(self, team_id: str) -> tuple[int, int, int]:
        discovered = self._discovered.get(team_id)
        if discovered is not None:
            # Push toward saturation so two pale kits stay distinguishable as dots.
            return tuple(int(np.clip(c * 1.15, 0, 255)) for c in discovered)  # type: ignore[return-value]
        rgb = self.cfg.team_colors.get(team_id, self.cfg.team_colors["unknown"])
        return (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # config is RGB, OpenCV is BGR

    def blank(self, frame_idx: int, timestamp_s: float, message: str) -> np.ndarray:
        """A radar frame for an uncalibrated moment, labelled as such."""
        canvas = self._background.copy()
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)
        cv2.putText(
            canvas, message, (self.padding, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA,
        )
        self._draw_header(canvas, frame_idx, timestamp_s, 0.0, 0)
        return canvas
