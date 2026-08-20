"""Broadcast-frame annotation.

Draws tracks and the ball onto the source video. Two conventions worth noting:

* Players get an **ellipse at their feet** rather than a full box. The ellipse
  sits on the ground-contact point that is actually used for projection, so the
  annotated video shows the same quantity the pitch coordinates were computed
  from -- if the feet marker is wrong, the radar will be wrong too, and that is
  visible at a glance.
* Interpolated positions are drawn **hollow**. A viewer must always be able to
  tell an observation from an inference.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionpitch.common.config import VisualizationConfig
from visionpitch.common.types import BallState, Role, Track

_WHITE = (255, 255, 255)
_BLACK = (20, 20, 20)


class FrameAnnotator:
    """Overlays tracking and ball state on broadcast frames."""

    def __init__(self, config: VisualizationConfig) -> None:
        self.cfg = config
        self._ball_trail: list[tuple[float, float]] = []
        self._discovered: dict[str, tuple[int, int, int]] = {}

    def set_discovered_colours(self, team_colours: dict[str, list[int]]) -> None:
        """Annotate each team in its own discovered kit colour, not a fixed palette."""
        for team_id, bgr in team_colours.items():
            if len(bgr) == 3:
                self._discovered[team_id] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def _colour(self, team_id: str) -> tuple[int, int, int]:
        discovered = self._discovered.get(team_id)
        if discovered is not None:
            return tuple(int(np.clip(c * 1.15, 0, 255)) for c in discovered)  # type: ignore[return-value]
        rgb = self.cfg.team_colors.get(team_id, self.cfg.team_colors["unknown"])
        return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    # -- people -------------------------------------------------------------- #

    def _draw_person(
        self, image: np.ndarray, track: Track, obs, label: str
    ) -> None:
        colour = self._colour(track.team_id.value)
        x1, y1, x2, y2 = (int(round(v)) for v in obs.bbox.to_xyxy())
        cx, cy = obs.bbox.ground_contact
        width = max(6, int(round(obs.bbox.width)))

        thickness = 1 if obs.interpolated else 2
        # Foot ellipse marking the projection point.
        cv2.ellipse(
            image,
            center=(int(round(cx)), int(round(cy))),
            axes=(width // 2, max(4, width // 5)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=colour,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        if track.role is Role.GOALKEEPER:
            cv2.ellipse(
                image, (int(round(cx)), int(round(cy))), (width // 2 + 4, max(5, width // 4)),
                0, -45, 235, _WHITE, 1, cv2.LINE_AA,
            )
        elif track.role is Role.REFEREE:
            cv2.ellipse(
                image, (int(round(cx)), int(round(cy))), (width // 2 + 4, max(5, width // 4)),
                0, -45, 235, _BLACK, 1, cv2.LINE_AA,
            )

        if not label:
            return

        font_scale = 0.42
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        box_top = max(0, y1 - th - 8)
        cv2.rectangle(image, (x1, box_top), (x1 + tw + 8, box_top + th + 8), colour, -1)
        # Pick a legible text colour from the background's luminance rather than
        # assuming a dark kit colour.
        luminance = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
        text_colour = _BLACK if luminance > 140 else _WHITE
        cv2.putText(
            image, label, (x1 + 4, box_top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_colour, 1, cv2.LINE_AA,
        )
        _ = y2

    # -- ball ---------------------------------------------------------------- #

    def _draw_ball(self, image: np.ndarray, ball: BallState) -> None:
        if ball.position is None:
            self._ball_trail.clear()
            return

        x, y = int(round(ball.position[0])), int(round(ball.position[1]))
        self._ball_trail.append(ball.position)
        if len(self._ball_trail) > self.cfg.ball_trail_frames:
            self._ball_trail.pop(0)

        # Skip segments that imply an impossible jump: those are estimation
        # error, and drawing them reads as a pass the ball never made.
        max_step_px = 0.25 * image.shape[1]
        for i in range(1, len(self._ball_trail)):
            previous, current = self._ball_trail[i - 1], self._ball_trail[i]
            if float(np.hypot(current[0] - previous[0], current[1] - previous[1])) > max_step_px:
                continue
            alpha = i / len(self._ball_trail)
            p0 = tuple(int(round(v)) for v in previous)
            p1 = tuple(int(round(v)) for v in current)
            cv2.line(
                image, p0, p1, (int(80 + 175 * alpha),) * 3, max(1, int(3 * alpha)), cv2.LINE_AA
            )

        if ball.observed:
            cv2.circle(image, (x, y), 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 7, _BLACK, 2, cv2.LINE_AA)
        else:
            # Hollow marker plus an uncertainty ring: this position was inferred.
            cv2.circle(image, (x, y), 7, (200, 200, 255), 2, cv2.LINE_AA)
            radius = int(round(min(60.0, max(8.0, ball.uncertainty_px))))
            cv2.circle(image, (x, y), radius, (160, 160, 255), 1, cv2.LINE_AA)

    # -- public -------------------------------------------------------------- #

    def annotate(
        self,
        image: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        tracks: dict[int, Track],
        ball: BallState | None,
        calibration_confidence: float,
        labels: dict[int, str] | None = None,
    ) -> np.ndarray:
        canvas = image.copy()
        labels = labels or {}

        drawn = 0
        for track in tracks.values():
            obs = track.observation_at(frame_idx)
            if obs is None:
                continue
            self._draw_person(canvas, track, obs, labels.get(track.track_id, ""))
            drawn += 1

        if ball is not None:
            self._draw_ball(canvas, ball)

        self._draw_hud(canvas, frame_idx, timestamp_s, drawn, ball, calibration_confidence)
        return canvas

    def _draw_hud(
        self,
        image: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        n_tracks: int,
        ball: BallState | None,
        calibration_confidence: float,
    ) -> None:
        h, w = image.shape[:2]
        panel = image[0:34, 0:w].copy()
        cv2.rectangle(panel, (0, 0), (w, 34), (0, 0, 0), -1)
        image[0:34, 0:w] = cv2.addWeighted(panel, 0.5, image[0:34, 0:w], 0.5, 0)

        minutes, seconds = divmod(timestamp_s, 60)
        if ball is None or ball.position is None:
            ball_state = "ball: not visible"
        elif ball.observed:
            ball_state = f"ball: tracked ({ball.confidence:.2f})"
        else:
            ball_state = f"ball: inferred (+/-{ball.uncertainty_px:.0f}px)"

        text = (
            f"frame {frame_idx}  {int(minutes):02d}:{seconds:05.2f}  |  "
            f"{n_tracks} tracked  |  {ball_state}  |  calib {calibration_confidence:.2f}"
        )
        cv2.putText(
            image, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA
        )

    def reset(self) -> None:
        self._ball_trail.clear()


def stack_side_by_side(broadcast: np.ndarray, radar: np.ndarray) -> np.ndarray:
    """Combine broadcast and radar into one frame, radar below, matched width."""
    bw = broadcast.shape[1]
    scale = bw / radar.shape[1]
    radar_resized = cv2.resize(
        radar, (bw, int(round(radar.shape[0] * scale))), interpolation=cv2.INTER_AREA
    )
    return np.vstack([broadcast, radar_resized])
