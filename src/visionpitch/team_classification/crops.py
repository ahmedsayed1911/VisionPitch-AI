"""Jersey region extraction.

Getting the crop right matters more than the model that consumes it. A naive
full-box crop of a player is roughly 45% pitch, 20% shorts and socks, 10% skin,
and only 25% shirt -- and the pitch is the single most saturated, consistent
colour in the image. Feed that to any clustering method and it happily separates
"player on grass" from "player on the touchline mud" instead of separating the
two teams.

So: crop the torso, drop grass pixels, and refuse crops that are too small or
too occluded to carry a signal, rather than emitting a low-quality feature that
looks like evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from visionpitch.common.config import TeamClassificationConfig


@dataclass(slots=True)
class JerseyCrop:
    """One usable torso crop, with the identity needed for temporal voting.

    Both the raw and the grass-suppressed crop are kept, because the two feature
    extractors want different things. Zeroing grass pixels to black helps a
    colour histogram enormously -- it is the entire point of the operation -- but
    it *hurts* a pretrained vision transformer, which has never seen an image
    with hard black holes punched through it and reacts to the artificial edges
    rather than to the kit. Handing each extractor the wrong one costs real
    accuracy, so neither is discarded.
    """

    track_id: int
    frame_idx: int
    #: raw torso crop, unmodified. Used by learned embedders.
    image: np.ndarray  # BGR
    #: same crop with grass pixels zeroed. Used by the colour embedder.
    masked: np.ndarray  # BGR
    #: fraction of the crop that survived grass suppression
    coverage: float
    #: mean BGR of the non-grass pixels, for the review screen's colour swatch
    mean_colour: tuple[float, float, float]


class JerseyCropExtractor:
    """Extracts grass-suppressed torso crops from person boxes."""

    def __init__(self, config: TeamClassificationConfig) -> None:
        self.cfg = config

    # -- masking ------------------------------------------------------------ #

    def grass_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Boolean mask, ``True`` where the pixel is *not* grass."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo, hi = self.cfg.grass_hue_range
        is_grass = (
            (hsv[:, :, 0] >= lo)
            & (hsv[:, :, 0] <= hi)
            & (hsv[:, :, 1] >= self.cfg.grass_sat_min)
        )
        return ~is_grass

    def _torso_box(
        self, bbox: tuple[float, float, float, float], width: int, height: int
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx1 = int(round(x1 + bw * self.cfg.jersey_side_margin_frac))
        cx2 = int(round(x2 - bw * self.cfg.jersey_side_margin_frac))
        cy1 = int(round(y1 + bh * self.cfg.jersey_top_frac))
        cy2 = int(round(y1 + bh * self.cfg.jersey_bottom_frac))
        return (
            max(0, cx1),
            max(0, cy1),
            min(width, cx2),
            min(height, cy2),
        )

    # -- extraction --------------------------------------------------------- #

    def extract(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
        track_id: int,
        frame_idx: int,
    ) -> JerseyCrop | None:
        """Return a usable crop, or ``None`` when this box cannot supply one."""
        h, w = image.shape[:2]
        cx1, cy1, cx2, cy2 = self._torso_box(bbox, w, h)
        if cx2 - cx1 < self.cfg.min_crop_px or cy2 - cy1 < self.cfg.min_crop_px:
            return None

        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        if self.cfg.remove_grass:
            mask = self.grass_mask(crop)
            coverage = float(mask.mean())
            # A torso that is mostly grass means the box is badly localised or
            # the player is heavily occluded. Either way the colour it would
            # contribute is the pitch, not the kit.
            if coverage < 0.25:
                return None
            pixels = crop[mask]
            masked = crop.copy()
            masked[~mask] = 0
        else:
            coverage = 1.0
            pixels = crop.reshape(-1, 3)
            masked = crop

        if pixels.size == 0:
            return None

        mean_colour = tuple(float(v) for v in pixels.reshape(-1, 3).mean(axis=0))
        return JerseyCrop(
            track_id=track_id,
            frame_idx=frame_idx,
            image=crop.copy(),
            masked=masked,
            coverage=coverage,
            mean_colour=mean_colour,  # type: ignore[arg-type]
        )

    # -- features ----------------------------------------------------------- #

    def colour_descriptor(self, crop: JerseyCrop, h_bins: int = 24, s_bins: int = 8) -> np.ndarray:
        """Hue-saturation histogram descriptor -- the offline fallback feature.

        Value (brightness) is deliberately excluded: it varies enormously
        between sunlit and shadowed halves of the same pitch while the kit's
        hue does not.
        """
        hsv = cv2.cvtColor(crop.masked, cv2.COLOR_BGR2HSV)
        nonzero = (crop.masked.sum(axis=2) > 0).astype(np.uint8) * 255
        hist = cv2.calcHist([hsv], [0, 1], nonzero, [h_bins, s_bins], [0, 180, 0, 256]).ravel()
        norm = np.linalg.norm(hist)
        return (hist / norm).astype(np.float32) if norm > 0 else hist.astype(np.float32)
