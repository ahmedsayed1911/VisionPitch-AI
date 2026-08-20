"""Appearance descriptors used for association.

Choice of descriptor
--------------------
A learned person-ReID embedding (OSNet, TransReID) is the strongest option in
general, but football is an adversarial case for it: ten outfield players per
side wear *identical* kit, so a ReID model trained to separate individuals by
clothing has almost no signal to work with. The realistic gain from appearance
in football tracking is separating the two *teams*, and that is a colour problem.

So the default descriptor here is a grass-suppressed HSV histogram over the
torso region: cheap enough to run on every detection at 25 fps, and directly
targeted at the failure mode that actually matters (a track jumping between
opposing players during a tackle or a crowd of bodies at a corner).

The interface is a protocol, so a learned embedding can be dropped in when the
extra cost is justified -- and it should be evaluated, not assumed better. See
``docs/LIMITATIONS.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class AppearanceExtractor(Protocol):
    """Turns image crops into L2-normalised feature vectors."""

    dim: int

    def extract(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """``(N, 4)`` xyxy boxes -> ``(N, dim)`` descriptors."""
        ...


class TorsoHistogramAppearance:
    """Grass-suppressed HSV histogram over the torso region of a person box."""

    def __init__(
        self,
        h_bins: int = 16,
        s_bins: int = 8,
        top_frac: float = 0.15,
        bottom_frac: float = 0.55,
        side_margin_frac: float = 0.15,
        grass_hue_range: tuple[int, int] = (30, 90),
        grass_sat_min: int = 40,
        min_crop_px: int = 6,
    ) -> None:
        self.h_bins, self.s_bins = h_bins, s_bins
        self.dim = h_bins * s_bins
        self.top_frac = top_frac
        self.bottom_frac = bottom_frac
        self.side_margin_frac = side_margin_frac
        self.grass_hue_range = grass_hue_range
        self.grass_sat_min = grass_sat_min
        self.min_crop_px = min_crop_px

    # -- region selection --------------------------------------------------- #

    def torso_region(self, box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        """Crop the shirt, excluding head, shorts and socks.

        Legs contribute the opponent's shorts colour in a tackle and the pitch
        between the legs everywhere else; the head contributes skin and hair.
        Both are noise for team separation.
        """
        x1, y1, x2, y2 = box
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx1 = int(round(x1 + bw * self.side_margin_frac))
        cx2 = int(round(x2 - bw * self.side_margin_frac))
        cy1 = int(round(y1 + bh * self.top_frac))
        cy2 = int(round(y1 + bh * self.bottom_frac))
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(width, cx2), min(height, cy2)
        return cx1, cy1, cx2, cy2

    def grass_mask(self, hsv: np.ndarray) -> np.ndarray:
        """255 where the pixel is *not* grass."""
        lo, hi = self.grass_hue_range
        is_grass = (
            (hsv[:, :, 0] >= lo)
            & (hsv[:, :, 0] <= hi)
            & (hsv[:, :, 1] >= self.grass_sat_min)
        )
        return np.where(is_grass, 0, 255).astype(np.uint8)

    # -- extraction --------------------------------------------------------- #

    def _descriptor(self, crop: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = self.grass_mask(hsv)
        # If the crop is essentially all grass the box is junk; an all-zero
        # descriptor scores zero similarity against everything, which is the
        # correct behaviour -- it contributes nothing rather than misleading.
        if mask.sum() < 255 * 8:
            return np.zeros(self.dim, dtype=np.float32)

        # Hue and saturation only. Value (brightness) is deliberately excluded:
        # the same shirt in sunlight and in the stand's shadow differs by far
        # more in V than two different kits do, so including it makes a player
        # look like a different person the moment they cross a shadow line. A
        # histogram binned on V also has a discretisation cliff -- two shirts
        # differing by ten grey levels can fall either side of a bin edge and
        # score as maximally dissimilar.
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            mask,
            [self.h_bins, self.s_bins],
            [0, 180, 0, 256],
        ).ravel()
        norm = np.linalg.norm(hist)
        return (hist / norm).astype(np.float32) if norm > 0 else np.zeros(self.dim, np.float32)

    def extract(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if boxes.size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        h, w = image.shape[:2]
        out = np.zeros((boxes.shape[0], self.dim), dtype=np.float32)
        for i, box in enumerate(boxes):
            cx1, cy1, cx2, cy2 = self.torso_region(box, w, h)
            if cx2 - cx1 < self.min_crop_px or cy2 - cy1 < self.min_crop_px:
                continue
            out[i] = self._descriptor(image[cy1:cy2, cx1:cx2])
        return out


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(N, D)`` x ``(M, D)`` -> pairwise cosine distance.

    A zero vector means the crop contained no usable appearance evidence.  Its
    pairwise entries are NaN so association can abstain from the appearance
    term instead of treating missing evidence as maximum dissimilarity.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if a.size == 0 or b.size == 0:
        return np.ones((a.shape[0], b.shape[0]), dtype=np.float32)

    sim = a @ b.T
    usable = (np.linalg.norm(a, axis=1) > 0)[:, None] & (np.linalg.norm(b, axis=1) > 0)[None, :]
    return np.where(usable, np.clip(1.0 - sim, 0.0, 1.0), np.nan).astype(np.float32)


class ExponentialFeatureBank:
    """Per-track smoothed appearance, updated only on confident observations.

    A track's descriptor is an exponential moving average rather than the most
    recent crop. During a partial occlusion the visible crop is half an opponent,
    and letting that overwrite the track's appearance is precisely how an ID
    switch becomes permanent.
    """

    def __init__(self, momentum: float = 0.9, min_confidence: float = 0.5) -> None:
        self.momentum = momentum
        self.min_confidence = min_confidence
        self._features: dict[int, np.ndarray] = {}

    def update(self, track_id: int, feature: np.ndarray, confidence: float) -> None:
        if feature is None or not np.any(feature):
            return
        if confidence < self.min_confidence and track_id in self._features:
            return
        current = self._features.get(track_id)
        if current is None:
            smoothed = feature.astype(np.float32)
        else:
            smoothed = self.momentum * current + (1.0 - self.momentum) * feature
        norm = np.linalg.norm(smoothed)
        self._features[track_id] = smoothed / norm if norm > 0 else smoothed

    def get(self, track_id: int, dim: int) -> np.ndarray:
        return self._features.get(track_id, np.zeros(dim, dtype=np.float32))

    def drop(self, track_id: int) -> None:
        self._features.pop(track_id, None)
