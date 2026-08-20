"""Jersey number recognition.

**Status: experimental, disabled by default.** The interfaces and the data model
are complete and stable -- ``Track.jersey_number``, its confidence, the stored
alternatives and the ``display_name`` fallback all exist and are exercised by
the rest of the pipeline. The recogniser itself is not accurate enough on
broadcast-resolution footage to be enabled without review, and the brief is
explicit that a number must never be invented when confidence is insufficient.

Why it is hard here
-------------------
In a wide broadcast shot a jersey number occupies roughly 12x16 pixels, is
printed on a curved deforming surface, is frequently rotated, and is visible
only when the player faces away from the camera. Single-frame OCR accuracy in
that regime is poor. The design therefore never trusts one frame.

Pipeline
--------
1. **Best-view selection** -- rank a track's crops by resolution, sharpness and
   how frontal/rear-facing the torso appears, and keep only the top N.
2. **Preprocessing** -- upscale, contrast-normalise, deskew.
3. **OCR** -- read digits with per-character confidence.
4. **Track-level voting** -- accumulate confidence-weighted votes across frames
   and assign only above a threshold, keeping the runners-up as alternatives.

Step 4 is the important one: the number is a property of the *track*, not of any
frame.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from visionpitch.common.config import Config
from visionpitch.common.logging import StageCounters, get_logger
from visionpitch.common.types import ObjectClass, Track
from visionpitch.team_classification.crops import JerseyCrop, JerseyCropExtractor

log = get_logger("reid.jersey")

_DIGITS = re.compile(r"\d{1,2}")


@dataclass
class JerseyReading:
    """Track-level jersey number result, with everything needed to audit it."""

    track_id: int
    number: int | None
    confidence: float
    n_votes: int
    #: runner-up candidates, ``[(number, vote_share), ...]``, best first
    alternatives: list[tuple[int, float]] = field(default_factory=list)
    #: frames that contributed to the winning vote
    supporting_frames: list[int] = field(default_factory=list)
    status: str = "unknown"  # "assigned" | "ambiguous" | "unknown"

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "number": self.number,
            "confidence": round(self.confidence, 4),
            "n_votes": self.n_votes,
            "alternatives": [(n, round(s, 4)) for n, s in self.alternatives],
            "supporting_frames": self.supporting_frames,
            "status": self.status,
        }


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian: high for crisp crops, near zero for blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def view_score(crop: JerseyCrop) -> float:
    """Rank a crop's suitability for reading a number.

    Combines size, sharpness and jersey coverage. Size dominates: a large blurry
    crop can still be read, a sharp 8-pixel one cannot.
    """
    h, w = crop.image.shape[:2]
    area_term = float(np.clip((w * h) / 4000.0, 0.0, 1.0))
    sharp_term = float(np.clip(sharpness(crop.image) / 500.0, 0.0, 1.0))
    return 0.55 * area_term + 0.30 * sharp_term + 0.15 * crop.coverage


class JerseyNumberRecogniser:
    """Temporal, track-level jersey number recognition."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cfg = config.reid
        self.extractor = JerseyCropExtractor(config.team_classification)
        self.counters = StageCounters("reid")
        self._reader = None

    # -- OCR backend -------------------------------------------------------- #

    def _ocr(self):
        """Lazily construct the OCR backend. ``None`` when unavailable."""
        if self._reader is not None:
            return self._reader
        try:
            import easyocr

            use_gpu = self.config.runtime.device != "cpu"
            self._reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            log.info("jersey OCR backend ready (easyocr, gpu=%s)", use_gpu)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "jersey OCR unavailable (%s); install the 'ocr' extra to enable it. "
                "Tracks will keep their fallback identities.",
                exc,
            )
            self._reader = None
        return self._reader

    # -- preprocessing ------------------------------------------------------ #

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Upscale and contrast-normalise a jersey crop for OCR."""
        scale = max(1, self.cfg.upscale_factor)
        upscaled = cv2.resize(
            image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        # CLAHE rather than global equalisation: a number in shadow on a bright
        # shirt is a local contrast problem.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def read_crop(self, crop: JerseyCrop) -> list[tuple[int, float]]:
        """Read candidate numbers from one crop as ``[(number, confidence)]``."""
        reader = self._ocr()
        if reader is None:
            return []

        prepared = self.preprocess(crop.image)
        try:
            results = reader.readtext(prepared, allowlist="0123456789", detail=1)
        except Exception as exc:  # noqa: BLE001
            log.debug("OCR failed on a crop: %s", exc)
            return []

        out: list[tuple[int, float]] = []
        for _, text, confidence in results:
            if confidence < self.cfg.min_digit_confidence:
                continue
            for match in _DIGITS.findall(str(text)):
                number = int(match)
                # Squad numbers run 1-99; a leading-zero read is an OCR artefact.
                if 1 <= number <= 99:
                    out.append((number, float(confidence)))
        return out

    # -- track-level -------------------------------------------------------- #

    def recognise(
        self, frames: dict[int, np.ndarray], tracks: dict[int, Track]
    ) -> dict[int, JerseyReading]:
        """Read numbers from cached frames. Convenience for tests and tools."""
        crops_by_track: dict[int, list[JerseyCrop]] = defaultdict(list)
        for track in tracks.values():
            for obs in track.observations:
                if obs.interpolated:
                    continue
                image = frames.get(obs.frame_idx)
                if image is None:
                    continue
                crop = self.extractor.extract(
                    image, obs.bbox.to_xyxy(), track.track_id, obs.frame_idx
                )
                if crop is not None:
                    crops_by_track[track.track_id].append(crop)
        return self.recognise_from_crops(crops_by_track, tracks)

    def recognise_from_crops(
        self, crops_by_track: dict[int, list[JerseyCrop]], tracks: dict[int, Track]
    ) -> dict[int, JerseyReading]:
        """Assign a jersey number to each player track, or leave it unknown."""
        readings: dict[int, JerseyReading] = {}
        if not self.cfg.jersey_ocr_enabled:
            log.info("jersey OCR disabled by config; tracks keep fallback identities")
            return readings

        for track in tracks.values():
            if track.object_class not in (ObjectClass.PLAYER, ObjectClass.GOALKEEPER):
                continue

            crops = crops_by_track.get(track.track_id, [])
            if not crops:
                readings[track.track_id] = JerseyReading(track.track_id, None, 0.0, 0)
                continue

            best = sorted(crops, key=view_score, reverse=True)[: self.cfg.crops_per_track]
            reading = self._vote(track.track_id, best)
            readings[track.track_id] = reading

            if reading.status == "assigned":
                track.jersey_number = reading.number
                track.jersey_confidence = reading.confidence
                self.counters.ok()
            else:
                # Explicitly leave the number unset. The display name falls back
                # to the stable track-based identity.
                track.jersey_number = None
                track.jersey_confidence = 0.0
                self.counters.warn(f"jersey_{reading.status}")

        assigned = sum(1 for r in readings.values() if r.status == "assigned")
        log.info("jersey numbers: %d assigned of %d player tracks", assigned, len(readings))
        return readings

    def _vote(self, track_id: int, crops: list[JerseyCrop]) -> JerseyReading:
        votes: dict[int, float] = defaultdict(float)
        frames_for: dict[int, list[int]] = defaultdict(list)
        n_votes = 0

        for crop in crops:
            for number, confidence in self.read_crop(crop):
                # Weight each vote by both OCR confidence and how good a view it
                # came from, so one lucky read on a tiny blurred crop cannot
                # outvote several clean ones.
                votes[number] += confidence * view_score(crop)
                frames_for[number].append(crop.frame_idx)
                n_votes += 1

        if not votes:
            return JerseyReading(track_id, None, 0.0, 0, status="unknown")

        total = sum(votes.values())
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        winner, winner_weight = ranked[0]
        share = winner_weight / total if total > 0 else 0.0

        alternatives = [(n, w / total) for n, w in ranked[1:4]]

        if n_votes < self.cfg.min_votes or share < self.cfg.vote_threshold:
            return JerseyReading(
                track_id,
                None,
                float(share),
                n_votes,
                alternatives=[(winner, share), *alternatives],
                supporting_frames=frames_for[winner],
                status="ambiguous",
            )

        return JerseyReading(
            track_id,
            winner,
            float(share),
            n_votes,
            alternatives=alternatives,
            supporting_frames=frames_for[winner],
            status="assigned",
        )
