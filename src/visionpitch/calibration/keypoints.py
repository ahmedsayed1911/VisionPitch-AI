"""Pitch landmark detection.

Approach choice
---------------
The classical route is Hough line detection followed by geometric reasoning
about which line is which. It is attractive because it needs no training data,
and it fails badly on real broadcast footage: line contrast collapses in shadow
and in wet conditions, players occlude the lines that matter, and a partial view
of the pitch gives the reasoning step no anchor. Worst of all, the failures are
silent -- it returns a confident, wrong assignment.

A keypoint regression model predicts all 32 landmarks jointly with per-point
confidence, including *hallucinated* positions for landmarks currently off
screen. That per-point confidence is the signal the homography solver needs, and
it is what makes principled rejection possible: this pipeline can say "I cannot
calibrate this frame" instead of quietly producing a 40-metre error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger
from visionpitch.detection.base import DetectorInfo, file_sha256, resolve_device

log = get_logger("calibration.keypoints")


@dataclass(slots=True)
class KeypointObservation:
    """Detected pitch landmarks for one frame."""

    frame_idx: int
    #: (32, 2) image coordinates, in the pitch model's landmark order
    points: np.ndarray
    #: (32,) per-landmark confidence
    confidences: np.ndarray

    def confident(self, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(indices, points, confidences)`` above a confidence floor."""
        mask = self.confidences >= threshold
        indices = np.flatnonzero(mask)
        return indices, self.points[mask], self.confidences[mask]


class PitchKeypointDetector:
    """Wraps a YOLO-pose checkpoint that regresses the 32 pitch landmarks."""

    def __init__(self, config: Config, n_expected: int = 32) -> None:
        from ultralytics import YOLO

        self.config = config
        self.cfg = config.calibration
        self.device = resolve_device(config.runtime.device)
        self.half = config.runtime.half_precision and self.device.startswith("cuda")
        self.n_expected = n_expected

        self.model = YOLO(self.cfg.model_path)
        self.model.to(self.device)

        kpt_shape = getattr(getattr(self.model, "model", None), "kpt_shape", None)
        if kpt_shape is not None and int(kpt_shape[0]) != n_expected:
            raise RuntimeError(
                f"pitch model predicts {kpt_shape[0]} keypoints but the pitch "
                f"geometry defines {n_expected}; the landmark ordering contract "
                f"is broken and every homography would be wrong"
            )

        self.info = DetectorInfo(
            name="pitch_keypoints",
            weights=self.cfg.model_path,
            weights_sha256=file_sha256(self.cfg.model_path),
            classes=["pitch"],
            imgsz=self.cfg.imgsz,
            device=self.device,
            half_precision=self.half,
            extra={"n_keypoints": n_expected},
        )
        log.info("loaded pitch keypoint model %s on %s", self.cfg.model_path, self.device)

    def detect_batch(
        self, images: list[np.ndarray], frame_indices: list[int]
    ) -> list[KeypointObservation | None]:
        if not images:
            return []

        results = self.model.predict(
            images,
            imgsz=self.cfg.imgsz,
            conf=0.01,  # keypoint confidence is filtered downstream, not here
            device=self.device,
            quantize="fp16" if self.half else None,
            verbose=False,
        )

        out: list[KeypointObservation | None] = []
        for result, frame_idx in zip(results, frame_indices, strict=True):
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
                out.append(None)
                continue

            xy = keypoints.xy[0].cpu().numpy().astype(np.float64)
            if keypoints.conf is not None:
                conf = keypoints.conf[0].cpu().numpy().astype(np.float64)
            else:
                conf = np.ones(xy.shape[0], dtype=np.float64)

            if xy.shape[0] != self.n_expected:
                log.warning(
                    "frame %d: model returned %d keypoints, expected %d",
                    frame_idx,
                    xy.shape[0],
                    self.n_expected,
                )
                out.append(None)
                continue

            # A landmark predicted at exactly (0, 0) is the model's way of
            # saying "not visible"; treat it as unconfident rather than as a
            # detection in the top-left corner.
            at_origin = (np.abs(xy) < 1e-3).all(axis=1)
            conf = np.where(at_origin, 0.0, conf)

            out.append(KeypointObservation(frame_idx=frame_idx, points=xy, confidences=conf))
        return out
