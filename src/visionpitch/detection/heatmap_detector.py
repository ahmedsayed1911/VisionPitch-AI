"""Pipeline adapter for the centre-heatmap ball model.

Cross-domain tiny-ball study, Part 9.

Phase 2D established the rule this module exists to serve: a detector that wins
on a benchmark can still make the product worse, so nothing is promoted on
benchmark numbers alone. To measure a representation *downstream* it has to run
inside the real pipeline, which means presenting the same surface the existing
``BallDetector`` does.

The one honest compromise
-------------------------
The pipeline's ``Detection`` type carries a bounding box, and this model does not
predict one. Part 1 measured why: annotated width and height on an ~11 px ball
disagree with each other by 9.5% at the median, and only 19% of annotations have
w == h for what is a circular object. So the box here is **synthesised** from a
nominal ball diameter around the predicted centre, and that fact is recorded in
the detector's manifest entry rather than left for someone to discover.

Nothing downstream reads ball box extent -- possession uses the centre, the
trajectory search uses the centre, the analytics use the centre. The synthesised
box exists to satisfy a type, not to make a claim.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger
from visionpitch.common.types import BBox, Detection, ObjectClass
from visionpitch.detection.base import DetectorInfo, file_sha256, resolve_device
from visionpitch.detection.heatmap import BallHeatmapNet, HeatmapConfig, decode

log = get_logger("detection.heatmap_detector")

#: Nominal diameter for the synthesised box, in pixels. The median annotated
#: ball is 11.5 px (Roboflow) and 16.0 px (SN-GSR); 14 sits between them. It is
#: a placeholder for a type, not an estimate.
NOMINAL_BALL_PX = 14.0


class HeatmapBallDetector:
    """Drop-in replacement for ``BallDetector`` backed by centre heatmaps."""

    name = "ball_heatmap"

    def __init__(self, config: Config, checkpoint: str | Path | None = None) -> None:
        import torch

        self.config = config
        self.cfg = config.ball_detection
        self.device = resolve_device(config.runtime.device)

        path = Path(checkpoint or self.cfg.model_path)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        stored = dict(payload.get("config") or {})
        stored.pop("schema_version", None)
        self.heatmap_cfg = HeatmapConfig(**stored)
        # The pipeline's configured confidence wins over whatever the checkpoint
        # was saved with, so a run's manifest and its behaviour cannot disagree.
        if self.cfg.conf_threshold > 0:
            self.heatmap_cfg.peak_threshold = self.cfg.conf_threshold

        self.model = BallHeatmapNet(self.heatmap_cfg)
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()
        self._torch = torch

        self.info = DetectorInfo(
            name=self.name,
            weights=str(path),
            weights_sha256=file_sha256(str(path)),
            classes=["ball"],
            imgsz=self.heatmap_cfg.input_size,
            device=self.device,
            half_precision=False,
            extra={
                "representation": "centre_heatmap",
                "output_stride": self.heatmap_cfg.output_stride,
                "peak_threshold": self.heatmap_cfg.peak_threshold,
                "bbox_is_synthesised": True,
                "nominal_ball_px": NOMINAL_BALL_PX,
                "bbox_note": (
                    "this model predicts a centre and an uncertainty radius, not "
                    "a box; the reported box is a fixed-size square drawn around "
                    "the centre so it satisfies the Detection type. Do not read "
                    "ball width or height from it."
                ),
                "selected_epoch": payload.get("epoch"),
            },
        )
        log.info(
            "loaded heatmap ball detector %s on %s (stride %d, peak %.2f)",
            path, self.device, self.heatmap_cfg.output_stride,
            self.heatmap_cfg.peak_threshold,
        )

    # -- inference ------------------------------------------------------------ #

    def _letterbox(self, image: np.ndarray):
        size = self.heatmap_cfg.input_size
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        ox, oy = (size - nw) // 2, (size - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(image, (nw, nh))
        return canvas, scale, ox, oy

    def _predict(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Heatmap plus the letterbox transform needed to undo it."""
        canvas, scale, ox, oy = self._letterbox(image)
        tensor = (
            self._torch.from_numpy(
                cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
            ).float().div(255.0)[None].to(self.device)
        )
        with self._torch.no_grad():
            heatmap = self.model(tensor)[0, 0].detach().cpu().numpy()
        return heatmap, scale, ox, oy

    def detect(
        self,
        image: np.ndarray,
        frame_idx: int,
        predicted_centre: tuple[float, float] | None = None,
        allow_tiled: bool = True,
    ) -> list[Detection]:
        """Full-frame centre detection.

        ``predicted_centre`` and ``allow_tiled`` are accepted for interface
        compatibility and deliberately ignored: this model already sees the whole
        frame at its native resolution, so there is no ROI to prefer and no tile
        sweep to fall back to. Silently accepting them keeps the pipeline
        unchanged; silently *using* them would make the two detectors
        incomparable.
        """
        del predicted_centre, allow_tiled

        if image is None or image.size == 0:
            return []
        heatmap, scale, ox, oy = self._predict(image)

        out: list[Detection] = []
        for detection in decode(heatmap, self.heatmap_cfg):
            cx = (detection.x - ox) / scale
            cy = (detection.y - oy) / scale
            half = NOMINAL_BALL_PX / 2.0
            out.append(
                Detection(
                    frame_idx=frame_idx,
                    object_class=ObjectClass.BALL,
                    bbox=BBox(cx - half, cy - half, cx + half, cy + half),
                    confidence=float(min(1.0, max(0.0, detection.confidence))),
                    source="ball_heatmap",
                )
            )
        return out

    def detect_batch(
        self, images: list[np.ndarray], frame_indices: list[int]
    ) -> list[list[Detection]]:
        return [
            self.detect(image, frame_idx)
            for image, frame_idx in zip(images, frame_indices, strict=True)
        ]
