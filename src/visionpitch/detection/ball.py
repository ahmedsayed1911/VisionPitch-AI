"""Specialist high-resolution ball detector.

Why a second detector at all
----------------------------
On its own published test split the multiclass football checkpoint scores
ball mAP50-95 = 0.338, while the dedicated ball checkpoint scores 0.551. The
gap is a resolution problem more than a capacity one: at 1280px inference a
broadcast ball is often 6-10px across, and after the backbone's stride-32
downsampling it survives as a fraction of one feature cell.

Strategy
--------
Rather than running a second full-frame pass at 4K (which would dominate
runtime), this detector spends its resolution budget where the ball actually is:

* **ROI mode** - crop a window around the predicted ball position and run the
  specialist model on it. The crop is small, so it is fed to the model at close
  to native resolution, which is the entire point.
* **Tiled sweep** - when the ball's position is unknown, sweep overlapping tiles
  across the frame. This is the expensive path, so it is rate-limited by
  ``tiled_every_n_frames``.

The detector is deliberately *stateless* about football: it is told where to
look by the trajectory estimator and returns candidates with confidences. All
temporal reasoning lives in ``ball_tracking``.
"""

from __future__ import annotations

import numpy as np

from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger
from visionpitch.common.types import BBox, Detection, ObjectClass
from visionpitch.detection.base import DetectorInfo, file_sha256, resolve_device

log = get_logger("detection.ball")


def build_ball_detector(config: Config):
    """The ball detector named by ``ball_detection.representation``.

    Kept as a factory rather than a branch at the call site so the pipeline has
    exactly one place that knows a second representation exists, and so the
    default path is unchanged for every existing run.
    """
    if config.ball_detection.representation == "heatmap":
        from visionpitch.detection.heatmap_detector import HeatmapBallDetector

        return HeatmapBallDetector(config)
    return BallDetector(config)


def _greedy_nms(detections: list[Detection], iou_threshold: float = 0.4) -> list[Detection]:
    """Confidence-ordered NMS. Needed because tiles overlap by design."""
    if len(detections) <= 1:
        return detections
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if all(candidate.bbox.iou(k.bbox) <= iou_threshold for k in kept):
            kept.append(candidate)
    return kept


class BallDetector:
    """ROI-guided and tiled ball detection."""

    name = "ball_specialist"

    def __init__(self, config: Config) -> None:
        from ultralytics import YOLO

        self.config = config
        self.cfg = config.ball_detection
        self.device = resolve_device(config.runtime.device)
        self.half = config.runtime.half_precision and self.device.startswith("cuda")

        self.model = YOLO(self.cfg.model_path)
        self.model.to(self.device)

        names = dict(self.model.names)
        self.ball_class_ids = {
            int(i) for i, n in names.items()
            if str(n).strip().lower() in ("ball", "football", "soccer ball", "sports ball")
        }
        if not self.ball_class_ids:
            # Single-class checkpoints sometimes label the class something odd;
            # with exactly one class there is no ambiguity about what it means.
            if len(names) == 1:
                self.ball_class_ids = {int(next(iter(names)))}
                log.warning(
                    "ball checkpoint's only class is %r; treating it as the ball",
                    next(iter(names.values())),
                )
            else:
                raise RuntimeError(
                    f"ball checkpoint {self.cfg.model_path} has no ball-like class "
                    f"among {list(names.values())}"
                )

        self.info = DetectorInfo(
            name=self.name,
            weights=self.cfg.model_path,
            weights_sha256=file_sha256(self.cfg.model_path),
            classes=[names[i] for i in sorted(names)],
            imgsz=self.cfg.imgsz,
            device=self.device,
            half_precision=self.half,
            extra={
                "roi_size_px": self.cfg.roi_size_px,
                "tiled_fallback": self.cfg.tiled_fallback,
                "tiles": f"{self.cfg.tile_rows}x{self.cfg.tile_cols}",
            },
        )
        log.info("loaded ball detector %s on %s", self.cfg.model_path, self.device)

    # -- crop helpers ------------------------------------------------------- #

    @staticmethod
    def _clamped_window(
        cx: float, cy: float, size: int, width: int, height: int
    ) -> tuple[int, int, int, int]:
        """A ``size``x``size`` window centred on (cx, cy), shifted to stay in frame.

        Shifting rather than truncating keeps the window at full size, which
        keeps the effective inference scale constant near the frame edges.
        """
        half = size // 2
        x1 = int(round(cx)) - half
        y1 = int(round(cy)) - half
        x1 = max(0, min(x1, max(0, width - size)))
        y1 = max(0, min(y1, max(0, height - size)))
        x2 = min(width, x1 + size)
        y2 = min(height, y1 + size)
        return x1, y1, x2, y2

    def _run(self, crops: list[np.ndarray]) -> list:
        return self.model.predict(
            crops,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf_threshold,
            iou=0.5,
            max_det=8,
            device=self.device,
            quantize="fp16" if self.half else None,
            verbose=False,
        )

    def _decode(
        self, result, frame_idx: int, offset: tuple[int, int], source: str
    ) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        ox, oy = offset
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        out = []
        for box, conf, cls_idx in zip(xyxy, confs, classes, strict=True):
            if int(cls_idx) not in self.ball_class_ids:
                continue
            out.append(
                Detection(
                    frame_idx=frame_idx,
                    object_class=ObjectClass.BALL,
                    bbox=BBox(
                        float(box[0]) + ox,
                        float(box[1]) + oy,
                        float(box[2]) + ox,
                        float(box[3]) + oy,
                    ),
                    confidence=float(np.clip(conf, 0.0, 1.0)),
                    source=source,
                )
            )
        return out

    # -- public API --------------------------------------------------------- #

    def detect_roi(
        self, image: np.ndarray, frame_idx: int, centre: tuple[float, float]
    ) -> list[Detection]:
        """Run the specialist model on a window around a predicted position."""
        h, w = image.shape[:2]
        size = min(self.cfg.roi_size_px, w, h)
        x1, y1, x2, y2 = self._clamped_window(centre[0], centre[1], size, w, h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        results = self._run([crop])
        return self._decode(results[0], frame_idx, (x1, y1), "ball_roi")

    def detect_tiled(self, image: np.ndarray, frame_idx: int) -> list[Detection]:
        """Sweep overlapping tiles across the whole frame."""
        h, w = image.shape[:2]
        rows, cols = self.cfg.tile_rows, self.cfg.tile_cols
        overlap = self.cfg.tile_overlap

        tile_w = int(np.ceil(w / cols))
        tile_h = int(np.ceil(h / rows))
        pad_x = int(tile_w * overlap / 2)
        pad_y = int(tile_h * overlap / 2)

        crops: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                x1 = max(0, c * tile_w - pad_x)
                y1 = max(0, r * tile_h - pad_y)
                x2 = min(w, (c + 1) * tile_w + pad_x)
                y2 = min(h, (r + 1) * tile_h + pad_y)
                crop = image[y1:y2, x1:x2]
                if crop.size:
                    crops.append(crop)
                    offsets.append((x1, y1))

        if not crops:
            return []

        results = self._run(crops)
        detections: list[Detection] = []
        for result, offset in zip(results, offsets, strict=True):
            detections.extend(self._decode(result, frame_idx, offset, "ball_tiled"))
        return _greedy_nms(detections)

    def detect(
        self,
        image: np.ndarray,
        frame_idx: int,
        predicted_centre: tuple[float, float] | None,
        allow_tiled: bool = True,
    ) -> list[Detection]:
        """Detect the ball, preferring the cheap ROI path when possible.

        ``predicted_centre`` comes from the trajectory estimator's forward
        prediction. When it is ``None`` the ball's whereabouts are unknown and
        only a sweep can find it.
        """
        if predicted_centre is not None:
            found = self.detect_roi(image, frame_idx, predicted_centre)
            if found:
                return found

        # The ROI found nothing, so the ball's whereabouts are unknown and only
        # a sweep can find it. Rate-limiting the sweep here was measurably
        # expensive: with the limiter at every 3rd frame, two thirds of the
        # frames where the ROI missed got no second look at all, and the
        # pipeline saw ball candidates in 64.8% of frames against the detector's
        # 91.2% recall on stills. The sweep is a nano model over six small
        # tiles; skipping it is not where the runtime budget should be spent.
        if not (self.cfg.tiled_fallback and allow_tiled):
            return []
        if self.cfg.tiled_every_n_frames > 1 and frame_idx % self.cfg.tiled_every_n_frames:
            return []
        return self.detect_tiled(image, frame_idx)
