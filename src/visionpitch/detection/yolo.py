"""Ultralytics-backed detectors.

Two backends:

``YoloFootballDetector``
    A checkpoint fine-tuned on broadcast football that predicts
    player / goalkeeper / referee / ball directly. Class ids are resolved from
    the checkpoint's own ``names`` mapping, never hard-coded, so a replacement
    model with a different label order works unchanged.

``CocoFallbackDetector``
    Stock COCO weights, mapping ``person`` -> player and ``sports ball`` -> ball.
    Offline-capable and license-friendlier, but it cannot distinguish
    goalkeepers or referees, and its ball recall on broadcast footage is poor.
    Roles then fall to the appearance stage. Present so the pipeline degrades
    rather than fails when the football weights are unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger
from visionpitch.common.types import BBox, Detection, ObjectClass
from visionpitch.detection.base import DetectorInfo, file_sha256, resolve_device

log = get_logger("detection.yolo")

#: Label aliases seen across football checkpoints, mapped onto our class enum.
_CLASS_ALIASES: dict[str, ObjectClass] = {
    "player": ObjectClass.PLAYER,
    "players": ObjectClass.PLAYER,
    "outfield player": ObjectClass.PLAYER,
    "goalkeeper": ObjectClass.GOALKEEPER,
    "goal keeper": ObjectClass.GOALKEEPER,
    "gk": ObjectClass.GOALKEEPER,
    "referee": ObjectClass.REFEREE,
    "ref": ObjectClass.REFEREE,
    "main referee": ObjectClass.REFEREE,
    "side referee": ObjectClass.REFEREE,
    "staff members": ObjectClass.REFEREE,
    "ball": ObjectClass.BALL,
    "football": ObjectClass.BALL,
    "soccer ball": ObjectClass.BALL,
    "sports ball": ObjectClass.BALL,
}


def _load_yolo(weights: str, device: str):
    from ultralytics import YOLO

    model = YOLO(weights)
    model.to(device)
    return model


class _UltralyticsDetector:
    """Shared plumbing for the Ultralytics-based backends."""

    name = "ultralytics"

    def __init__(self, config: Config, weights: str, imgsz: int) -> None:
        self.config = config
        self.device = resolve_device(config.runtime.device)
        self.half = config.runtime.half_precision and self.device.startswith("cuda")
        self.imgsz = imgsz
        self.model = _load_yolo(weights, self.device)

        raw_names: dict[int, str] = dict(self.model.names)
        self.class_map = self._build_class_map(raw_names)
        if not self.class_map:
            raise RuntimeError(
                f"none of the checkpoint's classes {list(raw_names.values())} could be "
                f"mapped onto VisionPitch classes; add an alias in _CLASS_ALIASES"
            )

        self.info = DetectorInfo(
            name=self.name,
            weights=weights,
            weights_sha256=file_sha256(weights),
            classes=[raw_names[i] for i in sorted(raw_names)],
            imgsz=imgsz,
            device=self.device,
            half_precision=self.half,
            extra={"class_map": {k: v.value for k, v in self.class_map.items()}},
        )
        log.info(
            "loaded %s on %s (fp16=%s), classes=%s",
            weights,
            self.device,
            self.half,
            self.info.extra["class_map"],
        )

    def _build_class_map(self, raw_names: dict[int, str]) -> dict[int, ObjectClass]:
        mapping: dict[int, ObjectClass] = {}
        for idx, name in raw_names.items():
            key = str(name).strip().lower()
            if key in _CLASS_ALIASES:
                mapping[int(idx)] = _CLASS_ALIASES[key]
            else:
                log.debug("ignoring unmapped detector class %r", name)
        return mapping

    # -- inference ---------------------------------------------------------- #

    def _predict(self, images: list[np.ndarray], conf: float, **kwargs: Any):
        return self.model.predict(
            images,
            imgsz=self.imgsz,
            conf=conf,
            iou=self.config.detection.iou_threshold,
            max_det=self.config.detection.max_detections,
            device=self.device,
            quantize="fp16" if self.half else None,
            augment=self.config.detection.augment,
            verbose=False,
            **kwargs,
        )

    def _to_detections(self, result, frame_idx: int, source: str) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        overrides = self.config.detection.class_conf_overrides
        out: list[Detection] = []
        for box, conf, cls_idx in zip(xyxy, confs, classes, strict=True):
            object_class = self.class_map.get(int(cls_idx))
            if object_class is None:
                continue
            # Per-class floors let ball recall run hot while keeping person
            # precision high; a single global threshold cannot do both.
            floor = overrides.get(object_class.value)
            if floor is not None and float(conf) < floor:
                continue
            out.append(
                Detection(
                    frame_idx=frame_idx,
                    object_class=object_class,
                    bbox=BBox.from_xyxy(box),
                    confidence=float(np.clip(conf, 0.0, 1.0)),
                    source=source,
                )
            )
        return out

    def detect_batch(
        self, images: list[np.ndarray], frame_indices: list[int]
    ) -> list[list[Detection]]:
        if not images:
            return []
        if len(images) != len(frame_indices):
            raise ValueError("images and frame_indices must be the same length")

        # The lowest per-class floor is what the model must be run at; the
        # per-class floors are then applied during conversion.
        floors = [self.config.detection.conf_threshold]
        floors.extend(
            v for k, v in self.config.detection.class_conf_overrides.items()
            if any(c.value == k for c in self.class_map.values())
        )
        conf = max(0.001, min(floors))

        results = self._predict(images, conf=conf)
        return [
            self._to_detections(res, idx, self.name)
            for res, idx in zip(results, frame_indices, strict=True)
        ]


class YoloFootballDetector(_UltralyticsDetector):
    """Multiclass player / goalkeeper / referee / ball detector."""

    name = "football_multiclass"

    def __init__(self, config: Config) -> None:
        super().__init__(config, config.detection.model_path, config.detection.imgsz)
        missing = {ObjectClass.PLAYER} - set(self.class_map.values())
        if missing:
            raise RuntimeError(
                f"checkpoint {config.detection.model_path} does not predict {missing}"
            )


class CocoFallbackDetector(_UltralyticsDetector):
    """Stock COCO detector, used when football weights are unavailable.

    Reports every person as ``player``: it has no notion of goalkeeper or
    referee, so those roles must be recovered downstream from appearance and
    position, with correspondingly lower confidence.
    """

    name = "coco_fallback"

    def __init__(self, config: Config) -> None:
        super().__init__(config, config.detection.model_path, config.detection.imgsz)
        log.warning(
            "COCO fallback detector active: goalkeepers and referees are not "
            "detected as distinct classes and ball recall will be substantially "
            "lower than with football-specific weights"
        )

    def _build_class_map(self, raw_names: dict[int, str]) -> dict[int, ObjectClass]:
        mapping: dict[int, ObjectClass] = {}
        for idx, name in raw_names.items():
            key = str(name).strip().lower()
            if key == "person":
                mapping[int(idx)] = ObjectClass.PLAYER
            elif key in ("sports ball", "ball"):
                mapping[int(idx)] = ObjectClass.BALL
        return mapping


def build_detector(config: Config):
    """Instantiate the configured detection backend."""
    if config.detection.backend == "football":
        return YoloFootballDetector(config)
    if config.detection.backend == "coco":
        return CocoFallbackDetector(config)
    raise ValueError(f"unknown detection backend {config.detection.backend!r}")
