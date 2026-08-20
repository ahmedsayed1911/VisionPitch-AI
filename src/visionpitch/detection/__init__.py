"""Object detection: multiclass football detector plus a specialist ball pass."""

from visionpitch.detection.ball import BallDetector
from visionpitch.detection.base import Detector, DetectorInfo
from visionpitch.detection.fusion import fuse_detections
from visionpitch.detection.yolo import CocoFallbackDetector, YoloFootballDetector, build_detector

__all__ = [
    "BallDetector",
    "CocoFallbackDetector",
    "Detector",
    "DetectorInfo",
    "YoloFootballDetector",
    "build_detector",
    "fuse_detections",
]
