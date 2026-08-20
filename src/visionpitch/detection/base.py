"""Detector interface.

Keeping detection behind a narrow protocol is what allows the AGPL-licensed
checkpoints this project ships with to be swapped for differently-licensed or
better-performing ones without touching tracking, calibration or analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from visionpitch.common.types import Detection


@dataclass
class DetectorInfo:
    """Provenance recorded in the run manifest."""

    name: str
    weights: str
    weights_sha256: str | None
    classes: list[str]
    imgsz: int
    device: str
    half_precision: bool
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "weights": self.weights,
            "weights_sha256": self.weights_sha256,
            "classes": self.classes,
            "imgsz": self.imgsz,
            "device": self.device,
            "half_precision": self.half_precision,
            **self.extra,
        }


@runtime_checkable
class Detector(Protocol):
    """Stateless per-frame object detector.

    Implementations take a batch of BGR images and return one detection list per
    image, in the same order. Batching is part of the interface rather than an
    optimisation detail because GPU utilisation on a 25-minute clip depends on it.
    """

    info: DetectorInfo

    def detect_batch(
        self, images: list[np.ndarray], frame_indices: list[int]
    ) -> list[list[Detection]]:
        ...


def resolve_device(requested: str) -> str:
    """Map ``'auto'`` onto the best available device."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str | None:
    """Checkpoint hash for the manifest, so results trace to exact weights."""
    import hashlib
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    hasher = hashlib.sha256()
    with p.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()
