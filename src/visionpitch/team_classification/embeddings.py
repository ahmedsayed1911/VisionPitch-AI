"""Appearance embeddings for team discovery.

Two backends, both behind the same interface.

``ColourEmbedder`` -- the default
    Hue-saturation histograms over the grass-suppressed torso.

``SiglipEmbedder``
    A pretrained vision transformer over the raw torso crop.

Which one is actually better, measured
--------------------------------------
The intuition that a learned embedding beats a colour histogram is wrong for
broadcast football, and this was measured rather than assumed. On the U-17
validation clip (white kit vs red kit), fitting two clusters over 600 harvested
crops gave:

===================  ==================  ===================
backend              silhouette score    tracks left UNKNOWN
===================  ==================  ===================
colour histogram     0.501               39 of 117
SigLIP ViT           0.237               45 of 117
===================  ==================  ===================

The reason is resolution. At broadcast distance a torso crop is roughly
20x35 px. Upscaling that to the 224x224 the ViT expects produces an image that
is mostly interpolation blur, so its features end up dominated by pose, motion
blur and background rather than by the kit. A hue histogram computed over the
very same pixels keeps precisely the signal that separates two kits and throws
away everything else.

SigLIP is kept, not deleted, because the conditions that favour it are real and
foreseeable: kits that differ in *pattern* rather than hue (hoops vs plain in
similar colours), and larger crops from tactical-camera or close-range footage.
Switch with ``team_classification.method``, and check the reported silhouette
score before trusting either.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger
from visionpitch.team_classification.crops import JerseyCrop, JerseyCropExtractor

log = get_logger("team_classification.embeddings")


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, crops: list[JerseyCrop]) -> np.ndarray:
        ...


class ColourEmbedder:
    """Hue-saturation histogram features."""

    name = "colour_histogram"

    def __init__(self, extractor: JerseyCropExtractor, h_bins: int = 24, s_bins: int = 8) -> None:
        self.extractor = extractor
        self.h_bins, self.s_bins = h_bins, s_bins
        self.dim = h_bins * s_bins

    def embed(self, crops: list[JerseyCrop]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(
            [self.extractor.colour_descriptor(c, self.h_bins, self.s_bins) for c in crops]
        )


class SiglipEmbedder:
    """Pretrained SigLIP vision tower, pooled and L2-normalised."""

    name = "siglip"

    def __init__(self, config: Config) -> None:
        import torch
        from transformers import SiglipImageProcessor, SiglipVisionModel

        from visionpitch.detection.base import resolve_device

        model_id = config.team_classification.embedding_model
        self.device = resolve_device(config.runtime.device)
        self.torch = torch

        self.processor = SiglipImageProcessor.from_pretrained(model_id)
        self.model = SiglipVisionModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)
        self.batch_size = max(8, config.runtime.batch_size)
        log.info("loaded %s on %s (dim=%d)", model_id, self.device, self.dim)

    def embed(self, crops: list[JerseyCrop]) -> np.ndarray:
        import cv2

        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)

        out = np.zeros((len(crops), self.dim), dtype=np.float32)
        for start in range(0, len(crops), self.batch_size):
            batch = crops[start : start + self.batch_size]
            # The *raw* crop, not the grass-masked one: see JerseyCrop's docstring.
            images = [cv2.cvtColor(c.image, cv2.COLOR_BGR2RGB) for c in batch]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                features = self.model(**inputs).pooler_output
            features = features.float().cpu().numpy()
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            out[start : start + len(batch)] = np.divide(
                features, norms, out=np.zeros_like(features), where=norms > 0
            )
        return out


def build_embedder(config: Config, extractor: JerseyCropExtractor) -> Embedder:
    """Instantiate the configured embedder, degrading rather than failing.

    A missing model download or absent GPU must not abort a run: the colour
    fallback is worse but usable, and the substitution is logged loudly and
    recorded in the manifest so no one mistakes one for the other.
    """
    if config.team_classification.method == "colour" or (
        config.team_classification.method == "color"
    ):
        return ColourEmbedder(extractor)

    try:
        return SiglipEmbedder(config)
    except Exception as exc:  # noqa: BLE001 - any failure here should degrade, not crash
        log.warning(
            "could not load embedding model %r (%s); falling back to colour "
            "histograms. Team separation on similar kits will be weaker.",
            config.team_classification.embedding_model,
            exc,
        )
        return ColourEmbedder(extractor)
