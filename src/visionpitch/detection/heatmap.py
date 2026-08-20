"""Centre-heatmap ball localisation.

Cross-domain tiny-ball study, Part 4.

Why a heatmap rather than a box
-------------------------------
Part 1 measured that the ball is a median 11.5 x 11.6 px object whose annotated
width and height disagree by 9.5% at the median -- a circle recorded as a
rectangle, with roughly a pixel of quantisation noise on each side. Part 3 then
measured that the existing box detector, *when it fires*, localises the centre to
a median of 1.44 px. Its centre-recall curve is almost flat between a 5 px and a
25 px tolerance (0.452 -> 0.525).

Those two facts together say something specific: box regression is not costing
us localisation, it is costing us **detection**. So the case for a heatmap is not
"it will localise better" -- it already cannot much -- but that dense per-pixel
supervision with a penalty-reduced focal loss handles the extreme
foreground/background imbalance of an 11 px object better than anchor
assignment does, and that dropping the noisy width/height regression removes a
target that is partly noise.

That is a testable mechanism, and this module exists to test it rather than to
assume it.

What the output is
------------------
A centre, a confidence, and an **uncertainty radius** derived from the sharpness
of the peak. No width or height: Part 1 showed those carry roughly a pixel of
real signal at this scale, and the downstream consumer never reads them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visionpitch.common.logging import get_logger

log = get_logger("detection.heatmap")

HEATMAP_SCHEMA_VERSION = "1.0.0"


@dataclass
class HeatmapConfig:
    """Architecture and target parameters.

    ``output_stride`` is 2, not the 4 that CenterNet-style detectors normally
    use. At stride 4 an 11 px ball occupies under three output cells and its
    Gaussian collapses to a single pixel, which destroys the sub-pixel
    information that soft-argmax would otherwise recover.

    ``min_sigma`` floors the target Gaussian. A sigma derived purely from ball
    size would be under half a cell for the smallest balls, giving a target that
    is one hot pixel surrounded by zeros -- the degenerate case focal loss is
    worst at.
    """

    input_size: int = 640
    output_stride: int = 2
    base_channels: int = 16
    #: target Gaussian sigma = max(min_sigma, ball_radius_cells * sigma_scale)
    sigma_scale: float = 0.55
    min_sigma: float = 1.0
    #: focal loss shape, following the penalty-reduced formulation
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    #: peaks below this are not reported
    peak_threshold: float = 0.25
    #: non-maximum suppression window on the heatmap, in output cells
    nms_kernel: int = 5
    #: window used for sub-pixel soft-argmax around each peak, in cells
    refine_window: int = 3

    def to_dict_kwargs(self) -> dict:
        """Field values only, suitable for reconstructing a modified copy."""
        from dataclasses import asdict

        return asdict(self)

    def to_dict(self) -> dict:
        return {
            "schema_version": HEATMAP_SCHEMA_VERSION,
            "input_size": self.input_size,
            "output_stride": self.output_stride,
            "base_channels": self.base_channels,
            "sigma_scale": self.sigma_scale,
            "min_sigma": self.min_sigma,
            "focal_alpha": self.focal_alpha,
            "focal_beta": self.focal_beta,
            "peak_threshold": self.peak_threshold,
            "nms_kernel": self.nms_kernel,
            "refine_window": self.refine_window,
        }


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


def render_target(
    centres: list[tuple[float, float]],
    sizes: list[float],
    output_hw: tuple[int, int],
    stride: int,
    config: HeatmapConfig,
) -> np.ndarray:
    """Penalty-reduced Gaussian target.

    The peak is set to exactly 1.0 at the nearest cell and the Gaussian decays
    around it. Values are combined with ``maximum`` rather than addition, so two
    nearby balls cannot sum to a super-unit target that the focal loss would
    then treat as impossible to reach.
    """
    height, width = output_hw
    target = np.zeros((height, width), dtype=np.float32)

    for (cx, cy), size in zip(centres, sizes, strict=True):
        x = cx / stride
        y = cy / stride
        if not (0 <= x < width and 0 <= y < height):
            continue
        radius_cells = max(0.5, (size / 2.0) / stride)
        sigma = max(config.min_sigma, radius_cells * config.sigma_scale)

        extent = int(math.ceil(3 * sigma))
        x0, x1 = max(0, int(x) - extent), min(width, int(x) + extent + 1)
        y0, y1 = max(0, int(y) - extent), min(height, int(y) + extent + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        ys, xs = np.mgrid[y0:y1, x0:x1]
        gaussian = np.exp(
            -((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma * sigma)
        ).astype(np.float32)
        np.maximum(target[y0:y1, x0:x1], gaussian, out=target[y0:y1, x0:x1])

        # Guarantee an exact 1.0 somewhere, so every ball has a positive cell
        # even when it lands between cells and the Gaussian peaks below one.
        target[min(height - 1, int(round(y))), min(width - 1, int(round(x)))] = 1.0

    return target


def focal_loss(
    prediction: torch.Tensor, target: torch.Tensor, config: HeatmapConfig
) -> torch.Tensor:
    """Penalty-reduced focal loss (CornerNet / CenterNet formulation).

    Ordinary focal loss treats every non-peak cell as equally negative. Here a
    cell near the true centre is penalised *less* in proportion to its target
    value, which matters enormously for an 11 px object where "one cell off" is
    a good answer, not a wrong one.
    """
    prediction = prediction.clamp(1e-4, 1 - 1e-4)
    positive = target.ge(1.0).float()
    negative = 1.0 - positive

    positive_loss = (
        torch.log(prediction)
        * torch.pow(1 - prediction, config.focal_alpha)
        * positive
    )
    negative_loss = (
        torch.log(1 - prediction)
        * torch.pow(prediction, config.focal_alpha)
        * torch.pow(1 - target, config.focal_beta)
        * negative
    )

    n_positive = positive.sum()
    total = -(positive_loss.sum() + negative_loss.sum())
    # With no ball in the batch, normalising by zero positives would explode.
    return total / n_positive if n_positive > 0 else -negative_loss.sum()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class BallHeatmapNet(nn.Module):
    """A small U-Net that keeps a high-resolution path for tiny objects.

    Deliberately shallow. The ball is 11 px, so a deep encoder that reaches a
    stride-32 bottleneck has thrown the object away entirely by its deepest
    layer; the useful evidence lives in the first two or three stages. Capacity
    is spent on resolution rather than depth.
    """

    def __init__(self, config: HeatmapConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or HeatmapConfig()
        c = self.cfg.base_channels

        self.enc1 = conv_block(3, c)          # stride 1
        self.enc2 = conv_block(c, c * 2)      # stride 2
        self.enc3 = conv_block(c * 2, c * 4)  # stride 4
        self.enc4 = conv_block(c * 4, c * 8)  # stride 8
        self.pool = nn.MaxPool2d(2)

        self.dec3 = conv_block(c * 8 + c * 4, c * 4)
        self.dec2 = conv_block(c * 4 + c * 2, c * 2)

        self.head = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, 1, 1),
        )
        # Bias the head so training starts predicting a low probability
        # everywhere. Without it the first steps are dominated by the ~1e6
        # background cells and the model collapses to all-zero.
        nn.init.constant_(self.head[-1].bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                     # stride 1
        e2 = self.enc2(self.pool(e1))         # stride 2
        e3 = self.enc3(self.pool(e2))         # stride 4
        e4 = self.enc4(self.pool(e3))         # stride 8

        d3 = self.dec3(
            torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="nearest"), e3], 1)
        )
        d2 = self.dec2(
            torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="nearest"), e2], 1)
        )
        return torch.sigmoid(self.head(d2))   # stride 2


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


@dataclass
class HeatmapDetection:
    """One predicted ball centre, in input-image pixels."""

    x: float
    y: float
    confidence: float
    #: 1-sigma positional uncertainty, from the sharpness of the peak
    uncertainty_px: float

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "confidence": round(self.confidence, 4),
            "uncertainty_px": round(self.uncertainty_px, 2),
        }


def decode(
    heatmap: np.ndarray, config: HeatmapConfig, max_detections: int = 5
) -> list[HeatmapDetection]:
    """Peaks with sub-pixel refinement and an uncertainty estimate.

    Sub-pixel refinement is a soft-argmax over a small window rather than the
    integer peak. At stride 2 the integer peak alone quantises every prediction
    to 2 px, which would put a floor under centre error well above the 1.44 px
    the box detector already achieves.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"expected a 2-D heatmap, got shape {heatmap.shape}")

    tensor = torch.from_numpy(heatmap)[None, None]
    pooled = F.max_pool2d(
        tensor, config.nms_kernel, stride=1, padding=config.nms_kernel // 2
    )
    peaks = (tensor == pooled).float() * tensor
    peaks = peaks[0, 0].numpy()

    height, width = peaks.shape
    flat = peaks.reshape(-1)
    order = np.argsort(-flat)[:max_detections]

    out: list[HeatmapDetection] = []
    for index in order:
        score = float(flat[index])
        if score < config.peak_threshold:
            break
        py, px = divmod(int(index), width)

        window = config.refine_window
        y0, y1 = max(0, py - window), min(height, py + window + 1)
        x0, x1 = max(0, px - window), min(width, px + window + 1)
        patch = heatmap[y0:y1, x0:x1].astype(np.float64)
        weight = patch.sum()
        if weight <= 0:
            refined_x, refined_y, spread = float(px), float(py), float(window)
        else:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            refined_x = float((xs * patch).sum() / weight)
            refined_y = float((ys * patch).sum() / weight)
            variance = (
                ((xs - refined_x) ** 2 + (ys - refined_y) ** 2) * patch
            ).sum() / weight
            spread = float(np.sqrt(max(variance, 1e-6) / 2.0))

        out.append(
            HeatmapDetection(
                x=refined_x * config.output_stride,
                y=refined_y * config.output_stride,
                confidence=score,
                uncertainty_px=spread * config.output_stride,
            )
        )
    return out
