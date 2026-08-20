"""Projective geometry helpers shared by calibration, game state and evaluation."""

from __future__ import annotations

import numpy as np


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """``(N, 2)`` -> ``(N, 3)`` by appending a ones column."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project ``(N, 2)`` points through a 3x3 homography.

    Points whose projective weight is degenerate (on or behind the horizon) are
    returned as NaN rather than as a huge finite number, so callers must handle
    them explicitly instead of silently consuming a nonsense coordinate.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {H.shape}")

    pts = to_homogeneous(points)
    projected = pts @ H.T
    w = projected[:, 2]

    out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    safe = np.abs(w) > 1e-8
    out[safe] = projected[safe, :2] / w[safe, None]
    return out


def invert_homography(H: np.ndarray) -> np.ndarray | None:
    """Inverse homography, or ``None`` when the matrix is singular."""
    try:
        inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inv)):
        return None
    return inv


def reprojection_errors(
    H: np.ndarray, source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """Per-point Euclidean error of ``H @ source`` against ``target``.

    Units are those of ``target_points`` -- when the homography maps image
    pixels to pitch metres, the returned errors are in metres, which is the
    quantity a football analyst can actually reason about.
    """
    projected = apply_homography(H, source_points)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    return np.linalg.norm(projected - target, axis=1)


def normalise_homography(H: np.ndarray) -> np.ndarray:
    """Scale a homography so ``H[2, 2] == 1``, making matrices comparable.

    Homographies are defined up to scale; averaging or interpolating them
    without fixing the scale mixes incompatible magnitudes.
    """
    H = np.asarray(H, dtype=np.float64)
    denom = H[2, 2]
    if abs(denom) < 1e-12:
        norm = np.linalg.norm(H)
        return H / norm if norm > 0 else H
    return H / denom


def homography_distance(a: np.ndarray, b: np.ndarray, image_size: tuple[int, int]) -> float:
    """Disagreement between two homographies, in target units.

    Compares where each maps a grid of probe points. Far more meaningful than a
    matrix norm, which is dominated by the scale of individual entries.

    The probes deliberately avoid the image corners and the top of the frame. On
    a tilted broadcast camera the upper image region lies at or beyond the
    horizon, where a fraction of a degree of camera pitch moves the projected
    point by kilometres. Including those points makes the metric report enormous
    instability for a homography pair that agrees to centimetres everywhere a
    player could actually stand -- which is the only place the answer matters.
    """
    w, h = image_size
    xs = np.linspace(0.1 * w, 0.9 * w, 4)
    ys = np.linspace(0.55 * h, 0.95 * h, 3)
    probes = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)

    pa, pb = apply_homography(a, probes), apply_homography(b, probes)
    valid = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
    if not valid.any():
        return float("inf")
    return float(np.linalg.norm(pa[valid] - pb[valid], axis=1).mean())


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``(N, 4)`` / ``(M, 4)`` xyxy arrays."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that tolerates NaNs and short edges.

    Used for trajectory smoothing where dropping NaN frames outright would
    shift every subsequent sample in time.
    """
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return values.copy()

    half = window // 2
    out = np.full_like(values, np.nan)
    for i in range(values.shape[0]):
        lo, hi = max(0, i - half), min(values.shape[0], i + half + 1)
        chunk = values[lo:hi]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            out[i] = finite.mean()
    return out
