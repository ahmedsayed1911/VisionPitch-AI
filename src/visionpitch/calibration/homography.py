"""Homography estimation and validation.

The homography maps *image pixels* to *pitch metres* on the ground plane. Every
downstream physical quantity -- distance covered, sprint speed, the offside
line -- inherits its error directly from this matrix, so validation here is not
defensive programming, it is the accuracy ceiling of the whole product.

Three independent checks are applied, because each catches a different failure:

1. **Reprojection error**, in metres. Catches a fit that is merely imprecise.
2. **Geometric sanity** of the induced mapping. Catches a fit that is
   catastrophically wrong -- degenerate, mirrored, or folded through the horizon
   -- which can still show a low reprojection error when the keypoints used were
   nearly collinear.
3. **Inlier ratio.** Catches a fit that survived only by discarding most of its
   evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from visionpitch.common.geometry import apply_homography, reprojection_errors
from visionpitch.common.logging import get_logger
from visionpitch.pitch.geometry import PitchConfiguration

log = get_logger("calibration.homography")


@dataclass(slots=True)
class HomographyFit:
    """Result of one homography solve."""

    homography: np.ndarray | None
    n_keypoints: int
    n_inliers: int
    reprojection_error_m: float
    confidence: float
    rejection_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.homography is not None


def _jacobian_determinant(H: np.ndarray, point: np.ndarray) -> float:
    """Signed area scaling of the homography at one image point.

    A homography's global matrix determinant says nothing useful about local
    behaviour, because the mapping's scale varies across the image. The local
    Jacobian is the quantity that actually describes what happens to a small
    patch at ``point``.
    """
    x, y = float(point[0]), float(point[1])
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if abs(w) < 1e-9:
        return float("nan")
    u = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    v = H[1, 0] * x + H[1, 1] * y + H[1, 2]

    du_dx = (H[0, 0] * w - u * H[2, 0]) / (w * w)
    du_dy = (H[0, 1] * w - u * H[2, 1]) / (w * w)
    dv_dx = (H[1, 0] * w - v * H[2, 0]) / (w * w)
    dv_dy = (H[1, 1] * w - v * H[2, 1]) / (w * w)
    return float(du_dx * dv_dy - du_dy * dv_dx)


def validate_homography(
    H: np.ndarray,
    image_size: tuple[int, int],
    pitch: PitchConfiguration,
    keypoint_hull: np.ndarray | None = None,
) -> str | None:
    """Geometric plausibility check. Returns a rejection reason, or ``None``.

    Reprojection error alone is not sufficient: keypoints clustered along the
    halfway line are nearly collinear, and a homography fitted to them can
    reproduce those points almost perfectly while mapping the rest of the image
    to nonsense.

    What is deliberately *not* checked
    ----------------------------------
    The sign of the mapping's orientation. Image ``y`` increases downward while
    pitch ``y`` increases upward, so a correct homography reverses orientation --
    and whether it does depends on which touchline the camera sits behind. There
    is no fixed correct sign, so requiring one rejects valid calibrations. What
    *is* checked is that the sign stays *consistent* across the frame, because an
    inconsistent sign means the mapping folds the plane back on itself, which is
    always wrong.

    Image corners are also not projected and range-checked. On a tilted broadcast
    camera the upper corners legitimately lie beyond the horizon and project to
    enormous or infinite coordinates; that is expected geometry, not a failure.
    Validation is therefore done where the evidence is -- over the region the
    keypoints actually cover.
    """
    if H is None or not np.all(np.isfinite(H)):
        return "non_finite"
    if abs(float(np.linalg.det(H))) < 1e-12:
        return "degenerate"

    width, height = image_size

    # -- probe over the region we have evidence for -------------------------- #
    if keypoint_hull is not None and len(keypoint_hull) >= 3:
        lo = keypoint_hull.min(axis=0)
        hi = keypoint_hull.max(axis=0)
    else:
        lo = np.array([0.25 * width, 0.5 * height])
        hi = np.array([0.75 * width, 0.95 * height])

    xs = np.linspace(lo[0], hi[0], 3)
    ys = np.linspace(lo[1], hi[1], 3)
    probes = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)

    projected = apply_homography(H, probes)
    finite = np.isfinite(projected).all(axis=1)
    if finite.sum() < 4:
        return "horizon_crossing"

    # -- no fold ------------------------------------------------------------- #
    jacobians = np.array([_jacobian_determinant(H, p) for p in probes[finite]])
    usable = np.isfinite(jacobians) & (np.abs(jacobians) > 1e-12)
    if usable.sum() < 3:
        return "degenerate"
    signs = np.sign(jacobians[usable])
    if not (np.all(signs > 0) or np.all(signs < 0)):
        return "folded"

    # -- plausible physical scale -------------------------------------------- #
    # |Jacobian| is square metres of pitch per square pixel of image. A broadcast
    # frame ranges from roughly 0.01 m/px (tight) to 0.5 m/px (very wide) at the
    # near touchline; the bounds below are an order of magnitude either side.
    metres_per_px = np.sqrt(np.abs(jacobians[usable]))
    if np.median(metres_per_px) > 2.0:
        return "implausible_scale_large"
    if np.median(metres_per_px) < 1e-3:
        return "implausible_scale_small"

    # -- the evidence region must land on the pitch --------------------------- #
    # Not the image corners -- the area we actually saw landmarks in. Allowing a
    # full pitch length of slack keeps stands and run-off areas admissible.
    points = projected[finite]
    centre = np.median(points, axis=0)
    if not pitch.contains(float(centre[0]), float(centre[1]), margin=pitch.length):
        return "projects_off_pitch"

    return None


def estimate_homography(
    image_points: np.ndarray,
    pitch_indices: np.ndarray,
    pitch: PitchConfiguration,
    image_size: tuple[int, int],
    keypoint_confidences: np.ndarray | None = None,
    ransac_threshold_m: float = 2.0,
    max_reprojection_error_m: float = 3.0,
    min_keypoints: int = 5,
) -> HomographyFit:
    """Fit an image -> pitch homography from matched landmarks.

    Parameters
    ----------
    image_points:
        ``(N, 2)`` landmark positions in image pixels.
    pitch_indices:
        ``(N,)`` indices into :attr:`PitchConfiguration.vertices` naming which
        landmark each image point is.
    """
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    pitch_indices = np.asarray(pitch_indices, dtype=int).ravel()
    n = image_points.shape[0]

    if n < min_keypoints:
        return HomographyFit(None, n, 0, float("inf"), 0.0, "too_few_keypoints")

    world_points = pitch.vertices[pitch_indices]

    # Reject near-collinear configurations before fitting. A homography needs
    # points in general position; four points on a line produce a matrix that
    # fits them perfectly and generalises to nothing.
    centred = image_points - image_points.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    if singular_values[0] > 0 and singular_values[-1] / singular_values[0] < 0.02:
        return HomographyFit(None, n, 0, float("inf"), 0.0, "collinear_keypoints")

    # The threshold is in the *target* units -- metres of pitch -- because that is
    # what findHomography measures residuals in when the destination points are
    # pitch coordinates. Expressing it in pixels and converting by a made-up
    # factor is how this silently became a 0.4 m tolerance and rejected almost
    # every frame for having too few inliers.
    # Plain RANSAC rather than USAC_MAGSAC. MAGSAC++ is usually the better
    # estimator, but in OpenCV 5 it returns None outright on small, clean
    # correspondence sets whose destination coordinates are in metres -- verified
    # against an 8-point exact-fit case where RANSAC recovers the camera to
    # 0.0000 m and MAGSAC returns nothing. Since broadcast frames routinely offer
    # only 5-9 landmarks, that failure mode is the common case here, not an edge
    # case. Least squares is the last resort when even RANSAC cannot fit.
    H, inlier_mask = cv2.findHomography(
        image_points,
        world_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_m,
        maxIters=5000,
        confidence=0.999,
    )
    if H is None:
        H, inlier_mask = cv2.findHomography(image_points, world_points, method=0)
    if H is None:
        return HomographyFit(None, n, 0, float("inf"), 0.0, "ransac_failed")

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else n
    if inliers < min_keypoints:
        return HomographyFit(None, n, inliers, float("inf"), 0.0, "too_few_inliers")

    mask = inlier_mask.ravel().astype(bool) if inlier_mask is not None else np.ones(n, bool)
    errors = reprojection_errors(H, image_points[mask], world_points[mask])
    finite_errors = errors[np.isfinite(errors)]
    mean_error = float(finite_errors.mean()) if finite_errors.size else float("inf")

    reason = validate_homography(H, image_size, pitch, keypoint_hull=image_points[mask])
    if reason is not None:
        return HomographyFit(None, n, inliers, mean_error, 0.0, reason)

    if mean_error > max_reprojection_error_m:
        return HomographyFit(None, n, inliers, mean_error, 0.0, "reprojection_error_too_high")

    confidence = _confidence(
        mean_error, inliers, n, max_reprojection_error_m, keypoint_confidences, mask
    )
    return HomographyFit(H, n, inliers, mean_error, confidence)


def _confidence(
    mean_error_m: float,
    n_inliers: int,
    n_keypoints: int,
    max_error_m: float,
    keypoint_confidences: np.ndarray | None,
    inlier_mask: np.ndarray,
) -> float:
    """Blend three independent signals into one calibration confidence.

    Kept multiplicative rather than additive: any one of the three collapsing
    should collapse the result, because a fit with excellent reprojection error
    on four inliers out of thirty is not trustworthy however good that number
    looks in isolation.
    """
    error_term = float(np.clip(1.0 - mean_error_m / max_error_m, 0.0, 1.0))
    inlier_term = float(np.clip(n_inliers / max(1, n_keypoints), 0.0, 1.0))
    # More landmarks constrain the fit better; saturating at 12 of 32, beyond
    # which extra points add little.
    support_term = float(np.clip(n_inliers / 12.0, 0.0, 1.0))

    if keypoint_confidences is not None and keypoint_confidences.size:
        conf = np.asarray(keypoint_confidences, dtype=np.float64)
        used = conf[inlier_mask] if inlier_mask.shape == conf.shape else conf
        detector_term = float(np.clip(used.mean(), 0.0, 1.0)) if used.size else 0.5
    else:
        detector_term = 1.0

    return float(
        np.clip((error_term**0.5) * (inlier_term**0.5) * support_term * detector_term, 0.0, 1.0)
    )
