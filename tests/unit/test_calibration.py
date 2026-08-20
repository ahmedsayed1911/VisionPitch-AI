"""Homography estimation, validation and temporal handling."""

from __future__ import annotations

import numpy as np
import pytest

from visionpitch.calibration.homography import (
    estimate_homography,
    validate_homography,
)
from visionpitch.calibration.temporal import (
    ShotChangeDetector,
    average_homographies,
    reject_temporal_outliers,
    temporal_stability,
)
from visionpitch.common.geometry import apply_homography, invert_homography
from visionpitch.common.types import CalibrationResult, SegmentKind
from visionpitch.pitch.geometry import PitchConfiguration

IMAGE_SIZE = (1280, 720)


def project_landmarks(H_pitch_to_image: np.ndarray, pitch: PitchConfiguration, indices):
    """Where the given pitch landmarks appear in the image under a known camera."""
    world = pitch.vertices[list(indices)]
    return apply_homography(H_pitch_to_image, world)


class TestValidateHomography:
    def test_accepts_a_plausible_camera(
        self, synthetic_homography: np.ndarray, pitch: PitchConfiguration
    ) -> None:
        assert validate_homography(synthetic_homography, IMAGE_SIZE, pitch) is None

    def test_rejects_non_finite(self, pitch: PitchConfiguration) -> None:
        H = np.full((3, 3), np.nan)
        assert validate_homography(H, IMAGE_SIZE, pitch) == "non_finite"

    def test_rejects_degenerate(self, pitch: PitchConfiguration) -> None:
        assert validate_homography(np.zeros((3, 3)), IMAGE_SIZE, pitch) == "degenerate"

    def test_rejects_absurd_scale(self, pitch: PitchConfiguration) -> None:
        H = np.eye(3) * 1.0
        H[0, 0] = H[1, 1] = 500.0  # 500 metres of pitch per pixel
        assert validate_homography(H, IMAGE_SIZE, pitch) == "implausible_scale_large"

    def test_accepts_an_orientation_reversing_mapping(
        self, synthetic_homography: np.ndarray, pitch: PitchConfiguration
    ) -> None:
        """Image y points down, pitch y points up, so a correct homography
        reverses orientation. Requiring a fixed sign rejects valid cameras --
        this is the regression test for that bug."""
        flip = np.diag([1.0, -1.0, 1.0])
        mirrored = synthetic_homography @ flip
        reason = validate_homography(mirrored, IMAGE_SIZE, pitch)
        assert reason != "mirrored"


class TestEstimateHomography:
    def test_recovers_a_known_camera(self, pitch: PitchConfiguration) -> None:
        """Given exact landmark correspondences, the solve must be near-perfect."""
        import cv2

        # A synthetic camera: pitch -> image.
        world = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
        image = np.array([[150.0, 700.0], [1150.0, 700.0], [900.0, 250.0], [400.0, 250.0]])
        pitch_to_image, _ = cv2.findHomography(world, image, method=0)

        indices = [0, 5, 13, 16, 24, 29, 8, 21]
        image_points = project_landmarks(pitch_to_image, pitch, indices)

        fit = estimate_homography(
            image_points, np.array(indices), pitch, IMAGE_SIZE, min_keypoints=5
        )
        assert fit.ok, fit.rejection_reason
        assert fit.reprojection_error_m < 0.05
        assert fit.confidence > 0.5

        # And it must invert back to the camera we started from.
        recovered = invert_homography(fit.homography)
        assert recovered is not None
        check = apply_homography(recovered, pitch.vertices[[0, 24]])
        assert np.allclose(check, image[[0, 1]], atol=1.0)

    def test_rejects_too_few_keypoints(self, pitch: PitchConfiguration) -> None:
        fit = estimate_homography(
            np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]), pitch, IMAGE_SIZE
        )
        assert not fit.ok
        assert fit.rejection_reason == "too_few_keypoints"

    def test_rejects_collinear_keypoints(self, pitch: PitchConfiguration) -> None:
        """Points on a line fit perfectly and generalise to nothing."""
        points = np.array([[100.0 + 40 * i, 400.0] for i in range(8)])
        fit = estimate_homography(
            points, np.arange(8), pitch, IMAGE_SIZE, min_keypoints=5
        )
        assert not fit.ok
        assert fit.rejection_reason == "collinear_keypoints"

    def test_tolerance_is_in_metres_not_pixels(self, pitch: PitchConfiguration) -> None:
        """A 2 m tolerance must accept landmarks that are noisy by ~1 m.

        Regression test: the tolerance was once derived from a pixel value and
        became 0.4 m, which rejected almost every real frame for lack of inliers.
        """
        import cv2

        world = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
        image = np.array([[150.0, 700.0], [1150.0, 700.0], [900.0, 250.0], [400.0, 250.0]])
        pitch_to_image, _ = cv2.findHomography(world, image, method=0)

        indices = [0, 5, 13, 16, 24, 29, 8, 21, 9, 17]
        image_points = project_landmarks(pitch_to_image, pitch, indices)
        rng = np.random.default_rng(3)
        noisy = image_points + rng.normal(0, 3.0, image_points.shape)

        fit = estimate_homography(
            noisy, np.array(indices), pitch, IMAGE_SIZE,
            ransac_threshold_m=2.0, min_keypoints=5,
        )
        assert fit.ok, f"rejected: {fit.rejection_reason}"
        assert fit.n_inliers >= 8


class TestTemporal:
    def test_average_of_identical_homographies_is_unchanged(
        self, synthetic_homography: np.ndarray
    ) -> None:
        out = average_homographies(
            [synthetic_homography] * 5, [1.0] * 5, IMAGE_SIZE
        )
        assert out is not None
        probe = np.array([[640.0, 600.0]])
        assert np.allclose(
            apply_homography(out, probe),
            apply_homography(synthetic_homography, probe),
            atol=0.05,
        )

    def test_average_is_robust_to_one_wild_outlier(
        self, synthetic_homography: np.ndarray
    ) -> None:
        """A single catastrophic fit must not drag the smoothed result.

        This is why the combination is a weighted median rather than a mean.
        """
        wild = synthetic_homography.copy()
        wild[0, 2] += 400.0
        inputs = [synthetic_homography] * 6 + [wild]

        out = average_homographies(inputs, [1.0] * 7, IMAGE_SIZE)
        assert out is not None
        probe = np.array([[640.0, 600.0]])
        good = apply_homography(synthetic_homography, probe)
        got = apply_homography(out, probe)
        assert np.linalg.norm(got - good) < 5.0

    def test_rejects_a_teleporting_frame(
        self, synthetic_homography: np.ndarray
    ) -> None:
        frames = list(range(30))
        results = {
            f: CalibrationResult(f, synthetic_homography.copy(), 0.8, 0.3, 10, 9)
            for f in frames
        }
        rogue = synthetic_homography.copy()
        rogue[0, 2] += 60.0  # 60 m away from every neighbour
        results[15] = CalibrationResult(15, rogue, 0.8, 0.3, 10, 9)

        cleaned, n_rejected = reject_temporal_outliers(
            results, frames, IMAGE_SIZE, max_jump_m=8.0
        )
        assert n_rejected == 1
        assert cleaned[15].homography is None
        assert cleaned[14].homography is not None

    def test_keeps_a_smoothly_panning_camera(
        self, synthetic_homography: np.ndarray
    ) -> None:
        """Rejection must not fire on legitimate camera movement."""
        frames = list(range(40))
        results = {}
        for f in frames:
            H = synthetic_homography.copy()
            H[0, 2] += 0.3 * f  # a steady 0.3 m/frame pan
            results[f] = CalibrationResult(f, H, 0.8, 0.3, 10, 9)

        _, n_rejected = reject_temporal_outliers(results, frames, IMAGE_SIZE, max_jump_m=8.0)
        assert n_rejected == 0

    def test_stability_reports_median_and_mean(
        self, synthetic_homography: np.ndarray
    ) -> None:
        frames = list(range(20))
        results = {
            f: CalibrationResult(f, synthetic_homography.copy(), 0.8, 0.2, 10, 9)
            for f in frames
        }
        stats = temporal_stability(results, frames, IMAGE_SIZE)
        assert stats["median_delta_m"] == pytest.approx(0.0, abs=1e-6)
        assert "mean_delta_m" in stats and "p95_delta_m" in stats

    def test_invalid_frames_break_the_stability_chain(self) -> None:
        results = {f: CalibrationResult(f, None, 0.0, float("nan"), 0, 0) for f in range(10)}
        stats = temporal_stability(results, list(range(10)), IMAGE_SIZE)
        assert stats["n_pairs"] == 0


class TestShotChangeDetector:
    def test_no_change_on_an_identical_frame(self) -> None:
        rng = np.random.default_rng(1)
        image = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
        detector = ShotChangeDetector(threshold=0.45)
        detector.update(image)
        changed, dissimilarity = detector.update(image.copy())
        assert changed is False
        assert dissimilarity == pytest.approx(0.0, abs=1e-6)

    def test_detects_a_hard_cut(self) -> None:
        green = np.zeros((240, 320, 3), np.uint8)
        green[:, :] = (40, 140, 40)
        red = np.zeros((240, 320, 3), np.uint8)
        red[:, :] = (30, 30, 200)

        detector = ShotChangeDetector(threshold=0.45)
        detector.update(green)
        changed, _ = detector.update(red)
        assert changed is True

    def test_first_frame_is_never_a_cut(self) -> None:
        detector = ShotChangeDetector()
        changed, _ = detector.update(np.zeros((240, 320, 3), np.uint8))
        assert changed is False


def test_calibration_result_reports_validity() -> None:
    assert CalibrationResult(0, np.eye(3), 0.9, 0.1, 8, 8).is_valid
    assert not CalibrationResult(0, None, 0.0, float("nan"), 0, 0).is_valid
    assert CalibrationResult(0, None, 0.0, 0.0, 0, 0).segment_kind is SegmentKind.UNKNOWN
