"""Pitch model and projective geometry."""

from __future__ import annotations

import numpy as np
import pytest

from visionpitch.common.geometry import (
    apply_homography,
    homography_distance,
    invert_homography,
    normalise_homography,
    reprojection_errors,
    smooth_series,
)
from visionpitch.common.types import BBox
from visionpitch.pitch.geometry import PitchConfiguration, PitchZone, zone_of


class TestPitchConfiguration:
    def test_has_exactly_32_vertices(self, pitch: PitchConfiguration) -> None:
        assert pitch.vertices.shape == (32, 2)
        assert pitch.n_vertices == 32

    def test_every_vertex_lies_on_the_pitch(self, pitch: PitchConfiguration) -> None:
        for i, (x, y) in enumerate(pitch.vertices):
            assert pitch.contains(x, y), f"vertex {i} at ({x}, {y}) is off the pitch"

    def test_vertices_are_distinct(self, pitch: PitchConfiguration) -> None:
        # A duplicated landmark would silently degrade every homography, because
        # two image points would be told to map to the same world point.
        unique = {tuple(np.round(v, 6)) for v in pitch.vertices}
        assert len(unique) == 32

    def test_landmark_semantics(self, pitch: PitchConfiguration) -> None:
        """The ordering is a contract with the keypoint model; pin it down."""
        v = pitch.vertices
        # corners
        assert np.allclose(v[0], [0.0, 0.0])
        assert np.allclose(v[5], [0.0, pitch.width])
        assert np.allclose(v[24], [pitch.length, 0.0])
        assert np.allclose(v[29], [pitch.length, pitch.width])
        # halfway line meets both touchlines
        assert np.allclose(v[13], [pitch.length / 2, 0.0])
        assert np.allclose(v[16], [pitch.length / 2, pitch.width])
        # centre circle poles sit exactly one radius from the centre spot
        centre = np.array(pitch.centre)
        for idx in (14, 15, 30, 31):
            assert np.isclose(
                np.linalg.norm(v[idx] - centre), pitch.centre_circle_radius
            ), f"landmark {idx} is not on the centre circle"
        # penalty spots
        assert np.allclose(v[8], [pitch.penalty_spot_distance, pitch.width / 2])
        assert np.allclose(v[21], [pitch.length - pitch.penalty_spot_distance, pitch.width / 2])

    def test_left_and_right_halves_mirror(self, pitch: PitchConfiguration) -> None:
        v = pitch.vertices
        for left, right in ((1, 25), (2, 26), (3, 27), (4, 28), (9, 17), (12, 20)):
            assert np.isclose(v[left][0], pitch.length - v[right][0])
            assert np.isclose(v[left][1], v[right][1])

    def test_rejects_dimensions_outside_ifab_range(self) -> None:
        with pytest.raises(ValueError, match="length"):
            PitchConfiguration(length=200.0)
        with pytest.raises(ValueError, match="width"):
            PitchConfiguration(width=10.0)

    def test_penalty_area_membership(self, pitch: PitchConfiguration) -> None:
        assert pitch.in_penalty_area(5.0, pitch.width / 2, "left")
        assert not pitch.in_penalty_area(50.0, pitch.width / 2, "left")
        assert pitch.in_penalty_area(pitch.length - 5.0, pitch.width / 2, "right")
        assert not pitch.in_penalty_area(5.0, 1.0, "left")

    def test_normalise_maps_to_unit_square(self, pitch: PitchConfiguration) -> None:
        assert pitch.normalise(0.0, 0.0) == (0.0, 0.0)
        assert pitch.normalise(pitch.length, pitch.width) == (1.0, 1.0)

    def test_zone_classification(self, pitch: PitchConfiguration) -> None:
        assert zone_of(*pitch.centre, pitch) is PitchZone.MID_CENTRE
        assert zone_of(1.0, 1.0, pitch) is PitchZone.DEF_LEFT
        assert zone_of(-50.0, 0.0, pitch) is PitchZone.OFF_PITCH


class TestHomography:
    def test_identity_is_a_no_op(self) -> None:
        points = np.array([[1.0, 2.0], [30.0, 40.0]])
        assert np.allclose(apply_homography(np.eye(3), points), points)

    def test_round_trip_through_inverse(self, synthetic_homography: np.ndarray) -> None:
        image_points = np.array([[300.0, 600.0], [900.0, 400.0], [640.0, 500.0]])
        pitch_points = apply_homography(synthetic_homography, image_points)
        inverse = invert_homography(synthetic_homography)
        assert inverse is not None
        assert np.allclose(apply_homography(inverse, pitch_points), image_points, atol=1e-6)

    def test_degenerate_projection_yields_nan_not_a_huge_number(self) -> None:
        """A point on the horizon must be unusable, not silently enormous."""
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        out = apply_homography(H, np.array([[5.0, 0.0]]))
        assert np.isnan(out).all()

    def test_singular_matrix_has_no_inverse(self) -> None:
        assert invert_homography(np.zeros((3, 3))) is None

    def test_reprojection_error_is_zero_for_exact_correspondences(
        self, synthetic_homography: np.ndarray
    ) -> None:
        image_points = np.array([[180.0, 690.0], [1120.0, 690.0], [880.0, 300.0]])
        expected = apply_homography(synthetic_homography, image_points)
        errors = reprojection_errors(synthetic_homography, image_points, expected)
        assert np.allclose(errors, 0.0, atol=1e-9)

    def test_reprojection_error_is_in_target_units(
        self, synthetic_homography: np.ndarray
    ) -> None:
        image_points = np.array([[640.0, 600.0]])
        truth = apply_homography(synthetic_homography, image_points)
        errors = reprojection_errors(synthetic_homography, image_points, truth + 3.0)
        assert np.isclose(errors[0], np.sqrt(18.0))

    def test_normalise_fixes_scale(self) -> None:
        H = np.eye(3) * 7.0
        assert np.isclose(normalise_homography(H)[2, 2], 1.0)

    def test_distance_is_zero_between_identical_homographies(
        self, synthetic_homography: np.ndarray
    ) -> None:
        d = homography_distance(synthetic_homography, synthetic_homography, (1280, 720))
        assert np.isclose(d, 0.0)

    def test_distance_grows_monotonically_with_disagreement(
        self, synthetic_homography: np.ndarray
    ) -> None:
        """Note the metric is not linear in the matrix entry that was perturbed:
        a homography divides by a point-dependent projective weight, so the same
        entry shift moves near and far points by different amounts. What must
        hold is monotonicity."""
        distances = []
        for shift in (1.0, 5.0, 20.0):
            shifted = synthetic_homography.copy()
            shifted[0, 2] += shift
            distances.append(homography_distance(synthetic_homography, shifted, (1280, 720)))
        assert distances[0] < distances[1] < distances[2]
        assert distances[0] > 0.0


class TestBBox:
    def test_ground_contact_is_bottom_centre(self) -> None:
        box = BBox(100, 200, 140, 400)
        assert box.ground_contact == (120.0, 400.0)
        # Explicitly *not* the centre: projecting the centre through a ground
        # plane homography puts the player metres from where they stand.
        assert box.ground_contact != box.center

    def test_iou_of_identical_boxes_is_one(self) -> None:
        box = BBox(0, 0, 10, 10)
        assert np.isclose(box.iou(box), 1.0)

    def test_iou_of_disjoint_boxes_is_zero(self) -> None:
        assert BBox(0, 0, 10, 10).iou(BBox(50, 50, 60, 60)) == 0.0

    def test_iou_half_overlap(self) -> None:
        a, b = BBox(0, 0, 10, 10), BBox(5, 0, 15, 10)
        assert np.isclose(a.iou(b), 50 / 150)

    def test_clip_keeps_box_inside_frame(self) -> None:
        clipped = BBox(-20, -20, 2000, 2000).clip(1280, 720)
        assert clipped.x1 == 0 and clipped.y1 == 0
        assert clipped.x2 == 1279 and clipped.y2 == 719


class TestSmoothing:
    def test_tolerates_nan_without_shifting_the_series(self) -> None:
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        out = smooth_series(values, 3)
        assert out.shape == values.shape
        assert np.isfinite(out).all()

    def test_window_of_one_is_a_no_op(self) -> None:
        values = np.array([1.0, 5.0, 2.0])
        assert np.allclose(smooth_series(values, 1), values)

    def test_reduces_variance_of_noise(self) -> None:
        rng = np.random.default_rng(0)
        noisy = 10.0 + rng.normal(0, 1.0, 400)
        assert smooth_series(noisy, 9).std() < noisy.std()
