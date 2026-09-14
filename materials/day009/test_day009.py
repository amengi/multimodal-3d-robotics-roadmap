"""Regression tests for Day 009 broadcasting and vectorization."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from materials.day009.rigid_and_projection import (
    ContractError,
    benchmark_transforms,
    cosine_similarity_matrix,
    project_to_shared_space,
    rotation_z,
    run_demo,
    transform_points_loop,
    transform_points_vectorized,
    validate_points,
)


class Day009Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-1.0, 0.0, 1.0]])
        self.rotation = rotation_z(90.0)
        self.translation = np.array([10.0, 1.0, -1.0])

    def test_rotation_is_proper(self) -> None:
        np.testing.assert_allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(self.rotation)), 1.0)

    def test_hand_calculated_transform(self) -> None:
        expected = np.array([[10.0, 2.0, -1.0], [8.0, 1.0, -1.0], [10.0, 0.0, 0.0]])
        actual = transform_points_vectorized(self.points, self.rotation, self.translation)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

    def test_loop_and_vectorized_forms_agree(self) -> None:
        rng = np.random.default_rng(20260914)
        points = rng.normal(size=(137, 3))
        loop = transform_points_loop(points, rotation_z(-37.0), self.translation)
        vectorized = transform_points_vectorized(points, rotation_z(-37.0), self.translation)
        np.testing.assert_allclose(loop, vectorized, atol=1e-12, rtol=0.0)

    def test_translation_broadcasts_over_rows(self) -> None:
        actual = transform_points_vectorized(np.zeros((4, 3)), np.eye(3), self.translation)
        np.testing.assert_allclose(actual, np.tile(self.translation, (4, 1)))

    def test_transform_does_not_mutate_input(self) -> None:
        before = self.points.copy()
        transform_points_vectorized(self.points, self.rotation, self.translation)
        np.testing.assert_array_equal(self.points, before)

    def test_bad_point_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, r"\(N, 3\)"):
            validate_points(np.ones((3, 2)))

    def test_nonfinite_point_is_rejected(self) -> None:
        points = self.points.copy()
        points[1, 2] = np.nan
        with self.assertRaisesRegex(ContractError, "finite"):
            validate_points(points)

    def test_non_rotation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "R.T"):
            transform_points_vectorized(self.points, 2.0 * np.eye(3), self.translation)

    def test_shared_projection_accepts_different_input_widths(self) -> None:
        a = np.ones((5, 2))
        b = np.ones((5, 3))
        za, zb = project_to_shared_space(a, b, np.ones((2, 4)), np.ones((3, 4)))
        self.assertEqual(za.shape, (5, 4))
        self.assertEqual(zb.shape, (5, 4))

    def test_wrong_pair_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "same nonzero B"):
            project_to_shared_space(
                np.ones((5, 2)), np.ones((4, 3)), np.ones((2, 2)), np.ones((3, 2))
            )

    def test_pairwise_cosine_for_orthogonal_vectors(self) -> None:
        identity = np.eye(2)
        np.testing.assert_allclose(cosine_similarity_matrix(identity, identity), identity)

    def test_zero_vector_is_rejected_for_cosine(self) -> None:
        with self.assertRaisesRegex(ContractError, "zero vector"):
            cosine_similarity_matrix(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))

    def test_swapping_pairs_moves_the_similarity_diagonal(self) -> None:
        identity = np.eye(2)
        paired = cosine_similarity_matrix(identity, identity)
        swapped = cosine_similarity_matrix(identity, identity[::-1])
        self.assertGreater(float(np.trace(paired)), float(np.trace(swapped)))

    def test_benchmark_proves_equivalence(self) -> None:
        result = benchmark_transforms(n_points=300, repeats=2)
        self.assertLessEqual(float(result["max_abs_error_m"]), 1e-12)
        self.assertGreater(float(result["loop_median_ms"]), 0.0)
        self.assertGreater(float(result["vectorized_median_ms"]), 0.0)

    def test_demo_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_demo(Path(directory), n_points=300, repeats=2)
            path = Path(report["report_path"])
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
