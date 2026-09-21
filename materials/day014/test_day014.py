"""Regression tests for the Day 014 weekly project."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from materials.day014.weekly_project import (
    ContractError,
    cloud_summary,
    dependence_event_gap,
    fit_regression_baselines,
    make_cube_surface,
    make_nonlinear_data,
    paired_rmse_m,
    pearson_correlation,
    rotation_z,
    run,
    transform_points,
    validate_points,
    validate_rigid_transform,
)


class Day014Tests(unittest.TestCase):
    def test_cube_is_three_dimensional_and_centered(self):
        points = make_cube_surface()
        self.assertEqual(points.shape[1], 3)
        np.testing.assert_allclose(points.mean(axis=0), 0.0, atol=1e-15)

    def test_cube_aabb_matches_side_length(self):
        summary = cloud_summary(make_cube_surface(side_m=2.0))
        np.testing.assert_allclose(summary["aabb_extent_m"], [2.0, 2.0, 2.0])

    def test_bad_point_shape_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "shape"):
            validate_points(np.zeros((5, 2)))

    def test_nonfinite_point_is_rejected(self):
        points = np.zeros((4, 3))
        points[0, 0] = np.nan
        with self.assertRaisesRegex(ContractError, "finite"):
            validate_points(points)

    def test_rotation_is_proper(self):
        rotation = rotation_z(30.0)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_reflection_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "determinant"):
            validate_rigid_transform(np.diag([1.0, 1.0, -1.0]), np.zeros(3))

    def test_row_vector_transform(self):
        points = np.array([[1.0, 0.0, 0.0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        transformed = transform_points(points, rotation_z(90), [1, 2, 3])
        np.testing.assert_allclose(transformed[0], [1, 3, 3], atol=1e-12)

    def test_transform_is_reproducible_for_same_seed(self):
        points = make_cube_surface()
        first = transform_points(points, np.eye(3), np.zeros(3), noise_sigma_m=0.1, seed=7)
        second = transform_points(points, np.eye(3), np.zeros(3), noise_sigma_m=0.1, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_paired_rmse_is_known(self):
        reference = np.zeros((4, 3))
        estimate = np.tile([3.0, 4.0, 0.0], (4, 1))
        self.assertAlmostEqual(paired_rmse_m(reference, estimate), 5.0)

    def test_unpaired_shapes_are_rejected(self):
        with self.assertRaisesRegex(ContractError, "identical shape"):
            paired_rmse_m(np.zeros((4, 3)), np.zeros((5, 3)))

    def test_nonlinear_data_is_reproducible(self):
        first = make_nonlinear_data(seed=11)
        second = make_nonlinear_data(seed=11)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_zero_variance_pearson_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "zero variance"):
            pearson_correlation(np.ones(5), np.arange(5))

    def test_quadratic_baseline_beats_linear(self):
        x, y = make_nonlinear_data(seed=17)
        summary = fit_regression_baselines(x, y)
        self.assertLess(summary["quadratic"]["test_rmse"], summary["linear"]["test_rmse"])

    def test_event_factorization_has_visible_gap(self):
        x, y = make_nonlinear_data(seed=23, n_samples=10000)
        summary = dependence_event_gap(x, y)
        self.assertGreater(summary["absolute_gap"], 0.10)

    def test_full_run_writes_strict_json_and_two_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run(Path(directory))
            self.assertEqual(len(report["figures"]), 2)
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in report["figures"]))
            with Path(report["report_path"]).open(encoding="utf-8") as handle:
                parsed = json.load(handle, parse_constant=lambda value: self.fail(value))
            self.assertEqual(parsed["point_cloud_contract"]["unit"], "m")
            self.assertLess(
                parsed["dependence"]["regression_baselines"]["quadratic"]["test_rmse"],
                parsed["dependence"]["regression_baselines"]["linear"]["test_rmse"],
            )


if __name__ == "__main__":
    unittest.main()
