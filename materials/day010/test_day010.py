"""Regression tests for the Day 010 linear algebra lab."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from materials.day010.linear_algebra_lab import (
    ContractError,
    coordinate_drop_reconstruction,
    covariance_eigh,
    feature_rmse,
    fit_line_lstsq,
    make_line_data,
    make_rotated_features,
    pca_svd,
    reconstruct_pca,
    root_mean_square_error,
    run_demo,
    solve_square,
)


class Day010Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = make_rotated_features(n_samples=120)

    def test_solve_square_known_system(self) -> None:
        solution = solve_square([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])
        np.testing.assert_allclose(solution, [2.0, 3.0], atol=1e-12, rtol=0.0)

    def test_solve_rejects_singular_matrix(self) -> None:
        with self.assertRaisesRegex(ContractError, "nonsingular"):
            solve_square([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])

    def test_rmse_known_values(self) -> None:
        self.assertAlmostEqual(root_mean_square_error([0.0, 0.0], [3.0, 4.0]), np.sqrt(12.5))

    def test_line_fit_recovers_noiseless_coefficients(self) -> None:
        x = np.linspace(-2.0, 2.0, 9)
        fit, predicted, residual = fit_line_lstsq(x, 1.75 * x - 0.40)
        self.assertAlmostEqual(fit.slope, 1.75)
        self.assertAlmostEqual(fit.intercept_m, -0.40)
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)
        np.testing.assert_allclose(predicted, 1.75 * x - 0.40, atol=1e-12)

    def test_line_fit_beats_mean_baseline(self) -> None:
        x, y = make_line_data()
        fit, _, _ = fit_line_lstsq(x, y)
        self.assertLess(fit.rmse_m, fit.mean_baseline_rmse_m)

    def test_line_fit_rejects_constant_x(self) -> None:
        with self.assertRaisesRegex(ContractError, "must vary"):
            fit_line_lstsq(np.ones(5), np.arange(5.0))

    def test_nonfinite_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "finite"):
            pca_svd([[1.0, 2.0], [np.nan, 3.0]])

    def test_pca_shapes_and_centered_scores(self) -> None:
        result = pca_svd(self.features, n_components=1)
        self.assertEqual(result.components.shape, (1, 3))
        self.assertEqual(result.scores.shape, (120, 1))
        np.testing.assert_allclose(result.scores.mean(axis=0), 0.0, atol=1e-12)

    def test_pca_components_are_unit_vectors(self) -> None:
        result = pca_svd(self.features, n_components=3)
        np.testing.assert_allclose(result.components @ result.components.T, np.eye(3), atol=1e-12)

    def test_explained_variance_ratio_is_valid(self) -> None:
        result = pca_svd(self.features, n_components=3)
        self.assertTrue(np.all(result.explained_variance_ratio >= 0.0))
        self.assertAlmostEqual(float(result.explained_variance_ratio.sum()), 1.0)

    def test_all_components_reconstruct_input(self) -> None:
        result = pca_svd(self.features, n_components=3)
        np.testing.assert_allclose(reconstruct_pca(result), self.features, atol=1e-12, rtol=0.0)

    def test_first_pc_beats_keep_first_raw_coordinate(self) -> None:
        result = pca_svd(self.features, n_components=1)
        pca_error = feature_rmse(self.features, reconstruct_pca(result))
        drop_error = feature_rmse(self.features, coordinate_drop_reconstruction(self.features))
        self.assertLess(pca_error, drop_error)

    def test_svd_and_eigh_agree_on_eigenvalues(self) -> None:
        result = pca_svd(self.features, n_components=3)
        eigenvalues, _ = covariance_eigh(self.features)
        np.testing.assert_allclose(result.explained_variance, eigenvalues, atol=1e-12, rtol=1e-12)

    def test_svd_and_eigh_first_directions_align_up_to_sign(self) -> None:
        result = pca_svd(self.features, n_components=1)
        _, eigenvectors = covariance_eigh(self.features)
        self.assertAlmostEqual(abs(float(result.components[0] @ eigenvectors[:, 0])), 1.0)

    def test_demo_writes_nonempty_report_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_demo(Path(directory))
            self.assertGreater(Path(report["report_path"]).stat().st_size, 0)
            self.assertGreater(Path(report["residual_plot"]).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
