"""Regression tests for the Day 027 lab."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from materials.day027.svd_condition_mdl import (
    compact_svd,
    condition_demo,
    fusion_model_comparison,
    image_metrics,
    make_synthetic_image,
    mdl_proxy_bits,
    rank_reconstruct,
    run_experiment,
    validate_matrix,
)


class Day027Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = make_synthetic_image()
        cls.u, cls.s, cls.vh = compact_svd(cls.image)

    def test_01_image_contract(self) -> None:
        self.assertEqual(self.image.shape, (64, 96))
        self.assertGreaterEqual(float(self.image.min()), 0.0)
        self.assertLessEqual(float(self.image.max()), 1.0)

    def test_02_seed_is_deterministic(self) -> None:
        np.testing.assert_array_equal(self.image, make_synthetic_image())
        self.assertFalse(np.array_equal(self.image, make_synthetic_image(seed=7)))

    def test_03_validate_rejects_wrong_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D"):
            validate_matrix(np.zeros(3))

    def test_04_validate_rejects_nonfinite(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_matrix(np.array([[1.0, np.nan]]))

    def test_05_compact_shapes(self) -> None:
        self.assertEqual(self.u.shape, (64, 64))
        self.assertEqual(self.s.shape, (64,))
        self.assertEqual(self.vh.shape, (64, 96))

    def test_06_singular_values_descend(self) -> None:
        self.assertTrue(np.all(self.s[:-1] >= self.s[1:]))
        self.assertTrue(np.all(self.s >= 0))

    def test_07_svd_reconstructs(self) -> None:
        reconstructed = (self.u * self.s) @ self.vh
        np.testing.assert_allclose(reconstructed, self.image, atol=1e-12)

    def test_08_factors_are_orthonormal(self) -> None:
        np.testing.assert_allclose(self.u.T @ self.u, np.eye(64), atol=1e-12)
        np.testing.assert_allclose(self.vh @ self.vh.T, np.eye(64), atol=1e-12)

    def test_09_rank_zero_is_zero_matrix(self) -> None:
        self.assertFalse(np.any(rank_reconstruct(self.u, self.s, self.vh, 0)))

    def test_10_rank_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            rank_reconstruct(self.u, self.s, self.vh, 65)

    def test_11_error_decreases_with_rank(self) -> None:
        errors = [
            image_metrics(self.image, rank_reconstruct(self.u, self.s, self.vh, rank))["rmse"]
            for rank in (0, 1, 2, 4, 8, 16, 32, 64)
        ]
        self.assertTrue(all(a >= b - 1e-14 for a, b in zip(errors, errors[1:])))

    def test_12_rank_bound(self) -> None:
        approximation = rank_reconstruct(self.u, self.s, self.vh, 4)
        self.assertLessEqual(np.linalg.matrix_rank(approximation, tol=1e-10), 4)

    def test_13_frobenius_tail_identity(self) -> None:
        approximation = rank_reconstruct(self.u, self.s, self.vh, 4)
        actual = np.linalg.norm(self.image - approximation, "fro") ** 2
        expected = np.sum(self.s[4:] ** 2)
        self.assertAlmostEqual(float(actual), float(expected), places=10)

    def test_14_metrics_reject_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical"):
            image_metrics(np.zeros((2, 2)), np.zeros((2, 3)))

    def test_15_mdl_proxy_adds_parts(self) -> None:
        proxy = mdl_proxy_bits(np.zeros((2, 3)), parameter_count=5)
        self.assertEqual(proxy["model_bits"], 80)
        self.assertAlmostEqual(
            proxy["total_proxy_bits"], proxy["model_bits"] + proxy["residual_nll_bits"]
        )

    def test_16_condition_number_and_amplification(self) -> None:
        demo = condition_demo()
        self.assertAlmostEqual(demo["condition_number_2"], 10000.0)
        self.assertGreater(demo["relative_x_error"], 1000 * demo["relative_b_error"])
        self.assertLessEqual(demo["observed_amplification"], demo["condition_number_2"] * 1.01)

    def test_17_pinv_cutoff_tradeoff(self) -> None:
        demo = condition_demo()
        keep = demo["pinv_cutoffs"]["1e-12"]
        truncate = demo["pinv_cutoffs"]["1e-03"]
        self.assertLess(keep["relative_residual"], truncate["relative_residual"])
        self.assertLess(keep["relative_solution_error"], truncate["relative_solution_error"])

    def test_18_fusion_model_tradeoff(self) -> None:
        comparison = fusion_model_comparison()
        stable = comparison["models"]["stable_single_modality"]
        fusion = comparison["models"]["two_modality_fusion"]
        self.assertGreater(fusion["parameter_count"], stable["parameter_count"])
        self.assertLess(fusion["nll_nats_per_example"], stable["nll_nats_per_example"])
        self.assertLess(fusion["total_proxy_bits"], stable["total_proxy_bits"])

    def test_19_complete_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            report = run_experiment(output)
            self.assertEqual(report["seed"], 2709)
            self.assertIn(report["mdl_proxy_definition"]["best_rank"], (1, 2, 4, 8, 16, 32))
            for name in ("report.json", "reconstructions.png", "svd_mdl_curves.png"):
                self.assertGreater((output / name).stat().st_size, 100)
            loaded = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["data_contract"]["image_shape"], [64, 96])


if __name__ == "__main__":
    unittest.main()
