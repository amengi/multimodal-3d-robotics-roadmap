from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from materials.day026.pca_rate_distortion import (
    allocation_for,
    direction_angle_deg,
    encode_decode,
    fit_pca,
    fit_quantizer_ranges,
    isotropic_failure,
    line_angle_error_deg,
    make_dataset,
    project,
    reconstruct,
    run_experiment,
    uniform_quantize,
    validate_points,
)


class Day026Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="day026-test-")
        cls.output = Path(cls.temp.name)
        cls.report = run_experiment(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_input_contract(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_points(np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_points(np.array([[0, 0], [1, np.nan], [2, 2]]))

    def test_02_hand_covariance(self):
        fit = fit_pca(np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]))
        np.testing.assert_allclose(fit.covariance, [[1.0, -1.0], [-1.0, 1.0]])

    def test_03_eigenvalues_descend(self):
        values = np.asarray(self.report["eigenvalues_m2"])
        self.assertGreater(values[0], values[1])
        self.assertTrue(np.all(values >= 0))

    def test_04_eigen_equation_and_orthonormality(self):
        points, _, _, _ = make_dataset()
        fit = fit_pca(points[:600])
        np.testing.assert_allclose(
            fit.covariance @ fit.eigenvectors,
            fit.eigenvectors @ np.diag(fit.eigenvalues),
            atol=1e-12,
        )
        np.testing.assert_allclose(fit.eigenvectors.T @ fit.eigenvectors, np.eye(2), atol=1e-12)

    def test_05_projection_round_trip(self):
        points, _, _, _ = make_dataset()
        fit = fit_pca(points[:600])
        recovered = reconstruct(project(points[600:], fit), fit)
        self.assertLess(
            np.sqrt(np.mean(np.sum((points[600:] - recovered) ** 2, axis=1))),
            1e-12,
        )

    def test_06_principal_direction(self):
        self.assertLess(self.report["principal_angle_error_deg"], 2.0)
        self.assertGreater(self.report["explained_variance_ratio"][0], 0.97)

    def test_07_sign_invariant_angle(self):
        self.assertAlmostEqual(line_angle_error_deg(np.array([1, 0]), np.array([-1, 0])), 0.0)
        self.assertAlmostEqual(direction_angle_deg(np.array([-1, 0])), 0.0)

    def test_08_quantizer_zero_bits(self):
        np.testing.assert_array_equal(uniform_quantize(np.array([-2.0, 3.0]), 0, -2, 3), 0.0)

    def test_09_allocation_budget(self):
        for rate in (0, 1, 2, 4, 6, 8):
            for policy in ("variance", "task_aware"):
                self.assertEqual(sum(allocation_for(policy, rate)), rate)

    def test_10_raw_baseline(self):
        raw = self.report["raw_float64_baseline"]
        self.assertEqual(raw["rate_bits_per_point"], 128)
        self.assertEqual(raw["rmse_m"], 0.0)
        self.assertGreater(raw["task_accuracy"], 0.99)

    def test_11_variance_policy_loses_task_feature(self):
        rows = [
            row for row in self.report["codec_rows"]
            if row["policy"] == "variance"
        ]
        _, labels, _, _ = make_dataset()
        expected_tie_accuracy = float(np.mean(labels[600:] == 1))
        for row in rows:
            self.assertAlmostEqual(row["task_accuracy"], expected_tie_accuracy)

    def test_12_task_aware_policy_preserves_task(self):
        row = next(
            row for row in self.report["codec_rows"]
            if row["policy"] == "task_aware" and row["rate_bits_per_point"] == 8
        )
        self.assertGreater(row["task_accuracy"], 0.95)

    def test_13_same_rate_is_not_same_utility(self):
        rows = {
            row["policy"]: row for row in self.report["codec_rows"]
            if row["rate_bits_per_point"] == 4
        }
        self.assertGreater(
            rows["task_aware"]["task_accuracy"], rows["variance"]["task_accuracy"] + 0.3
        )

    def test_14_isotropic_failure_is_reported(self):
        failure = isotropic_failure()
        self.assertGreater(failure["angle_range_deg"], 90.0)
        self.assertLess(failure["median_eigengap_ratio"], 0.5)

    def test_15_artifacts(self):
        parsed = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(parsed["seed"], 2608)
        for name in ("pca_direction.png", "rate_distortion.png"):
            self.assertGreater((self.output / name).stat().st_size, 10_000)

    def test_16_seed_is_configurable(self):
        alternative = run_experiment(self.output / "seed-42", seed=42)
        self.assertEqual(alternative["seed"], 42)
        self.assertNotEqual(alternative["mean_m"], self.report["mean_m"])


if __name__ == "__main__":
    unittest.main()
