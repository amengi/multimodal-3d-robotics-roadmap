"""Regression tests for the Day 031 Jacobian and dependence lab."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from materials.day031.jacobian_dependence_lab import (
    DEFAULT_INTRINSICS,
    DEFAULT_POINT,
    analytic_projection_jacobian,
    canonical_correlations,
    central_difference_jacobian,
    dependence_audit,
    gaussian_mutual_information_bits,
    hsic_biased,
    jacobian_sweep,
    linear_probability_baseline,
    pearson_first_coordinate,
    project_point,
    run_experiment,
)


class Day031Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = dependence_audit(seed=31, sample_count=400)

    def test_01_projection_matches_hand_calculation(self) -> None:
        self.assertTrue(np.allclose(project_point(DEFAULT_POINT), np.array([800.0, 278.0])))

    def test_02_projection_contract_rejects_bad_depth(self) -> None:
        for depth in (0.0, -1.0):
            with self.assertRaises(ValueError):
                project_point([0.4, -0.2, depth])

    def test_03_projection_contract_rejects_shape(self) -> None:
        with self.assertRaises(ValueError):
            project_point([0.4, -0.2])

    def test_04_analytic_jacobian_matches_expected(self) -> None:
        expected = np.array([[400.0, 0.0, -80.0], [0.0, 410.0, 41.0]])
        self.assertTrue(np.allclose(analytic_projection_jacobian(DEFAULT_POINT), expected))

    def test_05_numeric_jacobian_matches_analytic(self) -> None:
        numeric = central_difference_jacobian(project_point, DEFAULT_POINT, 1e-5)
        analytic = analytic_projection_jacobian(DEFAULT_POINT)
        self.assertLess(float(np.max(np.abs(numeric - analytic))), 1e-6)

    def test_06_bad_finite_difference_step_rejected(self) -> None:
        for step in (0.0, -1.0, math.nan):
            with self.assertRaises(ValueError):
                central_difference_jacobian(project_point, DEFAULT_POINT, step)

    def test_07_jacobian_sweep_exposes_roundoff(self) -> None:
        report = jacobian_sweep()
        errors = report["max_absolute_errors_px_per_m"]
        self.assertLess(min(errors), errors[-1])
        self.assertEqual(len(errors), 11)

    def test_08_near_camera_is_more_sensitive(self) -> None:
        sensitivity = jacobian_sweep()["depth_sensitivity"]
        self.assertGreater(sensitivity["near_over_far"], 4.0)

    def test_09_one_dimensional_cca_equals_absolute_pearson(self) -> None:
        x = np.arange(10, dtype=float)
        y = -2.0 * x + np.linspace(-0.1, 0.1, 10)
        pearson = abs(pearson_first_coordinate(x, y))
        cca = float(canonical_correlations(x, y)[0])
        self.assertAlmostEqual(pearson, cca, places=7)

    def test_10_multiview_cca_range_and_count(self) -> None:
        rng = np.random.default_rng(10)
        x = rng.normal(size=(50, 3))
        y = rng.normal(size=(50, 2))
        correlations = canonical_correlations(x, y)
        self.assertEqual(correlations.shape, (2,))
        self.assertTrue(np.all((0.0 <= correlations) & (correlations <= 1.0)))

    def test_11_constant_pearson_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pearson_first_coordinate(np.ones(10), np.arange(10))

    def test_12_gaussian_mi_formula(self) -> None:
        self.assertAlmostEqual(gaussian_mutual_information_bits(0.0), 0.0)
        self.assertAlmostEqual(gaussian_mutual_information_bits(0.8), 0.7369655941662062)

    def test_13_singular_gaussian_correlation_rejected(self) -> None:
        for correlation in (-1.0, 1.0, math.nan):
            with self.assertRaises(ValueError):
                gaussian_mutual_information_bits(correlation)

    def test_14_hsic_detects_nonlinear_pairing(self) -> None:
        rows = self.audit["dependence"]
        self.assertGreater(rows["nonlinear"]["rbf_hsic_biased"], 10.0 * rows["nonlinear_shuffled"]["rbf_hsic_biased"])

    def test_15_linear_measures_miss_quadratic_dependence(self) -> None:
        row = self.audit["dependence"]["nonlinear"]
        self.assertLess(abs(row["pearson_first"]), 0.1)
        self.assertLess(row["cca_first"], 0.1)

    def test_16_shared_identifier_has_high_dependence(self) -> None:
        row = self.audit["dependence"]["shared_identifier"]
        self.assertGreater(row["cca_first"], 0.99)

    def test_17_shared_identifier_is_not_task_predictive(self) -> None:
        baseline = self.audit["task_baselines"]["shared_identifier_only"]
        self.assertLess(abs(baseline["accuracy"] - 0.5), 0.1)
        self.assertGreater(baseline["brier"], 0.23)

    def test_18_clean_fusion_improves_probability_baseline(self) -> None:
        baselines = self.audit["task_baselines"]
        self.assertGreater(baselines["clean_fusion"]["accuracy"], baselines["stable_single_view"]["accuracy"])
        self.assertLess(baselines["clean_fusion"]["brier"], baselines["stable_single_view"]["brier"])

    def test_19_hsic_rejects_identical_samples(self) -> None:
        with self.assertRaises(ValueError):
            hsic_biased(np.ones(10), np.arange(10))

    def test_20_probability_baseline_uses_valid_shapes(self) -> None:
        features = np.arange(20, dtype=float).reshape(10, 2)
        labels = np.arange(10) % 2
        with self.assertRaises(ValueError):
            linear_probability_baseline(features, labels[:-1], np.arange(7), np.arange(7, 10))

    def test_21_complete_run_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory)
            report = run_experiment(destination, seed=31, sample_count=120)
            self.assertEqual(report["configuration"]["seed"], 31)
            for filename in ("report.json", "jacobian_sweep.png", "dependence_audit.png"):
                self.assertGreater((destination / filename).stat().st_size, 100)
            loaded = json.loads((destination / "report.json").read_text(encoding="utf-8"))
            self.assertIn("dependence", loaded)


if __name__ == "__main__":
    unittest.main()
