"""Regression tests for Day 039."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from materials.day039.linear_gaussian_lab import (
    binary_entropy_bits,
    error_detection_auc,
    make_model,
    make_uncertainty_cases,
    monte_carlo_validation,
    probability_metrics,
    process_noise_scan,
    propagate_gaussian,
    run,
    simulate_trajectory,
    uncertainty_audit,
)


class Day039Tests(unittest.TestCase):
    def test_model_shapes(self) -> None:
        model = make_model()
        self.assertEqual(np.asarray(model["F"]).shape, (2, 2))
        self.assertEqual(np.asarray(model["B"]).shape, (2, 1))
        self.assertEqual(np.asarray(model["H"]).shape, (2, 2))

    def test_noise_covariances_are_positive_semidefinite(self) -> None:
        model = make_model()
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(np.asarray(model["Q"]))), -1e-12)
        self.assertGreater(np.min(np.linalg.eigvalsh(np.asarray(model["R"]))), 0.0)

    def test_invalid_model_scale_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_model(dt_s=0.0)

    def test_zero_noise_mean_has_closed_form(self) -> None:
        model = make_model()
        controls = np.zeros(4)
        means, _ = propagate_gaussian(np.array([1.0, 2.0]), np.zeros((2, 2)), controls, model)
        self.assertAlmostEqual(means[-1, 0], 5.0)
        self.assertAlmostEqual(means[-1, 1], 2.0)

    def test_covariance_stays_symmetric(self) -> None:
        model = make_model()
        _, covariance = propagate_gaussian(
            np.array([0.0, 2.0]), np.diag([0.16, 0.0225]), np.zeros(30), model
        )
        np.testing.assert_allclose(covariance, covariance.transpose(0, 2, 1))

    def test_simulation_shapes(self) -> None:
        result = simulate_trajectory(steps=12)
        self.assertEqual(np.asarray(result["states"]).shape, (13, 2))
        self.assertEqual(np.asarray(result["measurements"]).shape, (13, 2))
        self.assertEqual(np.asarray(result["analytic_covariance"]).shape, (13, 2, 2))

    def test_simulation_is_deterministic(self) -> None:
        first = simulate_trajectory(seed=39, steps=8)
        second = simulate_trajectory(seed=39, steps=8)
        np.testing.assert_array_equal(first["states"], second["states"])

    def test_measurements_are_not_truth(self) -> None:
        result = simulate_trajectory(seed=39, steps=8)
        self.assertFalse(np.array_equal(result["states"], result["measurements"]))

    def test_monte_carlo_matches_analytic_mean(self) -> None:
        audit = monte_carlo_validation(samples=6000)
        self.assertLess(audit["mean_max_abs_error"], 0.12)

    def test_monte_carlo_coverage_is_reasonable(self) -> None:
        audit = monte_carlo_validation(samples=6000)
        self.assertGreater(audit["position_95_coverage"], 0.92)
        self.assertLess(audit["position_95_coverage"], 0.98)

    def test_process_noise_scan_increases_spread(self) -> None:
        rows = process_noise_scan()
        spreads = [row["final_position_std_m"] for row in rows]
        self.assertTrue(all(left < right for left, right in zip(spreads, spreads[1:])))

    def test_entropy_endpoints(self) -> None:
        np.testing.assert_allclose(binary_entropy_bits(np.array([0.0, 1.0])), 0.0)

    def test_entropy_half_is_one_bit(self) -> None:
        self.assertAlmostEqual(float(binary_entropy_bits(0.5)), 1.0)

    def test_invalid_probability_rejected(self) -> None:
        with self.assertRaises(ValueError):
            binary_entropy_bits(1.01)

    def test_uncertainty_case_contract(self) -> None:
        labels, probabilities, names = make_uncertainty_cases()
        self.assertEqual(labels.shape, (12,))
        self.assertEqual(probabilities.shape, (12, 5))
        self.assertEqual(len(names), 12)

    def test_high_entropy_correct_counterexample_exists(self) -> None:
        audit = uncertainty_audit()
        self.assertTrue(audit["counterexamples"]["high_entropy_but_correct"])

    def test_low_entropy_wrong_counterexample_exists(self) -> None:
        audit = uncertainty_audit()
        self.assertTrue(audit["counterexamples"]["low_entropy_but_wrong"])

    def test_predictive_entropy_dominates_expected_entropy(self) -> None:
        rows = uncertainty_audit()["cases"]
        self.assertTrue(
            all(row["predictive_entropy_bits"] + 1e-12 >= row["expected_member_entropy_bits"] for row in rows)
        )

    def test_error_detection_auc_bounds(self) -> None:
        aucs = uncertainty_audit()["error_detection_auc"]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in aucs.values()))

    def test_pairwise_auc_perfect_ordering(self) -> None:
        auc = error_detection_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertEqual(auc, 1.0)

    def test_probability_metrics_perfect_predictions(self) -> None:
        metrics = probability_metrics(np.array([0.1, 0.9]), np.array([0, 1]))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["brier"], 0.01)

    def test_full_run_writes_parseable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = run(output, steps=12, mc_samples=1000)
            loaded = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["linear_gaussian"]["state_shape"], [13, 2])
            self.assertIn("predictive_uncertainty", report)
            self.assertGreater((output / "linear_gaussian_trajectory.png").stat().st_size, 1000)
            self.assertGreater((output / "uncertainty_audit.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
