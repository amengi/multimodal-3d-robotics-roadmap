"""Regression tests for Day 034."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from materials.day034.robust_bottleneck_lab import (
    F_SCALE_M,
    OUTLIER_FRACTION,
    TRUE_INTERCEPT_M,
    TRUE_SLOPE,
    binary_entropy_nats,
    cauchy_loss,
    dropout_baseline,
    encoder_metrics,
    fit_line,
    huber_loss,
    make_line_data,
    optimize_binary_bottleneck,
    run_experiment,
    save_plots,
)


class Day034Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = make_line_data(seed=34)
        cls.report = run_experiment(seed=34)

    def test_data_shapes(self) -> None:
        self.assertEqual(self.data["x_m"].shape, (100,))
        self.assertEqual(self.data["y_m"].shape, (100,))

    def test_exact_outlier_fraction(self) -> None:
        self.assertEqual(int(self.data["outlier_mask"].sum()), int(100 * OUTLIER_FRACTION))

    def test_data_is_deterministic(self) -> None:
        again = make_line_data(seed=34)
        np.testing.assert_array_equal(self.data["y_m"], again["y_m"])

    def test_linear_fit_succeeds(self) -> None:
        self.assertTrue(fit_line(self.data, "linear")["success"])

    def test_huber_fit_succeeds(self) -> None:
        self.assertTrue(fit_line(self.data, "huber")["success"])

    def test_cauchy_fit_succeeds(self) -> None:
        self.assertTrue(fit_line(self.data, "cauchy")["success"])

    def test_robust_slope_beats_l2(self) -> None:
        fits = self.report["robust_fits"]
        self.assertLess(fits["huber"]["slope_abs_error"], fits["linear"]["slope_abs_error"])
        self.assertLess(fits["cauchy"]["slope_abs_error"], fits["linear"]["slope_abs_error"])

    def test_robust_intercept_beats_l2(self) -> None:
        fits = self.report["robust_fits"]
        self.assertLess(fits["huber"]["intercept_abs_error_m"], fits["linear"]["intercept_abs_error_m"])

    def test_huber_formula_inside(self) -> None:
        self.assertAlmostEqual(float(huber_loss(0.1, 0.2)), 0.005)

    def test_huber_formula_outside(self) -> None:
        self.assertAlmostEqual(float(huber_loss(0.5, 0.2)), 0.08)

    def test_cauchy_is_even(self) -> None:
        self.assertAlmostEqual(float(cauchy_loss(-1.0, F_SCALE_M)), float(cauchy_loss(1.0, F_SCALE_M)))

    def test_binary_entropy_endpoints(self) -> None:
        np.testing.assert_allclose(binary_entropy_nats(np.array([0.0, 1.0])), 0.0)

    def test_binary_entropy_half(self) -> None:
        self.assertAlmostEqual(float(binary_entropy_nats(0.5)), math.log(2.0))

    def test_capacity_deterministic_two_bits(self) -> None:
        self.assertAlmostEqual(encoder_metrics(0.0, 0.0)["capacity_bits"], 2.0)

    def test_capacity_random_two_bits(self) -> None:
        self.assertAlmostEqual(encoder_metrics(0.5, 0.5)["capacity_bits"], 0.0)

    def test_vib_drops_nuisance(self) -> None:
        row = optimize_binary_bottleneck(0.1, "vib")
        self.assertAlmostEqual(row["nuisance_error"], 0.5)

    def test_beta_vae_treats_bits_symmetrically(self) -> None:
        row = optimize_binary_bottleneck(0.1, "beta_vae")
        self.assertAlmostEqual(row["signal_error"], row["nuisance_error"])

    def test_vib_capacity_nonincreasing(self) -> None:
        capacities = [row["capacity_bits"] for row in self.report["binary_bottleneck"]["vib"]]
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(capacities, capacities[1:])))

    def test_dropout_exact_metrics(self) -> None:
        row = dropout_baseline(0.30)
        self.assertAlmostEqual(row["task_accuracy"], 0.85)
        self.assertAlmostEqual(row["task_brier"], 0.075)

    def test_declared_truth_is_preserved(self) -> None:
        config = self.report["config"]
        self.assertEqual(config["true_slope"], TRUE_SLOPE)
        self.assertEqual(config["true_intercept_m"], TRUE_INTERCEPT_M)

    def test_report_round_trip_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_plots(self.report, output, seed=34)
            path = output / "report.json"
            path.write_text(json.dumps(self.report), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("robust_fits", loaded)
            self.assertGreater((output / "robust_line_fit.png").stat().st_size, 1000)
            self.assertGreater((output / "bottleneck_tradeoff.png").stat().st_size, 1000)

    def test_invalid_probability_rejected(self) -> None:
        with self.assertRaises(ValueError):
            binary_entropy_nats(1.1)


if __name__ == "__main__":
    unittest.main()
