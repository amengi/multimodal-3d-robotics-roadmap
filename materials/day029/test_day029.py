"""Regression tests for the Day 029 derivative and representation lab."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from materials.day029.derivative_representation_lab import (
    binary_entropy_bits,
    central_difference,
    conditional_y1,
    derivative_sweep,
    encode_joint,
    forward_difference,
    make_joint_xy,
    mutual_information_bits,
    prediction_metrics,
    representation_audit,
    run_experiment,
    sin_taylor,
    taylor_audit,
)


class Day029Tests(unittest.TestCase):
    def test_01_forward_difference_is_accurate(self) -> None:
        estimate = forward_difference(math.sin, 0.7, 1e-5)
        self.assertLess(abs(estimate - math.cos(0.7)), 1e-5)

    def test_02_central_beats_forward_at_moderate_h(self) -> None:
        forward_error = abs(forward_difference(math.sin, 0.7, 1e-3) - math.cos(0.7))
        central_error = abs(central_difference(math.sin, 0.7, 1e-3) - math.cos(0.7))
        self.assertLess(central_error, forward_error)

    def test_03_bad_step_rejected(self) -> None:
        for bad_step in (0.0, -1.0, math.nan):
            with self.assertRaises(ValueError):
                central_difference(math.sin, 0.7, bad_step)

    def test_04_quadratic_central_difference(self) -> None:
        function = lambda value: 3.0 * value**2 - 2.0 * value + 5.0
        self.assertAlmostEqual(central_difference(function, 1.5, 1e-4), 7.0, places=8)

    def test_05_taylor_matches_at_center(self) -> None:
        for order in (0, 1, 2):
            self.assertAlmostEqual(float(sin_taylor(0.7, 0.7, order)), math.sin(0.7))

    def test_06_taylor_bad_order_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sin_taylor(1.0, 0.7, 3)

    def test_07_second_order_improves_target(self) -> None:
        audit = taylor_audit()
        self.assertLess(audit["second_absolute_error"], audit["first_absolute_error"])

    def test_08_sweep_exposes_roundoff(self) -> None:
        sweep = derivative_sweep()
        errors = sweep["central_absolute_error_per_rad"]
        self.assertLess(min(errors), errors[-1])
        self.assertEqual(len(errors), 14)

    def test_09_binary_entropy_boundaries(self) -> None:
        self.assertEqual(binary_entropy_bits(0.0), 0.0)
        self.assertAlmostEqual(binary_entropy_bits(0.5), 1.0)

    def test_10_joint_is_normalized(self) -> None:
        joint = make_joint_xy()
        self.assertEqual(joint.shape, (4, 2))
        self.assertAlmostEqual(float(joint.sum()), 1.0)

    def test_11_bad_epsilon_rejected(self) -> None:
        for epsilon in (0.0, 0.5, math.nan):
            with self.assertRaises(ValueError):
                make_joint_xy(epsilon)

    def test_12_mutual_information_matches_analytic(self) -> None:
        actual = mutual_information_bits(make_joint_xy())
        expected = 1.0 - binary_entropy_bits(0.1)
        self.assertAlmostEqual(actual, expected)

    def test_13_signal_is_sufficient(self) -> None:
        audit = representation_audit()
        full = audit["representations"]["full_x"]
        signal = audit["representations"]["signal_z"]
        self.assertAlmostEqual(full["information_with_y_bits"], signal["information_with_y_bits"])
        self.assertAlmostEqual(signal["sufficiency_gap_max_probability"], 0.0)
        self.assertLess(signal["fixed_length_cost_bits"], full["fixed_length_cost_bits"])

    def test_14_nuisance_loses_task_information(self) -> None:
        nuisance = representation_audit()["representations"]["nuisance_z"]
        self.assertAlmostEqual(nuisance["information_with_y_bits"], 0.0)
        self.assertAlmostEqual(nuisance["accuracy"], 0.5)

    def test_15_data_processing_inequality(self) -> None:
        rows = representation_audit()["representations"]
        for name in ("full_x", "signal_z", "nuisance_z", "constant_z"):
            self.assertGreaterEqual(rows[name]["dpi_slack_bits"], -1e-12)

    def test_16_overconfidence_hurts_probabilistic_metrics(self) -> None:
        rows = representation_audit()["representations"]
        calibrated = rows["signal_z"]
        overconfident = rows["signal_overconfident"]
        self.assertAlmostEqual(calibrated["accuracy"], overconfident["accuracy"])
        self.assertGreater(overconfident["nll_nat_per_example"], calibrated["nll_nat_per_example"])
        self.assertGreater(overconfident["brier_per_example"], calibrated["brier_per_example"])
        self.assertGreater(overconfident["calibration_error"], calibrated["calibration_error"])

    def test_17_shift_hurts_calibration(self) -> None:
        shift = representation_audit()["robustness_shift"]
        self.assertAlmostEqual(shift["accuracy"], 0.7)
        self.assertAlmostEqual(shift["calibration_error"], 0.2)

    def test_18_encode_joint_checks_labels(self) -> None:
        with self.assertRaises(ValueError):
            encode_joint(make_joint_xy(), np.array([0, 2, 0, 2]))

    def test_19_conditional_probabilities(self) -> None:
        probabilities = conditional_y1(make_joint_xy())
        self.assertTrue(np.allclose(probabilities, np.array([0.1, 0.1, 0.9, 0.9])))

    def test_20_prediction_probability_contract(self) -> None:
        with self.assertRaises(ValueError):
            prediction_metrics(make_joint_xy(), np.array([0.0, 0.1, 0.9, 1.0]))

    def test_21_complete_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            report = run_experiment(destination)
            self.assertAlmostEqual(report["configuration"]["label_flip_probability"], 0.1)
            for filename in ("report.json", "derivative_taylor.png", "representation_audit.png"):
                self.assertGreater((destination / filename).stat().st_size, 100)
            loaded = json.loads((destination / "report.json").read_text(encoding="utf-8"))
            self.assertIn("representation", loaded)


if __name__ == "__main__":
    unittest.main()
