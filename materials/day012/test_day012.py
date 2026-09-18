"""Regression tests for the Day 012 plotting and probability lab."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from materials.day012.plotting_probability_lab import (
    ContractError,
    equal_axis_limits,
    make_camera_trajectories,
    probability_summary,
    run,
    trajectory_errors_m,
    validate_joint_table,
    validate_trajectory,
)


class Day012Tests(unittest.TestCase):
    def test_trajectory_generation_is_deterministic(self):
        first = make_camera_trajectories(seed=7)
        second = make_camera_trajectories(seed=7)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_trajectory_shape_contract(self):
        truth, estimate = make_camera_trajectories(n_frames=12)
        self.assertEqual(truth.shape, (12, 3))
        self.assertEqual(estimate.shape, (12, 3))

    def test_wrong_trajectory_shape_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "shape"):
            validate_trajectory(np.zeros((5, 2)))

    def test_nonfinite_trajectory_is_rejected(self):
        value = np.zeros((3, 3))
        value[0, 0] = np.nan
        with self.assertRaisesRegex(ContractError, "finite"):
            validate_trajectory(value)

    def test_identical_trajectory_has_zero_error(self):
        truth, _ = make_camera_trajectories(n_frames=8)
        np.testing.assert_array_equal(trajectory_errors_m(truth, truth), 0.0)

    def test_known_trajectory_error(self):
        truth = np.zeros((3, 3))
        estimate = np.tile([3.0, 4.0, 0.0], (3, 1))
        np.testing.assert_allclose(trajectory_errors_m(truth, estimate), 5.0)

    def test_shifted_pairing_is_worse(self):
        truth, estimate = make_camera_trajectories()
        aligned = trajectory_errors_m(truth, estimate)
        shifted = trajectory_errors_m(truth, np.roll(estimate, 1, axis=0))
        self.assertGreater(float(np.sqrt(np.mean(shifted**2))), float(np.sqrt(np.mean(aligned**2))))

    def test_equal_limits_have_same_span(self):
        truth, _ = make_camera_trajectories()
        limits = equal_axis_limits(truth)
        spans = [upper - lower for lower, upper in limits]
        np.testing.assert_allclose(spans, spans[0])

    def test_degenerate_axis_input_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "nonzero spatial extent"):
            equal_axis_limits(np.ones((3, 3)))

    def test_joint_table_contract(self):
        joint = validate_joint_table([[0.54, 0.06], [0.08, 0.32]])
        self.assertAlmostEqual(float(joint.sum()), 1.0)

    def test_negative_probability_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "nonnegative"):
            validate_joint_table([[0.7, -0.1], [0.1, 0.3]])

    def test_probability_sum_is_checked(self):
        with self.assertRaisesRegex(ContractError, "sum to 1"):
            validate_joint_table([[0.4, 0.1], [0.1, 0.1]])

    def test_zero_conditioning_marginal_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "marginals"):
            validate_joint_table([[0.5, 0.0], [0.5, 0.0]])

    def test_conditional_probability_and_bayes_agree(self):
        summary = probability_summary([[0.54, 0.06], [0.08, 0.32]])
        self.assertAlmostEqual(summary["p_x1"], 0.40)
        self.assertAlmostEqual(summary["p_y1"], 0.38)
        self.assertAlmostEqual(summary["p_x1_given_y1"], 0.32 / 0.38)
        self.assertAlmostEqual(summary["p_x1_given_y1"], summary["bayes_p_x1_given_y1"])

    def test_full_run_writes_strict_report_and_two_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run(Path(directory))
            self.assertGreater(report["shifted_pairing_rmse_m"], report["aligned_rmse_m"])
            self.assertEqual(len(report["figures"]), 2)
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in report["figures"]))
            with Path(report["report_path"]).open(encoding="utf-8") as handle:
                parsed = json.load(handle, parse_constant=lambda value: self.fail(value))
            self.assertEqual(parsed["trajectory_contract"]["shape"], [60, 3])


if __name__ == "__main__":
    unittest.main()
