"""Regression tests for the Day 028 weekly information audit."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from materials.day028.weekly_information_audit import (
    DEFAULT_SEED,
    N_BINS,
    SAMPLE_SIZES,
    binary_entropy_bits,
    discretize,
    entropy_bits,
    finite_sample_audit,
    generate_scenario,
    joint_pmf,
    js_bits,
    kl_bits,
    mdl_proxy_demo,
    mutual_information_bits,
    normalized_mi,
    pmf,
    prediction_baselines,
    rate_distortion_demo,
    run_experiment,
)


class Day028Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = finite_sample_audit(DEFAULT_SEED)

    def test_01_discretize_contract(self) -> None:
        labels = discretize(np.array([-10.0, -3.5, 0.0, 3.5, 10.0]))
        self.assertTrue(np.array_equal(labels, np.array([0, 0, 4, 7, 7])))

    def test_02_discretize_rejects_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            discretize(np.array([0.0, math.nan]))

    def test_03_pmf_sums_to_one(self) -> None:
        probabilities = pmf(np.array([0, 0, 1]), 3, alpha=0.5)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(probabilities > 0))

    def test_04_fair_coin_entropy(self) -> None:
        self.assertAlmostEqual(entropy_bits(np.array([0.5, 0.5])), 1.0)

    def test_05_entropy_rejects_bad_sum(self) -> None:
        with self.assertRaises(ValueError):
            entropy_bits(np.array([0.2, 0.2]))

    def test_06_kl_identity(self) -> None:
        p = np.array([0.25, 0.75])
        self.assertAlmostEqual(kl_bits(p, p), 0.0)

    def test_07_kl_infinite_without_support(self) -> None:
        self.assertTrue(math.isinf(kl_bits(np.array([0.5, 0.5]), np.array([1.0, 0.0]))))

    def test_08_js_symmetric_and_bounded(self) -> None:
        p, q = np.array([0.9, 0.1]), np.array([0.2, 0.8])
        self.assertAlmostEqual(js_bits(p, q), js_bits(q, p))
        self.assertGreaterEqual(js_bits(p, q), 0.0)
        self.assertLessEqual(js_bits(p, q), 1.0)

    def test_09_mi_known_perfect_pair(self) -> None:
        joint = np.array([[0.5, 0.0], [0.0, 0.5]])
        self.assertAlmostEqual(mutual_information_bits(joint), 1.0)
        self.assertAlmostEqual(normalized_mi(joint), 1.0)

    def test_10_joint_requires_pairing(self) -> None:
        with self.assertRaises(ValueError):
            joint_pmf(np.array([0, 1]), np.array([0]), 2)

    def test_11_unknown_scenario_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_scenario("unknown", 80, np.random.default_rng(1))

    def test_12_shift_is_unpaired(self) -> None:
        _, _, paired = generate_scenario("distribution_shift", 80, np.random.default_rng(1))
        self.assertFalse(paired)

    def test_13_baseline_returns_probabilities(self) -> None:
        x = np.tile(np.arange(N_BINS), 10)
        result = prediction_baselines(x, x)
        self.assertGreaterEqual(result["lookup_accuracy"], result["majority_accuracy"])
        self.assertLessEqual(result["lookup_accuracy"], 1.0)

    def test_14_audit_sizes_present(self) -> None:
        self.assertEqual(tuple(map(int, self.audit["linear"].keys())), SAMPLE_SIZES)

    def test_15_shift_mi_is_not_applicable(self) -> None:
        self.assertIsNone(self.audit["distribution_shift"]["800"]["mi_bits"])
        self.assertIsNone(self.audit["distribution_shift"]["800"]["pearson_r"])

    def test_16_independent_mi_shrinks_with_n(self) -> None:
        small = self.audit["independent"]["80"]["mi_bits"]["mean"]
        large = self.audit["independent"]["800"]["mi_bits"]["mean"]
        self.assertLess(large, small)

    def test_17_nonlinear_beats_pearson_only_view(self) -> None:
        nonlinear = self.audit["nonlinear"]["800"]
        self.assertLess(abs(nonlinear["pearson_r"]["mean"]), 0.1)
        self.assertGreater(nonlinear["mi_bits"]["mean"], 0.7)
        self.assertGreater(nonlinear["lookup_accuracy"]["mean"], nonlinear["majority_accuracy"]["mean"])

    def test_18_shift_js_exceeds_independent_marginal_js(self) -> None:
        shift = self.audit["distribution_shift"]["800"]["js_bits"]["mean"]
        independent = self.audit["independent"]["800"]["js_bits"]["mean"]
        self.assertGreater(shift, independent * 5)

    def test_19_binary_entropy_boundaries(self) -> None:
        self.assertEqual(binary_entropy_bits(0.0), 0.0)
        self.assertAlmostEqual(binary_entropy_bits(0.5), 1.0)

    def test_20_rate_distortion_is_nonincreasing(self) -> None:
        rates = [row["rate_bits_per_symbol"] for row in rate_distortion_demo()]
        self.assertTrue(all(left >= right for left, right in zip(rates, rates[1:])))
        self.assertEqual(rates[-1], 0.0)

    def test_21_mdl_proxy_tradeoff(self) -> None:
        models = mdl_proxy_demo()["models"]
        simple = models["marginal_y"]
        conditional = models["conditional_y_given_x"]
        self.assertGreater(conditional["parameter_count"], simple["parameter_count"])
        self.assertLess(conditional["test_nll_bits"], simple["test_nll_bits"])
        self.assertLess(conditional["total_proxy_bits"], simple["total_proxy_bits"])

    def test_22_complete_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            report = run_experiment(destination)
            self.assertEqual(report["seed"], DEFAULT_SEED)
            for filename in ("report.json", "finite_sample_metrics.png", "rate_distortion_mdl.png"):
                self.assertGreater((destination / filename).stat().st_size, 100)
            loaded = json.loads((destination / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["data_contract"]["repeats"], 40)


if __name__ == "__main__":
    unittest.main()
