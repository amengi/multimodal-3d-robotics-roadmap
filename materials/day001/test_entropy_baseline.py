"""Tests for the Day 001 Shannon entropy implementation."""

import math
import unittest

from materials.day001.entropy_baseline import shannon_entropy


class ShannonEntropyTests(unittest.TestCase):
    def test_fair_binary_distribution_is_one_bit(self) -> None:
        self.assertAlmostEqual(shannon_entropy([0.5, 0.5]), 1.0)

    def test_certain_distribution_is_zero(self) -> None:
        self.assertEqual(shannon_entropy([1.0, 0.0]), 0.0)

    def test_uniform_four_class_distribution_is_two_bits(self) -> None:
        self.assertAlmostEqual(shannon_entropy([0.25] * 4), 2.0)

    def test_zero_probability_contributes_zero(self) -> None:
        self.assertAlmostEqual(shannon_entropy([0.5, 0.5, 0.0]), 1.0)

    def test_probabilities_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            shannon_entropy([0.2, 0.2])

    def test_negative_probability_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            shannon_entropy([1.1, -0.1])


if __name__ == "__main__":
    unittest.main()
