"""Regression tests for the Day 004 examples."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from materials.day004.geometry import PointDataError, load_points_csv, point_stats
from materials.day004.gradient_lab import (
    finite_difference_gradient,
    multimodal_fusion_step,
    polynomial_value_and_gradient,
)


class GeometryTests(unittest.TestCase):
    def test_point_stats(self) -> None:
        result = point_stats([(0, 0), (2, 0), (2, 2), (0, 2)])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["centroid"], (1.0, 1.0))
        self.assertEqual(result["bbox"], ((0.0, 0.0), (2.0, 2.0)))

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(PointDataError, "must not be empty"):
            point_stats([])

    def test_nonfinite_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(PointDataError, "finite"):
            point_stats([(0.0, float("nan"))])

    def test_boolean_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(PointDataError, "real numbers"):
            point_stats([(True, 1.0)])

    def test_csv_loader(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "points.csv"
            source.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
            self.assertEqual(load_points_csv(source), [(1.0, 2.0), (3.0, 4.0)])

    def test_csv_error_keeps_line_number(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "bad.csv"
            source.write_text("x,y\n1,nope\n", encoding="utf-8")
            with self.assertRaisesRegex(PointDataError, r":2"):
                load_points_csv(source)


class AutogradTests(unittest.TestCase):
    def test_polynomial_gradient(self) -> None:
        value, gradient = polynomial_value_and_gradient(2.0)
        self.assertEqual(value, 10.0)
        self.assertEqual(gradient, 7.0)

    def test_autograd_matches_finite_difference(self) -> None:
        _, gradient = polynomial_value_and_gradient(-1.25)
        self.assertAlmostEqual(gradient, finite_difference_gradient(-1.25), places=6)

    def test_nonpositive_step_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            finite_difference_gradient(2.0, step=0.0)

    def test_fusion_shapes_and_values(self) -> None:
        result = multimodal_fusion_step()
        self.assertEqual(tuple(result["features"].shape), (3, 2))
        self.assertEqual(tuple(result["predictions"].shape), (3,))
        torch.testing.assert_close(
            result["predictions"], torch.tensor([0.5, 0.5, 1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result["weight_gradients"],
            torch.tensor([-1.0 / 3.0, 1.0 / 3.0], dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
