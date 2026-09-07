from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from materials.day003.paired_views import make_pairs, permute_y, plot_views
from materials.day003.point_stats import point_stats


class PointStatsTests(unittest.TestCase):
    def test_four_corners(self) -> None:
        result = point_stats([(0, 0), (2, 0), (2, 2), (0, 2)])
        self.assertEqual(result["centroid"], (1.0, 1.0))
        self.assertEqual(result["bbox"], ((0.0, 0.0), (2.0, 2.0)))
        self.assertEqual(result["nearest"], (0.0, 0.0))

    def test_single_point(self) -> None:
        result = point_stats([(3, -2)])
        self.assertEqual(result["centroid"], (3.0, -2.0))
        self.assertEqual(result["bbox"], ((3.0, -2.0), (3.0, -2.0)))

    def test_tie_keeps_first_point(self) -> None:
        result = point_stats([(-2, 0), (2, 0)])
        self.assertEqual(result["nearest"], (-2.0, 0.0))

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            point_stats([])

    def test_nonfinite_coordinate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            point_stats([(0, np.nan)])


class PairedViewsTests(unittest.TestCase):
    def test_pairs_are_reproducible(self) -> None:
        x1, y1 = make_pairs(seed=3, n=1000, rho=0.8)
        x2, y2 = make_pairs(seed=3, n=1000, rho=0.8)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_permutation_preserves_y_marginal(self) -> None:
        _, y = make_pairs()
        shuffled = permute_y(y)
        np.testing.assert_array_equal(np.sort(y), np.sort(shuffled))
        self.assertFalse(np.array_equal(y, shuffled))

    def test_invalid_rho_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rho"):
            make_pairs(rho=1.1)

    def test_plot_is_created(self) -> None:
        x, y = make_pairs(n=20)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "plot.png"
            plot_views(x, y, output, "Test")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
