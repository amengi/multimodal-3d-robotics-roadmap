from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from materials.day002.array_shape_lab import (
    axis_summaries,
    center_features,
    make_sensor_batch,
    project_features,
)
from materials.day002.check_terminal_tree import validate_tree


class ArrayShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = make_sensor_batch()

    def test_reproducible_shape(self) -> None:
        self.assertEqual(self.batch.shape, (2, 3, 4))
        np.testing.assert_array_equal(self.batch, make_sensor_batch())

    def test_axis_summaries(self) -> None:
        shapes = {key: value.shape for key, value in axis_summaries(self.batch).items()}
        self.assertEqual(
            shapes,
            {
                "per_sample_feature_mean": (2, 3),
                "per_sample_modality_mean": (2, 4),
                "dataset_mean": (4,),
            },
        )

    def test_broadcast_centering(self) -> None:
        centered, feature_mean = center_features(self.batch)
        self.assertEqual(feature_mean.shape, (1, 1, 4))
        self.assertEqual(centered.shape, self.batch.shape)
        np.testing.assert_allclose(centered.mean(axis=(0, 1)), 0.0, atol=1e-12)

    def test_matrix_projection(self) -> None:
        weights = np.ones((4, 2))
        self.assertEqual(project_features(self.batch, weights).shape, (2, 3, 2))

    def test_matrix_projection_rejects_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature size"):
            project_features(self.batch, np.ones((5, 2)))


class TerminalTreeTests(unittest.TestCase):
    def test_complete_tree(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("data/raw", "data/processed", "scripts", "notes"):
                (root / relative).mkdir(parents=True, exist_ok=True)
            for relative in ("README.md", "data/raw/sensors.csv", "notes/commands.md"):
                (root / relative).touch()
            (root / "data/processed/sensors_clean.csv").touch()
            self.assertEqual(validate_tree(root), [])

    def test_incomplete_tree_reports_problems(self) -> None:
        with TemporaryDirectory() as directory:
            problems = validate_tree(Path(directory))
            self.assertIn("missing directory: data/raw", problems)
            self.assertIn("missing file: README.md", problems)


if __name__ == "__main__":
    unittest.main()
