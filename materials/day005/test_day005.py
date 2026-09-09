"""Regression tests for the Day 005 NumPy and PyTorch examples."""

import unittest

import numpy as np
import torch

from materials.day005.mlp_shapes import (
    FeatureError,
    TwoLayerMLP,
    make_toy_batch,
    single_optimization_step,
)
from materials.day005.vectorized_points import (
    PointArrayError,
    benchmark,
    example_transform,
    transform_points_loop,
    transform_points_vectorized,
    validate_points,
    validate_rigid_transform,
)


class NumPyPointTests(unittest.TestCase):
    def test_example_has_expected_value(self) -> None:
        points, rotation, translation = example_transform()
        result = transform_points_vectorized(points, rotation, translation)
        np.testing.assert_allclose(result[0], [10.0, 1.0, -1.0])

    def test_loop_matches_vectorized(self) -> None:
        generator = np.random.default_rng(7)
        points = generator.normal(size=(50, 3))
        _, rotation, translation = example_transform()
        np.testing.assert_allclose(
            transform_points_vectorized(points, rotation, translation),
            transform_points_loop(points, rotation, translation),
        )

    def test_translation_broadcasts_over_rows(self) -> None:
        points = np.zeros((4, 3))
        result = transform_points_vectorized(points, np.eye(3), [1, 2, 3])
        np.testing.assert_array_equal(result, np.tile([1, 2, 3], (4, 1)))

    def test_wrong_point_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(PointArrayError, r"\(N, 3\)"):
            validate_points([[1.0, 2.0]])

    def test_nonfinite_point_is_rejected(self) -> None:
        with self.assertRaisesRegex(PointArrayError, "finite"):
            validate_points([[0.0, np.nan, 1.0]])

    def test_reflection_is_not_a_rotation(self) -> None:
        reflection = np.diag([-1.0, 1.0, 1.0])
        with self.assertRaisesRegex(PointArrayError, "determinant"):
            validate_rigid_transform(reflection, np.zeros(3))

    def test_benchmark_validates_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            benchmark(point_count=0)


class MLPTests(unittest.TestCase):
    def test_model_shapes(self) -> None:
        model = TwoLayerMLP(hidden_features=8)
        features, _ = make_toy_batch()
        logits, shapes = model.forward_with_shapes(features)
        self.assertEqual(shapes["input"], (6, 2))
        self.assertEqual(shapes["hidden_linear"], (6, 8))
        self.assertEqual(tuple(logits.shape), (6, 1))

    def test_labels_match_geometric_rule(self) -> None:
        _, labels = make_toy_batch()
        torch.testing.assert_close(
            labels.squeeze(1), torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        )

    def test_wrong_feature_shape_is_rejected(self) -> None:
        model = TwoLayerMLP()
        with self.assertRaisesRegex(FeatureError, r"\(B, 2\)"):
            model(torch.ones(3, 3))

    def test_missing_value_is_rejected(self) -> None:
        model = TwoLayerMLP()
        features = torch.tensor([[0.0, float("nan")]])
        with self.assertRaisesRegex(FeatureError, "finite"):
            model(features)

    def test_single_step_is_deterministic_and_updates(self) -> None:
        first = single_optimization_step()
        second = single_optimization_step()
        self.assertGreater(first["gradient_norm"], 0.0)
        self.assertNotEqual(first["loss_before"], first["loss_after"])
        self.assertEqual(first["loss_before"], second["loss_before"])
        torch.testing.assert_close(first["probabilities"], second["probabilities"])


if __name__ == "__main__":
    unittest.main()
