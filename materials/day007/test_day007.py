"""Regression tests for the Day 007 weekly assessment."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from materials.day007.weekly_assessment import (
    ContractError,
    FusionMLP,
    SingleSensorLinear,
    evaluate_on,
    make_paired_split,
    majority_metrics,
    metrics_from_logits,
    run_assessment,
    shuffled_sensor_b,
    train,
    validate_xy,
)


class Day007Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.split = make_paired_split()
        cls.single = train(SingleSensorLinear(), cls.split)
        cls.fusion = train(FusionMLP(), cls.split)

    def test_split_shapes_and_disjoint_indices(self) -> None:
        self.assertEqual(tuple(self.split.train_x.shape), (360, 2, 2))
        self.assertEqual(tuple(self.split.train_y.shape), (360, 1))
        self.assertEqual(tuple(self.split.validation_x.shape), (120, 2, 2))
        self.assertEqual(tuple(self.split.validation_y.shape), (120, 1))
        self.assertFalse(set(self.split.train_indices) & set(self.split.validation_indices))

    def test_split_is_reproducible(self) -> None:
        again = make_paired_split()
        torch.testing.assert_close(self.split.train_x, again.train_x)
        torch.testing.assert_close(self.split.validation_y, again.validation_y)

    def test_training_normalization_is_per_modality_and_feature(self) -> None:
        torch.testing.assert_close(
            self.split.train_x.mean(dim=0), torch.zeros(2, 2), atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            self.split.train_x.std(dim=0, unbiased=False),
            torch.ones(2, 2),
            atol=1e-6,
            rtol=0.0,
        )

    def test_invalid_feature_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, r"\(B, 2, 2\)"):
            validate_xy(torch.ones(8, 4))

    def test_invalid_label_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, r"\(B, 1\)"):
            validate_xy(torch.ones(8, 2, 2), torch.ones(8))

    def test_nonfinite_feature_is_rejected(self) -> None:
        features = torch.ones(8, 2, 2)
        features[3, 1, 0] = torch.nan
        with self.assertRaisesRegex(ContractError, "finite"):
            validate_xy(features)

    def test_model_output_shapes(self) -> None:
        self.assertEqual(tuple(SingleSensorLinear()(torch.ones(7, 2, 2)).shape), (7, 1))
        self.assertEqual(tuple(FusionMLP()(torch.ones(7, 2, 2)).shape), (7, 1))

    def test_metrics_have_expected_values_for_confident_correct_logits(self) -> None:
        metrics = metrics_from_logits(torch.tensor([[-8.0], [8.0]]), torch.tensor([[0.0], [1.0]]))
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertLess(metrics.nll_nats, 0.001)
        self.assertLess(metrics.brier, 0.001)

    def test_majority_baseline_is_fitted_without_validation_features(self) -> None:
        metrics = majority_metrics(self.split.train_y, self.split.validation_y)
        self.assertGreaterEqual(metrics.accuracy, 0.45)
        self.assertLessEqual(metrics.accuracy, 0.55)

    def test_training_reduces_loss(self) -> None:
        self.assertLess(self.single.losses[-1], self.single.losses[0])
        self.assertLess(self.fusion.losses[-1], self.fusion.losses[0])

    def test_fusion_beats_majority_on_same_validation_split(self) -> None:
        majority = majority_metrics(self.split.train_y, self.split.validation_y)
        self.assertGreater(self.fusion.metrics.accuracy, majority.accuracy + 0.20)

    def test_shuffle_preserves_shape_but_changes_pairing(self) -> None:
        shuffled = shuffled_sensor_b(self.split.validation_x)
        self.assertEqual(tuple(shuffled.shape), tuple(self.split.validation_x.shape))
        torch.testing.assert_close(shuffled[:, 0, :], self.split.validation_x[:, 0, :])
        self.assertFalse(torch.equal(shuffled[:, 1, :], self.split.validation_x[:, 1, :]))

    def test_shuffled_pairing_hurts_fusion_accuracy(self) -> None:
        shuffled_metrics = evaluate_on(
            self.fusion.model, shuffled_sensor_b(self.split.validation_x, 20260912), self.split.validation_y
        )
        self.assertLess(shuffled_metrics.accuracy, self.fusion.metrics.accuracy - 0.10)

    def test_full_report_and_plot_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_assessment(Path(directory))
            self.assertEqual(report["train_validation_overlap"], 0)
            self.assertGreater(Path(report["loss_plot"]).stat().st_size, 0)
            self.assertGreater(Path(report["report_path"]).stat().st_size, 0)

    def test_bad_training_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "epochs"):
            train(FusionMLP(), self.split, epochs=0)


if __name__ == "__main__":
    unittest.main()
