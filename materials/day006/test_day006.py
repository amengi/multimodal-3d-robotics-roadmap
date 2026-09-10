"""Regression tests for the Day 006 Git and training examples."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from materials.day006.compare_methods import (
    accuracy,
    fit_class_centroids,
    nearest_centroid_predict,
)
from materials.day006.git_sandbox import GitLabError, create_conflict_lab
from materials.day006.train_moons import (
    MoonMLP,
    TrainingDataError,
    decision_grid,
    evaluate,
    make_moon_split,
    save_plots,
    train_model,
    validate_batch,
)


class GitSandboxTests(unittest.TestCase):
    def test_lab_stops_at_expected_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "git-lab"
            result = create_conflict_lab(target)
            self.assertNotEqual(result["merge_returncode"], 0)
            self.assertIn("UU experiment.cfg", result["status"])
            self.assertIn("<<<<<<< HEAD", result["conflicted_text"])
            self.assertTrue((target / ".git" / "MERGE_HEAD").exists())

    def test_lab_has_three_premerge_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "git-lab"
            create_conflict_lab(target)
            completed = subprocess.run(
                ["git", "rev-list", "--all", "--count"],
                cwd=target,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(completed.stdout.strip(), "3")

    def test_nonempty_target_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "git-lab"
            target.mkdir()
            (target / "keep.txt").write_text("do not overwrite", encoding="utf-8")
            with self.assertRaisesRegex(GitLabError, "refusing to overwrite"):
                create_conflict_lab(target)

    def test_target_inside_existing_repository_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory) / "outer"
            outer.mkdir()
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=outer,
                text=True,
                capture_output=True,
                check=True,
            )
            with self.assertRaisesRegex(GitLabError, "outside an existing"):
                create_conflict_lab(outer / "nested-lab")


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.split = make_moon_split()
        cls.result = train_model(cls.split)

    def test_split_is_disjoint_and_has_expected_shapes(self) -> None:
        self.assertEqual(tuple(self.split.train_features.shape), (300, 2))
        self.assertEqual(tuple(self.split.train_labels.shape), (300, 1))
        self.assertEqual(tuple(self.split.validation_features.shape), (100, 2))
        self.assertEqual(tuple(self.split.validation_labels.shape), (100, 1))
        self.assertFalse(
            set(self.split.train_indices.tolist())
            & set(self.split.validation_indices.tolist())
        )

    def test_split_is_reproducible(self) -> None:
        again = make_moon_split()
        torch.testing.assert_close(self.split.train_features, again.train_features)
        np.testing.assert_array_equal(self.split.validation_indices, again.validation_indices)

    def test_standardization_uses_valid_training_statistics(self) -> None:
        torch.testing.assert_close(
            self.split.train_features.mean(dim=0),
            torch.zeros(2),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            self.split.train_features.std(dim=0, unbiased=False),
            torch.ones(2),
            atol=1e-6,
            rtol=0.0,
        )

    def test_invalid_feature_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(TrainingDataError, r"\(B, 2\)"):
            validate_batch(torch.ones(8, 3))

    def test_invalid_label_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(TrainingDataError, r"\(B, 1\)"):
            validate_batch(torch.ones(8, 2), torch.ones(8))

    def test_model_output_shape(self) -> None:
        model = MoonMLP()
        self.assertEqual(tuple(model(torch.ones(7, 2)).shape), (7, 1))

    def test_training_reduces_loss_and_reaches_useful_accuracy(self) -> None:
        self.assertLess(self.result.train_losses[-1], self.result.train_losses[0])
        self.assertGreaterEqual(self.result.validation_accuracy, 0.85)

    def test_evaluation_does_not_create_gradients(self) -> None:
        model = MoonMLP()
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        evaluate(model, self.split.validation_features, self.split.validation_labels)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_decision_grid_shapes_match(self) -> None:
        grid_x, grid_y, probabilities = decision_grid(
            self.result.model, self.split.validation_features, step=0.1
        )
        self.assertEqual(grid_x.shape, grid_y.shape)
        self.assertEqual(grid_x.shape, probabilities.shape)
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))

    def test_plots_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = save_plots(self.split, self.result, Path(directory))
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_nearest_centroid_baseline_uses_matching_shapes(self) -> None:
        centroids = fit_class_centroids(
            self.split.train_features, self.split.train_labels
        )
        predictions = nearest_centroid_predict(
            self.split.validation_features, centroids
        )
        score = accuracy(predictions, self.split.validation_labels)
        self.assertEqual(tuple(centroids.shape), (2, 2))
        self.assertEqual(tuple(predictions.shape), (100, 1))
        self.assertGreaterEqual(score, 0.70)

    def test_bad_training_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "epochs"):
            train_model(self.split, epochs=0)


if __name__ == "__main__":
    unittest.main()
