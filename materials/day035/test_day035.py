"""Regression tests for the Day 035 weekly project."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from materials.day035.day035_project import (
    centroid_probabilities,
    downstream_audit,
    gauss_newton_line,
    info_nce_metrics,
    line_jacobian,
    line_residuals,
    make_line_data,
    make_paired_embeddings,
    multiclass_scores,
    normalize_rows,
    numerical_jacobian,
    run,
    stable_softmax,
    total_least_squares_line,
)


class Day035Tests(unittest.TestCase):
    def test_line_data_shape_and_unit_contract(self) -> None:
        points, truth = make_line_data()
        self.assertEqual(points.shape, (120, 2))
        self.assertAlmostEqual(truth["intercept_m"], 1.2)

    def test_residual_shape(self) -> None:
        points, _ = make_line_data()
        self.assertEqual(line_residuals(np.array([1.7, 0.0]), points).shape, (120,))

    def test_rejects_wrong_point_shape(self) -> None:
        with self.assertRaises(ValueError):
            line_residuals(np.array([1.0, 0.0]), np.ones((5, 3)))

    def test_analytic_matches_numeric_jacobian(self) -> None:
        points, _ = make_line_data()
        params = np.array([1.7, 0.2])
        error = np.max(np.abs(line_jacobian(params, points) - numerical_jacobian(params, points)))
        self.assertLess(error, 1e-7)

    def test_numerical_jacobian_rejects_nonpositive_step(self) -> None:
        points, _ = make_line_data()
        with self.assertRaises(ValueError):
            numerical_jacobian(np.array([1.0, 0.0]), points, step=0)

    def test_gauss_newton_reduces_cost(self) -> None:
        points, _ = make_line_data()
        _, history = gauss_newton_line(points, np.array([1.7, 0.0]))
        self.assertLess(history[-1]["cost_m2"], history[0]["cost_m2"])

    def test_gauss_newton_matches_tls_baseline(self) -> None:
        points, _ = make_line_data()
        gn, _ = gauss_newton_line(points, np.array([1.7, 0.0]))
        tls = total_least_squares_line(points)
        self.assertLess(np.sqrt(np.mean(line_residuals(gn, points) ** 2)), 0.06)
        self.assertAlmostEqual(
            np.sqrt(np.mean(line_residuals(gn, points) ** 2)),
            np.sqrt(np.mean(line_residuals(tls, points) ** 2)),
            places=9,
        )

    def test_normalize_rows(self) -> None:
        result = normalize_rows(np.array([[3.0, 4.0], [5.0, 12.0]]))
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0)

    def test_normalize_rejects_zero_vector(self) -> None:
        with self.assertRaises(ValueError):
            normalize_rows(np.array([[0.0, 0.0]]))

    def test_softmax_rows_sum_to_one(self) -> None:
        probabilities = stable_softmax(np.array([[1000.0, 1001.0], [-1000.0, -999.0]]))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_paired_embedding_contract(self) -> None:
        image, text, labels = make_paired_embeddings()
        self.assertEqual(image.shape, (64, 8))
        self.assertEqual(text.shape, image.shape)
        self.assertEqual(labels.shape, (64,))

    def test_info_nce_metrics_are_bounded(self) -> None:
        image, text, labels = make_paired_embeddings()
        metrics = info_nce_metrics(image, text, labels, 0.07)
        self.assertGreaterEqual(metrics["strict_pair_top1"], 0.0)
        self.assertLessEqual(metrics["strict_pair_top1"], 1.0)
        self.assertGreaterEqual(metrics["semantic_top1"], metrics["strict_pair_top1"])

    def test_info_nce_rejects_temperature_zero(self) -> None:
        image, text, labels = make_paired_embeddings()
        with self.assertRaises(ValueError):
            info_nce_metrics(image, text, labels, 0.0)

    def test_false_negatives_are_counted(self) -> None:
        image, text, labels = make_paired_embeddings()
        metrics = info_nce_metrics(image, text, labels, 0.07)
        self.assertAlmostEqual(metrics["mean_same_class_negatives"], 7.0)

    def test_centroid_probability_contract(self) -> None:
        train = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        labels = np.array([0, 0, 1, 1])
        prob = centroid_probabilities(train, labels, train, 0.2)
        self.assertEqual(prob.shape, (4, 2))
        np.testing.assert_allclose(prob.sum(axis=1), 1.0)

    def test_perfect_scores(self) -> None:
        prob = np.array([[0.9, 0.1], [0.1, 0.9]])
        scores = multiclass_scores(prob, np.array([0, 1]))
        self.assertEqual(scores["accuracy"], 1.0)
        self.assertGreater(scores["nll_nats_per_example"], 0.0)

    def test_validation_test_split_is_disjoint_by_index(self) -> None:
        image, text, labels = make_paired_embeddings()
        audit = downstream_audit(image, text, labels)
        self.assertEqual(audit["split_sizes"], {"train": 32, "validation": 16, "test": 16})
        self.assertIn(audit["temperature_selected_on_validation"], [0.03, 0.07, 0.15, 0.30, 0.70, 1.00])

    def test_collapsed_semantics_expose_false_negative_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = run(Path(temporary))
            row = report["contrastive"]["false_negative_audit"][
                "two_classes_repeated_four_times_collapsed_semantics"
            ]
            self.assertLess(row["strict_pair_top1"], row["semantic_top1"])
            self.assertEqual(row["semantic_top1"], 1.0)

    def test_full_run_writes_parseable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = run(output)
            loaded = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["line_fit"]["data_shape"], [120, 2])
            self.assertEqual(report["contrastive"]["embedding_shape"], [64, 8])
            self.assertGreater((output / "gn_convergence.png").stat().st_size, 0)
            self.assertGreater((output / "contrastive_diagnostics.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
