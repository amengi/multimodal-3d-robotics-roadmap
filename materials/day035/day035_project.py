"""Day 035: Gauss–Newton line fitting and contrastive representation audit.

All datasets are deterministic synthetic teaching data. The contrastive section
computes an InfoNCE objective; it does not claim to estimate the true mutual
information of a real image–text distribution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized values."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape (N, D)")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError("cannot normalize a zero vector")
    return values / norms


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax after subtracting the per-axis maximum."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def make_line_data(seed: int = 35, n_points: int = 120) -> tuple[np.ndarray, dict]:
    """Create noisy 2D points around y = 0.7 x + 1.2, in metres."""
    if n_points < 8:
        raise ValueError("n_points must be at least 8")
    rng = np.random.default_rng(seed)
    slope = 0.7
    intercept = 1.2
    x = np.linspace(-3.0, 3.0, n_points)
    y = slope * x + intercept
    normal = np.array([-slope, 1.0])
    normal /= np.linalg.norm(normal)
    orthogonal_noise_m = rng.normal(0.0, 0.04, size=(n_points, 1))
    points_m = np.column_stack([x, y]) + orthogonal_noise_m * normal
    truth = {
        "slope": slope,
        "intercept_m": intercept,
        "normal": normal,
        "rho_m": float(intercept / math.sqrt(1.0 + slope**2)),
    }
    return points_m, truth


def line_residuals(params: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    """Signed orthogonal residuals for x*cos(theta)+y*sin(theta)-rho."""
    params = np.asarray(params, dtype=float)
    points_m = np.asarray(points_m, dtype=float)
    if params.shape != (2,):
        raise ValueError("params must have shape (2,) = [theta_rad, rho_m]")
    if points_m.ndim != 2 or points_m.shape[1] != 2:
        raise ValueError("points_m must have shape (N, 2)")
    if not np.all(np.isfinite(params)) or not np.all(np.isfinite(points_m)):
        raise ValueError("inputs must be finite")
    theta_rad, rho_m = params
    normal = np.array([math.cos(theta_rad), math.sin(theta_rad)])
    return points_m @ normal - rho_m


def line_jacobian(params: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    """Analytic Jacobian dr/d[theta,rho], shape (N, 2)."""
    theta_rad = float(np.asarray(params, dtype=float)[0])
    x_m = np.asarray(points_m, dtype=float)[:, 0]
    y_m = np.asarray(points_m, dtype=float)[:, 1]
    return np.column_stack(
        [-x_m * math.sin(theta_rad) + y_m * math.cos(theta_rad), -np.ones_like(x_m)]
    )


def numerical_jacobian(
    params: np.ndarray, points_m: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    """Central-difference Jacobian of line_residuals."""
    if step <= 0:
        raise ValueError("step must be positive")
    params = np.asarray(params, dtype=float)
    columns = []
    for column in range(params.size):
        delta = np.zeros_like(params)
        delta[column] = step
        columns.append(
            (line_residuals(params + delta, points_m) - line_residuals(params - delta, points_m))
            / (2.0 * step)
        )
    return np.column_stack(columns)


def gauss_newton_line(
    points_m: np.ndarray,
    initial_params: np.ndarray,
    max_iterations: int = 20,
    damping: float = 1e-8,
) -> tuple[np.ndarray, list[dict]]:
    """Minimize 1/2 sum_i r_i^2 with damped Gauss–Newton steps."""
    if max_iterations < 1 or damping < 0:
        raise ValueError("invalid optimizer settings")
    params = np.asarray(initial_params, dtype=float).copy()
    history: list[dict] = []
    for iteration in range(max_iterations):
        residual = line_residuals(params, points_m)
        jacobian = line_jacobian(params, points_m)
        cost_m2 = 0.5 * float(residual @ residual)
        gradient = jacobian.T @ residual
        hessian_approx = jacobian.T @ jacobian + damping * np.eye(2)
        step = np.linalg.solve(hessian_approx, -gradient)
        history.append(
            {
                "iteration": iteration,
                "cost_m2": cost_m2,
                "step_norm": float(np.linalg.norm(step)),
                "gradient_norm": float(np.linalg.norm(gradient)),
            }
        )
        params += step
        if np.linalg.norm(step) < 1e-11:
            break
    return params, history


def total_least_squares_line(points_m: np.ndarray) -> np.ndarray:
    """Closed-form geometric baseline using the smallest covariance eigenvector."""
    points_m = np.asarray(points_m, dtype=float)
    center = points_m.mean(axis=0)
    centered = points_m - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    if normal[1] < 0:
        normal = -normal
    return np.array([math.atan2(normal[1], normal[0]), float(normal @ center)])


def line_slope_intercept(params: np.ndarray) -> tuple[float, float]:
    """Convert normal-form parameters to y = slope*x + intercept."""
    theta_rad, rho_m = np.asarray(params, dtype=float)
    sin_theta = math.sin(theta_rad)
    if abs(sin_theta) < 1e-8:
        raise ValueError("vertical line has no finite slope-intercept form")
    return -math.cos(theta_rad) / sin_theta, rho_m / sin_theta


def make_paired_embeddings(
    seed: int = 350, n_classes: int = 8, per_class: int = 8, dimension: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create aligned synthetic image/text embeddings with repeated semantics."""
    if n_classes < 2 or per_class < 4 or dimension < n_classes:
        raise ValueError("need >=2 classes, >=4 pairs/class, and dimension >= n_classes")
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    centers = 0.8 * basis[:n_classes]
    labels = np.repeat(np.arange(n_classes), per_class)
    instance = centers[labels] + 0.35 * rng.normal(size=(labels.size, dimension))
    image = normalize_rows(instance + 0.12 * rng.normal(size=instance.shape))
    text = normalize_rows(instance + 0.12 * rng.normal(size=instance.shape))
    order = rng.permutation(labels.size)
    return image[order], text[order], labels[order]


def info_nce_metrics(
    image: np.ndarray, text: np.ndarray, labels: np.ndarray, temperature: float
) -> dict:
    """Compute symmetric InfoNCE and strict/semantic retrieval metrics."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    image = normalize_rows(image)
    text = normalize_rows(text)
    labels = np.asarray(labels)
    if image.shape != text.shape or labels.shape != (image.shape[0],):
        raise ValueError("paired embeddings/labels have inconsistent shapes")
    n = image.shape[0]
    logits = image @ text.T / temperature
    row_prob = stable_softmax(logits, axis=1)
    col_prob = stable_softmax(logits, axis=0)
    target = np.arange(n)
    loss_i2t = -float(np.mean(np.log(np.clip(row_prob[target, target], EPS, 1.0))))
    loss_t2i = -float(np.mean(np.log(np.clip(col_prob[target, target], EPS, 1.0))))
    i2t_choice = np.argmax(logits, axis=1)
    t2i_choice = np.argmax(logits, axis=0)
    strict_top1 = 0.5 * (
        np.mean(i2t_choice == target) + np.mean(t2i_choice == target)
    )
    semantic_top1 = 0.5 * (
        np.mean(labels[i2t_choice] == labels) + np.mean(labels[t2i_choice] == labels)
    )
    same_class_negatives = np.mean(
        np.sum(labels[:, None] == labels[None, :], axis=1) - 1
    )
    loss_nats = 0.5 * (loss_i2t + loss_t2i)
    return {
        "temperature": float(temperature),
        "batch_size": int(n),
        "loss_nats_per_pair": loss_nats,
        "batch_bound_bits": float((math.log(n) - loss_nats) / math.log(2.0)),
        "strict_pair_top1": float(strict_top1),
        "semantic_top1": float(semantic_top1),
        "mean_same_class_negatives": float(same_class_negatives),
    }


def multiclass_scores(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Return discrimination and calibration summaries."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n, n_classes = probabilities.shape
    target_prob = probabilities[np.arange(n), labels]
    predictions = np.argmax(probabilities, axis=1)
    one_hot = np.eye(n_classes)[labels]
    confidence = np.max(probabilities, axis=1)
    correct = predictions == labels
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if np.any(in_bin):
            ece += np.mean(in_bin) * abs(np.mean(correct[in_bin]) - np.mean(confidence[in_bin]))
    return {
        "accuracy": float(np.mean(correct)),
        "nll_nats_per_example": -float(np.mean(np.log(np.clip(target_prob, EPS, 1.0)))),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_5_bins": float(ece),
    }


def centroid_probabilities(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Predict classes from negative squared distance to training centroids."""
    classes = np.unique(train_labels)
    if not np.array_equal(classes, np.arange(classes.size)):
        raise ValueError("labels must be consecutive integers starting at zero")
    centroids = np.stack([train_embeddings[train_labels == label].mean(axis=0) for label in classes])
    squared_distance = np.sum((query_embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return stable_softmax(-squared_distance / temperature, axis=1)


def downstream_audit(image: np.ndarray, text: np.ndarray, labels: np.ndarray) -> dict:
    """Fit centroids on train, tune temperature on validation, evaluate once on test."""
    fused = normalize_rows(image + text)
    train_idx: list[int] = []
    validation_idx: list[int] = []
    test_idx: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        train_idx.extend(indices[:4])
        validation_idx.extend(indices[4:6])
        test_idx.extend(indices[6:8])
    train_idx = np.asarray(train_idx)
    validation_idx = np.asarray(validation_idx)
    test_idx = np.asarray(test_idx)
    grid = [0.03, 0.07, 0.15, 0.30, 0.70, 1.00]
    validation_rows = []
    for temperature in grid:
        prob = centroid_probabilities(
            fused[train_idx], labels[train_idx], fused[validation_idx], temperature
        )
        scores = multiclass_scores(prob, labels[validation_idx])
        validation_rows.append({"temperature": temperature, **scores})
    best = min(validation_rows, key=lambda row: row["nll_nats_per_example"])
    test_prob = centroid_probabilities(
        fused[train_idx], labels[train_idx], fused[test_idx], best["temperature"]
    )
    raw_test_prob = centroid_probabilities(
        fused[train_idx], labels[train_idx], fused[test_idx], 1.0
    )
    n_classes = np.unique(labels).size
    uniform_prob = np.full((test_idx.size, n_classes), 1.0 / n_classes)
    return {
        "split_sizes": {
            "train": int(train_idx.size),
            "validation": int(validation_idx.size),
            "test": int(test_idx.size),
        },
        "temperature_selected_on_validation": float(best["temperature"]),
        "validation_scan": validation_rows,
        "test_selected_temperature": multiclass_scores(test_prob, labels[test_idx]),
        "test_uncalibrated_temperature_1": multiclass_scores(raw_test_prob, labels[test_idx]),
        "test_uniform_baseline": multiclass_scores(uniform_prob, labels[test_idx]),
    }


def select_false_negative_batches(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-size diverse and repeated-semantic index sets."""
    unique_batch = np.array([np.flatnonzero(labels == label)[0] for label in range(8)])
    repeated_batch = np.concatenate(
        [np.flatnonzero(labels == label)[:4] for label in range(2)]
    )
    return unique_batch, repeated_batch


def save_line_figure(
    points_m: np.ndarray, gn_params: np.ndarray, tls_params: np.ndarray, history: list[dict], path: Path
) -> None:
    slope_gn, intercept_gn = line_slope_intercept(gn_params)
    slope_tls, intercept_tls = line_slope_intercept(tls_params)
    x_grid = np.linspace(points_m[:, 0].min(), points_m[:, 0].max(), 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(points_m[:, 0], points_m[:, 1], s=12, alpha=0.65, label="points")
    axes[0].plot(x_grid, slope_gn * x_grid + intercept_gn, label="Gauss–Newton")
    axes[0].plot(x_grid, slope_tls * x_grid + intercept_tls, "--", label="TLS baseline")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Orthogonal line fit")
    axes[0].legend()
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].semilogy(
        [row["iteration"] for row in history],
        [max(row["cost_m2"], EPS) for row in history],
        marker="o",
    )
    axes[1].set(xlabel="iteration", ylabel="0.5 sum residual^2 (m^2)", title="GN convergence")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_contrastive_figure(temperature_rows: list[dict], batch_rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    temperatures = [row["temperature"] for row in temperature_rows]
    axes[0].plot(temperatures, [row["loss_nats_per_pair"] for row in temperature_rows], marker="o")
    axes[0].set(xlabel="InfoNCE temperature", ylabel="loss (nat/pair)", title="Temperature scan")
    axes[0].set_xscale("log")
    axes[0].set_xticks(temperatures, labels=[f"{value:.2f}" for value in temperatures])
    axes[0].minorticks_off()
    axes[0].grid(alpha=0.3)
    batch_sizes = [row["batch_size"] for row in batch_rows]
    axes[1].plot(batch_sizes, [row["strict_pair_top1"] for row in batch_rows], marker="o", label="strict pair")
    axes[1].plot(batch_sizes, [row["semantic_top1"] for row in batch_rows], marker="s", label="semantic")
    axes[1].set(xlabel="batch size", ylabel="top-1", ylim=(0, 1.05), title="Retrieval is definition-dependent")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    points_m, truth = make_line_data()
    initial = np.array([1.7, 0.0])
    analytic = line_jacobian(initial, points_m)
    numeric = numerical_jacobian(initial, points_m)
    jacobian_max_abs_error = float(np.max(np.abs(analytic - numeric)))
    gn_params, history = gauss_newton_line(points_m, initial)
    tls_params = total_least_squares_line(points_m)
    gn_slope, gn_intercept = line_slope_intercept(gn_params)
    tls_slope, tls_intercept = line_slope_intercept(tls_params)
    gn_rmse_m = float(np.sqrt(np.mean(line_residuals(gn_params, points_m) ** 2)))
    tls_rmse_m = float(np.sqrt(np.mean(line_residuals(tls_params, points_m) ** 2)))

    image, text, labels = make_paired_embeddings()
    temperatures = [0.03, 0.07, 0.15, 0.30]
    temperature_rows = [info_nce_metrics(image, text, labels, t) for t in temperatures]
    batch_rows = [info_nce_metrics(image[:n], text[:n], labels[:n], 0.07) for n in [8, 16, 32, 64]]
    unique_idx, repeated_idx = select_false_negative_batches(labels)
    repeated_labels = labels[repeated_idx]
    collapsed_image = normalize_rows(
        np.stack([image[labels == label].mean(axis=0) for label in repeated_labels])
    )
    collapsed_text = normalize_rows(
        np.stack([text[labels == label].mean(axis=0) for label in repeated_labels])
    )
    false_negative_audit = {
        "eight_distinct_classes": info_nce_metrics(
            image[unique_idx], text[unique_idx], labels[unique_idx], 0.07
        ),
        "two_classes_repeated_four_times_collapsed_semantics": info_nce_metrics(
            collapsed_image, collapsed_text, repeated_labels, 0.07
        ),
    }
    downstream = downstream_audit(image, text, labels)

    line_figure = output_dir / "gn_convergence.png"
    contrastive_figure = output_dir / "contrastive_diagnostics.png"
    save_line_figure(points_m, gn_params, tls_params, history, line_figure)
    save_contrastive_figure(temperature_rows, batch_rows, contrastive_figure)

    report = {
        "seed": {"line": 35, "paired_embeddings": 350},
        "line_fit": {
            "data_shape": list(points_m.shape),
            "point_unit": "metre",
            "frame": "synthetic world xy",
            "initial_theta_rad_rho_m": initial.tolist(),
            "truth_slope_intercept_m": [truth["slope"], truth["intercept_m"]],
            "jacobian_max_abs_error": jacobian_max_abs_error,
            "gauss_newton_slope_intercept_m": [gn_slope, gn_intercept],
            "tls_slope_intercept_m": [tls_slope, tls_intercept],
            "gauss_newton_rmse_m": gn_rmse_m,
            "tls_rmse_m": tls_rmse_m,
            "iterations": len(history),
            "initial_final_cost_m2": [history[0]["cost_m2"], history[-1]["cost_m2"]],
        },
        "contrastive": {
            "embedding_shape": list(image.shape),
            "similarity_shape": [image.shape[0], image.shape[0]],
            "temperature_scan": temperature_rows,
            "batch_scan": batch_rows,
            "false_negative_audit": false_negative_audit,
            "downstream_classification": downstream,
            "claim_boundary": (
                "log(batch_size)-InfoNCE is a batch-dependent lower-bound diagnostic, "
                "not a measurement of true mutual information or causality"
            ),
        },
        "artifacts": [line_figure.name, contrastive_figure.name],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    assert jacobian_max_abs_error < 1e-7
    assert gn_rmse_m <= tls_rmse_m + 1e-10
    assert abs(gn_slope - tls_slope) < 1e-7
    assert downstream["split_sizes"] == {"train": 32, "validation": 16, "test": 16}
    assert line_figure.stat().st_size > 0 and contrastive_figure.stat().st_size > 0
    print(f"Jacobian max abs error: {jacobian_max_abs_error:.3e}")
    print(f"GN/TLS orthogonal RMSE: {gn_rmse_m:.6f}/{tls_rmse_m:.6f} m")
    selected = downstream["test_selected_temperature"]
    print(
        "Downstream test accuracy/NLL/Brier/ECE: "
        f"{selected['accuracy']:.3f}/{selected['nll_nats_per_example']:.3f}/"
        f"{selected['multiclass_brier']:.3f}/{selected['ece_5_bins']:.3f}"
    )
    print(f"Wrote {report_path}")
    print("SUCCESS")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/day035_outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output_dir)
