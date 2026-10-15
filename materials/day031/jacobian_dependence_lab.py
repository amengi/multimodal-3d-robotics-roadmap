"""Day 031: projection Jacobians and paired-view dependence audits."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_POINT = np.array([0.4, -0.2, 2.0], dtype=float)
DEFAULT_INTRINSICS = np.array([800.0, 820.0, 640.0, 360.0], dtype=float)


def _finite_vector(values: np.ndarray | list[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def project_point(
    point_camera_m: np.ndarray | list[float],
    intrinsics_px: np.ndarray | list[float] = DEFAULT_INTRINSICS,
) -> np.ndarray:
    """Project one camera-frame 3D point to one image pixel."""
    point = _finite_vector(point_camera_m, (3,), "point_camera_m")
    intrinsics = _finite_vector(intrinsics_px, (4,), "intrinsics_px")
    x_m, y_m, z_m = point
    fx_px, fy_px, cx_px, cy_px = intrinsics
    if z_m <= 0.0:
        raise ValueError("camera-frame depth Z must be positive")
    if fx_px <= 0.0 or fy_px <= 0.0:
        raise ValueError("focal lengths fx and fy must be positive")
    return np.array([fx_px * x_m / z_m + cx_px, fy_px * y_m / z_m + cy_px])


def analytic_projection_jacobian(
    point_camera_m: np.ndarray | list[float],
    intrinsics_px: np.ndarray | list[float] = DEFAULT_INTRINSICS,
) -> np.ndarray:
    """Return d[u,v]/d[X,Y,Z], shape (2,3), in pixel/metre."""
    point = _finite_vector(point_camera_m, (3,), "point_camera_m")
    intrinsics = _finite_vector(intrinsics_px, (4,), "intrinsics_px")
    x_m, y_m, z_m = point
    fx_px, fy_px, _, _ = intrinsics
    if z_m <= 0.0:
        raise ValueError("camera-frame depth Z must be positive")
    if fx_px <= 0.0 or fy_px <= 0.0:
        raise ValueError("focal lengths fx and fy must be positive")
    return np.array(
        [
            [fx_px / z_m, 0.0, -fx_px * x_m / (z_m**2)],
            [0.0, fy_px / z_m, -fy_px * y_m / (z_m**2)],
        ]
    )


def central_difference_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray | list[float],
    step: float | np.ndarray,
) -> np.ndarray:
    """Numerically estimate a vector-valued Jacobian with central differences."""
    point = np.asarray(x, dtype=float)
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise ValueError("x must be a finite 1D array")
    steps = np.broadcast_to(np.asarray(step, dtype=float), point.shape)
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("every finite-difference step must be positive and finite")
    baseline = np.asarray(function(point), dtype=float)
    if baseline.ndim != 1 or not np.all(np.isfinite(baseline)):
        raise ValueError("function output must be a finite 1D array")
    jacobian = np.empty((baseline.size, point.size), dtype=float)
    for column in range(point.size):
        delta = np.zeros_like(point)
        delta[column] = steps[column]
        positive = np.asarray(function(point + delta), dtype=float)
        negative = np.asarray(function(point - delta), dtype=float)
        if positive.shape != baseline.shape or negative.shape != baseline.shape:
            raise ValueError("function output shape changed during finite differences")
        jacobian[:, column] = (positive - negative) / (2.0 * steps[column])
    return jacobian


def jacobian_sweep() -> dict[str, Any]:
    """Compare analytic and central-difference projection Jacobians across step sizes."""
    point = DEFAULT_POINT.copy()
    intrinsics = DEFAULT_INTRINSICS.copy()
    analytic = analytic_projection_jacobian(point, intrinsics)
    steps = np.logspace(-1, -11, 11)
    errors = []
    for step in steps:
        numeric = central_difference_jacobian(lambda value: project_point(value, intrinsics), point, step)
        errors.append(float(np.max(np.abs(numeric - analytic))))
    best_index = int(np.argmin(errors))
    near_point = point.copy()
    near_point[2] = 0.5
    far_norm = float(np.linalg.norm(analytic, ord="fro"))
    near_norm = float(np.linalg.norm(analytic_projection_jacobian(near_point, intrinsics), ord="fro"))
    return {
        "point_camera_m": point.tolist(),
        "intrinsics_px": {"fx": intrinsics[0], "fy": intrinsics[1], "cx": intrinsics[2], "cy": intrinsics[3]},
        "projected_pixel": project_point(point, intrinsics).tolist(),
        "analytic_jacobian_px_per_m": analytic.tolist(),
        "steps_m": steps.tolist(),
        "max_absolute_errors_px_per_m": errors,
        "best": {"step_m": float(steps[best_index]), "max_absolute_error_px_per_m": errors[best_index]},
        "depth_sensitivity": {
            "far_depth_m": 2.0,
            "near_depth_m": 0.5,
            "far_frobenius_norm_px_per_m": far_norm,
            "near_frobenius_norm_px_per_m": near_norm,
            "near_over_far": near_norm / far_norm,
        },
    }


def pearson_first_coordinate(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation between the first coordinate of paired views."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 3:
        raise ValueError("x and y must be paired 1D/2D arrays with at least three rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite")
    if np.std(x[:, 0]) == 0.0 or np.std(y[:, 0]) == 0.0:
        raise ValueError("Pearson correlation is undefined for a constant coordinate")
    return float(np.corrcoef(x[:, 0], y[:, 0])[0, 1])


def _inverse_square_root(matrix: np.ndarray, regularization: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, regularization)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def canonical_correlations(x: np.ndarray, y: np.ndarray, regularization: float = 1e-9) -> np.ndarray:
    """Compute linear CCA correlations by covariance whitening and SVD."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 3:
        raise ValueError("x and y must be paired 1D/2D arrays with at least three rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite")
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be positive and finite")
    centered_x = x - x.mean(axis=0)
    centered_y = y - y.mean(axis=0)
    denominator = x.shape[0] - 1
    covariance_x = centered_x.T @ centered_x / denominator
    covariance_y = centered_y.T @ centered_y / denominator
    cross_covariance = centered_x.T @ centered_y / denominator
    whitened = (
        _inverse_square_root(covariance_x, regularization)
        @ cross_covariance
        @ _inverse_square_root(covariance_y, regularization)
    )
    correlations = np.linalg.svd(whitened, compute_uv=False)
    return np.clip(correlations, 0.0, 1.0)


def _rbf_kernel(values: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    squared = np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2)
    positive = squared[squared > 0.0]
    if positive.size == 0:
        raise ValueError("RBF bandwidth is undefined for identical samples")
    median_squared_distance = float(np.median(positive))
    return np.exp(-squared / (2.0 * median_squared_distance)), math.sqrt(median_squared_distance)


def hsic_biased(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return the biased empirical RBF-HSIC and median-distance bandwidths."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 3:
        raise ValueError("x and y must be paired arrays with at least three rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite")
    kernel_x, bandwidth_x = _rbf_kernel(x)
    kernel_y, bandwidth_y = _rbf_kernel(y)
    kernel_x -= kernel_x.mean(axis=0, keepdims=True)
    kernel_x -= kernel_x.mean(axis=1, keepdims=True)
    kernel_x += kernel_x.mean()
    kernel_y -= kernel_y.mean(axis=0, keepdims=True)
    kernel_y -= kernel_y.mean(axis=1, keepdims=True)
    kernel_y += kernel_y.mean()
    statistic = float(np.sum(kernel_x * kernel_y) / ((x.shape[0] - 1) ** 2))
    return {"statistic": statistic, "bandwidth_x": bandwidth_x, "bandwidth_y": bandwidth_y}


def gaussian_mutual_information_bits(correlation: float) -> float:
    """Return I(X;Y) for a scalar jointly Gaussian pair with correlation rho."""
    correlation = float(correlation)
    if not math.isfinite(correlation) or abs(correlation) >= 1.0:
        raise ValueError("a nonsingular Gaussian correlation must satisfy |rho| < 1")
    return -0.5 * math.log2(1.0 - correlation**2)


def linear_probability_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, float]:
    """Fit a ridge least-squares probability baseline on train only and score test."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("features must be (N,D) and labels must be (N,)")
    train_x = features[train_indices]
    test_x = features[test_indices]
    train_y = labels[train_indices]
    test_y = labels[test_indices]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_design = np.column_stack([np.ones(train_x.shape[0]), (train_x - mean) / scale])
    test_design = np.column_stack([np.ones(test_x.shape[0]), (test_x - mean) / scale])
    penalty = np.eye(train_design.shape[1]) * 1e-6
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(train_design.T @ train_design + penalty, train_design.T @ train_y)
    probability = np.clip(test_design @ weights, 1e-6, 1.0 - 1e-6)
    predictions = probability >= 0.5
    accuracy = float(np.mean(predictions == test_y))
    brier = float(np.mean((probability - test_y) ** 2))
    nll = float(-np.mean(test_y * np.log(probability) + (1.0 - test_y) * np.log(1.0 - probability)))
    return {"accuracy": accuracy, "brier": brier, "nll_nat_per_example": nll}


def dependence_audit(seed: int = 31, sample_count: int = 400) -> dict[str, Any]:
    """Compare linear, nonlinear, shuffled, and shared-identifier paired views."""
    if sample_count < 80:
        raise ValueError("sample_count must be at least 80")
    rng = np.random.default_rng(seed)

    rho = 0.8
    gaussian_x = rng.normal(size=sample_count)
    gaussian_y = rho * gaussian_x + math.sqrt(1.0 - rho**2) * rng.normal(size=sample_count)

    nonlinear_x = rng.uniform(-2.0, 2.0, size=sample_count)
    nonlinear_y = nonlinear_x**2 + rng.normal(0.0, 0.15, size=sample_count)
    shuffled_y = nonlinear_y[rng.permutation(sample_count)]

    shared_identifier = rng.normal(size=sample_count)
    identifier_x = shared_identifier + rng.normal(0.0, 0.02, size=sample_count)
    identifier_y = shared_identifier + rng.normal(0.0, 0.02, size=sample_count)

    task_signal = rng.normal(size=sample_count)
    labels = (task_signal + rng.normal(0.0, 0.35, size=sample_count) > 0.0).astype(float)
    stable_view = np.column_stack(
        [task_signal + rng.normal(0.0, 0.75, size=sample_count), rng.normal(size=sample_count)]
    )
    second_view = np.column_stack(
        [task_signal + rng.normal(0.0, 0.55, size=sample_count), rng.normal(size=sample_count)]
    )
    identifier_views = np.column_stack([identifier_x, identifier_y])
    permutation = rng.permutation(sample_count)
    split = int(0.7 * sample_count)
    train_indices, test_indices = permutation[:split], permutation[split:]

    def row(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        hsic = hsic_biased(x, y)
        return {
            "pearson_first": pearson_first_coordinate(x, y),
            "cca_first": float(canonical_correlations(x, y)[0]),
            "rbf_hsic_biased": hsic["statistic"],
            "rbf_bandwidth_x": hsic["bandwidth_x"],
            "rbf_bandwidth_y": hsic["bandwidth_y"],
        }

    gaussian_empirical_rho = pearson_first_coordinate(gaussian_x, gaussian_y)
    dependence_rows = {
        "linear_gaussian": {
            **row(gaussian_x, gaussian_y),
            "rho_population": rho,
            "gaussian_mi_population_bit_per_pair": gaussian_mutual_information_bits(rho),
            "gaussian_mi_plugin_bit_per_pair": gaussian_mutual_information_bits(gaussian_empirical_rho),
        },
        "nonlinear": row(nonlinear_x, nonlinear_y),
        "nonlinear_shuffled": row(nonlinear_x, shuffled_y),
        "shared_identifier": row(identifier_x, identifier_y),
    }
    task_baselines = {
        "stable_single_view": linear_probability_baseline(stable_view, labels, train_indices, test_indices),
        "clean_fusion": linear_probability_baseline(
            np.column_stack([stable_view, second_view]), labels, train_indices, test_indices
        ),
        "shared_identifier_only": linear_probability_baseline(
            identifier_views, labels, train_indices, test_indices
        ),
    }
    return {
        "seed": seed,
        "sample_count": sample_count,
        "dependence": dependence_rows,
        "task_baselines": task_baselines,
        "cost": {
            "hsic_kernel_memory_mib_two_float64_matrices": 2.0 * sample_count**2 * 8.0 / (1024.0**2),
            "hsic_time_order": "O(N^2)",
            "cca_covariance_note": "CCA regularization is fixed before evaluation; no test tuning.",
        },
        "warnings": [
            "The Gaussian closed form is valid only for a scalar jointly Gaussian pair.",
            "The reported finite-sample HSIC is kernel- and bandwidth-dependent, not mutual information.",
            "High dependence from a shared identifier is not causal or task-relevant evidence.",
            "No calibration parameter or decision threshold was fitted on the test split.",
        ],
    }


def plot_jacobian_sweep(report: dict[str, Any], destination: Path) -> None:
    steps = np.asarray(report["steps_m"])
    errors = np.asarray(report["max_absolute_errors_px_per_m"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].loglog(steps, errors, marker="o")
    axes[0].invert_xaxis()
    axes[0].set(
        xlabel="central-difference step h (m)",
        ylabel="max |J_numeric - J_analytic| (pixel/m)",
        title="Jacobian finite-difference check",
    )
    axes[0].grid(alpha=0.25, which="both")
    depths = [report["depth_sensitivity"]["far_depth_m"], report["depth_sensitivity"]["near_depth_m"]]
    norms = [
        report["depth_sensitivity"]["far_frobenius_norm_px_per_m"],
        report["depth_sensitivity"]["near_frobenius_norm_px_per_m"],
    ]
    axes[1].bar(["Z=2.0 m", "Z=0.5 m"], norms, color=["#4c78a8", "#f58518"])
    axes[1].set(ylabel="||J||_F (pixel/m)", title="Projection sensitivity grows near camera")
    axes[1].grid(alpha=0.25, axis="y")
    figure.suptitle("Day 031: pinhole projection Jacobian")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_dependence_audit(report: dict[str, Any], destination: Path) -> None:
    rows = report["dependence"]
    names = ["linear_gaussian", "nonlinear", "nonlinear_shuffled", "shared_identifier"]
    labels = ["Gaussian", "nonlinear", "shuffled", "shared ID"]
    pearson = [abs(rows[name]["pearson_first"]) for name in names]
    cca = [rows[name]["cca_first"] for name in names]
    hsic = [rows[name]["rbf_hsic_biased"] for name in names]
    positions = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    width = 0.35
    axes[0].bar(positions - width / 2, pearson, width, label="|Pearson| first coordinate")
    axes[0].bar(positions + width / 2, cca, width, label="first linear CCA correlation")
    axes[0].set_xticks(positions, labels, rotation=15)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set(ylabel="correlation (dimensionless)", title="Linear dependence measures")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, axis="y")
    axes[1].bar(labels, hsic, color=["#4c78a8", "#54a24b", "#bab0ac", "#e45756"])
    axes[1].set(ylabel="biased RBF-HSIC (kernel-scale dependent)", title="Nonlinear dependence statistic")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(alpha=0.25, axis="y")
    figure.suptitle("Day 031: paired-view dependence audit")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def run_experiment(output_dir: Path, seed: int = 31, sample_count: int = 400) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jacobian = jacobian_sweep()
    dependence = dependence_audit(seed=seed, sample_count=sample_count)
    report = {"configuration": {"seed": seed, "sample_count": sample_count}, "jacobian": jacobian, "dependence": dependence}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_jacobian_sweep(jacobian, output_dir / "jacobian_sweep.png")
    plot_dependence_audit(dependence, output_dir / "dependence_audit.png")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/day031-output"))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--sample-count", type=int, default=400)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    report = run_experiment(arguments.output_dir, arguments.seed, arguments.sample_count)
    jacobian = report["jacobian"]
    dependence = report["dependence"]
    print(f"projected_pixel={np.round(jacobian['projected_pixel'], 6).tolist()}")
    print(f"best_step_m={jacobian['best']['step_m']:.1e}")
    print(f"best_jacobian_error_px_per_m={jacobian['best']['max_absolute_error_px_per_m']:.6e}")
    print(f"near_over_far_sensitivity={jacobian['depth_sensitivity']['near_over_far']:.6f}")
    for name, row in dependence["dependence"].items():
        print(
            f"{name}: pearson={row['pearson_first']:.6f}, "
            f"cca1={row['cca_first']:.6f}, hsic={row['rbf_hsic_biased']:.6f}"
        )
    for name, row in dependence["task_baselines"].items():
        print(
            f"{name}: accuracy={row['accuracy']:.6f}, "
            f"brier={row['brier']:.6f}, nll={row['nll_nat_per_example']:.6f}"
        )
    print("SUCCESS")


if __name__ == "__main__":
    main()
