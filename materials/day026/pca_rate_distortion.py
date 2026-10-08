"""Day 026: PCA geometry and a task-aware empirical rate--distortion scan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PCAFit:
    mean: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def validate_points(points_m: np.ndarray) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("points_m must have shape (N, 2), N >= 2")
    if not np.isfinite(points).all():
        raise ValueError("points_m must contain only finite values")
    return points


def fit_pca(points_m: np.ndarray) -> PCAFit:
    points = validate_points(points_m)
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / (points.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for column in range(2):
        pivot = int(np.argmax(np.abs(eigenvectors[:, column])))
        if eigenvectors[pivot, column] < 0.0:
            eigenvectors[:, column] *= -1.0
    return PCAFit(mean, covariance, eigenvalues, eigenvectors)


def project(points_m: np.ndarray, fit: PCAFit) -> np.ndarray:
    return (validate_points(points_m) - fit.mean) @ fit.eigenvectors


def reconstruct(projected_m: np.ndarray, fit: PCAFit) -> np.ndarray:
    projected = np.asarray(projected_m, dtype=np.float64)
    if projected.ndim != 2 or projected.shape[1] != 2:
        raise ValueError("projected_m must have shape (N, 2)")
    if not np.isfinite(projected).all():
        raise ValueError("projected_m must be finite")
    return fit.mean + projected @ fit.eigenvectors.T


def fit_quantizer_ranges(projected_calibration_m: np.ndarray) -> np.ndarray:
    projected = np.asarray(projected_calibration_m, dtype=np.float64)
    if projected.ndim != 2 or projected.shape[1] != 2:
        raise ValueError("projected_calibration_m must have shape (N, 2)")
    if not np.isfinite(projected).all():
        raise ValueError("projected_calibration_m must be finite")
    maxima = np.max(np.abs(projected), axis=0)
    maxima = np.maximum(maxima * 1.05, np.finfo(np.float64).eps)
    return np.column_stack([-maxima, maxima])


def uniform_quantize(values: np.ndarray, bits: int, lower: float, upper: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if bits < 0:
        raise ValueError("bits must be non-negative")
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError("lower and upper must be finite with lower < upper")
    if bits == 0:
        return np.zeros_like(array)
    levels = 2**bits
    width = (upper - lower) / levels
    clipped = np.clip(array, lower, np.nextafter(upper, -np.inf))
    indices = np.floor((clipped - lower) / width)
    return lower + (indices + 0.5) * width


def allocation_for(policy: str, rate_bits_per_point: int) -> tuple[int, int]:
    if rate_bits_per_point < 0:
        raise ValueError("rate_bits_per_point must be non-negative")
    if policy == "variance":
        return rate_bits_per_point, 0
    if policy == "task_aware":
        if rate_bits_per_point == 0:
            return 0, 0
        return rate_bits_per_point - 1, 1
    raise ValueError("policy must be 'variance' or 'task_aware'")


def encode_decode(
    points_m: np.ndarray,
    fit: PCAFit,
    ranges_m: np.ndarray,
    allocation_bits: tuple[int, int],
) -> np.ndarray:
    coordinates = project(points_m, fit)
    ranges = np.asarray(ranges_m, dtype=np.float64)
    if ranges.shape != (2, 2):
        raise ValueError("ranges_m must have shape (2, 2)")
    quantized = np.column_stack(
        [
            uniform_quantize(
                coordinates[:, component],
                allocation_bits[component],
                ranges[component, 0],
                ranges[component, 1],
            )
            for component in range(2)
        ]
    )
    return fit.mean + quantized @ fit.eigenvectors.T


def line_angle_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0.0, 1.0))))


def direction_angle_deg(direction: np.ndarray) -> float:
    vector = np.asarray(direction, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    return float(np.degrees(np.arccos(np.clip(abs(vector[0]), 0.0, 1.0))))


def make_dataset(
    seed: int = 2608, n_calibration: int = 600, n_evaluation: int = 600
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_calibration + n_evaluation
    theta = np.deg2rad(30.0)
    major_axis = np.array([np.cos(theta), np.sin(theta)])
    task_axis = np.array([-np.sin(theta), np.cos(theta)])
    major_coordinate_m = rng.normal(0.0, 2.5, size=n)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    task_coordinate_m = signs * rng.uniform(0.16, 0.28, size=n)
    points_m = (
        np.array([1.0, -0.5])
        + major_coordinate_m[:, None] * major_axis
        + task_coordinate_m[:, None] * task_axis
    )
    labels = (task_coordinate_m >= 0.0).astype(np.int64)
    return points_m, labels, major_axis, task_axis


def _rmse_m(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((reference - estimate) ** 2, axis=1))))


def _task_accuracy(
    reconstructed_m: np.ndarray, fit: PCAFit, orientation: float, labels: np.ndarray
) -> float:
    task_scores = project(reconstructed_m, fit)[:, 1] * orientation
    # A zero-bit task coordinate should remain an exact decision-boundary tie.
    # Reconstructing and projecting can otherwise introduce signed round-off at
    # roughly 1e-16 and make the documented >= 0 tie rule platform-dependent.
    task_scores[np.isclose(task_scores, 0.0, atol=1e-12, rtol=0.0)] = 0.0
    predicted = (task_scores >= 0.0).astype(np.int64)
    return float(np.mean(predicted == labels))


def isotropic_failure(seed: int = 2608, trials: int = 80, n: int = 40) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    orientations: list[float] = []
    gaps: list[float] = []
    for _ in range(trials):
        cloud = rng.normal(0.0, 1.0, size=(n, 2))
        fit = fit_pca(cloud)
        raw_angle = float(np.degrees(np.arctan2(fit.eigenvectors[1, 0], fit.eigenvectors[0, 0])))
        orientations.append(raw_angle % 180.0)
        gaps.append(float((fit.eigenvalues[0] - fit.eigenvalues[1]) / fit.eigenvalues[0]))
    return {
        "angle_range_deg": float(max(orientations) - min(orientations)),
        "median_eigengap_ratio": float(np.median(gaps)),
        "trials": float(trials),
    }


def run_experiment(output_dir: Path, seed: int = 2608) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    points_m, labels, major_axis, _ = make_dataset(seed=seed)
    calibration_m, evaluation_m = points_m[:600], points_m[600:]
    evaluation_labels = labels[600:]
    fit = fit_pca(calibration_m)
    ranges_m = fit_quantizer_ranges(project(calibration_m, fit))
    calibration_task_scores = project(calibration_m, fit)[:, 1]
    class_one_mean = float(calibration_task_scores[labels[:600] == 1].mean())
    class_zero_mean = float(calibration_task_scores[labels[:600] == 0].mean())
    orientation = 1.0 if class_one_mean > class_zero_mean else -1.0

    raw_accuracy = _task_accuracy(evaluation_m, fit, orientation, evaluation_labels)
    codec_rows: list[dict[str, float | int | str | list[int]]] = []
    for rate in (0, 1, 2, 4, 6, 8):
        for policy in ("variance", "task_aware"):
            allocation = allocation_for(policy, rate)
            reconstructed = encode_decode(evaluation_m, fit, ranges_m, allocation)
            codec_rows.append(
                {
                    "policy": policy,
                    "rate_bits_per_point": rate,
                    "allocation_bits": list(allocation),
                    "rmse_m": _rmse_m(evaluation_m, reconstructed),
                    "point_mse_m2": float(
                        np.mean(np.sum((evaluation_m - reconstructed) ** 2, axis=1))
                    ),
                    "task_accuracy": _task_accuracy(reconstructed, fit, orientation, evaluation_labels),
                }
            )

    report: dict[str, object] = {
        "seed": seed,
        "calibration_shape": [600, 2],
        "evaluation_shape": [600, 2],
        "coordinate_unit": "m",
        "covariance_unit": "m^2",
        "rate_unit": "bit/point",
        "mean_m": fit.mean.tolist(),
        "covariance_m2": fit.covariance.tolist(),
        "eigenvalues_m2": fit.eigenvalues.tolist(),
        "eigenvectors_columns": fit.eigenvectors.tolist(),
        "explained_variance_ratio": (fit.eigenvalues / fit.eigenvalues.sum()).tolist(),
        "principal_angle_error_deg": line_angle_error_deg(fit.eigenvectors[:, 0], major_axis),
        "eigengap_ratio": float((fit.eigenvalues[0] - fit.eigenvalues[1]) / fit.eigenvalues[0]),
        "task_axis_orientation_fitted_on_calibration": orientation,
        "raw_float64_baseline": {
            "rate_bits_per_point": 128,
            "rmse_m": 0.0,
            "task_accuracy": raw_accuracy,
        },
        "majority_baseline_accuracy": float(
            max(np.mean(evaluation_labels == 0), np.mean(evaluation_labels == 1))
        ),
        "codec_rows": codec_rows,
        "near_isotropic_failure": isotropic_failure(seed=seed),
        "boundary": (
            "These are operational points for a fixed scalar codec; they are not "
            "the Shannon R(D), a clinical guarantee, calibration, or causal evidence."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _save_pca_figure(output_dir / "pca_direction.png", evaluation_m, evaluation_labels, fit)
    _save_rate_figure(output_dir / "rate_distortion.png", codec_rows)
    return report


def _save_pca_figure(path: Path, points_m: np.ndarray, labels: np.ndarray, fit: PCAFit) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    scatter = ax.scatter(points_m[:, 0], points_m[:, 1], c=labels, cmap="coolwarm", s=12, alpha=0.6)
    colors = ("black", "tab:green")
    for index in range(2):
        vector = fit.eigenvectors[:, index] * 2.0 * np.sqrt(fit.eigenvalues[index])
        ax.quiver(
            fit.mean[0], fit.mean[1], vector[0], vector[1],
            angles="xy", scale_units="xy", scale=1, width=0.009,
            color=colors[index], label=f"eigenvector {index + 1}",
        )
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("PCA directions; colour is the synthetic task label")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Y")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_rate_figure(path: Path, rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for policy, marker in (("variance", "o"), ("task_aware", "s")):
        selected = [row for row in rows if row["policy"] == policy]
        rates = [row["rate_bits_per_point"] for row in selected]
        axes[0].plot(rates, [row["rmse_m"] for row in selected], marker + "-", label=policy)
        axes[1].plot(rates, [row["task_accuracy"] for row in selected], marker + "-", label=policy)
    axes[0].set_xlabel("nominal rate (bit/point)")
    axes[0].set_ylabel("geometry RMSE (m)")
    axes[0].set_title("Geometric distortion")
    axes[1].set_xlabel("nominal rate (bit/point)")
    axes[1].set_ylabel("task accuracy (fraction)")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].set_title("Task retention at the same rates")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Operational codec points—not the Shannon R(D) lower envelope")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/day026-output"))
    parser.add_argument("--seed", type=int, default=2608)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    report = run_experiment(args.output_dir, seed=args.seed)
    print(f"principal_angle_error_deg={report['principal_angle_error_deg']:.6f}")
    print(f"explained_variance_ratio={report['explained_variance_ratio'][0]:.6f}")
    print(f"raw_task_accuracy={report['raw_float64_baseline']['task_accuracy']:.6f}")
    if args.self_test:
        print("SELF_TEST_OK: report and figures generated")
    print("SUCCESS")


if __name__ == "__main__":
    main()
