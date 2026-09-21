"""Reproducible Day 014 point-cloud and nonlinear-dependence project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class ContractError(ValueError):
    """Raised when an input violates a documented data contract."""


def validate_points(points: ArrayLike, *, name: str = "points") -> FloatArray:
    """Return a finite ``(N, 3)`` float array with at least four points."""
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] < 4:
        raise ContractError(f"{name} must have shape (N, 3) with N >= 4")
    if not np.all(np.isfinite(value)):
        raise ContractError(f"{name} must contain only finite values")
    return value


def make_cube_surface(side_m: float = 2.0, points_per_edge: int = 8) -> FloatArray:
    """Create unique, symmetric samples on a cube surface centered at the origin."""
    if not np.isfinite(side_m) or side_m <= 0:
        raise ContractError("side_m must be finite and positive")
    if points_per_edge < 2:
        raise ContractError("points_per_edge must be at least 2")
    grid = np.linspace(-side_m / 2.0, side_m / 2.0, points_per_edge)
    u, v = np.meshgrid(grid, grid, indexing="ij")
    faces: list[FloatArray] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            face = np.empty((u.size, 3), dtype=np.float64)
            free_axes = [candidate for candidate in range(3) if candidate != axis]
            face[:, axis] = sign * side_m / 2.0
            face[:, free_axes[0]] = u.ravel()
            face[:, free_axes[1]] = v.ravel()
            faces.append(face)
    return np.unique(np.vstack(faces), axis=0)


def rotation_z(degrees: float) -> FloatArray:
    """Return an active right-handed rotation about +z."""
    if not np.isfinite(degrees):
        raise ContractError("degrees must be finite")
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def validate_rigid_transform(rotation: ArrayLike, translation_m: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Validate a proper 3D rotation and a finite translation in metres."""
    matrix = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation_m, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ContractError("rotation must have shape (3, 3)")
    if translation.shape != (3,):
        raise ContractError("translation_m must have shape (3,)")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(translation)):
        raise ContractError("rotation and translation must be finite")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10):
        raise ContractError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10):
        raise ContractError("rotation determinant must be +1")
    return matrix, translation


def transform_points(
    points_m: ArrayLike,
    rotation: ArrayLike,
    translation_m: ArrayLike,
    *,
    noise_sigma_m: float = 0.0,
    seed: int = 20260921,
) -> FloatArray:
    """Apply ``p_target = p_source @ R.T + t + noise`` using row vectors."""
    points = validate_points(points_m, name="points_m")
    matrix, translation = validate_rigid_transform(rotation, translation_m)
    if not np.isfinite(noise_sigma_m) or noise_sigma_m < 0:
        raise ContractError("noise_sigma_m must be finite and nonnegative")
    transformed = points @ matrix.T + translation
    if noise_sigma_m > 0:
        rng = np.random.default_rng(seed)
        transformed = transformed + rng.normal(0.0, noise_sigma_m, size=transformed.shape)
    return transformed


def cloud_summary(points_m: ArrayLike) -> dict[str, FloatArray]:
    """Return centroid, axis-aligned bounds, and AABB extent in metres."""
    points = validate_points(points_m, name="points_m")
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return {
        "centroid_m": points.mean(axis=0),
        "aabb_min_m": minimum,
        "aabb_max_m": maximum,
        "aabb_extent_m": maximum - minimum,
    }


def paired_rmse_m(reference_m: ArrayLike, estimate_m: ArrayLike) -> float:
    """Compute pointwise RMSE for two paired ``(N,3)`` clouds."""
    reference = validate_points(reference_m, name="reference_m")
    estimate = validate_points(estimate_m, name="estimate_m")
    if reference.shape != estimate.shape:
        raise ContractError("paired point clouds must have identical shape")
    squared_distance = np.sum((estimate - reference) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_distance)))


def make_nonlinear_data(
    *, seed: int = 20260921, n_samples: int = 1200, noise_sigma: float = 0.25
) -> tuple[FloatArray, FloatArray]:
    """Generate ``X~N(0,1)`` and ``Y=X^2+epsilon``."""
    if n_samples < 40:
        raise ContractError("n_samples must be at least 40")
    if not np.isfinite(noise_sigma) or noise_sigma < 0:
        raise ContractError("noise_sigma must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_samples)
    y = x**2 + rng.normal(0.0, noise_sigma, size=n_samples)
    return x.astype(np.float64), y.astype(np.float64)


def pearson_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Compute Pearson correlation after checking a one-dimensional contract."""
    x_value = np.asarray(x, dtype=np.float64)
    y_value = np.asarray(y, dtype=np.float64)
    if x_value.ndim != 1 or y_value.ndim != 1 or x_value.shape != y_value.shape:
        raise ContractError("x and y must be one-dimensional arrays of equal shape")
    if x_value.size < 2 or not np.all(np.isfinite(x_value)) or not np.all(np.isfinite(y_value)):
        raise ContractError("x and y must have at least two finite values")
    if np.std(x_value) == 0 or np.std(y_value) == 0:
        raise ContractError("Pearson correlation is undefined for zero variance")
    return float(np.corrcoef(x_value, y_value)[0, 1])


def fit_regression_baselines(x: ArrayLike, y: ArrayLike) -> dict[str, Any]:
    """Compare intercept-only, linear, and quadratic least-squares baselines."""
    x_value = np.asarray(x, dtype=np.float64)
    y_value = np.asarray(y, dtype=np.float64)
    if x_value.ndim != 1 or y_value.ndim != 1 or x_value.shape != y_value.shape:
        raise ContractError("x and y must be one-dimensional arrays of equal shape")
    if x_value.size < 40 or not np.all(np.isfinite(x_value)) or not np.all(np.isfinite(y_value)):
        raise ContractError("x and y must contain at least 40 finite paired values")
    split = int(0.75 * x_value.size)
    x_train, x_test = x_value[:split], x_value[split:]
    y_train, y_test = y_value[:split], y_value[split:]
    designs_train = {
        "constant": np.ones((split, 1)),
        "linear": np.column_stack((np.ones(split), x_train)),
        "quadratic": np.column_stack((np.ones(split), x_train, x_train**2)),
    }
    designs_test = {
        "constant": np.ones((x_test.size, 1)),
        "linear": np.column_stack((np.ones(x_test.size), x_test)),
        "quadratic": np.column_stack((np.ones(x_test.size), x_test, x_test**2)),
    }
    result: dict[str, Any] = {"train_size": split, "test_size": x_test.size}
    for name, design in designs_train.items():
        coefficients, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
        prediction = designs_test[name] @ coefficients
        rmse = float(np.sqrt(np.mean((prediction - y_test) ** 2)))
        result[name] = {"coefficients": coefficients.tolist(), "test_rmse": rmse}
    return result


def dependence_event_gap(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Give one event pair whose joint probability violates independence factorization."""
    x_value = np.asarray(x, dtype=np.float64)
    y_value = np.asarray(y, dtype=np.float64)
    if x_value.ndim != 1 or y_value.ndim != 1 or x_value.shape != y_value.shape:
        raise ContractError("x and y must be one-dimensional arrays of equal shape")
    event_a = np.abs(x_value) > 1.0
    event_b = y_value > 1.0
    p_a = float(event_a.mean())
    p_b = float(event_b.mean())
    p_joint = float(np.logical_and(event_a, event_b).mean())
    return {
        "p_abs_x_gt_1": p_a,
        "p_y_gt_1": p_b,
        "p_joint": p_joint,
        "p_product_if_independent": p_a * p_b,
        "absolute_gap": abs(p_joint - p_a * p_b),
    }


def _equal_limits(*clouds: FloatArray) -> list[tuple[float, float]]:
    combined = np.vstack(clouds)
    minimum, maximum = combined.min(axis=0), combined.max(axis=0)
    span = float(np.max(maximum - minimum))
    if span <= 0:
        raise ContractError("point clouds must have nonzero spatial extent")
    center = (minimum + maximum) / 2.0
    return [(float(value - span / 2), float(value + span / 2)) for value in center]


def _save_cloud_figure(source: FloatArray, target: FloatArray, destination: Path) -> None:
    figure = plt.figure(figsize=(10, 4.5), layout="constrained")
    axes = [figure.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    limits = _equal_limits(source, target)
    for axis, points, title, color in zip(
        axes,
        (source, target),
        ("Source cube: source frame", "Transformed noisy cube: target frame"),
        ("tab:blue", "tab:orange"),
        strict=True,
    ):
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=7, alpha=0.65, color=color)
        axis.scatter(*points.mean(axis=0), s=70, marker="x", color="black", label="centroid")
        axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=title)
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_zlim(*limits[2])
        axis.set_box_aspect((1, 1, 1))
        axis.legend()
    figure.suptitle("Day 014: paired point-cloud geometry audit")
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_dependence_figure(x: FloatArray, y: FloatArray, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    axes[0].scatter(x, y, s=8, alpha=0.35, color="tab:purple")
    axes[0].set(xlabel="X (unitless)", ylabel="Y (unitless)", title="Y = X² + noise")
    order = np.argsort(x)
    designs = np.column_stack((np.ones(x.size), x, x**2))
    coefficients, _, _, _ = np.linalg.lstsq(designs, y, rcond=None)
    axes[0].plot(x[order], (designs @ coefficients)[order], color="black", label="quadratic fit")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=(18, 18))
    image = axes[1].pcolormesh(x_edges, y_edges, counts.T, shading="auto", cmap="Blues")
    axes[1].set(xlabel="X (unitless)", ylabel="Y (unitless)", title="Empirical joint counts")
    figure.colorbar(image, ax=axes[1], label="sample count")
    figure.suptitle("Zero Pearson correlation does not establish independence")
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run(output: Path, *, seed: int = 20260921) -> dict[str, Any]:
    """Run both project tracks and write strict JSON plus two figures."""
    output.mkdir(parents=True, exist_ok=True)
    source = make_cube_surface()
    rotation = rotation_z(30.0)
    translation = np.array([1.5, -0.5, 0.8])
    clean_target = transform_points(source, rotation, translation)
    noisy_target = transform_points(
        source, rotation, translation, noise_sigma_m=0.01, seed=seed
    )
    source_summary = cloud_summary(source)
    target_summary = cloud_summary(noisy_target)
    centroid_shift = target_summary["centroid_m"] - source_summary["centroid_m"]

    x, y = make_nonlinear_data(seed=seed)
    baselines = fit_regression_baselines(x, y)
    event_gap = dependence_event_gap(x, y)
    cloud_figure = output / "cube_transform.png"
    dependence_figure = output / "nonlinear_dependence.png"
    _save_cloud_figure(source, noisy_target, cloud_figure)
    _save_dependence_figure(x, y, dependence_figure)

    report: dict[str, Any] = {
        "seed": seed,
        "point_cloud_contract": {
            "shape": list(source.shape),
            "unit": "m",
            "source_frame": "cube_local",
            "target_frame": "world",
            "pairing_key": "point_index",
            "row_vector_formula": "p_target = p_source @ R.T + t + noise",
        },
        "geometry": {
            "rotation_deg_about_positive_z": 30.0,
            "translation_m": translation.tolist(),
            "noise_sigma_m": 0.01,
            "source_centroid_m": source_summary["centroid_m"].tolist(),
            "target_centroid_m": target_summary["centroid_m"].tolist(),
            "estimated_centroid_shift_m": centroid_shift.tolist(),
            "source_aabb_extent_m": source_summary["aabb_extent_m"].tolist(),
            "target_aabb_extent_m": target_summary["aabb_extent_m"].tolist(),
            "paired_rmse_to_clean_target_m": paired_rmse_m(clean_target, noisy_target),
        },
        "dependence": {
            "definition": "X~N(0,1); Y=X^2+epsilon; epsilon~N(0,0.25^2)",
            "pearson_x_y": pearson_correlation(x, y),
            "pearson_abs_x_y": pearson_correlation(np.abs(x), y),
            "event_factorization_check": event_gap,
            "regression_baselines": baselines,
        },
        "figures": [str(cloud_figure), str(dependence_figure)],
        "limits": [
            "AABB extent depends on coordinate axes and changes under rotation",
            "near-zero sample correlation does not prove independence",
            "synthetic results do not establish causality or deployment performance",
        ],
    }
    report_path = output / "day014_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/day014-output"))
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    report = run(args.output, seed=args.seed)
    geometry = report["geometry"]
    dependence = report["dependence"]
    baselines = dependence["regression_baselines"]
    print(f"cube points: {report['point_cloud_contract']['shape']}, unit=m")
    print("centroid shift (m):", np.round(geometry["estimated_centroid_shift_m"], 6).tolist())
    print("source/target AABB extent (m):", np.round(geometry["source_aabb_extent_m"], 6).tolist(), np.round(geometry["target_aabb_extent_m"], 6).tolist())
    print(f"paired noise RMSE={geometry['paired_rmse_to_clean_target_m']:.6f} m")
    print(f"Pearson r(X,Y)={dependence['pearson_x_y']:.6f}; r(|X|,Y)={dependence['pearson_abs_x_y']:.6f}")
    print("test RMSE constant/linear/quadratic:", *(f"{baselines[name]['test_rmse']:.6f}" for name in ("constant", "linear", "quadratic")))
    event = dependence["event_factorization_check"]
    print(f"event P(A,B)={event['p_joint']:.4f} vs P(A)P(B)={event['p_product_if_independent']:.4f}; gap={event['absolute_gap']:.4f}")
    for path in report["figures"] + [report["report_path"]]:
        print("saved:", path)
    print("SUCCESS: geometry, dependence, baselines, and artifacts validated")


if __name__ == "__main__":
    main()
