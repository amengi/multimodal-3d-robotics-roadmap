"""Day 010: solve, least squares, SVD/eigh PCA, and audited baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class ContractError(ValueError):
    """Raised when an input violates a documented shape or value contract."""


@dataclass(frozen=True)
class LineFit:
    slope: float
    intercept_m: float
    rmse_m: float
    mean_baseline_rmse_m: float
    residual_sum_squares_m2: float
    rank: int
    singular_values: list[float]


@dataclass(frozen=True)
class PCAResult:
    mean: FloatArray
    components: FloatArray
    scores: FloatArray
    singular_values: FloatArray
    explained_variance: FloatArray
    explained_variance_ratio: FloatArray


def _finite_array(value: Any, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    return array


def solve_square(matrix: Any, target: Any) -> FloatArray:
    """Solve A x = b without forming A inverse."""

    a = _finite_array(matrix, "matrix")
    b = _finite_array(target, "target")
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ContractError("matrix must have non-empty square shape (n, n)")
    if b.shape not in {(a.shape[0],), (a.shape[0], 1)}:
        raise ContractError("target must have shape (n,) or (n, 1)")
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError as error:
        raise ContractError("matrix must be nonsingular") from error


def root_mean_square_error(actual: Any, predicted: Any) -> float:
    """Return RMSE using the physical unit of the inputs."""

    y = _finite_array(actual, "actual")
    y_hat = _finite_array(predicted, "predicted")
    if y.shape != y_hat.shape or y.ndim != 1 or y.size == 0:
        raise ContractError("actual and predicted must share non-empty shape (N,)")
    return float(np.sqrt(np.mean(np.square(y - y_hat))))


def fit_line_lstsq(x_m: Any, y_m: Any) -> tuple[LineFit, FloatArray, FloatArray]:
    """Fit y = slope*x + intercept with np.linalg.lstsq, never inv(A.T@A)."""

    x = _finite_array(x_m, "x_m")
    y = _finite_array(y_m, "y_m")
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 3:
        raise ContractError("x_m and y_m must share shape (N,) with N >= 3")
    if np.allclose(x, x[0]):
        raise ContractError("x_m must vary so slope and intercept are identifiable")

    design = np.column_stack([x, np.ones_like(x)])
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(
        design, y, rcond=None
    )
    if rank != 2:
        raise ContractError("line design matrix must have rank 2")
    predicted = design @ coefficients
    residual = y - predicted
    residual_sum_squares = float(residual @ residual)
    if residuals.size:
        # LAPACK and an explicit dot can accumulate roundoff differently near zero.
        np.testing.assert_allclose(
            residuals[0], residual_sum_squares, atol=1e-24, rtol=1e-12
        )
    fit = LineFit(
        slope=float(coefficients[0]),
        intercept_m=float(coefficients[1]),
        rmse_m=root_mean_square_error(y, predicted),
        mean_baseline_rmse_m=root_mean_square_error(y, np.full_like(y, y.mean())),
        residual_sum_squares_m2=residual_sum_squares,
        rank=int(rank),
        singular_values=singular_values.tolist(),
    )
    return fit, predicted, residual


def pca_svd(features: Any, n_components: int = 1) -> PCAResult:
    """Fit PCA by centering and SVD; input features are dimensionless."""

    x = _finite_array(features, "features")
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ContractError("features must have shape (N, F) with N >= 2 and F >= 1")
    maximum = min(x.shape)
    if not isinstance(n_components, int) or not 1 <= n_components <= maximum:
        raise ContractError(f"n_components must be an integer in [1, {maximum}]")
    mean = x.mean(axis=0)
    centered = x - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    scores = centered @ components.T
    explained_variance_all = np.square(singular_values) / (x.shape[0] - 1)
    total = float(explained_variance_all.sum())
    if total <= 0.0:
        raise ContractError("features must contain nonzero centered variation")
    return PCAResult(
        mean=mean,
        components=components,
        scores=scores,
        singular_values=singular_values,
        explained_variance=explained_variance_all[:n_components],
        explained_variance_ratio=explained_variance_all[:n_components] / total,
    )


def reconstruct_pca(result: PCAResult) -> FloatArray:
    """Map retained scores back to the original feature coordinates."""

    return result.scores @ result.components + result.mean


def feature_rmse(actual: Any, reconstructed: Any) -> float:
    """Elementwise RMSE for dimensionless feature matrices."""

    x = _finite_array(actual, "actual features")
    x_hat = _finite_array(reconstructed, "reconstructed features")
    if x.ndim != 2 or x.shape != x_hat.shape or x.size == 0:
        raise ContractError("feature arrays must share non-empty shape (N, F)")
    return float(np.sqrt(np.mean(np.square(x - x_hat))))


def coordinate_drop_reconstruction(features: Any, keep_index: int = 0) -> FloatArray:
    """Baseline: keep one raw centered coordinate and replace others by means."""

    x = _finite_array(features, "features")
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ContractError("features must have non-empty shape (N, F)")
    if not 0 <= keep_index < x.shape[1]:
        raise ContractError("keep_index is outside the feature axis")
    mean = x.mean(axis=0)
    reconstructed = np.broadcast_to(mean, x.shape).copy()
    reconstructed[:, keep_index] = x[:, keep_index]
    return reconstructed


def covariance_eigh(features: Any) -> tuple[FloatArray, FloatArray]:
    """Return descending eigenpairs of a symmetric sample covariance matrix."""

    x = _finite_array(features, "features")
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ContractError("features must have shape (N, F) with N >= 2 and F >= 1")
    centered = x - x.mean(axis=0)
    covariance = centered.T @ centered / (x.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def make_line_data(seed: int = 20260915, n_samples: int = 40) -> tuple[FloatArray, FloatArray]:
    if n_samples < 3:
        raise ContractError("n_samples must be at least 3")
    rng = np.random.default_rng(seed)
    x_m = np.linspace(-3.0, 3.0, n_samples)
    y_m = 1.75 * x_m - 0.40 + rng.normal(0.0, 0.28, n_samples)
    return x_m, y_m


def make_rotated_features(seed: int = 20260915, n_samples: int = 240) -> FloatArray:
    """Create a rotated, anisotropic, dimensionless 3-D cloud for PCA."""

    if n_samples < 3:
        raise ContractError("n_samples must be at least 3")
    rng = np.random.default_rng(seed)
    direction = np.array([0.40, 0.80, -0.45])
    direction /= np.linalg.norm(direction)
    latent = rng.normal(0.0, 2.0, size=(n_samples, 1))
    noise = rng.normal(0.0, 0.20, size=(n_samples, 3))
    return latent * direction + noise


def save_residual_plot(
    x_m: FloatArray,
    y_m: FloatArray,
    predicted_m: FloatArray,
    residual_m: FloatArray,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "line_fit_residuals.png"
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    axes[0].scatter(x_m, y_m, label="observations", s=24)
    axes[0].plot(x_m, predicted_m, color="tab:orange", label="lstsq fit")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Overdetermined line fit")
    axes[0].legend()
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].scatter(x_m, residual_m, color="tab:red", s=24)
    axes[1].set(xlabel="x (m)", ylabel="residual y-y_hat (m)", title="Residual audit")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def run_demo(output_dir: Path, seed: int = 20260915) -> dict[str, Any]:
    square_solution = solve_square([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])
    x_m, y_m = make_line_data(seed=seed)
    line_fit, predicted_m, residual_m = fit_line_lstsq(x_m, y_m)
    plot_path = save_residual_plot(x_m, y_m, predicted_m, residual_m, output_dir)

    features = make_rotated_features(seed=seed)
    pca = pca_svd(features, n_components=1)
    pca_reconstruction = reconstruct_pca(pca)
    drop_reconstruction = coordinate_drop_reconstruction(features, keep_index=0)
    pca_rmse = feature_rmse(features, pca_reconstruction)
    drop_rmse = feature_rmse(features, drop_reconstruction)
    eigenvalues, eigenvectors = covariance_eigh(features)
    alignment = abs(float(pca.components[0] @ eigenvectors[:, 0]))

    report: dict[str, Any] = {
        "seed": seed,
        "square_system": {
            "A_shape": [2, 2],
            "b_shape": [2],
            "solution": square_solution.tolist(),
        },
        "line_fit": asdict(line_fit),
        "pca": {
            "input_shape": list(features.shape),
            "scores_shape": list(pca.scores.shape),
            "component_shape": list(pca.components.shape),
            "first_explained_variance_ratio": float(pca.explained_variance_ratio[0]),
            "pca_1d_reconstruction_rmse": pca_rmse,
            "keep_raw_coordinate_0_rmse": drop_rmse,
            "svd_eigh_first_direction_abs_dot": alignment,
            "eigh_eigenvalues": eigenvalues.tolist(),
        },
        "limits": [
            "PCA maximizes sample variance, not label performance",
            "variance retained is not entropy or mutual information retained",
            "a principal direction is not a causal direction",
            "synthetic results do not establish deployment performance",
        ],
        "residual_plot": str(plot_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "day010_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/Day010/reference_outputs"),
        help="directory for the JSON report and residual plot",
    )
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    report = run_demo(args.output, seed=args.seed)
    line = report["line_fit"]
    pca = report["pca"]
    print("solve A@x=b solution:", np.array(report["square_system"]["solution"]))
    print(
        "lstsq line: slope={:.4f}, intercept={:.4f} m, RMSE={:.4f} m, "
        "mean-baseline RMSE={:.4f} m, rank={}".format(
            line["slope"],
            line["intercept_m"],
            line["rmse_m"],
            line["mean_baseline_rmse_m"],
            line["rank"],
        )
    )
    print(
        "PCA shapes: input={}, scores={}, component={}".format(
            tuple(pca["input_shape"]),
            tuple(pca["scores_shape"]),
            tuple(pca["component_shape"]),
        )
    )
    print(
        "PCA 1D: explained_variance_ratio={:.4f}, RMSE={:.4f}; "
        "keep-raw-coordinate RMSE={:.4f}".format(
            pca["first_explained_variance_ratio"],
            pca["pca_1d_reconstruction_rmse"],
            pca["keep_raw_coordinate_0_rmse"],
        )
    )
    print("SVD/eigh first-direction |dot|: {:.12f}".format(pca["svd_eigh_first_direction_abs_dot"]))
    print("saved:", report["residual_plot"])
    print("saved:", report["report_path"])
    print("All Day 010 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
