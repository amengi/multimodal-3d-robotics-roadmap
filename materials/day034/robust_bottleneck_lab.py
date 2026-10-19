"""Day 034: robust line fitting and an exact binary bottleneck audit.

The first experiment compares ordinary squared residuals with Huber and
Cauchy losses on the same synthetic line data.  The second enumerates a
two-bit stochastic encoder, so its task distortion and KL capacity proxy are
exact rather than estimated from samples.
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
from scipy.optimize import least_squares


TRUE_SLOPE = 1.7
TRUE_INTERCEPT_M = -0.4
INLIER_SIGMA_M = 0.08
OUTLIER_FRACTION = 0.20
F_SCALE_M = 0.12


def make_line_data(seed: int = 34, n: int = 100) -> dict[str, np.ndarray]:
    """Return x/y coordinates in metres plus a boolean outlier mask."""
    if n < 20:
        raise ValueError("n must be at least 20")
    rng = np.random.default_rng(seed)
    x_m = np.linspace(-2.0, 2.0, n)
    y_clean_m = TRUE_SLOPE * x_m + TRUE_INTERCEPT_M
    y_m = y_clean_m + rng.normal(0.0, INLIER_SIGMA_M, size=n)
    count = int(round(OUTLIER_FRACTION * n))
    indices = rng.choice(n, size=count, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    shifts = rng.uniform(1.5, 3.0, size=count) * rng.choice([-1.0, 1.0], size=count)
    y_m[indices] += shifts
    return {"x_m": x_m, "y_m": y_m, "y_clean_m": y_clean_m, "outlier_mask": mask}


def residuals(params: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Residual prediction-observation, shape (N,), unit metre."""
    slope, intercept_m = params
    return slope * x_m + intercept_m - y_m


def fit_line(data: dict[str, np.ndarray], loss: str) -> dict[str, float | bool]:
    """Fit one line with a SciPy least-squares loss."""
    result = least_squares(
        residuals,
        x0=np.array([0.0, 0.0]),
        args=(data["x_m"], data["y_m"]),
        loss=loss,
        f_scale=F_SCALE_M,
    )
    slope, intercept_m = result.x
    all_residuals_m = residuals(result.x, data["x_m"], data["y_m"])
    inlier_residuals_m = all_residuals_m[~data["outlier_mask"]]
    return {
        "success": bool(result.success),
        "slope": float(slope),
        "intercept_m": float(intercept_m),
        "slope_abs_error": float(abs(slope - TRUE_SLOPE)),
        "intercept_abs_error_m": float(abs(intercept_m - TRUE_INTERCEPT_M)),
        "inlier_rmse_m": float(np.sqrt(np.mean(inlier_residuals_m**2))),
        "all_rmse_m": float(np.sqrt(np.mean(all_residuals_m**2))),
        "nfev": int(result.nfev),
    }


def huber_loss(residual: np.ndarray | float, delta: float) -> np.ndarray:
    """Textbook Huber loss with residual and delta in the same unit."""
    r = np.asarray(residual, dtype=float)
    absolute = np.abs(r)
    return np.where(absolute <= delta, 0.5 * r**2, delta * (absolute - 0.5 * delta))


def cauchy_loss(residual: np.ndarray | float, scale: float) -> np.ndarray:
    """Cauchy loss, scaled so the result has squared-residual units."""
    r = np.asarray(residual, dtype=float)
    return 0.5 * scale**2 * np.log1p((r / scale) ** 2)


def binary_entropy_nats(p: np.ndarray | float) -> np.ndarray:
    """Binary entropy in nats, with exact handling of p=0 and p=1."""
    p_array = np.asarray(p, dtype=float)
    if np.any((p_array < 0.0) | (p_array > 1.0)):
        raise ValueError("probability must lie in [0, 1]")
    out = np.zeros_like(p_array)
    interior = (p_array > 0.0) & (p_array < 1.0)
    q = p_array[interior]
    out[interior] = -q * np.log(q) - (1.0 - q) * np.log(1.0 - q)
    return out


def encoder_metrics(error_signal: float, error_nuisance: float) -> dict[str, float]:
    """Exact metrics for two independent uniform bits through binary channels.

    X=(S,N), Y=S and Z=(Zs,Zn).  error_signal and error_nuisance are bit-flip
    probabilities.  The reference r(z) is uniform, which equals the aggregate
    marginal for this symmetric example.
    """
    if not 0.0 <= error_signal <= 0.5 or not 0.0 <= error_nuisance <= 0.5:
        raise ValueError("bit errors must lie in [0, 0.5]")
    hs = float(binary_entropy_nats(error_signal))
    hn = float(binary_entropy_nats(error_nuisance))
    capacity_nats = 2.0 * math.log(2.0) - hs - hn
    return {
        "signal_error": float(error_signal),
        "nuisance_error": float(error_nuisance),
        "task_error": float(error_signal),
        "task_accuracy": float(1.0 - error_signal),
        "task_nll_nats": hs,
        "task_brier": float(error_signal * (1.0 - error_signal)),
        "reconstruction_hamming_per_bit": float((error_signal + error_nuisance) / 2.0),
        "capacity_nats": float(capacity_nats),
        "capacity_bits": float(capacity_nats / math.log(2.0)),
    }


def optimize_binary_bottleneck(beta: float, objective: str, grid_size: int = 501) -> dict[str, float]:
    """Enumerate a VIB-like or beta-VAE-like rate-distortion objective."""
    if beta < 0.0:
        raise ValueError("beta must be nonnegative")
    if objective not in {"vib", "beta_vae"}:
        raise ValueError("objective must be 'vib' or 'beta_vae'")
    grid = np.linspace(0.0, 0.5, grid_size)
    es, en = np.meshgrid(grid, grid, indexing="ij")
    hs = binary_entropy_nats(es)
    hn = binary_entropy_nats(en)
    capacity = 2.0 * math.log(2.0) - hs - hn
    if objective == "vib":
        distortion = es
    else:
        distortion = 0.5 * (es + en)
    values = distortion + beta * capacity
    flat_index = int(np.argmin(values))
    i, j = np.unravel_index(flat_index, values.shape)
    metrics = encoder_metrics(float(grid[i]), float(grid[j]))
    metrics["beta"] = float(beta)
    metrics["objective_value"] = float(values[i, j])
    return metrics


def dropout_baseline(drop_probability: float = 0.30) -> dict[str, float | str]:
    """Exact task metrics when the signal bit is erased with probability p."""
    if not 0.0 <= drop_probability <= 1.0:
        raise ValueError("drop_probability must lie in [0, 1]")
    return {
        "name": "dropout-erasure baseline",
        "drop_probability": float(drop_probability),
        "task_accuracy": float(1.0 - 0.5 * drop_probability),
        "task_nll_nats": float(drop_probability * math.log(2.0)),
        "task_brier": float(0.25 * drop_probability),
        "capacity_bits": "not an optimized KL capacity",
    }


def run_experiment(seed: int = 34) -> dict[str, object]:
    data = make_line_data(seed=seed)
    fits = {loss: fit_line(data, loss) for loss in ("linear", "huber", "cauchy")}
    betas = [0.02, 0.10, 0.30, 0.60, 1.00]
    bottleneck = {
        "vib": [optimize_binary_bottleneck(beta, "vib") for beta in betas],
        "beta_vae": [optimize_binary_bottleneck(beta, "beta_vae") for beta in betas],
        "dropout": dropout_baseline(0.30),
    }
    return {
        "config": {
            "seed": seed,
            "n_samples": int(data["x_m"].size),
            "outlier_fraction": OUTLIER_FRACTION,
            "inlier_sigma_m": INLIER_SIGMA_M,
            "f_scale_m": F_SCALE_M,
            "true_slope": TRUE_SLOPE,
            "true_intercept_m": TRUE_INTERCEPT_M,
            "beta_values": betas,
        },
        "robust_fits": fits,
        "binary_bottleneck": bottleneck,
        "scope_note": (
            "The binary KL is exact for the declared toy distribution; it is not an "
            "estimate of mutual information in a continuous deterministic network."
        ),
    }


def save_plots(report: dict[str, object], output_dir: Path, seed: int) -> None:
    data = make_line_data(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    inliers = ~data["outlier_mask"]
    axes[0].scatter(data["x_m"][inliers], data["y_m"][inliers], s=18, label="inliers")
    axes[0].scatter(data["x_m"][~inliers], data["y_m"][~inliers], s=28, marker="x", label="outliers")
    x_grid = np.linspace(-2.1, 2.1, 200)
    axes[0].plot(x_grid, TRUE_SLOPE * x_grid + TRUE_INTERCEPT_M, "k--", label="true line")
    for loss, values in report["robust_fits"].items():
        axes[0].plot(x_grid, values["slope"] * x_grid + values["intercept_m"], label=loss)
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Same data, different residual losses")
    axes[0].legend(fontsize=8)

    bins = np.linspace(-0.35, 0.35, 25)
    for loss, values in report["robust_fits"].items():
        params = np.array([values["slope"], values["intercept_m"]])
        inlier_residual = residuals(params, data["x_m"], data["y_m"])[inliers]
        axes[1].hist(inlier_residual, bins=bins, alpha=0.42, label=loss)
    axes[1].set(xlabel="inlier residual: prediction - observation (m)", ylabel="count", title="Inlier residual distribution")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "robust_line_fit.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, marker in (("vib", "o"), ("beta_vae", "s")):
        rows = report["binary_bottleneck"][name]
        capacity = [row["capacity_bits"] for row in rows]
        axes[0].plot(capacity, [row["task_error"] for row in rows], marker=marker, label=name)
        axes[1].plot(capacity, [row["reconstruction_hamming_per_bit"] for row in rows], marker=marker, label=name)
        for row in rows:
            axes[0].annotate(f"β={row['beta']:g}", (row["capacity_bits"], row["task_error"]), fontsize=7)
    axes[0].set(xlabel="KL capacity proxy (bit/example)", ylabel="task error probability", title="Task distortion")
    axes[1].set(xlabel="KL capacity proxy (bit/example)", ylabel="reconstruction Hamming distortion / bit", title="Reconstruction distortion")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bottleneck_tradeoff.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/day034-output"))
    parser.add_argument("--seed", type=int, default=34)
    args = parser.parse_args()
    report = run_experiment(seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_plots(report, args.output_dir, args.seed)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fits = report["robust_fits"]
    print("loss      slope   intercept(m)  |slope err|  inlier RMSE(m)")
    for loss in ("linear", "huber", "cauchy"):
        row = fits[loss]
        print(f"{loss:7s}  {row['slope']:7.4f}  {row['intercept_m']:12.4f}  {row['slope_abs_error']:11.4f}  {row['inlier_rmse_m']:14.4f}")
    vib = report["binary_bottleneck"]["vib"]
    print("\nVIB-like exact binary scan")
    print("beta  capacity(bit)  signal error  nuisance error  task accuracy")
    for row in vib:
        print(f"{row['beta']:4.2f}  {row['capacity_bits']:13.4f}  {row['signal_error']:12.3f}  {row['nuisance_error']:14.3f}  {row['task_accuracy']:13.3f}")
    print(f"\nWROTE: {report_path}")
    print("SUCCESS: robust fits and exact binary bottleneck audit completed")


if __name__ == "__main__":
    main()
