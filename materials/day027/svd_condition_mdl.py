"""Day 027: SVD image compression, conditioning, and an explicit MDL proxy.

The image is synthetic and deterministic.  All image intensities are dimensionless
numbers in [0, 1].  The MDL calculation is a teaching proxy, not an exact NML code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RANKS = (0, 1, 2, 4, 8, 16, 32, 64)


def validate_matrix(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    """Return a finite float64 2-D matrix or raise a helpful ValueError."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    if min(array.shape) < 1:
        raise ValueError(f"{name} must have non-zero dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def make_synthetic_image(seed: int = 2709, shape: tuple[int, int] = (64, 96)) -> np.ndarray:
    """Create a structured grayscale image with low-amplitude sensor noise."""
    if len(shape) != 2 or min(shape) < 8:
        raise ValueError("shape must contain two dimensions, each at least 8")
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    x = xx / (cols - 1)
    y = yy / (rows - 1)
    background = 0.12 + 0.32 * x + 0.12 * y
    disk = ((x - 0.30) ** 2 + (y - 0.38) ** 2 <= 0.13**2).astype(float) * 0.38
    rectangle = ((x > 0.58) & (x < 0.85) & (y > 0.18) & (y < 0.42)).astype(float) * 0.28
    ridge = 0.18 * np.exp(-((y - (0.68 + 0.08 * np.sin(5 * np.pi * x))) ** 2) / 0.002)
    texture = 0.035 * np.sin(14 * np.pi * x) * np.cos(8 * np.pi * y)
    noise = np.random.default_rng(seed).normal(0.0, 0.008, size=shape)
    return np.clip(background + disk + rectangle + ridge + texture + noise, 0.0, 1.0)


def compact_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the compact SVD A=U diag(s) Vh."""
    array = validate_matrix(matrix)
    u, s, vh = np.linalg.svd(array, full_matrices=False)
    return u, s, vh


def rank_reconstruct(
    u: np.ndarray, s: np.ndarray, vh: np.ndarray, rank: int
) -> np.ndarray:
    """Return the rank-r truncated-SVD reconstruction."""
    k = len(s)
    if not isinstance(rank, (int, np.integer)) or not 0 <= int(rank) <= k:
        raise ValueError(f"rank must be an integer in [0, {k}]")
    rank = int(rank)
    if rank == 0:
        return np.zeros((u.shape[0], vh.shape[1]), dtype=np.float64)
    return (u[:, :rank] * s[:rank]) @ vh[:rank, :]


def image_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Compute normalized-image MSE, RMSE, and PSNR."""
    ref = validate_matrix(reference, "reference")
    est = validate_matrix(estimate, "estimate")
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have identical shapes")
    mse = float(np.mean((ref - est) ** 2))
    rmse = math.sqrt(mse)
    psnr_db = math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)
    return {"mse": mse, "rmse": rmse, "psnr_db": psnr_db}


def mdl_proxy_bits(
    residual: np.ndarray,
    parameter_count: int,
    *,
    parameter_bits: int = 16,
    noise_sigma: float = 0.03,
    quantization_step: float = 1.0 / 255.0,
) -> dict[str, float | int]:
    """Return a declared two-part teaching proxy in bits.

    The residual part uses a fixed zero-mean Gaussian density and converts density
    to approximate 8-bit-bin probability with a fixed quantization step.  The model
    part assumes a fixed number of bits for every stored scalar.
    """
    errors = validate_matrix(residual, "residual")
    if parameter_count < 0 or parameter_bits <= 0:
        raise ValueError("parameter counts and bit widths must be non-negative/positive")
    if noise_sigma <= 0 or not 0 < quantization_step <= 1:
        raise ValueError("noise_sigma and quantization_step must be positive")
    nll_nats = np.sum(
        0.5 * np.log(2.0 * np.pi * noise_sigma**2)
        + (errors**2) / (2.0 * noise_sigma**2)
    )
    residual_bits = float(nll_nats / np.log(2.0) - errors.size * np.log2(quantization_step))
    model_bits = int(parameter_count * parameter_bits)
    return {
        "parameter_count": int(parameter_count),
        "model_bits": model_bits,
        "residual_nll_bits": residual_bits,
        "total_proxy_bits": float(model_bits + residual_bits),
    }


def condition_demo(epsilon: float = 1e-4, perturbation: float = 1e-6) -> dict[str, object]:
    """Show forward-error amplification and a truncated-pseudoinverse trade-off."""
    if not 0 < epsilon < 1 or perturbation <= 0:
        raise ValueError("epsilon must be in (0,1) and perturbation must be positive")
    a = np.diag([1.0, epsilon])
    x_true = np.array([1.0, 1.0])
    b = a @ x_true
    b_noisy = b + np.array([0.0, perturbation])
    x_direct = np.linalg.solve(a, b_noisy)
    relative_b_error = float(np.linalg.norm(b_noisy - b) / np.linalg.norm(b))
    relative_x_error = float(np.linalg.norm(x_direct - x_true) / np.linalg.norm(x_true))
    amplification = relative_x_error / relative_b_error

    cutoffs: dict[str, dict[str, object]] = {}
    for rcond in (1e-12, 1e-6, 1e-3):
        x_hat = np.linalg.pinv(a, rcond=rcond) @ b_noisy
        cutoffs[f"{rcond:.0e}"] = {
            "x_hat": [float(value) for value in x_hat],
            "relative_solution_error": float(np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)),
            "relative_residual": float(np.linalg.norm(a @ x_hat - b_noisy) / np.linalg.norm(b_noisy)),
        }
    return {
        "A": a.tolist(),
        "singular_values": [1.0, epsilon],
        "condition_number_2": float(np.linalg.cond(a, 2)),
        "x_true": x_true.tolist(),
        "b": b.tolist(),
        "b_noisy": b_noisy.tolist(),
        "x_direct": x_direct.tolist(),
        "relative_b_error": relative_b_error,
        "relative_x_error": relative_x_error,
        "observed_amplification": float(amplification),
        "pinv_cutoffs": cutoffs,
    }


def fusion_model_comparison(repeats: int = 100) -> dict[str, object]:
    """Compare two fixed toy predictors with NLL and a declared code proxy."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    labels = np.tile(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int), repeats)
    stable_prob = np.tile(np.array([0.10, 0.20, 0.35, 0.45, 0.55, 0.65, 0.80, 0.90]), repeats)
    fusion_prob = np.tile(np.array([0.04, 0.08, 0.20, 0.40, 0.60, 0.80, 0.92, 0.96]), repeats)
    models = {
        "stable_single_modality": {"prob": stable_prob, "parameter_count": 4},
        "two_modality_fusion": {"prob": fusion_prob, "parameter_count": 18},
    }
    results: dict[str, object] = {}
    for name, specification in models.items():
        probability = specification["prob"]
        p_true = np.where(labels == 1, probability, 1.0 - probability)
        nll_nats = float(-np.sum(np.log(p_true)))
        data_bits = nll_nats / np.log(2.0)
        model_bits = int(specification["parameter_count"] * 8)
        prediction = (probability >= 0.5).astype(int)
        results[name] = {
            "parameter_count": specification["parameter_count"],
            "accuracy": float(np.mean(prediction == labels)),
            "nll_nats_per_example": nll_nats / labels.size,
            "data_nll_bits": data_bits,
            "model_bits": model_bits,
            "total_proxy_bits": model_bits + data_bits,
        }
    return {
        "n_examples": int(labels.size),
        "model_scalar_bits": 8,
        "warning": "fixed synthetic probabilities; proxy is not exact MDL and not a calibration study",
        "models": results,
    }


def _plot_reconstructions(image: np.ndarray, records: list[dict[str, object]], output: Path) -> None:
    selected = [0, 4, 16, min(image.shape)]
    lookup = {int(record["rank"]): record for record in records}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    for ax, rank in zip(axes, selected):
        array = image if rank == min(image.shape) else np.asarray(lookup[rank]["reconstruction"])
        ax.imshow(array, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        metric = image_metrics(image, array)
        title = "original/full" if rank == min(image.shape) else f"rank {rank}"
        ax.set_title(f"{title}\nRMSE={metric['rmse']:.4f}")
        ax.set_xlabel("column (pixel)")
        ax.set_ylabel("row (pixel)")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_curves(s: np.ndarray, compact_records: list[dict[str, object]], output: Path) -> None:
    ranks = np.array([int(record["rank"]) for record in compact_records])
    rmse = np.array([float(record["rmse"]) for record in compact_records])
    total = np.array([float(record["total_proxy_bits"]) for record in compact_records]) / 1000.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].semilogy(np.arange(1, len(s) + 1), s, marker=".")
    axes[0].set(xlabel="component index", ylabel="singular value", title="Singular-value spectrum")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ranks, rmse, marker="o")
    axes[1].set(xlabel="retained rank", ylabel="RMSE (intensity)", title="Reconstruction error")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(ranks, total, marker="o", label="model + residual")
    best_index = int(np.argmin(total))
    axes[2].scatter([ranks[best_index]], [total[best_index]], color="red", zorder=3, label="minimum proxy")
    axes[2].set(xlabel="retained rank", ylabel="description proxy (kbit)", title="Declared MDL proxy")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_experiment(
    output_dir: Path, seed: int = 2709, ranks: Iterable[int] = DEFAULT_RANKS
) -> dict[str, object]:
    """Run the complete deterministic experiment and write JSON/PNG artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image = make_synthetic_image(seed=seed)
    rows, cols = image.shape
    u, s, vh = compact_svd(image)
    max_rank = min(image.shape)
    clean_ranks = sorted({int(rank) for rank in ranks})
    if not clean_ranks or clean_ranks[0] < 0 or clean_ranks[-1] > max_rank:
        raise ValueError(f"all ranks must be in [0, {max_rank}]")
    if 0 not in clean_ranks or max_rank not in clean_ranks:
        raise ValueError("rank scan must contain both 0 and full rank")

    records_for_plot: list[dict[str, object]] = []
    compact_records: list[dict[str, object]] = []
    for rank in clean_ranks:
        reconstruction = rank_reconstruct(u, s, vh, rank)
        metrics = image_metrics(image, reconstruction)
        parameter_count = 0 if rank == 0 else rank * (rows + cols + 1)
        proxy = mdl_proxy_bits(image - reconstruction, parameter_count)
        record = {"rank": rank, **metrics, **proxy}
        records_for_plot.append({**record, "reconstruction": reconstruction.tolist()})
        compact_records.append(record)

    best = min(compact_records, key=lambda item: float(item["total_proxy_bits"]))
    energy = np.cumsum(s**2) / np.sum(s**2)
    report: dict[str, object] = {
        "seed": seed,
        "data_contract": {
            "image_shape": [rows, cols],
            "intensity_range": [0.0, 1.0],
            "intensity_unit": "dimensionless normalized grayscale",
            "pixel_axes": "row increases downward; column increases rightward",
        },
        "svd": {
            "U_shape": list(u.shape),
            "s_shape": list(s.shape),
            "Vh_shape": list(vh.shape),
            "reconstruction_max_abs_error": float(np.max(np.abs((u * s) @ vh - image))),
            "orthogonality_U_max_abs_error": float(np.max(np.abs(u.T @ u - np.eye(len(s))))),
            "orthogonality_V_max_abs_error": float(np.max(np.abs(vh @ vh.T - np.eye(len(s))))),
            "first_10_singular_values": [float(value) for value in s[:10]],
            "energy_fraction_rank_4": float(energy[3]),
            "energy_fraction_rank_16": float(energy[15]),
        },
        "rank_scan": compact_records,
        "mdl_proxy_definition": {
            "model": "16 bits per stored scalar; r*(m+n+1) scalars",
            "residual": "fixed Gaussian sigma=0.03 plus 1/255 intensity bin",
            "warning": "teaching two-part proxy, not exact MDL/NML/stochastic complexity",
            "best_rank": int(best["rank"]),
        },
        "fusion_model_comparison": fusion_model_comparison(),
        "conditioning": condition_demo(),
    }

    _plot_reconstructions(image, records_for_plot, output_dir / "reconstructions.png")
    _plot_curves(s, compact_records, output_dir / "svd_mdl_curves.png")
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/day027-output"))
    parser.add_argument("--seed", type=int, default=2709)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_experiment(args.output_dir, seed=args.seed)
    rank4 = next(item for item in report["rank_scan"] if item["rank"] == 4)
    rank16 = next(item for item in report["rank_scan"] if item["rank"] == 16)
    cond = report["conditioning"]
    fusion = report["fusion_model_comparison"]["models"]
    print(f"image_shape={tuple(report['data_contract']['image_shape'])}")
    print(f"rank4_rmse={rank4['rmse']:.6f} rank4_psnr_db={rank4['psnr_db']:.3f}")
    print(f"rank16_rmse={rank16['rmse']:.6f} rank16_psnr_db={rank16['psnr_db']:.3f}")
    print(f"mdl_proxy_best_rank={report['mdl_proxy_definition']['best_rank']}")
    print(f"condition_number_2={cond['condition_number_2']:.1f}")
    print(f"relative_b_error={cond['relative_b_error']:.8f}")
    print(f"relative_x_error={cond['relative_x_error']:.8f}")
    print(
        "fusion_nll_nats_per_example="
        f"{fusion['two_modality_fusion']['nll_nats_per_example']:.6f} "
        "fusion_total_proxy_bits="
        f"{fusion['two_modality_fusion']['total_proxy_bits']:.3f}"
    )
    print("SUCCESS")


if __name__ == "__main__":
    main()
