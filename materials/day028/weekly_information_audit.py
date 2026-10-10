"""Finite-sample information-metric audit for Day 028.

All data are synthetic and dimensionless.  The plug-in estimators in this file
are teaching tools, not ground-truth entropy or mutual information estimators.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SEED = 2810
SAMPLE_SIZES = (80, 200, 800)
REPEATS = 40
BIN_EDGES = np.linspace(-4.0, 4.0, 9)
N_BINS = len(BIN_EDGES) - 1


def _as_labels(values: np.ndarray, name: str) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{name} must contain integer labels")
    if np.any(labels < 0):
        raise ValueError(f"{name} labels must be non-negative")
    return labels.astype(np.int64, copy=False)


def discretize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return np.clip(np.digitize(array, BIN_EDGES[1:-1]), 0, N_BINS - 1).astype(np.int64)


def pmf(labels: np.ndarray, n_classes: int | None = None, alpha: float = 0.0) -> np.ndarray:
    labels = _as_labels(labels, "labels")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    inferred = int(labels.max()) + 1
    size = inferred if n_classes is None else int(n_classes)
    if size < inferred or size <= 0:
        raise ValueError("n_classes is too small")
    counts = np.bincount(labels, minlength=size).astype(float) + alpha
    return counts / counts.sum()


def entropy_bits(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.size == 0 or np.any(p < 0) or not np.isfinite(p).all():
        raise ValueError("probabilities must be a finite non-negative vector")
    if not np.isclose(p.sum(), 1.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    positive = p > 0
    return float(-np.sum(p[positive] * np.log2(p[positive])))


def kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be same-shaped vectors")
    if np.any(p < 0) or np.any(q < 0) or not np.isclose(p.sum(), 1.0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("p and q must be probability vectors")
    if np.any((p > 0) & (q == 0)):
        return math.inf
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js_bits(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    midpoint = 0.5 * (p + q)
    return 0.5 * kl_bits(p, midpoint) + 0.5 * kl_bits(q, midpoint)


def joint_pmf(x: np.ndarray, y: np.ndarray, n_classes: int = N_BINS, alpha: float = 0.0) -> np.ndarray:
    x = _as_labels(x, "x")
    y = _as_labels(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must be paired and have the same shape")
    if np.any(x >= n_classes) or np.any(y >= n_classes):
        raise ValueError("labels exceed n_classes")
    counts = np.full((n_classes, n_classes), alpha, dtype=float)
    np.add.at(counts, (x, y), 1.0)
    return counts / counts.sum()


def mutual_information_bits(joint: np.ndarray) -> float:
    pxy = np.asarray(joint, dtype=float)
    if pxy.ndim != 2 or np.any(pxy < 0) or not np.isclose(pxy.sum(), 1.0):
        raise ValueError("joint must be a 2-D probability table summing to one")
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    product = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log2(pxy[mask] / product[mask])))


def normalized_mi(joint: np.ndarray) -> float:
    pxy = np.asarray(joint, dtype=float)
    hx = entropy_bits(pxy.sum(axis=1))
    hy = entropy_bits(pxy.sum(axis=0))
    denominator = hx + hy
    return 0.0 if denominator == 0 else float(2.0 * mutual_information_bits(pxy) / denominator)


def generate_scenario(name: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, bool]:
    if n < 20:
        raise ValueError("n must be at least 20")
    if name == "independent":
        return rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n), True
    if name == "linear":
        x = rng.normal(0.0, 1.0, n)
        return x, 0.85 * x + rng.normal(0.0, 0.45, n), True
    if name == "nonlinear":
        x = rng.uniform(-2.0, 2.0, n)
        return x, x * x + rng.normal(0.0, 0.25, n), True
    if name == "distribution_shift":
        # These are independent source/target samples, not aligned pairs.
        return rng.normal(-0.6, 0.8, n), rng.normal(0.8, 1.1, n), False
    raise ValueError(f"unknown scenario: {name}")


def prediction_baselines(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = _as_labels(x, "x")
    y = _as_labels(y, "y")
    if x.shape != y.shape or x.size < 20:
        raise ValueError("paired labels with at least 20 examples are required")
    split = max(10, int(0.7 * x.size))
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]
    majority = int(np.argmax(np.bincount(train_y, minlength=N_BINS)))
    majority_accuracy = float(np.mean(test_y == majority))
    lookup = np.full(N_BINS, majority, dtype=np.int64)
    for label in range(N_BINS):
        observed = train_y[train_x == label]
        if observed.size:
            lookup[label] = int(np.argmax(np.bincount(observed, minlength=N_BINS)))
    lookup_accuracy = float(np.mean(test_y == lookup[test_x]))
    return {"majority_accuracy": majority_accuracy, "lookup_accuracy": lookup_accuracy}


def one_estimate(name: str, n: int, rng: np.random.Generator) -> dict[str, float | None]:
    x_cont, y_cont, paired = generate_scenario(name, n, rng)
    x = discretize(x_cont)
    y = discretize(y_cont)
    # Laplace smoothing makes KL finite; the exact alpha is part of the estimator.
    px = pmf(x, N_BINS, alpha=0.5)
    py = pmf(y, N_BINS, alpha=0.5)
    result: dict[str, float | None] = {
        "entropy_x_bits": entropy_bits(px),
        "entropy_y_bits": entropy_bits(py),
        "kl_x_to_y_bits": kl_bits(px, py),
        "js_bits": js_bits(px, py),
        "mi_bits": None,
        "nmi_arithmetic": None,
        "pearson_r": None,
        "majority_accuracy": None,
        "lookup_accuracy": None,
    }
    if paired:
        joint = joint_pmf(x, y, N_BINS)
        baselines = prediction_baselines(x, y)
        result.update(
            {
                "mi_bits": mutual_information_bits(joint),
                "nmi_arithmetic": normalized_mi(joint),
                "pearson_r": float(np.corrcoef(x_cont, y_cont)[0, 1]),
                **baselines,
            }
        )
    return result


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def finite_sample_audit(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    scenarios = ("independent", "linear", "nonlinear", "distribution_shift")
    output: dict[str, Any] = {}
    for scenario_index, scenario in enumerate(scenarios):
        output[scenario] = {}
        for size in SAMPLE_SIZES:
            records: list[dict[str, float | None]] = []
            for repeat in range(REPEATS):
                child_seed = seed + 100_000 * scenario_index + 100 * size + repeat
                records.append(one_estimate(scenario, size, np.random.default_rng(child_seed)))
            summarized: dict[str, Any] = {}
            for key in records[0]:
                numeric = [float(record[key]) for record in records if record[key] is not None]
                summarized[key] = _summary(numeric) if numeric else None
            output[scenario][str(size)] = summarized
    return output


def binary_entropy_bits(d: float) -> float:
    if not 0.0 <= d <= 1.0:
        raise ValueError("d must be in [0,1]")
    if d in (0.0, 1.0):
        return 0.0
    return float(-d * math.log2(d) - (1.0 - d) * math.log2(1.0 - d))


def rate_distortion_demo() -> list[dict[str, float]]:
    rows = []
    for distortion in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        rate = max(0.0, 1.0 - binary_entropy_bits(distortion))
        rows.append({"hamming_distortion": distortion, "rate_bits_per_symbol": rate})
    return rows


def _conditional_probabilities(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    counts = np.full((N_BINS, N_BINS), alpha, dtype=float)
    np.add.at(counts, (x, y), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def mdl_proxy_demo(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 999)
    x_cont, y_cont, _ = generate_scenario("nonlinear", 900, rng)
    x, y = discretize(x_cont), discretize(y_cont)
    split = 600
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]

    marginal = pmf(train_y, N_BINS, alpha=1.0)
    conditional = _conditional_probabilities(train_x, train_y, alpha=1.0)
    models = {
        "marginal_y": {
            "parameter_count": N_BINS - 1,
            "probability": marginal[test_y],
            "prediction": np.full(test_y.shape, int(np.argmax(marginal))),
        },
        "conditional_y_given_x": {
            "parameter_count": N_BINS * (N_BINS - 1),
            "probability": conditional[test_x, test_y],
            "prediction": np.argmax(conditional[test_x], axis=1),
        },
    }
    result: dict[str, Any] = {}
    for name, model in models.items():
        nll_bits = float(-np.log2(model["probability"]).sum())
        model_bits = float(4 * model["parameter_count"])
        result[name] = {
            "parameter_count": int(model["parameter_count"]),
            "model_bits_at_4_per_parameter": model_bits,
            "test_nll_bits": nll_bits,
            "total_proxy_bits": model_bits + nll_bits,
            "test_accuracy": float(np.mean(model["prediction"] == test_y)),
        }
    return {
        "train_examples": split,
        "test_examples": int(test_y.size),
        "smoothing_alpha": 1.0,
        "warning": "Two-part classroom proxy; not exact MDL or NML.",
        "models": result,
    }


def _plot_interval(axis: Any, sizes: np.ndarray, rows: list[dict[str, float]], label: str) -> None:
    means = np.array([row["mean"] for row in rows])
    lows = np.array([row["p05"] for row in rows])
    highs = np.array([row["p95"] for row in rows])
    axis.plot(sizes, means, marker="o", label=label)
    axis.fill_between(sizes, lows, highs, alpha=0.15)


def plot_finite_sample(audit: dict[str, Any], destination: Path) -> None:
    sizes = np.asarray(SAMPLE_SIZES)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for scenario in ("independent", "linear", "nonlinear"):
        _plot_interval(axes[0, 0], sizes, [audit[scenario][str(n)]["mi_bits"] for n in sizes], scenario)
        _plot_interval(axes[1, 0], sizes, [audit[scenario][str(n)]["pearson_r"] for n in sizes], scenario)
        _plot_interval(axes[1, 1], sizes, [audit[scenario][str(n)]["lookup_accuracy"] for n in sizes], scenario)
    for scenario in ("independent", "linear", "nonlinear", "distribution_shift"):
        _plot_interval(axes[0, 1], sizes, [audit[scenario][str(n)]["js_bits"] for n in sizes], scenario)
    axes[0, 0].set(title="Plug-in MI (paired only)", ylabel="MI (bit)")
    axes[0, 1].set(title="JS between marginals", ylabel="JS (bit)")
    axes[1, 0].set(title="Ordinary dependence baseline", ylabel="Pearson r")
    axes[1, 1].set(title="Supervised lookup baseline", ylabel="test accuracy")
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xlabel("sample size n")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Day 028: mean and 5–95% interval over 40 synthetic repeats")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_rate_mdl(rate_rows: list[dict[str, float]], mdl: dict[str, Any], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    distortion = [row["hamming_distortion"] for row in rate_rows]
    rate = [row["rate_bits_per_symbol"] for row in rate_rows]
    axes[0].plot(distortion, rate, marker="o")
    axes[0].set(xlabel="Hamming distortion D", ylabel="R(D) (bit/symbol)", title="Bernoulli(0.5) rate–distortion")
    axes[0].grid(alpha=0.25)

    names = list(mdl["models"])
    model_bits = [mdl["models"][name]["model_bits_at_4_per_parameter"] for name in names]
    nll_bits = [mdl["models"][name]["test_nll_bits"] for name in names]
    positions = np.arange(len(names))
    axes[1].bar(positions, model_bits, label="model proxy")
    axes[1].bar(positions, nll_bits, bottom=model_bits, label="test NLL")
    axes[1].set_xticks(positions, ["marginal", "conditional"])
    axes[1].set(ylabel="two-part proxy (bit)", title="Declared MDL-like proxy")
    axes[1].legend()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_experiment(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = finite_sample_audit(seed)
    rate_rows = rate_distortion_demo()
    mdl = mdl_proxy_demo(seed)
    report = {
        "seed": seed,
        "data_contract": {
            "sample_sizes": list(SAMPLE_SIZES),
            "repeats": REPEATS,
            "bin_edges": BIN_EDGES.tolist(),
            "units": "synthetic dimensionless values; information in bit",
            "paired_scenarios": ["independent", "linear", "nonlinear"],
            "unpaired_scenario": "distribution_shift",
        },
        "estimator_contract": {
            "histogram": "8 fixed-width bins over [-4,4]",
            "marginal_smoothing_alpha": 0.5,
            "mi_smoothing_alpha": 0.0,
            "nmi_definition": "2 I(X;Y) / (H(X)+H(Y))",
            "interval": "empirical 5th and 95th percentiles over 40 repeats",
            "warning": "Finite-sample plug-in estimates; not ground-truth MI.",
        },
        "finite_sample_audit": audit,
        "rate_distortion": {
            "source": "Bernoulli(0.5)",
            "distortion": "Hamming error probability",
            "formula": "R(D)=1-h2(D), 0<=D<=0.5",
            "rows": rate_rows,
        },
        "mdl_proxy": mdl,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_finite_sample(audit, output_dir / "finite_sample_metrics.png")
    plot_rate_mdl(rate_rows, mdl, output_dir / "rate_distortion_mdl.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/day028-output"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    report = run_experiment(args.output_dir, args.seed)
    audit = report["finite_sample_audit"]
    mdl = report["mdl_proxy"]["models"]
    print(f"sample_sizes={report['data_contract']['sample_sizes']} repeats={REPEATS}")
    for scenario in ("independent", "linear", "nonlinear"):
        row = audit[scenario]["800"]
        print(
            f"{scenario}_n800_mi_bits={row['mi_bits']['mean']:.6f} "
            f"pearson_r={row['pearson_r']['mean']:.6f} "
            f"lookup_accuracy={row['lookup_accuracy']['mean']:.6f}"
        )
    shift = audit["distribution_shift"]["800"]
    print(f"shift_n800_kl_bits={shift['kl_x_to_y_bits']['mean']:.6f} js_bits={shift['js_bits']['mean']:.6f} mi=NA_unpaired")
    print("rate_D0.10_bits_per_symbol=" + f"{report['rate_distortion']['rows'][2]['rate_bits_per_symbol']:.6f}")
    print(
        "mdl_total_proxy_bits="
        f"marginal:{mdl['marginal_y']['total_proxy_bits']:.3f},"
        f"conditional:{mdl['conditional_y_given_x']['total_proxy_bits']:.3f}"
    )
    print("SUCCESS")


if __name__ == "__main__":
    main()
