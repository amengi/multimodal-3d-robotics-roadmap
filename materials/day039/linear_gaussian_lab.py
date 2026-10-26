"""Day 039: linear-Gaussian dynamics and predictive-uncertainty audit.

The dynamics use a one-dimensional position/velocity state.  The uncertainty
examples are deliberately tiny counterexamples: uncertainty scores are
evaluated against actual classification errors instead of being treated as
guarantees of correctness.
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


def make_model(
    dt_s: float = 0.5,
    acceleration_noise_std_m_s2: float = 0.25,
    position_noise_std_m: float = 1.2,
    velocity_noise_std_m_s: float = 0.35,
) -> dict[str, np.ndarray | float]:
    """Return a constant-velocity linear-Gaussian model and its noise scales."""
    scales = (
        dt_s,
        acceleration_noise_std_m_s2,
        position_noise_std_m,
        velocity_noise_std_m_s,
    )
    if not all(np.isfinite(value) and value > 0 for value in scales):
        raise ValueError("dt and all standard deviations must be positive and finite")

    f = np.array([[1.0, dt_s], [0.0, 1.0]])
    b = np.array([[0.5 * dt_s**2], [dt_s]])
    h = np.eye(2)
    g = b[:, 0]
    q = acceleration_noise_std_m_s2**2 * np.outer(g, g)
    r = np.diag([position_noise_std_m**2, velocity_noise_std_m_s**2])
    return {
        "dt_s": float(dt_s),
        "acceleration_noise_std_m_s2": float(acceleration_noise_std_m_s2),
        "position_noise_std_m": float(position_noise_std_m),
        "velocity_noise_std_m_s": float(velocity_noise_std_m_s),
        "F": f,
        "B": b,
        "H": h,
        "Q": q,
        "R": r,
    }


def propagate_gaussian(
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    controls_m_s2: np.ndarray,
    model: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate mean/covariance without measurement updates."""
    mean = np.asarray(initial_mean, dtype=float)
    covariance = np.asarray(initial_covariance, dtype=float)
    controls = np.asarray(controls_m_s2, dtype=float)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("initial mean/covariance must have shapes (2,) and (2,2)")
    if controls.ndim != 1 or not np.all(np.isfinite(controls)):
        raise ValueError("controls must have shape (T,) and contain finite accelerations")
    if not np.allclose(covariance, covariance.T) or np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
        raise ValueError("initial covariance must be symmetric positive semidefinite")

    f = np.asarray(model["F"])
    b = np.asarray(model["B"])[:, 0]
    q = np.asarray(model["Q"])
    means = np.empty((controls.size + 1, 2))
    covariances = np.empty((controls.size + 1, 2, 2))
    means[0] = mean
    covariances[0] = covariance
    for step, control in enumerate(controls):
        means[step + 1] = f @ means[step] + b * control
        covariances[step + 1] = f @ covariances[step] @ f.T + q
        covariances[step + 1] = 0.5 * (covariances[step + 1] + covariances[step + 1].T)
    return means, covariances


def simulate_trajectory(
    seed: int = 39,
    steps: int = 30,
    dt_s: float = 0.5,
) -> dict[str, np.ndarray | dict]:
    """Simulate one state trajectory and two sensor channels."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    model = make_model(dt_s=dt_s)
    rng = np.random.default_rng(seed)
    initial_mean = np.array([0.0, 2.0])
    initial_covariance = np.diag([0.4**2, 0.15**2])
    controls = np.zeros(steps)
    analytic_mean, analytic_covariance = propagate_gaussian(
        initial_mean, initial_covariance, controls, model
    )

    states = np.empty((steps + 1, 2))
    states[0] = rng.multivariate_normal(initial_mean, initial_covariance)
    f = np.asarray(model["F"])
    b = np.asarray(model["B"])[:, 0]
    acceleration_std = float(model["acceleration_noise_std_m_s2"])
    for step in range(steps):
        acceleration_noise = rng.normal(0.0, acceleration_std)
        states[step + 1] = f @ states[step] + b * (controls[step] + acceleration_noise)

    h = np.asarray(model["H"])
    r = np.asarray(model["R"])
    measurement_noise = rng.multivariate_normal(np.zeros(2), r, size=steps + 1)
    measurements = states @ h.T + measurement_noise
    times_s = np.arange(steps + 1) * dt_s
    return {
        "model": model,
        "times_s": times_s,
        "controls_m_s2": controls,
        "states": states,
        "measurements": measurements,
        "analytic_mean": analytic_mean,
        "analytic_covariance": analytic_covariance,
        "initial_mean": initial_mean,
        "initial_covariance": initial_covariance,
    }


def monte_carlo_validation(
    seed: int = 390,
    steps: int = 30,
    samples: int = 4000,
    dt_s: float = 0.5,
) -> dict[str, float | int | list]:
    """Compare final empirical moments with analytic Gaussian propagation."""
    if samples < 100:
        raise ValueError("samples must be at least 100")
    model = make_model(dt_s=dt_s)
    initial_mean = np.array([0.0, 2.0])
    initial_covariance = np.diag([0.4**2, 0.15**2])
    controls = np.zeros(steps)
    analytic_mean, analytic_covariance = propagate_gaussian(
        initial_mean, initial_covariance, controls, model
    )
    rng = np.random.default_rng(seed)
    states = rng.multivariate_normal(initial_mean, initial_covariance, size=samples)
    f = np.asarray(model["F"])
    b = np.asarray(model["B"])[:, 0]
    acceleration_std = float(model["acceleration_noise_std_m_s2"])
    for step in range(steps):
        acceleration_noise = rng.normal(0.0, acceleration_std, size=samples)
        states = states @ f.T + (controls[step] + acceleration_noise)[:, None] * b

    empirical_mean = states.mean(axis=0)
    empirical_covariance = np.cov(states, rowvar=False, ddof=1)
    final_mean = analytic_mean[-1]
    final_covariance = analytic_covariance[-1]
    position_std = math.sqrt(final_covariance[0, 0])
    lower = final_mean[0] - 1.96 * position_std
    upper = final_mean[0] + 1.96 * position_std
    coverage = np.mean((states[:, 0] >= lower) & (states[:, 0] <= upper))
    return {
        "samples": int(samples),
        "empirical_final_mean": empirical_mean.tolist(),
        "analytic_final_mean": final_mean.tolist(),
        "mean_max_abs_error": float(np.max(np.abs(empirical_mean - final_mean))),
        "empirical_final_covariance": empirical_covariance.tolist(),
        "analytic_final_covariance": final_covariance.tolist(),
        "covariance_max_abs_error": float(np.max(np.abs(empirical_covariance - final_covariance))),
        "position_95_interval_m": [float(lower), float(upper)],
        "position_95_coverage": float(coverage),
    }


def process_noise_scan(
    steps: int = 30,
    dt_s: float = 0.5,
    noise_stds_m_s2: tuple[float, ...] = (0.05, 0.25, 0.75),
) -> list[dict[str, float]]:
    """Change only acceleration-noise scale and report final position spread."""
    initial_mean = np.array([0.0, 2.0])
    initial_covariance = np.diag([0.4**2, 0.15**2])
    controls = np.zeros(steps)
    rows = []
    for noise_std in noise_stds_m_s2:
        model = make_model(dt_s=dt_s, acceleration_noise_std_m_s2=noise_std)
        _, covariance = propagate_gaussian(initial_mean, initial_covariance, controls, model)
        rows.append(
            {
                "acceleration_noise_std_m_s2": float(noise_std),
                "final_position_std_m": float(np.sqrt(covariance[-1, 0, 0])),
                "final_velocity_std_m_s": float(np.sqrt(covariance[-1, 1, 1])),
            }
        )
    return rows


def binary_entropy_bits(probability: np.ndarray | float) -> np.ndarray:
    """Binary Shannon entropy in bits, with the endpoint convention 0 log 0 = 0."""
    p = np.asarray(probability, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite and lie in [0,1]")
    p_safe = np.clip(p, EPS, 1.0 - EPS)
    entropy = -(p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))
    return np.where((p == 0.0) | (p == 1.0), 0.0, entropy)


def make_uncertainty_cases() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return labels and five-member probabilities for explicit failure cases."""
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    ensemble_probability_y1 = np.array(
        [
            [0.95, 0.94, 0.96, 0.95, 0.93],
            [0.05, 0.04, 0.06, 0.05, 0.07],
            [0.55, 0.49, 0.52, 0.50, 0.51],
            [0.45, 0.51, 0.48, 0.50, 0.49],
            [0.90, 0.10, 0.85, 0.20, 0.65],
            [0.10, 0.90, 0.15, 0.80, 0.35],
            [0.03, 0.04, 0.05, 0.03, 0.04],
            [0.97, 0.96, 0.95, 0.97, 0.96],
            [0.75, 0.70, 0.80, 0.72, 0.78],
            [0.25, 0.30, 0.20, 0.28, 0.22],
            [0.49, 0.47, 0.48, 0.46, 0.49],
            [0.51, 0.53, 0.52, 0.54, 0.51],
        ],
        dtype=float,
    )
    names = [
        "correct_confident_y1",
        "correct_confident_y0",
        "correct_high_entropy_y1",
        "correct_high_entropy_y0",
        "correct_member_disagreement_y1",
        "correct_member_disagreement_y0",
        "wrong_overconfident_y1",
        "wrong_overconfident_y0",
        "correct_medium_y1",
        "correct_medium_y0",
        "wrong_high_entropy_y1",
        "wrong_high_entropy_y0",
    ]
    return labels, ensemble_probability_y1, names


def error_detection_auc(error: np.ndarray, score: np.ndarray) -> float:
    """Pairwise ROC AUC where error=1 is the positive class."""
    error = np.asarray(error, dtype=int)
    score = np.asarray(score, dtype=float)
    if error.shape != score.shape or error.ndim != 1:
        raise ValueError("error and score must have matching shape (N,)")
    positive = score[error == 1]
    negative = score[error == 0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUC requires at least one error and one correct case")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def probability_metrics(probability_y1: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return discrimination and calibration summaries for binary probabilities."""
    probability = np.asarray(probability_y1, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probability.shape != labels.shape or probability.ndim != 1:
        raise ValueError("probability and labels must have matching shape (N,)")
    if np.any((probability < 0.0) | (probability > 1.0)) or np.any(~np.isin(labels, [0, 1])):
        raise ValueError("invalid probability or binary label")
    prediction = (probability >= 0.5).astype(int)
    target_probability = np.where(labels == 1, probability, 1.0 - probability)
    confidence = np.maximum(probability, 1.0 - probability)
    correct = prediction == labels
    ece = 0.0
    edges = np.linspace(0.5, 1.0, 6)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (confidence >= lower) & (confidence <= upper if index == 4 else confidence < upper)
        if np.any(in_bin):
            ece += np.mean(in_bin) * abs(np.mean(correct[in_bin]) - np.mean(confidence[in_bin]))
    return {
        "accuracy": float(np.mean(correct)),
        "nll_nats_per_case": -float(np.mean(np.log(np.clip(target_probability, EPS, 1.0)))),
        "brier": float(np.mean((probability - labels) ** 2)),
        "ece_5_bins": float(ece),
    }


def uncertainty_audit() -> dict:
    """Audit predictive entropy and ensemble disagreement against true errors."""
    labels, member_probability, names = make_uncertainty_cases()
    mean_probability = member_probability.mean(axis=1)
    prediction = (mean_probability >= 0.5).astype(int)
    error = (prediction != labels).astype(int)
    predictive_entropy = binary_entropy_bits(mean_probability)
    expected_member_entropy = binary_entropy_bits(member_probability).mean(axis=1)
    disagreement = predictive_entropy - expected_member_entropy
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "name": name,
                "label": int(labels[index]),
                "mean_probability_y1": float(mean_probability[index]),
                "prediction": int(prediction[index]),
                "error": int(error[index]),
                "predictive_entropy_bits": float(predictive_entropy[index]),
                "expected_member_entropy_bits": float(expected_member_entropy[index]),
                "disagreement_bits": float(disagreement[index]),
            }
        )
    return {
        "cases": rows,
        "ensemble_mean_metrics": probability_metrics(mean_probability, labels),
        "single_member_0_metrics": probability_metrics(member_probability[:, 0], labels),
        "error_detection_auc": {
            "predictive_entropy": error_detection_auc(error, predictive_entropy),
            "ensemble_disagreement": error_detection_auc(error, disagreement),
        },
        "cost": {
            "single_model_evaluations_per_case": 1,
            "ensemble_model_evaluations_per_case": int(member_probability.shape[1]),
        },
        "counterexamples": {
            "high_entropy_but_correct": [
                row["name"] for row in rows if row["error"] == 0 and row["predictive_entropy_bits"] > 0.99
            ],
            "low_entropy_but_wrong": [
                row["name"] for row in rows if row["error"] == 1 and row["predictive_entropy_bits"] < 0.30
            ],
        },
    }


def make_plots(trajectory: dict, uncertainty: dict, output_dir: Path) -> None:
    """Write one dynamics figure and one uncertainty audit figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(trajectory["times_s"])
    states = np.asarray(trajectory["states"])
    measurements = np.asarray(trajectory["measurements"])
    mean = np.asarray(trajectory["analytic_mean"])
    covariance = np.asarray(trajectory["analytic_covariance"])

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    labels = [("position", "m"), ("velocity", "m/s")]
    for dimension, (name, unit) in enumerate(labels):
        std = np.sqrt(covariance[:, dimension, dimension])
        axes[dimension].fill_between(
            times, mean[:, dimension] - 1.96 * std, mean[:, dimension] + 1.96 * std,
            alpha=0.25, label="analytic 95% marginal band",
        )
        axes[dimension].plot(times, mean[:, dimension], label="analytic mean", linewidth=2)
        axes[dimension].plot(times, states[:, dimension], label="one sampled truth", linewidth=1.5)
        axes[dimension].scatter(times, measurements[:, dimension], s=12, alpha=0.55, label="measurement")
        axes[dimension].set_ylabel(f"{name} ({unit})")
        axes[dimension].grid(alpha=0.25)
        axes[dimension].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle("Linear-Gaussian prior propagation (no measurement update)")
    figure.tight_layout()
    figure.savefig(output_dir / "linear_gaussian_trajectory.png", dpi=160)
    plt.close(figure)

    rows = uncertainty["cases"]
    names = [row["name"].replace("_", "\n") for row in rows]
    entropy = [row["predictive_entropy_bits"] for row in rows]
    disagreement = [row["disagreement_bits"] for row in rows]
    colors = ["#d62728" if row["error"] else "#2ca02c" for row in rows]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(x - 0.2, entropy, width=0.4, color=colors, alpha=0.75, label="predictive entropy")
    axis.bar(x + 0.2, disagreement, width=0.4, color="#1f77b4", alpha=0.85, label="ensemble disagreement")
    axis.set_xticks(x, names, fontsize=7)
    axis.set_ylabel("uncertainty score (bit)")
    axis.set_title("Green/red entropy bars are correct/error cases; blue is disagreement")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "uncertainty_audit.png", dpi=160)
    plt.close(figure)


def run(output_dir: Path, seed: int = 39, steps: int = 30, mc_samples: int = 4000) -> dict:
    """Run the complete lab, save outputs, and return the report."""
    trajectory = simulate_trajectory(seed=seed, steps=steps)
    mc = monte_carlo_validation(seed=seed * 10, steps=steps, samples=mc_samples)
    uncertainty = uncertainty_audit()
    model = trajectory["model"]
    states = np.asarray(trajectory["states"])
    measurements = np.asarray(trajectory["measurements"])
    analytic_mean = np.asarray(trajectory["analytic_mean"])
    report = {
        "config": {"seed": seed, "steps": steps, "mc_samples": mc_samples, "dt_s": model["dt_s"]},
        "linear_gaussian": {
            "state_shape": list(states.shape),
            "measurement_shape": list(measurements.shape),
            "state_contract": ["position_m", "velocity_m_s"],
            "measurement_contract": ["position_sensor_m", "velocity_sensor_m_s"],
            "F": np.asarray(model["F"]).tolist(),
            "B": np.asarray(model["B"]).tolist(),
            "H": np.asarray(model["H"]).tolist(),
            "Q": np.asarray(model["Q"]).tolist(),
            "R": np.asarray(model["R"]).tolist(),
            "sample_position_rmse_from_mean_m": float(np.sqrt(np.mean((states[:, 0] - analytic_mean[:, 0]) ** 2))),
            "position_measurement_rmse_m": float(np.sqrt(np.mean((measurements[:, 0] - states[:, 0]) ** 2))),
            "velocity_measurement_rmse_m_s": float(np.sqrt(np.mean((measurements[:, 1] - states[:, 1]) ** 2))),
            "final_position_std_m": float(np.sqrt(np.asarray(trajectory["analytic_covariance"])[-1, 0, 0])),
            "monte_carlo": mc,
            "process_noise_robustness_scan": process_noise_scan(steps=steps),
        },
        "predictive_uncertainty": uncertainty,
        "limitations": [
            "Linear-Gaussian dynamics and declared noise scales are assumptions, not learned truths.",
            "No Kalman measurement update is performed; that is the next lesson.",
            "The twelve classification cases are constructed counterexamples, not clinical evidence.",
            "Entropy and disagreement do not prove correctness, causality, calibration, or safety.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    make_plots(trajectory, uncertainty, output_dir)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/day039-output"))
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--mc-samples", type=int, default=4000)
    args = parser.parse_args()
    report = run(args.output_dir, seed=args.seed, steps=args.steps, mc_samples=args.mc_samples)
    linear = report["linear_gaussian"]
    mc = linear["monte_carlo"]
    uncertainty = report["predictive_uncertainty"]
    metrics = uncertainty["ensemble_mean_metrics"]
    print("F =", np.asarray(linear["F"]))
    print("Q =", np.asarray(linear["Q"]))
    print(f"Final position std: {linear['final_position_std_m']:.6f} m")
    print(f"Monte Carlo final-position 95% coverage: {mc['position_95_coverage']:.4f}")
    print(
        "Ensemble accuracy/NLL/Brier/ECE: "
        f"{metrics['accuracy']:.4f}/{metrics['nll_nats_per_case']:.4f}/"
        f"{metrics['brier']:.4f}/{metrics['ece_5_bins']:.4f}"
    )
    print("Error-detection AUC:", uncertainty["error_detection_auc"])
    print("High-entropy correct:", uncertainty["counterexamples"]["high_entropy_but_correct"])
    print("Low-entropy wrong:", uncertainty["counterexamples"]["low_entropy_but_wrong"])
    print(f"Wrote {args.output_dir / 'report.json'}")
    print("SUCCESS: linear-Gaussian propagation and uncertainty audit completed")


if __name__ == "__main__":
    main()
