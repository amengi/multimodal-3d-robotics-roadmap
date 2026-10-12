"""Day 029: finite differences, Taylor approximation, and task representations."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_X0 = 0.7
DEFAULT_EPSILON = 0.1


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def forward_difference(function: Callable[[float], float], x: float, h: float) -> float:
    """Approximate f'(x) with a first-order forward difference."""
    h = _positive_finite(h, "h")
    return (float(function(x + h)) - float(function(x))) / h


def central_difference(function: Callable[[float], float], x: float, h: float) -> float:
    """Approximate f'(x) with a second-order central difference."""
    h = _positive_finite(h, "h")
    return (float(function(x + h)) - float(function(x - h))) / (2.0 * h)


def sin_taylor(x: np.ndarray | float, center: float, order: int) -> np.ndarray:
    """Return the order-0, order-1, or order-2 Taylor polynomial of sin."""
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1, or 2")
    values = np.asarray(x, dtype=float)
    delta = values - float(center)
    approximation = np.full_like(values, math.sin(center), dtype=float)
    if order >= 1:
        approximation += math.cos(center) * delta
    if order >= 2:
        approximation -= 0.5 * math.sin(center) * delta**2
    return approximation


def derivative_sweep(x0: float = DEFAULT_X0) -> dict[str, Any]:
    """Sweep h from 1e-1 to 1e-14 for sin'(x0)."""
    if not math.isfinite(x0):
        raise ValueError("x0 must be finite")
    steps = np.logspace(-1, -14, 14)
    exact = math.cos(x0)
    forward = np.array([forward_difference(math.sin, x0, h) for h in steps])
    central = np.array([central_difference(math.sin, x0, h) for h in steps])
    forward_error = np.abs(forward - exact)
    central_error = np.abs(central - exact)
    best_forward = int(np.argmin(forward_error))
    best_central = int(np.argmin(central_error))
    return {
        "x0_rad": x0,
        "exact_derivative_per_rad": exact,
        "steps_rad": steps.tolist(),
        "forward_estimate_per_rad": forward.tolist(),
        "central_estimate_per_rad": central.tolist(),
        "forward_absolute_error_per_rad": forward_error.tolist(),
        "central_absolute_error_per_rad": central_error.tolist(),
        "best_forward": {
            "h_rad": float(steps[best_forward]),
            "absolute_error_per_rad": float(forward_error[best_forward]),
        },
        "best_central": {
            "h_rad": float(steps[best_central]),
            "absolute_error_per_rad": float(central_error[best_central]),
        },
    }


def taylor_audit(center: float = DEFAULT_X0, target: float = 1.2) -> dict[str, Any]:
    """Compare first- and second-order local models of sin at one target."""
    if not math.isfinite(center) or not math.isfinite(target):
        raise ValueError("center and target must be finite")
    exact = math.sin(target)
    first = float(sin_taylor(target, center, 1))
    second = float(sin_taylor(target, center, 2))
    return {
        "center_rad": center,
        "target_rad": target,
        "delta_rad": target - center,
        "exact_sin": exact,
        "first_order": first,
        "second_order": second,
        "first_absolute_error": abs(first - exact),
        "second_absolute_error": abs(second - exact),
    }


def binary_entropy_bits(probability: float) -> float:
    """Binary entropy H_b(p), in bit."""
    probability = float(probability)
    if probability < 0.0 or probability > 1.0 or not math.isfinite(probability):
        raise ValueError("probability must be finite and in [0,1]")
    if probability in (0.0, 1.0):
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability)


def make_joint_xy(epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Build p(X=(S,N),Y), with fair independent bits S,N and Y=S xor E."""
    epsilon = float(epsilon)
    if not 0.0 < epsilon < 0.5 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and in (0,0.5)")
    joint = np.zeros((4, 2), dtype=float)
    for signal in (0, 1):
        for nuisance in (0, 1):
            x_index = 2 * signal + nuisance
            joint[x_index, signal] = 0.25 * (1.0 - epsilon)
            joint[x_index, 1 - signal] = 0.25 * epsilon
    return joint


def mutual_information_bits(joint: np.ndarray) -> float:
    """Compute discrete mutual information from a normalized joint PMF."""
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2 or np.any(joint < 0.0) or not np.all(np.isfinite(joint)):
        raise ValueError("joint must be a finite nonnegative 2D array")
    if not math.isclose(float(joint.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("joint must sum to one")
    row = joint.sum(axis=1, keepdims=True)
    column = joint.sum(axis=0, keepdims=True)
    expected = row @ column
    mask = joint > 0.0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / expected[mask])))


def encode_joint(joint_xy: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Map X states deterministically to Z states and return p(Z,Y)."""
    joint_xy = np.asarray(joint_xy, dtype=float)
    mapping = np.asarray(mapping, dtype=int)
    if joint_xy.shape != (4, 2) or mapping.shape != (4,):
        raise ValueError("expected joint_xy shape (4,2) and mapping shape (4,)")
    if np.any(mapping < 0):
        raise ValueError("mapping labels must be nonnegative")
    unique = np.unique(mapping)
    if not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError("mapping labels must be contiguous from zero")
    joint_zy = np.zeros((unique.size, 2), dtype=float)
    for x_index, z_index in enumerate(mapping):
        joint_zy[z_index] += joint_xy[x_index]
    return joint_zy


def conditional_y1(joint: np.ndarray) -> np.ndarray:
    """Return p(Y=1 | row state) for a two-class joint PMF."""
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2 or joint.shape[1] != 2:
        raise ValueError("joint must have shape (states,2)")
    totals = joint.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("every state must have positive mass")
    return joint[:, 1] / totals


def prediction_metrics(joint_xy: np.ndarray, probability_y1_by_x: np.ndarray) -> dict[str, float]:
    """Compute exact accuracy, NLL, Brier, and calibration error."""
    joint_xy = np.asarray(joint_xy, dtype=float)
    probability = np.asarray(probability_y1_by_x, dtype=float)
    if joint_xy.shape != (4, 2) or probability.shape != (4,):
        raise ValueError("expected joint_xy shape (4,2) and four probabilities")
    if np.any(probability <= 0.0) or np.any(probability >= 1.0) or not np.all(np.isfinite(probability)):
        raise ValueError("predicted probabilities must be finite and in (0,1)")
    predicted_class = probability > 0.5
    accuracy = sum(joint_xy[x, int(predicted_class[x])] for x in range(4))
    nll = 0.0
    brier = 0.0
    for x_index in range(4):
        for label in (0, 1):
            observed_probability = probability[x_index] if label == 1 else 1.0 - probability[x_index]
            nll -= joint_xy[x_index, label] * math.log(observed_probability)
            brier += joint_xy[x_index, label] * (probability[x_index] - label) ** 2
    calibration_error = 0.0
    for predicted_probability in np.unique(probability):
        selected = np.isclose(probability, predicted_probability, rtol=0.0, atol=1e-15)
        weight = float(joint_xy[selected].sum())
        actual = float(joint_xy[selected, 1].sum() / weight)
        calibration_error += weight * abs(actual - float(predicted_probability))
    return {
        "accuracy": float(accuracy),
        "nll_nat_per_example": float(nll),
        "brier_per_example": float(brier),
        "calibration_error": float(calibration_error),
    }


def representation_audit(epsilon: float = DEFAULT_EPSILON) -> dict[str, Any]:
    """Audit deterministic representations Z=f(X) for task Y."""
    joint_xy = make_joint_xy(epsilon)
    information_xy = mutual_information_bits(joint_xy)
    conditional_x = conditional_y1(joint_xy)
    mappings = {
        "full_x": np.array([0, 1, 2, 3]),
        "signal_z": np.array([0, 0, 1, 1]),
        "nuisance_z": np.array([0, 1, 0, 1]),
        "constant_z": np.array([0, 0, 0, 0]),
    }
    rows: dict[str, Any] = {}
    for name, mapping in mappings.items():
        joint_zy = encode_joint(joint_xy, mapping)
        probability_by_z = conditional_y1(joint_zy)
        probability_by_x = probability_by_z[mapping]
        information_zy = mutual_information_bits(joint_zy)
        rows[name] = {
            "states": int(joint_zy.shape[0]),
            "fixed_length_cost_bits": math.log2(joint_zy.shape[0]),
            "information_with_y_bits": information_zy,
            "dpi_slack_bits": information_xy - information_zy,
            "sufficiency_gap_max_probability": float(np.max(np.abs(conditional_x - probability_by_x))),
            **prediction_metrics(joint_xy, probability_by_x),
        }

    signal_mapping = mappings["signal_z"]
    overconfident = np.where(signal_mapping == 1, 0.99, 0.01)
    rows["signal_overconfident"] = {
        **rows["signal_z"],
        "predictor_note": "Same representation and accuracy; probabilities 0.99/0.01 are miscalibrated.",
        **prediction_metrics(joint_xy, overconfident),
    }

    shifted_joint = make_joint_xy(0.3)
    training_probability = np.where(signal_mapping == 1, 1.0 - epsilon, epsilon)
    shifted_metrics = prediction_metrics(shifted_joint, training_probability)
    return {
        "data_generating_process": {
            "X": "X=(S,N), two fair independent bits",
            "Y": f"Y=S xor E, E~Bernoulli({epsilon})",
            "Z": "deterministic representation f(X)",
            "units": "information in bit/example; NLL in nat/example; other metrics dimensionless",
        },
        "information_x_y_bits": information_xy,
        "analytic_information_bits": 1.0 - binary_entropy_bits(epsilon),
        "representations": rows,
        "robustness_shift": {
            "test_epsilon": 0.3,
            "information_x_y_bits": mutual_information_bits(shifted_joint),
            "predictor_kept_at_training_probabilities": True,
            **shifted_metrics,
        },
        "warning": "Dependence and sufficiency in this synthetic PMF do not establish causality or medical validity.",
    }


def plot_derivative_and_taylor(sweep: dict[str, Any], destination: Path) -> None:
    steps = np.asarray(sweep["steps_rad"])
    x0 = float(sweep["x0_rad"])
    domain = np.linspace(0.0, 1.4, 300)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].loglog(steps, sweep["forward_absolute_error_per_rad"], marker="o", label="forward")
    axes[0].loglog(steps, sweep["central_absolute_error_per_rad"], marker="s", label="central")
    axes[0].invert_xaxis()
    axes[0].set(xlabel="step h (rad)", ylabel="absolute derivative error (per rad)", title="Truncation versus roundoff")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend()
    axes[1].plot(domain, np.sin(domain), label="sin(theta)")
    axes[1].plot(domain, sin_taylor(domain, x0, 1), label="first order")
    axes[1].plot(domain, sin_taylor(domain, x0, 2), label="second order")
    axes[1].axvline(x0, color="black", linestyle="--", linewidth=1, label="center 0.7 rad")
    axes[1].set(xlabel="theta (rad)", ylabel="dimensionless value", title="Local Taylor models")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Day 029: finite differences and Taylor approximation")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_representation_audit(audit: dict[str, Any], destination: Path) -> None:
    names = ["full_x", "signal_z", "nuisance_z", "constant_z"]
    labels = ["full X", "signal Z", "nuisance Z", "constant Z"]
    rows = audit["representations"]
    costs = [rows[name]["fixed_length_cost_bits"] for name in names]
    information = [rows[name]["information_with_y_bits"] for name in names]
    accuracy = [rows[name]["accuracy"] for name in names]
    nll = [rows[name]["nll_nat_per_example"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(costs, information, s=70)
    for x_value, y_value, label in zip(costs, information, labels):
        axes[0].annotate(label, (x_value, y_value), xytext=(4, 5), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="fixed-length representation cost (bit)", ylabel="I(Z;Y) (bit/example)", title="Compression and task information")
    axes[0].grid(alpha=0.25)
    positions = np.arange(len(names))
    width = 0.36
    axes[1].bar(positions - width / 2, accuracy, width, label="accuracy")
    axes[1].bar(positions + width / 2, nll, width, label="NLL (nat/example)")
    axes[1].set_xticks(positions, labels, rotation=15)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set(title="Task baselines", ylabel="metric value")
    axes[1].legend()
    axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle("Day 029: exact discrete representation audit")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_experiment(output_dir: Path, x0: float = DEFAULT_X0, epsilon: float = DEFAULT_EPSILON) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = derivative_sweep(x0)
    taylor = taylor_audit(x0)
    representation = representation_audit(epsilon)
    report = {
        "configuration": {"x0_rad": x0, "label_flip_probability": epsilon},
        "finite_difference": sweep,
        "taylor": taylor,
        "representation": representation,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_derivative_and_taylor(sweep, output_dir / "derivative_taylor.png")
    plot_representation_audit(representation, output_dir / "representation_audit.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/day029-output"))
    parser.add_argument("--x0", type=float, default=DEFAULT_X0)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    args = parser.parse_args()
    report = run_experiment(args.output_dir, args.x0, args.epsilon)
    derivative = report["finite_difference"]
    taylor = report["taylor"]
    representation = report["representation"]
    rows = representation["representations"]
    print(f"sin_x0={args.x0:.3f} exact_derivative={derivative['exact_derivative_per_rad']:.9f}_per_rad")
    print(
        "best_forward="
        f"h:{derivative['best_forward']['h_rad']:.1e}_rad,error:{derivative['best_forward']['absolute_error_per_rad']:.3e}_per_rad"
    )
    print(
        "best_central="
        f"h:{derivative['best_central']['h_rad']:.1e}_rad,error:{derivative['best_central']['absolute_error_per_rad']:.3e}_per_rad"
    )
    print(
        f"taylor_at_1.2_rad=first_error:{taylor['first_absolute_error']:.6f},"
        f"second_error:{taylor['second_absolute_error']:.6f}"
    )
    for name in ("full_x", "signal_z", "nuisance_z", "constant_z", "signal_overconfident"):
        row = rows[name]
        print(
            f"{name}=I:{row['information_with_y_bits']:.6f}_bit,acc:{row['accuracy']:.3f},"
            f"nll:{row['nll_nat_per_example']:.6f}_nat,brier:{row['brier_per_example']:.3f},"
            f"ece:{row['calibration_error']:.3f},cost:{row['fixed_length_cost_bits']:.1f}_bit"
        )
    shift = representation["robustness_shift"]
    print(
        f"shift_epsilon_0.3=acc:{shift['accuracy']:.3f},nll:{shift['nll_nat_per_example']:.6f}_nat,"
        f"brier:{shift['brier_per_example']:.3f},ece:{shift['calibration_error']:.3f}"
    )
    print("SUCCESS")


if __name__ == "__main__":
    main()
