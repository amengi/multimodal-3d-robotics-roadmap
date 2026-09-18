"""Day 012: auditable Matplotlib figures and conditional probabilities."""

from __future__ import annotations

import argparse
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
    """Raised when an input violates a documented data contract."""


def _finite_array(value: Any, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    return array


def validate_trajectory(value: Any, name: str = "trajectory") -> FloatArray:
    """Return a finite trajectory with shape (T, 3), in metres."""

    trajectory = _finite_array(value, name)
    if trajectory.ndim != 2 or trajectory.shape[1:] != (3,) or trajectory.shape[0] < 3:
        raise ContractError(f"{name} must have shape (T, 3) with T >= 3")
    return trajectory


def make_camera_trajectories(
    seed: int = 20260918, n_frames: int = 60
) -> tuple[FloatArray, FloatArray]:
    """Create deterministic true/estimated camera positions in a world frame."""

    if not isinstance(n_frames, int) or n_frames < 3:
        raise ContractError("n_frames must be an integer >= 3")
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 2.0 * np.pi, n_frames)
    truth_m = np.column_stack(
        [2.0 * np.cos(phase), 2.0 * np.sin(phase), 0.12 * phase]
    )
    drift_m = np.column_stack(
        [0.035 * phase, -0.020 * phase, 0.012 * phase]
    )
    estimate_m = truth_m + drift_m + rng.normal(0.0, 0.035, truth_m.shape)
    return truth_m, estimate_m


def trajectory_errors_m(truth_m: Any, estimate_m: Any) -> FloatArray:
    """Return per-frame Euclidean position error in metres."""

    truth = validate_trajectory(truth_m, "truth_m")
    estimate = validate_trajectory(estimate_m, "estimate_m")
    if truth.shape != estimate.shape:
        raise ContractError("truth_m and estimate_m must share shape (T, 3)")
    return np.linalg.norm(estimate - truth, axis=1)


def equal_axis_limits(points_m: Any, padding_fraction: float = 0.05) -> tuple[tuple[float, float], ...]:
    """Return equal-span x/y/z limits so geometry is not visually distorted."""

    points = validate_trajectory(points_m, "points_m")
    if not np.isfinite(padding_fraction) or padding_fraction < 0.0:
        raise ContractError("padding_fraction must be finite and nonnegative")
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = float(np.max(upper - lower))
    if span <= 0.0:
        raise ContractError("points_m must have nonzero spatial extent")
    half = 0.5 * span * (1.0 + 2.0 * padding_fraction)
    center = 0.5 * (lower + upper)
    return tuple((float(c - half), float(c + half)) for c in center)


def validate_joint_table(value: Any) -> FloatArray:
    """Validate a 2x2 joint PMF whose rows are X and columns are Y."""

    joint = _finite_array(value, "joint probability table")
    if joint.shape != (2, 2):
        raise ContractError("joint probability table must have shape (2, 2)")
    if np.any(joint < 0.0):
        raise ContractError("joint probabilities must be nonnegative")
    if not np.isclose(joint.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ContractError("joint probabilities must sum to 1")
    row_marginal = joint.sum(axis=1)
    column_marginal = joint.sum(axis=0)
    if np.any(row_marginal <= 0.0) or np.any(column_marginal <= 0.0):
        raise ContractError("all row and column marginals must be positive")
    return joint


def probability_summary(value: Any) -> dict[str, float]:
    """Compute marginals and both conditional directions from a joint PMF."""

    joint = validate_joint_table(value)
    p_x1 = float(joint[1, :].sum())
    p_y1 = float(joint[:, 1].sum())
    p_x1_y1 = float(joint[1, 1] / p_y1)
    p_y1_x1 = float(joint[1, 1] / p_x1)
    bayes_x1_y1 = p_y1_x1 * p_x1 / p_y1
    return {
        "p_x1": p_x1,
        "p_y1": p_y1,
        "p_x1_and_y1": float(joint[1, 1]),
        "p_x1_given_y1": p_x1_y1,
        "p_y1_given_x1": p_y1_x1,
        "bayes_p_x1_given_y1": bayes_x1_y1,
    }


def _save_trajectory_figure(
    truth_m: FloatArray,
    estimate_m: FloatArray,
    output_dir: Path,
) -> Path:
    errors_m = trajectory_errors_m(truth_m, estimate_m)
    frames = np.arange(truth_m.shape[0])
    destination = output_dir / "trajectory_error_report.png"
    figure = plt.figure(figsize=(11.0, 8.0), layout="constrained")
    grid = figure.add_gridspec(2, 2)
    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    axis_error = figure.add_subplot(grid[0, 1])
    axis_xy = figure.add_subplot(grid[1, 0])
    axis_hist = figure.add_subplot(grid[1, 1])

    axis_3d.plot(*truth_m.T, label="truth", color="tab:blue")
    axis_3d.scatter(*estimate_m.T, label="estimate", color="tab:orange", s=12, alpha=0.75)
    limits = equal_axis_limits(np.vstack([truth_m, estimate_m]))
    axis_3d.set_xlim(*limits[0])
    axis_3d.set_ylim(*limits[1])
    axis_3d.set_zlim(*limits[2])
    axis_3d.set_box_aspect((1, 1, 1))
    axis_3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="Camera trajectory")
    axis_3d.legend()

    axis_error.plot(frames, errors_m, color="tab:red", label="position error")
    axis_error.axhline(float(np.sqrt(np.mean(errors_m**2))), color="black", linestyle="--", label="RMSE")
    axis_error.set(xlabel="frame index", ylabel="error (m)", title="Per-frame error")
    axis_error.legend()
    axis_error.grid(alpha=0.25)

    axis_xy.scatter(truth_m[:, 0], truth_m[:, 1], s=16, label="truth")
    axis_xy.scatter(estimate_m[:, 0], estimate_m[:, 1], s=16, label="estimate", alpha=0.7)
    axis_xy.set(xlabel="x (m)", ylabel="y (m)", title="Top view with equal aspect")
    axis_xy.set_aspect("equal", adjustable="box")
    axis_xy.legend()
    axis_xy.grid(alpha=0.25)

    axis_hist.hist(errors_m, bins=10, color="tab:green", edgecolor="black")
    axis_hist.set(xlabel="position error (m)", ylabel="frame count", title="Error histogram")
    axis_hist.grid(axis="y", alpha=0.25)

    figure.suptitle("Day 012: geometry and error audit")
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def _save_probability_figure(joint: FloatArray, output_dir: Path) -> Path:
    summary = probability_summary(joint)
    destination = output_dir / "conditional_probability_report.png"
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), layout="constrained")
    image = axes[0].imshow(joint, vmin=0.0, vmax=float(joint.max()), cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[0].text(column, row, f"{joint[row, column]:.2f}", ha="center", va="center")
    axes[0].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Y=0", "Y=1"],
        yticklabels=["X=0", "X=1"],
        title="Joint PMF P(X,Y)",
    )
    figure.colorbar(image, ax=axes[0], label="probability")

    labels = ["P(X=1)", "P(Y=1)", "P(X=1|Y=1)", "P(Y=1|X=1)"]
    values = [
        summary["p_x1"],
        summary["p_y1"],
        summary["p_x1_given_y1"],
        summary["p_y1_given_x1"],
    ]
    axes[1].bar(np.arange(len(values)), values, color=["C0", "C1", "C2", "C3"])
    axes[1].set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set(ylabel="probability", title="Marginals are not conditionals")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Day 012: conditional-probability audit")
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def run(output_dir: Path, seed: int = 20260918) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_m, estimate_m = make_camera_trajectories(seed=seed)
    aligned_errors_m = trajectory_errors_m(truth_m, estimate_m)
    shifted_estimate_m = np.roll(estimate_m, 1, axis=0)
    shifted_errors_m = trajectory_errors_m(truth_m, shifted_estimate_m)
    joint = np.array([[0.54, 0.06], [0.08, 0.32]], dtype=np.float64)
    probabilities = probability_summary(joint)
    trajectory_plot = _save_trajectory_figure(truth_m, estimate_m, output_dir)
    probability_plot = _save_probability_figure(joint, output_dir)

    report: dict[str, Any] = {
        "seed": seed,
        "frames": int(truth_m.shape[0]),
        "trajectory_contract": {
            "shape": list(truth_m.shape),
            "unit": "m",
            "coordinate_frame": "world",
            "pairing_key": "frame_index",
        },
        "aligned_rmse_m": float(np.sqrt(np.mean(aligned_errors_m**2))),
        "aligned_max_error_m": float(aligned_errors_m.max()),
        "shifted_pairing_rmse_m": float(np.sqrt(np.mean(shifted_errors_m**2))),
        "joint_probability_table": joint.tolist(),
        "probabilities": probabilities,
        "limits": [
            "plot appearance is not a substitute for numeric metrics",
            "conditional probability does not establish causality",
            "this synthetic camera path is not deployment evidence",
        ],
        "figures": [str(trajectory_plot), str(probability_plot)],
    }
    report_path = output_dir / "day012_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/Day012/reference_outputs"),
        help="directory for the JSON report and two PNG figures",
    )
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    report = run(args.output, seed=args.seed)
    p = report["probabilities"]
    print(f"trajectory shape: {tuple(report['trajectory_contract']['shape'])} m in world frame")
    print(
        "aligned RMSE={:.6f} m, max={:.6f} m; shifted-pair RMSE={:.6f} m".format(
            report["aligned_rmse_m"],
            report["aligned_max_error_m"],
            report["shifted_pairing_rmse_m"],
        )
    )
    print(
        "P(X=1)={:.4f}, P(Y=1)={:.4f}, P(X=1|Y=1)={:.4f}, P(Y=1|X=1)={:.4f}".format(
            p["p_x1"], p["p_y1"], p["p_x1_given_y1"], p["p_y1_given_x1"]
        )
    )
    print("Bayes check |direct-formula|={:.3e}".format(abs(p["p_x1_given_y1"] - p["bayes_p_x1_given_y1"])))
    for path in report["figures"]:
        print("saved:", path)
    print("saved:", report["report_path"])
    print("SUCCESS: plots, probabilities, and failure signals validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
