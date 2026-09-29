from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def entropy_bits(probabilities: np.ndarray) -> float:
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def analyze_counts(counts: np.ndarray) -> dict[str, float]:
    probabilities = counts / counts.sum()
    p_x = probabilities.sum(axis=1)
    p_y = probabilities.sum(axis=0)
    h_x = entropy_bits(p_x)
    h_y = entropy_bits(p_y)
    h_joint = entropy_bits(probabilities.ravel())
    h_x_given_y = h_joint - h_y
    mi_conditional = h_x - h_x_given_y
    mi_entropies = h_x + h_y - h_joint
    product = p_x[:, None] * p_y[None, :]
    mask = probabilities > 0
    mi_kl = float(
        (probabilities[mask] * np.log2(probabilities[mask] / product[mask])).sum()
    )
    return {
        "h_x_bits": h_x,
        "h_y_bits": h_y,
        "h_joint_bits": h_joint,
        "h_x_given_y_bits": h_x_given_y,
        "mi_conditional_bits": mi_conditional,
        "mi_entropies_bits": mi_entropies,
        "mi_kl_bits": mi_kl,
        "match_rule_accuracy": float(np.trace(counts) / counts.sum()),
        "majority_y_accuracy": float(p_y.max()),
    }


def build_report() -> dict[str, object]:
    x = np.arange(5, dtype=np.float64)
    observations_m = np.array([1.1, 2.9, 5.2, 6.8, 9.1], dtype=np.float64)
    design = np.column_stack([x, np.ones_like(x)])
    beta, _, rank, singular_values = np.linalg.lstsq(
        design, observations_m, rcond=None
    )
    residual_m = design @ beta - observations_m
    paired = analyze_counts(np.array([[3.0, 1.0], [1.0, 3.0]]))
    shifted = analyze_counts(np.full((2, 2), 2.0))
    return {
        "numpy_version": np.__version__,
        "design_shape": list(design.shape),
        "observation_shape": list(observations_m.shape),
        "rank": int(rank),
        "slope": float(beta[0]),
        "intercept_m": float(beta[1]),
        "rmse_m": float(np.sqrt(np.mean(residual_m**2))),
        "singular_values": [float(value) for value in singular_values],
        "condition_number": float(singular_values[0] / singular_values[1]),
        "paired": paired,
        "shifted": shifted,
    }


def self_test(report: dict[str, object]) -> None:
    assert report["design_shape"] == [5, 2]
    assert report["rank"] == 2
    assert math.isclose(report["slope"], 1.99, abs_tol=1e-12)
    assert math.isclose(report["intercept_m"], 1.04, abs_tol=1e-12)
    assert math.isclose(report["rmse_m"], 0.1462873883832781, abs_tol=1e-12)
    paired = report["paired"]
    assert math.isclose(paired["mi_conditional_bits"], 0.18872187554086717)
    assert math.isclose(paired["mi_entropies_bits"], paired["mi_kl_bits"])
    assert math.isclose(paired["match_rule_accuracy"], 0.75)
    shifted = report["shifted"]
    assert math.isclose(shifted["mi_kl_bits"], 0.0, abs_tol=1e-12)
    assert math.isclose(shifted["match_rule_accuracy"], 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    report = build_report()
    self_test(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if args.self_test:
        print("NUMPY_SELF_TEST_OK: 10 checks")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
