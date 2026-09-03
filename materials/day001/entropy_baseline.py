"""Day 001: a dependency-free Shannon entropy baseline."""

from __future__ import annotations

import math
from collections.abc import Sequence


def shannon_entropy(
    probabilities: Sequence[float],
    *,
    base: float = 2.0,
    tolerance: float = 1e-9,
) -> float:
    """Return the Shannon entropy of a discrete probability distribution.

    Args:
        probabilities: Non-empty probabilities that sum to one.
        base: Logarithm base. Base 2 returns bits; math.e returns nats.
        tolerance: Absolute tolerance used when checking the probability sum.

    Raises:
        ValueError: If the distribution or logarithm base is invalid.
    """
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if base <= 0.0 or math.isclose(base, 1.0):
        raise ValueError("base must be positive and different from 1")
    if any(not math.isfinite(p) for p in probabilities):
        raise ValueError("probabilities must be finite")
    if any(p < 0.0 for p in probabilities):
        raise ValueError("probabilities must be non-negative")

    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"probabilities must sum to 1; got {total:.12g}")

    entropy = -math.fsum(
        p * math.log(p, base) for p in probabilities if p > 0.0
    )
    return 0.0 if math.isclose(entropy, 0.0, abs_tol=tolerance) else entropy


def main() -> None:
    """Run three known-answer baseline checks."""
    distributions = [
        [0.5, 0.5],
        [1.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ]
    expected = [1.0, 0.0, 2.0]

    for probabilities, expected_entropy in zip(distributions, expected):
        actual = shannon_entropy(probabilities)
        print(f"P={probabilities} -> H={actual:.6f} bits")
        assert math.isclose(actual, expected_entropy, abs_tol=1e-12)

    print("All baseline checks passed.")


if __name__ == "__main__":
    main()
