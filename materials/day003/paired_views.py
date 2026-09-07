"""Visualize paired variables before and after sample correspondence is broken."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_pairs(seed: int = 3, n: int = 1000, rho: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Generate unitless Gaussian variables with target Pearson correlation rho."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = rho * x + np.sqrt(max(0.0, 1.0 - rho * rho)) * noise
    return x, y


def permute_y(y: np.ndarray, seed: int = 4) -> np.ndarray:
    """Return the same y values in a reproducibly shuffled order."""
    return np.random.default_rng(seed).permutation(y)


def plot_views(x: np.ndarray, y: np.ndarray, output: Path, title_prefix: str) -> None:
    """Save scatter, marginal histogram, and joint histogram views."""
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays with equal shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].scatter(x, y, s=8, alpha=0.3)
    axes[0].set(xlabel="X (unitless)", ylabel="Y (unitless)", title=f"{title_prefix}: pairs")
    axes[1].hist(x, bins=30, alpha=0.5, label="X")
    axes[1].hist(y, bins=30, alpha=0.5, label="Y")
    axes[1].set(xlabel="Value (unitless)", ylabel="Count", title=f"{title_prefix}: marginals")
    axes[1].legend()
    histogram = axes[2].hist2d(x, y, bins=30)
    axes[2].set(xlabel="X (unitless)", ylabel="Y (unitless)", title=f"{title_prefix}: joint")
    fig.colorbar(histogram[3], ax=axes[2], label="Count")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/day003"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--rho", type=float, default=0.8)
    args = parser.parse_args()

    x, y = make_pairs(args.seed, args.n, args.rho)
    shuffled_y = permute_y(y, args.seed + 1)
    plot_views(x, y, args.out_dir / "paired_original.png", "Original")
    plot_views(x, shuffled_y, args.out_dir / "paired_shuffled.png", "Shuffled")

    original_r = float(np.corrcoef(x, y)[0, 1])
    shuffled_r = float(np.corrcoef(x, shuffled_y)[0, 1])
    marginal_difference = float(np.max(np.abs(np.sort(y) - np.sort(shuffled_y))))
    print(f"seed={args.seed}, n={args.n}, rho_target={args.rho}")
    print(f"original sample correlation: {original_r:.6f}")
    print(f"shuffled sample correlation: {shuffled_r:.6f}")
    print(f"sorted-y maximum difference: {marginal_difference:.1f}")
    print(f"saved: {args.out_dir / 'paired_original.png'}")
    print(f"saved: {args.out_dir / 'paired_shuffled.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
