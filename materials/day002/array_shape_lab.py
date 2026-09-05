"""Day 002: inspect axes, reductions, broadcasting, and matrix multiplication."""

from __future__ import annotations

import numpy as np


SENSOR_NAMES = ("rgb", "depth", "imu")


def make_sensor_batch(seed: int = 2) -> np.ndarray:
    """Return a reproducible float array with shape (batch=2, modality=3, feature=4)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10, size=(2, 3, 4)).astype(np.float64)


def axis_summaries(batch: np.ndarray) -> dict[str, np.ndarray]:
    """Compute three reductions while preserving their semantic axis names."""
    if batch.ndim != 3:
        raise ValueError("batch must have shape (B, M, F)")
    return {
        "per_sample_feature_mean": batch.mean(axis=2),  # (B, M)
        "per_sample_modality_mean": batch.mean(axis=1),  # (B, F)
        "dataset_mean": batch.mean(axis=(0, 1)),  # (F,)
    }


def center_features(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the feature mean using broadcasting; return centered data and mean."""
    if batch.ndim != 3:
        raise ValueError("batch must have shape (B, M, F)")
    feature_mean = batch.mean(axis=(0, 1), keepdims=True)  # (1, 1, F)
    return batch - feature_mean, feature_mean


def project_features(batch: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Map the last feature axis F to output axis O with matrix multiplication."""
    if batch.ndim != 3 or weights.ndim != 2:
        raise ValueError("expected batch (B, M, F) and weights (F, O)")
    if batch.shape[-1] != weights.shape[0]:
        raise ValueError("batch feature size must equal weights first dimension")
    return batch @ weights


def main() -> None:
    batch = make_sensor_batch()
    summaries = axis_summaries(batch)
    centered, feature_mean = center_features(batch)
    weights = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.5, 0.5]],
        dtype=np.float64,
    )
    projected = project_features(batch, weights)

    print(f"sensor order: {SENSOR_NAMES}")
    print(f"batch shape: {batch.shape}  # (B, M, F)")
    for name, value in summaries.items():
        print(f"{name}: {value.shape}")
    print(f"feature mean shape: {feature_mean.shape}")
    print(f"centered shape: {centered.shape}")
    print(f"max absolute centered feature mean: {abs(centered.mean(axis=(0, 1))).max():.3e}")
    print(f"weights shape: {weights.shape}")
    print(f"projected shape: {projected.shape}")
    print("All shape checks passed.")


if __name__ == "__main__":
    main()
