"""Deterministic Week 01 assessment with paired synthetic sensor modalities.

The example is deliberately small and CPU-only.  It compares a majority
baseline, a stable single-modality linear model, and a two-modality MLP on the
same held-out validation split.  A shuffled-modality evaluation supplies a
controlled failure case; it is not an estimate of mutual information.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn


class ContractError(ValueError):
    """Raised when an input violates the documented shape/value contract."""


@dataclass(frozen=True)
class PairedSplit:
    train_x: Tensor
    train_y: Tensor
    validation_x: Tensor
    validation_y: Tensor
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    nll_nats: float
    brier: float


@dataclass
class TrainedModel:
    model: nn.Module
    losses: list[float]
    metrics: Metrics


def set_seed(seed: int) -> None:
    """Seed all random sources used by this CPU example."""
    # Tiny full-batch models are faster and more repeatable with one CPU thread.
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stratified_indices(labels: np.ndarray, validation_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    validation_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for class_id in (0, 1):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        count = int(round(len(indices) * validation_fraction))
        validation_parts.append(indices[:count])
        train_parts.append(indices[count:])
    train = np.concatenate(train_parts)
    validation = np.concatenate(validation_parts)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train.astype(np.int64), validation.astype(np.int64)


def make_paired_split(
    n_samples: int = 480,
    validation_fraction: float = 0.25,
    seed: int = 20260911,
) -> PairedSplit:
    """Return paired sensor features X=(B,2,2) and binary labels Y=(B,1)."""
    if n_samples < 80:
        raise ContractError("n_samples must be at least 80")
    if not 0.1 <= validation_fraction <= 0.5:
        raise ContractError("validation_fraction must be between 0.1 and 0.5")

    rng = np.random.default_rng(seed)
    latent_a = rng.normal(size=n_samples)
    latent_b = rng.normal(size=n_samples)
    label_noise = 0.10 * rng.normal(size=n_samples)
    labels = (latent_a + latent_b + label_noise > 0.0).astype(np.float32)

    sensor_a = np.column_stack(
        [
            latent_a + 0.35 * rng.normal(size=n_samples),
            0.70 * latent_a + 0.45 * rng.normal(size=n_samples),
        ]
    )
    sensor_b = np.column_stack(
        [
            latent_b + 0.35 * rng.normal(size=n_samples),
            0.70 * latent_b + 0.45 * rng.normal(size=n_samples),
        ]
    )
    features = np.stack([sensor_a, sensor_b], axis=1).astype(np.float32)
    train_idx, validation_idx = _stratified_indices(
        labels.astype(np.int64), validation_fraction, seed
    )

    # Fit each modality/feature normalization parameter on training samples only.
    train_mean = features[train_idx].mean(axis=0, keepdims=True)
    train_std = features[train_idx].std(axis=0, keepdims=True)
    if not np.isfinite(train_std).all() or np.any(train_std <= 0.0):
        raise ContractError("training data contain a non-finite or constant feature")
    normalized = (features - train_mean) / train_std

    return PairedSplit(
        train_x=torch.as_tensor(normalized[train_idx], dtype=torch.float32),
        train_y=torch.as_tensor(labels[train_idx, None], dtype=torch.float32),
        validation_x=torch.as_tensor(normalized[validation_idx], dtype=torch.float32),
        validation_y=torch.as_tensor(labels[validation_idx, None], dtype=torch.float32),
        train_indices=train_idx,
        validation_indices=validation_idx,
    )


def validate_xy(features: Tensor, labels: Tensor | None = None) -> None:
    """Enforce X=(B,2,2), Y=(B,1), finite floating-point values."""
    if not isinstance(features, Tensor):
        raise TypeError("features must be a torch.Tensor")
    if features.ndim != 3 or tuple(features.shape[1:]) != (2, 2) or len(features) == 0:
        raise ContractError(
            f"features must have non-empty shape (B, 2, 2), got {tuple(features.shape)}"
        )
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ContractError("features must be finite floating-point values")
    if labels is not None:
        if tuple(labels.shape) != (len(features), 1):
            raise ContractError(f"labels must have shape (B, 1), got {tuple(labels.shape)}")
        if not labels.is_floating_point() or not torch.isfinite(labels).all():
            raise ContractError("labels must be finite floating-point values")
        if not torch.all((labels == 0.0) | (labels == 1.0)):
            raise ContractError("labels must contain only 0 and 1")


class SingleSensorLinear(nn.Module):
    """Ordinary supervised baseline that sees only stable sensor A."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, paired_features: Tensor) -> Tensor:
        validate_xy(paired_features)
        return self.linear(paired_features[:, 0, :])


class FusionMLP(nn.Module):
    """Small early-fusion classifier: (B,2,2) -> (B,4) -> (B,12) -> (B,1)."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Linear(12, 1))

    def forward(self, paired_features: Tensor) -> Tensor:
        validate_xy(paired_features)
        flattened = paired_features.reshape(len(paired_features), 4)
        return self.network(flattened)


def metrics_from_logits(logits: Tensor, labels: Tensor) -> Metrics:
    """Compute discrimination and probabilistic scores on the same examples."""
    if tuple(logits.shape) != tuple(labels.shape) or logits.ndim != 2:
        raise ContractError("logits and labels must share shape (B, 1)")
    if not torch.isfinite(logits).all():
        raise ContractError("logits must be finite")
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).to(labels.dtype)
    accuracy = (predictions == labels).float().mean()
    nll = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    brier = (probabilities - labels).square().mean()
    return Metrics(float(accuracy), float(nll), float(brier))


def majority_metrics(train_labels: Tensor, validation_labels: Tensor) -> Metrics:
    """Fit a constant class probability on train only and score validation."""
    probability = train_labels.mean().clamp(1e-6, 1.0 - 1e-6)
    logit = torch.logit(probability).expand_as(validation_labels)
    return metrics_from_logits(logit, validation_labels)


def train(
    model: nn.Module,
    split: PairedSplit,
    *,
    epochs: int = 250,
    learning_rate: float = 0.03,
    seed: int = 20260911,
) -> TrainedModel:
    """Fit a full-batch classifier and evaluate only after each parameter update."""
    if epochs <= 0:
        raise ContractError("epochs must be positive")
    if not 0.0 < learning_rate <= 1.0:
        raise ContractError("learning_rate must be in (0, 1]")
    validate_xy(split.train_x, split.train_y)
    validate_xy(split.validation_x, split.validation_y)
    set_seed(seed)
    model.apply(lambda module: module.reset_parameters() if hasattr(module, "reset_parameters") else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    losses: list[float] = []

    for _ in range(epochs):
        model.train()
        logits = model(split.train_x)
        loss = loss_function(logits, split.train_y)
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        validation_logits = model(split.validation_x)
    return TrainedModel(model, losses, metrics_from_logits(validation_logits, split.validation_y))


def shuffled_sensor_b(features: Tensor, seed: int = 20260911) -> Tensor:
    """Break A/B pairing in a copy while preserving both marginal value sets."""
    validate_xy(features)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(features), generator=generator)
    shuffled = features.clone()
    shuffled[:, 1, :] = features[permutation, 1, :]
    return shuffled


def evaluate_on(model: nn.Module, features: Tensor, labels: Tensor) -> Metrics:
    validate_xy(features, labels)
    model.eval()
    with torch.no_grad():
        return metrics_from_logits(model(features), labels)


def save_loss_plot(
    single: TrainedModel, fusion: TrainedModel, output_directory: Path
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / "loss_curve.png"
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    axis.plot(single.losses, label="single sensor A: train BCE")
    axis.plot(fusion.losses, label="paired A+B: train BCE")
    axis.set(xlabel="epoch", ylabel="BCE (nats/example)", title="Week 01 closed-loop check")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def run_assessment(output_directory: Path, seed: int = 20260911) -> dict[str, object]:
    split = make_paired_split(seed=seed)
    majority = majority_metrics(split.train_y, split.validation_y)
    single = train(SingleSensorLinear(), split, seed=seed)
    fusion = train(FusionMLP(), split, seed=seed)
    broken_pairing = evaluate_on(
        fusion.model, shuffled_sensor_b(split.validation_x, seed + 1), split.validation_y
    )
    plot_path = save_loss_plot(single, fusion, output_directory)
    report = {
        "seed": seed,
        "train_shape": list(split.train_x.shape),
        "validation_shape": list(split.validation_x.shape),
        "train_validation_overlap": int(
            len(set(split.train_indices.tolist()) & set(split.validation_indices.tolist()))
        ),
        "majority": asdict(majority),
        "single_sensor_a": asdict(single.metrics),
        "paired_fusion": asdict(fusion.metrics),
        "shuffled_sensor_b": asdict(broken_pairing),
        "loss_plot": str(plot_path),
        "limits": [
            "synthetic data are not deployment evidence",
            "one split and one seed do not establish general superiority",
            "NLL and Brier alone do not prove calibration",
            "a shuffle ablation does not estimate mutual information or causality",
        ],
    }
    report_path = output_directory / "assessment_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/Day007/reference_outputs"),
        help="directory for the JSON report and loss plot",
    )
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    report = run_assessment(args.output, args.seed)
    print("train/validation X shapes:", tuple(report["train_shape"]), tuple(report["validation_shape"]))
    print("train/validation index overlap:", report["train_validation_overlap"])
    for name in ("majority", "single_sensor_a", "paired_fusion", "shuffled_sensor_b"):
        values = report[name]
        print(
            f"{name:20s} accuracy={values['accuracy']:.3f} "
            f"NLL={values['nll_nats']:.3f} nats/example Brier={values['brier']:.3f}"
        )
    print("saved:", report["loss_plot"])
    print("saved:", report["report_path"])
    print("All Day 007 assessment checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
