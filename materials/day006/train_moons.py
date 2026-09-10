"""Deterministic full-batch training on a synthetic two-moons data set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor, nn


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class TrainingDataError(ValueError):
    """Raised when data or training configuration violates its contract."""


@dataclass(frozen=True)
class MoonSplit:
    train_features: Tensor
    train_labels: Tensor
    validation_features: Tensor
    validation_labels: Tensor
    train_indices: IntArray
    validation_indices: IntArray
    train_mean: FloatArray
    train_std: FloatArray


@dataclass
class TrainingResult:
    model: "MoonMLP"
    train_losses: list[float]
    validation_losses: list[float]
    train_accuracy: float
    validation_accuracy: float


def set_reproducible_seed(seed: int) -> None:
    """Seed the random-number generators used by this CPU-only example."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_moon_split(
    n_samples: int = 400,
    noise: float = 0.20,
    validation_fraction: float = 0.25,
    seed: int = 20260910,
) -> MoonSplit:
    """Generate, stratify, split, and standardize using training statistics only."""
    if n_samples < 40:
        raise TrainingDataError("n_samples must be at least 40")
    if not 0.0 <= noise <= 1.0:
        raise TrainingDataError("noise must be between 0 and 1")
    if not 0.1 <= validation_fraction <= 0.5:
        raise TrainingDataError("validation_fraction must be between 0.1 and 0.5")

    features, labels = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=seed,
        shuffle=True,
    )
    indices = np.arange(n_samples, dtype=np.int64)
    train_idx, validation_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    train_features = features[train_idx]
    validation_features = features[validation_idx]
    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0)
    if not np.isfinite(train_std).all() or np.any(train_std == 0.0):
        raise TrainingDataError("training features contain a degenerate column")

    train_features = (train_features - train_mean) / train_std
    validation_features = (validation_features - train_mean) / train_std

    def to_feature_tensor(values: FloatArray) -> Tensor:
        return torch.as_tensor(values, dtype=torch.float32)

    def to_label_tensor(values: IntArray) -> Tensor:
        return torch.as_tensor(values, dtype=torch.float32).unsqueeze(1)

    return MoonSplit(
        train_features=to_feature_tensor(train_features),
        train_labels=to_label_tensor(labels[train_idx]),
        validation_features=to_feature_tensor(validation_features),
        validation_labels=to_label_tensor(labels[validation_idx]),
        train_indices=np.asarray(train_idx, dtype=np.int64),
        validation_indices=np.asarray(validation_idx, dtype=np.int64),
        train_mean=np.asarray(train_mean, dtype=np.float64),
        train_std=np.asarray(train_std, dtype=np.float64),
    )


def validate_batch(features: Tensor, labels: Tensor | None = None) -> None:
    if not isinstance(features, Tensor):
        raise TypeError("features must be a torch.Tensor")
    if features.ndim != 2 or features.shape[1] != 2 or features.shape[0] == 0:
        raise TrainingDataError(
            f"features must have non-empty shape (B, 2), received {tuple(features.shape)}"
        )
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise TrainingDataError("features must be finite floating-point values")
    if labels is not None:
        if labels.shape != (features.shape[0], 1):
            raise TrainingDataError(
                f"labels must have shape (B, 1), received {tuple(labels.shape)}"
            )
        if not labels.is_floating_point() or not torch.isfinite(labels).all():
            raise TrainingDataError("labels must be finite floating-point values")
        if not torch.all((labels == 0.0) | (labels == 1.0)):
            raise TrainingDataError("labels must contain only 0 and 1")


class MoonMLP(nn.Module):
    """A small 2 -> hidden -> hidden -> 1 binary classifier."""

    def __init__(self, hidden_features: int = 16) -> None:
        super().__init__()
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        self.network = nn.Sequential(
            nn.Linear(2, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        validate_batch(features)
        return self.network(features)


def binary_accuracy(logits: Tensor, labels: Tensor) -> float:
    if logits.shape != labels.shape:
        raise TrainingDataError("logits and labels must have the same shape")
    if labels.ndim != 2 or labels.shape[1] != 1 or labels.shape[0] == 0:
        raise TrainingDataError("labels must have non-empty shape (B, 1)")
    if not logits.is_floating_point() or not labels.is_floating_point():
        raise TrainingDataError("logits and labels must be floating-point tensors")
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
        raise TrainingDataError("logits and labels must contain only finite values")
    if not torch.all((labels == 0.0) | (labels == 1.0)):
        raise TrainingDataError("labels must contain only 0 and 1")
    predictions = (logits >= 0.0).to(labels.dtype)
    return float((predictions == labels).float().mean())


def evaluate(model: MoonMLP, features: Tensor, labels: Tensor) -> tuple[float, float]:
    """Return BCE loss and accuracy without constructing a gradient graph."""
    validate_batch(features, labels)
    model.eval()
    loss_function = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(features)
        loss = loss_function(logits, labels)
    return float(loss), binary_accuracy(logits, labels)


def train_model(
    split: MoonSplit,
    epochs: int = 300,
    learning_rate: float = 0.02,
    seed: int = 20260910,
) -> TrainingResult:
    """Train with full batches and evaluate on a held-out validation set."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    validate_batch(split.train_features, split.train_labels)
    validate_batch(split.validation_features, split.validation_labels)
    set_reproducible_seed(seed)

    model = MoonMLP(hidden_features=16)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        logits = model(split.train_features)
        loss = loss_function(logits, split.train_labels)
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss, _ = evaluate(model, split.train_features, split.train_labels)
        validation_loss, _ = evaluate(
            model, split.validation_features, split.validation_labels
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

    train_loss, train_accuracy = evaluate(
        model, split.train_features, split.train_labels
    )
    validation_loss, validation_accuracy = evaluate(
        model, split.validation_features, split.validation_labels
    )
    return TrainingResult(
        model=model,
        train_losses=train_losses,
        validation_losses=validation_losses,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
    )


def decision_grid(
    model: MoonMLP, features: Tensor, step: float = 0.04
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate probabilities on a regular 2D grid for visualization."""
    validate_batch(features)
    if not 0.0 < step <= 0.5:
        raise ValueError("step must be in (0, 0.5]")
    values = features.detach().cpu().numpy()
    x_values = np.arange(values[:, 0].min() - 0.5, values[:, 0].max() + 0.5, step)
    y_values = np.arange(values[:, 1].min() - 0.5, values[:, 1].max() + 0.5, step)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid = torch.as_tensor(
        np.column_stack([grid_x.ravel(), grid_y.ravel()]), dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(grid)).reshape(grid_x.shape)
    return grid_x, grid_y, probabilities.cpu().numpy().astype(np.float64)


def save_plots(split: MoonSplit, result: TrainingResult, output_dir: Path) -> list[Path]:
    """Save a loss curve and a decision-boundary figure with explicit labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_path = output_dir / "loss_curve.png"
    boundary_path = output_dir / "decision_boundary.png"

    epochs = np.arange(1, len(result.train_losses) + 1)
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(epochs, result.train_losses, label="training BCE")
    axis.plot(epochs, result.validation_losses, label="validation BCE")
    axis.set(xlabel="epoch", ylabel="binary cross-entropy", title="Two-moons loss")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)

    all_features = torch.cat(
        [split.train_features, split.validation_features], dim=0
    )
    grid_x, grid_y, probabilities = decision_grid(result.model, all_features)
    validation = split.validation_features.numpy()
    labels = split.validation_labels.squeeze(1).numpy()
    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    filled = axis.contourf(
        grid_x, grid_y, probabilities, levels=np.linspace(0.0, 1.0, 11), cmap="RdBu", alpha=0.75
    )
    axis.scatter(
        validation[:, 0], validation[:, 1], c=labels, cmap="bwr", edgecolor="black", s=28
    )
    axis.contour(grid_x, grid_y, probabilities, levels=[0.5], colors="black", linewidths=1.2)
    axis.set(
        xlabel="standardized feature x1",
        ylabel="standardized feature x2",
        title="Validation points and MLP decision boundary",
    )
    fig.colorbar(filled, ax=axis, label="model sigmoid output")
    fig.savefig(boundary_path, dpi=160)
    plt.close(fig)
    return [loss_path, boundary_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a deterministic two-moons MLP.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/Day006/reference_outputs")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    split = make_moon_split(seed=args.seed)
    result = train_model(
        split, epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed
    )
    paths = save_plots(split, result, args.output_dir)
    print("task: synthetic two-moons binary classification")
    print("train features/labels:", tuple(split.train_features.shape), tuple(split.train_labels.shape))
    print(
        "validation features/labels:",
        tuple(split.validation_features.shape),
        tuple(split.validation_labels.shape),
    )
    print("seed / epochs / learning_rate:", args.seed, args.epochs, args.learning_rate)
    print("first/final training BCE: {:.6f} / {:.6f}".format(result.train_losses[0], result.train_losses[-1]))
    print("final validation BCE: {:.6f}".format(result.validation_losses[-1]))
    print("train/validation accuracy: {:.3f} / {:.3f}".format(result.train_accuracy, result.validation_accuracy))
    for path in paths:
        print("saved:", path)
    print("Training-loop checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
