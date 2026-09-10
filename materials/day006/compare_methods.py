"""Compare a nearest-centroid geometric baseline with the Day 006 MLP."""

from __future__ import annotations

import torch
from torch import Tensor

from materials.day006.train_moons import make_moon_split, train_model, validate_batch


def fit_class_centroids(features: Tensor, labels: Tensor) -> Tensor:
    """Return one 2D centroid for each binary class, fitted on training data."""
    validate_batch(features, labels)
    centroids = []
    flat_labels = labels.squeeze(1)
    for class_id in (0.0, 1.0):
        selected = features[flat_labels == class_id]
        if selected.shape[0] == 0:
            raise ValueError(f"class {int(class_id)} has no training samples")
        centroids.append(selected.mean(dim=0))
    return torch.stack(centroids)


def nearest_centroid_predict(features: Tensor, centroids: Tensor) -> Tensor:
    """Assign each point to the class with the nearest squared Euclidean distance."""
    validate_batch(features)
    if centroids.shape != (2, 2) or not torch.isfinite(centroids).all():
        raise ValueError("centroids must be finite with shape (2, 2)")
    squared_distances = (features[:, None, :] - centroids[None, :, :]).square().sum(dim=2)
    return squared_distances.argmin(dim=1, keepdim=True).to(torch.float32)


def accuracy(predictions: Tensor, labels: Tensor) -> float:
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    return float((predictions == labels).float().mean())


def compare(seed: int = 20260910) -> dict[str, float | int]:
    """Use the same split and metric for both methods."""
    split = make_moon_split(seed=seed)
    centroids = fit_class_centroids(split.train_features, split.train_labels)
    centroid_predictions = nearest_centroid_predict(split.validation_features, centroids)
    centroid_accuracy = accuracy(centroid_predictions, split.validation_labels)

    trained = train_model(split, seed=seed)
    return {
        "validation_samples": split.validation_features.shape[0],
        "centroid_accuracy": centroid_accuracy,
        "mlp_accuracy": trained.validation_accuracy,
    }


def main() -> int:
    result = compare()
    print("same held-out validation set; metric: accuracy")
    print("validation samples:", result["validation_samples"])
    print("nearest-centroid accuracy: {:.3f}".format(result["centroid_accuracy"]))
    print("MLP accuracy: {:.3f}".format(result["mlp_accuracy"]))
    print("This one synthetic split is a method check, not a universal ranking.")
    print("Method-comparison checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
