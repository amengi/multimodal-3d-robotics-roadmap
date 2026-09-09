"""A shape-audited two-layer MLP and one deterministic optimization step."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FeatureError(ValueError):
    """Raised when a feature batch violates the ``(B, 2)`` contract."""


def validate_features(features: Tensor) -> Tensor:
    """Require a finite floating-point batch of 2D points."""
    if not isinstance(features, Tensor):
        raise TypeError("features must be a torch.Tensor")
    if features.ndim != 2 or features.shape[1] != 2:
        raise FeatureError(
            f"features must have shape (B, 2), received {tuple(features.shape)}"
        )
    if features.shape[0] == 0:
        raise FeatureError("features must contain at least one sample")
    if not features.is_floating_point():
        raise FeatureError("features must use a floating-point dtype")
    if not torch.isfinite(features).all():
        raise FeatureError("features must contain only finite values")
    return features


class TwoLayerMLP(nn.Module):
    """Map one 2D point to one binary-classification logit."""

    def __init__(self, hidden_features: int = 8) -> None:
        super().__init__()
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        self.linear1 = nn.Linear(2, hidden_features)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, 1)

    def forward(self, features: Tensor) -> Tensor:
        checked = validate_features(features)
        hidden = self.activation(self.linear1(checked))
        return self.linear2(hidden)

    def forward_with_shapes(self, features: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        """Return logits and the observable shape at each boundary."""
        checked = validate_features(features)
        hidden_linear = self.linear1(checked)
        hidden_activated = self.activation(hidden_linear)
        logits = self.linear2(hidden_activated)
        return logits, {
            "input": tuple(checked.shape),
            "hidden_linear": tuple(hidden_linear.shape),
            "hidden_activated": tuple(hidden_activated.shape),
            "logits": tuple(logits.shape),
        }


def make_toy_batch() -> tuple[Tensor, Tensor]:
    """Return six 2D points and labels defined by ``x^2 + y^2 > 1``."""
    features = torch.tensor(
        [
            [-0.2, 0.1],
            [0.3, 0.4],
            [1.2, 0.2],
            [-1.1, -0.8],
            [0.1, 1.3],
            [0.6, -0.2],
        ],
        dtype=torch.float32,
    )
    labels = ((features[:, 0] ** 2 + features[:, 1] ** 2) > 1.0).to(
        dtype=torch.float32
    )
    return features, labels.unsqueeze(1)


def single_optimization_step(seed: int = 20260909) -> dict[str, object]:
    """Run one SGD update and return auditable values, shapes, and gradients."""
    torch.manual_seed(seed)
    features, labels = make_toy_batch()
    model = TwoLayerMLP(hidden_features=8)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    logits_before, shapes = model.forward_with_shapes(features)
    loss_before = loss_function(logits_before, labels)

    optimizer.zero_grad(set_to_none=True)
    loss_before.backward()
    gradient_norm = torch.sqrt(
        sum(parameter.grad.square().sum() for parameter in model.parameters() if parameter.grad is not None)
    )
    optimizer.step()

    with torch.no_grad():
        logits_after = model(features)
        probabilities = torch.sigmoid(logits_after)
        loss_after = loss_function(logits_after, labels)

    return {
        "features": features,
        "labels": labels,
        "shapes": shapes,
        "logits_before": logits_before.detach(),
        "logits_after": logits_after,
        "probabilities": probabilities,
        "loss_before": float(loss_before.detach()),
        "loss_after": float(loss_after),
        "gradient_norm": float(gradient_norm.detach()),
    }


def main() -> int:
    """Display every interface shape and one checked optimizer update."""
    result = single_optimization_step()
    print("task: synthetic 2D-point binary classification")
    for name, shape in result["shapes"].items():
        print(f"{name} shape: {shape}")
    print("labels shape:", tuple(result["labels"].shape))
    print("first probability:", round(float(result["probabilities"][0]), 6))
    print("loss before:", round(result["loss_before"], 6))
    print("loss after one SGD step:", round(result["loss_after"], 6))
    print("global gradient norm:", round(result["gradient_norm"], 6))
    print("MLP shape checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
