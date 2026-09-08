"""Small, deterministic PyTorch automatic-differentiation experiments."""

from __future__ import annotations

import torch


def polynomial_value_and_gradient(x_value: float) -> tuple[float, float]:
    """Evaluate y=x^2+3x and dy/dx at one dimensionless scalar."""
    x = torch.tensor(float(x_value), dtype=torch.float64, requires_grad=True)
    y = x**2 + 3 * x
    y.backward()
    if x.grad is None:
        raise RuntimeError("PyTorch did not populate x.grad")
    return float(y.item()), float(x.grad.item())


def finite_difference_gradient(x_value: float, step: float = 1e-6) -> float:
    """Approximate the same derivative with a centered finite difference."""
    if step <= 0:
        raise ValueError("step must be positive")
    f_plus = (x_value + step) ** 2 + 3 * (x_value + step)
    f_minus = (x_value - step) ** 2 + 3 * (x_value - step)
    return (f_plus - f_minus) / (2 * step)


def multimodal_fusion_step() -> dict[str, torch.Tensor]:
    """Differentiate mean squared error through a two-modality linear fusion.

    ``features`` has shape (B=3, M=2).  Column 0 and column 1 are standardized
    scalar features from two modalities, so both are dimensionless.
    """
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    targets = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
    predictions = features @ weights
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()
    if weights.grad is None:
        raise RuntimeError("PyTorch did not populate weights.grad")
    return {
        "features": features,
        "targets": targets,
        "weights": weights.detach(),
        "predictions": predictions.detach(),
        "loss": loss.detach(),
        "weight_gradients": weights.grad.detach(),
    }


def main() -> None:
    """Print expected values and fail fast if either experiment is wrong."""
    value, automatic = polynomial_value_and_gradient(2.0)
    numerical = finite_difference_gradient(2.0)
    fusion = multimodal_fusion_step()
    print(f"polynomial: x=2.0, y={value:.6f}, autograd={automatic:.6f}")
    print(f"finite difference gradient={numerical:.6f}")
    print("features shape:", tuple(fusion["features"].shape), "=(B, M)")
    print("predictions:", fusion["predictions"].tolist())
    print(f"MSE loss={fusion['loss'].item():.6f}")
    print("weight gradients:", fusion["weight_gradients"].tolist())
    assert value == 10.0
    assert automatic == 7.0
    assert abs(numerical - automatic) < 1e-6
    print("All autograd checks passed.")


if __name__ == "__main__":
    main()
