"""Verify three-class cross-entropy/NLL from logits with PyTorch."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


def validate_logits(logits: torch.Tensor, target: int) -> None:
    if logits.ndim != 1 or logits.numel() < 2:
        raise ValueError("logits must have shape (C,) with C >= 2")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if target < 0 or target >= logits.numel():
        raise ValueError("target index is outside [0, C)")


def manual_cross_entropy_nats(logits: torch.Tensor, target: int) -> float:
    """Return -log softmax(logits)[target] using stable log-sum-exp."""
    validate_logits(logits, target)
    shifted = logits - torch.max(logits)
    log_sum_exp = torch.max(logits) + torch.log(torch.exp(shifted).sum())
    return float(log_sum_exp - logits[target])


def multiclass_brier(probabilities: torch.Tensor, target: int) -> float:
    """Return sum_c (p_c - 1[c=target])^2 for one example."""
    validate_logits(probabilities, target)
    if bool((probabilities < 0.0).any()) or not math.isclose(
        float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("probabilities must be nonnegative and sum to one")
    one_hot = torch.zeros_like(probabilities)
    one_hot[target] = 1.0
    return float(torch.square(probabilities - one_hot).sum())


def loss_scan(true_class_logits: Iterable[float]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for value in true_class_logits:
        logits = torch.tensor([value, 0.0, 0.0], dtype=torch.float64)
        probabilities = torch.softmax(logits, dim=0)
        rows.append(
            {
                "true_logit": float(value),
                "p_true": float(probabilities[0]),
                "nll_nats": manual_cross_entropy_nats(logits, 0),
            }
        )
    return rows


def write_scan(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["true_logit", "p_true", "nll_nats"])
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    checks = 0
    logits = torch.tensor([2.0, 1.0, 0.1], dtype=torch.float64)
    probabilities = torch.softmax(logits, dim=0)
    manual = manual_cross_entropy_nats(logits, 0)
    torch_loss = float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([0])))
    one_hot_loss = float(-(torch.tensor([1.0, 0.0, 0.0]) * torch.log(probabilities)).sum())

    assert math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12)
    checks += 1
    assert math.isclose(manual, -math.log(float(probabilities[0])), abs_tol=1e-12)
    checks += 1
    assert math.isclose(manual, torch_loss, abs_tol=1e-12)
    checks += 1
    assert math.isclose(manual, one_hot_loss, abs_tol=1e-12)
    checks += 1
    assert int(torch.argmax(logits)) == 0
    checks += 1
    assert 0.0 <= multiclass_brier(probabilities, 0) <= 2.0
    checks += 1
    scan = loss_scan([-2.0, 0.0, 2.0, 4.0])
    assert all(scan[index]["nll_nats"] > scan[index + 1]["nll_nats"] for index in range(3))
    checks += 1
    print(f"CROSS_ENTROPY_SELF_TEST_OK: {checks} checks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    logits = torch.tensor([2.0, 1.0, 0.1], dtype=torch.float64)
    target = 0
    probabilities = torch.softmax(logits, dim=0)
    manual = manual_cross_entropy_nats(logits, target)
    torch_loss = float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target])))
    rows = loss_scan([-2.0, 0.0, 2.0, 4.0])
    if args.csv:
        write_scan(args.csv, rows)

    print(f"torch_version={torch.__version__}")
    print("logits_shape=(1, 3) target_shape=(1,)")
    print("probabilities=" + ",".join(f"{value:.6f}" for value in probabilities.tolist()))
    print(f"predicted_class={int(torch.argmax(logits))} target_class={target} accuracy=1.000000")
    print(f"p_true={float(probabilities[target]):.6f}")
    print(f"manual_nll_nats={manual:.6f}")
    print(f"torch_cross_entropy_nats={torch_loss:.6f}")
    print(f"multiclass_brier={multiclass_brier(probabilities, target):.6f}")
    print(f"absolute_error={abs(manual - torch_loss):.3e}")
    for row in rows:
        print(
            "scan "
            f"true_logit={row['true_logit']:.1f} "
            f"p_true={row['p_true']:.6f} "
            f"nll_nats={row['nll_nats']:.6f}"
        )
    if args.csv:
        print(f"scan_csv={args.csv}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
