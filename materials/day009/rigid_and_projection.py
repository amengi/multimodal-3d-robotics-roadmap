"""Day 009: vectorized rigid transforms and paired shared-space projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class ContractError(ValueError):
    """Raised when an array violates the documented data contract."""


def _as_float_array(value: Any, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    return array


def validate_points(points: Any) -> FloatArray:
    """Return an ``(N, 3)`` finite point array without mutating the input."""

    array = _as_float_array(points, "points")
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 1:
        raise ContractError("points must have shape (N, 3) with N >= 1")
    return array


def validate_transform(rotation: Any, translation: Any) -> tuple[FloatArray, FloatArray]:
    """Validate a proper 3-D rotation and a translation measured in metres."""

    r = _as_float_array(rotation, "rotation")
    t = _as_float_array(translation, "translation")
    if r.shape != (3, 3):
        raise ContractError("rotation must have shape (3, 3)")
    if t.shape != (3,):
        raise ContractError("translation must have shape (3,)")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-10, rtol=0.0):
        raise ContractError("rotation must satisfy R.T @ R == I")
    if not np.isclose(np.linalg.det(r), 1.0, atol=1e-10, rtol=0.0):
        raise ContractError("rotation must have determinant +1")
    return r, t


def rotation_z(degrees: float) -> FloatArray:
    """Create an active right-handed rotation about +z."""

    theta = np.deg2rad(float(degrees))
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def transform_points_loop(points: Any, rotation: Any, translation: Any) -> FloatArray:
    """Reference implementation of ``p_target = R @ p_source + t``."""

    p = validate_points(points)
    r, t = validate_transform(rotation, translation)
    result = np.empty_like(p)
    for index, point in enumerate(p):
        result[index] = r @ point + t
    return result


def transform_points_vectorized(points: Any, rotation: Any, translation: Any) -> FloatArray:
    """Vectorized row-array form: ``P_target = P_source @ R.T + t``."""

    p = validate_points(points)
    r, t = validate_transform(rotation, translation)
    return p @ r.T + t


def benchmark_transforms(
    n_points: int = 20_000, repeats: int = 5, seed: int = 20260914
) -> dict[str, float | int]:
    """Compare median wall-clock times after proving numerical equivalence."""

    if n_points < 1 or repeats < 1:
        raise ContractError("n_points and repeats must both be positive")
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, 3))
    rotation = rotation_z(31.0)
    translation = np.array([0.4, -0.2, 1.1])

    expected = transform_points_loop(points, rotation, translation)
    actual = transform_points_vectorized(points, rotation, translation)
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-12:
        raise AssertionError(f"implementations disagree: max error {max_error}")

    loop_times: list[float] = []
    vectorized_times: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        transform_points_loop(points, rotation, translation)
        loop_times.append(perf_counter() - start)

        start = perf_counter()
        transform_points_vectorized(points, rotation, translation)
        vectorized_times.append(perf_counter() - start)

    loop_median = float(np.median(loop_times))
    vectorized_median = float(np.median(vectorized_times))
    return {
        "n_points": n_points,
        "repeats": repeats,
        "loop_median_ms": 1000.0 * loop_median,
        "vectorized_median_ms": 1000.0 * vectorized_median,
        "speedup_ratio": loop_median / vectorized_median,
        "max_abs_error_m": max_error,
    }


def project_to_shared_space(
    modality_a: Any,
    modality_b: Any,
    projection_a: Any,
    projection_b: Any,
) -> tuple[FloatArray, FloatArray]:
    """Map paired rows from different feature widths into one dimension ``d``."""

    a = _as_float_array(modality_a, "modality_a")
    b = _as_float_array(modality_b, "modality_b")
    wa = _as_float_array(projection_a, "projection_a")
    wb = _as_float_array(projection_b, "projection_b")
    if a.ndim != 2 or b.ndim != 2:
        raise ContractError("modality arrays must both be 2-D")
    if a.shape[0] != b.shape[0] or a.shape[0] < 1:
        raise ContractError("paired modality arrays must have the same nonzero B")
    if wa.ndim != 2 or wb.ndim != 2:
        raise ContractError("projection matrices must both be 2-D")
    if a.shape[1] != wa.shape[0] or b.shape[1] != wb.shape[0]:
        raise ContractError("feature widths must match projection input dimensions")
    if wa.shape[1] != wb.shape[1] or wa.shape[1] < 1:
        raise ContractError("both projections must produce the same positive d")
    return a @ wa, b @ wb


def cosine_similarity_matrix(left: Any, right: Any) -> FloatArray:
    """Return all pairwise cosine similarities with shape ``(B_left, B_right)``."""

    x = _as_float_array(left, "left embeddings")
    y = _as_float_array(right, "right embeddings")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ContractError("embedding arrays must be 2-D with the same feature width")
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    if np.any(x_norm == 0.0) or np.any(y_norm == 0.0):
        raise ContractError("cosine similarity is undefined for a zero vector")
    return (x / x_norm) @ (y / y_norm).T


def run_demo(output_dir: Path, n_points: int = 20_000, repeats: int = 5) -> dict[str, Any]:
    """Run the hand-check, benchmark, paired projection, and one pairing failure."""

    source = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-1.0, 0.0, 1.0]])
    rotation = rotation_z(90.0)
    translation = np.array([10.0, 1.0, -1.0])
    transformed = transform_points_vectorized(source, rotation, translation)
    expected = np.array([[10.0, 2.0, -1.0], [8.0, 1.0, -1.0], [10.0, 0.0, 0.0]])
    np.testing.assert_allclose(transformed, expected, atol=1e-12, rtol=0.0)

    modality_a = np.array([[1.0, 0.0], [0.0, 1.0]])
    modality_b = np.array([[1.0, 9.0, 0.0], [0.0, 8.0, 1.0]])
    projection_a = np.eye(2)
    projection_b = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    embedding_a, embedding_b = project_to_shared_space(
        modality_a, modality_b, projection_a, projection_b
    )
    paired_similarity = cosine_similarity_matrix(embedding_a, embedding_b)
    swapped_similarity = cosine_similarity_matrix(embedding_a, embedding_b[::-1])

    report: dict[str, Any] = {
        "convention": "column: p_target=R@p_source+t; row batch: P_target=P_source@R.T+t",
        "units": "point coordinates and translation are metres; embeddings are dimensionless",
        "source_shape": list(source.shape),
        "translation_shape": list(translation.shape),
        "transformed_points_m": transformed.tolist(),
        "paired_similarity": paired_similarity.tolist(),
        "swapped_pairing_similarity": swapped_similarity.tolist(),
        "benchmark": benchmark_transforms(n_points=n_points, repeats=repeats),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "day009_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/Day009/reference_outputs"),
        help="directory for day009_report.json",
    )
    parser.add_argument("--n-points", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    report = run_demo(args.output, args.n_points, args.repeats)
    benchmark = report["benchmark"]
    print("source/translation shapes:", tuple(report["source_shape"]), tuple(report["translation_shape"]))
    print("transformed points (m):")
    print(np.array(report["transformed_points_m"]))
    print("paired similarity:")
    print(np.array(report["paired_similarity"]))
    print("swapped-pairing similarity:")
    print(np.array(report["swapped_pairing_similarity"]))
    print(
        "benchmark: "
        f"loop={benchmark['loop_median_ms']:.3f} ms, "
        f"vectorized={benchmark['vectorized_median_ms']:.3f} ms, "
        f"speedup={benchmark['speedup_ratio']:.1f}x, "
        f"max_error={benchmark['max_abs_error_m']:.3e} m"
    )
    print("saved:", report["report_path"])
    print("All Day 009 checks passed.")


if __name__ == "__main__":
    main()
