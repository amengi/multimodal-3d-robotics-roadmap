"""Validated NumPy examples for vectorized 3D point transforms.

Points are row-wise samples with shape ``(N, 3)`` in one Cartesian source
frame.  The mathematical convention is ``p_target = R @ p_source + t``.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class PointArrayError(ValueError):
    """Raised when point or transform arrays violate the declared contract."""


def validate_points(points: ArrayLike) -> FloatArray:
    """Return finite ``float64`` points with shape ``(N, 3)``."""
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PointArrayError("points must contain real numbers") from exc
    if array.ndim != 2 or array.shape[1] != 3:
        raise PointArrayError(
            f"points must have shape (N, 3), received {array.shape}"
        )
    if array.shape[0] == 0:
        raise PointArrayError("points must contain at least one row")
    if not np.isfinite(array).all():
        raise PointArrayError("points must contain only finite values")
    return array


def validate_rigid_transform(
    rotation: ArrayLike, translation: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return a proper 3D rotation and a finite translation vector."""
    try:
        matrix = np.asarray(rotation, dtype=np.float64)
        offset = np.asarray(translation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PointArrayError("rotation and translation must be real numbers") from exc
    if matrix.shape != (3, 3):
        raise PointArrayError(f"rotation must have shape (3, 3), received {matrix.shape}")
    if offset.shape != (3,):
        raise PointArrayError(f"translation must have shape (3,), received {offset.shape}")
    if not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        raise PointArrayError("transform values must be finite")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-10, rtol=1e-10):
        raise PointArrayError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10, rtol=1e-10):
        raise PointArrayError("rotation determinant must be +1")
    return matrix, offset


def transform_points_loop(
    points: ArrayLike, rotation: ArrayLike, translation: ArrayLike
) -> FloatArray:
    """Transform points using an explicit Python loop as a readable baseline."""
    source = validate_points(points)
    matrix, offset = validate_rigid_transform(rotation, translation)
    target = np.empty_like(source)
    for index, point in enumerate(source):
        target[index] = matrix @ point + offset
    return target


def transform_points_vectorized(
    points: ArrayLike, rotation: ArrayLike, translation: ArrayLike
) -> FloatArray:
    """Transform all rows at once using matrix multiplication and broadcasting."""
    source = validate_points(points)
    matrix, offset = validate_rigid_transform(rotation, translation)
    return source @ matrix.T + offset


def example_transform() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return a small checked transform: +90 degrees around the z axis."""
    points = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 2.0]],
        dtype=np.float64,
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translation = np.array([10.0, 0.0, -1.0], dtype=np.float64)
    return points, rotation, translation


def benchmark(point_count: int = 10_000, repeats: int = 5) -> dict[str, float]:
    """Benchmark both implementations and return minimum wall-clock durations."""
    if point_count <= 0 or repeats <= 0:
        raise ValueError("point_count and repeats must be positive")
    generator = np.random.default_rng(20260909)
    points = generator.normal(size=(point_count, 3))
    _, rotation, translation = example_transform()

    expected = transform_points_loop(points[:10], rotation, translation)
    actual = transform_points_vectorized(points[:10], rotation, translation)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    loop_samples: list[float] = []
    vector_samples: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        transform_points_loop(points, rotation, translation)
        loop_samples.append(perf_counter() - start)

        start = perf_counter()
        transform_points_vectorized(points, rotation, translation)
        vector_samples.append(perf_counter() - start)

    loop_seconds = min(loop_samples)
    vector_seconds = min(vector_samples)
    return {
        "point_count": float(point_count),
        "repeats": float(repeats),
        "loop_seconds": loop_seconds,
        "vectorized_seconds": vector_seconds,
        "speedup": loop_seconds / vector_seconds,
    }


def main() -> int:
    """Run the numerical check and a small local benchmark."""
    points, rotation, translation = example_transform()
    loop_result = transform_points_loop(points, rotation, translation)
    vector_result = transform_points_vectorized(points, rotation, translation)
    np.testing.assert_allclose(vector_result, loop_result)

    print("source frame -> target frame; coordinate unit: m")
    print("points shape/dtype:", points.shape, points.dtype)
    print("rotation shape:", rotation.shape)
    print("translation shape:", translation.shape)
    print("result shape:", vector_result.shape)
    print("first transformed point (m):", vector_result[0].tolist())

    result = benchmark()
    print(
        "benchmark: N={:d}, repeats={:d}, loop={:.6f}s, "
        "vectorized={:.6f}s, observed speedup={:.2f}x".format(
            int(result["point_count"]),
            int(result["repeats"]),
            result["loop_seconds"],
            result["vectorized_seconds"],
            result["speedup"],
        )
    )
    print("NumPy transform checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
