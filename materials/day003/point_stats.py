"""Compute beginner-friendly geometric statistics for 2D points."""

from __future__ import annotations

import math
from collections.abc import Iterable


Point = tuple[float, float]


def validate_points(points: Iterable[tuple[float, float]]) -> list[Point]:
    """Return finite 2D points as floats or raise a clear ValueError."""
    checked: list[Point] = []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"point {index} must contain exactly two coordinates")
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"point {index} coordinates must be numbers")
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"point {index} coordinates must be finite")
        checked.append((float(x), float(y)))
    if not checked:
        raise ValueError("points must not be empty")
    return checked


def point_stats(points: Iterable[tuple[float, float]]) -> dict[str, object]:
    """Return centroid, axis-aligned bounding box, and nearest point."""
    checked = validate_points(points)
    sx = sy = 0.0
    xmin = xmax = checked[0][0]
    ymin = ymax = checked[0][1]
    nearest = checked[0]
    best_d2 = nearest[0] ** 2 + nearest[1] ** 2

    for x, y in checked:
        sx += x
        sy += y
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
        d2 = x * x + y * y
        if d2 < best_d2:
            nearest = (x, y)
            best_d2 = d2

    return {
        "centroid": (sx / len(checked), sy / len(checked)),
        "bbox": ((xmin, ymin), (xmax, ymax)),
        "nearest": nearest,
        "nearest_distance_squared": best_d2,
    }


def main() -> None:
    points = [(0, 0), (2, 0), (2, 2), (0, 2)]
    result = point_stats(points)
    print("input frame: world; coordinate unit: m")
    print("centroid (m):", result["centroid"])
    print("bbox (m):", result["bbox"])
    print("nearest to origin (m):", result["nearest"])
    print("nearest squared distance (m^2):", result["nearest_distance_squared"])
    assert result["centroid"] == (1.0, 1.0)
    assert result["bbox"] == ((0.0, 0.0), (2.0, 2.0))
    assert result["nearest"] == (0.0, 0.0)
    print("All point-stat checks passed.")


if __name__ == "__main__":
    main()
