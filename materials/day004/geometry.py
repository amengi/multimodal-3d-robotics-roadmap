"""Validated geometry utilities for finite 2D points.

All input coordinates must be expressed in one Cartesian coordinate frame and
one length unit.  The module does not transform coordinates between frames.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from pathlib import Path


Point = tuple[float, float]


class PointDataError(ValueError):
    """Raised when a point collection cannot be interpreted safely."""


def validate_points(points: Iterable[tuple[float, float]]) -> list[Point]:
    """Return finite 2D points as floats or raise :class:`PointDataError`."""
    checked: list[Point] = []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise PointDataError(
                f"point {index} must contain exactly two coordinates"
            )
        x, y = point
        if isinstance(x, bool) or isinstance(y, bool):
            raise PointDataError(f"point {index} coordinates must be real numbers")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise PointDataError(f"point {index} coordinates must be real numbers")
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            raise PointDataError(f"point {index} coordinates must be finite")
        checked.append((float(x), float(y)))
    if not checked:
        raise PointDataError("points must not be empty")
    return checked


def point_stats(points: Iterable[tuple[float, float]]) -> dict[str, object]:
    """Return centroid, axis-aligned bounding box, and nearest-to-origin point."""
    checked = validate_points(points)
    xs = [point[0] for point in checked]
    ys = [point[1] for point in checked]
    nearest = min(checked, key=lambda point: point[0] ** 2 + point[1] ** 2)
    return {
        "count": len(checked),
        "centroid": (sum(xs) / len(xs), sum(ys) / len(ys)),
        "bbox": ((min(xs), min(ys)), (max(xs), max(ys))),
        "nearest": nearest,
        "nearest_distance_squared": nearest[0] ** 2 + nearest[1] ** 2,
    }


def load_points_csv(path: str | Path) -> list[Point]:
    """Load ``x,y`` columns from a UTF-8 CSV file with contextual errors."""
    source = Path(path)
    try:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"x", "y"}.issubset(reader.fieldnames):
                raise PointDataError(f"{source} must have x,y header columns")
            rows: list[Point] = []
            for line_number, row in enumerate(reader, start=2):
                try:
                    x = float(row["x"])
                    y = float(row["y"])
                except (TypeError, ValueError) as exc:
                    raise PointDataError(
                        f"{source}:{line_number} x and y must be numbers"
                    ) from exc
                if not math.isfinite(x) or not math.isfinite(y):
                    raise PointDataError(
                        f"{source}:{line_number} x and y must be finite"
                    )
                rows.append((x, y))
    except OSError as exc:
        raise PointDataError(f"cannot read {source}: {exc.strerror or exc}") from exc
    return validate_points(rows)


def main() -> int:
    """Run the checked example and return a shell-friendly exit status."""
    source = Path(__file__).with_name("sample_points.csv")
    try:
        points = load_points_csv(source)
        result = point_stats(points)
    except PointDataError as exc:
        print(f"input error: {exc}")
        return 2

    print("module: materials.day004.geometry")
    print("input frame: world; coordinate unit: m; shape: (N, 2)")
    print("point count:", result["count"])
    print("centroid (m):", result["centroid"])
    print("bbox (m):", result["bbox"])
    print("nearest squared distance (m^2):", result["nearest_distance_squared"])
    print("Geometry checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
