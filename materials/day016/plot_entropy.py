#!/usr/bin/env python3
"""Validate Day 016 entropy CSV and render a dependency-free SVG curve."""

from __future__ import annotations

import csv
import math
import pathlib
import sys


def read_rows(path: pathlib.Path) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["p", "entropy_bits"]:
            raise ValueError("CSV header must be p,entropy_bits")
        rows = [(float(row["p"]), float(row["entropy_bits"])) for row in reader]
    if len(rows) != 11:
        raise ValueError("CSV must contain 11 data rows")
    if any(not (math.isfinite(p) and math.isfinite(h)) for p, h in rows):
        raise ValueError("CSV values must be finite")
    if any(abs(p - index / 10.0) > 1e-9 for index, (p, _) in enumerate(rows)):
        raise ValueError("p values must be 0.0, 0.1, ..., 1.0")
    if abs(rows[0][1]) > 1e-9 or abs(rows[5][1] - 1.0) > 1e-9 or abs(rows[-1][1]) > 1e-9:
        raise ValueError("entropy endpoints/maximum do not match Bernoulli entropy")
    return rows


def render_svg(rows: list[tuple[float, float]], output: pathlib.Path) -> None:
    width, height = 720, 440
    left, right, top, bottom = 80, 680, 40, 370

    def point(p: float, entropy: float) -> tuple[float, float]:
        return left + p * (right - left), bottom - entropy * (bottom - top)

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(p, h) for p, h in rows))
    circles = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#d9485f"/>'
        for x, y in (point(p, h) for p, h in rows)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="360" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Bernoulli entropy H₂(p)</text>
<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222"/>
<line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222"/>
<line x1="{left}" y1="{top}" x2="{right}" y2="{top}" stroke="#bbb" stroke-dasharray="5 5"/>
<line x1="{(left + right) / 2}" y1="{bottom}" x2="{(left + right) / 2}" y2="{top}" stroke="#bbb" stroke-dasharray="5 5"/>
<polyline points="{points}" fill="none" stroke="#2864b7" stroke-width="3"/>
{circles}
<text x="{left}" y="{bottom + 24}" text-anchor="middle" font-family="sans-serif">0</text>
<text x="{(left + right) / 2}" y="{bottom + 24}" text-anchor="middle" font-family="sans-serif">0.5</text>
<text x="{right}" y="{bottom + 24}" text-anchor="middle" font-family="sans-serif">1</text>
<text x="{(left + right) / 2}" y="{height - 12}" text-anchor="middle" font-family="sans-serif">p = P(X=1), dimensionless</text>
<text x="24" y="{(top + bottom) / 2}" transform="rotate(-90 24 {(top + bottom) / 2})" text-anchor="middle" font-family="sans-serif">entropy (bit)</text>
<text x="{left - 12}" y="{bottom + 5}" text-anchor="end" font-family="sans-serif">0</text>
<text x="{left - 12}" y="{top + 5}" text-anchor="end" font-family="sans-serif">1</text>
</svg>
'''
    output.write_text(svg, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: plot_entropy.py INPUT.csv OUTPUT.svg", file=sys.stderr)
        return 2
    input_path = pathlib.Path(sys.argv[1])
    output_path = pathlib.Path(sys.argv[2])
    rows = read_rows(input_path)
    render_svg(rows, output_path)
    print(f"PLOT_OK:path={output_path},points={len(rows)},max_bits={max(h for _, h in rows):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
