"""Validate the directory tree created in the Day 002 terminal exercise."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_DIRECTORIES = ("data/raw", "data/processed", "scripts", "notes")
REQUIRED_FILES = ("README.md", "data/raw/sensors.csv", "notes/commands.md")


def validate_tree(root: Path) -> list[str]:
    problems: list[str] = []
    for relative in REQUIRED_DIRECTORIES:
        if not (root / relative).is_dir():
            problems.append(f"missing directory: {relative}")
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            problems.append(f"missing file: {relative}")
    processed = root / "data/processed/sensors_clean.csv"
    stale_copy = root / "data/processed/sensors.csv"
    if not processed.is_file():
        problems.append("missing moved file: data/processed/sensors_clean.csv")
    if stale_copy.exists():
        problems.append("data/processed/sensors.csv should have been moved, not copied")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="path to experiments/Day002/terminal_lab")
    args = parser.parse_args()
    problems = validate_tree(args.root)
    if problems:
        print("Tree check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("Tree check passed: 4 directories and 4 required files are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
