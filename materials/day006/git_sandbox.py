"""Create a disposable Git repository that intentionally stops at a conflict.

The target must be a new or empty directory.  The current course repository is
never modified by this module; every Git command runs inside the target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile


class GitLabError(RuntimeError):
    """Raised when the disposable Git lab cannot be created safely."""


@dataclass(frozen=True)
class GitResult:
    """A small record of one Git command."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def _git(repository: Path, *args: str, check: bool = True) -> GitResult:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    result = GitResult(
        args=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitLabError(f"git {' '.join(args)} failed: {detail}")
    return result


def _prepare_target(target: Path) -> Path:
    resolved = target.expanduser().resolve()
    for parent in resolved.parents:
        if (parent / ".git").exists():
            raise GitLabError(
                "target must be outside an existing Git repository; "
                f"found outer repository: {parent}"
            )
    if resolved.exists():
        if not resolved.is_dir():
            raise GitLabError(f"target is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise GitLabError(
                f"target must be new or empty; refusing to overwrite: {resolved}"
            )
    else:
        resolved.mkdir(parents=True)
    return resolved


def create_conflict_lab(target: Path) -> dict[str, object]:
    """Create three commits on two branches and leave an expected merge conflict."""
    repository = _prepare_target(target)
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Day 006 Learner")
    _git(repository, "config", "user.email", "day006@example.invalid")

    config_path = repository / "experiment.cfg"
    config_path.write_text(
        "seed=20260910\nlearning_rate=0.10\nepochs=200\n",
        encoding="utf-8",
    )
    _git(repository, "add", "experiment.cfg")
    _git(repository, "commit", "-m", "chore: add reproducible baseline config")

    _git(repository, "switch", "-c", "experiment-lr")
    config_path.write_text(
        "seed=20260910\nlearning_rate=0.05\nepochs=200\n",
        encoding="utf-8",
    )
    _git(repository, "add", "experiment.cfg")
    _git(repository, "commit", "-m", "experiment: lower learning rate")

    _git(repository, "switch", "main")
    config_path.write_text(
        "seed=20260910\nlearning_rate=0.20\nepochs=200\n",
        encoding="utf-8",
    )
    _git(repository, "add", "experiment.cfg")
    _git(repository, "commit", "-m", "experiment: raise learning rate")

    merge = _git(repository, "merge", "experiment-lr", check=False)
    conflicted_text = config_path.read_text(encoding="utf-8")
    merge_head = repository / ".git" / "MERGE_HEAD"
    if merge.returncode == 0 or not merge_head.exists() or "<<<<<<<" not in conflicted_text:
        raise GitLabError("expected a content conflict, but Git did not leave one")

    status = _git(repository, "status", "--short").stdout.strip()
    history = _git(
        repository, "log", "--all", "--graph", "--oneline", "--decorate"
    ).stdout.strip()
    return {
        "repository": repository,
        "merge_returncode": merge.returncode,
        "status": status,
        "history": history,
        "conflicted_text": conflicted_text,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a disposable repository paused at an expected merge conflict."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(tempfile.gettempdir()) / "day006-git-lab",
        help="new or empty directory outside any existing Git repository",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = create_conflict_lab(args.target)
    print("Disposable repository:", result["repository"])
    print("Expected merge conflict created; non-zero merge exit was observed.")
    print("\ngit status --short")
    print(result["status"])
    print("\ngit log --all --graph --oneline --decorate")
    print(result["history"])
    print("\nexperiment.cfg now contains conflict markers:")
    print(result["conflicted_text"])
    print("Next: cd into the lab, resolve experiment.cfg, then run:")
    print("  git diff")
    print("  git add experiment.cfg")
    print('  git commit -m "merge: resolve learning-rate experiment"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
