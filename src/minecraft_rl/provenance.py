"""Where an experiment run came from, for run metadata."""

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def git_commit() -> str | None:
    """The checked-out commit, suffixed `-dirty` when tracked files are modified.

    None outside a Git checkout.
    """
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0:
        return None
    dirty = _git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    return head.stdout.strip() + ("-dirty" if dirty else "")
