"""Read the DVC and Git provenance of a dataset so training can record it.

The agent later answers "which dataset did this run use" from these recorded values.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def dvc_md5(data_path: Path) -> str | None:
    """md5 recorded in the .dvc file next to the data file, if there is one."""
    dvc_file = data_path.with_name(data_path.name + ".dvc")
    if not dvc_file.is_file():
        return None
    match = re.search(r"md5:\s*([0-9a-f]{32})", dvc_file.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def dataset_provenance(data_path: str | Path, repo: str | Path = ".") -> dict[str, str]:
    """Tags describing the dataset used: path, DVC md5, Git commits and version tag.

    Missing values are simply left out, never guessed.
    """
    repo_path = Path(repo).resolve()
    data = Path(data_path)
    if not data.is_absolute():
        data = repo_path / data
    rel = data.relative_to(repo_path).as_posix()
    tags: dict[str, str] = {"dvc.dataset_path": rel}

    md5 = dvc_md5(data)
    if md5:
        tags["dvc.md5"] = md5
    head = _git(["rev-parse", "HEAD"], repo_path)
    if head:
        tags["git.commit"] = head
    dvc_file = rel + ".dvc"
    data_commit = _git(["log", "-1", "--format=%H", "--", dvc_file], repo_path)
    if data_commit:
        tags["dvc.dataset_commit"] = data_commit
        tag_list = _git(["tag", "--points-at", data_commit, "--list", "data-v*"], repo_path)
        if tag_list:
            tags["dataset.version"] = tag_list.splitlines()[0].removeprefix("data-")
    dirty = _git(["status", "--porcelain", "--", dvc_file], repo_path)
    if dirty:
        tags["dvc.uncommitted_change"] = "true"
    return tags
