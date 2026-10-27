"""Offline tests for the worktree hygiene tool.

The tool's promise is narrow and worth pinning: it removes only worktrees whose
commits are already reachable from another ref and whose working tree is clean.
Unique work and local edits must survive a run.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TOOL = ROOT / "tools" / "worktree_hygiene.py"


def _load():
    spec = importlib.util.spec_from_file_location("worktree_hygiene", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations through sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "file.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-m", "init")
    return repo


def _add_worktree(repo: Path, name: str, branch: str) -> Path:
    path = repo.parent / name
    _git(repo, "worktree", "add", "-b", branch, str(path))
    return path


def test_classify_keeps_unique_and_dirty_worktrees(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    contained = _add_worktree(repo, "wt-contained", "contained")
    unique = _add_worktree(repo, "wt-unique", "unique")
    dirty = _add_worktree(repo, "wt-dirty", "dirty")

    (unique / "file.txt").write_text("unique\n", encoding="utf-8")
    _git(unique, "commit", "-am", "unique work")
    (dirty / "untracked.txt").write_text("local edit\n", encoding="utf-8")

    module = _load()
    rows = module.classify(repo, module.list_worktrees(repo))
    by_path = {str(row.worktree.path): row for row in rows}

    assert by_path[str(repo)].state == "keep"
    assert by_path[str(repo)].reason == "main checkout"

    assert by_path[str(contained)].state == "remove"
    assert "another ref" in by_path[str(contained)].reason

    assert by_path[str(unique)].state == "keep"
    assert by_path[str(unique)].reason == "unique commits"

    assert by_path[str(dirty)].state == "keep"
    assert by_path[str(dirty)].reason == "uncommitted changes"


def test_apply_removes_safe_worktrees_and_prunes_their_branches(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    contained = _add_worktree(repo, "wt-contained", "contained")
    unique = _add_worktree(repo, "wt-unique", "unique")
    dirty = _add_worktree(repo, "wt-dirty", "dirty")

    (unique / "file.txt").write_text("unique\n", encoding="utf-8")
    _git(unique, "commit", "-am", "unique work")
    (dirty / "untracked.txt").write_text("local edit\n", encoding="utf-8")

    module = _load()
    module.main(["--root", str(repo), "--apply", "--prune-branches"])

    assert not contained.exists()
    assert unique.exists()
    assert dirty.exists()

    branches = _git(repo, "branch", "--format=%(refname:short)").stdout.split()
    assert "contained" not in branches
    assert "unique" in branches
    assert "dirty" in branches

    # The main checkout is never touched.
    assert (repo / "file.txt").exists()
