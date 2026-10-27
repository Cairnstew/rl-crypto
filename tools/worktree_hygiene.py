#!/usr/bin/env python3
"""Prune stale git worktrees left behind by model-improvement runs.

Two different tools create worktrees in this repo, and only one of them cleans
up after itself:

* ``tools/model_improve.py`` creates ``.model-improve/worktrees/<run>`` and
  removes it on a normal exit (``--keep-worktree`` opts out). A crash, a
  timeout, or an early stop leaves it behind.
* The ensemble runner's ``team_spawn(worktree=true)`` creates sibling
  ``../rl-crypto-<member>`` worktrees. Those are removed by ``team_cleanup``,
  which a paused or abandoned round never reaches — and ``team_cleanup`` only
  knows about the *current* team, so stale worktrees outlive the session.

The safe rule is: a worktree can go when it is not the main checkout, has no
modified or untracked files, and its HEAD is already reachable from another
ref (a local branch or a remote-tracking branch). Anything with unique commits
or local edits is reported and kept.

Report-only by default::

    python3 tools/worktree_hygiene.py                 # show the table
    python3 tools/worktree_hygiene.py --apply         # remove the safe ones
    python3 tools/worktree_hygiene.py --apply --prune-branches

``--prune-branches`` deletes local branches that no longer have a worktree and
whose tip is contained in another ref, so the removed round branches do not
linger as bare refs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Worktree:
    path: Path
    head: str
    branch: str | None  # short ref name (e.g. "main"), None when detached


@dataclass(frozen=True)
class Row:
    worktree: Worktree
    state: str  # "remove" | "keep"
    reason: str


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def list_worktrees(root: Path) -> list[Worktree]:
    """Parse ``git worktree list --porcelain`` into Worktree records."""
    out = _git(root, "worktree", "list", "--porcelain")
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "git worktree list failed")

    worktrees: list[Worktree] = []
    path: Path | None = None
    head = ""
    branch: str | None = None
    for line in [*out.stdout.splitlines(), ""]:
        if line.startswith("worktree "):
            path = Path(line.removeprefix("worktree "))
            head, branch = "", None
        elif line.startswith("HEAD "):
            head = line.removeprefix("HEAD ")
        elif line.startswith("branch "):
            branch = line.removeprefix("branch ").removeprefix("refs/heads/")
        elif line == "" and path is not None:
            worktrees.append(Worktree(path=path, head=head, branch=branch))
            path = None
    return worktrees


def all_refs(root: Path) -> list[str]:
    out = _git(root, "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes")
    return [line for line in out.stdout.splitlines() if line]


def is_contained(root: Path, sha: str, refs: list[str]) -> bool:
    """True when ``sha`` is already reachable from one of ``refs``."""
    return any(
        _git(root, "merge-base", "--is-ancestor", sha, ref).returncode == 0 for ref in refs
    )


def is_dirty(root: Path, worktree: Worktree) -> bool:
    out = _git(root, "-C", str(worktree.path), "status", "--porcelain", "--untracked-files=normal")
    return bool(out.stdout.strip())


def classify(root: Path, worktrees: list[Worktree]) -> list[Row]:
    """Decide what is safe to remove. The first worktree is the main checkout."""
    rows: list[Row] = []
    refs = all_refs(root)
    for index, worktree in enumerate(worktrees):
        if index == 0:
            rows.append(Row(worktree, "keep", "main checkout"))
            continue
        if not worktree.path.exists():
            rows.append(Row(worktree, "remove", "missing directory (prunable)"))
            continue
        if is_dirty(root, worktree):
            rows.append(Row(worktree, "keep", "uncommitted changes"))
            continue
        own = f"refs/heads/{worktree.branch}" if worktree.branch else None
        others = [ref for ref in refs if ref != own and not ref.endswith("/HEAD")]
        if worktree.branch is None or is_contained(root, worktree.head, others):
            rows.append(Row(worktree, "remove", "already in another ref"))
        else:
            rows.append(Row(worktree, "keep", "unique commits"))
    return rows


def apply(root: Path, rows: list[Row], *, prune_branches: bool) -> None:
    removed: list[Worktree] = []
    for row in rows:
        if row.state != "remove":
            continue
        path = row.worktree.path
        if path.exists():
            result = _git(root, "worktree", "remove", str(path))
            if result.returncode != 0:
                print(f"failed to remove {path}: {result.stderr.strip()}", file=sys.stderr)
                continue
        removed.append(row.worktree)
        print(f"removed worktree {path}")

    _git(root, "worktree", "prune")

    if not prune_branches:
        return
    current = _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    remaining = {w.branch for w in list_worktrees(root)}
    refs = all_refs(root)
    for worktree in removed:
        branch = worktree.branch
        if branch is None or branch == current or branch in remaining:
            continue
        if any(w.branch == branch for w in list_worktrees(root)):
            continue
        own = f"refs/heads/{branch}"
        others = [ref for ref in refs if ref != own and not ref.endswith("/HEAD")]
        if not is_contained(root, worktree.head, others):
            print(f"kept branch {branch} (unique commits)")
            continue
        result = _git(root, "branch", "-D", branch)
        if result.returncode == 0:
            print(f"deleted branch {branch}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    parser.add_argument("--apply", action="store_true", help="remove the safe worktrees")
    parser.add_argument(
        "--prune-branches",
        action="store_true",
        help="also delete contained branches that no longer have a worktree",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    worktrees = list_worktrees(root)
    rows = classify(root, worktrees)

    width = max(len(str(row.worktree.path)) for row in rows)
    print(f"{'worktree':<{width}}  {'branch':<38}  state   reason")
    for row in rows:
        branch = row.worktree.branch or "(detached)"
        print(f"{row.worktree.path!s:<{width}}  {branch:<38}  {row.state:<7} {row.reason}")

    safe = [row for row in rows if row.state == "remove"]
    print(f"\n{len(safe)} removable, {len(rows) - len(safe) - 1} kept, 1 main checkout")
    if safe and not args.apply:
        print("re-run with --apply to remove the safe worktrees")

    if args.apply:
        apply(root, rows, prune_branches=args.prune_branches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
