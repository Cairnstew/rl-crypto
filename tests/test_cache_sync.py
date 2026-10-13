"""Remote sync: fetch, fast-forward, and the ways it safely refuses.

Everything runs against throwaway git repositories on disk (``git init`` +
``git clone`` from a local path) — no network — so the suite stays offline.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from rl_crypto.cache import sync_cache

#: Hermetic git: ignore the user's global/system config (identities could be
#: missing, signing could be on) and never prompt for credentials.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_TERMINAL_PROMPT": "0",
}


def git(cwd: Path, *args: str) -> None:
    """One git call under ``cwd`` with a throwaway identity."""
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=GIT_ENV,
    )


def commit(root: Path, rel: str, content: str) -> None:
    """Write ``root/rel`` and commit it — what a runner's push looks like."""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(root, "add", rel)
    git(root, "commit", "-q", "-m", f"update {rel}")


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """The "remote": an ordinary repo the runners commit straight into."""
    repo = tmp_path / "remote"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    commit(repo, "data/eth-usd/candles.jsonl", "one\n")
    commit(repo, "models/eth-usd.json", "{}\n")
    return repo


@pytest.fixture
def clone(tmp_path: Path, remote: Path) -> Path:
    """The reader's checkout, tracking the remote like a real clone does."""
    local = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--quiet", str(remote), str(local)],
        check=True,
        capture_output=True,
        env=GIT_ENV,
    )
    return local


def test_sync_fast_forwards_to_the_remote_tip(clone, remote):
    commit(remote, "data/eth-usd/candles.jsonl", "one\ntwo\n")

    sync = sync_cache(clone)

    assert sync.pulled is True
    assert sync.fetched is True
    assert sync.up_to_date is True
    assert any("fast-forwarded" in note for note in sync.notes)
    assert sync.summary().startswith("cache: up to date with origin/main")
    assert (clone / "data/eth-usd/candles.jsonl").read_text() == "one\ntwo\n"


def test_sync_reports_a_level_clone(clone):
    sync = sync_cache(clone)

    assert sync.up_to_date is True
    assert sync.fetched is True
    assert sync.pulled is False
    assert sync.notes == ()


def test_sync_can_decline_the_pull(clone, remote):
    commit(remote, "data/eth-usd/candles.jsonl", "one\ntwo\n")

    sync = sync_cache(clone, pull=False)

    assert sync.pulled is False
    assert sync.behind == 1
    assert sync.up_to_date is False
    # the working copy was left exactly as it was
    assert (clone / "data/eth-usd/candles.jsonl").read_text() == "one\n"


def test_sync_refuses_to_clobber_a_dirty_cache(clone, remote):
    (clone / "data/eth-usd/candles.jsonl").write_text("one\nlocal edit\n", encoding="utf-8")
    commit(remote, "data/eth-usd/candles.jsonl", "one\nremote edit\n")

    sync = sync_cache(clone)

    assert sync.pulled is False
    assert sync.behind == 1
    assert sync.up_to_date is False
    assert any("ff-only" in note for note in sync.notes)
    # the local edit survived; refusing beats guessing
    assert (clone / "data/eth-usd/candles.jsonl").read_text() == "one\nlocal edit\n"


def test_sync_refuses_a_diverged_branch(clone, remote):
    commit(clone, "notebooks/nothing.json", "local\n")
    commit(remote, "data/eth-usd/candles.jsonl", "one\ntwo\n")

    sync = sync_cache(clone)

    assert (sync.behind, sync.ahead) == (1, 1)
    assert sync.pulled is False
    assert sync.up_to_date is False


def test_sync_reports_a_dirty_working_tree(clone):
    (clone / "models/eth-usd.json").write_text("{dirty}\n", encoding="utf-8")

    sync = sync_cache(clone)

    assert sync.dirty is True
    assert sync.pulled is False
    assert sync.up_to_date is False
    assert "uncommitted changes" in sync.summary()


def test_sync_without_a_remote_says_so(tmp_path):
    repo = tmp_path / "solo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    commit(repo, "data/eth-usd/candles.jsonl", "one\n")

    sync = sync_cache(repo)

    assert sync.upstream is None
    assert sync.pulled is False
    assert any("no git remote" in note for note in sync.notes)
    assert "files on disk" in sync.summary()


def test_sync_outside_a_checkout_says_so(tmp_path):
    sync = sync_cache(tmp_path)

    assert sync.upstream is None
    assert sync.notes == ("not a git checkout; reading the files on disk",)


def test_sync_survives_a_failing_fetch(clone, remote):
    sync_cache(clone)  # level, so a remote-tracking tip exists
    git(clone, "remote", "set-url", "origin", str(remote.parent / "gone"))
    commit(remote, "data/eth-usd/candles.jsonl", "one\ntwo\nthree\n")

    sync = sync_cache(clone)

    assert sync.fetched is False
    assert sync.pulled is False
    assert any("fetch of origin failed" in note for note in sync.notes)
    # with no successful fetch it can only compare the last-fetched tip
    assert sync.upstream == "origin/main"
