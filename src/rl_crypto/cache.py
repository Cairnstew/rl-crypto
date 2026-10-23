"""Commit-backed cache helpers shared by the production runners.

The git-tracked cache under ``data/<slug>/`` grows on every run but never
corrupts: candles are *upserted* by timestamp into a sorted archive (so the
history never shrinks and never duplicates), and live records are appended with
a cap. Shared here so ``rl_crypto.pipeline.run`` and ``rl_crypto.pipeline.live``
build the cache identically, and so a reader (the notebook, an analysis) can
first :func:`sync_cache` it against the remote that the runners push to.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .export import write_jsonl
from .models import DatasetRow, to_iso

#: Recent live records kept in ``data/<slug>/live.jsonl`` (bounded).
LIVE_CAP_DEFAULT = 1000

#: Wall-clock cap per git subprocess (a fetch is the slow one).
GIT_TIMEOUT_DEFAULT = 60.0

#: Cache paths :func:`sync_cache` keeps current with the remote.
CACHE_PATHS = ("data", "models")

T = TypeVar("T")


def call_with_timeout[T](
    target: Callable[[], T], *, timeout: float, label: str
) -> tuple[T | None, str]:
    """Run ``target()`` under a hard wall-clock cap; return ``(result, note)``.

    On timeout the orphaned worker is abandoned (daemon-less by design of
    ``ThreadPoolExecutor``); callers that must not hang afterwards should end
    the process with ``os._exit`` once all work is written.
    """
    executor = cf.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(target)
    try:
        return future.result(timeout=timeout), ""
    except cf.TimeoutError:
        return None, f"{label}: timed out after {timeout:.0f}s"
    except Exception as exc:  # tolerated, library-style partial failure
        return None, f"{label}: {type(exc).__name__}: {exc}"
    finally:
        executor.shutdown(wait=False)


# --------------------------------------------------------------------------- paths


def slug_for(ticker: str) -> str:
    """``ETH/USD`` -> ``eth-usd`` (the per-ticker cache directory name)."""
    return ticker.strip().replace("/", "-").replace(" ", "-").lower()


def data_dir(root: str | Path, ticker: str) -> Path:
    return Path(root) / "data" / slug_for(ticker)


def candles_path(root: str | Path, ticker: str) -> Path:
    return data_dir(root, ticker) / "candles.jsonl"


def live_path(root: str | Path, ticker: str) -> Path:
    return data_dir(root, ticker) / "live.jsonl"


def summary_path(root: str | Path, ticker: str) -> Path:
    return data_dir(root, ticker) / "summary.json"


# --------------------------------------------------------------------------- io


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, tolerating empty/garbled lines so a cache stays usable."""
    dest = Path(path)
    if not dest.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in dest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def append_capped(
    path: str | Path, record: Mapping[str, Any], *, cap: int = LIVE_CAP_DEFAULT
) -> Path:
    """Append one JSON record, keeping only the ``cap`` most recent lines."""
    lines = [json.dumps(line, ensure_ascii=False, sort_keys=True) for line in read_jsonl(path)]
    keep = max(0, cap - 1)
    lines = lines[-keep:] if keep > 0 else lines[:0]
    lines.append(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


# --------------------------------------------------------------------------- candles


def _canonical_timestamp(value: Any) -> str | None:
    """One canonical ISO-8601 ``...Z`` form for a timestamp, whatever it arrived as.

    Upstream emits ``datetime`` objects whose ``str()`` is space-separated
    (``2026-08-12 21:00:00+00:00``) while this project's own records use ``Z``.
    Keying the archive on the raw string therefore made the format load-bearing:
    a single upstream change would file the same hourly bar under a second key
    and silently duplicate the archive, while leaving the file itself mixed.
    Parsing first makes the key format-independent.
    """
    # Imported here, not at module scope: `dataset` imports this module for
    # `call_with_timeout`, so a top-level import of `parse_iso` would close a
    # cycle. The cost is a dict lookup per candle; the benefit is no import dance.
    from .dataset import parse_iso

    if value is None:
        return None
    parsed = parse_iso(value)
    return to_iso(parsed) if parsed else str(value)


def candle_dicts_from_rows(rows: Iterable[DatasetRow]) -> list[dict[str, Any]]:
    """The raw market bar shape, as the archive stores it (ticker + OHLCV)."""
    return [
        {
            "ticker": row.ticker,
            "timestamp": _canonical_timestamp(row.timestamp),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in rows
    ]


def merge_candles(path: str | Path, candles: Iterable[Mapping[str, Any]]) -> Path:
    """Upsert candles by timestamp into a sorted, append-only archive.

    Rows already archived are preserved (the archive grows beyond any single
    fetch window); a repeated timestamp is replaced by the newer candle, which
    is how a still-open bar gets corrected on the next run. Timestamps are
    canonicalised first, so the same bar always lands on the same key whatever
    format the upstream used.
    """
    merged: dict[str, dict[str, Any]] = {}
    for candle in read_jsonl(path):
        key = _canonical_timestamp(candle.get("timestamp"))
        if key:
            merged[key] = {**candle, "timestamp": key}
    for candle in candles:
        key = _canonical_timestamp(candle.get("timestamp"))
        if key:
            merged[key] = {**dict(candle), "timestamp": key}
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(dest, (merged[key] for key in sorted(merged)))


def load_candles(path: str | Path, *, ticker: str) -> tuple[DatasetRow, ...]:
    """Reconstruct market-only dataset rows from an archive (news-free by design).

    Timestamps are canonicalised on read too, so an archive written before the
    canonical form existed still yields rows the time-aware feature and target
    windows can measure against each other.
    """
    rows = []
    for candle in read_jsonl(path):
        stamp = _canonical_timestamp(candle.get("timestamp"))
        if stamp is None:
            continue
        rows.append(
            DatasetRow(
                ticker=candle.get("ticker") or ticker,
                timestamp=stamp,
                open=candle.get("open"),
                high=candle.get("high"),
                low=candle.get("low"),
                close=candle.get("close"),
                volume=candle.get("volume"),
                has_price=True,
                has_news=False,
            )
        )
    rows.sort(key=lambda row: row.timestamp)
    return tuple(rows)


# --------------------------------------------------------------------------- remote sync


@dataclass(frozen=True, slots=True)
class CacheSync:
    """How the on-disk cache compares to the git remote after a sync attempt.

    ``up_to_date`` is the single "trust the files on disk" flag; ``notes``
    explains in words what happened — or what could not happen (offline, dirty
    tree, diverged branch) — so a caller can print the reason, not a guess.
    """

    root: Path
    upstream: str | None
    local_head: str | None
    upstream_head: str | None
    behind: int
    ahead: int
    cache_stale: bool
    dirty: bool
    fetched: bool
    pulled: bool
    notes: tuple[str, ...] = ()

    @property
    def up_to_date(self) -> bool:
        return (
            self.upstream is not None
            and not self.cache_stale
            and not self.dirty
            and self.behind == 0
        )

    def summary(self) -> str:
        """One line for a human; the ``notes`` carry the detail."""
        local = self.local_head or "?"
        if self.upstream is None:
            return f"cache: no remote tracking ref (local @ {local}); using the files on disk"
        remote = self.upstream_head or "?"
        if self.up_to_date:
            return f"cache: up to date with {self.upstream} @ {remote}"
        reasons = []
        if self.behind:
            reasons.append(f"{self.behind} commit(s) behind")
        if self.ahead:
            reasons.append(f"{self.ahead} commit(s) ahead")
        if self.cache_stale:
            reasons.append("data/models differ")
        if self.dirty:
            reasons.append("uncommitted changes")
        detail = ", ".join(reasons) or "out of date"
        return f"cache: out of date ({detail}) — local @ {local}, {self.upstream} @ {remote}"


def _git(root: Path, *args: str, timeout: float) -> tuple[int, str, str]:
    """Run one git command under ``root``; return ``(returncode, stdout, stderr)``."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 128, "", f"{type(exc).__name__}: {exc}"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _tracking_ref(root: Path, *, timeout: float) -> str | None:
    """``origin/main`` for the current branch — via ``@{u}``, else the sole remote."""
    code, out, _ = _git(
        root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", timeout=timeout
    )
    if code == 0 and out.strip():
        return out.strip()
    code, remotes, _ = _git(root, "remote", timeout=timeout)
    names = remotes.split()
    code, branch, _ = _git(root, "symbolic-ref", "--short", "HEAD", timeout=timeout)
    if code == 0 and len(names) == 1 and branch.strip():
        return f"{names[0]}/{branch.strip()}"
    return None


def _cache_stale(root: Path, upstream: str, paths: tuple[str, ...], *, timeout: float) -> bool:
    """True when the committed cache paths differ between HEAD and ``upstream``."""
    existing = [name for name in paths if (root / name).exists()]
    if not existing:
        return False
    code, _, _ = _git(root, "diff", "--quiet", "HEAD", upstream, "--", *existing, timeout=timeout)
    return code == 1


def _dirty(root: Path, paths: tuple[str, ...], *, timeout: float) -> bool:
    """True when the working tree has uncommitted changes under ``paths``."""
    existing = [name for name in paths if (root / name).exists()]
    if not existing:
        return False
    code, out, _ = _git(root, "status", "--porcelain", "--", *existing, timeout=timeout)
    return code == 0 and bool(out.strip())


def sync_cache(
    root: str | Path,
    *,
    pull: bool = True,
    paths: tuple[str, ...] = CACHE_PATHS,
    timeout: float = GIT_TIMEOUT_DEFAULT,
) -> CacheSync:
    """Fetch the remote and fast-forward the checkout so the cache is current.

    Safe against a busy working tree: ``git fetch`` never touches it, and the
    update is a ``--ff-only`` pull, which refuses (leaving the tree intact)
    when local changes or local commits are in the way — the caller gets a note
    naming the command instead of a guessed-at overwrite. Every git call is
    bounded by ``timeout`` and every failure is reported in ``notes`` rather
    than raised, so a reader keeps working offline on whatever is on disk.
    """
    root = Path(root)
    notes: list[str] = []
    code, _, _ = _git(root, "rev-parse", "--is-inside-work-tree", timeout=timeout)
    if code != 0:
        return CacheSync(
            root=root,
            upstream=None,
            local_head=None,
            upstream_head=None,
            behind=0,
            ahead=0,
            cache_stale=False,
            dirty=False,
            fetched=False,
            pulled=False,
            notes=("not a git checkout; reading the files on disk",),
        )

    def rev(ref: str) -> str | None:
        code_, out_, _ = _git(root, "rev-parse", ref, timeout=timeout)
        return out_.strip() if code_ == 0 else None

    def short(sha: str | None) -> str | None:
        """A short sha for display; ``None`` passes through (unknown ref)."""
        return sha[:10] if sha else None

    def counts() -> tuple[int, int, bool]:
        """``(behind, ahead, cache_stale)`` for HEAD against the upstream ref."""
        head, tip = rev("HEAD"), rev(upstream) if upstream else None
        if not (head and tip) or head == tip:
            return 0, 0, False
        code_, out_, _ = _git(
            root, "rev-list", "--left-right", "--count", f"HEAD...{upstream}", timeout=timeout
        )
        ahead_ = behind_ = 0
        if code_ == 0 and out_.strip():
            left, right = out_.strip().split()
            ahead_, behind_ = int(left), int(right)
        stale = _cache_stale(root, upstream, paths, timeout=timeout) if upstream else False
        return behind_, ahead_, stale

    upstream = _tracking_ref(root, timeout=timeout)
    fetched = False
    if upstream is None:
        notes.append("no git remote for this branch; nothing to sync")
    else:
        remote = upstream.split("/", 1)[0]
        code, _, err = _git(root, "fetch", "--quiet", remote, timeout=timeout)
        fetched = code == 0
        if not fetched:
            reason = (err.strip().splitlines() or [f"git fetch exited {code}"])[-1]
            notes.append(
                f"fetch of {remote} failed ({reason}); "
                "comparing against the last-fetched tip"
            )

    behind, ahead, cache_stale = counts()
    dirty = _dirty(root, paths, timeout=timeout)

    pulled = False
    if upstream and "/" in upstream and (behind or cache_stale) and pull and fetched:
        remote, _, branch = upstream.partition("/")
        code, _, err = _git(root, "pull", "--ff-only", "--quiet", remote, branch, timeout=timeout)
        if code == 0:
            pulled = True
            notes.append(f"fast-forwarded to {upstream}")
        else:
            reason = (err.strip().splitlines() or [f"git pull exited {code}"])[-1]
            notes.append(
                f"remote is ahead but `git pull --ff-only` refused ({reason}); "
                "update the checkout to refresh the cache"
            )
        behind, ahead, cache_stale = counts()
        dirty = _dirty(root, paths, timeout=timeout)

    return CacheSync(
        root=root,
        upstream=upstream,
        local_head=short(rev("HEAD")),
        upstream_head=short(rev(upstream)) if upstream else None,
        behind=behind,
        ahead=ahead,
        cache_stale=cache_stale,
        dirty=dirty,
        fetched=fetched,
        pulled=pulled,
        notes=tuple(notes),
    )


__all__ = [
    "LIVE_CAP_DEFAULT",
    "CacheSync",
    "append_capped",
    "call_with_timeout",
    "candle_dicts_from_rows",
    "candles_path",
    "data_dir",
    "live_path",
    "load_candles",
    "merge_candles",
    "read_jsonl",
    "slug_for",
    "summary_path",
    "sync_cache",
]