#!/usr/bin/env python3
"""Resolve ``data/`` rebase conflicts by unioning, not by picking a side.

The cache writers are the 30-minute local timer and the 2-hourly Actions
pipeline, and both **append** to ``data/<slug>/live.jsonl``. When their runs
overlap, rebasing one onto the other conflicts on the tail of those files — and
picking either side would silently drop live records that were genuinely
collected.

So conflicts are resolved semantically:

* ``*.jsonl`` — append-only series: union both sides, drop exact duplicates,
  order by the record's timestamp. No record is lost, and the file stays sorted.
* anything else under ``data/`` (e.g. ``summary.json``, which is overwritten
  wholesale each run): prefer the rebased commit's version, since that is the
  newer local run.

Used by ``tools/record_live.sh`` after a rejected push, and safe to run when
there are no conflicts (it is a no-op then).

    uv run python tools/resolve_cache_conflicts.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

#: Keys that identify a record's position in time, best first.
_TIME_KEYS = ("generated_at", "_generated_at", "timestamp", "ts")

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def conflicted_paths() -> list[str]:
    """Paths git reports as unmerged, relative to the repo root."""
    out = _git("diff", "--name-only", "--diff-filter=U")
    return [line.strip() for line in out.splitlines() if line.strip()]


def _stage(path: str, stage: int) -> list[dict[str, Any]]:
    """Parse one side of a conflicted file into records (blank = empty side)."""
    try:
        raw = _git("show", f":{stage}:{path}")
    except subprocess.CalledProcessError:
        return []
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _sort_key(record: dict[str, Any]) -> tuple[str, str]:
    """Order records by their timestamp, then by serialised content.

    The second element keeps the ordering total so re-serialising an unchanged
    union twice produces byte-identical output (which keeps the cache diff
    quiet when a rebase replays a commit that changed nothing).
    """
    for key in _TIME_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            return (value, json.dumps(record, sort_keys=True))
    return ("", json.dumps(record, sort_keys=True))


def union_jsonl(ours: list[dict[str, Any]], theirs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union two append-only series: every record, de-duplicated, time-ordered."""
    seen: dict[str, dict[str, Any]] = {}
    for record in [*ours, *theirs]:
        seen.setdefault(json.dumps(record, sort_keys=True), record)
    return sorted(seen.values(), key=_sort_key)


def resolve() -> int:
    """Resolve every conflicted path; return the number resolved."""
    paths = conflicted_paths()
    if not paths:
        print("no conflicted paths")
        return 0

    for path in paths:
        # During a rebase, stage 2 is the upstream we are replaying onto and
        # stage 3 is the commit being replayed.
        base, theirs = _stage(path, 2), _stage(path, 3)

        if path.endswith(".jsonl"):
            merged = union_jsonl(base, theirs)
            body = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in merged)
            target = ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body + "\n" if body else "", encoding="utf-8")
            print(f"unioned {path}: {len(base)} + {len(theirs)} -> {len(merged)} records")
        else:
            # Overwritten artifact: the rebased commit holds the newer run.
            chosen = theirs or base
            target = ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(chosen[0], ensure_ascii=False, sort_keys=True, indent=2) + "\n"
                if chosen
                else "",
                encoding="utf-8",
            )
            print(f"kept rebased side of {path} ({len(theirs)} record(s))")

        subprocess.run(["git", "add", "--", path], cwd=ROOT, check=True)

    return len(paths)


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        count = resolve()
    except Exception as exc:
        print(f"conflict resolution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"resolved {count} conflicted path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
