#!/usr/bin/env python3
"""Resolve ``data/`` rebase conflicts by merging, not by picking a side.

The cache writers are the 30-minute local timer and the 2-hourly Actions
pipeline, and both write under ``data/``. When their runs overlap, rebasing one
onto the other conflicts — and picking either side silently drops data that was
genuinely collected. So conflicts are resolved semantically, per artifact:

* ``live.jsonl`` — append-only observations keyed by ``generated_at``. Union
  both sides, drop exact duplicates, order by the record's timestamp. Two
  records at the same timestamp with different payloads are different
  observations and both survive.
* ``candles.jsonl`` — an upsert archive, one bar per ``timestamp``. Merge by
  timestamp (never duplicating a bar). On a collision the *newer archive* wins,
  decided by each side's newest bar: a fresh local run beats an older remote
  tip, but a rebase replayed days later does not resurrect stale OHLCV.
* anything else under ``data/`` (``summary.json``, overwritten wholesale by each
  pipeline run) — keep the side with the newest ``generated_at`` whole, so a
  replayed commit never replaces a fresher remote run with an older one.

Used by ``tools/record_live.sh`` after a rejected push, and safe to run when
there are no conflicts (it is a no-op then)::

    rl-pipeline conflicts
    rl-pipeline conflicts --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Keys that identify a record's position in time, best first.
_TIME_KEYS = ("generated_at", "_generated_at", "timestamp", "ts")

ROOT = Path(os.environ.get("RL_CRYPTO_ROOT") or Path.cwd()).resolve()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def conflicted_paths(root: Path = ROOT) -> list[str]:
    """Paths git reports as unmerged, relative to the repo root."""
    out = _git(root, "diff", "--name-only", "--diff-filter=U")
    return [line.strip() for line in out.splitlines() if line.strip()]


def _stage(root: Path, path: str, stage_number: int) -> str:
    """One side of a conflicted file as raw text (empty = side absent)."""
    try:
        return _git(root, "show", f":{stage_number}:{path}")
    except subprocess.CalledProcessError:
        return ""


def _records(raw: str) -> list[dict[str, Any]]:
    """Parse a JSONL side into records (non-JSON lines are skipped)."""
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


def _instant(value: str) -> float:
    """A timestamp as epoch seconds; 0.0 when unparseable (sorts first)."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _doc_instant(doc: Any) -> float:
    """A whole-file artifact's run instant (summary and model bundles)."""
    if not isinstance(doc, dict):
        return 0.0
    for key in (*_TIME_KEYS, "trained_at"):
        value = doc.get(key)
        if isinstance(value, str) and value:
            return _instant(value)
    return 0.0


def _servable(doc: Any) -> bool:
    """Whether a Tier-1 bundle matches the running code's feature contract.

    A bundle trained against another feature definition is flagged on load and
    predicts nothing, so during a conflict it is never the better candidate —
    even when its ``trained_at`` is newer.
    """
    if not isinstance(doc, dict):
        return False
    try:
        from ..features import FEATURE_VERSION
    except ImportError:
        # Standalone copy: ``record_live.sh`` extracts this file to a temp path
        # and runs it mid-rebase, where the checkout may predate this package.
        # Field presence is the schema boundary that matters there.
        return doc.get("feature_version") is not None
    return doc.get("feature_version") == FEATURE_VERSION


def newest_stamp(records: list[dict[str, Any]]) -> float:
    """The newest record instant in a series (0.0 for an empty side)."""
    return max((_instant(_sort_key(record)[0]) for record in records), default=0.0)


def merge_candles(
    ours: list[dict[str, Any]], theirs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One bar per timestamp; on a collision the newer archive's bar wins."""
    # The archive is an upsert series, so both sides describe the same bars. The
    # fresher series is the authority: during a rebase "theirs" may be days
    # staler than "ours", and preferring it unconditionally would regress OHLCV.
    fresh, stale = (
        (theirs, ours) if newest_stamp(theirs) >= newest_stamp(ours) else (ours, theirs)
    )
    merged: dict[str, dict[str, Any]] = {}
    for record in stale:
        merged[_sort_key(record)[0]] = record
    for record in fresh:
        merged[_sort_key(record)[0]] = record
    return sorted(merged.values(), key=_sort_key)


def _jsonblob(raw: str) -> Any | None:
    """Whole-file JSON, or ``None`` when the side is absent/unparseable."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _merge(path: str, ours: str, theirs: str) -> tuple[str, str]:
    """Return ``(label, file contents)`` for one conflicted path."""
    if path.endswith(".jsonl"):
        if Path(path).name == "candles.jsonl":
            ours_records, theirs_records = _records(ours), _records(theirs)
            merged = merge_candles(ours_records, theirs_records)
            label = (
                f"candles: {len(ours_records)} + {len(theirs_records)}"
                f" -> {len(merged)} bars"
            )
        else:
            ours_records, theirs_records = _records(ours), _records(theirs)
            merged = union_jsonl(ours_records, theirs_records)
            label = (
                f"live: {len(ours_records)} + {len(theirs_records)}"
                f" -> {len(merged)} records"
            )
        body = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in merged)
        return label, body + "\n" if body else ""

    # Overwritten artifact (summary.json, models/<slug>.json): keep the newest
    # run whole. Validated as whole-file JSON — a line-wise parse of a
    # pretty-printed document finds no records and used to leave a 0-byte file.
    ours_doc, theirs_doc = _jsonblob(ours), _jsonblob(theirs)
    name = Path(path).name
    if ours_doc is not None and theirs_doc is not None:
        # A model bundle the running code cannot serve loses to one it can,
        # whatever their training dates say; otherwise recency decides.
        if path.startswith("models/") and _servable(ours_doc) != _servable(theirs_doc):
            if _servable(theirs_doc):
                return f"{name}: replayed side (servable)", theirs
            return f"{name}: upstream side (servable)", ours
        if _doc_instant(theirs_doc) >= _doc_instant(ours_doc):
            return f"{name}: replayed side", theirs
        return f"{name}: upstream side", ours
    if theirs_doc is not None:
        return f"{name}: replayed side", theirs
    if ours_doc is not None:
        return f"{name}: upstream side", ours
    raise ValueError(f"{path}: neither side is valid JSON")


def resolve(root: Path = ROOT, *, dry_run: bool = False) -> int:
    """Resolve every conflicted path; return the number resolved."""
    paths = conflicted_paths(root)
    if not paths:
        print("no conflicted paths")
        return 0

    for path in paths:
        # During a rebase, stage 2 is the upstream we are replaying onto and
        # stage 3 is the commit being replayed.
        label, body = _merge(path, _stage(root, path, 2), _stage(root, path, 3))
        print(f"{label}: {path}")
        if dry_run:
            continue
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        subprocess.run(["git", "add", "--", path], cwd=root, check=True)

    print(f"{'would resolve' if dry_run else 'resolved'} {len(paths)} conflicted path(s)")
    return len(paths)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the conflict-resolver flags (shared by the CLI and the shim)."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the merge plan for each conflicted path without writing",
    )
    parser.add_argument(
        "--root", default=None, help="repo root to resolve in (default: the cwd)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rl-pipeline conflicts",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    return resolve(Path(args.root or ROOT), dry_run=args.dry_run)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
