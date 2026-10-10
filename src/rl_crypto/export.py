"""Persistence helpers, mirroring ticker-news-signals' ``export`` module.

``write_jsonl`` writes the trailing-newline separator itself and supports an
explicit append mode, so a timer can grow a log without the truncate/append
race that a shell ``>>`` next to a ``>`` invites. Everything written here is
already JSON-ready (see :mod:`rl_crypto.models`).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION", "write_csv", "write_json", "write_jsonl"]


def write_json(path: str | Path, payload: Any, *, sort_keys: bool = False) -> Path:
    """Write a single JSON document (pretty), creating parent dirs."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )
    return dest


def write_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    append: bool = False,
) -> Path:
    """Write one JSON object per line.

    ``append=False`` (the default) replaces the file. ``append=True`` adds the
    new records to whatever the file already holds — the mode a timer wants.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with dest.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return dest


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write rows as CSV, using the first row's keys as the header.

    Writes an empty file (no header) when there are no rows, so a caller never
    has to guess a column set that was never observed.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    materialised = list(rows)
    with dest.open("w", newline="", encoding="utf-8") as handle:
        if not materialised:
            return dest
        writer = csv.DictWriter(handle, fieldnames=list(materialised[0].keys()))
        writer.writeheader()
        writer.writerows(materialised)
    return dest
