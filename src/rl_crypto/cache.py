"""Commit-backed cache helpers shared by the production runners.

The git-tracked cache under ``data/<slug>/`` grows on every run but never
corrupts: candles are *upserted* by timestamp into a sorted archive (so the
history never shrinks and never duplicates), and live records are appended with
a cap. Shared here so ``tools/run_pipeline.py`` and ``tools/record_live.py``
build the cache identically.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from .export import write_jsonl
from .models import DatasetRow

#: Recent live records kept in ``data/<slug>/live.jsonl`` (bounded).
LIVE_CAP_DEFAULT = 1000

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


def candle_dicts_from_rows(rows: Iterable[DatasetRow]) -> list[dict[str, Any]]:
    """The raw market bar shape, as the archive stores it (ticker + OHLCV)."""
    return [
        {
            "ticker": row.ticker,
            "timestamp": row.timestamp,
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
    is how a still-open bar gets corrected on the next run.
    """
    merged: dict[str, dict[str, Any]] = {
        str(candle["timestamp"]): dict(candle)
        for candle in read_jsonl(path)
        if candle.get("timestamp")
    }
    for candle in candles:
        timestamp = candle.get("timestamp")
        if timestamp:
            merged[str(timestamp)] = dict(candle)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(dest, (merged[key] for key in sorted(merged)))


def load_candles(path: str | Path, *, ticker: str) -> tuple[DatasetRow, ...]:
    """Reconstruct market-only dataset rows from an archive (news-free by design)."""
    rows = []
    for candle in read_jsonl(path):
        if candle.get("timestamp") is None:
            continue
        rows.append(
            DatasetRow(
                ticker=candle.get("ticker") or ticker,
                timestamp=str(candle["timestamp"]),
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


__all__ = [
    "LIVE_CAP_DEFAULT",
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
]