"""Commit-backed cache helpers: archive upsert, capped append, candle load."""

from __future__ import annotations

import json

from rl_crypto.cache import (
    append_capped,
    candle_dicts_from_rows,
    load_candles,
    merge_candles,
    read_jsonl,
)

from .conftest import make_dataset_rows


def test_merge_candles_dedupes_sorts_and_preserves_history(tmp_path):
    path = tmp_path / "candles.jsonl"
    merge_candles(
        path,
        [{"ticker": "ETH/USD", "timestamp": "2026-01-01T02:00:00Z", "close": 102.0},
         {"ticker": "ETH/USD", "timestamp": "2026-01-01T00:00:00Z", "close": 100.0},
         {"ticker": "ETH/USD", "timestamp": "2026-01-01T01:00:00Z", "close": 101.0}],
    )
    # second run: older rows preserved, the same bar updated
    merge_candles(
        path,
        [{"ticker": "ETH/USD", "timestamp": "2026-01-01T01:00:00Z", "close": 101.5},
         {"ticker": "ETH/USD", "timestamp": "2026-01-01T03:00:00Z", "close": 103.0}],
    )
    rows = read_jsonl(path)
    assert [r["timestamp"] for r in rows] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z",
        "2026-01-01T02:00:00Z",
        "2026-01-01T03:00:00Z",
    ]
    assert rows[1]["close"] == 101.5  # upserted, not duplicated


def test_append_capped_keeps_recent_records(tmp_path):
    path = tmp_path / "live.jsonl"
    for i in range(5):
        append_capped(path, {"record": i}, cap=3)
    payloads = [json.loads(line)["record"] for line in path.read_text().splitlines()]
    assert payloads == [2, 3, 4]


def test_load_candles_reconstructs_rows(tmp_path):
    rows = make_dataset_rows(10)
    path = merge_candles(tmp_path / "c.jsonl", candle_dicts_from_rows(rows))
    loaded = load_candles(path, ticker="ETH/USD")
    assert len(loaded) == 10
    assert all(row.has_price for row in loaded)
    assert loaded[0].timestamp == rows[0].timestamp
    assert loaded[0].close == rows[0].close


def test_read_jsonl_tolerates_garbage(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text('{"a": 1}\nnot json\n{"b": 2}\n\ntotally broken\n')
    assert len(read_jsonl(path)) == 2