"""Offline test of the high-frequency live recorder (no training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_crypto import build_samples
from rl_crypto.ml.store import save_bundle
from rl_crypto.ml.tier1 import train_tier1
from rl_crypto.models import UnifiedRecord
from rl_crypto.pipeline import live as tool

from .conftest import make_dataset_rows


def _write_bundle(tmp_path: Path, ticker: str = "ETH/USD") -> None:
    rows = make_dataset_rows(120, wiggle=0.04)
    bundle = train_tier1(build_samples(rows, horizon=1), horizon=1)
    slug = ticker.replace("/", "-").lower()
    save_bundle(bundle, tmp_path / "models" / f"{slug}.json")


def _candles(n: int = 60) -> list[dict]:
    rows = make_dataset_rows(n, wiggle=0.04)
    return [
        {
            "ticker": "ETH/USD",
            "timestamp": row.timestamp,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in rows
    ]


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    base = {
        "tickers": ["ETH/USD"],
        "ticker": None,
        "timeframe": "1h",
        "window": 60,
        "tail_hours": 6,
        "lookback_hours": 1,
        "fetch_timeout": 5.0,
        "live_cap": 10,
        "root": str(tmp_path),
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_record_live_appends_one_record(tmp_path, monkeypatch):
    _write_bundle(tmp_path)

    candles = _candles()
    monkeypatch.setattr(tool, "fetch_prices", lambda *a, **k: (candles, ()))
    monkeypatch.setattr(
        tool,
        "build_unified_many",
        lambda tickers, **k: [
            UnifiedRecord(ticker=t, market_ok=True, news_ok=True, price=2500.0)
            for t in tickers
        ],
    )

    payloads, paths, exit_code = tool.record_live(_args(tmp_path))

    assert exit_code == 0
    assert len(payloads) == 1
    assert payloads[0]["predictions"]["schema"].startswith("rl-crypto/ml/")
    assert payloads[0]["predictions"]["feature_version"]
    assert payloads[0]["predictions"]["metric_space"]
    assert payloads[0]["_generated_at"]
    assert "ETH/USD" in paths

    lines = (tmp_path / "data" / "eth-usd" / "live.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["market"]["price"] == 2500.0
    assert record["news"]["ok"] is True  # live news included
    assert record["predictions"]["metric_space"]["fwd_volatility"] == "log"


def test_record_live_reuses_the_latest_bundle(tmp_path, monkeypatch):
    _write_bundle(tmp_path)

    # A fresh run must keep appending to the same capped file.
    rows = make_dataset_rows(24)
    candles = [{"timestamp": r.timestamp, "close": r.close, "volume": r.volume} for r in rows]
    monkeypatch.setattr(tool, "fetch_prices", lambda *a, **k: (candles, ()))
    monkeypatch.setattr(
        tool,
        "build_unified_many",
        lambda tickers, **k: [
            UnifiedRecord(ticker=t, market_ok=True) for t in tickers
        ],
    )
    args = _args(tmp_path, window=24)
    tool.record_live(args)
    tool.record_live(args)
    assert len((tmp_path / "data" / "eth-usd" / "live.jsonl").read_text().splitlines()) == 2


def test_record_live_records_every_ticker(tmp_path, monkeypatch):
    """Every ticker is recorded on every run — no stagger, no rotation."""
    tickers = ["ETH/USD", "BTC/USD", "SOL/USD"]
    for ticker in tickers:
        _write_bundle(tmp_path, ticker)

    monkeypatch.setattr(tool, "fetch_prices", lambda *a, **k: (_candles(), ()))
    monkeypatch.setattr(
        tool,
        "build_unified_many",
        lambda ts, **k: [UnifiedRecord(ticker=t, market_ok=True, news_ok=True) for t in ts],
    )

    payloads, paths, exit_code = tool.record_live(_args(tmp_path, tickers=tickers))

    assert exit_code == 0
    assert len(payloads) == 3
    assert set(paths) == set(tickers)
    for slug in ("eth-usd", "btc-usd", "sol-usd"):
        assert len((tmp_path / "data" / slug / "live.jsonl").read_text().splitlines()) == 1


def test_record_live_isolates_a_failing_ticker(tmp_path, monkeypatch):
    """One broken ticker is reported and skipped; the rest still land."""
    tickers = ["ETH/USD", "BTC/USD"]
    for ticker in tickers:
        _write_bundle(tmp_path, ticker)

    def boom(*a, **k):
        raise RuntimeError("upstream exploded")

    real_fetch = lambda *a, **k: (_candles(), ())  # noqa: E731

    def fetch(ticker, *a, **k):
        return boom() if ticker == "BTC/USD" else real_fetch()

    monkeypatch.setattr(tool, "fetch_prices", fetch)
    monkeypatch.setattr(
        tool,
        "build_unified_many",
        lambda ts, **k: [UnifiedRecord(ticker=t, market_ok=True) for t in ts],
    )

    payloads, paths, exit_code = tool.record_live(_args(tmp_path, tickers=tickers))

    assert exit_code == 1  # a failure is reported, not silently swallowed
    assert set(paths) == {"ETH/USD"}  # BTC skipped, ETH still recorded
    assert len(payloads) == 1


def test_record_live_window_uses_archive_and_short_tail(tmp_path, monkeypatch):
    """The feature window comes from the archive plus a short tail fetch.

    The archive holds old-but-in-window candles; only a short tail is fetched,
    and the two are merged and de-duplicated by timestamp.
    """
    _write_bundle(tmp_path)

    from datetime import UTC, datetime, timedelta

    from rl_crypto.cache import merge_candles
    from rl_crypto.models import to_iso

    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    # One archived candle 5h old (inside a 48h window, outside the 1h tail).
    archived = {
        "ticker": "ETH/USD",
        "timestamp": (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "close": 111.0,
        "volume": 1.0,
    }
    merge_candles(tmp_path / "data" / "eth-usd" / "candles.jsonl", [archived])

    asked: list = []

    def fetch(ticker, **kwargs):
        asked.append(kwargs)
        # One live candle at the tail edge.
        return (
            [
                {
                    "ticker": "ETH/USD",
                    "timestamp": to_iso(now),
                    "close": 222.0,
                    "volume": 2.0,
                }
            ],
            (),
        )

    monkeypatch.setattr(tool, "fetch_prices", fetch)
    monkeypatch.setattr(
        tool,
        "build_unified_many",
        lambda ts, **k: [UnifiedRecord(ticker=t, market_ok=True) for t in ts],
    )

    tool.record_live(_args(tmp_path, window=48, tail_hours=1))

    assert asked, "the tail fetch should still run to cover the newest hour"
    # The tail request is bounded by the tail, not the whole feature window.
    tail_span = asked[0]["end"] - asked[0]["start"]
    assert tail_span <= timedelta(hours=1)


def test_record_live_single_ticker_alias(tmp_path, monkeypatch):
    """--ticker still works as an alias for --tickers (existing callers)."""
    _write_bundle(tmp_path)
    monkeypatch.setattr(tool, "fetch_prices", lambda *a, **k: (_candles(), ()))
    monkeypatch.setattr(
        tool,
        "build_unified_many",
        lambda ts, **k: [UnifiedRecord(ticker=t, market_ok=True) for t in ts],
    )
    args = _args(tmp_path, tickers=None, ticker="ETH/USD")
    # mirror main()'s alias handling
    if args.ticker and not args.tickers:
        args.tickers = [args.ticker]
    payloads, _, exit_code = tool.record_live(args)
    assert exit_code == 0
    assert [p["ticker"] for p in payloads] == ["ETH/USD"]
