"""Offline test of the high-frequency live recorder (no training)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from rl_crypto import build_samples
from rl_crypto.ml.store import save_bundle
from rl_crypto.ml.tier1 import train_tier1
from rl_crypto.models import UnifiedRecord

from .conftest import make_dataset_rows

_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "record_live.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("record_live", _TOOLS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bundle(tmp_path: Path) -> None:
    rows = make_dataset_rows(120, wiggle=0.04)
    bundle = train_tier1(build_samples(rows, horizon=1), horizon=1)
    save_bundle(bundle, tmp_path / "models" / "eth-usd.json")


def test_record_live_appends_one_record(tmp_path, monkeypatch):
    tool = _load_tool()
    _write_bundle(tmp_path)

    rows = make_dataset_rows(60, wiggle=0.04)
    candles = [
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
    monkeypatch.setattr(tool, "fetch_prices", lambda *a, **k: (candles, ()))
    monkeypatch.setattr(
        tool,
        "build_unified",
        lambda *a, **k: UnifiedRecord(ticker="ETH/USD", market_ok=True, news_ok=True, price=2500.0),
    )

    args = argparse.Namespace(
        ticker="ETH/USD", timeframe="1h", window=60, lookback_hours=1,
        fetch_timeout=5.0, live_cap=10, root=str(tmp_path),
    )
    payload, _, exit_code = tool.record_live(args)

    assert exit_code == 0
    assert payload["predictions"]["schema"].startswith("rl-crypto/ml/")
    assert payload["_generated_at"]

    lines = (tmp_path / "data" / "eth-usd" / "live.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["market"]["price"] == 2500.0
    assert record["news"]["ok"] is True  # live news included


def test_record_live_reuses_the_latest_bundle(tmp_path, monkeypatch):
    tool = _load_tool()
    _write_bundle(tmp_path)

    # A fresh run must keep appending to the same capped file.
    rows = make_dataset_rows(24)
    candles = [{"timestamp": r.timestamp, "close": r.close, "volume": r.volume} for r in rows]
    monkeypatch.setattr(tool, "fetch_prices", lambda *a, **k: (candles, ()))
    monkeypatch.setattr(
        tool, "build_unified", lambda *a, **k: UnifiedRecord(ticker="ETH/USD", market_ok=True)
    )
    args = argparse.Namespace(
        ticker="ETH/USD", timeframe="1h", window=24, lookback_hours=1,
        fetch_timeout=5.0, live_cap=10, root=str(tmp_path),
    )
    tool.record_live(args)
    tool.record_live(args)
    assert len((tmp_path / "data" / "eth-usd" / "live.jsonl").read_text().splitlines()) == 2