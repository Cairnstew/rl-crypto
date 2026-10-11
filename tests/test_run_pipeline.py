"""Offline test of the production pipeline runner (cache writing + exit code).

The runner itself hits the live upstreams; here its fetch/unify seams are
monkeypatched with deterministic dataset/unified-record doubles so the whole
cache-writing path (bundle, history, live capped log, summary) is exercised
with no network.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from rl_crypto.models import Dataset, UnifiedRecord

from .conftest import make_dataset_rows

_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "run_pipeline.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_pipeline", _TOOLS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_pipeline_writes_the_whole_cache(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "build_dataset",
        lambda *a, **k: Dataset(ticker="ETH/USD", interval="1h", rows=make_dataset_rows(120, wiggle=0.04)),
    )
    monkeypatch.setattr(
        runner,
        "build_unified",
        lambda *a, **k: UnifiedRecord(ticker="ETH/USD", market_ok=True, news_ok=True, price=101.0),
    )

    args = argparse.Namespace(
        ticker="ETH/USD",
        timeframe="1h",
        granularity="hour",
        horizon=1,
        train_days=21,
        lookback_hours=1,
        limit=None,
        root=str(tmp_path),
    )
    artifacts, summary, exit_code = runner.run_pipeline(args)

    assert exit_code == 0
    assert summary["ticker"] == "ETH/USD"
    assert summary["targets"].get("fwd_return") is not None

    assert (tmp_path / "models" / "eth-usd.json").is_file()
    history = json.loads((tmp_path / "data" / "eth-usd" / "history.jsonl").read_text().splitlines()[0])
    assert "ret_1" in history and "fwd_return" in history

    live_path = tmp_path / "data" / "eth-usd" / "live.jsonl"
    first = json.loads(live_path.read_text().splitlines()[0])
    assert first["market"]["price"] == 101.0
    assert first["predictions"]["schema"].startswith("rl-crypto/ml/")

    summary_payload = json.loads((tmp_path / "data" / "eth-usd" / "summary.json").read_text())
    assert summary_payload["schema"] == "rl-crypto/pipeline/1"
    assert all(name in artifacts for name in ("bundle", "history", "live", "summary"))


def test_live_cache_is_capped(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "build_dataset",
        lambda *a, **k: Dataset(ticker="ETH/USD", interval="1h", rows=make_dataset_rows(120)),
    )
    state = {"run": 0}

    def fake_unified(*a, **k):
        state["run"] += 1
        return UnifiedRecord(ticker="ETH/USD", market_ok=True, news_ok=True, price=float(state["run"]))

    monkeypatch.setattr(runner, "build_unified", fake_unified)

    args = argparse.Namespace(
        ticker="ETH/USD", timeframe="1h", granularity="hour", horizon=1, train_days=21,
        lookback_hours=1, limit=None, root=str(tmp_path),
    )
    runner.LIVE_CAP = 2
    runner.run_pipeline(args)  # run 1
    runner.run_pipeline(args)  # run 2
    runner.run_pipeline(args)  # run 3

    lines = (tmp_path / "data" / "eth-usd" / "live.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_exit_code_nonzero_when_everything_fails(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner, "build_dataset", lambda *a, **k: Dataset(ticker="ETH/USD", rows=()))
    monkeypatch.setattr(
        runner,
        "build_unified",
        lambda *a, **k: UnifiedRecord(ticker="ETH/USD", market_ok=False, news_ok=False),
    )

    args = argparse.Namespace(
        ticker="ETH/USD", timeframe="1h", granularity="hour", horizon=1, train_days=21,
        lookback_hours=1, limit=None, root=str(tmp_path),
    )
    _, _, exit_code = runner.run_pipeline(args)
    assert exit_code == 1