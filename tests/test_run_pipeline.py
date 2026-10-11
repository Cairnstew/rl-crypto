"""Offline test of the production pipeline runner (archive grow, train, live)."""

from __future__ import annotations

import argparse
import importlib.util
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


def _args(tmp_path: Path):
    return argparse.Namespace(
        ticker="ETH/USD",
        timeframe="1h",
        horizon=1,
        history_days=21,
        lookback_hours=1,
        limit=None,
        fetch_timeout=10.0,
        root=str(tmp_path),
    )


def test_run_pipeline_grows_archive_and_writes_cache(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "build_dataset",
        lambda *a, **k: Dataset(
            ticker="ETH/USD", interval="1h", rows=make_dataset_rows(120, wiggle=0.04)
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_unified",
        lambda *a, **k: UnifiedRecord(ticker="ETH/USD", market_ok=True, news_ok=True, price=101.0),
    )

    artifacts, summary, exit_code = runner.run_pipeline(_args(tmp_path))

    assert exit_code == 0
    assert summary["schema"] == "rl-crypto/pipeline/2"
    assert summary["archive"]["rows"] == 120
    assert summary["targets"].get("fwd_return") is not None

    assert (tmp_path / "models" / "eth-usd.json").is_file()
    candles = (tmp_path / "data" / "eth-usd" / "candles.jsonl").read_text().splitlines()
    assert len(candles) == 120
    first = jsonl(candles[0])
    assert "close" in first

    live = jsonl((tmp_path / "data" / "eth-usd" / "live.jsonl").read_text().splitlines()[0])
    assert live["market"]["price"] == 101.0
    assert live["predictions"]["schema"].startswith("rl-crypto/ml/")

    summary_payload = jsonl((tmp_path / "data" / "eth-usd" / "summary.json").read_text())
    assert summary_payload["schema"] == "rl-crypto/pipeline/2"
    assert set(artifacts) == {"bundle", "candles", "live", "summary"}


def test_run_pipeline_accumulates_across_runs(tmp_path, monkeypatch):
    runner = _load_runner()
    rows = make_dataset_rows(50)
    monkeypatch.setattr(runner, "build_dataset", lambda *a, **k: Dataset(ticker="ETH/USD", rows=rows))
    monkeypatch.setattr(
        runner, "build_unified", lambda *a, **k: UnifiedRecord(ticker="ETH/USD", market_ok=True)
    )

    runner.run_pipeline(_args(tmp_path))
    rows2 = make_dataset_rows(70)
    monkeypatch.setattr(runner, "build_dataset", lambda *a, **k: Dataset(ticker="ETH/USD", rows=rows2))

    _, summary, _ = runner.run_pipeline(_args(tmp_path))
    assert summary["archive"]["rows"] == 70  # upsert over the shared prefix + new tail


def test_run_pipeline_exit_nonzero_when_everything_fails(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner, "build_dataset", lambda *a, **k: Dataset(ticker="ETH/USD", rows=()))
    monkeypatch.setattr(
        runner,
        "build_unified",
        lambda *a, **k: UnifiedRecord(ticker="ETH/USD", market_ok=False, news_ok=False),
    )

    _, summary, exit_code = runner.run_pipeline(_args(tmp_path))
    assert exit_code == 1
    assert summary["archive"]["rows"] == 0


def jsonl(text: str) -> dict:
    import json

    return json.loads(text)