"""Offline test of the production pipeline runner (archive grow, train, live)."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_crypto.models import Dataset, UnifiedRecord
from rl_crypto.pipeline import run as runner

from .conftest import make_dataset_rows


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
    assert summary["schema"] == "rl-crypto/pipeline/3"
    assert summary["archive"]["rows"] == 120
    assert summary["targets"].get("fwd_return") is not None

    # Every published target carries an out-of-sample verdict next to the
    # in-sample fit metrics, so a reader cannot mistake one for the other.
    for name, entry in summary["targets"].items():
        assert "in_sample" in entry, name
        assert "holdout" in entry, name
        assert "trusted" in entry, name
    assert summary["targets"]["fwd_return"]["holdout"]["n_samples"] > 0
    assert isinstance(summary["targets"]["fwd_return"]["holdout"]["beats_baseline"], bool)
    assert summary["samples"]["holdout"] > 0
    assert summary["feature_version"]
    assert "used" in summary["features"] and "dropped_constant" in summary["features"]

    assert (tmp_path / "models" / "eth-usd.json").is_file()
    candles = (tmp_path / "data" / "eth-usd" / "candles.jsonl").read_text().splitlines()
    assert len(candles) == 120
    first = jsonl(candles[0])
    assert "close" in first

    live = jsonl((tmp_path / "data" / "eth-usd" / "live.jsonl").read_text().splitlines()[0])
    assert live["market"]["price"] == 101.0
    assert live["predictions"]["schema"].startswith("rl-crypto/ml/")

    summary_payload = jsonl((tmp_path / "data" / "eth-usd" / "summary.json").read_text())
    assert summary_payload["schema"] == "rl-crypto/pipeline/3"
    assert set(artifacts) == {"bundle", "candles", "live", "summary"}


def test_run_pipeline_accumulates_across_runs(tmp_path, monkeypatch):
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


def test_retrain_only_touches_no_network_and_no_live_stream(tmp_path, monkeypatch):
    """Rebuilding models after a feature bump must not spend upstream quota."""

    def _boom(*a, **k):  # pragma: no cover - any call here fails the test
        raise AssertionError("retrain-only must not fetch")

    monkeypatch.setattr(runner, "build_dataset", _boom)
    monkeypatch.setattr(runner, "build_unified", _boom)

    # Seed an archive by hand, then retrain over it.
    from rl_crypto.cache import candle_dicts_from_rows, candles_path, merge_candles

    merge_candles(
        candles_path(tmp_path, "ETH/USD"),
        candle_dicts_from_rows(make_dataset_rows(120, wiggle=0.04)),
    )
    live = tmp_path / "data" / "eth-usd" / "live.jsonl"
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_text('{"kept": true}\n')

    args = _args(tmp_path)
    args.retrain_only = True
    artifacts, summary, exit_code = runner.run_pipeline(args)

    assert exit_code == 0
    assert summary["archive"]["rows"] == 120
    assert summary["live"]["fetched"] is False
    assert "live" not in artifacts
    assert (tmp_path / "models" / "eth-usd.json").is_file()
    # The existing live stream is left exactly as it was.
    assert live.read_text() == '{"kept": true}\n'


def test_run_pipeline_exit_nonzero_when_everything_fails(tmp_path, monkeypatch):
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