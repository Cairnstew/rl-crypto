"""CLI parsing, delegation, and the fused commands (upstreams doubled)."""

from __future__ import annotations

import argparse
import json

import pytest

from rl_crypto import cli


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_parser_accepts_unified_flags():
    args = cli.build_parser().parse_args(["unified", "ETH/USD", "--json", "--lookback-hours", "3"])
    assert args.command == "unified"
    assert args.ticker == "ETH/USD"
    assert args.json is True
    assert args.lookback_hours == 3


def test_dataset_requires_window():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["dataset", "ETH/USD"])


def test_market_delegates_remaining_args(monkeypatch):
    market_cli = pytest.importorskip("tickerkit.cli")
    calls: list[list[str]] = []
    monkeypatch.setattr(market_cli, "main", lambda argv: calls.append(list(argv)) or 0)

    rc = cli.cmd_market(argparse.Namespace(ticker="BTC", rest=["--report", "--frames", "ohlcv"]))
    assert rc == 0
    assert calls == [["BTC", "--report", "--frames", "ohlcv"]]


def test_news_delegates_as_pull(monkeypatch):
    news_cli = pytest.importorskip("ticker_news_signals.cli")
    calls: list[list[str]] = []
    monkeypatch.setattr(news_cli, "main", lambda argv: calls.append(list(argv)) or 0)

    rc = cli.cmd_news(argparse.Namespace(ticker="ETH/USD", rest=["--append", "--output", "s.jsonl"]))
    assert rc == 0
    assert calls == [["pull", "--ticker", "ETH/USD", "--append", "--output", "s.jsonl"]]


def test_unified_prints_json(monkeypatch, capsys):
    from rl_crypto import config as config_mod
    from rl_crypto.models import UnifiedRecord

    monkeypatch.setattr(config_mod.Config, "news_config", lambda self: None)
    monkeypatch.setattr(
        "rl_crypto.unify.build_unified",
        lambda *a, **k: UnifiedRecord(ticker="ETH/USD", price=2500.0, market_ok=True),
    )
    rc = cli.main(["unified", "ETH/USD", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ticker"] == "ETH/USD"
    assert payload["market"]["price"] == 2500.0


def test_unified_exit_code_is_nonzero_only_when_both_sides_fail(monkeypatch):
    from rl_crypto import config as config_mod
    from rl_crypto.models import UnifiedRecord

    monkeypatch.setattr(config_mod.Config, "news_config", lambda self: None)
    monkeypatch.setattr(
        "rl_crypto.unify.build_unified",
        lambda *a, **k: UnifiedRecord(ticker="BTC", market_ok=False, news_ok=False),
    )
    assert cli.main(["unified", "BTC"]) == 1


def test_dataset_writes_jsonl(monkeypatch, tmp_path):
    from rl_crypto import config as config_mod
    from rl_crypto.models import Dataset, DatasetRow

    monkeypatch.setattr(config_mod.Config, "market_config", lambda self: None)
    monkeypatch.setattr(config_mod.Config, "news_config", lambda self: None)
    fake = Dataset(
        ticker="ETH/USD",
        interval="1h",
        rows=(DatasetRow(ticker="ETH/USD", timestamp="2026-01-01T00:00:00Z", close=100.0,
                         has_price=True),),
    )
    monkeypatch.setattr("rl_crypto.dataset.build_dataset", lambda *a, **k: fake)

    out = tmp_path / "eth.jsonl"
    rc = cli.main(["dataset", "ETH/USD", "--from", "2026-01-01", "--to", "2026-01-02",
                   "--output", str(out)])
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    assert json.loads(lines[0])["close"] == 100.0


def test_version(capsys):
    assert cli.main(["version"]) == 0
    assert "rl-crypto" in capsys.readouterr().out
