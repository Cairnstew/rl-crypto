"""The unification seam, exercised with upstream doubles (no network)."""

from __future__ import annotations

import pytest

from rl_crypto import unify
from rl_crypto.unify import build_unified

from .conftest import make_signal_result, make_snapshot, make_vector


def test_build_unified_fuses_both_sides(monkeypatch):
    monkeypatch.setattr(unify, "fetch_market", lambda *a, **k: make_snapshot(price=2500.0))
    monkeypatch.setattr(
        unify,
        "fetch_news",
        lambda *a, **k: make_signal_result(
            vectors=[
                make_vector("2026-09-27T18:00:00Z", 0.1),
                make_vector("2026-09-27T19:00:00Z", 0.5, novelty=True),
            ]
        ),
    )

    record = build_unified("ETH/USD", lookback_hours=3)
    assert record.market_ok and record.news_ok
    assert record.price == 2500.0
    assert record.resolved_name == "Ethereum"
    assert record.signal_hours == 2
    assert record.latest_sentiment == 0.5
    assert record.latest_sentiment_at == "2026-09-27T19:00:00Z"
    assert record.mean_sentiment == pytest.approx(0.3)
    assert record.novelty_flag is True
    assert record.notes == ()


def test_build_unified_tolerates_a_failing_market(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ccxt exploded")

    monkeypatch.setattr(unify, "fetch_market", boom)
    monkeypatch.setattr(unify, "fetch_news", lambda *a, **k: make_signal_result(
        vectors=[make_vector("2026-09-27T19:00:00Z", 0.4)]
    ))

    record = build_unified("ETH/USD")
    assert record.market_ok is False
    assert record.news_ok is True
    assert record.price is None
    assert record.latest_sentiment == 0.4
    assert record.notes and "market: RuntimeError" in record.notes[0]


def test_build_unified_tolerates_a_failing_news_side(monkeypatch):
    monkeypatch.setattr(unify, "fetch_market", lambda *a, **k: make_snapshot(price=10.0))

    def boom(*a, **k):
        raise ValueError("feed down")

    monkeypatch.setattr(unify, "fetch_news", boom)
    record = build_unified("ETH/USD")
    assert record.market_ok is True and record.news_ok is False
    assert record.price == 10.0
    assert record.latest_sentiment is None
    assert any("news: ValueError" in n for n in record.notes)


def test_build_unified_both_failing_is_an_empty_record(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(unify, "fetch_market", boom)
    monkeypatch.setattr(unify, "fetch_news", boom)
    record = build_unified("BTC")
    assert record.market_ok is False and record.news_ok is False
    assert len(record.notes) == 2


def test_build_unified_records_tolerated_error_counts(monkeypatch):
    monkeypatch.setattr(unify, "fetch_market", lambda *a, **k: make_snapshot(errors=2))
    monkeypatch.setattr(unify, "fetch_news", lambda *a, **k: make_signal_result(errors=1))
    record = build_unified("ETH/USD")
    assert record.market_errors == 2
    assert record.news_errors == 1
