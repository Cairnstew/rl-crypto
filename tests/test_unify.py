"""The unification seam, exercised with upstream doubles (no network)."""

from __future__ import annotations

import pytest

from rl_crypto import unify
from rl_crypto.unify import build_unified

from .conftest import make_signal_result, make_snapshot, make_vector


def _stub_tickerkit(monkeypatch, **attrs) -> None:
    """Inject a fake ``tickerkit`` module and neutralise ``asyncio.run``.

    ``fetch_markets`` does ``asyncio.run(aget_tickers(...))``. Replacing
    ``run`` with an identity means the stub only has to *return* the value, so
    these tests exercise the batching/mapping logic without real coroutines.
    """
    import sys

    monkeypatch.setitem(sys.modules, "tickerkit", type("M", (), attrs))
    monkeypatch.setattr(unify.asyncio, "run", lambda value: value, raising=False)


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


def test_build_unified_reuses_an_injected_snapshot_and_news_result(monkeypatch):
    """Pre-fetched market/news results are used instead of refetching."""
    def boom(*a, **k):
        raise AssertionError("must not refetch when results are injected")

    monkeypatch.setattr(unify, "fetch_market", boom)
    monkeypatch.setattr(unify, "fetch_news", boom)
    record = build_unified(
        "ETH/USD",
        snapshot=make_snapshot(price=99.0),
        news_result=make_signal_result(vectors=[make_vector("2026-01-01T00:00:00Z", 0.5)]),
    )
    assert record.market_ok is True
    assert record.price == 99.0
    assert record.news_ok is True
    assert record.articles == 1


def test_fetch_markets_batches_once_and_maps_by_ticker(monkeypatch):
    """The whole watchlist is fetched in ONE rate-limited batch call."""
    calls: list = []

    def fake_batch(tickers, **kwargs):
        calls.append(list(tickers))
        return [make_snapshot(symbol=t) for t in tickers]

    _stub_tickerkit(monkeypatch, aget_tickers=fake_batch)
    out = unify.fetch_markets(["BTC/USD", "ETH/USD"])
    assert calls == [["BTC/USD", "ETH/USD"]]
    assert set(out) == {"BTC/USD", "ETH/USD"}
    assert out["BTC/USD"].identity.symbol == "BTC/USD"


def test_fetch_markets_skips_whole_snapshot_failures(monkeypatch):
    def fake_batch(tickers, **kwargs):
        return [make_snapshot(symbol=tickers[0]), RuntimeError("snapshot blew up")]

    _stub_tickerkit(monkeypatch, aget_tickers=fake_batch)
    out = unify.fetch_markets(["BTC/USD", "ETH/USD"])
    assert set(out) == {"BTC/USD"}


def test_fetch_markets_on_empty_input_never_calls_upstream(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not call upstream for an empty watchlist")

    _stub_tickerkit(monkeypatch, aget_tickers=boom)
    assert unify.fetch_markets([]) == {}


def test_build_unified_many_returns_one_record_per_ticker(monkeypatch):
    monkeypatch.setattr(
        unify,
        "fetch_markets",
        lambda ts, **k: {t: make_snapshot(symbol=t, price=float(i + 1)) for i, t in enumerate(ts)},
    )
    monkeypatch.setattr(
        unify,
        "fetch_news",
        lambda t, **k: make_signal_result(symbol=t, vectors=[make_vector("2026-01-01T00:00:00Z", 0.1)]),
    )
    tickers = ["BTC/USD", "ETH/USD", "SOL/USD"]
    records = unify.build_unified_many(tickers)
    assert [r.ticker for r in records] == tickers
    assert [r.price for r in records] == [1.0, 2.0, 3.0]
    assert all(r.market_ok and r.news_ok for r in records)


def test_build_unified_many_notes_a_missing_snapshot(monkeypatch):
    monkeypatch.setattr(unify, "fetch_markets", lambda ts, **k: {})
    monkeypatch.setattr(unify, "fetch_news", lambda t, **k: make_signal_result(symbol=t))
    records = unify.build_unified_many(["BTC/USD"])
    assert len(records) == 1
    assert any("no snapshot returned" in n for n in records[0].notes)
