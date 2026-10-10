"""Shared fixtures: lightweight stand-ins for the upstream result objects.

The unified builders only read a handful of attributes off a tickerkit snapshot
and a ticker-news-signals result, so the tests use :class:`types.SimpleNamespace`
doubles rather than importing (or mocking HTTP for) the upstreams. This keeps the
offline suite fast and free of ccxt/gnews/pandas.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest


def make_snapshot(
    *,
    symbol: str = "ETH/USD",
    name: str = "Ethereum",
    price: float = 2500.0,
    errors: int = 0,
    venues: int = 3,
) -> SimpleNamespace:
    """A tickerkit-``TickerSnapshot``-shaped double."""
    market = SimpleNamespace(
        price=price,
        consensus_price=price,
        aggregated_price=price - 1.0,
        change_24h_pct=1.5,
        volume_24h=1_000_000.0,
        market_cap=300_000_000_000.0,
    )
    return SimpleNamespace(
        market=market,
        identity=SimpleNamespace(symbol=symbol, name=name),
        venues=tuple(range(venues)),
        data_quality=SimpleNamespace(errors=tuple(range(errors))),
    )


def make_vector(
    timestamp: str,
    score: float,
    *,
    articles: int = 1,
    novelty: bool = False,
) -> SimpleNamespace:
    """A ticker-news-signals-``SignalVector``-shaped double."""
    return SimpleNamespace(
        timestamp=timestamp,
        sentiment_score=score,
        article_count=articles,
        novelty_flag=novelty,
        to_dict=lambda: {
            "ticker": "ETH_USD",
            "timestamp": timestamp,
            "sentiment_score": score,
            "article_count": articles,
            "novelty_flag": novelty,
        },
    )


def make_signal_result(
    *,
    symbol: str = "ETH_USD",
    name: str = "Ethereum",
    vectors: list | None = None,
    errors: int = 0,
) -> SimpleNamespace:
    """A ticker-news-signals-``SignalResult``-shaped double."""
    return SimpleNamespace(
        ticker=symbol,
        resolved_symbol=symbol,
        resolved_name=name,
        vectors=list(vectors or []),
        data_quality=SimpleNamespace(errors=tuple(range(errors))),
    )


@pytest.fixture
def snapshot() -> SimpleNamespace:
    return make_snapshot()


@pytest.fixture
def signal_result() -> SimpleNamespace:
    return make_signal_result(
        vectors=[
            make_vector("2026-09-27T18:00:00Z", 0.1, articles=2),
            make_vector("2026-09-27T19:00:00Z", 0.5, articles=3, novelty=True),
        ]
    )


@pytest.fixture
def utcnow() -> datetime:
    return datetime(2026, 9, 27, 20, 0, tzinfo=UTC)
