"""The unification seam: one ticker, both pipelines, one record.

The two upstreams answer different questions about the same asset:

* **tickerkit** — *what is the market doing?* A provenance-tracked snapshot:
  consensus price across venues, 24h change, volume, market cap, plus OHLCV,
  order books, derivatives and on-chain data.
* **ticker-news-signals** — *what is the news saying?* Per-(ticker, UTC hour)
  sentiment vectors scored locally from many keyless feeds.

This module runs both and fuses them into a :class:`UnifiedRecord`. It is
**partial-failure tolerant**, exactly like the upstreams: if one side raises,
the other side's data is still returned and the failure is recorded in
``notes``. Nothing is fabricated — an unavailable side leaves its fields at
``None`` and flips its ``*_ok`` flag to ``False``.

Both upstreams are imported lazily inside the functions so that importing
``rl_crypto`` never pays for ccxt/pandas/gnews unless a fetch is actually made.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .models import UnifiedRecord, finite, to_iso, utcnow


def fetch_market(ticker: str, *, config: Any = None, **overrides: Any) -> Any:
    """Run one tickerkit snapshot fetch for ``ticker`` and return the snapshot.

    ``overrides`` are forwarded verbatim to
    :func:`tickerkit.aget_ticker` (``quote_currency``, ``exchanges``,
    ``coingecko_id``, …). Synchronous wrapper: the CLI is not inside an event
    loop, so :func:`asyncio.run` is the right primitive here.
    """
    from tickerkit import aget_ticker

    return asyncio.run(aget_ticker(ticker, config=config, **overrides))


def fetch_news(ticker: str, *, lookback_hours: int = 1, config: Any = None) -> Any:
    """Run one ticker-news-signals pull and return the :class:`SignalResult`."""
    from ticker_news_signals.pipeline import fetch_result

    return fetch_result(ticker, lookback_hours=lookback_hours, config=config)


def build_unified(
    ticker: str,
    *,
    lookback_hours: int = 1,
    market_config: Any = None,
    news_config: Any = None,
    **market_overrides: Any,
) -> UnifiedRecord:
    """Fuse a market snapshot and a news pull for ``ticker`` into one record.

    Neither side is required to succeed. A raising side contributes its
    ``*_error`` note and leaves its fields ``None``; the other side is returned
    intact.
    """
    record = UnifiedRecord(
        ticker=ticker,
        query=ticker,
        generated_at=to_iso(utcnow()),
        lookback_hours=lookback_hours,
    )
    notes: list[str] = []

    # -- market (tickerkit) --
    try:
        snap = fetch_market(ticker, config=market_config, **market_overrides)
        market = snap.market
        record.market_ok = True
        record.resolved_symbol = snap.identity.symbol
        record.resolved_name = snap.identity.name
        record.price = finite(market.price)
        record.consensus_price = finite(market.consensus_price)
        record.aggregated_price = finite(market.aggregated_price)
        record.change_24h_pct = finite(market.change_24h_pct)
        record.volume_24h = finite(market.volume_24h)
        record.market_cap = finite(market.market_cap)
        record.venue_count = len(snap.venues)
        record.market_errors = len(snap.data_quality.errors)
    except Exception as exc:
        notes.append(f"market: {type(exc).__name__}: {exc}")

    # -- news (ticker-news-signals) --
    try:
        result = fetch_news(ticker, lookback_hours=lookback_hours, config=news_config)
        record.news_ok = True
        if record.resolved_symbol is None and result.resolved_symbol:
            record.resolved_symbol = result.resolved_symbol
        if record.resolved_name is None and result.resolved_name:
            record.resolved_name = result.resolved_name
        vectors = result.vectors
        record.signal_hours = len(vectors)
        record.articles = sum(v.article_count for v in vectors)
        record.novelty_flag = any(v.novelty_flag for v in vectors)
        record.news_errors = len(result.data_quality.errors)
        if vectors:
            scores = [v.sentiment_score for v in vectors]
            record.mean_sentiment = finite(sum(scores) / len(scores))
            latest = max(vectors, key=lambda v: v.timestamp)
            record.latest_sentiment = finite(latest.sentiment_score)
            record.latest_sentiment_at = latest.timestamp
    except Exception as exc:
        notes.append(f"news: {type(exc).__name__}: {exc}")

    record.notes = tuple(notes)
    return record


__all__ = ["build_unified", "fetch_market", "fetch_news"]
