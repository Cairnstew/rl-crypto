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
from collections.abc import Sequence
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


def fetch_markets(
    tickers: Sequence[str], *, config: Any = None, **overrides: Any
) -> dict[str, Any]:
    """Fetch many tickerkit snapshots in one rate-limited batch.

    Uses :func:`tickerkit.aget_tickers`, which shares one rate limiter, one
    circuit breaker and one connection pool across the whole batch: adding
    tickers adds concurrency, never requests-per-minute. Fetching the watchlist
    with N separate :func:`fetch_market` calls instead would build N limiters
    and ask the free-tier providers for N times their documented rate — the
    exact overshoot that earns an IP-level block.

    Returns a ``{ticker: snapshot}`` mapping; a ticker whose snapshot failed
    outright is simply absent (per-provider failures are recorded *inside* the
    snapshot, as always, and do not raise).
    """
    if not tickers:
        return {}
    from tickerkit import aget_tickers

    snapshots = asyncio.run(
        aget_tickers(list(tickers), config=config, return_exceptions=True, **overrides)
    )
    out: dict[str, Any] = {}
    for ticker, snap in zip(tickers, snapshots, strict=False):
        if isinstance(snap, BaseException):
            continue
        out[ticker] = snap
    return out


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
    snapshot: Any = None,
    news_result: Any = None,
    **market_overrides: Any,
) -> UnifiedRecord:
    """Fuse a market snapshot and a news pull for ``ticker`` into one record.

    Neither side is required to succeed. A raising side contributes its
    ``*_error`` note and leaves its fields ``None``; the other side is returned
    intact.

    ``snapshot``/``news_result`` let a caller inject results it already has —
    the multi-ticker path fetches every market snapshot in one rate-limited
    batch (see :func:`fetch_markets`) and passes each one in, so no ticker is
    ever fetched twice.
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
        snap = snapshot if snapshot is not None else fetch_market(
            ticker, config=market_config, **market_overrides
        )
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
        result = news_result if news_result is not None else fetch_news(
            ticker, lookback_hours=lookback_hours, config=news_config
        )
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


def build_unified_many(
    tickers: Sequence[str],
    *,
    lookback_hours: int = 1,
    market_config: Any = None,
    news_config: Any = None,
    **market_overrides: Any,
) -> list[UnifiedRecord]:
    """Build one unified record per ticker, sharing a single market batch.

    All market snapshots are fetched in **one** rate-limited batch and then
    handed to :func:`build_unified`, so N tickers cost one shared request
    budget rather than N private ones.

    News has no upstream batch API, so each ticker gets its own pull, run
    serially and isolated: a provider that fails for one asset cannot take the
    rest of the watchlist down with it. One record is always returned per input
    ticker, in order.
    """
    snapshots = fetch_markets(
        tickers, config=market_config, **market_overrides
    )  # one shared limiter/breaker/pool for the whole batch

    records: list[UnifiedRecord] = []
    for ticker in tickers:
        snapshot = snapshots.get(ticker)
        if snapshot is None:
            # Only a whole-snapshot failure lands here; note it rather than
            # silently emitting a record with no market side and no reason.
            record = build_unified(
                ticker,
                lookback_hours=lookback_hours,
                news_config=news_config,
                **market_overrides,
            )
            record.notes = (
                *record.notes,
                "market: no snapshot returned by batch fetch",
            )
            records.append(record)
            continue
        records.append(
            build_unified(
                ticker,
                lookback_hours=lookback_hours,
                news_config=news_config,
                snapshot=snapshot,
                **market_overrides,
            )
        )
    return records


__all__ = [
    "build_unified",
    "build_unified_many",
    "fetch_market",
    "fetch_markets",
    "fetch_news",
]
