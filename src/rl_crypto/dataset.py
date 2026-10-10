"""The RL dataset: many hourly rows, market bar joined to news sentiment.

:func:`build_dataset` walks the *history* side of both pipelines — tickerkit's
cross-referenced OHLCV history (``aget_history``) and ticker-news-signals'
Wayback-reconstructed sentiment history (``fetch_history``) — then joins them on
the UTC interval.

The join is deliberately transparent:

* Both sides are floored onto the requested interval, so a ``1d`` market bar and
  every hourly news vector that falls inside that day land in one bucket.
* The bucket set is the **union** of both sides, sorted by time. An interval
  known only to one side is kept, with its other-side fields ``None`` and its
  ``has_price``/``has_news`` flag set accordingly — a sparse news archive shows
  up as ``has_news=False`` rows rather than silently shrinking the dataset.
* News vectors inside a bucket are aggregated (mean sentiment, summed article
  count, OR-ed novelty), mirroring the day/hour bucketing ticker-news-signals
  itself uses.

The pure join (:func:`join_buckets`) takes plain primitives, so it is testable
without any network or upstream objects.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from .models import Dataset, DatasetRow, finite, to_iso, utcnow

#: Seconds per market timeframe, mirroring tickerkit's history intervals.
_INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


def interval_seconds(timeframe: str) -> int:
    """Seconds for a timeframe string, defaulting to one hour for unknown ones."""
    return _INTERVAL_SECONDS.get(timeframe, 3600)


def parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 string (or pass a datetime through) to aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _floor(moment: datetime, seconds: int) -> datetime:
    """Floor an aware datetime onto an ``seconds`` boundary (UTC)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % seconds), tz=UTC)


def join_buckets(
    candles: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    *,
    ticker: str,
    interval_seconds: int = 3600,
) -> tuple[DatasetRow, ...]:
    """Join plain price candles and news vectors into sorted dataset rows.

    Parameters
    ----------
    candles:
        Dicts with a ``timestamp`` (ISO or datetime) and ``close`` plus optional
        ``open``/``high``/``low``/``volume``.
    vectors:
        Canonical news records (``timestamp``, ``sentiment_score``,
        ``article_count``, ``novelty_flag``) as emitted by ticker-news-signals.
    """
    prices: dict[datetime, dict[str, Any]] = {}
    for candle in candles:
        ts = parse_iso(candle.get("timestamp"))
        if ts is None:
            continue
        prices[_floor(ts, interval_seconds)] = candle

    news: dict[datetime, list[dict[str, Any]]] = {}
    for vector in vectors:
        ts = parse_iso(vector.get("timestamp"))
        if ts is None:
            continue
        news.setdefault(_floor(ts, interval_seconds), []).append(vector)

    rows: list[DatasetRow] = []
    for bucket in sorted(set(prices) | set(news)):
        candle = prices.get(bucket)
        bucket_news = news.get(bucket, [])
        scores = [finite(v.get("sentiment_score")) for v in bucket_news]
        scores = [s for s in scores if s is not None]
        rows.append(
            DatasetRow(
                ticker=ticker,
                timestamp=to_iso(bucket) or "",
                close=finite(candle.get("close")) if candle else None,
                open=finite(candle.get("open")) if candle else None,
                high=finite(candle.get("high")) if candle else None,
                low=finite(candle.get("low")) if candle else None,
                volume=finite(candle.get("volume")) if candle else None,
                sentiment_score=(sum(scores) / len(scores)) if scores else None,
                article_count=sum(int(v.get("article_count") or 0) for v in bucket_news),
                novelty_flag=any(bool(v.get("novelty_flag")) for v in bucket_news),
                has_price=candle is not None,
                has_news=bool(bucket_news),
            )
        )
    return tuple(rows)


# --------------------------------------------------------------------------- fetchers


def _candle_dicts(history_result: Any) -> list[dict[str, Any]]:
    """Normalise a tickerkit ``HistoryResult`` to plain candle dicts.

    Prefers the cross-referenced consensus series; falls back to the first raw
    OHLCV series when no consensus could be built.
    """
    candles: list[Any] = []
    consensus = getattr(history_result, "consensus", None)
    if consensus is not None and getattr(consensus, "candles", None):
        candles = list(consensus.candles)
    else:
        for series in getattr(history_result, "series", ()) or ():
            if getattr(series, "candles", None):
                candles = list(series.candles)
                break
    return [
        {
            "timestamp": getattr(c, "timestamp", None),
            "close": getattr(c, "close", None),
            "open": getattr(c, "open", None),
            "high": getattr(c, "high", None),
            "low": getattr(c, "low", None),
            "volume": getattr(c, "volume", None),
        }
        for c in candles
    ]


def fetch_market_history(
    ticker: str,
    *,
    timeframe: str = "1d",
    start: Any = None,
    end: Any = None,
    limit: int | None = None,
    config: Any = None,
    **overrides: Any,
) -> Any:
    """Run one tickerkit history fetch and return the ``HistoryResult``."""
    from tickerkit import aget_history

    return asyncio.run(
        aget_history(
            ticker,
            config=config,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            **overrides,
        )
    )


def fetch_news_history(
    ticker: str,
    *,
    granularity: str = "hour",
    start: Any,
    end: Any,
    config: Any = None,
) -> Any:
    """Run one ticker-news-signals history fetch and return the result.

    ``start``/``end`` may be ISO strings or datetimes; they are normalised to
    aware UTC before the call.
    """
    from ticker_news_signals.history import fetch_history

    start_dt = parse_iso(start) or start
    end_dt = parse_iso(end) or end
    return fetch_history(ticker, granularity=granularity, start=start_dt, end=end_dt, config=config)


def build_dataset(
    ticker: str,
    *,
    start: Any,
    end: Any,
    timeframe: str = "1d",
    granularity: str = "hour",
    limit: int | None = None,
    market_config: Any = None,
    news_config: Any = None,
    **market_overrides: Any,
) -> Dataset:
    """Build the joined per-interval dataset for ``ticker`` over a window.

    Either side may fail; its note is recorded and the other side's rows are
    still returned. The bucket set is the union of both sides.
    """
    seconds = interval_seconds(timeframe)
    notes: list[str] = []
    candles: list[dict[str, Any]] = []
    vectors: list[dict[str, Any]] = []

    try:
        history = fetch_market_history(
            ticker,
            timeframe=timeframe,
            start=parse_iso(start) or start,
            end=parse_iso(end) or end,
            limit=limit,
            config=market_config,
            **market_overrides,
        )
        candles = _candle_dicts(history)
        notes.extend(f"market: {err.message}" for err in _errors(history))
    except Exception as exc:
        notes.append(f"market: {type(exc).__name__}: {exc}")

    try:
        news = fetch_news_history(
            ticker, granularity=granularity, start=start, end=end, config=news_config
        )
        vectors = [v.to_dict() for v in news.vectors]
        notes.extend(f"news: {err.message}" for err in _errors(news))
    except Exception as exc:
        notes.append(f"news: {type(exc).__name__}: {exc}")

    rows = join_buckets(candles, vectors, ticker=ticker, interval_seconds=seconds)
    return Dataset(
        ticker=ticker,
        interval=timeframe,
        start=to_iso(parse_iso(start)) or (str(start) if start else None),
        end=to_iso(parse_iso(end)) or (str(end) if end else None),
        rows=rows,
        notes=tuple(notes),
        generated_at=to_iso(utcnow()),
    )


def _errors(result: Any) -> tuple[Any, ...]:
    """The tolerated-error records from a result's ``data_quality``, if any."""
    quality = getattr(result, "data_quality", None)
    return tuple(getattr(quality, "errors", ()) or ())


__all__ = [
    "build_dataset",
    "fetch_market_history",
    "fetch_news_history",
    "interval_seconds",
    "join_buckets",
    "parse_iso",
]
