"""Typed models for the unified RL dataset.

Two data-carrying types, both with JSON-ready ``to_dict`` (plain str/int/float/
bool/list/dict/None — no datetimes, no pandas):

- :class:`UnifiedRecord` — a single-ticker, point-in-time view that fuses one
  tickerkit market snapshot with one ticker-news-signals pull. This is the
  "one call, everything known about this ticker right now" shape.
- :class:`DatasetRow` — one per-bucket row of the training/backtest dataset:
  the market OHLCV bar joined to the news-sentiment vector for the same UTC
  interval. :class:`Dataset` bundles the rows with the request echo and the
  per-side notes, so a caller can always tell what was joined and what was
  missing.

Nothing here fabricates a value: an unavailable measurement is ``None`` (with a
``has_*`` flag on dataset rows and an explanatory ``notes`` entry on both
types), never a fabricated ``0``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "rl-crypto/1"


# --------------------------------------------------------------------------- helpers


def utcnow() -> datetime:
    """Aware UTC now."""
    return datetime.now(UTC)


def to_iso(value: datetime | None) -> str | None:
    """Serialise an aware datetime to the ``YYYY-MM-DDTHH:MM:SSZ`` form both
    upstream projects use, and echo ``None`` unchanged."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def finite(value: Any) -> float | None:
    """Coerce to a finite float or ``None`` (rejecting NaN/inf and bools)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# --------------------------------------------------------------------------- unified


@dataclass(slots=True)
class UnifiedRecord:
    """One ticker, one instant: the market snapshot and the news pull fused.

    The ``market_*`` fields come from tickerkit; the ``news_*``/sentiment fields
    from ticker-news-signals. ``market_ok``/``news_ok`` say which side answered,
    and ``notes`` carries the tolerated failure of any side that did not.
    """

    ticker: str
    query: str = ""
    resolved_symbol: str | None = None
    resolved_name: str | None = None
    generated_at: str | None = None

    # -- market (tickerkit) --
    price: float | None = None
    consensus_price: float | None = None
    aggregated_price: float | None = None
    change_24h_pct: float | None = None
    volume_24h: float | None = None
    market_cap: float | None = None
    venue_count: int = 0
    market_ok: bool = False
    market_errors: int = 0

    # -- news (ticker-news-signals) --
    lookback_hours: int = 1
    signal_hours: int = 0
    latest_sentiment: float | None = None
    latest_sentiment_at: str | None = None
    mean_sentiment: float | None = None
    articles: int = 0
    novelty_flag: bool = False
    news_ok: bool = False
    news_errors: int = 0

    notes: tuple[str, ...] = ()

    # -- ML feature addition (Tier 2 outlook; None until attached) --
    # Set by ``rl_crypto.ml.serve.attach_predictions``: the fused Tier-1/Tier-2
    # predictions as a JSON-ready dict. Kept as an opaque dict here so the core
    # models module need not depend on the ML layer.
    predictions: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (stable key order, no datetimes)."""
        return {
            "schema": SCHEMA_VERSION,
            "ticker": self.ticker,
            "query": self.query,
            "resolved": {"symbol": self.resolved_symbol, "name": self.resolved_name},
            "generated_at": self.generated_at,
            "market": {
                "ok": self.market_ok,
                "price": self.price,
                "consensus_price": self.consensus_price,
                "aggregated_price": self.aggregated_price,
                "change_24h_pct": self.change_24h_pct,
                "volume_24h": self.volume_24h,
                "market_cap": self.market_cap,
                "venue_count": self.venue_count,
                "errors": self.market_errors,
            },
            "news": {
                "ok": self.news_ok,
                "lookback_hours": self.lookback_hours,
                "signal_hours": self.signal_hours,
                "latest_sentiment": self.latest_sentiment,
                "latest_sentiment_at": self.latest_sentiment_at,
                "mean_sentiment": self.mean_sentiment,
                "articles": self.articles,
                "novelty_flag": self.novelty_flag,
                "errors": self.news_errors,
            },
            "notes": list(self.notes),
            "predictions": self.predictions,
        }

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)


# --------------------------------------------------------------------------- dataset


@dataclass(slots=True)
class DatasetRow:
    """One UTC interval of the joined dataset.

    Price columns are the tickerkit consensus bar (``None`` for an interval with
    no market data); sentiment columns are the aggregated ticker-news-signals
    vector (``None``/``0`` for an interval with no news). The ``has_*`` flags
    make each side's absence explicit.
    """

    ticker: str
    timestamp: str
    close: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    sentiment_score: float | None = None
    article_count: int = 0
    novelty_flag: bool = False
    has_price: bool = False
    has_news: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "close": self.close,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "sentiment_score": self.sentiment_score,
            "article_count": self.article_count,
            "novelty_flag": self.novelty_flag,
            "has_price": self.has_price,
            "has_news": self.has_news,
        }


@dataclass(slots=True)
class Dataset:
    """The joined per-interval dataset plus the request echo and per-side notes."""

    ticker: str
    interval: str = "1h"
    start: str | None = None
    end: str | None = None
    rows: tuple[DatasetRow, ...] = ()
    notes: tuple[str, ...] = ()
    generated_at: str | None = None

    @property
    def price_rows(self) -> int:
        return sum(1 for r in self.rows if r.has_price)

    @property
    def news_rows(self) -> int:
        return sum(1 for r in self.rows if r.has_news)

    @property
    def joined_rows(self) -> int:
        return sum(1 for r in self.rows if r.has_price and r.has_news)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "ticker": self.ticker,
            "interval": self.interval,
            "start": self.start,
            "end": self.end,
            "generated_at": self.generated_at,
            "coverage": {
                "rows": len(self.rows),
                "price_rows": self.price_rows,
                "news_rows": self.news_rows,
                "joined_rows": self.joined_rows,
            },
            "rows": [r.to_dict() for r in self.rows],
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)
