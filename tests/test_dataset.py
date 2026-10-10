"""The pure join: bucketing, flooring, union, and news aggregation."""

from __future__ import annotations

import pytest

from rl_crypto.dataset import interval_seconds, join_buckets, parse_iso


def _candle(ts: str, close: float, **kw) -> dict:
    return {"timestamp": ts, "close": close, **kw}


def test_join_aligns_price_and_news_on_the_same_hour():
    candles = [_candle("2026-01-01T00:00:00Z", 100.0), _candle("2026-01-01T01:00:00Z", 101.0)]
    vectors = [
        {"timestamp": "2026-01-01T00:00:00Z", "sentiment_score": 0.1, "article_count": 2},
        {"timestamp": "2026-01-01T01:00:00Z", "sentiment_score": 0.3, "article_count": 1},
    ]
    rows = join_buckets(candles, vectors, ticker="ETH", interval_seconds=3600)
    assert [r.timestamp for r in rows] == ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]
    assert all(r.has_price and r.has_news for r in rows)
    assert rows[1].close == 101.0
    assert rows[1].sentiment_score == 0.3


def test_join_is_a_union_and_flags_missing_sides():
    candles = [_candle("2026-01-01T00:00:00Z", 100.0)]
    vectors = [{"timestamp": "2026-01-01T05:00:00Z", "sentiment_score": -0.2, "article_count": 1}]
    rows = join_buckets(candles, vectors, ticker="ETH", interval_seconds=3600)
    assert len(rows) == 2
    price_only, news_only = rows
    assert price_only.has_price and not price_only.has_news
    assert price_only.sentiment_score is None
    assert news_only.has_news and not news_only.has_price
    assert news_only.close is None


def test_join_aggregates_multiple_news_vectors_in_a_bucket():
    candles = [_candle("2026-01-01T00:00:00Z", 100.0)]
    vectors = [
        {"timestamp": "2026-01-01T00:10:00Z", "sentiment_score": 0.2, "article_count": 2},
        {"timestamp": "2026-01-01T00:50:00Z", "sentiment_score": 0.4, "article_count": 3,
         "novelty_flag": True},
    ]
    rows = join_buckets(candles, vectors, ticker="ETH", interval_seconds=3600)
    assert len(rows) == 1
    assert rows[0].sentiment_score == pytest.approx(0.3)
    assert rows[0].article_count == 5
    assert rows[0].novelty_flag is True


def test_join_floors_daily_candles_and_hourly_news_into_one_bucket():
    candles = [_candle("2026-01-01T00:00:00Z", 100.0)]
    vectors = [
        {"timestamp": "2026-01-01T03:00:00Z", "sentiment_score": 0.2, "article_count": 1},
        {"timestamp": "2026-01-01T18:00:00Z", "sentiment_score": 0.6, "article_count": 1},
    ]
    rows = join_buckets(candles, vectors, ticker="ETH", interval_seconds=interval_seconds("1d"))
    assert len(rows) == 1
    assert rows[0].close == 100.0
    assert rows[0].sentiment_score == 0.4
    assert rows[0].article_count == 2


def test_join_skips_unparseable_timestamps():
    candles = [{"timestamp": "not-a-date", "close": 1.0}, _candle("2026-01-01T00:00:00Z", 2.0)]
    rows = join_buckets(candles, [], ticker="ETH", interval_seconds=3600)
    assert len(rows) == 1
    assert rows[0].close == 2.0


def test_parse_iso_and_interval_seconds():
    parsed = parse_iso("2026-01-01T00:00:00Z")
    assert parsed is not None and parsed.hour == 0
    assert parse_iso("") is None
    assert parse_iso(None) is None
    assert interval_seconds("1d") == 86400
    assert interval_seconds("4h") == 14400
    assert interval_seconds("nonsense") == 3600  # documented fallback
