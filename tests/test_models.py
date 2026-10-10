"""Model serialisation and coverage arithmetic."""

from __future__ import annotations

import json

from rl_crypto.models import SCHEMA_VERSION, Dataset, DatasetRow, UnifiedRecord, finite, to_iso


def test_unified_record_to_dict_is_json_ready():
    record = UnifiedRecord(
        ticker="ETH_USD",
        query="ETH/USD",
        resolved_symbol="ETH/USD",
        resolved_name="Ethereum",
        generated_at="2026-09-27T20:00:00Z",
        price=2500.0,
        latest_sentiment=0.42,
        news_ok=True,
        market_ok=True,
    )
    payload = record.to_dict()
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["market"]["price"] == 2500.0
    assert payload["news"]["latest_sentiment"] == 0.42
    # Round-trips through json with no bespoke encoder.
    assert json.loads(json.dumps(payload))["ticker"] == "ETH_USD"


def test_unified_record_marks_unavailable_sides():
    record = UnifiedRecord(ticker="BTC", market_ok=False, news_ok=False, notes=("market: boom",))
    payload = record.to_dict()
    assert payload["market"]["ok"] is False
    assert payload["news"]["ok"] is False
    assert payload["market"]["price"] is None
    assert payload["notes"] == ["market: boom"]


def test_dataset_coverage_counts():
    rows = (
        DatasetRow(ticker="ETH", timestamp="2026-01-01T00:00:00Z", close=1.0, has_price=True),
        DatasetRow(ticker="ETH", timestamp="2026-01-01T01:00:00Z", sentiment_score=0.1, has_news=True),
        DatasetRow(
            ticker="ETH",
            timestamp="2026-01-01T02:00:00Z",
            close=2.0,
            sentiment_score=0.2,
            has_price=True,
            has_news=True,
        ),
    )
    dataset = Dataset(ticker="ETH", rows=rows)
    assert dataset.price_rows == 2
    assert dataset.news_rows == 2
    assert dataset.joined_rows == 1
    assert dataset.to_dict()["coverage"]["rows"] == 3


def test_to_iso_normalises_naive_and_aware():
    from datetime import UTC, datetime

    assert to_iso(None) is None
    assert to_iso(datetime(2026, 1, 1, tzinfo=UTC)) == "2026-01-01T00:00:00Z"
    assert to_iso(datetime(2026, 1, 1)) == "2026-01-01T00:00:00Z"


def test_finite_rejects_non_finite():
    assert finite(1.5) == 1.5
    assert finite(None) is None
    assert finite(True) is None
    assert finite(float("nan")) is None
    assert finite(float("inf")) is None
    assert finite("nope") is None
