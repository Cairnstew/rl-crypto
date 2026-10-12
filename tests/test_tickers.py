"""Tests for the tracked ticker set and its argument parser."""

from __future__ import annotations

from rl_crypto.tickers import TRACKED_TICKERS, parse_tickers


def test_default_is_the_tracked_watchlist():
    assert parse_tickers(None) == TRACKED_TICKERS


def test_watchlist_is_multi_asset_and_normalised():
    assert len(TRACKED_TICKERS) >= 5
    assert all("/" in ticker for ticker in TRACKED_TICKERS)
    assert len(set(TRACKED_TICKERS)) == len(TRACKED_TICKERS)


def test_comma_and_space_separated_strings_parse():
    assert parse_tickers("BTC/USD,ETH/USD") == ("BTC/USD", "ETH/USD")
    assert parse_tickers("BTC/USD ETH/USD") == ("BTC/USD", "ETH/USD")


def test_iterables_and_duplicates_fold():
    assert parse_tickers(["SOL/USD", "BTC/USD", "SOL/USD"]) == ("SOL/USD", "BTC/USD")
    assert parse_tickers(["BTC/USD,ETH/USD", "BTC/USD"]) == ("BTC/USD", "ETH/USD")


def test_blank_entries_are_dropped():
    assert parse_tickers(" , ") == ()
    assert parse_tickers([]) == ()
