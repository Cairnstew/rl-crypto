#!/usr/bin/env python3
"""Probe candidate tickers for pipeline eligibility (not a pipeline stage).

For each candidate this measures the three things the scheduled pipeline
actually needs, cheaply and read-only (nothing is written to ``data/``):

* **market** — how many clean 1h candles a 60-day window yields, plus a live
  snapshot (price + whether the snapshot itself resolved),
* **news** — whether ticker-news-signals resolves the symbol to real search
  terms, and how many headlines one live pull returns,
* **errors** — what failed, so a thin ticker is visibly thin rather than
  silently empty.

    uv run python tools/probe_tickers.py --tickers BTC/USD SOL/USD --history-days 60
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import timedelta

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from rl_crypto.models import utcnow


def probe_market(ticker: str, days: int, timeout: float) -> dict:
    """Count usable candles over a ``days``-long 1h window + one live snapshot."""
    from rl_crypto.dataset import fetch_prices

    end = utcnow()
    start = end - timedelta(days=days)
    candles, notes = fetch_prices(
        ticker, start=start, end=end, timeframe="1h", timeout=timeout
    )
    out: dict = {"candles": len(candles), "notes": list(notes)}
    if candles:
        closes = [c.get("close") for c in candles if c.get("close")]
        out["first"] = candles[0].get("timestamp")
        out["last"] = candles[-1].get("timestamp")
        out["last_close"] = closes[-1] if closes else None
    # live snapshot (identity + price), same call the live recorder makes
    try:
        from tickerkit import aget_ticker

        snap = asyncio.run(aget_ticker(ticker))
        out["snapshot_price"] = snap.market.price
        out["resolved_symbol"] = getattr(snap.identity, "symbol", None)
    except Exception as exc:
        out["snapshot_error"] = f"{type(exc).__name__}: {exc}"[:120]
    return out


def probe_news(ticker: str, lookback_hours: int) -> dict:
    """Resolve search terms and count headlines from one live news pull."""
    from rl_crypto.unify import fetch_news

    try:
        from ticker_news_signals.resolve import resolve_ticker

        res = resolve_ticker(ticker)
        out = {
            "resolved_symbol": res.symbol,
            "name": res.name,
            "terms": list(res.terms)[:4],
        }
    except Exception as exc:
        return {"resolve_error": f"{type(exc).__name__}: {exc}"[:120]}

    try:
        result = fetch_news(ticker, lookback_hours=lookback_hours)
        vectors = getattr(result, "vectors", None) or []
        out["hour_vectors"] = len(vectors)
        out["articles"] = len(getattr(result, "articles", ()) or ())
        # Non-empty hourly buckets are what actually reach the feature join.
        populated = [v for v in vectors if (getattr(v, "article_count", 0) or 0) > 0]
        out["populated_hours"] = len(populated)
        errs = getattr(getattr(result, "data_quality", None), "errors", ()) or ()
        out["news_errors"] = len(errs)
    except Exception as exc:
        out["news_error"] = f"{type(exc).__name__}: {exc}"[:120]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tickers",
        nargs="+",
        default=["BTC/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "LINK/USD"],
    )
    ap.add_argument("--history-days", type=int, default=60)
    ap.add_argument("--lookback-hours", type=int, default=3)
    ap.add_argument("--fetch-timeout", type=float, default=180.0)
    args = ap.parse_args(argv)

    for ticker in args.tickers:
        print(f"=== {ticker} ===", flush=True)
        market = probe_market(ticker, args.history_days, args.fetch_timeout)
        news = probe_news(ticker, args.lookback_hours)
        print(f"  market : {market}", flush=True)
        print(f"  news   : {news}", flush=True)
    os._exit(0)  # skip atexit/thread teardown (ccxt keeps sockets alive)


if __name__ == "__main__":
    raise SystemExit(main())
