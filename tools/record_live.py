#!/usr/bin/env python3
"""High-frequency live recorder: one live record per run, no retraining.

The heavier history-archive + training work happens on the pipeline workflow's
slower cadence; this tool is the *frequent* live path:

* load the latest committed Tier-1 bundle,
* fetch a short recent market window (just enough for Tier-0 features),
* take a live unified snapshot — market **and live news**,
* compose the Tier-2 outlook and **append** one record to
  ``data/<slug>/live.jsonl`` (capped by ``--live-cap``).

Run by ``.github/workflows/live.yml`` on a frequent cron (default every 30
minutes; each run is cheap — no training, a small window fetch, one record).

    uv run python tools/record_live.py --ticker ETH/USD
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from rl_crypto.cache import LIVE_CAP_DEFAULT, append_capped, live_path
from rl_crypto.dataset import fetch_prices
from rl_crypto.ml import attach_predictions
from rl_crypto.ml.serve import outlook_for_rows
from rl_crypto.ml.store import default_bundle_path, load_bundle
from rl_crypto.models import DatasetRow, to_iso, utcnow
from rl_crypto.unify import build_unified

ROOT = Path(__file__).resolve().parents[1]


def record_live(args: argparse.Namespace) -> tuple[dict[str, Any], Path, int]:
    """Fetch one live outlook and append it; return ``(record_dict, path, code)``."""
    ticker = args.ticker
    timeline = utcnow()
    end = timeline
    start = end - timedelta(hours=max(1, args.window))

    bundle = load_bundle(default_bundle_path(ticker, directory=Path(args.root) / "models"))

    # Short, bounded market window — just enough for causal Tier-0 features.
    candles, _ = fetch_prices(
        ticker, start=start, end=end, timeframe=args.timeframe, timeout=args.fetch_timeout
    )
    rows = tuple(
        DatasetRow(
            ticker=ticker,
            timestamp=str(candle.get("timestamp") or ""),
            open=candle.get("open"),
            high=candle.get("high"),
            low=candle.get("low"),
            close=candle.get("close"),
            volume=candle.get("volume"),
            has_price=True,
            has_news=False,
        )
        for candle in candles
        if candle.get("timestamp")
    )

    outlook = outlook_for_rows(rows, bundle, ticker=ticker)
    record = build_unified(ticker, lookback_hours=args.lookback_hours)
    attach_predictions(record, outlook)

    dest = append_capped(live_path(args.root, ticker), record.to_dict(), cap=args.live_cap)
    payload = record.to_dict()
    payload["_generated_at"] = to_iso(timeline)
    ok = outlook.feature_timestamp is not None or record.market_ok or record.news_ok
    return payload, dest, 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="ETH/USD", help="asset to record live data for")
    parser.add_argument("--timeframe", default="1h", help="market interval for the feature window")
    parser.add_argument(
        "--window", type=int, default=48, help="hours of market history for features"
    )
    parser.add_argument("--lookback-hours", type=int, default=1, help="live news window")
    parser.add_argument(
        "--fetch-timeout", type=float, default=120.0, help="per-fetch wall-clock cap (s)"
    )
    parser.add_argument(
        "--live-cap", type=int, default=LIVE_CAP_DEFAULT, help="max live.jsonl records"
    )
    parser.add_argument("--root", default=str(ROOT), help="repo root for models/ and data/")
    args = parser.parse_args(argv)

    exit_code = 1
    try:
        payload, dest, exit_code = record_live(args)
        outlook = payload["predictions"]
        print(
            f"live {args.ticker} @ {payload['_generated_at']}"
            f" | signal={outlook['signal']} conf={outlook['confidence']}"
            f" | mkt={payload['market']['ok']} news={payload['news']['ok']}"
            f" price={payload['market']['price']}"
        )
        print(f"  appended -> {dest}")
    except Exception as exc:  # tolerated: report and still exit promptly
        print(f"record_live failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 1

    sys.stdout.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())