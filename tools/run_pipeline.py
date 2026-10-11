#!/usr/bin/env python3
"""Production pipeline run for one ticker: grow the archive, train, push live.

This is the deterministic invocation the scheduled GitHub Actions workflow
(``.github/workflows/pipeline.yml``) runs on a cron and commits back to the repo:

* fetch the **market history** window (tickerkit OHLCV; news history stays off
  until RL training — live news still flows through the unified record),
* **upsert the candles into a growing archive** ``data/<slug>/candles.jsonl`` —
  the archive never shrinks and never duplicates, so the repo accumulates as
  much clean historical data as we can fetch,
* rebuild features + forward targets over the **whole archive**, fit the
  Tier-1 bundle, and cache it under ``models/``,
* take a live unified snapshot (market **and** live news), attach the Tier-2
  outlook, and append it to ``data/<slug>/live.jsonl`` (capped).

Labels are never cached: a row's forward target only becomes defined once its
future bar is archived, and training recomputes them every run against the full
history. Fetches are bounded by ``--fetch-timeout`` per side; a slow/absent
upstream is recorded as a note and the good sides still land. The run always
ends with a hard ``os._exit`` so abandoned timeout workers cannot keep it alive
after the work is flushed.

    uv run python tools/run_pipeline.py --ticker ETH/USD --history-days 60
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from rl_crypto import build_samples
from rl_crypto.cache import (
    append_capped,
    call_with_timeout,
    candle_dicts_from_rows,
    candles_path,
    live_path,
    load_candles,
    merge_candles,
    summary_path,
)
from rl_crypto.dataset import candle_dicts, fetch_market_history, parse_iso
from rl_crypto.export import write_json
from rl_crypto.ml import attach_predictions, outlook_for_rows
from rl_crypto.ml.store import default_bundle_path, save_bundle
from rl_crypto.ml.tier1 import train_tier1
from rl_crypto.models import Dataset, DatasetRow, UnifiedRecord, to_iso, utcnow
from rl_crypto.unify import build_unified

ROOT = Path(__file__).resolve().parents[1]


def build_dataset(  # local, bounded — patched by tests
    ticker: str,
    *,
    start: Any,
    end: Any,
    timeframe: str,
    limit: int | None,
    fetch_timeout: float,
) -> Dataset:
    """Fetch the market-history window (capped) as price rows."""
    notes: list[str] = []

    histories, market_note = call_with_timeout(
        lambda: fetch_market_history(
            ticker,
            timeframe=timeframe,
            start=parse_iso(start) or start,
            end=parse_iso(end) or end,
            limit=limit,
        ),
        timeout=fetch_timeout,
        label="market history",
    )
    if market_note:
        notes.append(market_note)

    rows = [
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
        for candle in (candle_dicts(histories) if histories is not None else [])
        if candle.get("timestamp")
    ]
    return Dataset(
        ticker=ticker,
        interval=timeframe,
        start=to_iso(parse_iso(start)) or (str(start) if start else None),
        end=to_iso(parse_iso(end)) or (str(end) if end else None),
        rows=tuple(rows),
        notes=tuple(notes),
        generated_at=to_iso(utcnow()),
    )


def run_pipeline(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Grow the cache (archive -> train -> live) and return the run summary."""
    root = Path(args.root)
    now = utcnow()
    start = now - timedelta(days=max(1, args.history_days))
    ticker = args.ticker

    model_path = default_bundle_path(ticker, directory=root / "models")
    archive_path = merge_candles(
        candles_path(root, ticker),
        candle_dicts_from_rows(
            build_dataset(
                ticker,
                start=start,
                end=now,
                timeframe=args.timeframe,
                limit=args.limit,
                fetch_timeout=args.fetch_timeout,
            ).rows
        ),
    )

    # Train over the WHOLE archive — features and labels are recomputed from the
    # full, continuously-growing clean history.
    train_rows = load_candles(archive_path, ticker=ticker)
    samples = build_samples(train_rows, horizon=args.horizon)
    bundle = train_tier1(samples, horizon=args.horizon)

    # Live snapshot (market + live news) fused with an outlook from the newest
    # archived bar.
    outlook = outlook_for_rows(train_rows, bundle, ticker=ticker)
    live_record, live_note = call_with_timeout(
        lambda: build_unified(ticker, lookback_hours=args.lookback_hours),
        timeout=args.fetch_timeout,
        label="live",
    )
    if live_record is None:
        live_record = UnifiedRecord(ticker=ticker, notes=(live_note,) if live_note else ())
    attach_predictions(live_record, outlook)

    live_dest = append_capped(live_path(root, ticker), live_record.to_dict())

    summary = {
        "schema": "rl-crypto/pipeline/2",
        "ticker": ticker,
        "generated_at": to_iso(now),
        "timeframe": args.timeframe,
        "horizon": args.horizon,
        "fetch_window": {"start": to_iso(start), "end": to_iso(now)},
        "archive": {
            "rows": len(train_rows),
            "candles_path": str(archive_path),
            "spans": [train_rows[0].timestamp, train_rows[-1].timestamp] if train_rows else None,
        },
        "samples": {"total": len(samples), "feature_complete": bundle.n_samples},
        "targets": {name: model.metrics for name, model in bundle.targets.items()},
        "outlook": {
            "signal": outlook.signal,
            "confidence": outlook.confidence,
            "predictions": outlook.predictions,
            "probabilities": outlook.probabilities,
            "notes": list(outlook.notes),
        },
        "live": {
            "market_ok": live_record.market_ok,
            "news_ok": live_record.news_ok,
            "price": live_record.price,
            "latest_sentiment": live_record.latest_sentiment,
            "notes": list(live_record.notes),
        },
    }
    summary_path(root, ticker).parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_path(root, ticker), summary)

    artifacts = {
        "bundle": save_bundle(bundle, model_path),
        "candles": archive_path,
        "live": live_dest,
        "summary": summary_path(root, ticker),
    }
    exit_code = 0 if (train_rows or live_record.market_ok or live_record.news_ok) else 1
    return artifacts, summary, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="ETH/USD", help="asset to test the pipeline on")
    parser.add_argument("--timeframe", default="1h", help="market+feature interval")
    parser.add_argument("--horizon", type=int, default=1, help="forward horizon in intervals")
    parser.add_argument(
        "--history-days", type=int, default=60, help="days of market history to fetch per run"
    )
    parser.add_argument("--lookback-hours", type=int, default=1, help="live news window")
    parser.add_argument("--limit", type=int, default=None, help="maximum market candles")
    parser.add_argument(
        "--fetch-timeout", type=float, default=300.0, help="per-fetch wall-clock cap (s)"
    )
    parser.add_argument("--root", default=str(ROOT), help="repo root for models/ and data/")
    args = parser.parse_args(argv)

    exit_code = 1
    try:
        artifacts, summary, exit_code = run_pipeline(args)
        print(f"pipeline {summary['ticker']} @ {summary['generated_at']}")
        archive = summary["archive"]
        print(
            f"  archive: {archive['rows']} candles"
            f" ({archive['spans'][0]} .. {archive['spans'][1]})"
            if archive["spans"]
            else "  archive: empty"
        )
        print(
            f"  fitted targets: {', '.join(sorted(summary['targets'])) or 'none'}"
            f"  | samples complete: {summary['samples']['feature_complete']}"
        )
        print(
            f"  live: market_ok={summary['live']['market_ok']}"
            f" news_ok={summary['live']['news_ok']}"
            f" | signal={summary['outlook']['signal']}"
            f" conf={summary['outlook']['confidence']}"
        )
        for name, path in artifacts.items():
            print(f"  wrote {name:<8} {path}")
    except Exception as exc:  # tolerated: report and still exit promptly
        print(f"pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 1

    # Hard exit: flushed work is done, but abandoned fetch workers (survivors of
    # a per-side timeout) must not keep the process alive.
    sys.stdout.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())