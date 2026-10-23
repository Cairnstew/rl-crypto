#!/usr/bin/env python3
"""Frequent live recorder: one live record per ticker per run, no retraining.

The heavier history-archive + training work happens on the pipeline workflow's
slower cadence; this tool is the *frequent* live path, and it records the **whole
tracked watchlist** (see :mod:`rl_crypto.tickers`) on every run:

* load the latest committed Tier-1 bundle for each ticker,
* build the Tier-0 feature window from the committed candle archive topped up
  with a short live tail fetch (no deep backfill every 30 minutes),
* take one live unified snapshot per ticker — market snapshots come from a
  **single rate-limited batch** (upstream ``aget_tickers``) and news is pulled
  per ticker and isolated,
* compose the Tier-2 outlook and **append** one record to each
  ``data/<slug>/live.jsonl`` (capped by ``--live-cap``).

Run by the local systemd timer (and manually):

    rl-pipeline live
    rl-pipeline live --tickers BTC/USD ETH/USD
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from rl_crypto.cache import (
    LIVE_CAP_DEFAULT,
    append_capped,
    candles_path,
    live_path,
    load_candles,
)
from rl_crypto.dataset import fetch_prices, interval_seconds, parse_iso
from rl_crypto.ml import attach_predictions
from rl_crypto.ml.serve import outlook_for_rows
from rl_crypto.ml.store import default_bundle_path, load_bundle
from rl_crypto.models import DatasetRow, to_iso, utcnow
from rl_crypto.tickers import parse_tickers
from rl_crypto.unify import build_unified_many

#: Repo root holding ``data/`` and ``models/``. Not derived from ``__file__``:
#: a hermetic install puts this module in a store path, so the caller's working
#: directory (the repo root, for every runner) is the only stable anchor.
ROOT = Path(os.environ.get("RL_CRYPTO_ROOT") or Path.cwd()).resolve()

#: Tail of live candles fetched per ticker per run. The rest of the feature
#: window comes from the committed archive, so a 30-minute tick costs one small
#: fetch per ticker instead of a full history pull.
TAIL_HOURS_DEFAULT = 6


def _window_rows(
    ticker: str,
    *,
    root: Path,
    start: datetime,
    end: datetime,
    timeframe: str,
    tail_hours: int,
    fetch_timeout: float,
) -> tuple[tuple[DatasetRow, ...], tuple[str, ...]]:
    """Return the feature-window rows for ``ticker`` plus any fetch notes.

    Prefers the committed archive (``data/<slug>/candles.jsonl``) for the bulk of
    the window and fetches only the last ``tail_hours`` live, then merges and
    de-duplicates by timestamp. Falls back to a full-window fetch when no
    archive exists yet (a ticker's first run).
    """
    notes: list[str] = []
    rows: dict[str, DatasetRow] = {}

    def add(row: DatasetRow, stamp: datetime | None) -> None:
        """Insert a row under a canonical key (dedupe across mixed formats).

        Archive timestamps and freshly fetched ones do not share a string
        format (``"… 21:00:00+00:00"`` vs ``"…T21:00:00Z"``), so the dedupe
        key is normalised and the row keeps its own timestamp string.
        """
        if stamp is None:
            return
        key = to_iso(stamp)
        if key is not None:
            rows[key] = row

    archived = load_candles(candles_path(root, ticker), ticker=ticker)
    if archived:
        for row in archived:
            stamp = parse_iso(row.timestamp)
            if stamp is not None and start <= stamp <= end:
                add(row, stamp)
    else:
        notes.append("features: no archive yet, fetching the full window")

    tail_start = max(start, end - timedelta(hours=max(1, tail_hours)))
    candles, tail_notes = fetch_prices(
        ticker,
        start=tail_start,
        end=end,
        timeframe=timeframe,
        timeout=fetch_timeout,
    )
    notes.extend(tail_notes)
    for candle in candles:
        raw = candle.get("timestamp")
        stamp = parse_iso(raw)
        key = to_iso(stamp)
        if stamp is None or key is None:
            continue
        rows[key] = DatasetRow(
            ticker=ticker,
            timestamp=key,
            open=candle.get("open"),
            high=candle.get("high"),
            low=candle.get("low"),
            close=candle.get("close"),
            volume=candle.get("volume"),
            has_price=True,
            has_news=False,
        )

    # Sort by the canonical key, not by the row's own timestamp string.
    ordered = tuple(rows[key] for key in sorted(rows))
    return ordered, tuple(notes)


def record_live(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Path], int]:
    """Record one live point for every ticker; return ``(payloads, paths, code)``.

    Market snapshots are fetched once for the whole watchlist in a single
    rate-limited batch; each ticker is then feature-scored and appended to its
    own cache. A ticker that fails is reported and skipped without stopping the
    rest of the watchlist.
    """
    root = Path(args.root or ROOT)
    tickers = parse_tickers(args.tickers)
    timeline = utcnow()
    end = timeline
    start = end - timedelta(hours=max(1, args.window))
    tail_hours = getattr(args, "tail_hours", TAIL_HOURS_DEFAULT) or TAIL_HOURS_DEFAULT

    # One batch for every ticker: shared rate limiter, so the watchlist costs one
    # request budget rather than one per ticker.
    records = build_unified_many(tickers, lookback_hours=args.lookback_hours)

    payloads: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    failures = 0

    for ticker, record in zip(tickers, records, strict=True):
        try:
            rows, notes = _window_rows(
                ticker,
                root=root,
                start=start,
                end=end,
                timeframe=args.timeframe,
                tail_hours=tail_hours,
                fetch_timeout=args.fetch_timeout,
            )
            bundle = load_bundle(default_bundle_path(ticker, directory=root / "models"))
            outlook = outlook_for_rows(
                rows,
                bundle,
                ticker=ticker,
                interval_seconds=interval_seconds(args.timeframe),
            )
            attach_predictions(record, outlook)
            if notes:
                record.notes = (*record.notes, *notes)

            payload = record.to_dict()
            payload["_generated_at"] = to_iso(timeline)
            dest = append_capped(live_path(root, ticker), payload, cap=args.live_cap)
            paths[ticker] = dest
            payloads.append(payload)

            preds = payload["predictions"]
            # `confidence` is None whenever the outlook has no earned signal —
            # format it as a string, never as a float.
            confidence = preds["confidence"]
            shown = "n/a" if confidence is None else f"{confidence:.3f}"
            print(
                f"live {ticker} @ {payload['_generated_at']}"
                f" | signal={preds['signal']} conf={shown}"
                f" | mkt={payload['market']['ok']} news={payload['news']['ok']}"
                f" price={payload['market']['price']}"
                f" | candles={len(rows)}"
            )
            if not (
                outlook.feature_timestamp is not None
                or record.market_ok
                or record.news_ok
            ):
                failures += 1
        except Exception as exc:  # one bad ticker must not stop the watchlist
            failures += 1
            print(f"live {ticker} failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    return payloads, paths, (1 if failures or not payloads else 0)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the live-recorder flags (shared by the CLI and the legacy shim)."""
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="tickers to record (default: the tracked watchlist); accepts a "
        "comma- or space-separated list",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="single-ticker alias for --tickers (kept for existing callers)",
    )
    parser.add_argument(
        "--timeframe", default="1h", help="market interval for the feature window"
    )
    parser.add_argument(
        "--window", type=int, default=48, help="hours of market history for features"
    )
    parser.add_argument(
        "--tail-hours",
        type=int,
        default=TAIL_HOURS_DEFAULT,
        help="hours of live candles fetched per ticker (the archive fills the rest)",
    )
    parser.add_argument("--lookback-hours", type=int, default=1, help="live news window")
    parser.add_argument(
        "--fetch-timeout", type=float, default=120.0, help="per-fetch wall-clock cap (s)"
    )
    parser.add_argument(
        "--live-cap", type=int, default=LIVE_CAP_DEFAULT, help="max live.jsonl records"
    )
    parser.add_argument(
        "--root", default=None, help="repo root for models/ and data/ (default: the cwd)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rl-pipeline live",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Record one live point per selected ticker; return the exit status."""
    if args.ticker and not args.tickers:
        args.tickers = [args.ticker]

    exit_code = 1
    try:
        payloads, paths, exit_code = record_live(args)
        for ticker, dest in paths.items():
            print(f"  appended -> {ticker}: {dest}")
        print(f"recorded {len(payloads)} ticker(s)")
    except Exception as exc:  # tolerated: report and still exit promptly
        print(f"record_live failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 1

    sys.stdout.flush()
    return exit_code


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
