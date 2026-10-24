#!/usr/bin/env python3
"""Production pipeline run for one ticker: grow the archive, train, push live.

This is the deterministic invocation the scheduled GitHub Actions workflow
(``.github/workflows/production.yml``) runs on a cron and commits back to the repo:

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

    rl-pipeline run --ticker ETH/USD --history-days 60
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
from rl_crypto.dataset import candle_dicts, fetch_market_history, interval_seconds, parse_iso
from rl_crypto.export import write_json
from rl_crypto.ml import attach_predictions, outlook_for_rows
from rl_crypto.ml.store import default_bundle_path, save_bundle
from rl_crypto.ml.tier1 import train_tier1
from rl_crypto.ml.training_config import load_training_config
from rl_crypto.models import Dataset, DatasetRow, UnifiedRecord, to_iso, utcnow
from rl_crypto.tickers import parse_tickers
from rl_crypto.unify import build_unified

#: Repo root holding ``data/`` and ``models/``. Deliberately *not* derived from
#: ``__file__``: under a hermetic install (``nix run``, ``uv run`` from a built
#: venv) this module lives in a store path, so the only reliable anchor is the
#: caller's working directory — every runner ``cd``s to the repo root first.
ROOT = Path(os.environ.get("RL_CRYPTO_ROOT") or Path.cwd()).resolve()


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


def _fmt_confidence(value: float | None) -> str:
    """Render a confidence for the console; ``None`` means no signal was earned."""
    return "n/a" if value is None else f"{value:.3f}"


def run_pipeline(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Grow the cache (archive -> train -> live) and return the run summary."""
    root = Path(args.root or ROOT)
    now = utcnow()
    start = now - timedelta(days=max(1, args.history_days))
    ticker = args.ticker
    retrain_only = bool(getattr(args, "retrain_only", False))
    cadence = interval_seconds(args.timeframe)

    model_path = default_bundle_path(ticker, directory=root / "models")
    if retrain_only:
        # Rebuild the models from the archive already on disk. Used after a
        # feature-contract change, when the labels and features must be
        # recomputed but the bars have not changed — spending upstream quota to
        # re-fetch identical candles would buy nothing.
        #
        # The merge still runs (with nothing to add) because it canonicalises
        # timestamps in place: that is how an archive written before the
        # canonical ISO form existed gets normalised, for free and without a
        # single network call.
        archive_path = merge_candles(candles_path(root, ticker), [])
    else:
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
    # full, continuously-growing clean history. The archive is stored at the
    # run's timeframe, so the feature/target windows are measured in that same
    # interval rather than assuming the 1h default.
    #
    # Feature set, per-target estimators and horizon come from the committed
    # config, not from code defaults: `promote --apply` edits that file, so a
    # promoted candidate survives the next scheduled retrain.
    training = load_training_config(root).for_ticker(ticker)
    horizon = training.horizon if training.horizon is not None else args.horizon
    train_rows = load_candles(archive_path, ticker=ticker)
    samples = build_samples(train_rows, horizon=horizon, interval_seconds=cadence)
    bundle = train_tier1(
        samples,
        horizon=horizon,
        feature_names=training.feature_names,
        model_names=training.model_names or None,
    )

    # Live snapshot (market + live news) fused with an outlook from the newest
    # archived bar.
    outlook = outlook_for_rows(
        train_rows,
        bundle,
        ticker=ticker,
        interval_seconds=cadence,
    )

    live_dest: Path | None = None
    live_record: UnifiedRecord | None = None
    if not retrain_only:
        record, live_note = call_with_timeout(
            lambda: build_unified(ticker, lookback_hours=args.lookback_hours),
            timeout=args.fetch_timeout,
            label="live",
        )
        if record is None:
            record = UnifiedRecord(ticker=ticker, notes=(live_note,) if live_note else ())
        attach_predictions(record, outlook)
        live_record = record
        live_dest = append_capped(live_path(root, ticker), record.to_dict())

    summary = {
        "schema": "rl-crypto/pipeline/3",
        "ticker": ticker,
        "generated_at": to_iso(now),
        "timeframe": args.timeframe,
        "horizon": horizon,
        "training_config": {
            "source": training.source,
            "features": len(training.feature_names),
            "model_names": dict(training.model_names),
        },
        "feature_version": bundle.feature_version,
        "fetch_window": {"start": to_iso(start), "end": to_iso(now)},
        "archive": {
            "rows": len(train_rows),
            "candles_path": str(archive_path),
            "spans": [train_rows[0].timestamp, train_rows[-1].timestamp] if train_rows else None,
        },
        "samples": {
            "total": len(samples),
            "feature_complete": bundle.n_samples,
            "holdout": bundle.holdout_n_samples,
        },
        # What the models actually do on unseen data, and whether that beats the
        # bar a constant predictor would clear. `in_sample` is kept for the
        # fit-quality view it always was, now explicitly labelled so it cannot
        # be mistaken for the headline number.
        "targets": {
            name: {
                "in_sample": model.metrics,
                # "raw" or "log": the scale the regression metrics are on, so a
                # log-space model is not compared to a raw-scale constant.
                "metric_space": model.metric_space,
                "holdout": None
                if model.holdout is None
                else {
                    "n_samples": model.holdout.n_samples,
                    "metrics": model.holdout.metrics,
                    "baseline": model.holdout.baseline,
                    "beats_baseline": model.holdout.beats_baseline,
                },
                "trusted": bundle.trusted(name),
            }
            for name, model in bundle.targets.items()
        },
        "features": {
            "used": list(bundle.feature_names),
            "dropped_constant": list(bundle.dropped_features),
        },
        "outlook": {
            "signal": outlook.signal,
            "confidence": outlook.confidence,
            "predictions": outlook.predictions,
            "probabilities": outlook.probabilities,
            "notes": list(outlook.notes),
        },
        "live": (
            {
                "fetched": False,
                "note": "retrain-only run: no live snapshot was taken",
            }
            if live_record is None
            else {
                "fetched": True,
                "market_ok": live_record.market_ok,
                "news_ok": live_record.news_ok,
                "price": live_record.price,
                "latest_sentiment": live_record.latest_sentiment,
                "notes": list(live_record.notes),
            }
        ),
        "notes": list(bundle.notes),
    }
    summary_path(root, ticker).parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_path(root, ticker), summary)

    artifacts = {
        "bundle": save_bundle(bundle, model_path),
        "candles": archive_path,
        "summary": summary_path(root, ticker),
    }
    if live_dest is not None:
        artifacts["live"] = live_dest
    exit_code = (
        0
        if (
            train_rows
            or (live_record is not None and (live_record.market_ok or live_record.news_ok))
        )
        else 1
    )
    return artifacts, summary, exit_code


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the pipeline-run flags (shared by the CLI and the legacy shim)."""
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="tickers to run the pipeline on (default: the tracked watchlist)",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="single-ticker alias for --tickers (kept for existing callers)",
    )
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
    parser.add_argument(
        "--retrain-only",
        action="store_true",
        help="rebuild models/summaries from the committed archive without fetching",
    )
    parser.add_argument(
        "--root", default=None, help="repo root for models/ and data/ (default: the cwd)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rl-pipeline run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Run the pipeline over every selected ticker; return the exit status."""
    if args.ticker and not args.tickers:
        args.tickers = [args.ticker]
    tickers = parse_tickers(args.tickers)

    exit_code = 1
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        # One ticker at a time, deliberately: a deep history backfill spends many
        # upstream calls per series, and upstream tickerkit paces --history as a
        # single-asset operation. Parallel jobs would multiply that request rate.
        for ticker in tickers:
            args.ticker = ticker
            try:
                artifacts, summary, code = run_pipeline(args)
            except Exception as exc:  # one ticker's failure must not stop the rest
                failures.append(ticker)
                print(
                    f"pipeline {ticker} failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            # Reporting lives inside the try: a formatting slip over one ticker's
            # summary must not take the rest of the watchlist down with it.
            summaries.append(summary)
            exit_code = 0 if code == 0 else exit_code
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
            trusted = [name for name in summary["targets"] if summary["targets"][name]["trusted"]]
            untrusted = [name for name in summary["targets"] if name not in trusted]
            print(f"  beats baseline out of sample: {', '.join(trusted) or 'none'}")
            if untrusted:
                print(f"  below baseline (no signal): {', '.join(untrusted)}")
            live = summary["live"]
            live_state = (
                f"market_ok={live.get('market_ok')} news_ok={live.get('news_ok')}"
                if live.get("fetched")
                else "skipped (retrain-only)"
            )
            print(
                f"  live: {live_state}"
                f" | signal={summary['outlook']['signal']}"
                # confidence is None whenever no signal is earned — never a float format
                f" conf={_fmt_confidence(summary['outlook']['confidence'])}"
            )
            for name, path in artifacts.items():
                print(f"  wrote {name:<8} {path}")
        print(
            f"pipeline: {len(summaries)}/{len(tickers)} ticker(s) ok"
            + (f", failed: {', '.join(failures)}" if failures else "")
        )
        if not summaries:
            exit_code = 1
    except Exception as exc:  # tolerated: report and still exit promptly
        print(f"pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 1

    # Flushed work is done; the CLI turns this into a hard exit so abandoned
    # fetch workers (survivors of a per-side timeout) cannot keep it alive.
    sys.stdout.flush()
    return exit_code


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())