#!/usr/bin/env python3
"""Production pipeline run for one ticker: history cache, bundle, live outlook.

This is the deterministic invocation the scheduled GitHub Actions workflow
(``.github/workflows/pipeline.yml``) runs on a cron and commits back to the repo:

* fetch the *joined* history window (tickerkit OHLCV + tns sentiment),
* materialise Tier-0 features + forward targets, fit the Tier-1 bundle,
* take a live unified snapshot and attach the Tier-2 outlook (the feature
  addition),
* cache everything under ``models/`` and ``data/<slug>/``.

The cache is **bounded**: the training window, bundle, and summary are
overwritten each run, and ``live.jsonl`` is appended but capped at recent
records — so the committed cache never grows without limit. Any side that
fails is recorded in ``notes``/``summary.json`` and the still-good data is
still written; the exit code is non-zero only when nothing at all succeeded.

    uv run python tools/run_pipeline.py --ticker ETH/USD --train-days 21
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from rl_crypto import build_samples
from rl_crypto.dataset import build_dataset
from rl_crypto.export import write_json, write_jsonl
from rl_crypto.ml.serve import attach_predictions, outlook_for_rows
from rl_crypto.ml.store import default_bundle_path, save_bundle
from rl_crypto.ml.tier1 import train_tier1
from rl_crypto.models import to_iso, utcnow
from rl_crypto.unify import build_unified

ROOT = Path(__file__).resolve().parents[1]

#: Recent live records kept in ``data/<slug>/live.jsonl`` (bounded cache).
LIVE_CAP = 32


def _append_capped(path: Path, record: dict, cap: int | None = None) -> None:
    """Append one JSON record, keeping only the ``cap`` most recent lines.

    ``cap`` resolves at call time (module-level :data:`LIVE_CAP`) so a test (or
    operator) can tune it without re-importing.
    """
    if cap is None:
        cap = LIVE_CAP
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    keep = max(0, cap - 1)
    lines = lines[-keep:] if keep > 0 else lines[:0]
    lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> tuple[dict, dict, int]:
    """Execute the full pipeline and return ``(artifacts, summary, exit_code)``."""
    root = Path(args.root)
    now = utcnow()
    start = now - timedelta(days=max(1, args.train_days))
    ticker = args.ticker

    model_path = default_bundle_path(ticker, directory=root / "models")
    slug = model_path.stem
    data_dir = root / "data" / slug

    # 1. history: the fused window (both sides tolerate failure)
    dataset = build_dataset(
        ticker,
        start=start,
        end=now,
        timeframe=args.timeframe,
        granularity=args.granularity,
        limit=args.limit,
    )

    # 2. features + targets -> Tier-1 bundle (trained on the window)
    samples = build_samples(dataset.rows, horizon=args.horizon)
    bundle = train_tier1(samples, horizon=args.horizon)

    # 3. live snapshot fused with an outlook from the newest complete feature row
    outlook = outlook_for_rows(dataset.rows, bundle, ticker=ticker)
    record = build_unified(ticker, lookback_hours=args.lookback_hours)
    attach_predictions(record, outlook)

    # 4. cache everything
    live_path = data_dir / "live.jsonl"
    _append_capped(live_path, record.to_dict())
    artifacts: dict[str, Path] = {
        "bundle": save_bundle(bundle, model_path),
        "history": write_jsonl(data_dir / "history.jsonl", (s.to_dict() for s in samples)),
        "live": live_path,
    }

    summary = {
        "schema": "rl-crypto/pipeline/1",
        "ticker": ticker,
        "generated_at": to_iso(now),
        "timeframe": args.timeframe,
        "granularity": args.granularity,
        "horizon": args.horizon,
        "train_window": {"start": to_iso(start), "end": to_iso(now)},
        "dataset": {
            "rows": len(dataset.rows),
            "price_rows": dataset.price_rows,
            "news_rows": dataset.news_rows,
            "joined_rows": dataset.joined_rows,
            "notes": list(dataset.notes),
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
            "market_ok": record.market_ok,
            "news_ok": record.news_ok,
            "price": record.price,
            "latest_sentiment": record.latest_sentiment,
            "notes": list(record.notes),
        },
    }
    artifacts["summary"] = write_json(data_dir / "summary.json", summary)

    exit_code = 0 if (dataset.rows or record.market_ok or record.news_ok) else 1
    return artifacts, summary, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="ETH/USD", help="asset to test the pipeline on")
    parser.add_argument("--timeframe", default="1h", help="market+feature interval")
    parser.add_argument("--granularity", default="hour", choices=["hour", "day"])
    parser.add_argument("--horizon", type=int, default=1, help="forward horizon in intervals")
    parser.add_argument("--train-days", type=int, default=21, help="days of history to train on")
    parser.add_argument("--lookback-hours", type=int, default=1, help="live news window")
    parser.add_argument("--limit", type=int, default=None, help="maximum market candles")
    parser.add_argument("--root", default=str(ROOT), help="repo root for models/ and data/")
    args = parser.parse_args(argv)

    artifacts, summary, exit_code = run_pipeline(args)
    print(f"pipeline {summary['ticker']} @ {summary['generated_at']}")
    print(
        f"  dataset: {summary['dataset']['rows']} rows "
        f"(price {summary['dataset']['price_rows']}, news {summary['dataset']['news_rows']})"
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())