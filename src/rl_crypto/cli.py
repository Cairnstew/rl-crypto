"""The unified ``rl-crypto`` command line.

One entry point over two pipelines:

    rl-crypto market  TICKER [tickerkit args…]      # delegate: market snapshot
    rl-crypto news    TICKER [tns pull args…]       # delegate: live news pull
    rl-crypto news-history TICKER --from … --to …   # delegate: archived news
    rl-crypto unified TICKER                        # fused market + news record
    rl-crypto dataset TICKER --from … --to …        # joined hourly dataset
    rl-crypto features TICKER --from … --to …       # Tier-0 features + targets
    rl-crypto train TICKER --from … --to …          # fit Tier-1 bundle (json)
    rl-crypto predict TICKER --model models/….json  # live outlook → feature addition
    rl-crypto sources                               # every upstream source
    rl-crypto version

``market`` / ``news`` / ``news-history`` are thin **delegations**: they hand the
ticker and the remaining arguments to the upstream CLI, so every upstream flag
(``--report``, ``--frames``, ``--append``, ``--sources`` …) keeps working. The
``unified`` and ``dataset`` commands are rl-crypto's own fused output.

Everything goes through ``args.func`` and returns an int exit status; import
errors from the heavy upstreams surface as a friendly message, not a traceback.
"""

from __future__ import annotations

import argparse
import json as _json
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .export import write_csv, write_json, write_jsonl
from .render import (
    dataset_flat_metrics,
    render_dataset,
    render_outlook,
    render_unified,
    render_unified_markdown,
    unified_flat_metrics,
)
from .targets import TARGET_NAMES


def _emit_json(payload: object, *, pretty: bool = False) -> None:
    print(_json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False, sort_keys=True))


# --------------------------------------------------------------------------- passthroughs


def cmd_market(args: argparse.Namespace) -> int:
    """Delegate to the tickerkit CLI: ``tickerkit TICKER [args…]``."""
    from tickerkit.cli import main as market_main

    return market_main([args.ticker, *args.rest])


def cmd_news(args: argparse.Namespace) -> int:
    """Delegate to the ticker-news-signals CLI as ``pull``."""
    from ticker_news_signals.cli import main as news_main

    return news_main(["pull", "--ticker", args.ticker, *args.rest])


def cmd_news_history(args: argparse.Namespace) -> int:
    """Delegate to the ticker-news-signals CLI as ``history``."""
    from ticker_news_signals.cli import main as news_main

    return news_main(["history", "--ticker", args.ticker, *args.rest])


# --------------------------------------------------------------------------- fused


def cmd_unified(args: argparse.Namespace) -> int:
    from .unify import build_unified

    config = Config.from_env().with_overrides(lookback_hours=args.lookback_hours)
    record = build_unified(
        args.ticker,
        lookback_hours=config.lookback_hours,
        news_config=config.news_config(),
        quote=args.quote,
    )

    if args.json:
        _emit_json(record.to_dict(), pretty=True)
    elif args.metrics:
        _emit_json(unified_flat_metrics(record))
    elif args.markdown:
        print(render_unified_markdown(record))
    else:
        print(render_unified(record))
    # Non-zero only when *both* sides failed: a half-answer is still an answer.
    return 0 if (record.market_ok or record.news_ok) else 1


def cmd_dataset(args: argparse.Namespace) -> int:
    from .dataset import build_dataset

    config = Config.from_env()
    dataset = build_dataset(
        args.ticker,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe or config.timeframe,
        granularity=args.granularity or config.granularity,
        limit=args.limit,
        market_config=config.market_config(),
        news_config=config.news_config(),
    )

    if args.output:
        fmt = args.format or config.format
        if fmt == "csv":
            path = write_csv(args.output, (row.to_dict() for row in dataset.rows))
        elif args.json:
            path = write_json(args.output, dataset.to_dict())
        else:
            path = write_jsonl(args.output, (row.to_dict() for row in dataset.rows))
        print(f"dataset -> {path}", file=sys.stderr)

    if not dataset.rows and not (args.metrics or args.json or args.report):
        print(f"no rows in window for {args.ticker}", file=sys.stderr)

    if args.metrics:
        _emit_json(dataset_flat_metrics(dataset))
    elif args.json:
        _emit_json(dataset.to_dict(), pretty=True)
    elif args.report:
        print(render_dataset(dataset))
    elif not args.output:
        for row in dataset.rows:
            _emit_json(row.to_dict())
    return 0


# --------------------------------------------------------------------------- ML


def cmd_features(args: argparse.Namespace) -> int:
    """Tier-0 feature rows + forward targets for a window (no learning)."""
    from .dataset import build_dataset
    from .samples import build_samples

    config = Config.from_env()
    dataset = build_dataset(
        args.ticker,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe or config.timeframe,
        granularity=args.granularity or config.granularity,
        limit=args.limit,
        market_config=config.market_config(),
        news_config=config.news_config(),
    )
    samples = build_samples(dataset.rows, horizon=args.horizon)

    if args.output:
        fmt = args.format or config.format
        rows = (sample.to_dict() for sample in samples)
        path = write_csv(args.output, rows) if fmt == "csv" else write_jsonl(args.output, rows)
        print(f"features -> {path}", file=sys.stderr)
    elif args.json:
        _emit_json([sample.to_dict() for sample in samples], pretty=True)
    else:
        for sample in samples:
            _emit_json(sample.to_dict())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Fit Tier-1 models from a window's samples and write a JSON bundle."""
    from .dataset import build_dataset
    from .ml.store import default_bundle_path
    from .ml.tier1 import train_tier1
    from .samples import build_samples

    config = Config.from_env()
    dataset = build_dataset(
        args.ticker,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe or config.timeframe,
        granularity=args.granularity or config.granularity,
        limit=args.limit,
        market_config=config.market_config(),
        news_config=config.news_config(),
    )
    samples = build_samples(dataset.rows, horizon=args.horizon)
    targets = args.target or list(TARGET_NAMES)
    bundle = train_tier1(samples, horizon=args.horizon, targets=targets)

    path = Path(args.output) if args.output else default_bundle_path(args.ticker)
    from .ml.store import save_bundle

    save_bundle(bundle, path)
    print(
        f"trained {len(bundle.targets)} target(s) from {len(samples)} rows "
        f"({bundle.n_samples} feature-complete) -> {path}",
        file=sys.stderr,
    )
    for note in bundle.notes:
        print(f"  - {note}", file=sys.stderr)

    if args.json:
        _emit_json(bundle.to_dict(), pretty=True)
    return 0 if bundle.targets else 1


def cmd_predict(args: argparse.Namespace) -> int:
    """Live Tier-1/Tier-2 outlook, attached to the unified record as features."""
    from .ml.serve import attach_predictions, load_bundle_for, predict_live
    from .ml.store import default_bundle_path
    from .unify import build_unified

    config = Config.from_env().with_overrides(lookback_hours=args.lookback_hours)
    try:
        bundle = load_bundle_for(args.ticker, args.model)
    except FileNotFoundError:
        print(
            f"error: no model bundle for {args.ticker} — run `rl-crypto train {args.ticker} …` "
            f"(looked for {args.model or default_bundle_path(args.ticker)})",
            file=sys.stderr,
        )
        return 2

    record = build_unified(
        args.ticker,
        lookback_hours=config.lookback_hours,
        news_config=config.news_config(),
        quote=args.quote,
    )
    outlook = predict_live(
        args.ticker,
        bundle,
        timeframe=args.timeframe or "1h",
        window=args.window,
        granularity=args.granularity or config.granularity,
        market_config=config.market_config(),
        news_config=config.news_config(),
    )
    attach_predictions(record, outlook)

    if args.json:
        _emit_json(record.to_dict(), pretty=True)
    else:
        print(render_unified(record))
        print()
        print(render_outlook(outlook))
    return 0 if (record.market_ok or record.news_ok) else 1


def cmd_sources(_: argparse.Namespace) -> int:
    """List the news sources and market providers both upstreams expose."""
    from ticker_news_signals.config import Config as NewsConfig
    from ticker_news_signals.history import HISTORY_PROVIDERS
    from ticker_news_signals.providers import registry as news_registry
    from tickerkit.providers import PROVIDERS

    news_config = NewsConfig.from_env()
    print("news sources (ticker-news-signals)")
    print(f"  {'source':<16} {'available':<10} capabilities")
    print("  " + "-" * 56)
    for cls in (*news_registry(), *HISTORY_PROVIDERS):
        try:
            ok, _ = cls.available(news_config)
        except Exception:
            ok = False
        caps = ",".join(sorted(str(c) for c in getattr(cls, "supports", ())))
        print(f"  {cls.name:<16} {'yes' if ok else 'no':<10} {caps}")

    print()
    print("market providers (tickerkit)")
    print(f"  {'provider':<16} capabilities")
    print("  " + "-" * 56)
    for cls in PROVIDERS:
        caps = ",".join(sorted(str(c) for c in getattr(cls, "supports", ())))
        print(f"  {getattr(cls, 'name', cls.__name__):<16} {caps}")
    return 0


def cmd_version(_: argparse.Namespace) -> int:
    print(f"rl-crypto {__version__}")
    return 0


# --------------------------------------------------------------------------- parser


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="pretty JSON envelope")
    parser.add_argument("--markdown", action="store_true", help="report as a fenced markdown block")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl-crypto",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"rl-crypto {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("market", help="market snapshot (delegates to tickerkit)")
    p.add_argument("ticker", help="e.g. BTC, ETH, SOL, BTC/USDT")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="args forwarded to tickerkit")
    p.set_defaults(func=cmd_market)

    p = sub.add_parser("news", help="live news pull (delegates to `ticker-news-signals pull`)")
    p.add_argument("ticker", help="e.g. ETH/USD or ETH")
    p.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="args forwarded to `ticker-news-signals pull`",
    )
    p.set_defaults(func=cmd_news)

    p = sub.add_parser(
        "news-history",
        help="archived news history (delegates to `ticker-news-signals history`)",
    )
    p.add_argument("ticker", help="e.g. ETH/USD or ETH")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="args forwarded to `... history`")
    p.set_defaults(func=cmd_news_history)

    p = sub.add_parser("unified", help="fuse a market snapshot and a news pull for one ticker")
    p.add_argument("ticker", help="e.g. ETH/USD")
    p.add_argument("--lookback-hours", type=int, default=None, help="news window (default 1)")
    p.add_argument("--quote", default=None, help="market quote currency (default USD)")
    p.add_argument("--metrics", action="store_true", help="one flat JSON metrics row")
    _add_output_flags(p)
    p.set_defaults(func=cmd_unified)

    p = sub.add_parser("dataset", help="join market history and news history into one dataset")
    p.add_argument("ticker", help="e.g. ETH/USD")
    p.add_argument("--from", dest="start", required=True, help="window start (ISO date/datetime)")
    p.add_argument("--to", dest="end", required=True, help="window end (ISO date/datetime)")
    p.add_argument("--timeframe", default=None, help="market interval, e.g. 1h, 1d (default 1d)")
    p.add_argument(
        "--granularity", default=None, choices=["hour", "day"], help="news bucket (default hour)"
    )
    p.add_argument("--limit", type=int, default=None, help="maximum market candles")
    p.add_argument("--output", default=None, help="output file (default: stdout JSONL)")
    p.add_argument(
        "--format", default=None, choices=["jsonl", "csv"], help="output format (default jsonl)"
    )
    p.add_argument("--metrics", action="store_true", help="coverage metrics only")
    _add_output_flags(p)
    p.add_argument("--report", action="store_true", help="human-readable dataset report")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser(
        "features", help="Tier-0 features + forward targets for a window (no learning)"
    )
    p.add_argument("ticker", help="e.g. ETH/USD")
    p.add_argument("--from", dest="start", required=True, help="window start (ISO)")
    p.add_argument("--to", dest="end", required=True, help="window end (ISO)")
    p.add_argument("--timeframe", default=None, help="market interval, e.g. 1h, 1d (default 1d)")
    p.add_argument(
        "--granularity", default=None, choices=["hour", "day"], help="news bucket (default hour)"
    )
    p.add_argument("--horizon", type=int, default=1, help="forward horizon in intervals")
    p.add_argument("--limit", type=int, default=None, help="maximum market candles")
    p.add_argument("--output", default=None, help="output file (default: stdout JSONL)")
    p.add_argument(
        "--format", default=None, choices=["jsonl", "csv"], help="output format (default jsonl)"
    )
    p.add_argument("--json", action="store_true", help="pretty JSON array of sample rows")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser(
        "train", help="fit Tier-1 models from a window's samples into a JSON bundle"
    )
    p.add_argument("ticker", help="e.g. ETH/USD")
    p.add_argument("--from", dest="start", required=True, help="window start (ISO)")
    p.add_argument("--to", dest="end", required=True, help="window end (ISO)")
    p.add_argument("--timeframe", default=None, help="market interval, e.g. 1h, 1d (default 1d)")
    p.add_argument(
        "--granularity", default=None, choices=["hour", "day"], help="news bucket (default hour)"
    )
    p.add_argument("--horizon", type=int, default=1, help="forward horizon in intervals")
    p.add_argument("--limit", type=int, default=None, help="maximum market candles")
    p.add_argument(
        "--target",
        action="append",
        choices=list(TARGET_NAMES),
        help="target metric to fit (repeatable; default: all)",
    )
    p.add_argument("--output", default=None, help="bundle path (default models/<slug>.json)")
    p.add_argument("--json", action="store_true", help="also print the bundle as JSON")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser(
        "predict",
        help="live outlook: run Tier-1 models on a recent window and attach to the record",
    )
    p.add_argument("ticker", help="e.g. ETH/USD")
    p.add_argument("--model", default=None, help="bundle path (default models/<slug>.json)")
    p.add_argument("--timeframe", default="1h", help="window interval for features (default 1h)")
    p.add_argument(
        "--window", type=int, default=200, help="intervals of history to fetch (default 200)"
    )
    p.add_argument(
        "--granularity", default=None, choices=["hour", "day"], help="news bucket (default hour)"
    )
    p.add_argument("--lookback-hours", type=int, default=None, help="news window (default 1)")
    p.add_argument("--quote", default=None, help="market quote currency (default USD)")
    p.add_argument("--json", action="store_true", help="pretty JSON record with predictions")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("sources", help="list news sources and market providers")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("version", help="print the package version")
    p.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ImportError as exc:  # a missing upstream dependency
        print(
            f"error: could not import an upstream pipeline ({exc}). "
            "Run inside the dev shell: `nix develop` (or `uv sync`).",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__: list[str] = ["build_parser", "main"]
