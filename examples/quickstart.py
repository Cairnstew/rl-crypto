#!/usr/bin/env python3
"""Quickstart: run the unified pipeline for one ticker, offline-safe.

    uv run python examples/quickstart.py

Both sides are fetched from the live upstreams, so this needs network. A
failing side is recorded in ``notes`` rather than raising.
"""

from __future__ import annotations

from rl_crypto import build_dataset, build_unified
from rl_crypto.render import render_dataset, render_unified


def main() -> int:
    ticker = "ETH/USD"

    record = build_unified(ticker, lookback_hours=3)
    print(render_unified(record))
    print()

    dataset = build_dataset(
        ticker,
        start="2026-01-01",
        end="2026-01-03",
        timeframe="1h",
    )
    print(render_dataset(dataset, max_rows=8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())