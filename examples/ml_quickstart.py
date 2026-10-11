"""Offline ML quickstart: synthetic window -> train a bundle -> live outlook.

Runs with no network and no upstreams, so it is a safe first play with the
tiered pipeline. Swap the synthetic rows for a real
``rl_crypto.dataset.build_dataset`` call to get the actual thing.

    uv run python examples/ml_quickstart.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import exp, sin

from rl_crypto import build_samples
from rl_crypto.ml import save_bundle, train_tier1
from rl_crypto.ml.serve import outlook_for_rows
from rl_crypto.models import DatasetRow, to_iso


def synthetic_rows(n: int = 120) -> tuple[DatasetRow, ...]:
    """A deterministic hourly price path with news, enough history to fit."""
    rows: list[DatasetRow] = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(n):
        close = 100.0 * exp(0.01 * i + 0.04 * sin(i))
        rows.append(
            DatasetRow(
                ticker="ETH/USD",
                timestamp=to_iso(start + timedelta(hours=i)) or "",
                open=close * 0.999,
                high=close * 1.01,
                low=close * 0.99,
                volume=1000.0 + 50.0 * i,
                close=close,
                sentiment_score=0.1 + 0.02 * sin(i),
                article_count=(i % 3) + 1,
                novelty_flag=(i % 7 == 0),
                has_price=True,
                has_news=True,
            )
        )
    return tuple(rows)


def main() -> None:
    rows = synthetic_rows()
    bundle = train_tier1(build_samples(rows, horizon=1), horizon=1)
    save_bundle(bundle, "models/eth-usd-demo.json")

    outlook = outlook_for_rows(rows, bundle, ticker="ETH/USD")
    print(outlook.to_dict())
    print("signal:", outlook.signal, "| confidence:", outlook.confidence)


if __name__ == "__main__":
    main()