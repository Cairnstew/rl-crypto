"""rl-crypto — one CLI and one dataset over two crypto pipelines.

Two independently useful upstreams answer different questions about the same
asset:

* **tickerkit** — the market: a provenance-tracked snapshot (consensus price
  across venues, 24h change/volume/cap) and a cross-referenced OHLCV history.
* **ticker-news-signals** — the narrative: per-(ticker, UTC hour) news-sentiment
  vectors scored locally from many keyless feeds, live and via Wayback.

``rl-crypto`` fuses them. The two-step surface:

    from rl_crypto import build_unified, build_dataset

    record = build_unified("ETH/USD", lookback_hours=3)
    print(record.price, record.latest_sentiment, record.notes)

    dataset = build_dataset("ETH/USD", start="2026-01-01", end="2026-01-07",
                            timeframe="1h")
    for row in dataset.rows:
        print(row.timestamp, row.close, row.sentiment_score)

Both calls are **partial-failure tolerant**: if one upstream raises, the other
side's data is still returned and the failure lands in ``notes``. Nothing is
fabricated — an unavailable measurement is ``None``.

The heavy upstreams are imported lazily inside the fetch functions, so
``import rl_crypto`` stays cheap.
"""

from __future__ import annotations

from .config import Config
from .dataset import (
    build_dataset,
    fetch_market_history,
    fetch_news_history,
    interval_seconds,
    join_buckets,
)
from .features import (
    FEATURE_NAMES,
    FeatureRow,
    build_feature_table,
    feature_row,
    latest_complete,
)
from .models import SCHEMA_VERSION, Dataset, DatasetRow, UnifiedRecord
from .samples import Sample, build_dataset_samples, build_samples
from .targets import TARGET_KINDS, TARGET_NAMES, compute_targets
from .unify import build_unified, fetch_market, fetch_news

__version__ = "0.1.0"

__all__ = [
    "FEATURE_NAMES",
    "SCHEMA_VERSION",
    "TARGET_KINDS",
    "TARGET_NAMES",
    "Config",
    "Dataset",
    "DatasetRow",
    "FeatureRow",
    "Sample",
    "UnifiedRecord",
    "__version__",
    "build_dataset",
    "build_dataset_samples",
    "build_feature_table",
    "build_samples",
    "build_unified",
    "compute_targets",
    "feature_row",
    "fetch_market",
    "fetch_market_history",
    "fetch_news",
    "fetch_news_history",
    "interval_seconds",
    "join_buckets",
    "latest_complete",
]
