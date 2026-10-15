"""The serve seam — turn a recent window of rows into a live :class:`Outlook`.

Training and serving share :mod:`rl_crypto.features`, so the live feature row is
built exactly the way the training rows were: same windows, same causality, same
missing-value rules. The only difference is *where the rows come from* — history
during training, a short recent window at serve time.

The end product is the "feature addition": :func:`attach_predictions` hangs the
:class:`~rl_crypto.ml.types.Outlook` (and its flat ``as_features`` view) off a
live :class:`~rl_crypto.models.UnifiedRecord`, so downstream RL/analysis code
sees model output as just more columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..dataset import build_dataset, interval_seconds
from ..features import DEFAULT_INTERVAL_SECONDS, dataset_interval, latest_complete
from ..models import Dataset, DatasetRow, UnifiedRecord, to_iso, utcnow
from .store import load_bundle
from .tier1 import Tier1Bundle
from .tier2 import compose_outlook
from .types import Outlook

#: Intervals of history fetched by :func:`predict_live`. Comfortably above the
#: longest feature window (``MA_LONG`` = 24) plus room for sparse news rows.
DEFAULT_WINDOW = 200


def _empty(ticker: str, bundle: Tier1Bundle, generated_at: str | None, note: str) -> Outlook:
    return Outlook(
        ticker=ticker,
        generated_at=generated_at,
        horizon=bundle.horizon,
        feature_timestamp=None,
        notes=(note,),
    )


def outlook_for_rows(
    rows: Sequence[DatasetRow],
    bundle: Tier1Bundle,
    *,
    ticker: str,
    generated_at: str | None = None,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> Outlook:
    """Predict from the newest row in ``rows`` that has a complete feature set.

    ``interval_seconds`` must match the cadence ``rows`` are actually stored at
    — the time-aware windows measure gaps in real time, so passing the wrong
    cadence silently withholds every windowed feature.
    """
    stamp = generated_at or to_iso(utcnow())
    if not rows:
        return _empty(ticker, bundle, stamp, "window has no rows")
    feature_row = latest_complete(
        rows, names=bundle.feature_names, interval_seconds=interval_seconds
    )
    if feature_row is None:
        return _empty(
            ticker,
            bundle,
            stamp,
            f"no row with all {len(bundle.feature_names)} features (need more history)",
        )
    return compose_outlook(
        bundle, ticker=ticker, feature_row=feature_row, generated_at=stamp
    )


def outlook_for_dataset(
    dataset: Dataset,
    bundle: Tier1Bundle,
    *,
    generated_at: str | None = None,
) -> Outlook:
    """Predict from a built :class:`~rl_crypto.models.Dataset`.

    The dataset's own ``interval`` sets the cadence, matching how its samples
    were built.
    """
    return outlook_for_rows(
        dataset.rows,
        bundle,
        ticker=dataset.ticker,
        generated_at=generated_at,
        interval_seconds=dataset_interval(dataset),
    )


def predict_live(
    ticker: str,
    bundle: Tier1Bundle,
    *,
    timeframe: str = "1h",
    window: int = DEFAULT_WINDOW,
    granularity: str = "hour",
    market_config: Any = None,
    news_config: Any = None,
    generated_at: str | None = None,
) -> Outlook:
    """Fetch a recent window and predict — the network-touching serve path.

    Kept separate from :func:`outlook_for_rows` so the prediction logic is
    testable offline with hand-built rows.
    """
    end = utcnow()
    start = end - timedelta(seconds=interval_seconds(timeframe) * max(1, window))
    dataset = build_dataset(
        ticker,
        start=start,
        end=end,
        timeframe=timeframe,
        granularity=granularity,
        market_config=market_config,
        news_config=news_config,
    )
    return outlook_for_dataset(dataset, bundle, generated_at=generated_at)


def load_bundle_for(ticker: str, path: str | Path | None = None) -> Tier1Bundle:
    """Load a bundle from ``path``, or the conventional ``models/<slug>.json``."""
    from .store import default_bundle_path

    resolved = Path(path) if path is not None else default_bundle_path(ticker)
    return load_bundle(resolved)


def attach_predictions(record: UnifiedRecord, outlook: Outlook) -> UnifiedRecord:
    """Attach an outlook to a live record — the feature addition, in place."""
    record.predictions = outlook.to_dict()
    return record


__all__ = [
    "DEFAULT_WINDOW",
    "attach_predictions",
    "load_bundle_for",
    "outlook_for_dataset",
    "outlook_for_rows",
    "predict_live",
]
