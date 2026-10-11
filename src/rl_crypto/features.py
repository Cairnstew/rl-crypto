"""Tier 0 — deterministic, causal features from the joined dataset.

This is the bottom of the tiered ML pipeline. Everything here is a pure
function of the rows **at or before** a given index: no target, no future bar,
no fitted parameter. That matters twice over:

* **No leakage.** A feature at row ``i`` is computable the moment row ``i``
  closes, so training labels (which look forward) can never bleed into the
  inputs.
* **Train/serve parity.** The live path (:mod:`rl_crypto.ml.serve`) builds the
  *same* feature row from a recent window that training built from history, so
  the model sees one feature definition, not two.

Feature groups are the stable input contract; :data:`FEATURE_NAMES` is the
canonical order a model expects. Missing inputs become ``None`` — a feature is
never imputed to a fabricated value except for the one *explicit* case below.

Encoding absence
----------------
Price-derived features are ``None`` when the row has no price or lacks enough
history; a model trained on them simply skips those rows. News features are
different: a sparse archive would otherwise drop most rows. So they are
**encoded**, not imputed: ``sentiment`` is ``0.0`` when there is no news and
``has_news`` is ``0`` — the indicator tells the model that ``0`` means "no
news", not "neutral sentiment". Nothing else is fabricated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

from .dataset import parse_iso
from .models import Dataset, DatasetRow

FEATURE_VERSION = "rl-crypto/features/1"

#: Fixed look-back windows, in rows (intervals).
MA_SHORT, MA_LONG = 6, 24
VOL_SHORT, VOL_LONG = 3, 6
RETURN_LAGS = (1, 3, 6)
VOLUME_Z_WINDOW = 12
EWMA_SPAN = 6

#: Feature groups in emit order. The concatenation is :data:`FEATURE_NAMES`.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "return": ("ret_1", "ret_3", "ret_6"),
    "volatility": ("vol_3", "vol_6"),
    "bar": ("range_1", "body_1"),
    "trend": ("ma_gap_6", "ma_gap_24"),
    "volume": ("volume_log", "volume_z_12"),
    "news": (
        "sentiment",
        "sentiment_delta",
        "sentiment_ewma_6",
        "articles_log",
        "novelty",
        "has_news",
    ),
    "calendar": ("hour_sin", "hour_cos", "dow_sin", "dow_cos"),
}

FEATURE_NAMES: tuple[str, ...] = tuple(
    name for group in FEATURE_GROUPS.values() for name in group
)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "ret_1": "log(close_i / close_{i-1})",
    "ret_3": "log(close_i / close_{i-3})",
    "ret_6": "log(close_i / close_{i-6})",
    "vol_3": "stdev of the last 3 log returns",
    "vol_6": "stdev of the last 6 log returns",
    "range_1": "(high - low) / close",
    "body_1": "(close - open) / close",
    "ma_gap_6": "close / SMA(6) - 1",
    "ma_gap_24": "close / SMA(24) - 1",
    "volume_log": "log1p(volume)",
    "volume_z_12": "(volume - mean(volume, 12)) / stdev(volume, 12)",
    "sentiment": "bucket mean sentiment, or 0.0 when has_news is 0",
    "sentiment_delta": "sentiment_i - sentiment_{i-1}",
    "sentiment_ewma_6": "mean sentiment over the last 6 rows that had news",
    "articles_log": "log1p(article_count)",
    "novelty": "1.0 when any novelty flag in the bucket, else 0.0",
    "has_news": "1.0 when the bucket has news, else 0.0",
    "hour_sin": "sin(2*pi*hour/24)",
    "hour_cos": "cos(2*pi*hour/24)",
    "dow_sin": "sin(2*pi*weekday/7)",
    "dow_cos": "cos(2*pi*weekday/7)",
}

#: Rows of price history required before a price feature set is complete.
MIN_HISTORY = MA_LONG


@dataclass(slots=True)
class FeatureRow:
    """One row's feature vector plus its completeness metadata."""

    timestamp: str
    values: dict[str, float | None]

    @property
    def missing(self) -> tuple[str, ...]:
        """Feature names that are ``None`` (unavailable), in canonical order."""
        return tuple(name for name in FEATURE_NAMES if self.values.get(name) is None)

    @property
    def complete(self) -> bool:
        """True when every canonical feature has a value."""
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-ready row: ``timestamp`` + one key per feature + metadata."""
        return {
            "timestamp": self.timestamp,
            **self.values,
            "complete": self.complete,
            "missing": list(self.missing),
        }


# --------------------------------------------------------------------------- helpers


def _finite_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _log_ret(previous: Any, current: Any) -> float | None:
    a = _finite_or_none(previous)
    b = _finite_or_none(current)
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return None
    return math.log(b / a)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _closes(rows: Sequence[DatasetRow], start: int, stop: int) -> list[float] | None:
    """The closes in ``[start, stop]`` (inclusive), or ``None`` if any is unusable."""
    if start < 0:
        return None
    out: list[float] = []
    for j in range(start, stop + 1):
        close = _finite_or_none(rows[j].close)
        if close is None or close <= 0.0:
            return None
        out.append(close)
    return out


def _log_returns_back(rows: Sequence[DatasetRow], index: int, window: int) -> list[float]:
    """Log returns for the ``window`` steps ending at ``index`` (causal)."""
    out: list[float] = []
    for j in range(max(1, index - window + 1), index + 1):
        ret = _log_ret(rows[j - 1].close, rows[j].close)
        if ret is not None:
            out.append(ret)
    return out


def _return(rows: Sequence[DatasetRow], index: int, lag: int) -> float | None:
    if index - lag < 0:
        return None
    return _log_ret(rows[index - lag].close, rows[index].close)


def _volatility(rows: Sequence[DatasetRow], index: int, window: int) -> float | None:
    returns = _log_returns_back(rows, index, window)
    return pstdev(returns) if len(returns) >= 2 else None


def _ma_gap(rows: Sequence[DatasetRow], index: int, window: int) -> float | None:
    closes = _closes(rows, index - window + 1, index)
    if closes is None:
        return None
    moving_average = sum(closes) / window
    return closes[-1] / moving_average - 1.0


def _range(row: DatasetRow) -> float | None:
    high = _finite_or_none(row.high)
    low = _finite_or_none(row.low)
    close = _finite_or_none(row.close)
    if high is None or low is None or close is None or close == 0.0:
        return None
    return (high - low) / close


def _body(row: DatasetRow) -> float | None:
    open_ = _finite_or_none(row.open)
    close = _finite_or_none(row.close)
    if open_ is None or close is None or close == 0.0:
        return None
    return (close - open_) / close


def _volume_log(row: DatasetRow) -> float | None:
    volume = _finite_or_none(row.volume)
    if volume is None or volume < 0.0:
        return None
    return math.log1p(volume)


def _volume_z(rows: Sequence[DatasetRow], index: int, window: int) -> float | None:
    if index - window + 1 < 0:
        return None
    volumes: list[float] = []
    for j in range(index - window + 1, index + 1):
        volume = _finite_or_none(rows[j].volume)
        if volume is None or volume < 0.0:
            return None
        volumes.append(volume)
    spread = pstdev(volumes)
    if spread == 0.0:
        return None
    return (volumes[-1] - sum(volumes) / len(volumes)) / spread


def _sentiment(rows: Sequence[DatasetRow], index: int) -> float:
    """Encoded sentiment: the bucket score, or ``0.0`` when there is no news."""
    row = rows[index]
    if not row.has_news:
        return 0.0
    return _finite_or_none(row.sentiment_score) or 0.0


def _sentiment_delta(rows: Sequence[DatasetRow], index: int) -> float | None:
    if index == 0:
        return None
    return _sentiment(rows, index) - _sentiment(rows, index - 1)


def _sentiment_ewma(rows: Sequence[DatasetRow], index: int) -> float:
    scores = [
        _finite_or_none(rows[j].sentiment_score)
        for j in range(max(0, index - EWMA_SPAN + 1), index + 1)
        if rows[j].has_news
    ]
    usable = [s for s in scores if s is not None]
    return sum(usable) / len(usable) if usable else 0.0


def _calendar(timestamp: str) -> dict[str, float | None]:
    parsed = parse_iso(timestamp)
    if parsed is None:
        return {"hour_sin": None, "hour_cos": None, "dow_sin": None, "dow_cos": None}
    hour_angle = 2.0 * math.pi * parsed.hour / 24.0
    dow_angle = 2.0 * math.pi * parsed.weekday() / 7.0
    return {
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "dow_sin": math.sin(dow_angle),
        "dow_cos": math.cos(dow_angle),
    }


# --------------------------------------------------------------------------- public


def feature_row(rows: Sequence[DatasetRow], index: int) -> FeatureRow:
    """The Tier-0 feature row for ``rows[index]``, using only rows ``<= index``."""
    if not 0 <= index < len(rows):
        raise IndexError(f"index {index} out of range for {len(rows)} rows")
    row = rows[index]
    values: dict[str, float | None] = {
        "ret_1": _return(rows, index, 1),
        "ret_3": _return(rows, index, 3),
        "ret_6": _return(rows, index, 6),
        "vol_3": _volatility(rows, index, VOL_SHORT),
        "vol_6": _volatility(rows, index, VOL_LONG),
        "range_1": _range(row),
        "body_1": _body(row),
        "ma_gap_6": _ma_gap(rows, index, MA_SHORT),
        "ma_gap_24": _ma_gap(rows, index, MA_LONG),
        "volume_log": _volume_log(row),
        "volume_z_12": _volume_z(rows, index, VOLUME_Z_WINDOW),
        "sentiment": _sentiment(rows, index),
        "sentiment_delta": _sentiment_delta(rows, index),
        "sentiment_ewma_6": _sentiment_ewma(rows, index),
        "articles_log": math.log1p(max(0, int(row.article_count or 0))),
        "novelty": 1.0 if row.novelty_flag else 0.0,
        "has_news": 1.0 if row.has_news else 0.0,
        **_calendar(row.timestamp),
    }
    return FeatureRow(timestamp=row.timestamp, values=values)


def build_feature_table(rows: Sequence[DatasetRow]) -> tuple[FeatureRow, ...]:
    """One :class:`FeatureRow` per row, in order."""
    return tuple(feature_row(rows, index) for index in range(len(rows)))


def build_dataset_features(dataset: Dataset) -> tuple[FeatureRow, ...]:
    """Features for a whole :class:`~rl_crypto.models.Dataset`."""
    return build_feature_table(dataset.rows)


def latest_complete(
    rows: Sequence[DatasetRow],
    *,
    names: Sequence[str] = FEATURE_NAMES,
) -> FeatureRow | None:
    """The newest row whose ``names`` are all present — the live inference row.

    Returns ``None`` when no row in the window qualifies (e.g. too little
    history), so a caller reports "no outlook" rather than a fabricated one.
    """
    for index in range(len(rows) - 1, -1, -1):
        candidate = feature_row(rows, index)
        if all(candidate.values.get(name) is not None for name in names):
            return candidate
    return None


def as_vector(row: FeatureRow, names: Sequence[str] = FEATURE_NAMES) -> list[float | None]:
    """The row's values in ``names`` order (``None`` preserved for the caller)."""
    return [row.values.get(name) for name in names]


__all__ = [
    "FEATURE_DESCRIPTIONS",
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "FeatureRow",
    "as_vector",
    "build_dataset_features",
    "build_feature_table",
    "feature_row",
    "latest_complete",
]
