"""Tier 1 labels — forward-looking targets built from future rows.

Every target here looks *forward* from row ``i`` by a ``horizon`` of intervals.
That is the whole point of a supervised setup, and it is also the easiest place
to leak: the feature builder (:mod:`rl_crypto.features`) only ever reads rows
``<= i``, so a label that reads ``i + horizon`` cannot contaminate it. A label is
``None`` whenever the future it needs is unavailable (past the end of the window,
or a missing bar) — the training step then simply drops that row rather than
inventing a value.

The four targets:

``fwd_return``
    ``log(close_{i+h} / close_i)`` — the realised return over the horizon.
``fwd_direction``
    ``1.0`` when ``fwd_return > 0`` else ``0.0`` — the sign, as a binary class.
``fwd_volatility``
    population stdev of the log returns inside the horizon — realised risk.
``fwd_volume_ratio``
    ``log(mean(volume in horizon) / mean(volume in trailing window))`` — forward
    activity relative to what came before.

Every target is anchored to *time*, not to row position. A label is only
defined when the future it needs sits exactly ``horizon`` intervals away: a
forward return measured across a data gap is not an ``h``-interval return, it is
a longer one wearing the wrong label, and it would be indistinguishable from a
genuine one in training. Missing the exact spacing yields ``None`` — the same
"drop the row rather than invent a value" rule the rest of the pipeline uses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import pstdev
from typing import Any

from .dataset import parse_iso
from .features import DEFAULT_INTERVAL_SECONDS, _log_ret, _mean
from .models import DatasetRow

#: Default forward horizon, in intervals (predict the next bar).
HORIZON_DEFAULT = 1

#: Trailing window whose mean volume is the denominator of ``fwd_volume_ratio``.
TRAILING_VOLUME_WINDOW = 12

#: target name -> task kind, consumed by the Tier-1 trainer.
TARGET_KINDS: dict[str, str] = {
    "fwd_return": "regression",
    "fwd_direction": "classification",
    "fwd_volatility": "regression",
    "fwd_volume_ratio": "regression",
}

TARGET_NAMES: tuple[str, ...] = tuple(TARGET_KINDS)

TARGET_DESCRIPTIONS: dict[str, str] = {
    "fwd_return": "log(close_{i+h} / close_i)",
    "fwd_direction": "1 if fwd_return > 0 else 0",
    "fwd_volatility": "stdev of the log returns over the next max(h, 2) intervals",
    "fwd_volume_ratio": "log(mean(volume, i..i+h) / mean(volume, trailing))",
}

#: The smallest span a realised volatility can be measured over.
MIN_VOL_SPAN = 2


def _ahead_by(
    rows: Sequence[DatasetRow], index: int, span: int, interval_seconds: int
) -> bool:
    """True when ``rows[index + span]`` sits exactly ``span`` intervals ahead.

    Row-counting alone would accept "the next bar whenever it turns up"; this
    is what makes a label's horizon mean what its name says.
    """
    if span < 1 or index + span >= len(rows):
        return False
    here, there = parse_iso(rows[index].timestamp), parse_iso(rows[index + span].timestamp)
    if here is None or there is None:
        return False
    return (there - here).total_seconds() == span * interval_seconds


def forward_return(
    rows: Sequence[DatasetRow],
    index: int,
    horizon: int = HORIZON_DEFAULT,
    *,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> float | None:
    """Realised log return from ``index`` to ``index + horizon`` intervals."""
    if not _ahead_by(rows, index, horizon, interval_seconds):
        return None
    return _log_ret(rows[index].close, rows[index + horizon].close)


def forward_direction(
    rows: Sequence[DatasetRow],
    index: int,
    horizon: int = HORIZON_DEFAULT,
    *,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> float | None:
    """``1.0`` if the forward return is positive, else ``0.0`` (``None`` if unknown)."""
    ret = forward_return(rows, index, horizon, interval_seconds=interval_seconds)
    if ret is None:
        return None
    return 1.0 if ret > 0.0 else 0.0


def forward_volatility(
    rows: Sequence[DatasetRow],
    index: int,
    horizon: int = HORIZON_DEFAULT,
    *,
    min_span: int = MIN_VOL_SPAN,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> float | None:
    """Realised volatility over the next ``max(horizon, min_span)`` intervals.

    A single forward return carries no dispersion, so the measurement span is
    never smaller than ``min_span`` (2) — with the default ``horizon=1`` you get
    the two-interval realised vol, which is a coherent short-window volatility.
    """
    span = max(horizon, min_span)
    if not _ahead_by(rows, index, span, interval_seconds):
        return None
    returns: list[float] = []
    for j in range(index + 1, index + span + 1):
        ret = _log_ret(rows[j - 1].close, rows[j].close)
        if ret is None:
            return None
        returns.append(ret)
    return pstdev(returns) if len(returns) >= 2 else None


def forward_volume_ratio(
    rows: Sequence[DatasetRow],
    index: int,
    horizon: int = HORIZON_DEFAULT,
    *,
    trailing: int = TRAILING_VOLUME_WINDOW,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> float | None:
    """Forward volume relative to the trailing window, as a log ratio."""
    if index - trailing + 1 < 0:
        return None
    if not _ahead_by(rows, index, horizon, interval_seconds):
        return None
    # The trailing window must be contiguous too, or "relative to what came
    # before" is measured against a differently-spaced run of bars.
    previous: Any = None
    for j in range(index - trailing + 1, index + 1):
        current = parse_iso(rows[j].timestamp)
        if current is None:
            return None
        if previous is not None and (current - previous).total_seconds() != interval_seconds:
            return None
        previous = current
    future = [_positive(rows[j].volume) for j in range(index + 1, index + horizon + 1)]
    past = [_positive(rows[j].volume) for j in range(index - trailing + 1, index + 1)]
    if any(v is None for v in future) or any(v is None for v in past):
        return None
    future_mean = _mean([v for v in future if v is not None])
    past_mean = _mean([v for v in past if v is not None])
    if future_mean is None or past_mean is None or future_mean <= 0.0 or past_mean <= 0.0:
        return None
    return math.log(future_mean / past_mean)


def _positive(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out >= 0.0 else None


def compute_targets(
    rows: Sequence[DatasetRow],
    index: int,
    *,
    horizon: int = HORIZON_DEFAULT,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> dict[str, float | None]:
    """Every target for ``rows[index]`` at the given ``horizon``."""
    return {
        "fwd_return": forward_return(rows, index, horizon, interval_seconds=interval_seconds),
        "fwd_direction": forward_direction(rows, index, horizon, interval_seconds=interval_seconds),
        "fwd_volatility": forward_volatility(
            rows, index, horizon, interval_seconds=interval_seconds
        ),
        "fwd_volume_ratio": forward_volume_ratio(
            rows, index, horizon, interval_seconds=interval_seconds
        ),
    }


def build_target_table(
    rows: Sequence[DatasetRow],
    *,
    horizon: int = HORIZON_DEFAULT,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> tuple[dict[str, Any], ...]:
    """One ``{"timestamp", **targets}`` dict per row."""
    return tuple(
        {
            "timestamp": rows[index].timestamp,
            **compute_targets(
                rows, index, horizon=horizon, interval_seconds=interval_seconds
            ),
        }
        for index in range(len(rows))
    )


__all__ = [
    "HORIZON_DEFAULT",
    "MIN_VOL_SPAN",
    "TARGET_DESCRIPTIONS",
    "TARGET_KINDS",
    "TARGET_NAMES",
    "TRAILING_VOLUME_WINDOW",
    "build_target_table",
    "compute_targets",
    "forward_direction",
    "forward_return",
    "forward_volatility",
    "forward_volume_ratio",
]
