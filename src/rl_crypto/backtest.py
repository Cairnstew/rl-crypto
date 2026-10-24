"""After-cost policy economics, shared by the lab and the forward scorer.

One implementation so a backtest and a forward test cannot disagree about what
a trade costs or how a position is applied to a bar.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

#: Bars per year for the default hourly cadence, used to annualise Sharpe.
HOURLY_BARS_PER_YEAR = 365.0 * 24.0

_EMPTY = {
    "bars": 0,
    "trades": 0,
    "turnover": 0.0,
    "total_return": None,
    "sharpe": None,
    "max_drawdown": None,
    "hit_rate": None,
}


def pnl_metrics(
    positions: Sequence[float],
    returns: Sequence[float],
    *,
    fee_bps: float,
    slippage_bps: float,
    bars_per_year: float = HOURLY_BARS_PER_YEAR,
) -> dict[str, float | int | None]:
    """Return net of per-side costs, applied to each bar's position.

    ``positions[i]`` is held over ``returns[i]``. A position change is charged
    on the bar it happens, and the final position is charged again to close it —
    otherwise a strategy that ends holding a losing position would look free.
    """
    if not positions:
        return dict(_EMPTY)
    cost = (fee_bps + slippage_bps) / 10_000.0
    pnl: list[float] = []
    previous = 0.0
    trades = 0
    turnover = 0.0
    for pos, realised in zip(positions, returns, strict=True):
        delta = abs(pos - previous)
        trades += 1 if delta > 0 else 0
        turnover += delta
        pnl.append(pos * realised - cost * delta)
        previous = pos
    pnl[-1] -= cost * abs(previous)
    arr = np.asarray(pnl, dtype=float)
    equity = np.exp(np.cumsum(arr))
    peak = np.maximum.accumulate(equity)
    std = float(np.std(arr))
    return {
        "bars": len(arr),
        "trades": trades,
        "turnover": float(turnover),
        "total_return": float(equity[-1] - 1.0),
        "sharpe": float(np.mean(arr) / std * math.sqrt(bars_per_year)) if std > 0 else None,
        "max_drawdown": float(np.min(equity / peak - 1.0)),
        "hit_rate": float(np.mean(arr[arr != 0] > 0)) if np.any(arr != 0) else None,
    }


__all__ = ["HOURLY_BARS_PER_YEAR", "pnl_metrics"]
