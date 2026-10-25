"""A fitted HAR-RV volatility baseline and a windowed-volatility label.

The model lab already carries a volatility baseline called ``har``. It fits an
OLS regression on trailing *means of a causal volatility proxy* (``vol_3`` /
``vol_6``), which is a HAR-shaped predictor but not the textbook HAR-RV one: the
regressors are averages of a standard-deviation proxy, not realised variances
built from squared returns.

This module supplies the textbook version:

* the predictors are realised variance over the trailing 1, 24 and 168 bars,
  computed from squared log returns of the price series itself;
* the regression is fitted in log space, on the **training slice only**, and
  never sees a test row;
* predictions for a row use only returns at or before that row, so the baseline
  is causal by construction.

It also supplies :func:`forward_volatility_window`, a realised-volatility label
measured over a forward window of ``window`` bars (the diagnostic variant used
by ``tools/baseline_report.py``). That label is *diagnostic only*: it is not
registered in :data:`rl_crypto.targets.TARGET_KINDS`, so it can never be trained
by the production pipeline or consume a ledger trial.

Why the split matters: a baseline that is fitted on the same slice it is scored
on would collapse to "the model" and make every comparison against it
meaningless. Every fit here is therefore restricted to ``rows[:n_train]`` and
Fitted-HAR is compared out of sample, exactly like the other baselines.

Everything is deterministic: no sampling, no global state, numpy only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .features import DEFAULT_INTERVAL_SECONDS
from .models import DatasetRow
from .targets import _ahead_by, _log_ret

#: HAR-RV regressor windows (bars): 1h, 24h, 168h.
HAR_RV_WINDOWS: tuple[int, ...] = (1, 24, 168)

#: Minimum complete training rows for the OLS fit. Below this the three-regressor
#: design plus intercept is too saturated to mean anything, and the caller falls
#: back to a rolling mean.
MIN_HAR_RV_ROWS = 30

#: Default forward window for the diagnostic volatility label (bars).
DIAGNOSTIC_VOL_WINDOW = 24

#: The volatility baseline family reported by the diagnostic, fitted HAR-RV
#: included. Kept here (not in the lab's ``BASELINE_MODES``) so adding it cannot
#: silently change a verdict that was already computed without it.
VOLATILITY_BASELINES: tuple[str, ...] = (
    "constant",
    "seasonal",
    "persistence",
    "ewma",
    "har",
    "har_rv",
)


def realised_variance(
    returns: Sequence[float | None], index: int, window: int
) -> float | None:
    """Mean squared log return over ``window`` bars ending at ``index``.

    ``None`` when the window is not fully available or any value is missing: a
    silently shortened window would let an early row masquerade as a full 168-bar
    regressor and bias the fit. Mean (not sum) keeps the three regressors on the
    same scale so their coefficients are comparable across windows.
    """
    span = max(1, int(window))
    start = index - span + 1
    if start < 0:
        return None
    window_values = returns[start : index + 1]
    if len(window_values) != span:
        return None
    squared: list[float] = []
    for value in window_values:
        if value is None:
            return None
        squared.append(float(value) ** 2)
    return float(np.mean(np.asarray(squared, dtype=float)))


def rv_components(
    returns: Sequence[float | None],
    index: int,
    *,
    windows: Sequence[int] = HAR_RV_WINDOWS,
) -> tuple[float, ...] | None:
    """The realised-variance components for one row, or ``None`` if incomplete."""
    components: list[float] = []
    for window in windows:
        value = realised_variance(returns, index, window)
        if value is None:
            return None
        components.append(value)
    return tuple(components)


def _design_row(components: Sequence[float]) -> list[float] | None:
    """``[1, log(rv_1), log(rv_24), log(rv_168)]`` or ``None`` if any is non-positive."""
    values = [1.0]
    for component in components:
        if component <= 0.0:
            return None
        values.append(float(np.log(component)))
    return values


def fit_log_har_rv(
    returns: Sequence[float | None],
    targets: Sequence[float | None],
    n_train: int,
    *,
    windows: Sequence[int] = HAR_RV_WINDOWS,
    min_rows: int = MIN_HAR_RV_ROWS,
) -> np.ndarray | None:
    """OLS coefficients fitted on ``rows[:n_train]`` only.

    The response is ``log(target)``; rows with a non-positive target, a missing
    target, or an incomplete regressor window are skipped. Returns ``None`` when
    there are too few rows (``min_rows``) or the least-squares solve fails, so
    the caller can fall back to a rolling mean instead of inventing a fit.
    """
    limit = max(0, min(int(n_train), len(targets), len(returns)))
    design: list[list[float]] = []
    response: list[float] = []
    for index in range(limit):
        truth = targets[index]
        if truth is None or float(truth) <= 0.0:
            continue
        components = rv_components(returns, index, windows=windows)
        if components is None:
            continue
        vector = _design_row(components)
        if vector is None:
            continue
        design.append(vector)
        response.append(float(np.log(float(truth))))
    if len(design) < max(1, int(min_rows)):
        return None
    matrix = np.asarray(design, dtype=float)
    vector = np.asarray(response, dtype=float)
    try:
        coefficients, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coefficients)):
        return None
    return coefficients


def har_rv_predictions(
    returns: Sequence[float | None],
    targets: Sequence[float | None],
    n_train: int,
    *,
    windows: Sequence[int] = HAR_RV_WINDOWS,
    min_rows: int = MIN_HAR_RV_ROWS,
) -> list[float | None] | None:
    """Out-of-sample fitted-HAR-RV predictions for rows ``n_train..len(returns)``.

    ``returns`` holds the causal lag-1 log return for every row (``None`` where
    unavailable); ``targets`` holds the label being predicted. Coefficients come
    from :func:`fit_log_har_rv` on ``rows[:n_train]`` only. A row whose regressor
    window is incomplete yields ``None`` so the caller can substitute its
    deterministic fallback; ``None`` for the whole series means "no usable fit".
    """
    coefficients = fit_log_har_rv(
        returns, targets, n_train, windows=windows, min_rows=min_rows
    )
    if coefficients is None:
        return None
    start = max(0, int(n_train))
    out: list[float | None] = []
    for index in range(start, len(returns)):
        components = rv_components(returns, index, windows=windows)
        if components is None:
            out.append(None)
            continue
        vector = _design_row(components)
        if vector is None:
            out.append(None)
            continue
        value = float(np.dot(coefficients, np.asarray(vector, dtype=float)))
        out.append(float(np.exp(value)) if np.isfinite(value) else None)
    return out


def forward_volatility_window(
    rows: Sequence[DatasetRow],
    index: int,
    window: int = DIAGNOSTIC_VOL_WINDOW,
    *,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> float | None:
    """Realised volatility (stdev of log returns) over the next ``window`` bars.

    The diagnostic counterpart of :func:`rl_crypto.targets.forward_volatility`,
    which measures at least two bars. Requiring ``window`` bars makes the label
    symmetric with the HAR-RV regressors; ``None`` is returned when the forward
    row is not exactly ``window`` intervals away or any bar in the window is
    missing, so a gap cannot shorten the measurement silently.
    """
    span = max(1, int(window))
    if not _ahead_by(rows, index, span, interval_seconds):
        return None
    values: list[float] = []
    for offset in range(1, span + 1):
        if not _ahead_by(rows, index + offset - 1, 1, interval_seconds):
            return None
        ret = _log_ret(rows[index + offset - 1].close, rows[index + offset].close)
        if ret is None:
            return None
        values.append(ret)
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


__all__ = [
    "DIAGNOSTIC_VOL_WINDOW",
    "HAR_RV_WINDOWS",
    "MIN_HAR_RV_ROWS",
    "VOLATILITY_BASELINES",
    "fit_log_har_rv",
    "forward_volatility_window",
    "har_rv_predictions",
    "realised_variance",
    "rv_components",
]
