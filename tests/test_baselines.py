"""Offline tests for the fitted HAR-RV baseline and the windowed-vol label.

The properties that matter are the ones a baseline can quietly cheat on:

* the regressors are causal (only past returns),
* the OLS fit sees the training slice only,
* a missing window yields ``None`` instead of a silently shortened estimate,
* the diagnostic label is not a registered production target.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import pstdev

import numpy as np
import pytest

from rl_crypto.baselines import (
    DIAGNOSTIC_VOL_WINDOW,
    HAR_RV_WINDOWS,
    VOLATILITY_BASELINES,
    fit_log_har_rv,
    forward_volatility_window,
    har_rv_predictions,
    realised_variance,
    rv_components,
)
from rl_crypto.models import DatasetRow
from rl_crypto.samples import Sample
from rl_crypto.targets import TARGET_KINDS, TARGET_NAMES

ROOT = Path(__file__).resolve().parents[1]

_LAB_SPEC = importlib.util.spec_from_file_location("model_lab", ROOT / "tools" / "model_lab.py")
assert _LAB_SPEC is not None and _LAB_SPEC.loader is not None
lab = importlib.util.module_from_spec(_LAB_SPEC)
sys.modules["model_lab"] = lab
_LAB_SPEC.loader.exec_module(lab)


def _rows(n: int, *, step_seconds: int = 3600) -> list[DatasetRow]:
    out: list[DatasetRow] = []
    price = 100.0
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(n):
        price *= 1.0 + 0.001 * ((index % 5) - 2)
        stamp = (start + timedelta(seconds=index * step_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append(DatasetRow(ticker="BTC/USD", timestamp=stamp, close=price))
    return out


def _returns(n: int) -> list[float | None]:
    return [None if index == 0 else 0.01 * math.sin(index) for index in range(n)]


def test_realised_variance_is_the_mean_squared_return_over_the_trailing_window():
    returns = [0.5, -0.5, 0.25, 0.0]
    assert realised_variance(returns, 3, 1) == pytest.approx(0.0)
    assert realised_variance(returns, 3, 2) == pytest.approx((0.25**2 + 0.0**2) / 2)
    assert realised_variance(returns, 3, 4) == pytest.approx((0.25 + 0.25 + 0.0625 + 0.0) / 4)


def test_realised_variance_refuses_an_incomplete_or_gapped_window():
    returns: list[float | None] = [0.1, 0.2, None, 0.4]
    assert realised_variance(returns, 1, 3) is None  # not enough history
    assert realised_variance(returns, 3, 2) is None  # a missing bar in the window
    assert realised_variance(returns, 3, 1) == pytest.approx(0.16)


def test_rv_components_returns_one_value_per_har_window():
    returns = _returns(300)
    components = rv_components(returns, 250)
    assert components is not None
    assert len(components) == len(HAR_RV_WINDOWS)
    assert [realised_variance(returns, 250, w) for w in HAR_RV_WINDOWS] == list(components)
    assert rv_components(returns, 10) is None  # 168-bar history unavailable


def _log_linear_series(n: int = 400) -> tuple[list[float | None], list[float | None]]:
    """Returns whose RV components drive a known log-linear volatility target."""
    rng = np.random.default_rng(7)
    returns: list[float | None] = [None]
    for _ in range(n - 1):
        returns.append(float(rng.normal(0.0, 0.01)))
    targets: list[float | None] = []
    for index in range(n):
        components = rv_components(returns, index)
        if components is None:
            targets.append(None)
            continue
        log_rv = [math.log(value) for value in components]
        targets.append(math.exp(0.1 * log_rv[0] + 0.2 * log_rv[1] + 0.3 * log_rv[2] - 2.0))
    return returns, targets


def test_fit_log_har_rv_recovers_a_known_log_linear_relationship():
    returns, targets = _log_linear_series()
    coefficients = fit_log_har_rv(returns, targets, len(targets))
    assert coefficients is not None
    intercept, beta_1, beta_24, beta_168 = (float(value) for value in coefficients)
    assert intercept == pytest.approx(-2.0, abs=1e-8)
    assert beta_1 == pytest.approx(0.1, abs=1e-8)
    assert beta_24 == pytest.approx(0.2, abs=1e-8)
    assert beta_168 == pytest.approx(0.3, abs=1e-8)


def test_har_rv_predictions_are_positive_and_aligned_with_the_test_slice():
    returns, targets = _log_linear_series()
    predictions = har_rv_predictions(returns, targets, 300)
    assert predictions is not None
    assert len(predictions) == len(returns) - 300
    assert all(value is not None and value > 0.0 for value in predictions)


def test_har_rv_fit_ignores_targets_at_and_after_the_split():
    returns, targets = _log_linear_series()
    perturbed = list(targets)
    for index in range(300, len(perturbed)):
        perturbed[index] = 123.0
    first = har_rv_predictions(returns, targets, 300)
    second = har_rv_predictions(returns, perturbed, 300)
    assert first is not None and second is not None
    assert first == pytest.approx(second)


def test_har_rv_predictions_are_causal_in_the_return_series():
    returns, targets = _log_linear_series()
    shifted = list(returns)
    for index in range(350, len(shifted)):
        shifted[index] = 0.5  # a violently different future
    first = har_rv_predictions(returns, targets, 300)
    second = har_rv_predictions(shifted, targets, 300)
    assert first is not None and second is not None
    # Rows whose 168-bar window still ends before the perturbation must match.
    for offset in range(350 - 300 - 167):
        assert first[offset] == pytest.approx(second[offset])


def test_har_rv_refuses_to_fit_with_too_few_training_rows():
    returns, targets = _log_linear_series(60)
    assert fit_log_har_rv(returns, targets, 20) is None
    assert har_rv_predictions(returns, targets, 20) is None


def test_forward_volatility_window_is_the_stdev_of_the_forward_returns():
    rows = _rows(80)
    label = forward_volatility_window(rows, 10, 24)
    assert label is not None
    forward = [
        math.log(rows[index + 1].close / rows[index].close) for index in range(10, 34)
    ]
    assert label == pytest.approx(pstdev(forward))
    # The final rows have no complete forward window.
    assert forward_volatility_window(rows, len(rows) - 1, 24) is None
    assert forward_volatility_window(rows, len(rows) - DIAGNOSTIC_VOL_WINDOW, 24) is None


def test_forward_volatility_window_refuses_a_gapped_window():
    rows = list(_rows(60))
    # Drop one bar in the middle of the forward window so the spacing breaks.
    del rows[15]
    assert forward_volatility_window(rows, 10, 24) is None


def test_diagnostic_24_bar_label_is_not_a_registered_target():
    assert DIAGNOSTIC_VOL_WINDOW == 24
    assert "fwd_volatility_24" not in TARGET_KINDS
    assert "fwd_volatility_24" not in TARGET_NAMES


def test_volatility_baseline_family_includes_fitted_har_rv():
    assert VOLATILITY_BASELINES == (
        "constant",
        "seasonal",
        "persistence",
        "ewma",
        "har",
        "har_rv",
    )
    # The lab's default tuple is deliberately unchanged so no earlier verdict
    # is silently recomputed against a different baseline set.
    assert "har_rv" not in lab.BASELINE_MODES


def _vol_samples(n: int = 400) -> list[Sample]:
    rng = np.random.default_rng(11)
    out: list[Sample] = []
    level = 0.01
    for index in range(n):
        level = 0.9 * level + 0.1 * abs(float(rng.normal(0.0, 0.01)))
        ret = float(rng.normal(0.0, level))
        target = math.exp(math.log(level) * 0.8 - 4.0)
        out.append(
            Sample(
                timestamp=f"2026-01-{1 + index // 24:02d}T{index % 24:02d}:00:00Z",
                features={"ret_1": ret, "vol_3": level},
                targets={"fwd_volatility": target},
            )
        )
    return out


def test_lab_accepts_har_rv_as_an_explicit_mode():
    samples = _vol_samples()
    train, test = samples[:300], samples[300:]
    predictions = lab.baseline_predictions(
        "har_rv", "regression", train, test, "fwd_volatility", space="log", horizon=1
    )
    assert len(predictions) == len(test)
    assert all(value > 0.0 for value in predictions)


def test_lab_har_rv_falls_back_to_har_when_no_fit_is_possible():
    samples = _vol_samples(60)
    train, test = samples[:40], samples[40:]
    fitted = lab.baseline_predictions(
        "har_rv", "regression", train, test, "fwd_volatility", space="log", horizon=1
    )
    har = lab.baseline_predictions(
        "har", "regression", train, test, "fwd_volatility", space="log", horizon=1
    )
    assert fitted == pytest.approx(har)
