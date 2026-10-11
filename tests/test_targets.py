"""Tier-1 labels: forward-looking targets and their None rules."""

from __future__ import annotations

import math

import pytest

from rl_crypto.targets import (
    compute_targets,
    forward_direction,
    forward_return,
    forward_volatility,
    forward_volume_ratio,
)

from .conftest import make_dataset_rows


def test_forward_return_is_the_log_ratio():
    rows = make_dataset_rows(10)
    got = forward_return(rows, 3, horizon=2)
    now = rows[3].close
    later = rows[5].close
    assert now is not None and later is not None
    assert got == math.log(later / now)


def test_forward_direction_matches_the_sign():
    rows = make_dataset_rows(40, wiggle=0.05)
    ret = forward_return(rows, 20, horizon=1)
    assert forward_direction(rows, 20, horizon=1) == (1.0 if ret and ret > 0 else 0.0)


def test_forward_volatility_of_constant_returns_is_zero():
    # close_i = base * exp(drift * i) doubles every step -> constant log2 returns.
    rows = make_dataset_rows(10, base=1.0, drift=math.log(2.0))
    assert forward_volatility(rows, 1, horizon=3) == pytest.approx(0.0, abs=1e-12)


def test_forward_volume_ratio_uses_future_over_trailing():
    rows = make_dataset_rows(30)
    ratio = forward_volume_ratio(rows, 20, horizon=2)
    future = [float(v) for v in (rows[21].volume, rows[22].volume) if v is not None]
    past = [float(v) for v in (row.volume for row in rows[9:21]) if v is not None]
    assert len(future) == 2 and len(past) == 12
    future_mean = sum(future) / 2.0
    past_mean = sum(past) / 12.0
    assert ratio == math.log(future_mean / past_mean)


def test_targets_are_none_past_the_window():
    rows = make_dataset_rows(5)
    for name, value in compute_targets(rows, 4, horizon=1).items():
        assert value is None, name


def test_targets_are_none_when_future_has_no_price():
    rows = [
        *make_dataset_rows(5),
    ]
    # Force the bar after row 3 to be news-only (no closes).
    from rl_crypto.models import DatasetRow

    rows = (*rows[:4], DatasetRow(ticker="ETH/USD", timestamp="x", has_news=True))
    assert forward_return(rows, 3, horizon=1) is None