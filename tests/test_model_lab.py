"""Offline tests for the rl-model-lab engine's pure scoring helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from rl_crypto.samples import Sample

_SPEC = importlib.util.spec_from_file_location(
    "model_lab", Path(__file__).resolve().parents[1] / "tools" / "model_lab.py"
)
assert _SPEC is not None and _SPEC.loader is not None
lab = importlib.util.module_from_spec(_SPEC)
sys.modules["model_lab"] = lab
_SPEC.loader.exec_module(lab)


def _samples(n: int = 200) -> list[Sample]:
    return [
        Sample(
            timestamp=f"2026-01-01T{i % 24:02d}:00:00Z",
            features={"ret_1": 0.01 if i % 2 == 0 else -0.01},
            targets={
                "fwd_return": 0.01 if i % 3 == 0 else -0.005,
                "fwd_direction": 1.0 if i % 3 == 0 else 0.0,
            },
        )
        for i in range(n)
    ]


def test_walk_forward_is_chronological_and_purges_the_horizon():
    samples = _samples(200)
    splits = lab.walk_forward(samples, folds=3, horizon=2, min_train_fraction=0.3)
    assert len(splits) == 3
    for train, test in splits:
        assert train and test
        # The purge keeps the last `horizon` labels out of training.
        assert train[-1].timestamp != test[0].timestamp
        assert len(train) >= 20


def test_walk_forward_refuses_too_few_samples():
    assert lab.walk_forward(_samples(20), folds=3) == []


def test_pnl_metrics_charges_costs_and_handles_flat():
    returns = [0.01, 0.02, 0.005]
    winning = lab._pnl_metrics([1.0, 1.0, 1.0], returns, fee_bps=0, slippage_bps=0)
    assert winning["total_return"] > 0
    assert winning["trades"] == 1
    assert winning["sharpe"] is not None

    costly = lab._pnl_metrics([1.0, 1.0, 1.0], returns, fee_bps=100, slippage_bps=50)
    assert costly["total_return"] < winning["total_return"]

    flat = lab._pnl_metrics([0.0, 0.0, 0.0], returns, fee_bps=10, slippage_bps=5)
    assert flat["total_return"] == pytest.approx(0.0)
    assert flat["trades"] == 0


def test_feature_sets_are_explicit():
    market = lab.resolve_feature_names("market")
    assert "sentiment" not in market
    assert "ret_1" in market
    core = lab.resolve_feature_names("core")
    assert "sentiment" not in core
    assert "range_1" not in core  # bar group excluded from core
    price = lab.resolve_feature_names("price")
    assert "range_1" in price
    with pytest.raises(SystemExit):
        lab.resolve_feature_names("not-a-group")


def test_relative_improvement_is_relative_to_the_baseline():
    assert lab.relative_improvement("regression", {"rmse": 9.0}, {"rmse": 10.0}) == pytest.approx(
        0.1
    )
    assert lab.relative_improvement(
        "classification", {"log_loss": 0.69}, {"log_loss": 0.70}
    ) == pytest.approx((0.70 - 0.69) / 0.70)
    assert lab.relative_improvement("regression", {"rmse": None}, {"rmse": 1.0}) is None


def test_promotion_gate_accepts_a_real_improvement_and_rejects_a_tie():
    candidate = {
        "targets": {"fwd_direction": {"beats": True, "model": 0.65}},
        "economic": {"model": {"trades": 20, "sharpe": 0.5, "max_drawdown": -0.1}},
    }
    current = {
        "targets": {"fwd_direction": {"beats": False, "model": 0.70}},
        "economic": {"model": {"trades": 20, "sharpe": 0.2, "max_drawdown": -0.1}},
    }
    assert lab.promotion_gate(candidate, current, "fwd_direction") == []

    tied = {
        "targets": {"fwd_direction": {"beats": True, "model": 0.70}},
        "economic": {"model": {"trades": 20, "sharpe": 0.5, "max_drawdown": -0.1}},
    }
    reasons = lab.promotion_gate(tied, current, "fwd_direction")
    assert any("1%" in reason for reason in reasons)


def test_bars_per_year_is_hourly():
    assert math.isclose(lab.BARS_PER_YEAR, 365 * 24, rel_tol=1e-9)


def test_help_lists_actions_and_flags(capsys):
    assert lab.main(["help"]) == 0
    out = capsys.readouterr().out
    for action in ("status", "evaluate", "sweep", "promote", "report", "help"):
        assert action in out
    assert "--candidate" in out
    assert "final 20%" in out
