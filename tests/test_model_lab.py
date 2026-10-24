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


# --------------------------------------------------------------------------- statistical hardening


def _stamped(n=120, *, positive=True):
    return [
        Sample(
            timestamp=f"2026-01-{1 + i // 24:02d}T{i % 24:02d}:00:00Z",
            features={"ret_1": 0.01 if i % 2 == 0 else -0.01},
            targets={"fwd_direction": 1.0 if positive else 0.0},
        )
        for i in range(n)
    ]


def test_bootstrap_ci_is_deterministic_and_separates_signal_from_noise():
    signal = {
        "BTC/USD": [{"timestamp": str(i), "improvement": 0.02} for i in range(60)],
        "ETH/USD": [{"timestamp": str(i), "improvement": 0.015} for i in range(60)],
    }
    first = lab.bootstrap_pooled(
        signal, alpha=0.05, seed=7, block_bars=6, bootstrap_samples=300
    )
    second = lab.bootstrap_pooled(
        signal, alpha=0.05, seed=7, block_bars=6, bootstrap_samples=300
    )
    assert first == second
    assert first["ci_low"] > 0
    assert first["mean_improvement"] > 0

    values = []
    for block in range(20):
        centre = 0.01 if block % 2 == 0 else -0.01
        values.extend([centre + 0.005, centre - 0.005] * 3)
    noise = {
        "BTC/USD": [
            {"timestamp": str(i), "improvement": value}
            for i, value in enumerate(values)
        ]
    }
    null = lab.bootstrap_pooled(
        noise, alpha=0.05, seed=7, block_bars=6, bootstrap_samples=300
    )
    assert null["ci_low"] < 0


def test_controls_destroy_alignment_without_reading_new_bars():
    samples = [
        Sample(
            timestamp=f"2026-01-01T{i:02d}:00:00Z",
            features={"ret_1": 0.01},
            targets={"fwd_direction": 1.0 if i % 2 else 0.0},
        )
        for i in range(60)
    ]
    shuffled = lab.transform_targets(samples, "fwd_direction", "shuffle", seed=3, shift_bars=4)
    assert [s.timestamp for s in shuffled] == [s.timestamp for s in samples]
    # Same multiset of labels, but not the same alignment.
    assert sorted(s.targets["fwd_direction"] for s in shuffled) == sorted(
        s.targets["fwd_direction"] for s in samples
    )
    assert [s.targets["fwd_direction"] for s in shuffled] != [
        s.targets["fwd_direction"] for s in samples
    ]

    shifted = lab.transform_targets(samples, "fwd_direction", "shift", seed=3, shift_bars=4)
    assert shifted[0].targets["fwd_direction"] is None
    assert shifted[4].targets["fwd_direction"] == samples[0].targets["fwd_direction"]


def test_holdout_filter_and_split_use_a_timestamp_not_a_fraction():
    rows = [
        type("Row", (), {"timestamp": f"2026-01-01T{index:02d}:00:00Z"})()
        for index in range(10)
    ]
    kept, cutoff = lab.filter_rows_before(rows, "2026-01-01T06:00:00Z")
    assert cutoff is not None
    assert [row.timestamp for row in kept] == [row.timestamp for row in rows[:6]]

    samples = _stamped(10)
    split = lab.sample_split(samples, "2026-01-01T06:00:00Z")
    assert split == 6


def test_walk_forward_purges_the_volatility_span_even_at_horizon_one():
    samples = _stamped(240)
    splits = lab.walk_forward(samples, folds=3, horizon=1)
    assert splits
    for train, test in splits:
        test_start = next(
            index for index, sample in enumerate(samples) if sample.timestamp == test[0].timestamp
        )
        train_end = next(
            index for index, sample in enumerate(samples) if sample.timestamp == train[-1].timestamp
        )
        # fwd_volatility reads i+1 and i+2, so the last training row must be
        # at least two rows before the first test row.
        assert train_end <= test_start - 2


def test_the_old_fold_rule_passes_pure_noise_more_often_than_the_pooled_ci():
    import random

    rng = random.Random(1234)
    old_passes = 0
    new_passes = 0
    trials = 200
    for _ in range(trials):
        folds = [rng.gauss(0.0, 0.01) for _ in range(4)]
        if all(value > 0 for value in folds):
            old_passes += 1
        rows = {
            "BTC/USD": [
                {"timestamp": str(i), "improvement": rng.gauss(0.0, 0.01)}
                for i in range(600)
            ]
        }
        pooled = lab.bootstrap_pooled(
            rows, alpha=0.05, seed=7, block_bars=24, bootstrap_samples=300
        )
        if pooled["ci_low"] is not None and pooled["ci_low"] > 0 and pooled["mean_improvement"] > 0:
            new_passes += 1

    # The v1 rule's minimum-improvement bar is really "all four folds positive",
    # which noise clears about 6% of the time. The pooled CI should be at or
    # below the nominal alpha and strictly rarer.
    assert old_passes / trials > 0.01
    assert new_passes / trials <= 0.06
    assert new_passes < old_passes


def test_the_pooled_rule_passes_a_real_signal():
    rows = {
        "BTC/USD": [
            {"timestamp": str(i), "improvement": 0.02 + (0.001 if i % 2 else -0.001)}
            for i in range(80)
        ],
        "ETH/USD": [
            {"timestamp": str(i), "improvement": 0.015 + (0.001 if i % 2 else -0.001)}
            for i in range(80)
        ],
    }
    pooled = lab.bootstrap_pooled(
        rows, alpha=0.05, seed=7, block_bars=8, bootstrap_samples=300
    )
    assert pooled["mean_improvement"] > 0
    assert pooled["ci_low"] > 0
