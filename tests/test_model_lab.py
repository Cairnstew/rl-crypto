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
        signal, alpha=0.05, seed=7, block_bars=4, bootstrap_samples=1000
    )
    second = lab.bootstrap_pooled(
        signal, alpha=0.05, seed=7, block_bars=4, bootstrap_samples=1000
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
        noise, alpha=0.05, seed=7, block_bars=4, bootstrap_samples=1000
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
            rows, alpha=0.05, seed=7, block_bars=24, bootstrap_samples=1000
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
        rows, alpha=0.05, seed=7, block_bars=4, bootstrap_samples=1000
    )
    assert pooled["mean_improvement"] > 0
    assert pooled["ci_low"] > 0


def test_promote_apply_refuses_without_a_frozen_cutoff():
    with pytest.raises(SystemExit, match="requires --holdout-start"):
        lab.main(["promote", "--candidate", "candidate.json", "--apply"])


def _stamped_samples(n: int = 200) -> list[Sample]:
    return [
        Sample(timestamp=f"2026-01-01T{i:06d}:00:00Z", features={}, targets={})
        for i in range(n)
    ]


def test_walk_forward_purges_a_single_bar_for_a_one_bar_target():
    samples = _stamped_samples(200)
    splits = lab.walk_forward(samples, folds=2, horizon=1, targets=("fwd_return",))
    assert splits
    for train, test in splits:
        assert samples.index(train[-1]) + 1 <= samples.index(test[0])


def test_walk_forward_purges_the_volatility_span_not_just_the_horizon():
    """``fwd_volatility`` reads two bars at horizon 1; one-bar purge would leak."""
    samples = _stamped_samples(200)
    splits = lab.walk_forward(samples, folds=2, horizon=1, targets=("fwd_volatility",))
    assert splits
    for train, test in splits:
        assert samples.index(train[-1]) + 2 <= samples.index(test[0])


def test_walk_forward_uses_the_widest_target_reach():
    samples = _stamped_samples(200)
    splits = lab.walk_forward(
        samples, folds=2, horizon=1, targets=("fwd_return", "fwd_volatility")
    )
    assert splits
    for train, test in splits:
        assert samples.index(train[-1]) + 2 <= samples.index(test[0])


def test_reference_volatility_is_scored_in_log_space():
    from rl_crypto.ml.tier1 import train_tier1
    from rl_crypto.samples import build_samples

    from .conftest import make_dataset_rows

    samples = build_samples(make_dataset_rows(160, wiggle=0.03))
    bundle = train_tier1(
        samples,
        targets=("fwd_volatility",),
        model_names=lab.MODEL_SETS["reference"],
        holdout_fraction=0.2,
    )
    model = bundle.targets["fwd_volatility"]
    assert model.metric_space == "log"
    assert model.predict_row(samples[-1].features) is not None
    assert model.predict_row(samples[-1].features) > 0


def test_parse_thresholds_and_cost_levels():
    assert lab._parse_thresholds("0.5,0.55 0.6") == [0.5, 0.55, 0.6]
    assert lab._parse_thresholds(None) is None
    assert lab._parse_cost_levels("10:5,0:0") == [(10.0, 5.0), (0.0, 0.0)]
    assert lab._parse_cost_levels("10/5") == [(10.0, 5.0)]
    assert lab._parse_cost_levels(None) is None
    with pytest.raises(SystemExit):
        lab._parse_thresholds("0")
    with pytest.raises(SystemExit):
        lab._parse_cost_levels("10")


def _write_archive(root, rows):
    import json

    directory = root / "data" / "eth-usd"
    directory.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "ticker": row.ticker,
                "timestamp": row.timestamp,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            }
        )
        for row in rows
    ]
    (directory / "candles.jsonl").write_text("\n".join(lines) + "\n")


def test_evaluate_grid_fits_each_fold_once(tmp_path, monkeypatch):
    from .conftest import make_dataset_rows

    monkeypatch.setattr(lab, "ROOT", tmp_path)
    _write_archive(tmp_path, make_dataset_rows(240, wiggle=0.03))
    config = lab.Config(
        name="grid",
        feature_names=lab.resolve_feature_names("all"),
        model_names=lab.MODEL_SETS["baseline"],
        horizon=1,
    )

    calls = {"n": 0}
    real_train = lab.train_tier1

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real_train(*args, **kwargs)

    monkeypatch.setattr(lab, "train_tier1", spy)

    result = lab.evaluate_ticker(
        "ETH/USD",
        config,
        ("fwd_return", "fwd_direction"),
        folds=3,
        threshold=0.55,
        position="long-only",
        fee_bps=10.0,
        slippage_bps=5.0,
        thresholds=[0.5, 0.55, 0.9],
        cost_levels=[(10.0, 5.0), (0.0, 0.0)],
    )

    assert calls["n"] == result["folds"] == 3
    assert len(result["economic_grid"]) == 3 * 2

    headline = result["economic"]
    cell = next(
        row
        for row in result["economic_grid"]
        if row["threshold"] == 0.55
        and row["fee_bps"] == 10.0
        and row["slippage_bps"] == 5.0
    )
    assert headline["model"] == cell["model"]

    by_key = {
        (row["threshold"], row["fee_bps"], row["slippage_bps"]): row
        for row in result["economic_grid"]
    }
    free = by_key[(0.55, 0.0, 0.0)]["model"]["total_return"]
    costly = by_key[(0.55, 10.0, 5.0)]["model"]["total_return"]
    assert free >= costly

    low = by_key[(0.5, 10.0, 5.0)]["model"]["trades"]
    high = by_key[(0.9, 10.0, 5.0)]["model"]["trades"]
    assert high <= low


def test_promote_apply_records_the_durable_training_config(tmp_path, monkeypatch):
    """A promoted bundle only survives if the committed config names it too."""
    import importlib.util
    import json
    import types
    from datetime import UTC, datetime, timedelta

    spec = importlib.util.spec_from_file_location(
        "model_lab_promote_apply",
        Path(__file__).resolve().parents[1] / "tools" / "model_lab.py",
    )
    assert spec is not None and spec.loader is not None
    lab = importlib.util.module_from_spec(spec)
    sys.modules["model_lab_promote_apply"] = lab
    spec.loader.exec_module(lab)

    start = datetime(2026, 1, 1, tzinfo=UTC)
    cutoff = (start + timedelta(hours=100)).strftime("%Y-%m-%dT%H:%M:%SZ")
    samples = [
        type(
            "S",
            (),
            {
                "timestamp": (start + timedelta(hours=index)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "features": {},
                "targets": {"fwd_direction": 1.0},
            },
        )()
        for index in range(130)
    ]
    candidate = types.SimpleNamespace(
        horizon=1,
        feature_version="rl-crypto/features/2",
        feature_names=["ret_1"],
        targets={
            "fwd_direction": types.SimpleNamespace(
                model=types.SimpleNamespace(to_dict=lambda: {"kind": "logistic"})
            )
        },
    )
    current = types.SimpleNamespace(
        horizon=1,
        feature_names=["ret_1"],
        targets={
            "fwd_direction": types.SimpleNamespace(
                model=types.SimpleNamespace(to_dict=lambda: {"kind": "logistic"})
            )
        },
    )
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "load_ticker_rows", lambda ticker: samples)
    monkeypatch.setattr(lab, "build_samples", lambda rows, **kwargs: tuple(samples))
    monkeypatch.setattr(
        lab, "load_bundle", lambda path: candidate if "candidate" in str(path) else current
    )
    monkeypatch.setattr(lab, "default_bundle_path", lambda ticker, directory: Path("current.json"))
    monkeypatch.setattr(lab, "train_tier1", lambda train, **kwargs: types.SimpleNamespace())
    monkeypatch.setattr(
        lab,
        "_bundle_view",
        lambda bundle, samples, targets, **kwargs: {
            "targets": {"fwd_direction": {"beats": True, "model": 0.5}},
            "economic": {"model": {"trades": 20, "sharpe": 0.5, "max_drawdown": -0.1}},
        },
    )
    monkeypatch.setattr(lab, "promotion_gate", lambda *args, **kwargs: [])
    monkeypatch.setattr(lab, "_save", lambda action, payload: Path("artifact.json"))
    monkeypatch.setattr(lab, "save_bundle", lambda bundle, path: None)
    monkeypatch.setattr(lab, "LAB_DIR", tmp_path / "lab")

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}", encoding="utf-8")
    args = types.SimpleNamespace(
        candidate=str(candidate_path),
        tickers=["BTC/USD"],
        ticker="BTC/USD",
        targets=["fwd_direction"],
        holdout_start=cutoff,
        champion_retrain=True,
        threshold=0.55,
        position="long-only",
        fee_bps=10.0,
        slippage_bps=5.0,
        apply=True,
        json=True,
    )
    assert lab.promote_action(args) == 0
    config = json.loads((tmp_path / "config" / "training.json").read_text())
    assert config["tickers"]["BTC/USD"] == {
        "feature_names": ["ret_1"],
        "model_names": {"fwd_direction": "logistic"},
        "horizon": 1,
    }


def test_positive_canary_leak_is_detected_mechanically():
    samples = [
        Sample(
            timestamp=f"2026-01-01T{index:04d}:00:00Z",
            features={"ret_1": 0.001 * (index % 5)},
            targets={"fwd_return": 0.001 * (index % 7), "fwd_direction": float(index % 2)},
        )
        for index in range(80)
    ]
    assert lab.detect_leaks(samples, lab.TARGET_NAMES) == []
    leaky = lab.inject_leaky_feature(samples, "leak_next_bar_return")
    leaks = lab.detect_leaks(leaky, lab.TARGET_NAMES)
    assert leaks
    assert leaks[0]["feature"] == "leak_next_bar_return"
    assert leaks[0]["target"] == "fwd_return"
    assert leaks[0]["correlation"] == pytest.approx(1.0)
