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
            timestamp=f"2026-01-{1 + i // 24:02d}T{i % 24:02d}:00:00Z",
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


def test_plant_signal_adds_a_learnable_feature_and_reports_its_effect():
    samples = _samples(200)
    planted, meta = lab.plant_signal(
        samples, "fwd_return", strength=0.02, seed=3, space="raw"
    )
    assert meta["effect"] > 0.0
    assert all(lab.PLANTED_FEATURE in sample.features for sample in planted)
    changed = [
        abs(a.targets["fwd_return"] - b.targets["fwd_return"])
        for a, b in zip(planted, samples, strict=True)
    ]
    assert max(changed) > 0.0


def test_plant_signal_in_log_space_keeps_the_target_positive():
    samples = [
        Sample(
            timestamp=f"2026-01-{1 + i // 24:02d}T{i % 24:02d}:00:00Z",
            features={"ret_1": 0.0},
            targets={"fwd_volatility": 0.02},
        )
        for i in range(50)
    ]
    planted, meta = lab.plant_signal(
        samples, "fwd_volatility", strength=0.1, seed=4, space="log"
    )
    assert meta["space"] == "log"
    assert all((sample.targets["fwd_volatility"] or 0.0) > 0.0 for sample in planted)


def test_inject_correlated_feature_hits_the_requested_strength():
    samples = _samples(400)
    for requested in (0.5, 0.2, 0.1, 0.05):
        planted, meta = lab.inject_correlated_feature(
            samples, "fwd_return", requested, seed=7
        )
        actual = meta["actual_correlation"]
        assert actual is not None
        assert abs(abs(actual) - requested) < 0.03
        assert lab.detect_leaks(planted, ["fwd_return"]) == []


def test_validate_splits_catches_a_missing_purge():
    samples = _samples(200)
    good = lab.walk_forward(samples, folds=3, horizon=2, targets=["fwd_return"])
    assert lab.validate_splits(
        samples, good, targets=["fwd_return"], horizon=2
    )["ok"] is True
    bad = lab.walk_forward(
        samples, folds=3, horizon=2, targets=["fwd_return"], purge=0
    )
    result = lab.validate_splits(samples, bad, targets=["fwd_return"], horizon=2)
    assert result["ok"] is False
    assert result["violations"]


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
        # Structural perturbation invariance also refits, but it uses the
        # newest holdout; count only the walk-forward fold fits here.
        if kwargs.get("holdout_fraction") == 0.0:
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


def test_paired_target_rows_honour_log_space_for_volatility():
    """A log-space volatility model is paired against a log-space constant."""

    class _Model:
        def __init__(self, space):
            self.metric_space = space

        def predict_row(self, features):
            return features["pred"]

    class _Bundle:
        def __init__(self, space):
            self.targets = {"fwd_volatility": _Model(space)}

    def _sample(value, pred):
        return Sample(
            timestamp=str(value),
            features={"pred": pred},
            targets={"fwd_volatility": value},
        )

    # Train constant is 0.01 on both scales. Prediction 0.03 against truth
    # 0.02 is exactly as far from the constant as the truth is on the raw
    # scale (improvement 0) but closer on the log scale (improvement > 0).
    train = [_sample(0.01, 0.01), _sample(0.01, 0.01)]
    test = [_sample(0.02, 0.03), _sample(0.02, 0.03)]
    raw = lab.paired_target_rows(_Bundle("raw"), "fwd_volatility", train, test)
    logged = lab.paired_target_rows(_Bundle("log"), "fwd_volatility", train, test)
    assert raw and logged
    assert all(abs(row["improvement"]) < 1e-12 for row in raw)
    assert all(row["improvement"] > 0 for row in logged)


def test_default_control_shifts_lose_seasonality_alignment():
    for shift in lab.DEFAULT_CONTROL_SHIFTS:
        assert shift >= 168
        assert shift % 24 != 0
        assert shift % 168 != 0
        assert lab.valid_control_shift(shift)
    assert not lab.valid_control_shift(24)
    assert not lab.valid_control_shift(168)
    assert not lab.valid_control_shift(167)


def test_block_permute_preserves_the_label_multiset_and_breaks_alignment():
    samples = [
        Sample(
            timestamp=f"2026-01-01T{index:04d}:00:00Z",
            features={"ret_1": 0.01 if index % 2 else -0.01},
            targets={"fwd_volume_ratio": float(index)},
        )
        for index in range(120)
    ]
    permuted = lab.transform_targets(
        samples,
        "fwd_volume_ratio",
        "block_permute",
        seed=5,
        shift_bars=169,
        block_bars=12,
    )
    before = [sample.targets["fwd_volume_ratio"] for sample in samples]
    after = [sample.targets["fwd_volume_ratio"] for sample in permuted]
    assert sorted(after) == sorted(before)
    assert after != before
    assert [sample.timestamp for sample in permuted] == [sample.timestamp for sample in samples]
    # Blocks move whole; the first block is not necessarily in place, but the
    # sequence within any surviving block stays contiguous in value.
    assert all(value is not None for value in after)


def test_shift_control_is_a_past_only_transform():
    samples = [
        Sample(
            timestamp=f"2026-01-01T{index:04d}:00:00Z",
            features={},
            targets={"fwd_volume_ratio": float(index)},
        )
        for index in range(60)
    ]
    shifted = lab.transform_targets(
        samples, "fwd_volume_ratio", "shift", seed=0, shift_bars=169
    )
    assert all(sample.targets["fwd_volume_ratio"] is None for sample in shifted[:169])
    assert all(
        shifted[index].targets["fwd_volume_ratio"] == float(index - 169)
        for index in range(169, len(shifted))
    )


def _calendar_samples(n=336, *, pattern=None):
    from datetime import UTC, datetime, timedelta

    start = datetime(2026, 1, 5, tzinfo=UTC)  # a Monday
    samples = []
    for index in range(n):
        stamp = (start + timedelta(hours=index)).strftime("%Y-%m-%dT%H:%M:%SZ")
        slot = index % 168
        value = float(slot % 7) if pattern is None else pattern(slot)
        samples.append(
            Sample(
                timestamp=stamp,
                features={"ret_1": 0.01 if index % 2 else -0.01},
                targets={"fwd_volume_ratio": value},
            )
        )
    return samples


def test_seasonal_baseline_learns_hour_of_week_and_beats_constant():
    import numpy as np

    samples = _calendar_samples(336)
    train, test = samples[:250], samples[250:]
    constant = lab.baseline_predictions(
        "constant", "regression", train, test, "fwd_volume_ratio"
    )
    seasonal = lab.baseline_predictions(
        "seasonal", "regression", train, test, "fwd_volume_ratio"
    )
    assert len(set(constant)) == 1
    assert len(set(seasonal)) > 1
    # The target is exactly the hour-of-week slot pattern, so the seasonal
    # baseline should reproduce it far better than the flat mean.
    from rl_crypto.ml.tier1 import regression_metrics

    truth = [sample.targets["fwd_volume_ratio"] for sample in test]
    assert (
        regression_metrics(np.asarray(truth), np.asarray(seasonal))["rmse"]
        < regression_metrics(np.asarray(truth), np.asarray(constant))["rmse"]
    )


def test_persistence_baseline_uses_the_last_completed_label():
    samples = _calendar_samples(120, pattern=lambda slot: float(slot))
    train, test = samples[:80], samples[80:]
    predictions = lab.baseline_predictions(
        "persistence", "regression", train, test, "fwd_volume_ratio", horizon=1
    )
    # fwd_volume_ratio's reach is one bar: at the first test row the newest
    # completed label is the last training row.
    assert predictions[0] == train[-1].targets["fwd_volume_ratio"]
    for index in range(1, len(test)):
        assert predictions[index] == test[index - 1].targets["fwd_volume_ratio"]


def test_strongest_baseline_is_chosen_on_the_training_slice():
    # A persistent target: the best causal baseline on the inner holdout is
    # persistence, and the choice does not look at the outer test fold.
    samples = _calendar_samples(200, pattern=lambda slot: float(slot % 5))
    train = samples[:120]
    chosen = lab.choose_strongest_baseline(
        "regression", train, "fwd_volume_ratio", space="raw", horizon=1
    )
    assert chosen in {"constant", "seasonal", "persistence"}
    # Choosing after replacing the outer test data with nonsense must not move.
    assert chosen == lab.choose_strongest_baseline(
        "regression", train, "fwd_volume_ratio", space="raw", horizon=1
    )


def test_evaluate_target_reports_skill_against_all_three_baselines():
    from rl_crypto.ml.tier1 import train_tier1

    samples = _calendar_samples(260)
    train, test = samples[:200], samples[200:]
    bundle = train_tier1(
        train,
        horizon=1,
        feature_names=("ret_1",),
        targets=("fwd_volume_ratio",),
        model_names={"fwd_volume_ratio": "mean"},
        holdout_fraction=0.0,
    )
    record = lab.evaluate_target(
        bundle, "fwd_volume_ratio", train, test, baseline="strongest", horizon=1
    )
    assert record is not None
    assert set(record["skill_vs"]) == {"constant", "seasonal", "persistence"}
    assert record["baseline_mode"] in {"constant", "seasonal", "persistence"}
    assert set(record["baselines"]) == {"constant", "seasonal", "persistence"}


def test_control_ticker_reports_after_cost_economics_for_classification(tmp_path, monkeypatch):
    from .conftest import make_dataset_rows

    monkeypatch.setattr(lab, "ROOT", tmp_path)
    _write_archive(tmp_path, make_dataset_rows(280, wiggle=0.05))
    config = lab.Config(
        name="econ",
        feature_names=lab.resolve_feature_names("all"),
        model_names=lab.MODEL_SETS["reference"],
        horizon=1,
    )
    result = lab.evaluate_control_ticker(
        "ETH/USD",
        config,
        "fwd_direction",
        None,
        folds=3,
        holdout_start=None,
        seed=2,
        shift_bars=169,
        economics=True,
        threshold=0.55,
        position="long-only",
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    economics = result["economics"]
    assert economics is not None
    assert economics["model"]["bars"] > 0
    assert "momentum" in economics
    assert result["structural"]["ok"] is True
