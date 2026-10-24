"""Offline tests for the committed training config that makes a promotion durable."""

from __future__ import annotations

import json

import pytest

from rl_crypto.features import FEATURE_NAMES
from rl_crypto.ml.training_config import (
    SCHEMA,
    TrainingConfig,
    apply_promotion,
    config_path,
    load_training_config,
    parse_config,
    save_training_config,
)


def test_missing_config_means_production_defaults(tmp_path):
    config = load_training_config(tmp_path)
    resolved = config.for_ticker("BTC/USD")

    assert resolved.horizon == 1
    assert resolved.feature_names == FEATURE_NAMES
    assert resolved.model_names == {}
    assert resolved.source == "default"


def test_roundtrip_is_stable_and_empty_features_mean_all(tmp_path):
    config = TrainingConfig(
        horizon=1,
        feature_names=FEATURE_NAMES,
        model_names={"fwd_volatility": "log_ridge"},
        tickers={"BTC/USD": {"feature_names": ["ret_1"], "horizon": 1}},
    )
    path = save_training_config(tmp_path, config)
    payload = json.loads(path.read_text())

    assert payload["schema"] == SCHEMA
    assert payload["feature_names"] == []  # the full set, not a frozen copy
    loaded = load_training_config(tmp_path)
    assert loaded.for_ticker("ETH/USD").feature_names == FEATURE_NAMES
    assert loaded.for_ticker("BTC/USD").feature_names == ("ret_1",)
    assert loaded.for_ticker("ETH/USD").model_names == {"fwd_volatility": "log_ridge"}


def test_ticker_override_does_not_retune_the_rest_of_the_watchlist(tmp_path):
    config = TrainingConfig(model_names={"fwd_volatility": "log_ridge"})
    save_training_config(tmp_path, config)

    apply_promotion(
        tmp_path,
        ticker="BTC/USD",
        feature_names=("ret_1", "ret_3"),
        model_names={"fwd_direction": "logistic"},
        horizon=1,
    )
    loaded = load_training_config(tmp_path)

    btc = loaded.for_ticker("BTC/USD")
    eth = loaded.for_ticker("ETH/USD")
    assert btc.feature_names == ("ret_1", "ret_3")
    assert btc.model_names == {
        "fwd_volatility": "log_ridge",
        "fwd_direction": "logistic",
    }
    assert btc.source == "ticker:BTC/USD"
    assert eth.feature_names == FEATURE_NAMES
    assert eth.model_names == {"fwd_volatility": "log_ridge"}
    assert eth.source == "default"


def test_unknown_schema_is_rejected():
    with pytest.raises(ValueError, match="schema"):
        parse_config({"schema": "rl-crypto/training/0"})


def test_malformed_shapes_are_rejected():
    with pytest.raises(ValueError, match="feature_names"):
        parse_config({"schema": SCHEMA, "feature_names": "ret_1"})
    with pytest.raises(ValueError, match="model_names"):
        parse_config({"schema": SCHEMA, "model_names": []})
    with pytest.raises(ValueError, match="tickers"):
        parse_config({"schema": SCHEMA, "tickers": {"BTC/USD": []}})
    with pytest.raises(ValueError, match="horizon"):
        parse_config({"schema": SCHEMA, "horizon": 0})


def test_apply_promotion_creates_the_config_when_absent(tmp_path):
    assert not config_path(tmp_path).exists()
    apply_promotion(
        tmp_path,
        ticker="SOL/USD",
        feature_names=("ret_1",),
        model_names={"fwd_return": "ridge"},
        horizon=2,
    )
    loaded = load_training_config(tmp_path)
    sol = loaded.for_ticker("SOL/USD")
    assert sol.feature_names == ("ret_1",)
    assert sol.horizon == 2
    assert sol.source == "ticker:SOL/USD"
