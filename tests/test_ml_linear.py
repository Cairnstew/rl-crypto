"""Reference estimators: learning, baselines, and JSON round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from rl_crypto.ml.base import make_model, model_from_dict, model_names


def test_ridge_recovers_a_linear_relation():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(200, 3))
    y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + 1.0
    model = make_model("regression", "ridge", alpha=1e-6).fit(x, y)
    assert np.mean(np.abs(model.predict(x) - y)) < 1e-3


def test_logistic_separates_two_clusters():
    rng = np.random.default_rng(7)
    negative = rng.normal(-2.0, 0.5, size=(150, 2))
    positive = rng.normal(2.0, 0.5, size=(150, 2))
    x = np.vstack([negative, positive])
    y = np.concatenate([np.zeros(150), np.ones(150)])
    model = make_model("classification").fit(x, y)
    assert np.mean((model.predict(x) >= 0.5) == y) > 0.98


def test_log_ridge_learns_in_log_space_and_stays_positive():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(300, 2))
    # A multiplicative relation: the raw target is exponential in the features.
    y = np.exp(0.4 * x[:, 0] - 0.2 * x[:, 1] + 1.0)
    model = make_model("regression", "log_ridge", alpha=1e-6).fit(x, y)
    predicted = model.predict(x)
    assert np.all(predicted > 0.0)
    assert np.mean(np.abs(predicted - y)) < 1e-3


def test_log_ridge_survives_adversarial_extrapolation():
    # A plain ridge can go negative far outside its training range; the log
    # estimator cannot, because exp() is positive for every input.
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 1))
    y = np.exp(0.5 * x[:, 0])
    model = make_model("regression", "log_ridge", alpha=1e-6).fit(x, y)
    extreme = np.full((20, 1), -1e6)
    assert np.all(model.predict(extreme) > 0.0)


def test_log_ridge_ignores_non_positive_targets_when_fitting():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 1))
    y = np.exp(0.3 * x[:, 0])
    y[::5] = 0.0  # a zero realised spread carries no log
    model = make_model("regression", "log_ridge", alpha=1e-6).fit(x, y)
    assert np.all(model.predict(x) > 0.0)


def test_registry_names_and_defaults():
    assert set(model_names()) == {"base_rate", "log_ridge", "logistic", "mean", "ridge"}
    model = make_model("regression")
    assert model.name == "ridge"
    model = make_model("classification")
    assert model.name == "logistic"


def test_baselines_are_dumb_and_serialisable():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = np.zeros((5, 2))
    mean = make_model("regression", "mean").fit(x, y)
    assert np.allclose(mean.predict(np.zeros((3, 2))), 3.0)
    assert mean.to_dict()["value"] == 3.0

    yb = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    rate = make_model("classification", "base_rate").fit(x, yb)
    assert np.allclose(rate.predict(np.zeros((2, 2))), 0.6)


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        ("regression", "ridge"),
        ("regression", "log_ridge"),
        ("regression", "mean"),
        ("classification", "logistic"),
    ],
)
def test_estimator_json_round_trip(kind, name):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4))
    # Keep regression targets strictly positive so log_ridge can fit them.
    y = (
        (x[:, 0] + x[:, 1] > 0).astype(float)
        if kind == "classification"
        else np.exp(0.2 * x[:, 0]) + 0.1
    )
    model = make_model(kind, name).fit(x, y)
    rebuilt = model_from_dict(kind, model.to_dict())
    assert np.allclose(model.predict(x), rebuilt.predict(x))


def test_make_model_rejects_wrong_kind():
    with pytest.raises(ValueError):
        make_model("regression", "logistic")