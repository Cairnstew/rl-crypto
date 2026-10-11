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


def test_registry_names_and_defaults():
    assert set(model_names()) == {"base_rate", "logistic", "mean", "ridge"}
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
    [("regression", "ridge"), ("regression", "mean"), ("classification", "logistic")],
)
def test_estimator_json_round_trip(kind, name):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(float) if kind == "classification" else x[:, 0] - x[:, 2] + 0.5
    model = make_model(kind, name).fit(x, y)
    rebuilt = model_from_dict(kind, model.to_dict())
    assert np.allclose(model.predict(x), rebuilt.predict(x))


def test_make_model_rejects_wrong_kind():
    with pytest.raises(ValueError):
        make_model("regression", "logistic")