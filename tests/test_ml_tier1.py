"""Tier 1: training bundles, predictions, JSON round-trips, honesty."""

from __future__ import annotations

import numpy as np
import pytest

from rl_crypto import build_samples
from rl_crypto.ml.tier1 import Tier1Bundle, baseline_for, train_tier1

from .conftest import make_dataset_rows

ALL_TARGETS = {"fwd_return", "fwd_direction", "fwd_volatility", "fwd_volume_ratio"}


@pytest.fixture
def samples() -> tuple:
    return build_samples(make_dataset_rows(100, wiggle=0.04), horizon=1)


def test_train_fits_every_target(samples):
    bundle = train_tier1(samples, horizon=1)
    assert set(bundle.targets) == ALL_TARGETS
    assert bundle.targets["fwd_direction"].kind == "classification"
    assert all(
        model.kind == "regression" for name, model in bundle.targets.items() if name != "fwd_direction"
    )
    assert all(model.n_samples >= 8 for model in bundle.targets.values())
    assert all(model.metrics for model in bundle.targets.values())


def test_bundle_predictions_are_finite(samples):
    bundle = train_tier1(samples, horizon=1)
    points, probabilities = bundle.predict(samples[-2].features)
    assert points["fwd_return"] is not None
    probability = probabilities["fwd_direction"]
    assert probability is not None and 0.0 <= probability <= 1.0
    assert points["fwd_volatility"] is not None


def test_volatility_uses_a_non_negative_estimator(samples):
    # A spread cannot be negative, and the plain ridge is unbounded. The
    # volatility target must therefore be fitted in log space by default, and
    # its served point prediction must stay in the target's domain.
    bundle = train_tier1(samples, horizon=1)
    assert bundle.targets["fwd_volatility"].model.name == "log_ridge"
    points, _ = bundle.predict(samples[-2].features)
    assert points["fwd_volatility"] is not None
    assert points["fwd_volatility"] > 0.0


def test_log_space_target_is_scored_on_log_scale(samples):
    bundle = train_tier1(samples, horizon=1)
    vol = bundle.targets["fwd_volatility"]
    assert vol.metric_space == "log"
    rebuilt = Tier1Bundle.from_dict(bundle.to_dict())
    assert rebuilt.targets["fwd_volatility"].metric_space == "log"


def test_log_baseline_ignores_non_positive_rows():
    # A zero realised spread has no log. If it were floored instead of dropped,
    # the log baseline would become finite only by accident and one quiet bar
    # could decide the verdict.
    y_train = np.array([0.0, 0.01, 0.02, 0.03])
    y_test = np.array([0.0, 0.04])
    report = baseline_for("regression", y_train, y_test, space="log")
    assert report["rmse"] is not None
    assert np.isfinite(report["rmse"])
    assert report["r2"] is None  # a single scored row has no variance


def test_bundle_json_round_trip(samples):
    bundle = train_tier1(samples, horizon=1)
    rebuilt = Tier1Bundle.from_dict(bundle.to_dict())
    assert set(rebuilt.targets) == set(bundle.targets)
    assert rebuilt.feature_names == bundle.feature_names
    assert rebuilt.horizon == bundle.horizon
    original = bundle.predict(samples[-1].features)
    restored = rebuilt.predict(samples[-1].features)
    assert restored[0]["fwd_return"] == pytest.approx(original[0]["fwd_return"])


def test_incomplete_features_predict_none(samples):
    bundle = train_tier1(samples, horizon=1)
    features = dict(samples[-1].features)
    features["ret_1"] = None
    points, _ = bundle.predict(features)
    assert points["fwd_return"] is None


def test_skips_underpowered_targets():
    tiny = build_samples(make_dataset_rows(12), horizon=1)
    bundle = train_tier1(tiny, horizon=1)
    assert not bundle.targets
    assert any("skipped" in note for note in bundle.notes)