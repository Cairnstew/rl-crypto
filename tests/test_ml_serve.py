"""The serve seam: live outlooks, honest gaps, and the feature addition."""

from __future__ import annotations

import pytest

from rl_crypto import build_samples
from rl_crypto.ml.serve import (
    attach_predictions,
    outlook_for_rows,
)
from rl_crypto.ml.store import default_bundle_path, save_bundle
from rl_crypto.ml.tier1 import train_tier1
from rl_crypto.models import UnifiedRecord

from .conftest import make_dataset_rows


@pytest.fixture
def bundle():
    rows = make_dataset_rows(100, wiggle=0.04)
    return train_tier1(build_samples(rows, horizon=1), horizon=1)


def test_outlook_for_rows_composes(bundle):
    rows = make_dataset_rows(60, wiggle=0.04)
    outlook = outlook_for_rows(rows, bundle, ticker="ETH/USD")
    assert outlook.predictions["fwd_return"] is not None
    probability = outlook.probabilities.get("fwd_direction")
    assert probability is not None and 0.0 <= probability <= 1.0
    assert outlook.signal in (-1, 0, 1)
    assert outlook.feature_timestamp is not None


def test_outlook_for_too_few_rows_is_honest(bundle):
    outlook = outlook_for_rows(make_dataset_rows(5), bundle, ticker="ETH/USD")
    assert not outlook.predictions
    assert outlook.notes
    assert outlook.signal is None


def test_attach_predictions_adds_features_to_the_record(bundle):
    rows = make_dataset_rows(60, wiggle=0.04)
    outlook = outlook_for_rows(rows, bundle, ticker="ETH/USD")
    record = UnifiedRecord(ticker="ETH/USD", market_ok=True)
    attach_predictions(record, outlook)

    payload = record.to_dict()
    assert payload["predictions"]["schema"].startswith("rl-crypto/ml/")
    assert payload["predictions"]["feature_version"]
    assert payload["predictions"]["metric_space"]["fwd_volatility"] == "log"
    flat = outlook.as_features()
    assert flat["pred_fwd_return"] is not None
    assert "pred_signal" in flat
    assert flat["pred_signal"] is not None


def test_as_features_preserves_none_for_unavailable_predictions():
    from rl_crypto.ml.types import Outlook

    outlook = Outlook(
        ticker="ETH/USD", generated_at="2026-01-01T00:00:00Z", horizon=1,
        predictions={"fwd_return": None}, signal=None,
    )
    flat = outlook.as_features()
    assert flat["pred_fwd_return"] is None
    assert flat["pred_signal"] is None


def test_load_default_path_round_trip(tmp_path, bundle):
    path = default_bundle_path("ETH/USD", directory=tmp_path)
    save_bundle(bundle, path)

    from rl_crypto.ml.serve import load_bundle_for

    loaded = load_bundle_for("ETH/USD", path)
    assert set(loaded.targets) == set(bundle.targets)