"""Model-bundle persistence: JSON save/load and the schema guard."""

from __future__ import annotations

import json

import pytest

from rl_crypto import build_samples
from rl_crypto.ml.store import default_bundle_path, load_bundle, save_bundle
from rl_crypto.ml.tier1 import train_tier1

from .conftest import make_dataset_rows


def test_save_load_round_trip(tmp_path):
    bundle = train_tier1(build_samples(make_dataset_rows(80), horizon=1), horizon=1)
    path = tmp_path / "eth-usd.json"

    save_bundle(bundle, path)
    loaded = load_bundle(path)

    assert loaded.schema == bundle.schema
    assert set(loaded.targets) == set(bundle.targets)
    assert loaded.feature_names == bundle.feature_names


def test_load_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "rl-crypto/0", "feature_names": []}))
    with pytest.raises(ValueError):
        load_bundle(path)


def test_default_bundle_path_slugs_the_ticker(tmp_path):
    resolved = default_bundle_path("ETH/USD", directory=tmp_path)
    assert resolved == tmp_path / "eth-usd.json"