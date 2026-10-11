"""Aligning features and targets into training samples."""

from __future__ import annotations

import math

from rl_crypto import build_samples

from .conftest import make_dataset_rows


def test_samples_align_features_and_targets():
    rows = make_dataset_rows(40)
    samples = build_samples(rows, horizon=1)
    assert len(samples) == len(rows)

    sample = samples[25]
    now = rows[25].close
    later = rows[26].close
    previous = rows[24].close
    assert now is not None and later is not None and previous is not None
    assert sample.targets["fwd_return"] == math.log(later / now)
    assert sample.features["ret_1"] == math.log(now / previous)
    assert sample.features_complete()


def test_early_samples_are_feature_incomplete():
    rows = make_dataset_rows(10)
    samples = build_samples(rows)
    assert not samples[0].features_complete()


def test_sample_to_dict_is_flat_and_json_ready():
    rows = make_dataset_rows(10)
    payload = build_samples(rows)[1].to_dict()
    assert "timestamp" in payload
    assert "ret_1" in payload
    assert "fwd_return" in payload