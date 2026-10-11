"""Tier-0 features: causality, stability, and the no-fabrication contract."""

from __future__ import annotations

import math

from rl_crypto.features import (
    FEATURE_NAMES,
    build_feature_table,
    feature_row,
    latest_complete,
)

from .conftest import make_dataset_rows


def test_feature_names_are_stable_and_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert "ret_1" in FEATURE_NAMES
    assert "sentiment_ewma_6" in FEATURE_NAMES
    assert "hour_sin" in FEATURE_NAMES


def test_features_are_causal():
    """Truncating the future must not change a past feature row."""
    rows = make_dataset_rows(40, wiggle=0.02)
    prefix = rows[:25]
    full = feature_row(rows, 24).values
    cut = feature_row(prefix, 24).values
    for name in FEATURE_NAMES:
        assert full[name] == cut[name], name


def test_return_matches_closes():
    rows = make_dataset_rows(10)
    got = feature_row(rows, 1).values["ret_1"]
    previous = rows[0].close
    current = rows[1].close
    assert previous is not None and current is not None
    assert got == math.log(current / previous)


def test_early_rows_are_incomplete_and_late_rows_are_not():
    rows = make_dataset_rows(30)
    table = build_feature_table(rows)
    assert not table[0].complete
    assert "ret_1" in table[0].missing
    assert table[-1].complete


def test_news_absence_is_encoded_not_fabricated():
    rows = make_dataset_rows(40, news_every=3)
    bare = next(i for i in range(40) if i % 3 != 0)
    row = feature_row(rows, bare).values
    assert row["has_news"] == 0.0
    assert row["sentiment"] == 0.0  # encoded absence, documented
    assert row["articles_log"] == 0.0


def test_latest_complete_prefers_the_last_row():
    rows = make_dataset_rows(30)
    last = latest_complete(rows)
    assert last is not None
    assert last.timestamp == rows[-1].timestamp