"""Tests for the measurement contract: holdout honesty, feature gating, parity.

These cover the guarantees that make a Tier-1 bundle's claims checkable:

* a data gap withholds a windowed feature/label instead of silently re-timing it,
* ``sentiment_ewma_6`` really is decay-weighted,
* a constant feature is dropped at training time and ignored at serve time,
* every fitted model carries an out-of-sample verdict against a constant baseline,
* a bundle trained against another feature version predicts nothing and says why.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import pytest

from rl_crypto.cache import merge_candles, read_jsonl
from rl_crypto.features import FEATURE_NAMES, FeatureRow, build_feature_table, feature_row
from rl_crypto.ml.store import load_bundle, save_bundle
from rl_crypto.ml.tier1 import (
    baseline_for,
    chronological_split,
    train_tier1,
    usable_features,
)
from rl_crypto.ml.tier2 import _clip_domain, compose_outlook
from rl_crypto.models import DatasetRow
from rl_crypto.samples import Sample, build_samples
from rl_crypto.targets import forward_return, forward_volatility, label_purge

from .conftest import make_dataset_rows

# --------------------------------------------------------------------------- gaps


def _rows(hours: list[int], *, base: float = 100.0) -> tuple[DatasetRow, ...]:
    """Rows at the given hour offsets, so a test can leave holes on purpose."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return tuple(
        DatasetRow(
            ticker="ETH/USD",
            timestamp=(start + timedelta(hours=h)).isoformat(),
            open=base + h,
            high=base + h + 1,
            low=base + h - 1,
            close=base + h,
            volume=100.0,
            has_price=True,
        )
        for h in hours
    )


def test_gap_withholds_windowed_features() -> None:
    """A hole in the past must cost the row its windowed features, not re-time them."""
    contiguous = _rows(list(range(30)))
    gapped = _rows(list(range(25)) + list(range(30, 36)))

    # Index 25 of the gapped series is the first bar after the hole.
    f = feature_row(gapped, 25)
    assert f.values["ret_1"] is None
    assert f.values["ret_3"] is None
    assert f.values["vol_3"] is None
    assert f.values["ma_gap_6"] is None
    assert f.values["volume_z_12"] is None
    # Same-bar features need no history, so they survive.
    assert f.values["range_1"] is not None
    assert f.values["body_1"] is not None

    # The equivalent contiguous row keeps them all.
    assert feature_row(contiguous, 25).values["ret_1"] is not None


def test_gap_withholds_forward_labels() -> None:
    """A horizon label is only defined when the future sits exactly that far away."""
    # Index 20 is followed by +3h (hour 23), so an h=1 label is not available.
    rows = _rows(list(range(21)) + list(range(23, 45)))
    assert forward_return(rows, 20, horizon=1) is None
    assert forward_volatility(rows, 20, horizon=1) is None

    contiguous = _rows(list(range(45)))
    assert forward_return(contiguous, 20, horizon=1) is not None


def test_gapless_rows_keep_every_feature() -> None:
    """The guard must not cost anything on a clean series."""
    rows = _rows(list(range(40)))
    table = build_feature_table(rows)
    last = table[-1]
    assert last.values["ret_6"] is not None
    assert last.values["ma_gap_24"] is not None
    assert last.values["vol_6"] is not None


# --------------------------------------------------------------------------- ewma


def test_sentiment_ewma_weights_recent_news_more_heavily() -> None:
    """``sentiment_ewma_6`` decays: the newest bucket must outweigh the oldest.

    Five stale buckets of -1.0 then one of +1.0 gives a flat mean of -0.667;
    a decaying mean must sit above that because it discounts the five. The
    mirror case must sit below its flat mean, which a flat average could never do.
    """
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def _row(index: int, score: float) -> DatasetRow:
        return DatasetRow(
            ticker="ETH/USD",
            timestamp=(start + timedelta(hours=index)).isoformat(),
            close=100.0,
            volume=10.0,
            has_news=True,
            sentiment_score=score,
        )

    span = 6
    rising = build_feature_table((*(_row(i, -1.0) for i in range(span - 1)), _row(span - 1, 1.0)))
    falling = build_feature_table((*(_row(i, 1.0) for i in range(span - 1)), _row(span - 1, -1.0)))

    rising_ewma = rising[-1].values["sentiment_ewma_6"]
    falling_ewma = falling[-1].values["sentiment_ewma_6"]
    flat_mean = ((span - 1) * -1.0 + 1.0) / span

    assert rising_ewma is not None and falling_ewma is not None
    # Discounting the stale buckets pulls both toward the newest value...
    assert rising_ewma > flat_mean
    assert falling_ewma < -flat_mean
    # ...and the two orderings are exact mirrors, as they must be.
    assert rising_ewma == pytest.approx(-falling_ewma)


def test_sentiment_ewma_is_zero_without_news() -> None:
    rows = make_dataset_rows(10, news_every=0)
    assert build_feature_table(rows)[-1].values["sentiment_ewma_6"] == 0.0


# --------------------------------------------------------------------------- constants


def test_usable_features_drops_constant_columns() -> None:
    """A feature that never varies is dropped, keeping only informative ones."""
    samples = build_samples(make_dataset_rows(40, news_every=0))
    names = ("ret_1", "sentiment", "has_news")
    kept, dropped = usable_features(samples, names)
    assert kept == ("ret_1",)
    assert set(dropped) == {"sentiment", "has_news"}


def test_usable_features_keeps_columns_that_vary() -> None:
    samples = build_samples(make_dataset_rows(40, news_every=3))
    kept, dropped = usable_features(samples, ("sentiment", "has_news", "ret_1"))
    assert "sentiment" in kept
    assert dropped == ()


def test_constant_news_features_are_excluded_from_the_bundle() -> None:
    """The market-only archive must not hand the model six dead news columns."""
    samples = build_samples(make_dataset_rows(80, news_every=0))
    bundle = train_tier1(samples, horizon=1, targets=("fwd_return",))
    assert "has_news" not in bundle.feature_names
    assert "sentiment" not in bundle.feature_names
    assert set(bundle.dropped_features) >= {"has_news", "sentiment", "novelty"}
    assert any("constant" in note for note in bundle.notes)


# --------------------------------------------------------------------------- holdout


def test_every_fitted_model_carries_a_holdout_verdict() -> None:
    samples = build_samples(make_dataset_rows(120))
    bundle = train_tier1(samples, horizon=1)
    assert bundle.targets, "expected the reference data to fit something"
    assert bundle.holdout_n_samples > 0
    for name, model in bundle.targets.items():
        assert model.holdout is not None, name
        assert model.holdout.n_samples > 0, name
        assert isinstance(model.holdout.beats_baseline, bool), name
        assert model.holdout.baseline, name


def test_holdout_rows_are_excluded_from_training() -> None:
    """The shipped weights must never have seen the rows they are scored on."""
    samples = build_samples(make_dataset_rows(120))
    bundle = train_tier1(samples, horizon=1, targets=("fwd_return", "fwd_direction"))
    for model in bundle.targets.values():
        assert model.n_samples + model.holdout.n_samples <= len(samples)
        assert model.n_samples < len(samples)


def test_a_holdout_is_skipped_when_there_are_too_few_rows() -> None:
    """Too few usable rows for a meaningful holdout -> fit, but mark unmeasured.

    ``ma_gap_24`` needs 24 rows before any sample is feature-complete, so the
    fixture is sized to clear MIN_TRAIN_SAMPLES while leaving too little to
    spare: 10 usable rows yields a holdout of 2, under MIN_HOLDOUT_SAMPLES.
    """
    samples = build_samples(make_dataset_rows(34))
    bundle = train_tier1(samples, horizon=1, targets=("fwd_return",))
    model = bundle.targets.get("fwd_return")
    assert model is not None, "fixture should produce exactly the usable-rows-too-few case"
    assert model.holdout is None
    assert bundle.holdout_n_samples == 0
    assert bundle.trusted("fwd_return") is False
    assert any("no holdout" in note for note in bundle.notes)


def test_in_sample_metrics_still_reported_and_labelled_separately() -> None:
    samples = build_samples(make_dataset_rows(120))
    bundle = train_tier1(samples, horizon=1, targets=("fwd_return",))
    model = bundle.targets["fwd_return"]
    assert model.metrics  # in-sample fit quality, kept for the record
    assert model.holdout is not None and model.holdout.metrics
    # They are distinct blocks in JSON, not one merged "score".
    payload = model.to_dict()
    assert "metrics" in payload and "holdout" in payload


def test_baseline_is_the_constant_predictor() -> None:
    import numpy as np

    train = np.asarray([1.0, 2.0, 3.0])
    test = np.asarray([10.0, 12.0])
    baseline = baseline_for("regression", train, test)
    # Always predicting the training mean (2.0) is the only constant available.
    assert baseline["rmse"] is not None and baseline["rmse"] > 0


def test_trusted_is_false_for_an_unmeasured_model() -> None:
    samples = build_samples(make_dataset_rows(120))
    bundle = train_tier1(samples, horizon=1, targets=("fwd_return",))
    model = bundle.targets["fwd_return"]
    assert model.beats_baseline in (True, False)
    assert bundle.trusted("fwd_return") is (model.beats_baseline is True)
    assert bundle.trusted("a_target_that_does_not_exist") is False


# --------------------------------------------------------------------------- tier 2


def test_signal_withheld_when_the_model_did_not_beat_its_baseline() -> None:
    """A noisy direction model must not publish a confident signal."""
    samples = build_samples(make_dataset_rows(120))
    bundle = train_tier1(samples, horizon=1)
    row = build_feature_table_row(bundle)
    outlook = compose_outlook(bundle, ticker="ETH/USD", feature_row=row)
    if not bundle.trusted("fwd_direction"):
        assert outlook.signal is None
        assert outlook.confidence is None
        assert any("withheld" in note for note in outlook.notes)
    # Point predictions survive regardless — they are inputs, not claims.
    assert "fwd_return" in outlook.predictions


def test_signal_published_for_a_trusted_model() -> None:
    """The positive half of the gate: a model that passed its baseline speaks.

    The verdict is stamped directly rather than waited for, because whether any
    given series produces a skilled direction model is a property of the data,
    not of the gate. This exercises the publish path unconditionally.
    """
    bundle = train_tier1(build_samples(make_dataset_rows(120)), horizon=1)
    model = bundle.targets["fwd_direction"]
    model.holdout.beats_baseline = True
    assert bundle.trusted("fwd_direction") is True

    row = build_feature_table_row(bundle)
    outlook = compose_outlook(bundle, ticker="ETH/USD", feature_row=row)
    assert outlook.signal in (1, 0, -1)
    assert outlook.confidence is not None
    assert not any("withheld" in note for note in outlook.notes)


def test_outlook_notes_the_features_it_ignored() -> None:
    bundle = train_tier1(build_samples(make_dataset_rows(80, news_every=0)), horizon=1)
    row = build_feature_table_row(bundle)
    outlook = compose_outlook(bundle, ticker="ETH/USD", feature_row=row)
    assert any("excluded as constant" in note for note in outlook.notes)


def test_negative_volatility_prediction_is_clipped_to_its_domain() -> None:
    """A ridge fit is unbounded; a standard deviation is not.

    Serving a negative spread would publish a physically impossible number, so
    it is withheld as ``None`` and reported rather than passed through.
    """
    points, notes = _clip_domain({"fwd_volatility": -0.0011, "fwd_return": -0.002})
    # An honest gap, not a fabricated zero: no asset has zero volatility.
    assert points["fwd_volatility"] is None
    # A signed target keeps its sign: only the non-negative domain is enforced.
    assert points["fwd_return"] == -0.002
    assert any("domain" in note and "fwd_volatility" in note for note in notes)


def test_clip_domain_leaves_valid_and_missing_predictions_alone() -> None:
    points, notes = _clip_domain({"fwd_volatility": 0.0018, "fwd_volume_ratio": -0.4})
    assert points == {"fwd_volatility": 0.0018, "fwd_volume_ratio": -0.4}
    assert notes == []

    points, notes = _clip_domain({"fwd_volatility": None})
    assert points == {"fwd_volatility": None}
    assert notes == []


def test_compose_outlook_reports_a_clipped_spread() -> None:
    """The clip is visible in the outlook notes, not silent."""

    class _Bundle:
        feature_version_mismatch = False
        dropped_features: tuple[str, ...] = ()
        horizon = 1
        targets: ClassVar[dict[str, object]] = {}

        @staticmethod
        def predict(values):
            return {"fwd_volatility": -0.0005}, {}

    row = FeatureRow(timestamp="2026-10-23T20:00:00Z", values={"vol_3": 0.001})
    outlook = compose_outlook(_Bundle(), ticker="ETH/USD", feature_row=row)  # type: ignore[arg-type]

    assert outlook.predictions["fwd_volatility"] is None
    assert any("domain" in note for note in outlook.notes)
    # The gap is reported for the real reason, not as a missing feature.
    assert not any("required feature missing" in note for note in outlook.notes)


# --------------------------------------------------------------------------- parity


def test_bundle_records_the_feature_version(tmp_path) -> None:
    bundle = train_tier1(build_samples(make_dataset_rows(120)), horizon=1)
    path = tmp_path / "bundle.json"
    save_bundle(bundle, path)
    assert json.loads(path.read_text())["feature_version"] == bundle.feature_version
    assert load_bundle(path).feature_version == bundle.feature_version


def test_stale_bundle_is_flagged_not_crashed(tmp_path) -> None:
    """The 30-minute timer must keep recording when a bundle goes stale."""
    bundle = train_tier1(build_samples(make_dataset_rows(120)), horizon=1)
    path = tmp_path / "bundle.json"
    save_bundle(bundle, path)

    payload = json.loads(path.read_text())
    payload["feature_version"] = "rl-crypto/features/0"
    path.write_text(json.dumps(payload))

    stale = load_bundle(path)
    assert stale.feature_version_mismatch is True
    assert all(not stale.trusted(name) for name in stale.targets)
    assert any("feature version mismatch" in note for note in stale.notes)

    row = build_feature_table_row(stale)
    outlook = compose_outlook(stale, ticker="ETH/USD", feature_row=row)
    assert outlook.signal is None
    assert any("feature version" in note for note in outlook.notes)


# --------------------------------------------------------------------------- cache


def test_merge_candles_does_not_duplicate_across_timestamp_formats(tmp_path) -> None:
    """The same bar under two formats must upsert, not double the archive."""
    path = tmp_path / "candles.jsonl"
    merge_candles(path, [{"ticker": "T", "timestamp": "2026-01-01 00:00:00+00:00", "close": 1.0}])
    merge_candles(path, [{"ticker": "T", "timestamp": "2026-01-01T00:00:00Z", "close": 2.0}])
    rows = read_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["close"] == 2.0
    assert rows[0]["timestamp"] == "2026-01-01T00:00:00Z"


# --------------------------------------------------------------------------- helper


def build_feature_table_row(bundle):
    """The inference row the serve path would pick for ``bundle``."""
    from rl_crypto.features import latest_complete

    rows = make_dataset_rows(120)
    row = latest_complete(rows, names=bundle.feature_names)
    assert row is not None, "expected a complete inference row"
    return row

# --------------------------------------------------------------------------- purge


def _ordered_samples(n: int) -> list[Sample]:
    """Ordered samples with distinct timestamps, for split-boundary assertions."""
    return [
        Sample(timestamp=f"2026-01-01T{i:05d}:00:00Z", features={}, targets={})
        for i in range(n)
    ]


@pytest.mark.parametrize(
    ("target", "horizon", "expected_purge"),
    [
        ("fwd_return", 1, 1),
        ("fwd_direction", 1, 1),
        ("fwd_volume_ratio", 1, 1),
        ("fwd_volatility", 1, 2),
        ("fwd_volatility", 4, 4),
    ],
)
def test_chronological_split_purges_each_targets_true_reach(
    target: str, horizon: int, expected_purge: int
) -> None:
    """The last training label must end before the holdout begins.

    ``fwd_volatility`` is the case a flat ``horizon`` purge gets wrong: at
    ``horizon=1`` its realised span is two bars, so one bar of the holdout would
    be read by a training label.
    """
    samples = _ordered_samples(100)
    train, holdout, n_holdout = chronological_split(
        samples, target=target, horizon=horizon, holdout_fraction=0.2
    )

    assert n_holdout == 20
    assert len(holdout) == 20
    assert len(train) == 100 - 20 - expected_purge
    assert samples.index(train[-1]) + expected_purge < samples.index(holdout[0])


def test_label_purge_for_volatility_is_never_below_two() -> None:
    assert label_purge("fwd_volatility", 1) == 2
    assert label_purge("fwd_volatility", 0) == 2
    assert label_purge("fwd_volatility", 3) == 3
    assert label_purge("fwd_return", 1) == 1


def test_trainer_fits_volatility_on_the_purged_split() -> None:
    """The fitted row count proves the volatility purge reached the trainer."""
    rows = make_dataset_rows(200, wiggle=0.02)
    samples = build_samples(rows)
    bundle = train_tier1(samples, targets=("fwd_volatility",), holdout_fraction=0.2)
    model = bundle.targets["fwd_volatility"]
    assert model.holdout is not None
    n_usable = len(
        [
            s
            for s in samples
            if s.features_complete(FEATURE_NAMES)
            and s.targets.get("fwd_volatility") is not None
        ]
    )
    n_holdout = model.holdout.n_samples
    assert n_holdout == int(n_usable * 0.2)
    # Two bars are purged at horizon 1, not one.
    assert model.n_samples == n_usable - n_holdout - 2
