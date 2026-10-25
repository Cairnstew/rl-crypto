"""A stricter, opt-in Tier-1 trust gate: beat the *strongest* baseline.

The shipped verdict (:func:`rl_crypto.ml.tier1.beats_baseline`) compares a fitted
model with the best *constant* predictor. That bar is easy to clear for two
targets whose labels carry structure a constant cannot express:

``fwd_volume_ratio``
    Trading activity has an intraday cycle. Predicting the mean ratio for the
    same hour of day (a *seasonal* baseline) is a far higher bar than one global
    mean — on the committed archive it lowers the holdout RMSE by roughly
    10-20%, and it flips at least one currently-trusted target.

``fwd_volatility``
    Realised volatility is persistent. A HAR-style log-log regression on the
    trailing realised-vol lags the feature set already exposes (``vol_3``,
    ``vol_6``) is a higher bar than a constant spread on most tickers.

This module recomputes those baselines on the *same* chronologically held-out
rows the bundle was scored on, takes whichever of (constant, strong) is harder
to beat, and can rewrite the bundle's holdout verdict to that stricter bar.

OFF BY DEFAULT — nothing here changes a bundle unless :data:`ENV_FLAG` is truthy.
Enabling it makes the production gate *stricter*, so targets that report
themselves trusted today may stop emitting a signal. That is a deliberate
human-approved decision, not something a retrain may flip on its own.
Rollback: unset the variable (or ``git revert`` the commit).

Honest limits:

* The HAR leg uses the two lagged-vol features that exist today (``vol_3``,
  ``vol_6``). The classic 1/5/22-day legs would need a longer realised-vol
  feature that the feature set does not currently emit, so this is a
  *short-horizon* HAR. Where it is weaker than the constant the constant stays
  the bar: the gate can only tighten a verdict, never loosen one.
* The seasonal baseline buckets by UTC hour of day with a minimum bucket size;
  hours with too little history fall back to the global training mean.
* The gate replays the training split from samples. It is only faithful when the
  samples reproduce the bundle's holdout; :func:`compare_target` reports a
  ``skipped`` reason instead of guessing when they do not.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..dataset import parse_iso
from ..features import FEATURE_NAMES
from ..samples import Sample
from .tier1 import (
    HOLDOUT_FRACTION,
    Tier1Bundle,
    Tier1Model,
    beats_baseline,
    chronological_split,
    classification_metrics,
    regression_metrics,
)

#: Environment variable that turns the stricter gate on. Anything except a
#: truthy token leaves every bundle exactly as it is.
ENV_FLAG = "RL_CRYPTO_STRONG_BASELINE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: A seasonal bucket needs at least this many training rows to be used; below it
#: the bucket is too noisy and the global mean is the honest prediction.
SEASONAL_MIN_ROWS = 3

#: Trailing realised-vol features combined by the HAR-style baseline.
HAR_LAGS: tuple[str, ...] = ("vol_3", "vol_6")

#: Smallest positive value allowed before a log transform. Mirrors tier1's own
#: log-space scoring floor so zero target rows are excluded, not fabricated.
_LOG_FLOOR = 1e-12


def strong_baseline_enabled() -> bool:
    """Whether the stricter gate is switched on for this process."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in _TRUTHY


def _hour_of_day(timestamp: str) -> int | None:
    moment = parse_iso(timestamp)
    return None if moment is None else moment.hour


def _target_vector(samples: Sequence[Sample], target: str) -> np.ndarray:
    return np.asarray([sample.targets[target] for sample in samples], dtype=float)


def _log_space_metrics(
    truth: np.ndarray, predicted: np.ndarray
) -> dict[str, float | None]:
    """Score raw-space predictions on the log scale a log-space estimator uses.

    Rows whose truth is non-positive have no log and are excluded, exactly as
    ``tier1.__regression_metrics_in_space`` does; a test asserts the two agree.
    """
    positive = truth > 0.0
    if not np.any(positive):
        return {"rmse": None, "mae": None, "r2": None}
    clipped = np.clip(predicted[positive], _LOG_FLOOR, None)
    return regression_metrics(np.log(truth[positive]), np.log(clipped))


def _seasonal_metrics(
    train: Sequence[Sample], test: Sequence[Sample], target: str
) -> dict[str, float | None] | None:
    """Hour-of-day seasonal baseline for an activity target."""
    train_labels = _target_vector(train, target)
    if train_labels.size == 0:
        return None
    buckets: dict[int, list[float]] = {}
    for sample, label in zip(train, train_labels, strict=True):
        hour = _hour_of_day(sample.timestamp)
        if hour is not None:
            buckets.setdefault(hour, []).append(float(label))
    fallback = float(np.mean(train_labels))
    means = {
        hour: float(np.mean(values))
        for hour, values in buckets.items()
        if len(values) >= SEASONAL_MIN_ROWS
    }

    def _predict(sample: Sample) -> float:
        hour = _hour_of_day(sample.timestamp)
        return means.get(hour, fallback) if hour is not None else fallback

    predicted = np.asarray([_predict(sample) for sample in test], dtype=float)
    return regression_metrics(_target_vector(test, target), predicted)


def _har_metrics(
    train: Sequence[Sample], test: Sequence[Sample], target: str, *, space: str
) -> dict[str, float | None] | None:
    """Short-horizon HAR baseline: regress the label on its trailing vol lags."""
    train_labels = _target_vector(train, target)
    if train_labels.size == 0:
        return None

    def _design(samples: Sequence[Sample]) -> np.ndarray:
        columns = [np.ones(len(samples), dtype=float)]
        for name in HAR_LAGS:
            columns.append(
                np.asarray([sample.features.get(name) for sample in samples], dtype=float)
            )
        return np.column_stack(columns)

    if space == "log":
        # Log-log HAR: fit log(vol) on log(lags). Only rows with a positive label
        # and positive lags are informative; zeros carry no log.
        keep = train_labels > 0.0
        for name in HAR_LAGS:
            keep &= np.asarray(
                [sample.features.get(name) for sample in train], dtype=float
            ) > 0.0
        if int(keep.sum()) < SEASONAL_MIN_ROWS:
            return None
        kept_train = [sample for sample, use in zip(train, keep, strict=True) if use]
        design = np.log(_design(kept_train))
        design[:, 0] = np.ones(len(kept_train))  # intercept stays a plain 1
        response = np.log(train_labels[keep])
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        test_design = np.log(
            np.clip(_design(test)[:, 1:], _LOG_FLOOR, None)
        )
        test_design = np.column_stack([np.ones(len(test)), test_design])
        predicted = np.exp(test_design @ coefficients)
        return _log_space_metrics(_target_vector(test, target), predicted)

    keep = train_labels > 0.0
    if int(keep.sum()) < SEASONAL_MIN_ROWS:
        return None
    coefficients, *_ = np.linalg.lstsq(_design(train)[keep], train_labels[keep], rcond=None)
    predicted = _design(test) @ coefficients
    return regression_metrics(_target_vector(test, target), predicted)


def strong_metrics(
    target: str,
    kind: str,
    train: Sequence[Sample],
    test: Sequence[Sample],
    *,
    space: str = "raw",
) -> dict[str, float | None] | None:
    """Metrics for the strongest simple baseline this target admits.

    Returns ``None`` when no stronger-than-constant baseline is defined for the
    target (the return and direction targets keep the constant bar).
    """
    if kind != "regression" or not test or not train:
        return None
    if target == "fwd_volume_ratio":
        return _seasonal_metrics(train, test, target)
    if target == "fwd_volatility":
        return _har_metrics(train, test, target, space=space)
    return None


def stricter_baseline(
    kind: str,
    constant: Mapping[str, float | None] | None,
    strong: Mapping[str, float | None] | None,
) -> tuple[dict[str, float | None], str]:
    """Return the harder-to-beat baseline and its name.

    The verdict key is RMSE for regression and log-loss for classification. The
    comparison is a strict ``<`` on that key, so a tie / unmeasurable strong
    baseline keeps the constant bar and a verdict can never be loosened.
    """
    if strong is None:
        return dict(constant or {}), "constant"
    key = "rmse" if kind == "regression" else "log_loss"
    strong_value = strong.get(key)
    constant_value = (constant or {}).get(key)
    if strong_value is None or constant_value is None:
        return dict(constant or {}), "constant"
    if strong_value < constant_value:
        return dict(strong), "strong"
    return dict(constant or {}), "constant"


def baseline_label(target: str) -> str:
    """Human name for the strong baseline a target uses, for notes and reports."""
    if target == "fwd_volume_ratio":
        return "seasonal hour-of-day"
    if target == "fwd_volatility":
        return "HAR(3,6)"
    return "constant"


def compare_target(
    target: str,
    model: Tier1Model,
    train: Sequence[Sample],
    test: Sequence[Sample],
    *,
    expected_holdout: int | None = None,
) -> dict[str, Any]:
    """Compare a model's stored verdict with the stricter-baseline verdict.

    ``before`` is the verdict the bundle ships; ``after`` is what the gate would
    write. ``skipped`` explains why nothing was computed when the replay does not
    reproduce the stored holdout size.
    """
    row: dict[str, Any] = {
        "target": target,
        "kind": model.kind,
        "before": model.holdout.beats_baseline if model.holdout else None,
        "after": model.holdout.beats_baseline if model.holdout else None,
        "baseline_used": "constant",
        "constant": dict(model.holdout.baseline) if model.holdout else {},
        "strong": None,
        "skipped": None,
        "replay_ok": True,
    }
    if model.holdout is None:
        row["skipped"] = "no holdout recorded"
        return row
    if expected_holdout is not None and len(test) != expected_holdout:
        row["skipped"] = (
            f"holdout replay has {len(test)} rows, bundle recorded {expected_holdout}"
        )
        row["replay_ok"] = False
        return row
    strong = strong_metrics(target, model.kind, train, test, space=model.metric_space)
    if strong is None:
        row["skipped"] = "no stronger baseline defined"
        return row
    chosen, which = stricter_baseline(model.kind, model.holdout.baseline, strong)
    row["strong"] = dict(strong)
    row["baseline_used"] = which
    row["after"] = beats_baseline(model.kind, model.holdout.metrics, chosen)
    return row


def replay_split(
    bundle: Tier1Bundle,
    samples: Sequence[Sample],
    target: str,
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> tuple[list[Sample], list[Sample]]:
    names = tuple(feature_names)
    complete = [
        sample
        for sample in samples
        if all(sample.features.get(name) is not None for name in names)
    ]
    usable = [sample for sample in complete if sample.targets.get(target) is not None]
    train, test, _ = chronological_split(
        usable,
        target=target,
        horizon=bundle.horizon,
        holdout_fraction=holdout_fraction,
    )
    return list(train), list(test)


def gate_bundle(
    bundle: Tier1Bundle,
    samples: Sequence[Sample],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    holdout_fraction: float = HOLDOUT_FRACTION,
    force: bool = False,
) -> Tier1Bundle:
    """Rewrite ``bundle``'s trust verdicts to the stricter baseline, in place.

    No-op unless the flag is set (or ``force`` is passed for offline reporting).
    The gate only tightens: a target whose stored verdict already fails, or whose
    strong baseline is weaker than the stored constant, is left alone.
    """
    if not force and not strong_baseline_enabled():
        return bundle
    notes = list(bundle.notes)
    for target, model in bundle.targets.items():
        if model.holdout is None:
            continue
        train, test = replay_split(
            bundle,
            samples,
            target,
            feature_names=feature_names,
            holdout_fraction=holdout_fraction,
        )
        row = compare_target(
            target, model, train, test, expected_holdout=model.holdout.n_samples
        )
        if row["skipped"] is not None:
            notes.append(f"{target}: stronger-baseline gate skipped ({row['skipped']})")
            continue
        if row["baseline_used"] != "strong":
            notes.append(
                f"{target}: {baseline_label(target)} baseline is no stronger than the "
                "constant; constant kept"
            )
            continue
        chosen, _ = stricter_baseline(model.kind, model.holdout.baseline, row["strong"])
        model.holdout.baseline = chosen
        model.holdout.beats_baseline = bool(row["after"])
        notes.append(
            f"{target}: trust verdict gated on the stronger "
            f"{baseline_label(target)} baseline"
        )
    bundle.notes = tuple(notes)
    return bundle


__all__ = [
    "ENV_FLAG",
    "HAR_LAGS",
    "SEASONAL_MIN_ROWS",
    "baseline_label",
    "compare_target",
    "gate_bundle",
    "replay_split",
    "stricter_baseline",
    "strong_baseline_enabled",
    "strong_metrics",
]
