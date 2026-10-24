"""Tier 1 — one structured model per target metric, honestly measured.

The trainer turns a sequence of :class:`~rl_crypto.samples.Sample` into a
:class:`Tier1Bundle`: for each target it keeps the samples whose features are
complete *and* whose forward label exists, fits an estimator, and reports both
**in-sample** and **held-out** metrics. A target with too few usable samples is
skipped with a note rather than fit on noise.

Why the split is not optional
-----------------------------
A fit scored on the rows it was fitted on measures how much of the *training*
noise it memorised, not how well it predicts. On this data that difference is
not marginal: ``fwd_direction`` scores ~0.55 in-sample and ~0.49 on unseen rows
— the latter being *worse than always predicting "up"*. Reporting only the
former is how a pipeline ends up publishing confident signals from a model that
has none. So every model here carries an out-of-sample verdict:

* the **shipped weights are the ones fitted on the training split only**, so the
  published holdout metrics describe exactly the artifact in the bundle — no
  "but we refit afterwards" ambiguity. The withheld tail is the most recent
  data, which is also the data the live path predicts *forward* into, so the
  evaluation matches the deployment question.
* the withheld rows are **not discarded**. The production runner retrains every
  two hours over a growing archive, so each run folds them back in.

The bundle is the artifact the live path loads. Its JSON form is stable and
diffable (see :mod:`rl_crypto.ml.store`), and every model inside it can be
rebuilt from that JSON alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..features import FEATURE_NAMES, FEATURE_VERSION
from ..models import to_iso, utcnow
from ..samples import Sample
from ..targets import HORIZON_DEFAULT, TARGET_KINDS, TARGET_NAMES
from .base import Model, make_model, model_from_dict
from .types import ML_SCHEMA

#: A target is only fitted when it has at least this many usable samples.
MIN_TRAIN_SAMPLES = 8

#: Fraction of usable samples reserved, newest-first, as an untouched holdout.
HOLDOUT_FRACTION = 0.2

#: The holdout needs at least this many rows to mean anything at all.
MIN_HOLDOUT_SAMPLES = 5

#: A model must beat its trivial baseline by this *relative* margin to be
#: trusted. Absolute "any improvement" is too loose: on a ~300-row holdout a
#: 0.1% log-loss difference is noise, and treating it as skill is how a model
#: earns the right to emit a signal it has not earned. Measured on this data the
#: threshold cleanly separates the real margins (volatility 2-13%, volume
#: ratio 14-24%) from the noise ones (direction 0.1-0.3%).
RELATIVE_SKILL_MARGIN = 0.01

#: Per-target estimator overrides, applied when the caller names none.
#:
#: ``fwd_volatility`` is a standard deviation, so it has a non-negative domain.
#: The reference ridge is unbounded: on live feature rows its sign is arbitrary,
#: and before this override 8 of the 10 "trusted" volatility models served a
#: negative spread. ``log_ridge`` fits ``log(vol)`` and exponentiates back, so
#: the served value is positive by construction without changing the target
#: definition or the raw-scale holdout comparison.
DEFAULT_MODEL_BY_TARGET: dict[str, str] = {"fwd_volatility": "log_ridge"}

#: Smallest strictly-positive value used when scoring in log space. A zero
#: realised spread carries no log; such rows are excluded from log-space
#: scoring rather than floored, so one quiet bar cannot dominate the metric.
_LOG_FLOOR = 1e-12


# --------------------------------------------------------------------------- metrics


def regression_metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    """RMSE, MAE, and R2 of a regression fit."""
    truth = np.asarray(y, dtype=float).reshape(-1)
    guess = np.asarray(predicted, dtype=float).reshape(-1)
    residual = truth - guess
    total = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "rmse": float(math.sqrt(float(np.mean(residual**2)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": (1.0 - float(np.sum(residual**2)) / total) if total > 0.0 else None,
    }


def classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    """Accuracy, log-loss, and the positive base rate of a classifier."""
    truth = np.clip(np.asarray(y, dtype=float).reshape(-1), 0.0, 1.0)
    proba = np.clip(np.asarray(probability, dtype=float).reshape(-1), 1e-12, 1.0 - 1e-12)
    correct = (proba >= 0.5).astype(float) == truth
    return {
        "accuracy": float(np.mean(correct)) if truth.size else None,
        "log_loss": float(
            -np.mean(truth * np.log(proba) + (1.0 - truth) * np.log(1.0 - proba))
        ),
        "base_rate": float(truth.mean()) if truth.size else None,
    }


def baseline_for(
    kind: str, y_train: np.ndarray, y_test: np.ndarray, *, space: str = "raw"
) -> dict[str, float | None]:
    """Metrics for the best *constant* predictor of ``y_test`` known from ``y_train``.

    This is the bar a real model has to clear. A model that cannot beat it has
    learned nothing an average would not have given it, and its predictions
    should not be presented as signal.

    ``space`` selects the scale the comparison happens on. ``"raw"`` is the
    ordinary target scale. ``"log"`` is used by a log-space estimator: both the
    truth and the constant are scored as logs, so the model is judged on the
    loss it actually minimises. Rows with a non-positive target have no log and
    are excluded; a single zero would otherwise make the log baseline infinite
    and hand the model a fake win.
    """
    if kind == "regression":
        if space == "log":
            positive = y_test > 0.0
            train = y_train[y_train > 0.0]
            if train.size == 0 or not np.any(positive):
                return {"rmse": None, "mae": None, "r2": None}
            constant = float(np.mean(np.log(train)))
            held = np.log(y_test[positive])
            return regression_metrics(held, np.full(held.shape, constant))
        prediction = float(np.mean(y_train)) if y_train.size else 0.0
        return regression_metrics(y_test, np.full(y_test.shape, prediction))
    rate = float(np.mean(y_train)) if y_train.size else 0.5
    # Both the always-up and always-down constant are available to an oracle
    # that knows the test base rate; scoring the better of the two is the fair,
    # slightly generous bar.
    return classification_metrics(
        y_test, np.full(y_test.shape, max(rate, 1.0 - rate))
    )


def beats_baseline(
    kind: str,
    metrics: Mapping[str, float | None],
    baseline: Mapping[str, float | None],
) -> bool:
    """Whether ``metrics`` beat ``baseline`` by a margin worth acting on.

    Regression compares RMSE (lower is better); classification compares log-loss
    rather than accuracy, because accuracy on a ~50/50 label moves in noisy
    steps while log-loss moves smoothly and punishes confident-but-wrong. In
    both cases the comparison is *relative* (:data:`RELATIVE_SKILL_MARGIN`), so
    a rounding-error improvement does not read as skill.
    """
    if kind == "regression":
        got, bar = metrics.get("rmse"), baseline.get("rmse")
    else:
        got, bar = metrics.get("log_loss"), baseline.get("log_loss")
    if got is None or bar is None:
        return False
    if bar <= 0.0:
        return got < bar
    return bool(got < bar * (1.0 - RELATIVE_SKILL_MARGIN))


def _regression_metrics_in_space(
    truth: np.ndarray, predicted: np.ndarray, space: str
) -> dict[str, float | None]:
    """Score a regression fit on the scale its estimator optimises."""
    if space != "log":
        return regression_metrics(truth, predicted)
    positive = truth > 0.0
    if not np.any(positive):
        return {"rmse": None, "mae": None, "r2": None}
    return regression_metrics(
        np.log(truth[positive]),
        np.log(np.clip(predicted[positive], _LOG_FLOOR, None)),
    )


def usable_features(
    samples: Sequence[Sample], feature_names: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ``feature_names`` into those kept and those dropped as constant.

    A feature that never varies across the training rows carries no information,
    but it still occupies a column — and, worse, it becomes an out-of-distribution
    input the moment live data does vary it. The news group is exactly this case
    today: the archive is market-only, so every news feature is a constant, yet
    the live path has news. Dropping such columns here means the model only ever
    reads inputs it was actually trained on, and they light up on their own once
    news history lands.
    """
    if not samples:
        return tuple(feature_names), ()
    kept: list[str] = []
    dropped: list[str] = []
    for name in feature_names:
        values = [sample.features.get(name) for sample in samples]
        present = [float(v) for v in values if v is not None]
        if not present or max(present) == min(present):
            dropped.append(name)
        else:
            kept.append(name)
    return tuple(kept), tuple(dropped)


# --------------------------------------------------------------------------- tier-1 model


@dataclass(slots=True)
class HoldoutReport:
    """The out-of-sample verdict for one fitted model.

    ``metrics`` and ``baseline`` are both computed on rows the model never saw.
    ``beats_baseline`` is the single field downstream code should gate on: it is
    ``False`` for a model that is no better than predicting a constant, whatever
    its in-sample score says.
    """

    n_samples: int = 0
    metrics: dict[str, float | None] = field(default_factory=dict)
    baseline: dict[str, float | None] = field(default_factory=dict)
    beats_baseline: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "metrics": dict(self.metrics),
            "baseline": dict(self.baseline),
            "beats_baseline": self.beats_baseline,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> HoldoutReport | None:
        if payload is None:
            return None
        beats = payload.get("beats_baseline")
        return cls(
            n_samples=int(payload.get("n_samples", 0)),
            metrics=dict(payload.get("metrics") or {}),
            baseline=dict(payload.get("baseline") or {}),
            beats_baseline=None if beats is None else bool(beats),
        )


@dataclass(slots=True)
class Tier1Model:
    """One fitted estimator plus the feature contract and its training metrics.

    ``metrics`` is **in-sample** (fit quality on the training split) and
    ``holdout`` is the out-of-sample verdict. Readers should prefer ``holdout``.
    """

    target: str
    kind: str
    feature_names: tuple[str, ...]
    model: Model
    metrics: dict[str, float | None] = field(default_factory=dict)
    n_samples: int = 0
    #: Scale the regression metrics are computed on: ``"raw"`` or ``"log"``.
    #: A log-space estimator is judged on log loss against a log-space constant;
    #: scoring it on the raw scale would report a loss it never optimised.
    metric_space: str = "raw"
    holdout: HoldoutReport | None = None

    @property
    def beats_baseline(self) -> bool | None:
        """Whether this model beat a constant predictor out of sample."""
        return None if self.holdout is None else self.holdout.beats_baseline

    def predict_row(self, features: Mapping[str, float | None]) -> float | None:
        """Predict one row, or ``None`` when any required feature is missing."""
        vector = [features.get(name) for name in self.feature_names]
        if any(value is None for value in vector):
            return None
        matrix = np.asarray([vector], dtype=float)
        return float(self.model.predict(matrix)[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "n_samples": self.n_samples,
            "metric_space": self.metric_space,
            "metrics": dict(self.metrics),
            "holdout": None if self.holdout is None else self.holdout.to_dict(),
            "model": self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Tier1Model:
        kind = str(payload["kind"])
        return cls(
            target=str(payload["target"]),
            kind=kind,
            feature_names=tuple(payload["feature_names"]),
            model=model_from_dict(kind, payload["model"]),  # type: ignore[arg-type]
            metrics=dict(payload.get("metrics") or {}),
            n_samples=int(payload.get("n_samples", 0)),
            metric_space=str(payload.get("metric_space", "raw")),
            holdout=HoldoutReport.from_dict(payload.get("holdout")),
        )


@dataclass(slots=True)
class Tier1Bundle:
    """Every Tier-1 model for one ticker, with the shared feature contract.

    ``feature_version`` is recorded at training time and re-checked on load: a
    bundle fitted against one feature definition and served against another is
    silent corruption, and the names alone do not catch a reordering or a
    changed meaning under an unchanged name.
    """

    feature_names: tuple[str, ...]
    horizon: int
    targets: dict[str, Tier1Model] = field(default_factory=dict)
    trained_at: str | None = None
    n_samples: int = 0
    notes: tuple[str, ...] = ()
    schema: str = ML_SCHEMA
    feature_version: str = FEATURE_VERSION
    dropped_features: tuple[str, ...] = ()
    holdout_n_samples: int = 0
    feature_version_mismatch: bool = False

    def predict(
        self, features: Mapping[str, float | None]
    ) -> tuple[dict[str, float | None], dict[str, float | None]]:
        """Return ``(point_predictions, probabilities)`` for one feature row.

        A classification target contributes its ``P(class=1)`` to both: the
        probability under its name, and the 0/1 decision per the 0.5 threshold
        hashed into ``probabilities``. A regression target contributes only a
        point value.
        """
        points: dict[str, float | None] = {}
        probabilities: dict[str, float | None] = {}
        for target, model in self.targets.items():
            value = model.predict_row(features)
            if model.kind == "classification":
                probabilities[target] = value
                points[target] = None if value is None else (1.0 if value >= 0.5 else 0.0)
            else:
                points[target] = value
        return points, probabilities

    def trusted(self, target: str) -> bool:
        """Whether ``target`` may be presented as signal.

        True only when the model carries an out-of-sample verdict and that
        verdict says it beat a constant predictor. An unmeasured model (older
        bundle, too few rows for a holdout) is not trusted — absence of evidence
        is not evidence of quality.
        """
        model = self.targets.get(target)
        if model is None or self.feature_version_mismatch:
            return False
        return model.beats_baseline is True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "feature_version": self.feature_version,
            "trained_at": self.trained_at,
            "horizon": self.horizon,
            "n_samples": self.n_samples,
            "holdout_n_samples": self.holdout_n_samples,
            "feature_names": list(self.feature_names),
            "dropped_features": list(self.dropped_features),
            "targets": {name: model.to_dict() for name, model in self.targets.items()},
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Tier1Bundle:
        targets = {
            name: Tier1Model.from_dict(model_payload)
            for name, model_payload in (payload.get("targets") or {}).items()
        }
        return cls(
            feature_names=tuple(payload["feature_names"]),
            horizon=int(payload.get("horizon", HORIZON_DEFAULT)),
            targets=targets,
            trained_at=payload.get("trained_at"),
            n_samples=int(payload.get("n_samples", 0)),
            notes=tuple(payload.get("notes") or ()),
            schema=str(payload.get("schema", ML_SCHEMA)),
            feature_version=str(payload.get("feature_version", "")),
            dropped_features=tuple(payload.get("dropped_features") or ()),
            holdout_n_samples=int(payload.get("holdout_n_samples", 0)),
        )


# --------------------------------------------------------------------------- training


def _complete(sample: Sample, feature_names: Sequence[str]) -> bool:
    return all(sample.features.get(name) is not None for name in feature_names)


def train_tier1(
    samples: Sequence[Sample],
    *,
    horizon: int = HORIZON_DEFAULT,
    feature_names: Sequence[str] = FEATURE_NAMES,
    targets: Sequence[str] | None = None,
    model_names: Mapping[str, str] | None = None,
    trained_at: str | None = None,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> Tier1Bundle:
    """Fit one estimator per requested target over the usable samples.

    Samples are split chronologically — the newest ``holdout_fraction`` are
    reserved and never fitted on — and each model is scored against a constant
    baseline on that reserved tail. Models are fitted on the training split only,
    so the ``holdout`` verdict in the bundle describes the very weights being
    shipped.
    """
    names = tuple(feature_names)
    wanted = tuple(targets) if targets else TARGET_NAMES
    overrides = dict(model_names or {})

    complete = [s for s in samples if _complete(s, names)]
    kept_names, dropped_names = usable_features(complete, names)
    notes: list[str] = []
    if dropped_names:
        notes.append(
            f"dropped {len(dropped_names)} constant feature(s) with no variation in "
            f"training data: {', '.join(dropped_names)}"
        )
    if not kept_names:
        notes.append("no feature varied in the training data; nothing can be fitted")
        return Tier1Bundle(
            feature_names=(),
            horizon=horizon,
            trained_at=trained_at or to_iso(utcnow()),
            n_samples=len(complete),
            dropped_features=dropped_names,
            notes=tuple(notes),
        )

    def _vectors(usable: Sequence[Sample]) -> np.ndarray:
        return np.asarray(
            [[s.features[name] for name in kept_names] for s in usable], dtype=float
        )

    fitted: dict[str, Tier1Model] = {}
    holdout_n = 0
    for target in wanted:
        kind = TARGET_KINDS.get(target)
        if kind is None:
            notes.append(f"{target}: unknown target, skipped")
            continue
        usable = [s for s in complete if s.targets.get(target) is not None]
        if len(usable) < MIN_TRAIN_SAMPLES:
            notes.append(
                f"{target}: skipped ({len(usable)} usable < {MIN_TRAIN_SAMPLES})"
            )
            continue

        # Chronological split: samples arrive in time order, so the tail is the
        # most recent data — the same "predict forward from here" question the
        # live path asks.
        n_holdout = int(len(usable) * holdout_fraction)
        if n_holdout < MIN_HOLDOUT_SAMPLES:
            notes.append(
                f"{target}: no holdout ({len(usable)} usable rows cannot spare "
                f"{MIN_HOLDOUT_SAMPLES}); reported as unmeasured"
            )
            split = len(usable)
        else:
            split = len(usable) - n_holdout
            holdout_n = max(holdout_n, n_holdout)

        train_rows, test_rows = usable[:split], usable[split:]
        matrix = _vectors(train_rows)
        labels = np.asarray([s.targets[target] for s in train_rows], dtype=float)
        chosen = overrides.get(target) or DEFAULT_MODEL_BY_TARGET.get(target)
        model = make_model(kind, chosen)  # type: ignore[arg-type]
        model.fit(matrix, labels)
        predicted = model.predict(matrix)
        space = str(getattr(model, "metric_space", "raw"))
        metrics = (
            _regression_metrics_in_space(labels, predicted, space)
            if kind == "regression"
            else classification_metrics(labels, predicted)
        )

        holdout: HoldoutReport | None = None
        if test_rows:
            test_matrix = _vectors(test_rows)
            test_labels = np.asarray([s.targets[target] for s in test_rows], dtype=float)
            test_pred = model.predict(test_matrix)
            held_metrics = (
                _regression_metrics_in_space(test_labels, test_pred, space)
                if kind == "regression"
                else classification_metrics(test_labels, test_pred)
            )
            baseline = baseline_for(kind, labels, test_labels, space=space)
            holdout = HoldoutReport(
                n_samples=len(test_rows),
                metrics=held_metrics,
                baseline=baseline,
                beats_baseline=beats_baseline(kind, held_metrics, baseline),
            )
            verdict = "beats" if holdout.beats_baseline else "does not beat"
            key = "rmse" if kind == "regression" else "log_loss"
            got, bar = held_metrics.get(key), baseline.get(key)
            scale = "log-space " if space == "log" else ""
            if got is None or bar is None:
                detail = f"{scale}{key}=unmeasurable"
            else:
                improvement = (
                    f" ({100.0 * (bar - got) / bar:+.2f}% vs baseline)" if bar else ""
                )
                detail = f"{scale}{key}={got:.5f} vs baseline {bar:.5f}{improvement}"
            needed = f"need {RELATIVE_SKILL_MARGIN:.0%} to count as skill"
            notes.append(
                f"{target}: out-of-sample {detail} — {verdict} it "
                f"({len(test_rows)} held-out rows, {needed})"
            )

        fitted[target] = Tier1Model(
            target=target,
            kind=kind,
            feature_names=kept_names,
            model=model,
            metrics=metrics,
            n_samples=len(train_rows),
            metric_space=space,
            holdout=holdout,
        )

    if not fitted:
        notes.append("no targets had enough usable samples to fit")
    return Tier1Bundle(
        feature_names=kept_names,
        horizon=horizon,
        targets=fitted,
        trained_at=trained_at or to_iso(utcnow()),
        n_samples=len(complete),
        notes=tuple(notes),
        dropped_features=dropped_names,
        holdout_n_samples=holdout_n,
    )


__all__ = [
    "HOLDOUT_FRACTION",
    "MIN_HOLDOUT_SAMPLES",
    "MIN_TRAIN_SAMPLES",
    "HoldoutReport",
    "Tier1Bundle",
    "Tier1Model",
    "baseline_for",
    "beats_baseline",
    "classification_metrics",
    "regression_metrics",
    "train_tier1",
    "usable_features",
]
