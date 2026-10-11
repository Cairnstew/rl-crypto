"""Tier 1 — one structured model per target metric.

The trainer turns a sequence of :class:`~rl_crypto.samples.Sample` into a
:class:`Tier1Bundle`: for each target it keeps the samples whose features are
complete *and* whose forward label exists, fits an estimator, and records
in-sample metrics. A target with too few usable samples is skipped with a note
rather than fit on noise.

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

from ..features import FEATURE_NAMES
from ..models import to_iso, utcnow
from ..samples import Sample
from ..targets import HORIZON_DEFAULT, TARGET_KINDS, TARGET_NAMES
from .base import Model, make_model, model_from_dict
from .types import ML_SCHEMA

#: A target is only fitted when it has at least this many usable samples.
MIN_TRAIN_SAMPLES = 8


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


# --------------------------------------------------------------------------- tier-1 model


@dataclass(slots=True)
class Tier1Model:
    """One fitted estimator plus the feature contract and its training metrics."""

    target: str
    kind: str
    feature_names: tuple[str, ...]
    model: Model
    metrics: dict[str, float | None] = field(default_factory=dict)
    n_samples: int = 0

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
            "metrics": dict(self.metrics),
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
        )


@dataclass(slots=True)
class Tier1Bundle:
    """Every Tier-1 model for one ticker, with the shared feature contract."""

    feature_names: tuple[str, ...]
    horizon: int
    targets: dict[str, Tier1Model] = field(default_factory=dict)
    trained_at: str | None = None
    n_samples: int = 0
    notes: tuple[str, ...] = ()
    schema: str = ML_SCHEMA

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trained_at": self.trained_at,
            "horizon": self.horizon,
            "n_samples": self.n_samples,
            "feature_names": list(self.feature_names),
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
) -> Tier1Bundle:
    """Fit one estimator per requested target over the usable samples."""
    names = tuple(feature_names)
    wanted = tuple(targets) if targets else TARGET_NAMES
    overrides = dict(model_names or {})
    complete = [sample for sample in samples if _complete(sample, names)]

    fitted: dict[str, Tier1Model] = {}
    notes: list[str] = []
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
        matrix = np.asarray(
            [[s.features[name] for name in names] for s in usable], dtype=float
        )
        labels = np.asarray([s.targets[target] for s in usable], dtype=float)
        model = make_model(kind, overrides.get(target))  # type: ignore[arg-type]
        model.fit(matrix, labels)
        predicted = model.predict(matrix)
        metrics = (
            regression_metrics(labels, predicted)
            if kind == "regression"
            else classification_metrics(labels, predicted)
        )
        fitted[target] = Tier1Model(
            target=target,
            kind=kind,
            feature_names=names,
            model=model,
            metrics=metrics,
            n_samples=len(usable),
        )

    if not fitted:
        notes.append("no targets had enough usable samples to fit")
    return Tier1Bundle(
        feature_names=names,
        horizon=horizon,
        targets=fitted,
        trained_at=trained_at or to_iso(utcnow()),
        n_samples=len(complete),
        notes=tuple(notes),
    )


__all__ = [
    "MIN_TRAIN_SAMPLES",
    "Tier1Bundle",
    "Tier1Model",
    "classification_metrics",
    "regression_metrics",
    "train_tier1",
]
