"""Shared ML primitives: the :class:`Model` protocol, standardiser, and registry.

Tier 1 models are deliberately small and interchangeable. A model is anything
that can ``fit`` on a feature matrix, ``predict`` a vector, and serialise itself
to JSON — the reference implementations in :mod:`rl_crypto.ml.linear` are pure
numpy, so the pipeline stays hermetic and offline-testable. Because a model is
identified by its ``name`` and rebuilt from :meth:`to_dict`, a trained bundle
is a plain, reviewable JSON file rather than a pickle.

Classification models return the **probability of the positive class** from
``predict`` (there is no separate ``predict_proba``), so the Tier-1 layer can
treat both task kinds uniformly.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

TaskKind = Literal["regression", "classification"]


class Standardizer:
    """Column-wise z-scoring, fitted once and stored with the model.

    Linear models without it are dominated by whichever feature happens to have
    the largest scale (volume and macd-like gaps differ by orders of magnitude),
    so every reference estimator standardises its inputs and remembers how.
    """

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> None:
        self.mean_ = mean
        self.std_ = std

    def fit(self, X: np.ndarray) -> Standardizer:
        matrix = np.asarray(X, dtype=float)
        self.mean_ = matrix.mean(axis=0)
        spread = matrix.std(axis=0)
        spread[spread == 0.0] = 1.0
        self.std_ = spread
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer used before fit")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Standardizer:
        mean = payload.get("mean")
        std = payload.get("std")
        return cls(
            mean=np.asarray(mean, dtype=float) if mean is not None else None,
            std=np.asarray(std, dtype=float) if std is not None else None,
        )


@runtime_checkable
class Model(Protocol):
    """A tiny fitted-model interface shared by every Tier-1 estimator."""

    name: str
    kind: TaskKind

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def to_dict(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- registry

MODEL_TYPES: dict[str, type] = {}

DEFAULT_MODEL: dict[str, str] = {"regression": "ridge", "classification": "logistic"}


def register_model(cls: type) -> type:
    """Register an estimator class under its ``name`` (used as a decorator)."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    existing = MODEL_TYPES.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"model name {name!r} already registered by {existing.__name__}")
    MODEL_TYPES[name] = cls
    return cls


def _ensure_registered() -> None:
    """Import the reference estimators so the registry is populated."""
    from . import linear  # noqa: F401  (import side-effect registers the models)


def models_for(kind: TaskKind) -> dict[str, type]:
    """Every registered estimator for a task kind, keyed by name."""
    _ensure_registered()
    return {name: cls for name, cls in MODEL_TYPES.items() if cls.kind == kind}


def model_names() -> tuple[str, ...]:
    """All registered estimator names, sorted."""
    _ensure_registered()
    return tuple(sorted(MODEL_TYPES))


def make_model(kind: TaskKind, name: str | None = None, **params: Any) -> Model:
    """Construct an unfitted estimator of ``kind`` (defaults to the reference one)."""
    _ensure_registered()
    chosen = name or DEFAULT_MODEL[kind]
    cls = MODEL_TYPES.get(chosen)
    if cls is None:
        raise KeyError(f"unknown model {chosen!r}; known: {sorted(MODEL_TYPES)}")
    if cls.kind != kind:
        raise ValueError(f"model {chosen!r} is a {cls.kind} model, not {kind}")
    return cls(**params)


def model_from_dict(kind: TaskKind, payload: dict[str, Any]) -> Model:
    """Rebuild a fitted estimator from its :meth:`Model.to_dict` output."""
    _ensure_registered()
    name = payload.get("kind") or payload.get("name")
    cls = MODEL_TYPES.get(name) if isinstance(name, str) else None
    if cls is None:
        raise KeyError(f"unknown serialised model {name!r}")
    if cls.kind != kind:
        raise ValueError(f"serialised model {name!r} is a {cls.kind} model, not {kind}")
    return cls.from_dict(payload)


__all__ = [
    "DEFAULT_MODEL",
    "MODEL_TYPES",
    "Model",
    "Standardizer",
    "TaskKind",
    "make_model",
    "model_from_dict",
    "model_names",
    "models_for",
    "register_model",
]
