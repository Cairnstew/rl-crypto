"""Reference Tier-1 estimators — pure numpy, no model-serving dependency.

Five estimators, all serialising to JSON:

======================  =================  ====================================
name                    kind               idea
======================  =================  ====================================
``ridge``               regression         L2-regularised least squares (closed form)
``log_ridge``           regression         ridge fitted on ``log(y)``, exponentiated back
``mean``                regression         predict the training mean (baseline)
``logistic``            classification     L2-regularised logistic regression (GD)
``base_rate``           classification     predict the training positive rate
======================  =================  ====================================

``log_ridge`` exists for targets with a non-negative domain (a realised
volatility). A plain ``ridge`` is unbounded, so on live feature rows it can
predict a negative spread — a physically impossible number — even when its
holdout score looks fine. Fitting in log space makes ``exp(...)`` positive by
construction while keeping the linear closed form; the inverse-transformed value
is on the target's own scale, so raw-scale metrics and the constant baseline
stay comparable.

They exist so the tiered pipeline runs with zero extra dependencies and can be
exercised offline. A heavier backend (scikit-learn, gradient boosting) can be
dropped in behind :class:`~rl_crypto.ml.base.Model` without touching Tier-1/2.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import Model, Standardizer, TaskKind, register_model

_SIGMOID_CLIP = 60.0

#: Bound the log-space linear predictor so ``exp`` neither overflows nor
#: underflows to ``0.0`` on an extreme out-of-range feature row. ``exp(±700)``
#: stays inside float64 and, crucially, remains strictly positive.
_EXP_CLIP = 700.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -_SIGMOID_CLIP, _SIGMOID_CLIP)))


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _as_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {matrix.shape}")
    return matrix


def _as_target(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape(-1)


@register_model
class Ridge:
    """L2-regularised least squares, fitted in closed form on standardised inputs."""

    name = "ridge"
    kind: TaskKind = "regression"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self._scaler = Standardizer()
        self._coef: np.ndarray | None = None
        self._intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> Ridge:
        matrix = _as_matrix(X)
        target = _as_target(y)
        if matrix.shape[0] != target.shape[0]:
            raise ValueError("X and y disagree on the number of samples")
        self._scaler.fit(matrix)
        scaled = self._scaler.transform(matrix)
        gram = scaled.T @ scaled + self.alpha * np.eye(scaled.shape[1])
        self._coef = np.linalg.solve(gram, scaled.T @ (target - target.mean()))
        # Standardised columns are centred, so no bias term is needed in the
        # normal equations; the intercept is the training mean of y.
        self._intercept = float(target.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("Ridge used before fit")
        scaled = self._scaler.transform(_as_matrix(X))
        return scaled @ self._coef + self._intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.name,
            "alpha": self.alpha,
            "scaler": self._scaler.to_dict(),
            "coef": None if self._coef is None else self._coef.tolist(),
            "intercept": self._intercept,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Ridge:
        model = cls(alpha=float(payload.get("alpha", 1.0)))
        model._scaler = Standardizer.from_dict(payload["scaler"])
        coef = payload.get("coef")
        model._coef = np.asarray(coef, dtype=float) if coef is not None else None
        model._intercept = float(payload.get("intercept", 0.0))
        return model


@register_model
class LogRidge:
    """Ridge fitted in ``log`` space and exponentiated back to the target scale.

    For a non-negative target such as :func:`~rl_crypto.targets.forward_volatility`
    (a standard deviation), an unconstrained linear fit is not merely imprecise:
    its sign on a live feature row is arbitrary, and a negative spread is a
    number no market can produce. This estimator keeps the same closed-form
    solve, fits ``log(y)``, and returns ``exp(...)``, so every prediction is
    positive by construction. Metrics are still computed on the raw scale
    against the same constant baseline, so the holdout verdict stays comparable.

    Rows whose target is not strictly positive (a zero realised spread) carry no
    log and are excluded from the fit; they are still scored, because the model
    predicts for them like any other row.
    """

    name = "log_ridge"
    kind: TaskKind = "regression"
    #: This estimator minimises squared error in log space, so its metrics and
    #: constant baseline must be computed there too (see ``Tier1.train_tier1``).
    metric_space = "log"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self._scaler = Standardizer()
        self._coef: np.ndarray | None = None
        self._intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogRidge:
        matrix = _as_matrix(X)
        target = _as_target(y)
        if matrix.shape[0] != target.shape[0]:
            raise ValueError("X and y disagree on the number of samples")
        positive = target > 0.0
        if not np.any(positive):
            raise ValueError("log_ridge needs at least one strictly positive target")
        matrix = matrix[positive]
        log_target = np.log(target[positive])
        self._scaler.fit(matrix)
        scaled = self._scaler.transform(matrix)
        gram = scaled.T @ scaled + self.alpha * np.eye(scaled.shape[1])
        self._coef = np.linalg.solve(gram, scaled.T @ (log_target - log_target.mean()))
        self._intercept = float(log_target.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("LogRidge used before fit")
        scaled = self._scaler.transform(_as_matrix(X))
        return np.exp(np.clip(scaled @ self._coef + self._intercept, -_EXP_CLIP, _EXP_CLIP))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.name,
            "alpha": self.alpha,
            "scaler": self._scaler.to_dict(),
            "coef": None if self._coef is None else self._coef.tolist(),
            "intercept": self._intercept,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LogRidge:
        model = cls(alpha=float(payload.get("alpha", 1.0)))
        model._scaler = Standardizer.from_dict(payload["scaler"])
        coef = payload.get("coef")
        model._coef = np.asarray(coef, dtype=float) if coef is not None else None
        model._intercept = float(payload.get("intercept", 0.0))
        return model


@register_model
class LogisticRegression:
    """L2-regularised logistic regression, fit by full-batch gradient descent."""

    name = "logistic"
    kind: TaskKind = "classification"

    def __init__(self, l2: float = 1e-3, learning_rate: float = 0.5, epochs: int = 500) -> None:
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self._scaler = Standardizer()
        self._weights: np.ndarray | None = None
        self._bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        matrix = _as_matrix(X)
        target = _as_target(y)
        if matrix.shape[0] != target.shape[0]:
            raise ValueError("X and y disagree on the number of samples")
        target = np.clip(target, 0.0, 1.0)
        self._scaler.fit(matrix)
        scaled = self._scaler.transform(matrix)
        n, d = scaled.shape
        weights = np.zeros(d)
        bias = _logit(float(target.mean()))
        for _ in range(max(1, self.epochs)):
            error = _sigmoid(scaled @ weights + bias) - target
            weights -= self.learning_rate * (scaled.T @ error / n + self.l2 * weights)
            bias -= self.learning_rate * float(error.mean())
        self._weights = weights
        self._bias = float(bias)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class."""
        if self._weights is None:
            raise RuntimeError("LogisticRegression used before fit")
        scaled = self._scaler.transform(_as_matrix(X))
        return _sigmoid(scaled @ self._weights + self._bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.name,
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "scaler": self._scaler.to_dict(),
            "weights": None if self._weights is None else self._weights.tolist(),
            "bias": self._bias,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LogisticRegression:
        model = cls(
            l2=float(payload.get("l2", 1e-3)),
            learning_rate=float(payload.get("learning_rate", 0.5)),
            epochs=int(payload.get("epochs", 500)),
        )
        model._scaler = Standardizer.from_dict(payload["scaler"])
        weights = payload.get("weights")
        model._weights = np.asarray(weights, dtype=float) if weights is not None else None
        model._bias = float(payload.get("bias", 0.0))
        return model


@register_model
class MeanRegressor:
    """Baseline: always predict the training mean."""

    name = "mean"
    kind: TaskKind = "regression"

    def __init__(self) -> None:
        self._value = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> MeanRegressor:
        target = _as_target(y)
        self._value = float(target.mean()) if target.size else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(_as_matrix(X).shape[0], self._value)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.name, "value": self._value}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeanRegressor:
        model = cls()
        model._value = float(payload.get("value", 0.0))
        return model


@register_model
class BaseRateClassifier:
    """Baseline: always predict the training positive rate."""

    name = "base_rate"
    kind: TaskKind = "classification"

    def __init__(self) -> None:
        self._rate = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseRateClassifier:
        target = np.clip(_as_target(y), 0.0, 1.0)
        self._rate = float(target.mean()) if target.size else 0.5
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(_as_matrix(X).shape[0], self._rate)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.name, "rate": self._rate}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BaseRateClassifier:
        model = cls()
        model._rate = float(payload.get("rate", 0.5))
        return model


__all__ = [
    "BaseRateClassifier",
    "LogisticRegression",
    "MeanRegressor",
    "Model",
    "Ridge",
]
