"""Typed, JSON-ready results for the ML tiers.

Two shapes, both plain-data (no numpy, no datetimes):

* :class:`Prediction` — one target's outcome from one Tier-1 model.
* :class:`Outlook` — Tier 2's fusion of every Tier-1 prediction for a ticker at
  an instant, plus a composite ``signal``/``confidence`` and the feature row it
  was built from.

:meth:`Outlook.as_features` is the bridge to the live path: it flattens the
predictions into scalar columns that can be appended to a live row (the
"feature addition"), so downstream RL/analysis code consumes model output the
same way it consumes any other feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Schema tag for ML artifacts and prediction payloads.
ML_SCHEMA = "rl-crypto/ml/1"


@dataclass(slots=True)
class Prediction:
    """One Tier-1 model's output for one target."""

    target: str
    kind: str
    value: float | None
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "kind": self.kind,
            "value": self.value,
            "probability": self.probability,
        }


@dataclass(slots=True)
class Outlook:
    """Tier 2's combined view of every Tier-1 prediction for one ticker."""

    ticker: str
    generated_at: str | None
    horizon: int
    feature_timestamp: str | None = None
    predictions: dict[str, float | None] = field(default_factory=dict)
    probabilities: dict[str, float | None] = field(default_factory=dict)
    signal: int | None = None
    confidence: float | None = None
    features: dict[str, float | None] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    #: Which bundle produced this outlook. Recorded so a forward test can
    #: attribute realised skill to a specific deployed version instead of
    #: pooling every prediction the ticker ever made.
    bundle_trained_at: str | None = None
    feature_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ML_SCHEMA,
            "ticker": self.ticker,
            "generated_at": self.generated_at,
            "horizon": self.horizon,
            "feature_timestamp": self.feature_timestamp,
            "bundle_trained_at": self.bundle_trained_at,
            "feature_version": self.feature_version,
            "predictions": dict(self.predictions),
            "probabilities": dict(self.probabilities),
            "signal": self.signal,
            "confidence": self.confidence,
            "features": dict(self.features),
            "notes": list(self.notes),
        }

    def as_features(self, *, prefix: str = "pred_") -> dict[str, float | None]:
        """Flatten the outlook into scalar columns for a live row.

        ``pred_<target>`` for each point prediction, ``pred_prob_<target>`` for
        each classification probability, plus ``pred_signal`` and
        ``pred_confidence``. ``None`` is preserved — an unavailable prediction
        must not become a fabricated ``0``.
        """
        out: dict[str, float | None] = {
            f"{prefix}{target}": value for target, value in self.predictions.items()
        }
        out.update(
            {f"{prefix}prob_{target}": prob for target, prob in self.probabilities.items()}
        )
        out[f"{prefix}signal"] = None if self.signal is None else float(self.signal)
        out[f"{prefix}confidence"] = self.confidence
        return out


__all__ = ["ML_SCHEMA", "Outlook", "Prediction"]
