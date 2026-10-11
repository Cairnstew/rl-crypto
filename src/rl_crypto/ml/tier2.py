"""Tier 2 — fuse the Tier-1 predictions into one :class:`Outlook`.

Where Tier 1 answers "what does each model expect?", Tier 2 answers "what is the
combined view?". For now the fusion is deliberately transparent and untrained —
a composite ``signal`` and ``confidence`` derived from the direction model,
falling back to the sign of the return model — so there is no hidden second
model to explain. A learned meta-model (stacking a logistic over the Tier-1
outputs) drops in later behind the same :class:`Outlook` shape.

Every field is ``None`` when the model behind it could not predict, so a missing
input propagates as an honest gap rather than a fabricated number.
"""

from __future__ import annotations

from ..features import FeatureRow
from .tier1 import Tier1Bundle
from .types import Outlook

#: Probability band outside which the direction model is called a signal.
DIRECTION_BULL = 0.55
DIRECTION_BEAR = 0.45


def _composite(
    points: dict[str, float | None],
    probabilities: dict[str, float | None],
) -> tuple[int | None, float | None]:
    """A composite ``(signal, confidence)`` from the direction/return models."""
    probability = probabilities.get("fwd_direction")
    if probability is not None:
        if probability >= DIRECTION_BULL:
            signal = 1
        elif probability <= DIRECTION_BEAR:
            signal = -1
        else:
            signal = 0
        return signal, abs(probability - 0.5) * 2.0

    expected_return = points.get("fwd_return")
    if expected_return is None:
        return None, None
    return (1 if expected_return > 0.0 else -1 if expected_return < 0.0 else 0), None


def compose_outlook(
    bundle: Tier1Bundle,
    *,
    ticker: str,
    feature_row: FeatureRow,
    generated_at: str | None = None,
) -> Outlook:
    """Run every Tier-1 model on ``feature_row`` and fuse the results."""
    points, probabilities = bundle.predict(feature_row.values)
    signal, confidence = _composite(points, probabilities)

    notes = tuple(
        f"{target}: prediction skipped (required feature missing)"
        for target, value in points.items()
        if value is None and target not in probabilities
    )
    return Outlook(
        ticker=ticker,
        generated_at=generated_at,
        horizon=bundle.horizon,
        feature_timestamp=feature_row.timestamp,
        predictions=points,
        probabilities=probabilities,
        signal=signal,
        confidence=confidence,
        features=dict(feature_row.values),
        notes=notes,
    )


__all__ = ["DIRECTION_BEAR", "DIRECTION_BULL", "compose_outlook"]
