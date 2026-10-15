"""Tier 2 — fuse the Tier-1 predictions into one :class:`Outlook`.

Where Tier 1 answers "what does each model expect?", Tier 2 answers "what is the
combined view?". For now the fusion is deliberately transparent and untrained —
a composite ``signal`` and ``confidence`` derived from the direction model,
falling back to the sign of the return model — so there is no hidden second
model to explain. A learned meta-model (stacking a logistic over the Tier-1
outputs) drops in later behind the same :class:`Outlook` shape.

A composite is only published when it is *earned*
-----------------------------------------------
The directional models on hourly crypto bars do not beat "always up" out of
sample, and a confident number derived from them is worse than no number: it
looks like information to every downstream consumer. So :func:`compose_outlook`
consults the bundle's holdout verdict (:meth:`Tier1Bundle.trusted`) and emits
``signal``/``confidence`` as ``None`` — with a note naming the reason — whenever
the backing model is unmeasured or failed its baseline. Point predictions stay
available either way; they are inputs for an RL observation vector, not claims.

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
    bundle: Tier1Bundle,
    points: dict[str, float | None],
    probabilities: dict[str, float | None],
) -> tuple[int | None, float | None, str | None]:
    """A composite ``(signal, confidence)``, plus a note when withheld.

    Returns ``(None, None, reason)`` rather than a number the models have not
    earned; ``reason`` is ``None`` when a signal is published.
    """
    if bundle.feature_version_mismatch:
        return None, None, "signal withheld: bundle feature version does not match this code"

    probability = probabilities.get("fwd_direction")
    if probability is not None:
        if not bundle.trusted("fwd_direction"):
            report = bundle.targets["fwd_direction"].holdout
            detail = (
                "did not beat the baseline out of sample"
                if report is not None and report.beats_baseline is False
                else "has no out-of-sample measurement"
            )
            return None, None, f"signal withheld: fwd_direction {detail}"
        if probability >= DIRECTION_BULL:
            signal = 1
        elif probability <= DIRECTION_BEAR:
            signal = -1
        else:
            signal = 0
        return signal, abs(probability - 0.5) * 2.0, None

    expected_return = points.get("fwd_return")
    if expected_return is None:
        return None, None, "signal withheld: no direction or return prediction available"
    if not bundle.trusted("fwd_return"):
        return None, None, "signal withheld: fwd_return did not beat the baseline out of sample"
    return (1 if expected_return > 0.0 else -1 if expected_return < 0.0 else 0), None, None


def compose_outlook(
    bundle: Tier1Bundle,
    *,
    ticker: str,
    feature_row: FeatureRow,
    generated_at: str | None = None,
) -> Outlook:
    """Run every Tier-1 model on ``feature_row`` and fuse the results."""
    points, probabilities = bundle.predict(feature_row.values)
    signal, confidence, withheld = _composite(bundle, points, probabilities)

    notes = [
        f"{target}: prediction skipped (required feature missing)"
        for target, value in points.items()
        if value is None and target not in probabilities
    ]
    if withheld:
        notes.append(withheld)
    if bundle.dropped_features:
        notes.append(
            f"{len(bundle.dropped_features)} feature(s) excluded as constant in training "
            f"and ignored here: {', '.join(bundle.dropped_features)}"
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
        notes=tuple(notes),
    )


__all__ = ["DIRECTION_BEAR", "DIRECTION_BULL", "compose_outlook"]
