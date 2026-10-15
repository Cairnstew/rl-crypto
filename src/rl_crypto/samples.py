"""Align Tier-0 features with Tier-1 targets into training/inference samples.

A :class:`Sample` pairs the causal feature vector for a row with that row's
forward targets. Features and targets are built by two independent passes over
the same rows, so a feature can never see the future a label uses, and a label
is ``None`` wherever its future is unavailable. The Tier-1 trainer decides which
``(features, target)`` pairs are usable; nothing is dropped here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .features import DEFAULT_INTERVAL_SECONDS, FEATURE_NAMES, dataset_interval, feature_row
from .models import Dataset, DatasetRow
from .targets import HORIZON_DEFAULT, compute_targets


@dataclass(slots=True)
class Sample:
    """One row's features and its forward targets."""

    timestamp: str
    features: dict[str, float | None]
    targets: dict[str, float | None]

    def features_complete(self, names: Sequence[str] = FEATURE_NAMES) -> bool:
        """True when every requested feature is present."""
        return all(self.features.get(name) is not None for name in names)

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-ready row: timestamp, features, then targets."""
        return {"timestamp": self.timestamp, **self.features, **self.targets}


def build_samples(
    rows: Sequence[DatasetRow],
    *,
    horizon: int = HORIZON_DEFAULT,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> tuple[Sample, ...]:
    """Pair every row's causal features with its forward targets."""
    return tuple(
        Sample(
            timestamp=rows[index].timestamp,
            features=feature_row(rows, index, interval_seconds=interval_seconds).values,
            targets=compute_targets(
                rows, index, horizon=horizon, interval_seconds=interval_seconds
            ),
        )
        for index in range(len(rows))
    )


def build_dataset_samples(
    dataset: Dataset,
    *,
    horizon: int = HORIZON_DEFAULT,
) -> tuple[Sample, ...]:
    """Samples for a whole :class:`~rl_crypto.models.Dataset`.

    The interval is taken from the dataset itself, so features and labels agree
    on the cadence without the caller restating it.
    """
    return build_samples(
        dataset.rows, horizon=horizon, interval_seconds=dataset_interval(dataset)
    )


__all__ = ["Sample", "build_dataset_samples", "build_samples"]
