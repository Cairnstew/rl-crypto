"""Committed training configuration — the durable half of a promotion.

``models/<slug>.json`` is rebuilt by the scheduled train every two hours, so a
bundle written by ``promote --apply`` survives only until the next run. This
module reads the *source* the trainer uses: a committed ``config/training.json``
holding the feature set, per-target estimators and horizon, with per-ticker
overrides.

``promote --apply`` edits the file on the branch. That makes a promoted
candidate survive a retrain, because the retrain reads the same decision back
from git instead of from a default in code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..export import write_json
from ..features import FEATURE_NAMES

SCHEMA = "rl-crypto/training/1"


def config_path(root: str | Path = ".") -> Path:
    return Path(root) / "config" / "training.json"


@dataclass(frozen=True, slots=True)
class ResolvedTraining:
    """The settings one ticker's training run should use."""

    horizon: int | None
    feature_names: tuple[str, ...]
    model_names: dict[str, str]
    source: str


@dataclass(slots=True)
class TrainingConfig:
    """The committed training decisions, with per-ticker overrides."""

    horizon: int | None = 1
    feature_names: tuple[str, ...] = FEATURE_NAMES
    model_names: dict[str, str] = field(default_factory=dict)
    tickers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def for_ticker(self, ticker: str) -> ResolvedTraining:
        """Merge the ticker override over the repo defaults."""
        override = self.tickers.get(ticker) or {}
        raw_names = override.get("feature_names") or self.feature_names
        names = tuple(str(name) for name in raw_names) or FEATURE_NAMES
        models = dict(self.model_names)
        models.update(
            {str(k): str(v) for k, v in (override.get("model_names") or {}).items()}
        )
        horizon = override.get("horizon", self.horizon)
        return ResolvedTraining(
            horizon=int(horizon) if horizon is not None else None,
            feature_names=names,
            model_names=models,
            source=f"ticker:{ticker}" if override else "default",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            # An empty feature list means "the full FEATURE_NAMES set"; spelling
            # the list out would need editing on every feature-version bump.
            "feature_names": (
                list(self.feature_names) if self.feature_names != FEATURE_NAMES else []
            ),
            "horizon": self.horizon,
            "model_names": dict(self.model_names),
            "tickers": {ticker: dict(value) for ticker, value in sorted(self.tickers.items())},
        }


def default_config() -> TrainingConfig:
    """Production defaults when no committed config exists."""
    return TrainingConfig()


def parse_config(payload: Any) -> TrainingConfig:
    """Validate and build a :class:`TrainingConfig` from JSON data."""
    if not isinstance(payload, dict):
        raise ValueError("training config must be a JSON object")
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"unsupported training config schema {schema!r} (expected {SCHEMA!r})")

    raw_names = payload.get("feature_names") or []
    if not isinstance(raw_names, list) or not all(isinstance(name, str) for name in raw_names):
        raise ValueError("feature_names must be a list of strings")
    raw_models = payload.get("model_names", {})
    if raw_models is None:
        raw_models = {}
    if not isinstance(raw_models, dict):
        raise ValueError("model_names must be an object of target -> estimator")
    raw_tickers = payload.get("tickers", {})
    if raw_tickers is None:
        raw_tickers = {}
    if not isinstance(raw_tickers, dict) or not all(
        isinstance(value, dict) for value in raw_tickers.values()
    ):
        raise ValueError("tickers must be an object of ticker -> override object")
    horizon = payload.get("horizon", 1)
    if horizon is not None and (not isinstance(horizon, int) or horizon < 1):
        raise ValueError("horizon must be a positive integer or null")

    return TrainingConfig(
        horizon=horizon,
        feature_names=tuple(raw_names) or FEATURE_NAMES,
        model_names={str(k): str(v) for k, v in raw_models.items()},
        tickers={str(k): dict(v) for k, v in raw_tickers.items()},
    )


def load_training_config(root: str | Path = ".") -> TrainingConfig:
    """Read ``config/training.json``; a missing file means production defaults."""
    path = config_path(root)
    if not path.is_file():
        return default_config()
    return parse_config(json.loads(path.read_text(encoding="utf-8")))


def save_training_config(root: str | Path, config: TrainingConfig) -> Path:
    """Write the config as pretty, diffable JSON."""
    return write_json(config_path(root), config.to_dict())


def apply_promotion(
    root: str | Path,
    *,
    ticker: str,
    feature_names: tuple[str, ...] | list[str],
    model_names: dict[str, str],
    horizon: int,
) -> Path:
    """Record a promoted candidate as a per-ticker override, durably.

    Only the promoted ticker changes; every other ticker keeps the repo
    defaults, so one promotion cannot silently retune the rest of the watchlist.
    """
    config = load_training_config(root)
    config.tickers[ticker] = {
        "feature_names": [str(name) for name in feature_names],
        "model_names": {str(k): str(v) for k, v in sorted(model_names.items())},
        "horizon": int(horizon),
    }
    return save_training_config(root, config)


__all__ = [
    "SCHEMA",
    "ResolvedTraining",
    "TrainingConfig",
    "apply_promotion",
    "config_path",
    "default_config",
    "load_training_config",
    "parse_config",
    "save_training_config",
]
