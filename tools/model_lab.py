#!/usr/bin/env python3
"""rl-model-lab — an offline, out-of-sample model improvement loop.

Actions:

* ``status``   — what is deployed, what is trusted, and what is stale?
* ``evaluate`` — does the model beat a constant baseline *and* a simple
  momentum policy across walk-forward folds, after fees and slippage?
* ``sweep``    — do controlled feature/model/horizon variations improve the
  out-of-sample result consistently, or is the gain noise?
* ``promote``  — validate a candidate on a final holdout (dry run by default).

It never guarantees profit. It is designed to stop an overfit model from being
mistaken for a real edge: every score is out of sample, fold stability is
reported, costs are charged, and the final holdout is reserved for promotion.

Artifacts live outside the repo by default (``~/.local/share/rl-crypto/model-lab``)
so experiments never pollute ``data/`` or ``models/``. Override with
``RL_MODEL_LAB_DIR``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rl_crypto.backtest import pnl_metrics
from rl_crypto.cache import candles_path, load_candles, slug_for, summary_path
from rl_crypto.features import (
    DEFAULT_INTERVAL_SECONDS,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_VERSION,
)
from rl_crypto.ml.store import default_bundle_path, load_bundle, save_bundle
from rl_crypto.ml.tier1 import (
    Tier1Bundle,
    _regression_metrics_in_space,
    baseline_for,
    beats_baseline,
    classification_metrics,
    train_tier1,
)
from rl_crypto.ml.training_config import apply_promotion
from rl_crypto.models import to_iso, utcnow
from rl_crypto.samples import Sample, build_samples
from rl_crypto.targets import TARGET_KINDS, TARGET_NAMES, label_purge

ROOT = Path(os.environ.get("RL_CRYPTO_ROOT") or Path.cwd()).resolve()
LAB_DIR = Path(
    os.environ.get("RL_MODEL_LAB_DIR")
    or Path.home() / ".local" / "share" / "rl-crypto" / "model-lab"
)
INTERVAL_SECONDS = DEFAULT_INTERVAL_SECONDS
BARS_PER_YEAR = 365.0 * 24.0 * 3600.0 / INTERVAL_SECONDS

#: Minimum time blocks for a percentile CI. The effective sample size of a
#: block bootstrap is the number of blocks; eight is too few for a 2.5% tail.
#: Twenty-four is six blocks per fold at the default four folds.
MIN_BOOTSTRAP_BLOCKS = 24

#: A percentile bootstrap resolves quantiles at ``1 / B``; a two-sided tail of
#: ``alpha / 2`` needs at least two order statistics in that tail.
RESOLUTION_ORDER_STATS = 2

MODEL_SETS: dict[str, dict[str, str]] = {
    "reference": {
        "fwd_return": "ridge",
        "fwd_direction": "logistic",
        # fwd_volatility is non-negative and production fits it in log space;
        # evaluating a raw ridge here would score a different model than ships.
        "fwd_volatility": "log_ridge",
        "fwd_volume_ratio": "ridge",
    },
    "baseline": {
        "fwd_return": "mean",
        "fwd_direction": "base_rate",
        "fwd_volatility": "mean",
        "fwd_volume_ratio": "mean",
    },
}
#: Alias kept because the lab's job is to evaluate what production ships.
MODEL_SETS["production"] = MODEL_SETS["reference"]

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "all": FEATURE_NAMES,
    "market": tuple(n for n in FEATURE_NAMES if n not in FEATURE_GROUPS["news"]),
    "core": tuple(
        n
        for group in ("return", "volatility", "trend", "volume", "calendar")
        for n in FEATURE_GROUPS[group]
    ),
    "price": tuple(
        n for group in ("return", "volatility", "bar", "trend") for n in FEATURE_GROUPS[group]
    ),
}


@dataclass(frozen=True, slots=True)
class Config:
    """One experiment: feature set, model set, horizon."""

    name: str
    feature_names: tuple[str, ...]
    model_names: dict[str, str]
    horizon: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "features": len(self.feature_names),
            "horizon": self.horizon,
            "models": dict(self.model_names),
        }


# --------------------------------------------------------------------------- helpers


def resolve_tickers(values: Sequence[str] | None) -> tuple[str, ...]:
    out: list[str] = []
    for value in values or ():
        out.extend(part for part in str(value).replace(",", " ").split() if part.strip())
    if out:
        return tuple(dict.fromkeys(out))
    dirs = [
        p.name for p in (ROOT / "data").iterdir() if p.is_dir() and (p / "candles.jsonl").exists()
    ]
    return tuple(ticker_for_slug(d) for d in sorted(dirs))


def ticker_for_slug(slug: str) -> str:
    return slug.replace("-", "/").upper()


def load_ticker_rows(ticker: str) -> tuple[Any, ...]:
    return load_candles(candles_path(ROOT, ticker), ticker=ticker)


def resolve_targets(values: Sequence[str] | None) -> tuple[str, ...]:
    out: list[str] = []
    for value in values or ():
        out.extend(part for part in str(value).replace(",", " ").split() if part.strip())
    unknown = [t for t in out if t not in TARGET_KINDS]
    if unknown:
        raise SystemExit(f"unknown target(s): {', '.join(unknown)}; choose {TARGET_NAMES}")
    return tuple(dict.fromkeys(out)) or TARGET_NAMES


def resolve_feature_names(spec: str) -> tuple[str, ...]:
    spec = (spec or "all").strip()
    if spec in FEATURE_SETS:
        return FEATURE_SETS[spec]
    groups = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise SystemExit(
            f"unknown feature group(s): {', '.join(unknown)}; "
            f"choose {', '.join(FEATURE_GROUPS)} or {', '.join(FEATURE_SETS)}"
        )
    return tuple(n for g in groups for n in FEATURE_GROUPS[g])


def resolve_model_set(spec: str) -> dict[str, str]:
    spec = (spec or "reference").strip()
    if spec not in MODEL_SETS:
        raise SystemExit(f"unknown model set {spec!r}; choose {', '.join(MODEL_SETS)}")
    return dict(MODEL_SETS[spec])


def sample_span(samples: Sequence[Sample]) -> tuple[str, str]:
    return (samples[0].timestamp, samples[-1].timestamp) if samples else ("", "")


def walk_forward(
    samples: Sequence[Sample],
    *,
    folds: int = 4,
    horizon: int = 1,
    targets: Sequence[str] = (),
    purge: int | None = None,
    min_train_fraction: float = 0.34,
    min_test: int = 20,
) -> list[tuple[tuple[Sample, ...], tuple[Sample, ...]]]:
    """Expanding-window walk-forward splits with a label-reach purge.

    A label anchored at row ``i`` reads up to ``i + horizon``; ``fwd_volatility``
    reads ``max(horizon, MIN_VOL_SPAN)``. The purge is the largest reach among
    ``targets`` (from the shared :func:`rl_crypto.targets.label_purge`), so the
    last training label never reads a test bar. Passing a flat ``horizon`` here
    leaks one bar for ``fwd_volatility`` at the default horizon, which is why
    the shared helper is the only definition of a label's reach.
    """
    n = len(samples)
    if n < min_test * 2:
        return []
    reach = max((label_purge(target, horizon) for target in targets), default=max(0, horizon))
    effective_purge = max(0, purge if purge is not None else reach)
    min_train = max(min_test, int(n * min_train_fraction))
    available = n - min_train
    folds = max(1, min(folds, available // min_test))
    test_size = max(min_test, available // folds)
    split: list[tuple[tuple[Sample, ...], tuple[Sample, ...]]] = []
    for i in range(folds):
        train_end = min_train + i * test_size
        test_end = min(n, train_end + test_size)
        train = tuple(samples[: max(0, train_end - effective_purge)])
        test = tuple(samples[train_end:test_end])
        if len(train) >= min_test and test:
            split.append((train, test))
    return split


def purge_required(targets: Sequence[str], horizon: int) -> int:
    """Rows that must sit between the last training label and the first test row."""
    return max((label_purge(target, horizon) for target in targets), default=max(0, horizon))


def validate_splits(
    samples: Sequence[Sample],
    splits: Sequence[tuple[tuple[Sample, ...], tuple[Sample, ...]]],
    *,
    targets: Sequence[str],
    horizon: int,
    purge: int | None = None,
) -> dict[str, Any]:
    """Mechanically check that no training label reaches into its test fold.

    This is what turns the purge from a convention into a checkable layer. A
    canary that deliberately skips the purge is reported as a violation, and
    the v2 verdict refuses it instead of scoring the leak.
    """
    index = {sample.timestamp: position for position, sample in enumerate(samples)}
    required = max(0, int(purge if purge is not None else purge_required(targets, horizon)))
    violations: list[dict[str, Any]] = []
    for number, (train, test) in enumerate(splits):
        if not train or not test:
            violations.append({"fold": number, "reason": "empty train or test"})
            continue
        test_start = index.get(test[0].timestamp)
        train_last = index.get(train[-1].timestamp)
        if test_start is None or train_last is None:
            violations.append({"fold": number, "reason": "unknown timestamp"})
            continue
        train_end = train_last + 1
        gap = test_start - train_end
        if gap < required:
            violations.append(
                {
                    "fold": number,
                    "reason": "label reach overlaps the test fold",
                    "gap": gap,
                    "required": required,
                }
            )
        if any(
            (position := index.get(sample.timestamp)) is not None
            and position >= test_start
            for sample in train
        ):
            violations.append({"fold": number, "reason": "train sample at/after test start"})
    return {
        "ok": not violations,
        "required_purge": required,
        "violations": violations,
    }


def metric_key(kind: str) -> str:
    return "rmse" if kind == "regression" else "log_loss"


def relative_improvement(
    kind: str, metrics: dict[str, Any], baseline: dict[str, Any]
) -> float | None:
    key = metric_key(kind)
    got, bar = metrics.get(key), baseline.get(key)
    if not isinstance(got, int | float) or not isinstance(bar, int | float) or bar == 0:
        return None
    return float((bar - got) / abs(bar))


def target_labels(samples: Sequence[Sample], target: str) -> np.ndarray:
    return np.asarray(
        [s.targets[target] for s in samples if s.targets.get(target) is not None],
        dtype=float,
    )


def evaluate_target(
    bundle: Tier1Bundle,
    target: str,
    train: Sequence[Sample],
    test: Sequence[Sample],
) -> dict[str, Any] | None:
    model = bundle.targets.get(target)
    if model is None:
        return None
    kind = TARGET_KINDS[target]
    rows = [(s, model.predict_row(s.features)) for s in test if s.targets.get(target) is not None]
    rows = [(s, p) for s, p in rows if p is not None]
    if not rows:
        return None
    y_true = np.asarray([s.targets[target] for s, _ in rows], dtype=float)
    y_pred = np.asarray([p for _, p in rows], dtype=float)
    # Score on the scale the estimator actually minimises, so a log-space
    # volatility model is compared with a log-space constant (the same rule the
    # trainer and forward scorer use). The metadata on the fitted model is the
    # single source of truth for that scale.
    space = str(getattr(model, "metric_space", "raw"))
    metrics = (
        _regression_metrics_in_space(y_true, y_pred, space)
        if kind == "regression"
        else classification_metrics(y_true, y_pred)
    )
    baseline = baseline_for(kind, target_labels(train, target), y_true, space=space)
    key = metric_key(kind)
    return {
        "target": target,
        "kind": kind,
        "n": len(rows),
        "metric": key,
        "model": metrics.get(key),
        "baseline": baseline.get(key),
        "improvement": relative_improvement(kind, metrics, baseline),
        "beats": beats_baseline(kind, metrics, baseline),
    }


def _policy_components(
    bundle: Tier1Bundle,
    test: Sequence[Sample],
    *,
    position: str,
) -> tuple[list[float | None], list[float | None], list[float], list[float], list[float]]:
    """Predictions and realised returns, computed once per fold.

    The direction probability and return prediction are evaluated once here and
    reused for every threshold in the cost grid, so a grid never refits or
    re-predicts per cell. A missing prediction or target is skipped, never
    imputed.
    """
    direction = bundle.targets.get("fwd_direction")
    ret = bundle.targets.get("fwd_return")
    probabilities: list[float | None] = []
    expected: list[float | None] = []
    momentum: list[float] = []
    always: list[float] = []
    realised: list[float] = []
    for sample in test:
        target = sample.targets.get("fwd_return")
        if target is None:
            continue
        probabilities.append(
            direction.predict_row(sample.features) if direction is not None else None
        )
        expected.append(ret.predict_row(sample.features) if ret is not None else None)
        ret1 = sample.features.get("ret_1")
        momentum.append(
            (1.0 if ret1 is not None and ret1 > 0 else -1.0)
            if position == "long-short"
            else (1.0 if ret1 is not None and ret1 > 0 else 0.0)
        )
        always.append(1.0)
        realised.append(float(target))
    return probabilities, expected, momentum, always, realised


def _positions_for(
    probabilities: Sequence[float | None],
    expected: Sequence[float | None],
    *,
    threshold: float,
    position: str,
) -> list[float]:
    """Turn a direction probability (or return sign) into a position."""
    out: list[float] = []
    for probability, prediction in zip(probabilities, expected, strict=True):
        held = 0.0
        if probability is not None:
            if probability >= threshold:
                held = 1.0
            elif position == "long-short" and probability <= 1.0 - threshold:
                held = -1.0
        elif prediction is not None:
            held = 1.0 if prediction > 0 else (-1.0 if position == "long-short" else 0.0)
        out.append(held)
    return out


def _policy_rows(
    bundle: Tier1Bundle,
    test: Sequence[Sample],
    *,
    threshold: float,
    position: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Model/momentum/always-long positions plus realised returns (one threshold)."""
    probabilities, expected, momentum, always, realised = _policy_components(
        bundle, test, position=position
    )
    model = _positions_for(probabilities, expected, threshold=threshold, position=position)
    return model, momentum, always, realised


def _pnl_metrics(
    positions: Sequence[float],
    returns: Sequence[float],
    *,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    """Per-side-cost economics, from the one shared backtest implementation."""
    return pnl_metrics(
        positions,
        returns,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        bars_per_year=BARS_PER_YEAR,
    )


def aggregate_target(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    improvements = [r["improvement"] for r in records if r["improvement"] is not None]
    model_values = [r["model"] for r in records if r["model"] is not None]
    baseline_values = [r["baseline"] for r in records if r["baseline"] is not None]
    beats = [bool(r["beats"]) for r in records]
    return {
        "target": records[0]["target"],
        "kind": records[0]["kind"],
        "metric": records[0]["metric"],
        "folds": len(records),
        "n": sum(int(r["n"]) for r in records),
        "model_mean": float(np.mean(model_values)) if model_values else None,
        "baseline_mean": float(np.mean(baseline_values)) if baseline_values else None,
        "mean_improvement": float(np.mean(improvements)) if improvements else None,
        "min_improvement": float(np.min(improvements)) if improvements else None,
        "folds_beating": sum(beats),
        "beats_ratio": sum(beats) / len(beats),
    }


# --------------------------------------------------------------------------- holdout / paired


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


def filter_rows_before(
    rows: Sequence[Any], holdout_start: str | None
) -> tuple[tuple[Any, ...], datetime | None]:
    """Keep rows strictly before the frozen cutoff (never a fraction)."""
    cutoff = parse_timestamp(holdout_start)
    if cutoff is None:
        return tuple(rows), None
    kept = tuple(
        row
        for row in rows
        if (stamp := parse_timestamp(row.timestamp)) is not None and stamp < cutoff
    )
    return kept, cutoff


def sample_split(samples: Sequence[Sample], holdout_start: str | None) -> int | None:
    """Index of the first sample at or after the cutoff (the holdout starts there)."""
    cutoff = parse_timestamp(holdout_start)
    if cutoff is None:
        return None
    for index, sample in enumerate(samples):
        stamp = parse_timestamp(sample.timestamp)
        if stamp is None or stamp >= cutoff:
            return index
    return len(samples)


def _baseline_probability(labels: Sequence[float | None]) -> float:
    values = [float(v) for v in labels if v is not None]
    rate = float(np.mean(values)) if values else 0.5
    # Match baseline_for: the better of always-up / always-down.
    return max(rate, 1.0 - rate)


def paired_target_rows(
    bundle: Tier1Bundle,
    target: str,
    train: Sequence[Sample],
    test: Sequence[Sample],
) -> list[dict[str, Any]]:
    """Per-row paired loss improvement: positive means the model beat the constant."""
    model = bundle.targets.get(target)
    if model is None:
        return []
    kind = TARGET_KINDS[target]
    space = str(getattr(model, "metric_space", "raw"))
    train_labels = [sample.targets.get(target) for sample in train]
    if kind == "regression" and space == "log":
        # The estimator minimises squared error on log(y); the constant must be
        # fitted on the same scale, and non-positive rows carry no log.
        positive = [float(v) for v in train_labels if v is not None and float(v) > 0.0]
        baseline = float(np.mean(np.log(positive))) if positive else 0.0
    else:
        baseline = (
            float(np.mean([float(v) for v in train_labels if v is not None]))
            if kind == "regression"
            else _baseline_probability(train_labels)
        )
    if kind == "regression" and not any(v is not None for v in train_labels):
        baseline = 0.0
    out: list[dict[str, Any]] = []
    for sample in test:
        truth = sample.targets.get(target)
        if truth is None:
            continue
        prediction = model.predict_row(sample.features)
        if prediction is None:
            continue
        if kind == "regression" and space == "log":
            if float(truth) <= 0.0 or float(prediction) <= 0.0:
                continue
            model_loss = (math.log(float(truth)) - math.log(float(prediction))) ** 2
            baseline_loss = (math.log(float(truth)) - baseline) ** 2
        elif kind == "regression":
            model_loss = (float(truth) - float(prediction)) ** 2
            baseline_loss = (float(truth) - baseline) ** 2
        else:
            probability = min(max(float(prediction), 1e-12), 1.0 - 1e-12)
            constant = min(max(baseline, 1e-12), 1.0 - 1e-12)
            model_loss = -(
                float(truth) * math.log(probability)
                + (1.0 - float(truth)) * math.log(1.0 - probability)
            )
            baseline_loss = -(
                float(truth) * math.log(constant)
                + (1.0 - float(truth)) * math.log(1.0 - constant)
            )
        out.append(
            {
                "timestamp": sample.timestamp,
                "improvement": float(baseline_loss - model_loss),
            }
        )
    return out


def bootstrap_pooled(
    rows_by_ticker: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    alpha: float,
    seed: int,
    block_bars: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Time-block bootstrap over pooled paired improvements.

    Blocks never span tickers, so the correlation between BTC/ETH/SOL is not
    laundered into extra independent observations. The point estimate is the
    mean over every pooled row; the interval comes from resampling whole
    consecutive blocks with replacement.
    """
    blocks: list[list[float]] = []
    per_ticker: dict[str, Any] = {}
    for ticker, rows in rows_by_ticker.items():
        values = [float(row["improvement"]) for row in rows]
        if values:
            per_ticker[str(ticker)] = {
                "n": len(values),
                "mean_improvement": float(np.mean(values)),
            }
        width = max(1, int(block_bars))
        for start in range(0, len(values), width):
            block = values[start : start + width]
            if block:
                blocks.append(block)
    pooled_values = [value for block in blocks for value in block]
    result: dict[str, Any] = {
        "n": len(pooled_values),
        "blocks": len(blocks),
        "block_bars": int(block_bars),
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        "alpha": float(alpha),
        "mean_improvement": float(np.mean(pooled_values)) if pooled_values else None,
        "ci_low": None,
        "ci_high": None,
        "per_ticker": per_ticker,
    }
    # Resolution: can B draws place the required order statistics in both
    # tails? Below that the boundary is interpolated noise, not a quantile.
    tail = max(float(alpha), 1e-12) / 2.0
    required_samples = max(
        1000, math.ceil(RESOLUTION_ORDER_STATS / tail)
    )
    result["tail"] = tail
    result["required_samples"] = required_samples
    result["resolution_ok"] = (
        int(bootstrap_samples) >= required_samples
        and len(blocks) >= MIN_BOOTSTRAP_BLOCKS
        and len(pooled_values) >= 60
    )
    result["min_blocks"] = MIN_BOOTSTRAP_BLOCKS
    # Too few blocks makes the percentile interval anti-conservative; refuse to
    # manufacture a CI from a handful of blocks.
    if len(blocks) < MIN_BOOTSTRAP_BLOCKS or len(pooled_values) < 60:
        result["insufficient"] = True
        result["resolution_reason"] = (
            f"only {len(blocks)} time blocks (need {MIN_BOOTSTRAP_BLOCKS})"
        )
        return result
    if int(bootstrap_samples) < required_samples:
        result["insufficient"] = True
        result["resolution_reason"] = (
            f"{bootstrap_samples} resamples cannot resolve alpha={alpha:g} "
            f"(need {required_samples})"
        )
        return result
    # Resample block means and counts, not the flattened rows: the statistic is
    # identical and a 10k-draw CI over hundreds of blocks stays cheap.
    sums = np.asarray([float(np.sum(block)) for block in blocks], dtype=float)
    counts = np.asarray([len(block) for block in blocks], dtype=float)
    rng = np.random.default_rng(int(seed))
    picks = rng.integers(0, len(blocks), size=(int(bootstrap_samples), len(blocks)))
    drawn_sums = sums[picks].sum(axis=1)
    drawn_counts = counts[picks].sum(axis=1)
    means = drawn_sums / drawn_counts
    result["ci_low"] = float(np.quantile(means, alpha / 2.0))
    result["ci_high"] = float(np.quantile(means, 1.0 - alpha / 2.0))
    return result


def pool_fold_improvements(
    folds_by_ticker: Mapping[str, Sequence[float | None]], folds: int
) -> list[float | None]:
    out: list[float | None] = []
    for index in range(folds):
        values = [
            float(series[index])
            for series in folds_by_ticker.values()
            if len(series) > index and series[index] is not None
        ]
        out.append(float(np.mean(values)) if values else None)
    return out


def transform_targets(
    samples: Sequence[Sample],
    target: str,
    mode: str,
    *,
    seed: int,
    shift_bars: int,
) -> tuple[Sample, ...]:
    """Destroy feature/label alignment without reading any new bar.

    ``shuffle`` permutes the observed labels; ``shift`` moves labels backward
    by ``shift_bars`` (a past label can never contain future information).
    """
    values = [sample.targets.get(target) for sample in samples]
    if mode == "shuffle":
        positions = [index for index, value in enumerate(values) if value is not None]
        permuted = list(values)
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(positions))
        for destination, source in zip(positions, order, strict=True):
            permuted[destination] = values[positions[int(source)]]
        new_values = permuted
    elif mode == "shift":
        width = max(1, int(shift_bars))
        new_values = [None] * len(values)
        for index in range(width, len(values)):
            new_values[index] = values[index - width]
    else:  # pragma: no cover - guarded by the CLI choices
        raise SystemExit(f"unknown control mode {mode!r}")
    return tuple(
        Sample(
            timestamp=sample.timestamp,
            features=sample.features,
            targets={**sample.targets, target: new_values[index]},
        )
        for index, sample in enumerate(samples)
    )


#: Feature name used by the power calibration to plant a learnable signal.
PLANTED_FEATURE = "planted_signal"


def plant_signal(
    samples: Sequence[Sample],
    target: str,
    *,
    strength: float,
    seed: int,
    space: str = "raw",
    feature: str = PLANTED_FEATURE,
) -> tuple[tuple[Sample, ...], dict[str, Any]]:
    """Plant a known linear signal into ``target`` and expose it as a feature.

    The synthetic feature is an independent standard-normal draw, so the model
    can learn it only from the planted component; the original target remains
    as noise. The *oracle* paired improvement is ``strength**2 * var(z)`` in the
    metric space of the target (raw squared error, or log-space squared error
    for a ``log_ridge`` volatility model). ``space="log"`` plants
    multiplicatively so the target stays positive:
    ``log(y') = log(y) + strength * z``.
    """
    width = len(samples)
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(width)
    planted: list[Sample] = []
    for index, sample in enumerate(samples):
        value = sample.targets.get(target)
        new_targets: dict[str, float | None] = dict(sample.targets)
        if value is not None and float(value) > 0.0:
            if space == "log":
                new_targets[target] = float(value) * math.exp(
                    float(strength) * float(noise[index])
                )
            else:
                new_targets[target] = float(value) + float(strength) * float(noise[index])
        elif value is not None and space != "log":
            new_targets[target] = float(value) + float(strength) * float(noise[index])
        planted.append(
            Sample(
                timestamp=sample.timestamp,
                features={**sample.features, feature: float(noise[index])},
                targets=new_targets,
            )
        )
    effect = float(strength) ** 2 * float(np.var(noise))
    return tuple(planted), {
        "feature": feature,
        "strength": float(strength),
        "space": space,
        "effect": effect,
        "n": width,
    }


LEAK_CORRELATION_FEATURE = "leak_correlation"


def inject_correlated_feature(
    samples: Sequence[Sample],
    target: str,
    correlation: float,
    *,
    seed: int,
    feature: str = LEAK_CORRELATION_FEATURE,
) -> tuple[tuple[Sample, ...], dict[str, Any]]:
    """Add a feature with a graded, finite-sample correlation to ``target``.

    The deterministic positive canary uses ``|r| = 1``. Graded leaks matter
    because a real implementation bug (a normalised future window, a centred
    target) produces a strong but not perfect correlation, which the strict
    ``detect_leaks`` threshold cannot see. This helper makes that scenario
    reproducible so the calibration can report which layer, if any, catches it.
    """
    values = [sample.targets.get(target) for sample in samples]
    usable = [index for index, value in enumerate(values) if value is not None]
    usable_values = {index: float(value) for index, value in enumerate(values) if value is not None}
    rng = np.random.default_rng(int(seed))
    rho = max(-0.999, min(0.999, float(correlation)))
    signal = np.zeros(len(samples), dtype=float)
    noise = rng.standard_normal(len(samples))
    if usable:
        observed = np.asarray([usable_values[index] for index in usable], dtype=float)
        spread = float(np.std(observed))
        standardised = (
            (observed - float(np.mean(observed))) / spread
            if spread > 0.0
            else np.zeros_like(observed)
        )
        # Orthogonalise the noise against the standardised target so the sample
        # correlation is the requested value, not a finite-sample draw around it.
        residual = noise[usable] - float(np.mean(noise[usable]))
        projection = (
            float(np.dot(residual, standardised))
            / float(np.dot(standardised, standardised))
            if float(np.dot(standardised, standardised)) > 0.0
            else 0.0
        )
        residual = residual - projection * standardised
        residual_spread = float(np.std(residual))
        if residual_spread > 0.0:
            residual = residual / residual_spread
        signal[usable] = (
            rho * standardised
            + math.sqrt(max(0.0, 1.0 - rho * rho)) * residual
        )
    planted = tuple(
        Sample(
            timestamp=sample.timestamp,
            features={**sample.features, feature: float(signal[index])},
            targets=dict(sample.targets),
        )
        for index, sample in enumerate(samples)
    )
    actual = None
    if len(usable) >= 2:
        xs = np.asarray([float(planted[i].features[feature]) for i in usable])
        ys = np.asarray([usable_values[i] for i in usable])
        if float(np.std(xs)) > 0.0 and float(np.std(ys)) > 0.0:
            actual = float(np.corrcoef(xs, ys)[0, 1])
    return planted, {
        "feature": feature,
        "requested_correlation": rho,
        "actual_correlation": actual,
        "n": len(usable),
    }


def inject_leaky_feature(
    samples: Sequence[Sample], feature: str, target: str = "fwd_return"
) -> tuple[Sample, ...]:
    """Add a feature that is a deterministic function of the realised label.

    This exists only for the positive canary: it makes the label readable at
    prediction time. A harness that cannot see that as leakage is void.
    """
    return tuple(
        Sample(
            timestamp=sample.timestamp,
            features={**sample.features, feature: sample.targets.get(target)},
            targets=dict(sample.targets),
        )
        for sample in samples
    )


def detect_leaks(
    samples: Sequence[Sample], targets: Sequence[str]
) -> list[dict[str, Any]]:
    """Flag a feature that is (nearly) a deterministic function of a target.

    Pearson |r| >= 0.999 over at least 50 complete rows is not a subtle
    overfit: a feature that moves one-for-one with the label it predicts was
    computed from the future. The check is deliberately mechanical so the
    positive canary cannot depend on an LLM noticing it.
    """
    leaks: list[dict[str, Any]] = []
    if not samples:
        return leaks
    feature_names = sorted({name for sample in samples for name in sample.features})
    for target in targets:
        truth = [sample.targets.get(target) for sample in samples]
        pairs = [(index, value) for index, value in enumerate(truth) if value is not None]
        if len(pairs) < 50:
            continue
        for name in feature_names:
            xs: list[float] = []
            ys: list[float] = []
            for index, value in pairs:
                feature_value = samples[index].features.get(name)
                if isinstance(feature_value, int | float):
                    xs.append(float(feature_value))
                    ys.append(float(value))
            if len(xs) < 50 or len(set(xs)) < 2 or len(set(ys)) < 2:
                continue
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
            correlation = float(np.corrcoef(x, y)[0, 1])
            if abs(correlation) >= 0.999:
                leaks.append(
                    {
                        "feature": name,
                        "target": target,
                        "correlation": correlation,
                        "n": len(xs),
                    }
                )
    return leaks


def evaluate_control_ticker(
    ticker: str,
    config: Config,
    target: str,
    mode: str | None,
    *,
    folds: int,
    holdout_start: str | None,
    seed: int,
    shift_bars: int,
    leak_feature: str | None = None,
    plant_strength: float = 0.0,
    plant_space: str = "raw",
    plant_feature: str | None = None,
    inject_correlation: float | None = None,
    inject_feature: str | None = None,
) -> dict[str, Any]:
    """One walk-forward pass with an optional planted signal and control.

    ``mode=None`` is the planted candidate; ``"shuffle"``/``"shift"`` transform
    the (possibly planted) labels so the same signal can no longer be read.
    """
    rows, _ = filter_rows_before(load_ticker_rows(ticker), holdout_start)
    samples = build_samples(rows, horizon=config.horizon, interval_seconds=INTERVAL_SECONDS)
    planted: dict[str, Any] | None = None
    if plant_strength:
        samples, planted = plant_signal(
            samples,
            target,
            strength=plant_strength,
            seed=seed,
            space=plant_space,
            feature=plant_feature or PLANTED_FEATURE,
        )
    if inject_correlation is not None:
        samples, _ = inject_correlated_feature(
            samples,
            target,
            inject_correlation,
            seed=seed,
            feature=inject_feature or LEAK_CORRELATION_FEATURE,
        )
    if leak_feature:
        samples = inject_leaky_feature(samples, leak_feature)
    if mode is not None:
        samples = transform_targets(
            samples, target, mode, seed=seed, shift_bars=shift_bars
        )
    splits = walk_forward(samples, folds=folds, horizon=config.horizon, targets=[target])
    paired: list[dict[str, Any]] = []
    fold_improvements: list[float | None] = []
    for train, test in splits:
        bundle = train_tier1(
            train,
            horizon=config.horizon,
            feature_names=config.feature_names,
            targets=[target],
            model_names=config.model_names,
            holdout_fraction=0.0,
        )
        rows_out = paired_target_rows(bundle, target, train, test)
        paired.extend(rows_out)
        values = [float(row["improvement"]) for row in rows_out]
        fold_improvements.append(float(np.mean(values)) if values else None)
    return {
        "ticker": ticker,
        "paired_rows": paired,
        "fold_improvements": fold_improvements,
        "planted": planted,
        "purge": validate_splits(
            samples, splits, targets=[target], horizon=config.horizon
        ),
    }


def evaluate_ticker(
    ticker: str,
    config: Config,
    targets: Sequence[str],
    *,
    folds: int,
    threshold: float,
    position: str,
    fee_bps: float,
    slippage_bps: float,
    thresholds: Sequence[float] | None = None,
    cost_levels: Sequence[tuple[float, float]] | None = None,
    reserve_final: float = 0.0,
    holdout_start: str | None = None,
    full_archive: bool = False,
    primary_target: str | None = None,
    leak_feature: str | None = None,
) -> dict[str, Any]:
    """Walk-forward metrics plus a threshold x cost policy grid.

    Each fold is fitted exactly once; the direction probability is computed once
    per fold and re-thresholded for every grid cell. The headline ``economic``
    block is the cell matching ``threshold``/``fee_bps``/``slippage_bps``.
    """
    rows = load_ticker_rows(ticker)
    if len(rows) < 120:
        return {"ticker": ticker, "error": f"only {len(rows)} candles"}
    if holdout_start and not full_archive:
        rows, _ = filter_rows_before(rows, holdout_start)
    threshold_list = _unique(list(thresholds or [threshold]))
    cost_list = _unique(list(cost_levels or [(fee_bps, slippage_bps)]))
    samples = build_samples(rows, horizon=config.horizon, interval_seconds=INTERVAL_SECONDS)
    if leak_feature:
        samples = inject_leaky_feature(samples, leak_feature)
    # Check against every target, not only the requested ones: a feature
    # computed from any realised outcome is a leak regardless of what is scored.
    leaks = detect_leaks(samples, TARGET_NAMES)
    if reserve_final > 0.0 and not holdout_start:
        keep = int(len(samples) * (1.0 - reserve_final))
        samples = samples[:keep]
    splits = walk_forward(samples, folds=folds, horizon=config.horizon, targets=targets)
    if not splits:
        return {"ticker": ticker, "error": "not enough samples for walk-forward"}
    target_records: dict[str, list[dict[str, Any]]] = {t: [] for t in targets}
    model_pos: dict[float, list[float]] = {value: [] for value in threshold_list}
    mom_pos: list[float] = []
    always_pos: list[float] = []
    returns: list[float] = []
    paired: list[dict[str, Any]] = []
    fold_improvements: list[float | None] = []
    paired_target = primary_target or (targets[0] if targets else None)
    for train, test in splits:
        bundle = train_tier1(
            train,
            horizon=config.horizon,
            feature_names=config.feature_names,
            targets=targets,
            model_names=config.model_names,
            holdout_fraction=0.0,
        )
        for target in targets:
            record = evaluate_target(bundle, target, train, test)
            if record is not None:
                target_records[target].append(record)
        if paired_target is not None:
            rows_out = paired_target_rows(bundle, paired_target, train, test)
            paired.extend(rows_out)
            values = [float(row["improvement"]) for row in rows_out]
            fold_improvements.append(float(np.mean(values)) if values else None)
        probabilities, expected, mom, always, rets = _policy_components(
            bundle, test, position=position
        )
        for value in threshold_list:
            model_pos[value].extend(
                _positions_for(probabilities, expected, threshold=value, position=position)
            )
        mom_pos.extend(mom)
        always_pos.extend(always)
        returns.extend(rets)
    aggregated = {
        target: agg
        for target, records in target_records.items()
        if (agg := aggregate_target(records)) is not None
    }
    grid = [
        {
            "threshold": value,
            "fee_bps": fee,
            "slippage_bps": slip,
            "model": _pnl_metrics(
                model_pos[value], returns, fee_bps=fee, slippage_bps=slip
            ),
            "momentum": _pnl_metrics(mom_pos, returns, fee_bps=fee, slippage_bps=slip),
            "always_long": _pnl_metrics(
                always_pos, returns, fee_bps=fee, slippage_bps=slip
            ),
        }
        for value in threshold_list
        for fee, slip in cost_list
    ]
    headline = next(
        (
            cell
            for cell in grid
            if cell["threshold"] == threshold
            and cell["fee_bps"] == fee_bps
            and cell["slippage_bps"] == slippage_bps
        ),
        grid[0],
    )
    economic = {
        "model": headline["model"],
        "momentum": headline["momentum"],
        "always_long": headline["always_long"],
    }
    paired_values = [float(row["improvement"]) for row in paired]
    return {
        "ticker": ticker,
        "config": config.as_dict(),
        "folds": len(splits),
        "samples": len(samples),
        "reserved_final": reserve_final,
        "holdout_start": holdout_start,
        "diagnostic_full_archive": bool(full_archive),
        "span": sample_span(samples),
        "targets": aggregated,
        "economic": economic,
        "economic_grid": grid,
        "policy": {
            "threshold": threshold,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "model": list(model_pos.get(threshold, [])),
            "momentum": list(mom_pos),
            "always_long": list(always_pos),
            "returns": list(returns),
        },
        "paired_target": paired_target,
        "paired": {
            "n": len(paired_values),
            "mean_improvement": float(np.mean(paired_values)) if paired_values else None,
        },
        "paired_rows": paired,
        "fold_improvements": fold_improvements,
        "purge": validate_splits(
            samples, splits, targets=targets, horizon=config.horizon
        ),
        "leak_detected": bool(leaks),
        "leak_features": leaks,
    }


def _unique(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def status_payload(tickers: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ticker in tickers:
        rows = load_ticker_rows(ticker)
        summary: dict[str, Any] = {}
        path = summary_path(ROOT, ticker)
        if path.exists():
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {}
        bundle: Tier1Bundle | None = None
        error: str | None = None
        try:
            bundle = load_bundle(default_bundle_path(ticker, directory=ROOT / "models"))
        except FileNotFoundError:
            error = "no bundle"
        except Exception as exc:  # status must report, not crash
            error = f"{type(exc).__name__}: {exc}"
        trusted = [t for t in TARGET_NAMES if bundle is not None and bundle.trusted(t)]
        direction = bundle.targets.get("fwd_direction") if bundle else None
        holdout = direction.holdout if direction else None
        target_holdouts: dict[str, dict[str, Any]] = {}
        if bundle is not None:
            for name in TARGET_NAMES:
                model = bundle.targets.get(name)
                if model is not None and model.holdout is not None:
                    target_holdouts[name] = {
                        "n": model.holdout.n_samples,
                        "beats": model.holdout.beats_baseline,
                    }
        out.append(
            {
                "ticker": ticker,
                "bars": len(rows),
                "span": (rows[0].timestamp, rows[-1].timestamp) if rows else None,
                "bundle": str(default_bundle_path(ticker, directory=ROOT / "models")),
                "bundle_error": error,
                "feature_version": bundle.feature_version if bundle else None,
                "feature_mismatch": bool(bundle and bundle.feature_version != FEATURE_VERSION),
                "trusted": trusted,
                "holdout": {
                    "target": "fwd_direction",
                    "n": holdout.n_samples if holdout else None,
                    "beats": holdout.beats_baseline if holdout else None,
                    "metrics": holdout.metrics if holdout else None,
                },
                "target_holdouts": target_holdouts,
                "summary_schema": summary.get("schema"),
                "summary_generated_at": summary.get("generated_at"),
            }
        )
    return out


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _save(action: str, payload: dict[str, Any]) -> Path:
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = LAB_DIR / f"{action}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    ledger = {
        "at": to_iso(utcnow()),
        "action": action,
        "artifact": str(path),
        "summary": payload.get("summary", {}),
    }
    with (LAB_DIR / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ledger, default=str) + "\n")
    return path


def aggregate_purge(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One purge verdict for a multi-ticker payload."""
    rows = [result.get("purge") or {} for result in results if "error" not in result]
    violations = [
        {"ticker": result.get("ticker"), **violation}
        for result in results
        for violation in ((result.get("purge") or {}).get("violations") or [])
    ]
    return {
        "ok": not violations,
        "required_purge": max(
            (int(row.get("required_purge") or 0) for row in rows), default=0
        ),
        "violations": violations,
    }


def contract_block(
    *,
    fit_scope: str = "train",
    selection_scope: str = "not-applicable",
) -> dict[str, Any]:
    """Declarative provenance blocks the mechanical verdict can refuse.

    The lab reports what it actually did; these are the fields that make a
    fit-on-everything scaler or a holdout-selected candidate expressible as a
    deterministic ``invalid`` rather than something only a human reviewer might
    notice.
    """
    return {
        "preprocessing": {"fit_scope": fit_scope},
        "selection": {"scope": selection_scope},
    }


def _pool_results(
    results: Sequence[dict[str, Any]],
    *,
    target: str,
    folds: int,
    alpha: float,
    seed: int,
    block_bars: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    rows_by_ticker = {
        str(result["ticker"]): list(result.get("paired_rows") or [])
        for result in results
        if "error" not in result
    }
    folds_by_ticker = {
        str(result["ticker"]): list(result.get("fold_improvements") or [])
        for result in results
        if "error" not in result
    }
    pooled = bootstrap_pooled(
        rows_by_ticker,
        alpha=alpha,
        seed=seed,
        block_bars=block_bars,
        bootstrap_samples=bootstrap_samples,
    )
    pooled["fold_improvements"] = pool_fold_improvements(folds_by_ticker, folds)
    pooled["target"] = target
    for result in results:
        result.pop("paired_rows", None)
        result.pop("fold_improvements", None)
    return pooled


def _controls_for(
    tickers: Sequence[str],
    config: Config,
    target: str,
    *,
    folds: int,
    holdout_start: str | None,
    seed: int,
    shift_bars: int,
    alpha: float,
    block_bars: int,
    bootstrap_samples: int,
    modes: Sequence[str],
    leak_feature: str | None = None,
) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    for mode in modes:
        per_ticker = {
            ticker: evaluate_control_ticker(
                ticker,
                config,
                target,
                mode,
                folds=folds,
                holdout_start=holdout_start,
                seed=seed,
                shift_bars=shift_bars,
                leak_feature=leak_feature,
            )
            for ticker in tickers
        }
        pooled = bootstrap_pooled(
            {ticker: row["paired_rows"] for ticker, row in per_ticker.items()},
            alpha=alpha,
            seed=seed,
            block_bars=block_bars,
            bootstrap_samples=bootstrap_samples,
        )
        pooled["fold_improvements"] = pool_fold_improvements(
            {ticker: row["fold_improvements"] for ticker, row in per_ticker.items()},
            folds,
        )
        controls[mode] = {
            "pooled": pooled,
            "per_ticker": pooled.get("per_ticker") or {},
            "shift_bars": int(shift_bars) if mode == "shift" else None,
        }
    return controls


def _parse_thresholds(value: str | None) -> list[float] | None:
    """``--thresholds "0.5 0.55 0.6"`` (commas or spaces) or ``None``."""
    if value is None:
        return None
    values = [float(part) for part in str(value).replace(",", " ").split() if part.strip()]
    for threshold in values:
        if not 0.0 < threshold < 1.0:
            raise SystemExit(f"threshold {threshold} must be strictly between 0 and 1")
    return values or None


def _parse_cost_levels(value: str | None) -> list[tuple[float, float]] | None:
    """``--cost-levels "10:5 0:0"`` as ``fee:slippage`` per-side pairs."""
    if value is None:
        return None
    out: list[tuple[float, float]] = []
    for part in str(value).replace(";", " ").replace(",", " ").split():
        separator = ":" if ":" in part else ("/" if "/" in part else None)
        if separator is None:
            raise SystemExit(f"cost level {part!r} must look like fee:slippage (e.g. 10:5)")
        fee, slippage = part.split(separator, 1)
        out.append((float(fee), float(slippage)))
    return out or None


def _unique_leaks(leaks: Any) -> list[dict[str, Any]]:
    """One entry per (feature, target), even when every ticker reports it."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for leak in leaks:
        key = (leak.get("feature"), leak.get("target"))
        if key in seen:
            continue
        seen.add(key)
        out.append(leak)
    return out


def _pooled_economics(
    results: Sequence[dict[str, Any]], *, fee_bps: float, slippage_bps: float
) -> dict[str, Any]:
    """After-cost economics on the pooled portfolio, not the first result row.

    A single ticker's economics is one correlated bet; a v2 claim about
    ``fwd_direction``/``fwd_return`` must be judged on every scoped ticker's
    positions concatenated in time. The per-ticker ``policy`` block is consumed
    here and removed so it is not persisted twice.
    """
    model: list[float] = []
    momentum: list[float] = []
    always: list[float] = []
    returns: list[float] = []
    for result in results:
        if "error" in result:
            continue
        policy = result.pop("policy", None) or {}
        model.extend(policy.get("model") or [])
        momentum.extend(policy.get("momentum") or [])
        always.extend(policy.get("always_long") or [])
        returns.extend(policy.get("returns") or [])
    if not returns:
        return {"model": _pnl_metrics([], [], fee_bps=fee_bps, slippage_bps=slippage_bps)}
    return {
        "model": _pnl_metrics(model, returns, fee_bps=fee_bps, slippage_bps=slippage_bps),
        "momentum": _pnl_metrics(momentum, returns, fee_bps=fee_bps, slippage_bps=slippage_bps),
        "always_long": _pnl_metrics(
            always, returns, fee_bps=fee_bps, slippage_bps=slippage_bps
        ),
        "bars": len(returns),
        "tickers": len([r for r in results if "error" not in r]),
    }


def evaluate_action(args: argparse.Namespace) -> int:
    leak_feature = getattr(args, "canary_leak", None)
    feature_names = resolve_feature_names(args.feature_set)
    if leak_feature:
        feature_names = tuple(dict.fromkeys((*feature_names, leak_feature)))
    config = Config(
        name=f"{args.feature_set}+{args.model_set}+h{args.horizon}",
        feature_names=feature_names,
        model_names=resolve_model_set(args.model_set),
        horizon=args.horizon,
    )
    tickers = resolve_tickers(args.tickers)
    targets = resolve_targets(args.targets)
    primary_target = args.primary_target or (
        "fwd_direction" if "fwd_direction" in targets else targets[0]
    )
    if primary_target not in targets:
        raise SystemExit(f"primary target {primary_target!r} is not in --targets")
    thresholds = _parse_thresholds(getattr(args, "thresholds", None))
    cost_levels = _parse_cost_levels(getattr(args, "cost_levels", None))
    results = [
        evaluate_ticker(
            ticker,
            config,
            targets,
            folds=args.folds,
            threshold=args.threshold,
            position=args.position,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            thresholds=thresholds,
            cost_levels=cost_levels,
            holdout_start=args.holdout_start,
            full_archive=args.full_archive,
            primary_target=primary_target,
            leak_feature=leak_feature,
        )
        for ticker in tickers
    ]
    pooled_economics = _pooled_economics(
        results,
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
    )
    pooled_economics["threshold"] = float(args.threshold)
    pooled_economics["fee_bps"] = float(args.fee_bps)
    pooled_economics["slippage_bps"] = float(args.slippage_bps)
    pooled = _pool_results(
        results,
        target=primary_target,
        folds=args.folds,
        alpha=args.alpha,
        seed=args.seed,
        block_bars=args.bootstrap_block,
        bootstrap_samples=args.bootstrap_samples,
    )
    controls: dict[str, Any] = {}
    if args.controls != "none":
        modes = ["shuffle", "shift"] if args.controls == "both" else [args.controls]
        controls = _controls_for(
            tickers,
            config,
            primary_target,
            folds=args.folds,
            holdout_start=args.holdout_start,
            seed=args.seed,
            shift_bars=args.shift_bars,
            alpha=args.alpha,
            block_bars=args.bootstrap_block,
            bootstrap_samples=args.bootstrap_samples,
            modes=modes,
            leak_feature=leak_feature,
        )
    payload = {
        "action": "evaluate",
        "config": config.as_dict(),
        "primary": config.name,
        "primary_target": primary_target,
        "targets": list(targets),
        "holdout_start": args.holdout_start,
        "diagnostic_full_archive": bool(args.full_archive),
        "seed": int(args.seed),
        "alpha": float(args.alpha),
        "pooled": pooled,
        "pooled_economics": pooled_economics,
        "leak_detected": any(result.get("leak_detected") for result in results),
        "leak_features": _unique_leaks(
            leak
            for result in results
            for leak in (result.get("leak_features") or [])
        ),
        "purge": aggregate_purge(results),
        **contract_block(
            fit_scope="train",
            selection_scope=(
                "pre-cutoff"
                if args.holdout_start
                else "reserved-final"
                if not args.full_archive
                else "not-applicable"
            ),
        ),
        "controls": controls,
        "results": results,
        "summary": {
            "tickers": len(results),
            "errors": sum(1 for r in results if "error" in r),
            "targets_beating": sum(
                1
                for r in results
                for t in (r.get("targets") or {}).values()
                if t.get("beats_ratio", 0) >= 0.6
            ),
            "pooled_mean": pooled.get("mean_improvement"),
            "ci_low": pooled.get("ci_low"),
            "ci_high": pooled.get("ci_high"),
        },
    }
    path = _save("evaluate", payload)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print("target metrics (walk-forward, chronological, constant baseline):")
    _print_table(
        ["ticker", "target", "folds", "model", "baseline", "mean%", "min%", "beats"],
        [
            [
                r["ticker"],
                t["target"],
                t["folds"],
                f"{t['model_mean']:.5f}" if t["model_mean"] is not None else "—",
                f"{t['baseline_mean']:.5f}" if t["baseline_mean"] is not None else "—",
                f"{100 * t['mean_improvement']:+.2f}" if t["mean_improvement"] is not None else "—",
                f"{100 * t['min_improvement']:+.2f}" if t["min_improvement"] is not None else "—",
                f"{t['folds_beating']}/{t['folds']}",
            ]
            for r in results
            if "error" not in r
            for t in (r.get("targets") or {}).values()
        ],
    )
    print("\npooled paired improvement (primary config; time-block bootstrap):")
    print(
        f"  mean {pooled.get('mean_improvement')}  "
        f"CI[{pooled.get('ci_low')}, {pooled.get('ci_high')}]  "
        f"n={pooled.get('n')} blocks={pooled.get('blocks')} seed={pooled.get('seed')}"
    )
    if controls:
        for name, block in controls.items():
            control = block.get("pooled") or {}
            print(
                f"  {name} control: mean {control.get('mean_improvement')}  "
                f"CI[{control.get('ci_low')}, {control.get('ci_high')}]"
            )
    print("\nafter-cost policy (direction-preferred; 10 bps + 5 bps per side):")
    _print_table(
        ["ticker", "bars", "trades", "sharpe", "ret", "dd", "mom", "always"],
        [
            [
                r["ticker"],
                r["economic"]["model"]["bars"],
                r["economic"]["model"]["trades"],
                f"{r['economic']['model']['sharpe']:.2f}"
                if r["economic"]["model"]["sharpe"] is not None
                else "—",
                f"{100 * r['economic']['model']['total_return']:+.2f}%"
                if r["economic"]["model"]["total_return"] is not None
                else "—",
                f"{100 * r['economic']['model']['max_drawdown']:.2f}%"
                if r["economic"]["model"]["max_drawdown"] is not None
                else "—",
                f"{r['economic']['momentum']['sharpe']:.2f}"
                if r["economic"]["momentum"]["sharpe"] is not None
                else "—",
                f"{r['economic']['always_long']['sharpe']:.2f}"
                if r["economic"]["always_long"]["sharpe"] is not None
                else "—",
            ]
            for r in results
            if "error" not in r
        ],
    )
    print(f"\nartifact: {path}")
    return 0


def _configs(args: argparse.Namespace) -> list[Config]:
    feature_sets = [s.strip() for s in args.feature_sets.split(",") if s.strip()]
    model_sets = [s.strip() for s in args.model_sets.split(",") if s.strip()]
    horizons = [int(h) for h in str(args.horizons).replace(",", " ").split() if h.strip()]
    configs = [
        Config(
            name=f"{fs}+{ms}+h{h}",
            feature_names=resolve_feature_names(fs),
            model_names=resolve_model_set(ms),
            horizon=h,
        )
        for fs, ms, h in itertools.product(feature_sets, model_sets, horizons)
    ]
    return configs[: max(1, args.max_configs)]


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, int | float) else default


def _rank_key(result: dict[str, Any], primary: str) -> tuple[Any, ...]:
    agg = (result.get("targets") or {}).get(primary) or {}
    econ = (result.get("economic") or {}).get("model") or {}
    beats = _number(agg.get("beats_ratio"), 0.0)
    min_imp = _number(agg.get("min_improvement"), -1.0)
    mean_imp = _number(agg.get("mean_improvement"), -1.0)
    sharpe = _number(econ.get("sharpe"), -999.0)
    dd = _number(econ.get("max_drawdown"), -1.0)
    gate = int(beats >= 0.6 and min_imp > 0.0)
    return (gate, beats, min_imp, mean_imp, sharpe, dd)


def sweep_action(args: argparse.Namespace) -> int:
    tickers = resolve_tickers(args.tickers) if args.tickers else ("BTC/USD", "ETH/USD", "SOL/USD")
    targets = resolve_targets(args.targets)
    primary_target = args.primary_target or (
        "fwd_direction" if "fwd_direction" in targets else targets[0]
    )
    configs = _configs(args)
    primary_config = args.primary_config or configs[0].name
    if primary_config not in {config.name for config in configs}:
        raise SystemExit(
            f"primary config {primary_config!r} is not one of {[c.name for c in configs]}"
        )
    results: list[dict[str, Any]] = []
    for ticker in tickers:
        for config in configs:
            result = evaluate_ticker(
                ticker,
                config,
                targets,
                folds=args.folds,
                threshold=args.threshold,
                position=args.position,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                reserve_final=0.2,
                holdout_start=args.holdout_start,
                primary_target=primary_target,
            )
            result["primary"] = config.name == primary_config
            result["rank"] = _rank_key(result, primary_target)
            results.append(result)
    primary_rows = [r for r in results if r.get("primary") and "error" not in r]
    pooled = _pool_results(
        primary_rows,
        target=primary_target,
        folds=args.folds,
        alpha=args.alpha,
        seed=args.seed,
        block_bars=args.bootstrap_block,
        bootstrap_samples=args.bootstrap_samples,
    )
    for result in results:
        if not result.get("primary"):
            result.pop("paired_rows", None)
            result.pop("fold_improvements", None)
    ranked = sorted(results, key=lambda r: r.get("rank", ()), reverse=True)
    top = [r for r in ranked if r.get("rank", (0,))[0]][: args.limit]
    candidates: list[str] = []
    for result in top:
        if not result.get("primary"):
            continue
        ticker = result["ticker"]
        config_name = result["config"]["name"]
        rows = load_ticker_rows(ticker)
        if args.holdout_start:
            rows, _ = filter_rows_before(rows, args.holdout_start)
        samples = build_samples(
            rows, horizon=result["config"]["horizon"], interval_seconds=INTERVAL_SECONDS
        )
        split = len(samples) if args.holdout_start else int(len(samples) * 0.8)
        if split < 100:
            continue
        config = next(c for c in configs if c.name == config_name)
        bundle = train_tier1(
            samples[:split],
            horizon=config.horizon,
            feature_names=config.feature_names,
            targets=targets,
            model_names=config.model_names,
            holdout_fraction=0.0,
        )
        candidate = (
            LAB_DIR / "candidates" / f"{slug_for(ticker)}-{config.name.replace('+', '-')}.json"
        )
        candidate.parent.mkdir(parents=True, exist_ok=True)
        save_bundle(bundle, candidate)
        candidates.append(str(candidate))
        result["candidate"] = str(candidate)
    payload = {
        "action": "sweep",
        "tickers": list(tickers),
        "targets": list(targets),
        "primary": primary_config,
        "primary_target": primary_target,
        "holdout_start": args.holdout_start,
        "seed": int(args.seed),
        "alpha": float(args.alpha),
        "pooled": pooled,
        "configs": [c.as_dict() for c in configs],
        "results": results,
        "top": top,
        "candidates": candidates,
        "primary_candidates": candidates,
        "purge": aggregate_purge(results),
        **contract_block(
            fit_scope="train",
            selection_scope="pre-cutoff" if args.holdout_start else "reserved-final",
        ),
        "summary": {
            "tickers": len(tickers),
            "configs": len(configs),
            "runs": len(results),
            "gated": sum(1 for r in results if r.get("rank", (0,))[0]),
            "candidates": len(candidates),
            "reserved_final": 0.2,
            "holdout_start": args.holdout_start,
            "pooled_mean": pooled.get("mean_improvement"),
            "ci_low": pooled.get("ci_low"),
        },
    }
    path = _save("sweep", payload)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    holdout = (
        f"before {args.holdout_start}"
        if args.holdout_start
        else "the first 80% (final 20% reserved)"
    )
    print(f"sweep evaluates {holdout}; only {primary_config} is verdict-eligible.")
    print(
        f"pooled paired: mean {pooled.get('mean_improvement')} "
        f"CI[{pooled.get('ci_low')}, {pooled.get('ci_high')}]"
    )
    if not top:
        print("No config beat the baseline consistently across folds.")
        print(
            "That is a valid result: the current data/features do not support a directional edge."
        )
        print(f"artifact: {path}")
        return 0
    _print_table(
        [
            "ticker",
            "config",
            "primary",
            "beats",
            "min%",
            "mean%",
            "sharpe",
            "ret",
            "dd",
            "candidate",
        ],
        [
            [
                r["ticker"],
                r["config"]["name"],
                "yes" if r.get("primary") else "—",
                f"{r['targets'][primary_target]['folds_beating']}/{r['targets'][primary_target]['folds']}",
                f"{100 * (r['targets'][primary_target]['min_improvement'] or 0):+.2f}",
                f"{100 * (r['targets'][primary_target]['mean_improvement'] or 0):+.2f}",
                f"{r['economic']['model']['sharpe']:.2f}"
                if r["economic"]["model"]["sharpe"] is not None
                else "—",
                f"{100 * r['economic']['model']['total_return']:+.2f}%"
                if r["economic"]["model"]["total_return"] is not None
                else "—",
                f"{100 * r['economic']['model']['max_drawdown']:.2f}%"
                if r["economic"]["model"]["max_drawdown"] is not None
                else "—",
                "yes" if r.get("candidate") else "—",
            ]
            for r in top
        ],
    )
    print(f"\nartifact: {path}")
    return 0


def _bundle_view(
    bundle: Tier1Bundle,
    samples: Sequence[Sample],
    targets: Sequence[str],
    *,
    split: int,
    threshold: float,
    position: str,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    targets_out: dict[str, Any] = {}
    for target in targets:
        record = evaluate_target(bundle, target, samples[:split], samples[split:])
        if record is not None:
            targets_out[target] = record
    model, mom, always, rets = _policy_rows(
        bundle, samples[split:], threshold=threshold, position=position
    )
    return {
        "targets": targets_out,
        "economic": {
            "model": _pnl_metrics(model, rets, fee_bps=fee_bps, slippage_bps=slippage_bps),
            "momentum": _pnl_metrics(mom, rets, fee_bps=fee_bps, slippage_bps=slippage_bps),
            "always_long": _pnl_metrics(always, rets, fee_bps=fee_bps, slippage_bps=slippage_bps),
        },
    }


def _infer_ticker(candidate: Path, tickers: Sequence[str]) -> str | None:
    for ticker in tickers:
        if candidate.name.startswith(f"{slug_for(ticker)}-"):
            return ticker
    return None


def boundary_purge(targets: Sequence[str], horizon: int) -> int:
    """Rows to drop before a holdout boundary so no training label reads into it."""
    return max((label_purge(target, horizon) for target in targets), default=max(0, horizon))


def promotion_gate(
    candidate: dict[str, Any], current: dict[str, Any] | None, primary: str
) -> list[str]:
    reasons: list[str] = []
    cand_target = (candidate.get("targets") or {}).get(primary)
    if cand_target is None:
        return [f"candidate has no {primary} model"]
    if not cand_target.get("beats"):
        reasons.append("candidate does not beat the constant baseline on the final holdout")
    cur_target = ((current or {}).get("targets") or {}).get(primary)
    if cur_target is not None:
        got, bar = cand_target.get("model"), cur_target.get("model")
        if isinstance(got, int | float) and isinstance(bar, int | float) and got > bar * 0.99:
            reasons.append("candidate is not at least 1% better than the current bundle")
    cand_econ = (candidate.get("economic") or {}).get("model") or {}
    cur_econ = ((current or {}).get("economic") or {}).get("model") or {}
    trades = cand_econ.get("trades") or 0
    if trades < 10:
        reasons.append(f"only {trades} trades on the final holdout (need >= 10)")
    if (
        isinstance(cur_econ.get("sharpe"), int | float)
        and isinstance(cand_econ.get("sharpe"), int | float)
        and cand_econ["sharpe"] < cur_econ["sharpe"]
    ):
        reasons.append("candidate Sharpe is below the current bundle")
    if (
        isinstance(cur_econ.get("max_drawdown"), int | float)
        and isinstance(cand_econ.get("max_drawdown"), int | float)
        and cand_econ["max_drawdown"] < cur_econ["max_drawdown"] * 1.25
    ):
        reasons.append("candidate drawdown is materially worse than the current bundle")
    return reasons


def promote_action(args: argparse.Namespace) -> int:
    if args.apply and not args.holdout_start:
        raise SystemExit(
            "promote --apply requires --holdout-start: without a frozen cutoff the "
            "comparison is against a deployed bundle that may have seen the holdout"
        )
    candidate_path = Path(args.candidate).expanduser().resolve()
    if not candidate_path.exists():
        raise SystemExit(f"candidate not found: {candidate_path}")
    tickers = resolve_tickers(args.tickers)
    ticker = args.ticker or _infer_ticker(candidate_path, tickers)
    if ticker is None:
        raise SystemExit("could not infer ticker from candidate name; pass --ticker BTC/USD")
    targets = resolve_targets(args.targets)
    candidate = load_bundle(candidate_path)
    rows = load_ticker_rows(ticker)
    if len(rows) < 120:
        raise SystemExit(f"{ticker}: only {len(rows)} candles")
    samples = build_samples(rows, horizon=candidate.horizon, interval_seconds=INTERVAL_SECONDS)
    if args.holdout_start:
        split = sample_split(samples, args.holdout_start)
        if split is None or split < 100:
            raise SystemExit(f"{ticker}: not enough samples before the holdout cutoff")
    else:
        split = int(len(samples) * 0.8)
    if split < 100:
        raise SystemExit(f"{ticker}: not enough samples for a final holdout ({split})")
    if split >= len(samples):
        raise SystemExit(f"{ticker}: the holdout cutoff is at or after the end of the archive")
    try:
        current = load_bundle(default_bundle_path(ticker, directory=ROOT / "models"))
    except FileNotFoundError:
        current = None
    common = {
        "split": split,
        "threshold": args.threshold,
        "position": args.position,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
    }
    cand_view = _bundle_view(candidate, samples, targets, **common)
    deployed_view = (
        _bundle_view(current, samples, targets, **common) if current is not None else None
    )
    champion_view = deployed_view
    comparison_basis = "deployed-bundle"
    if current is not None and args.champion_retrain and args.holdout_start:
        # The deployed bundle may have trained on the holdout. Retrain its
        # configuration on exactly the pre-cutoff slice and compare against that.
        model_names = {
            target: current.targets[target].model.to_dict().get("kind")
            for target in targets
            if target in current.targets
        }
        # The last training label must not read the first holdout bar: purge the
        # widest target reach at the frozen cutoff, exactly as walk_forward does.
        train_slice = samples[: max(0, split - boundary_purge(targets, current.horizon))]
        champion_bundle = train_tier1(
            train_slice,
            horizon=current.horizon,
            feature_names=current.feature_names,
            targets=list(targets),
            model_names=model_names,
            holdout_fraction=0.0,
        )
        champion_view = _bundle_view(champion_bundle, samples, targets, **common)
        comparison_basis = "retrained-champion"
    primary = "fwd_direction" if "fwd_direction" in targets else targets[0]
    reasons = promotion_gate(cand_view, champion_view, primary)
    if candidate.feature_version != FEATURE_VERSION:
        reasons.append(
            f"candidate feature_version {candidate.feature_version!r} != {FEATURE_VERSION!r}"
        )
    if current is not None and current.horizon != candidate.horizon:
        reasons.append(
            f"current bundle horizon {current.horizon} differs from candidate "
            f"{candidate.horizon}; retrain on one horizon before promoting"
        )
    recommendation = {
        "feature_names": list(candidate.feature_names),
        "horizon": candidate.horizon,
        "targets": list(candidate.targets),
        "models": {
            name: model.model.to_dict().get("kind") for name, model in candidate.targets.items()
        },
        "note": (
            "To make this durable, pass these feature_names/model_names to "
            "train_tier1 in pipeline/run.py; the scheduled train overwrites "
            "models/<slug>.json otherwise."
        ),
    }
    payload = {
        "action": "promote",
        "ticker": ticker,
        "candidate": str(candidate_path),
        "current_bundle": str(default_bundle_path(ticker, directory=ROOT / "models")),
        "apply": bool(args.apply),
        "primary": primary,
        "holdout_start": args.holdout_start,
        "comparison_basis": comparison_basis,
        "candidate_view": cand_view,
        "champion_view": champion_view,
        "deployed_view": deployed_view,
        "recommendation": recommendation,
        "conservative_note": (
            "The deployed bundle may have been trained on data that includes this "
            "holdout; it is informational only. The gate uses the champion "
            "retrained on the pre-cutoff slice when --holdout-start is supplied."
        ),
        "reasons": reasons,
        "passed": not reasons,
        "summary": {
            "passed": not reasons,
            "ticker": ticker,
            "reasons": reasons,
            "comparison_basis": comparison_basis,
        },
    }
    if args.apply and not reasons:
        current_path = default_bundle_path(ticker, directory=ROOT / "models")
        if current_path.exists():
            LAB_DIR.mkdir(parents=True, exist_ok=True)
            backup = (
                LAB_DIR
                / "backups"
                / f"{slug_for(ticker)}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
            )
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_text(current_path.read_text(encoding="utf-8"), encoding="utf-8")
            payload["backup"] = str(backup)
        save_bundle(candidate, current_path)
        # The bundle alone is overwritten by the next scheduled retrain. Record
        # the decision in the committed training config too, so `pipeline/run.py`
        # rebuilds the same feature/model/horizon set from git.
        apply_promotion(
            ROOT,
            ticker=ticker,
            feature_names=candidate.feature_names,
            model_names={
                name: str(model.model.to_dict().get("kind"))
                for name, model in candidate.targets.items()
            },
            horizon=candidate.horizon,
        )
        payload["written"] = str(current_path)
        payload["training_config"] = str(ROOT / "config" / "training.json")
        payload["warning"] = (
            "The promoted feature/model/horizon set is recorded in "
            "config/training.json, so the next scheduled retrain rebuilds it."
        )
    path = _save("promote", payload)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"ticker:    {ticker}")
    print(f"candidate: {candidate_path}")
    print(f"gate:      {'PASS' if not reasons else 'FAIL'}")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"durable:   {len(candidate.feature_names)} features, horizon {candidate.horizon}")
    print(f"basis:     {payload['comparison_basis']} (deployed bundle is informational)")
    if args.apply:
        print(f"applied:   {'yes' if not reasons else 'no (gate failed)'}")
        if payload.get("written"):
            print(f"written:   {payload['written']}")
            print(f"warning:   {payload['warning']}")
    else:
        print("dry run:   pass --apply to write models/<slug>.json after a passing gate")
    print(f"artifact:  {path}")
    return 0


def report_action(args: argparse.Namespace) -> int:
    ledger = LAB_DIR / "ledger.jsonl"
    rows: list[dict[str, Any]] = []
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines()[-args.limit :]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    payload = {"action": "report", "ledger": str(ledger), "entries": rows}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if not rows:
        print(f"no ledger entries yet ({ledger})")
        return 0
    _print_table(
        ["at", "action", "artifact", "summary"],
        [
            [
                r.get("at", ""),
                r.get("action", ""),
                Path(r.get("artifact", "")).name,
                json.dumps(r.get("summary", {}), default=str)[:80],
            ]
            for r in rows
        ],
    )
    return 0


def _holdout_beats(row: dict[str, Any]) -> str:
    short = {
        "fwd_return": "ret",
        "fwd_direction": "dir",
        "fwd_volatility": "vol",
        "fwd_volume_ratio": "vratio",
    }
    parts = []
    for name, info in (row.get("target_holdouts") or {}).items():
        beats = info.get("beats")
        if beats is None:
            continue
        parts.append(f"{short.get(name, name)}={'T' if beats else 'F'}")
    return " ".join(parts) or "—"


def status_action(args: argparse.Namespace) -> int:
    payload = {"action": "status", "tickers": status_payload(resolve_tickers(args.tickers))}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    _print_table(
        ["ticker", "bars", "feature", "trusted", "holdout beats", "error"],
        [
            [
                r["ticker"],
                r["bars"],
                r["feature_version"] or "—",
                ",".join(r["trusted"]) or "—",
                _holdout_beats(r),
                r["bundle_error"] or "",
            ]
            for r in payload["tickers"]
        ],
    )
    print(
        "\nholdout beats: ret=fwd_return dir=fwd_direction "
        "vol=fwd_volatility vratio=fwd_volume_ratio"
    )
    print("trusted = targets whose own holdout beat the baseline.")
    return 0


HELP_TEXT = """\
rl-model-lab — offline, out-of-sample model loop

Actions:
  status    deployed bundles, trust, freshness
  evaluate  walk-forward target metrics + after-cost policy
  sweep     controlled feature/model/horizon experiments
  promote   validate a candidate on the reserved final holdout
  report    experiment ledger
  help      this reference

Common arguments (evaluate / sweep / promote):
  --tickers BTC/USD ETH/USD     comma- or space-separated; default all
                                (sweep default: BTC/USD, ETH/USD, SOL/USD)
  --targets fwd_direction ...   default all four targets
  --folds N                     walk-forward folds (default 4)
  --threshold P                 direction probability for a long (default 0.55)
  --position long-only|long-short
  --fee-bps N                   per-side fee (default 10)
  --slippage-bps N              per-side slippage (default 5)
  --holdout-start ISO           frozen timestamp cutoff (strictly before it)
  --seed N                      controls + bootstrap seed (default 0)
  --alpha F                     CI alpha (driver passes alpha/trials)
  --bootstrap-samples N         block-bootstrap draws (default 1000)
  --bootstrap-block N           bars per block (default 24)
  --primary-target TARGET       target of record (default fwd_direction)

Action-specific:
  status    --tickers, --json
  evaluate  --feature-set all|market|core|price (default all)
            --model-set reference|baseline (default reference)
            --horizon N (default 1; the economic proxy is only honest at 1)
            --controls none|shuffle|shift|both
            --shift-bars N (default 24)
            --thresholds 0.5,0.55,0.6   one fit, many policy thresholds
            --cost-levels 10:5,0:0      one fit, many cost levels
            --full-archive (diagnostic only; cannot feed a verdict)
            --limit N (default 20)
  sweep     --feature-sets all,market,core,price
            --model-sets reference,baseline
            --horizons 1
            --max-configs N (default 16)
            --primary-config NAME (the only verdict-eligible config)
            --limit N (default 20)
  promote   --candidate PATH (required)
            --ticker BTC/USD (inferred from the filename when possible)
            --holdout-start ISO (retrain the champion on the pre-cutoff slice)
            --no-champion-retrain (compare against the deployed bundle)
            --apply (write models/<slug>.json only after a passing gate)
  report    --limit N (default 10)

Every action accepts --json. Artifacts and the ledger live under
~/.local/share/rl-crypto/model-lab (override RL_MODEL_LAB_DIR).
Nothing here guarantees profit: the final 20% is reserved for promotion,
costs are charged per side, and the gate compares against a constant baseline.
"""


def help_action(args: argparse.Namespace) -> int:
    print(HELP_TEXT)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-model-lab", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    help_parser = sub.add_parser("help", help="show actions and arguments")
    help_parser.set_defaults(func=help_action)

    status = sub.add_parser("status", help="current bundles, trust and freshness")
    status.add_argument("--tickers", nargs="+", default=None)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=status_action)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tickers", nargs="+", default=None)
        p.add_argument("--targets", nargs="+", default=None)
        p.add_argument("--primary-target", default=None)
        p.add_argument("--folds", type=int, default=4)
        p.add_argument("--threshold", type=float, default=0.55)
        p.add_argument("--position", choices=("long-only", "long-short"), default="long-only")
        p.add_argument("--fee-bps", type=float, default=10.0)
        p.add_argument("--slippage-bps", type=float, default=5.0)
        p.add_argument("--holdout-start", default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument("--bootstrap-samples", type=int, default=1000)
        p.add_argument("--bootstrap-block", type=int, default=24)
        p.add_argument("--json", action="store_true")
        p.add_argument("--limit", type=int, default=20)

    evaluate = sub.add_parser("evaluate", help="walk-forward metrics + economic proxy")
    common(evaluate)
    evaluate.add_argument("--feature-set", default="all")
    evaluate.add_argument("--model-set", default="reference")
    evaluate.add_argument("--horizon", type=int, default=1)
    evaluate.add_argument("--full-archive", action="store_true")
    evaluate.add_argument(
        "--controls", choices=("none", "shuffle", "shift", "both"), default="none"
    )
    evaluate.add_argument("--shift-bars", type=int, default=24)
    evaluate.add_argument(
        "--canary-leak",
        default=None,
        help="positive-canary only: inject this feature from the realised label",
    )
    evaluate.add_argument(
        "--thresholds",
        default=None,
        help="direction thresholds to sweep in one fit, e.g. '0.5,0.55,0.6'",
    )
    evaluate.add_argument(
        "--cost-levels",
        default=None,
        help="per-side fee:slippage pairs to sweep in one fit, e.g. '10:5,0:0'",
    )
    evaluate.set_defaults(func=evaluate_action)

    sweep = sub.add_parser("sweep", help="controlled feature/model/horizon experiments")
    common(sweep)
    sweep.add_argument("--feature-sets", default="all,market,core,price")
    sweep.add_argument("--model-sets", default="reference,baseline")
    sweep.add_argument("--horizons", default="1")
    sweep.add_argument("--max-configs", type=int, default=16)
    sweep.add_argument("--primary-config", default=None)
    sweep.set_defaults(func=sweep_action)

    promote = sub.add_parser("promote", help="validate a candidate on the final holdout")
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--ticker", default=None)
    promote.add_argument("--tickers", nargs="+", default=None)
    promote.add_argument("--targets", nargs="+", default=None)
    promote.add_argument("--holdout-start", default=None)
    promote.add_argument("--seed", type=int, default=0)
    promote.add_argument(
        "--champion-retrain", dest="champion_retrain", action="store_true", default=True
    )
    promote.add_argument("--no-champion-retrain", dest="champion_retrain", action="store_false")

    promote.add_argument("--threshold", type=float, default=0.55)
    promote.add_argument("--position", choices=("long-only", "long-short"), default="long-only")
    promote.add_argument("--fee-bps", type=float, default=10.0)
    promote.add_argument("--slippage-bps", type=float, default=5.0)
    promote.add_argument("--apply", action="store_true")
    promote.add_argument("--json", action="store_true")
    promote.set_defaults(func=promote_action)

    report = sub.add_parser("report", help="summarize the experiment ledger")
    report.add_argument("--limit", type=int, default=10)
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=report_action)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
