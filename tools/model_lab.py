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
    baseline_for,
    beats_baseline,
    classification_metrics,
    regression_metrics,
    train_tier1,
)
from rl_crypto.models import to_iso, utcnow
from rl_crypto.samples import Sample, build_samples
from rl_crypto.targets import TARGET_KINDS, TARGET_NAMES

ROOT = Path(os.environ.get("RL_CRYPTO_ROOT") or Path.cwd()).resolve()
LAB_DIR = Path(
    os.environ.get("RL_MODEL_LAB_DIR")
    or Path.home() / ".local" / "share" / "rl-crypto" / "model-lab"
)
INTERVAL_SECONDS = DEFAULT_INTERVAL_SECONDS
BARS_PER_YEAR = 365.0 * 24.0 * 3600.0 / INTERVAL_SECONDS

MODEL_SETS: dict[str, dict[str, str]] = {
    "reference": {
        "fwd_return": "ridge",
        "fwd_direction": "logistic",
        "fwd_volatility": "ridge",
        "fwd_volume_ratio": "ridge",
    },
    "baseline": {
        "fwd_return": "mean",
        "fwd_direction": "base_rate",
        "fwd_volatility": "mean",
        "fwd_volume_ratio": "mean",
    },
}

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
    min_train_fraction: float = 0.34,
    min_test: int = 20,
) -> list[tuple[tuple[Sample, ...], tuple[Sample, ...]]]:
    """Expanding-window walk-forward splits with a horizon purge.

    The last ``horizon`` training rows are purged because their forward labels
    overlap the test window. This is the smallest honest split for this repo; a
    longer horizon needs a larger purge/embargo.
    """
    n = len(samples)
    if n < min_test * 2:
        return []
    min_train = max(min_test, int(n * min_train_fraction))
    available = n - min_train
    folds = max(1, min(folds, available // min_test))
    test_size = max(min_test, available // folds)
    # ``fwd_volatility`` spans max(horizon, 2) bars, so the purge must cover
    # that longer label even when the configured horizon is 1.
    purge = max(0, horizon, 2)
    split: list[tuple[tuple[Sample, ...], tuple[Sample, ...]]] = []
    for i in range(folds):
        train_end = min_train + i * test_size
        test_end = min(n, train_end + test_size)
        train = tuple(samples[: max(0, train_end - purge)])
        test = tuple(samples[train_end:test_end])
        if len(train) >= min_test and test:
            split.append((train, test))
    return split


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
    metrics = (
        regression_metrics(y_true, y_pred)
        if kind == "regression"
        else classification_metrics(y_true, y_pred)
    )
    baseline = baseline_for(kind, target_labels(train, target), y_true)
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


def _policy_rows(
    bundle: Tier1Bundle,
    test: Sequence[Sample],
    *,
    threshold: float,
    position: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return model/momentum/always-long positions plus realised returns.

    The direction model is preferred; the return model's sign is the fallback.
    A missing prediction or target is skipped, never imputed.
    """
    direction = bundle.targets.get("fwd_direction")
    ret = bundle.targets.get("fwd_return")
    model: list[float] = []
    momentum: list[float] = []
    always: list[float] = []
    realised: list[float] = []
    for sample in test:
        target = sample.targets.get("fwd_return")
        if target is None:
            continue
        pos = 0.0
        if direction is not None:
            probability = direction.predict_row(sample.features)
            if probability is not None:
                if probability >= threshold:
                    pos = 1.0
                elif position == "long-short" and probability <= 1.0 - threshold:
                    pos = -1.0
        elif ret is not None:
            prediction = ret.predict_row(sample.features)
            if prediction is not None:
                pos = 1.0 if prediction > 0 else (-1.0 if position == "long-short" else 0.0)
        ret1 = sample.features.get("ret_1")
        mom = (
            (1.0 if ret1 is not None and ret1 > 0 else -1.0)
            if position == "long-short"
            else (1.0 if ret1 is not None and ret1 > 0 else 0.0)
        )
        model.append(pos)
        momentum.append(mom)
        always.append(1.0)
        realised.append(float(target))
    return model, momentum, always, realised


def _pnl_metrics(
    positions: Sequence[float],
    returns: Sequence[float],
    *,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    if not positions:
        return {
            "bars": 0,
            "trades": 0,
            "turnover": 0.0,
            "total_return": None,
            "sharpe": None,
            "max_drawdown": None,
            "hit_rate": None,
        }
    cost = (fee_bps + slippage_bps) / 10_000.0
    pnl: list[float] = []
    previous = 0.0
    trades = 0
    turnover = 0.0
    for pos, realised in zip(positions, returns, strict=True):
        delta = abs(pos - previous)
        trades += 1 if delta > 0 else 0
        turnover += delta
        pnl.append(pos * realised - cost * delta)
        previous = pos
    pnl[-1] -= cost * abs(previous)
    arr = np.asarray(pnl, dtype=float)
    equity = np.exp(np.cumsum(arr))
    peak = np.maximum.accumulate(equity)
    std = float(np.std(arr))
    return {
        "bars": len(arr),
        "trades": trades,
        "turnover": float(turnover),
        "total_return": float(equity[-1] - 1.0),
        "sharpe": float(np.mean(arr) / std * math.sqrt(BARS_PER_YEAR)) if std > 0 else None,
        "max_drawdown": float(np.min(equity / peak - 1.0)),
        "hit_rate": float(np.mean(arr[arr != 0] > 0)) if np.any(arr != 0) else None,
    }


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
    train_labels = [sample.targets.get(target) for sample in train]
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
        if kind == "regression":
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
    # Too few blocks makes the percentile interval anti-conservative; refuse to
    # manufacture a CI from a handful of blocks.
    if len(blocks) < 8 or len(pooled_values) < 60:
        result["insufficient"] = True
        return result
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(bootstrap_samples), dtype=float)
    for index in range(int(bootstrap_samples)):
        picks = rng.integers(0, len(blocks), size=len(blocks))
        sampled = [blocks[pick] for pick in picks]
        means[index] = float(np.mean([value for block in sampled for value in block]))
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


def evaluate_control_ticker(
    ticker: str,
    config: Config,
    target: str,
    mode: str,
    *,
    folds: int,
    holdout_start: str | None,
    seed: int,
    shift_bars: int,
) -> dict[str, Any]:
    rows, _ = filter_rows_before(load_ticker_rows(ticker), holdout_start)
    samples = build_samples(rows, horizon=config.horizon, interval_seconds=INTERVAL_SECONDS)
    samples = transform_targets(
        samples, target, mode, seed=seed, shift_bars=shift_bars
    )
    splits = walk_forward(samples, folds=folds, horizon=config.horizon)
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
    reserve_final: float = 0.0,
    holdout_start: str | None = None,
    full_archive: bool = False,
    primary_target: str | None = None,
) -> dict[str, Any]:
    rows = load_ticker_rows(ticker)
    if len(rows) < 120:
        return {"ticker": ticker, "error": f"only {len(rows)} candles"}
    if holdout_start and not full_archive:
        rows, _ = filter_rows_before(rows, holdout_start)
    samples = build_samples(rows, horizon=config.horizon, interval_seconds=INTERVAL_SECONDS)
    if reserve_final > 0.0 and not holdout_start:
        keep = int(len(samples) * (1.0 - reserve_final))
        samples = samples[:keep]
    splits = walk_forward(samples, folds=folds, horizon=config.horizon)
    if not splits:
        return {"ticker": ticker, "error": "not enough samples for walk-forward"}
    target_records: dict[str, list[dict[str, Any]]] = {t: [] for t in targets}
    model_pos: list[float] = []
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
        m, mom, always, rets = _policy_rows(bundle, test, threshold=threshold, position=position)
        model_pos.extend(m)
        mom_pos.extend(mom)
        always_pos.extend(always)
        returns.extend(rets)
    aggregated = {
        target: agg
        for target, records in target_records.items()
        if (agg := aggregate_target(records)) is not None
    }
    economic = {
        "model": _pnl_metrics(model_pos, returns, fee_bps=fee_bps, slippage_bps=slippage_bps),
        "momentum": _pnl_metrics(mom_pos, returns, fee_bps=fee_bps, slippage_bps=slippage_bps),
        "always_long": _pnl_metrics(
            always_pos, returns, fee_bps=fee_bps, slippage_bps=slippage_bps
        ),
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
        "paired_target": paired_target,
        "paired": {
            "n": len(paired_values),
            "mean_improvement": float(np.mean(paired_values)) if paired_values else None,
        },
        "paired_rows": paired,
        "fold_improvements": fold_improvements,
    }


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


def evaluate_action(args: argparse.Namespace) -> int:
    config = Config(
        name=f"{args.feature_set}+{args.model_set}+h{args.horizon}",
        feature_names=resolve_feature_names(args.feature_set),
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
            holdout_start=args.holdout_start,
            full_archive=args.full_archive,
            primary_target=primary_target,
        )
        for ticker in tickers
    ]
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
        champion_bundle = train_tier1(
            samples[:split],
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
        payload["written"] = str(current_path)
        payload["warning"] = (
            "The scheduled train run will overwrite this bundle unless the default "
            "training config in pipeline/run.py is updated to the same feature/model set."
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
