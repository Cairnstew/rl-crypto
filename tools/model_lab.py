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
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rl_crypto.backtest import pnl_metrics
from rl_crypto.cache import candles_path, load_candles, slug_for, summary_path
from rl_crypto.experiments import (
    check_promote_allowed,
    config_hash,
    ledger_path,
    read_entries,
    record_holdout,
    write_scoreboard,
)
from rl_crypto.experiments import (
    counts as experiment_counts,
)
from rl_crypto.experiments import (
    preregister as ledger_preregister,
)
from rl_crypto.features import (
    DEFAULT_INTERVAL_SECONDS,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_VERSION,
)
from rl_crypto.ml.base import DEFAULT_MODEL
from rl_crypto.ml.store import default_bundle_path, load_bundle, save_bundle
from rl_crypto.ml.tier1 import (
    DEFAULT_MODEL_BY_TARGET,
    Tier1Bundle,
    baseline_for,
    beats_baseline,
    classification_metrics,
    regression_metrics,
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

#: The estimator production actually ships for each target: the trainer's
#: per-target override, else the default estimator for the task kind. The lab's
#: ``reference`` set is derived from this, not hand-written, so a lab verdict is
#: comparable to the deployed bundle (notably ``log_ridge`` for volatility).
PRODUCTION_MODELS: dict[str, str] = {
    target: DEFAULT_MODEL_BY_TARGET.get(target, DEFAULT_MODEL[TARGET_KINDS[target]])
    for target in TARGET_NAMES
}

MODEL_SETS: dict[str, dict[str, str]] = {
    "reference": dict(PRODUCTION_MODELS),
    "production": dict(PRODUCTION_MODELS),
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
    targets: Sequence[str] = (),
    purge: int | None = None,
    min_train_fraction: float = 0.34,
    min_test: int = 20,
) -> list[tuple[tuple[Sample, ...], tuple[Sample, ...]]]:
    """Expanding-window walk-forward splits with a label-reach purge.

    A training label anchored at row ``i`` reads up to ``i + horizon``, and
    ``fwd_volatility`` reads ``max(horizon, MIN_VOL_SPAN)``. The purge therefore
    defaults to the largest reach among ``targets`` so the last training label
    never reads a test bar; passing ``horizon`` alone leaks one bar for
    ``fwd_volatility`` at the default horizon.
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


def _policy_components(
    bundle: Tier1Bundle,
    test: Sequence[Sample],
    *,
    position: str,
) -> tuple[list[float | None], list[float | None], list[float], list[float], list[float]]:
    """Predictions and realised returns, computed once per fold.

    The direction probability is evaluated once here and reused for every
    threshold in the cost grid, so a grid does not refit or re-predict per cell.
    A missing prediction or target is skipped, never imputed.
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
        pos = 0.0
        if probability is not None:
            if probability >= threshold:
                pos = 1.0
            elif position == "long-short" and probability <= 1.0 - threshold:
                pos = -1.0
        elif prediction is not None:
            pos = 1.0 if prediction > 0 else (-1.0 if position == "long-short" else 0.0)
        out.append(pos)
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
    """Per-side-cost economics, from the shared backtest implementation."""
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


def _unique(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


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
) -> dict[str, Any]:
    """Walk-forward metrics plus a threshold x cost policy grid.

    Each fold is fitted exactly once; the direction probability is computed once
    per fold and re-thresholded for every grid cell. The headline ``economic``
    block is the cell matching ``threshold``/``fee_bps``/``slippage_bps``.
    """
    threshold_list = _unique(list(thresholds or [threshold]))
    cost_list = _unique(list(cost_levels or [(fee_bps, slippage_bps)]))
    rows = load_ticker_rows(ticker)
    if len(rows) < 120:
        return {"ticker": ticker, "error": f"only {len(rows)} candles"}
    samples = build_samples(rows, horizon=config.horizon, interval_seconds=INTERVAL_SECONDS)
    if reserve_final > 0.0:
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
            row
            for row in grid
            if row["threshold"] == threshold
            and row["fee_bps"] == fee_bps
            and row["slippage_bps"] == slippage_bps
        ),
        grid[0],
    )
    economic = {
        "model": headline["model"],
        "momentum": headline["momentum"],
        "always_long": headline["always_long"],
    }
    return {
        "ticker": ticker,
        "config": config.as_dict(),
        "folds": len(splits),
        "samples": len(samples),
        "reserved_final": reserve_final,
        "span": sample_span(samples),
        "targets": aggregated,
        "economic": economic,
        "economic_grid": grid,
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


def evaluate_action(args: argparse.Namespace) -> int:
    config = Config(
        name=f"{args.feature_set}+{args.model_set}+h{args.horizon}",
        feature_names=resolve_feature_names(args.feature_set),
        model_names=resolve_model_set(args.model_set),
        horizon=args.horizon,
    )
    tickers = resolve_tickers(args.tickers)
    targets = resolve_targets(args.targets)
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
        )
        for ticker in tickers
    ]
    payload = {
        "action": "evaluate",
        "config": config.as_dict(),
        "targets": list(targets),
        "policy_grid": {
            "thresholds": thresholds or [args.threshold],
            "cost_levels": cost_levels or [(args.fee_bps, args.slippage_bps)],
        },
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
    grid_rows = [
        [r["ticker"], cell["threshold"], cell["fee_bps"], cell["slippage_bps"],
         f"{cell['model']['sharpe']:.2f}"
         if cell["model"]["sharpe"] is not None
         else "—",
         f"{100 * cell['model']['total_return']:+.2f}%"
         if cell["model"]["total_return"] is not None
         else "—",
         cell["model"]["trades"],
         f"{cell['momentum']['sharpe']:.2f}"
         if cell["momentum"]["sharpe"] is not None
         else "—"]
        for r in results
        if "error" not in r
        for cell in r.get("economic_grid", [])
    ]
    if len(grid_rows) > len([r for r in results if "error" not in r]):
        print("\nthreshold x cost grid (same folds, fitted once per fold):")
        _print_table(
            ["ticker", "thr", "fee", "slip", "sharpe", "ret", "trades", "mom"],
            grid_rows,
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
    primary = "fwd_direction" if "fwd_direction" in targets else targets[0]
    configs = _configs(args)
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
            )
            result["rank"] = _rank_key(result, primary)
            results.append(result)
    ranked = sorted(results, key=lambda r: r.get("rank", ()), reverse=True)
    top = [r for r in ranked if r.get("rank", (0,))[0]][: args.limit]
    candidates: list[str] = []
    for result in top:
        ticker = result["ticker"]
        config_name = result["config"]["name"]
        rows = load_ticker_rows(ticker)
        samples = build_samples(
            rows, horizon=result["config"]["horizon"], interval_seconds=INTERVAL_SECONDS
        )
        split = int(len(samples) * 0.8)
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
        "configs": [c.as_dict() for c in configs],
        "results": results,
        "top": top,
        "candidates": candidates,
        "summary": {
            "tickers": len(tickers),
            "configs": len(configs),
            "runs": len(results),
            "gated": sum(1 for r in results if r.get("rank", (0,))[0]),
            "candidates": len(candidates),
            "reserved_final": 0.2,
        },
    }
    path = _save("sweep", payload)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print("sweep evaluates the first 80%; the final 20% is reserved for promote.")
    if not top:
        print("No config beat the baseline consistently across folds.")
        print(
            "That is a valid result: the current data/features do not support a directional edge."
        )
        print(f"artifact: {path}")
        return 0
    _print_table(
        ["ticker", "config", "beats", "min%", "mean%", "sharpe", "ret", "dd", "candidate"],
        [
            [
                r["ticker"],
                r["config"]["name"],
                f"{r['targets'][primary]['folds_beating']}/{r['targets'][primary]['folds']}",
                f"{100 * (r['targets'][primary]['min_improvement'] or 0):+.2f}",
                f"{100 * (r['targets'][primary]['mean_improvement'] or 0):+.2f}",
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
    hypothesis = args.hypothesis or "adhoc"
    lineage = args.lineage or f"{hypothesis}:{ticker}"
    candidate_hash = config_hash(
        {
            "feature_names": list(candidate.feature_names),
            "horizon": candidate.horizon,
            "models": {
                name: model.model.to_dict().get("kind")
                for name, model in candidate.targets.items()
            },
        }
    )
    refusal = check_promote_allowed(
        read_entries(ROOT),
        lineage_id=lineage,
        candidate_hash=candidate_hash,
        apply=bool(args.apply),
    )
    if refusal:
        # Refuse before evaluating anything: the final holdout is the only
        # untouched data left, so it must not be read speculatively.
        raise SystemExit(f"promote refused: {refusal}")
    rows = load_ticker_rows(ticker)
    if len(rows) < 120:
        raise SystemExit(f"{ticker}: only {len(rows)} candles")
    samples = build_samples(rows, horizon=candidate.horizon, interval_seconds=INTERVAL_SECONDS)
    split = int(len(samples) * 0.8)
    if split < 100:
        raise SystemExit(f"{ticker}: not enough samples for a final holdout ({split})")
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
    cur_view = _bundle_view(current, samples, targets, **common) if current is not None else None
    primary = "fwd_direction" if "fwd_direction" in targets else targets[0]
    reasons = promotion_gate(cand_view, cur_view, primary)
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
        "hypothesis": hypothesis,
        "lineage": lineage,
        "candidate_hash": candidate_hash,
        "candidate": str(candidate_path),
        "current_bundle": str(default_bundle_path(ticker, directory=ROOT / "models")),
        "apply": bool(args.apply),
        "primary": primary,
        "candidate_view": cand_view,
        "current_view": cur_view,
        "recommendation": recommendation,
        "conservative_note": (
            "The current bundle may have been trained on data that includes this "
            "holdout, so the gate is conservative in the candidate's favour."
        ),
        "reasons": reasons,
        "passed": not reasons,
        "summary": {"passed": not reasons, "ticker": ticker, "reasons": reasons},
    }
    # The holdout is spent here, whether the gate passes or fails: this is the
    # touch the ledger exists to bound. Recorded before `--apply` so an apply
    # can never happen without its dry-run record.
    record_holdout(
        ROOT,
        hypothesis_id=hypothesis,
        lineage_id=lineage,
        candidate=str(candidate_path),
        candidate_hash=candidate_hash,
        passed=not reasons,
        apply=bool(args.apply),
        reasons=reasons,
    )
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
        # The bundle alone is overwritten by the next scheduled retrain; the
        # committed config is what makes the promotion durable.
        payload["training_config"] = str(
            apply_promotion(
                ROOT,
                ticker=ticker,
                feature_names=candidate.feature_names,
                model_names=recommendation["models"],
                horizon=candidate.horizon,
            )
        )
        payload["warning"] = (
            "models/<slug>.json is rebuilt by the next scheduled train; the "
            "committed config/training.json now reproduces this feature/model set."
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
    print("note:      the current bundle may have seen this holdout; the gate is conservative")
    if args.apply:
        print(f"applied:   {'yes' if not reasons else 'no (gate failed)'}")
        if payload.get("written"):
            print(f"written:   {payload['written']}")
            print(f"warning:   {payload['warning']}")
    else:
        print("dry run:   pass --apply to write models/<slug>.json after a passing gate")
    print(f"artifact:  {path}")
    return 0


def _parse_configs(value: str) -> list[dict[str, Any]]:
    """Parse ``--configs`` as one JSON object or a list of them."""
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--configs must be JSON: {exc}") from exc
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise SystemExit("--configs must be a JSON object or a list of objects")
    return payload


def preregister_action(args: argparse.Namespace) -> int:
    entry = ledger_preregister(
        ROOT,
        hypothesis_id=args.hypothesis,
        family=args.family,
        lineage_id=args.lineage
        or f"{args.hypothesis}:{','.join(resolve_tickers(args.tickers))}",
        mechanism=args.mechanism,
        configs=_parse_configs(args.configs),
        tickers=list(resolve_tickers(args.tickers)),
        targets=list(resolve_targets(args.targets)),
        metric=args.metric,
        criteria=args.criteria,
        negative=args.negative,
        seed=args.seed,
        folds=args.folds,
    )
    if args.json:
        print(json.dumps(entry, indent=2, default=str))
        return 0
    print(f"preregistered {entry['hypothesis_id']} ({entry['lineage_id']})")
    print(f"  family:      {entry['family']}  trial {entry['trial_index']}")
    print(f"  git:         {entry['git_sha'] or 'unknown'}")
    print(f"  config hash: {entry['config_hash'][:16]}")
    print(f"  data:        {json.dumps(entry['data_hash'], default=str)}")
    print(f"  ledger:      {ledger_path(ROOT)}")
    return 0


def scoreboard_action(args: argparse.Namespace) -> int:
    path = write_scoreboard(ROOT)
    totals = experiment_counts(read_entries(ROOT))
    if args.json:
        print(json.dumps({"scoreboard": str(path), "counts": totals}, indent=2, default=str))
        return 0
    print(f"scoreboard: {path}")
    print(
        f"  entries={totals['entries']} lineages={totals['lineages']} "
        f"final-holdout touches={totals['holdout_touches']}"
    )
    return 0


def report_action(args: argparse.Namespace) -> int:
    path = ledger_path(ROOT)
    entries = read_entries(ROOT)
    totals = experiment_counts(entries)
    rows = entries[-args.limit :]
    payload = {
        "action": "report",
        "ledger": str(path),
        "counts": totals,
        "entries": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if not entries:
        print(f"no experiment ledger entries yet ({path})")
        return 0
    print(
        f"experiments: {totals['entries']} entries, {totals['lineages']} lineages, "
        f"{totals['holdout_touches']} final-holdout touch(es)"
    )
    for family, bucket in sorted(totals["families"].items()):
        print(
            f"  {family}: trials={bucket['trials']} results={bucket['results']} "
            f"holdouts={bucket['holdouts']}"
        )
    print()
    _print_table(
        ["at", "kind", "lineage", "trial", "verdict", "holdout"],
        [
            [
                r.get("at", ""),
                r.get("kind", ""),
                r.get("lineage_id", ""),
                r.get("trial_index", ""),
                r.get("verdict", "—"),
                "used" if r.get("holdout_touched") else "—",
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
    payload = {
        "action": "status",
        "tickers": status_payload(resolve_tickers(args.tickers)),
        "experiments": experiment_counts(read_entries(ROOT)),
    }
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
    totals = payload["experiments"]
    print(
        f"\nexperiments: {totals['entries']} ledger entries, "
        f"{totals['lineages']} lineages, "
        f"{totals['holdout_touches']} final-holdout touch(es)"
    )
    return 0


HELP_TEXT = """\
rl-model-lab — offline, out-of-sample model loop

Actions:
  status       deployed bundles, trust, freshness
  evaluate     walk-forward target metrics + after-cost policy
  sweep        controlled feature/model/horizon experiments
  preregister  record a hypothesis + pass/fail criteria before it runs
  promote      validate a candidate on the reserved final holdout
  report       experiment ledger with trial/holdout counts
  scoreboard   regenerate experiments/SCOREBOARD.md from the ledger
  help         this reference

Common arguments (evaluate / sweep / promote):
  --tickers BTC/USD ETH/USD     comma- or space-separated; default all
                                (sweep default: BTC/USD, ETH/USD, SOL/USD)
  --targets fwd_direction ...   default all four targets
  --folds N                     walk-forward folds (default 4)
  --threshold P                 direction probability for a long (default 0.55)
  --position long-only|long-short
  --fee-bps N                   per-side fee (default 10)
  --slippage-bps N              per-side slippage (default 5)

Action-specific:
  status    --tickers, --json
  evaluate  --feature-set all|market|core|price (default all)
            --model-set reference|baseline (default reference)
            --horizon N (default 1; the economic proxy is only honest at 1)
            --thresholds 0.5,0.55,0.6   one fit, many policy thresholds
            --cost-levels 10:5,0:0      one fit, many cost levels
            --limit N (default 20)
  sweep     --feature-sets all,market,core,price
            --model-sets reference,baseline
            --horizons 1
            --max-configs N (default 16)
            --limit N (default 20)
  preregister --hypothesis H1 --family pooled --lineage H1:BTC/USD
              --mechanism "why this should help" --configs JSON
              --metric "metric of record" --criteria "numeric pass/fail"
              --negative "what would count as a negative finding"
  promote   --candidate PATH (required)
            --ticker BTC/USD (inferred from the filename when possible)
            --hypothesis ID --lineage ID (default: <hypothesis>:<ticker>)
            --apply (write models/<slug>.json only after a passing gate)
  report    --limit N (default 10)
  scoreboard

Every action accepts --json. Artifacts live under
~/.local/share/rl-crypto/model-lab (override RL_MODEL_LAB_DIR); the committed
experiment ledger is experiments/ledger.jsonl. The final holdout is spent at
most once per lineage, and only by promote.
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
        p.add_argument("--folds", type=int, default=4)
        p.add_argument("--threshold", type=float, default=0.55)
        p.add_argument("--position", choices=("long-only", "long-short"), default="long-only")
        p.add_argument("--fee-bps", type=float, default=10.0)
        p.add_argument("--slippage-bps", type=float, default=5.0)
        p.add_argument("--json", action="store_true")
        p.add_argument("--limit", type=int, default=20)

    evaluate = sub.add_parser("evaluate", help="walk-forward metrics + economic proxy")
    common(evaluate)
    evaluate.add_argument("--feature-set", default="all")
    evaluate.add_argument("--model-set", default="reference")
    evaluate.add_argument("--horizon", type=int, default=1)
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
    sweep.set_defaults(func=sweep_action)

    promote = sub.add_parser("promote", help="validate a candidate on the final holdout")
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--ticker", default=None)
    promote.add_argument(
        "--hypothesis", default=None, help="hypothesis id the candidate belongs to"
    )
    promote.add_argument(
        "--lineage",
        default=None,
        help="candidate lineage whose holdout budget this spends (default: <hypothesis>:<ticker>)",
    )
    promote.add_argument("--tickers", nargs="+", default=None)
    promote.add_argument("--targets", nargs="+", default=None)

    promote.add_argument("--threshold", type=float, default=0.55)
    promote.add_argument("--position", choices=("long-only", "long-short"), default="long-only")
    promote.add_argument("--fee-bps", type=float, default=10.0)
    promote.add_argument("--slippage-bps", type=float, default=5.0)
    promote.add_argument("--apply", action="store_true")
    promote.add_argument("--json", action="store_true")
    promote.set_defaults(func=promote_action)

    preregister = sub.add_parser(
        "preregister", help="record a hypothesis and its pass/fail criteria before running"
    )
    preregister.add_argument("--hypothesis", required=True)
    preregister.add_argument("--family", required=True)
    preregister.add_argument("--lineage", default=None)
    preregister.add_argument("--mechanism", required=True)
    preregister.add_argument("--configs", required=True, help="JSON object or list of objects")
    preregister.add_argument("--tickers", nargs="+", default=None)
    preregister.add_argument("--targets", nargs="+", default=None)
    preregister.add_argument("--metric", required=True)
    preregister.add_argument("--criteria", required=True)
    preregister.add_argument("--negative", required=True)
    preregister.add_argument("--seed", type=int, default=0)
    preregister.add_argument("--folds", type=int, default=4)
    preregister.add_argument("--json", action="store_true")
    preregister.set_defaults(func=preregister_action)

    report = sub.add_parser("report", help="summarize the experiment ledger")
    report.add_argument("--limit", type=int, default=10)
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=report_action)

    scoreboard = sub.add_parser(
        "scoreboard", help="regenerate experiments/SCOREBOARD.md from the ledger"
    )
    scoreboard.add_argument("--json", action="store_true")
    scoreboard.set_defaults(func=scoreboard_action)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
