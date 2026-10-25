#!/usr/bin/env python3
"""Diagnostic baseline report: model skill vs six causal volatility baselines.

This is a **diagnostic** tool, not an experiment. It never writes the ledger, a
bundle, or ``data/``; it prints (or writes) a JSON table and nothing else.

What it measures
----------------
For each ticker and each target variant, a reference model is trained
walk-forward and its per-row loss is compared with six baselines:

``constant``, ``seasonal``, ``persistence``, ``ewma``, ``har`` and the fitted
``har_rv`` supplied by :mod:`rl_crypto.baselines`. The paired per-row loss
improvement (baseline loss − model loss; positive means the model beat that
baseline) is pooled across folds and tickers, and the interval is the lab's
active **v3** rule: the effective-sample Student-t interval from
:func:`rl_crypto.power.effective_sample_t_interval_grouped`, which charges the
lag-1 autocorrelation through the effective sample size and never pools BTC /
ETH / SOL as independent rows.

A cell is labelled ``edge`` only when the interval's lower bound is above zero
**and** the interval is resolved (enough effective blocks). Point estimates
alone never earn the word.

Variants
--------
* ``fwd_volatility`` — the shipped label: realised vol over ``max(h, 2)`` bars.
* ``fwd_volatility_24`` — diagnostic only: realised vol over the next 24 bars.
  It is the shipped label *relabelled* in the sample set (same key) so the
  trainer and the baselines need no new target kind; the walk-forward purge is
  raised to 24 bars so the longer label cannot read into the test fold. It is
  never registered in ``TARGET_KINDS`` and consumes no ledger trial.

The tool also reports which baseline the *inner holdout* of the training slice
selects, with and without ``har_rv`` in the candidate set — the evidence that
fitted HAR-RV does (or does not) change the "strongest baseline".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rl_crypto.baselines import (
    DIAGNOSTIC_VOL_WINDOW,
    VOLATILITY_BASELINES,
    forward_volatility_window,
)
from rl_crypto.cache import candles_path, load_candles
from rl_crypto.features import FEATURE_NAMES
from rl_crypto.ml.tier1 import train_tier1
from rl_crypto.samples import Sample, build_samples
from rl_crypto.targets import TARGET_KINDS, TARGET_NAMES

ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location("model_lab", ROOT / "tools" / "model_lab.py")
assert _SPEC is not None and _SPEC.loader is not None
lab = importlib.util.module_from_spec(_SPEC)
sys.modules["model_lab"] = lab
_SPEC.loader.exec_module(lab)

DEFAULT_TICKERS: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD")
VARIANT_24 = "fwd_volatility_24"
ALPHA = 0.05
SEED = 0
BLOCK_BARS = 24
BOOTSTRAP_SAMPLES = 1000


def _variant_samples(rows: Sequence[Any], *, window: int | None) -> list[Sample]:
    """Samples with the shipped labels, or with ``fwd_volatility`` re-measured."""
    samples = list(build_samples(rows, horizon=1, interval_seconds=lab.INTERVAL_SECONDS))
    if window is None:
        return samples
    out: list[Sample] = []
    for index, sample in enumerate(samples):
        value = forward_volatility_window(
            rows, index, window, interval_seconds=lab.INTERVAL_SECONDS
        )
        out.append(
            Sample(
                timestamp=sample.timestamp,
                features=dict(sample.features),
                targets={**sample.targets, "fwd_volatility": value},
            )
        )
    return out


def _series(
    mode: str,
    kind: str,
    train: Sequence[Sample],
    test: Sequence[Sample],
    target: str,
    *,
    space: str,
    horizon: int,
) -> list[float]:
    return lab.baseline_predictions(
        mode, kind, train, test, target, space=space, horizon=horizon
    )


def _paired_rows(
    model: Any,
    kind: str,
    space: str,
    train: Sequence[Sample],
    test: Sequence[Sample],
    target: str,
    mode: str,
    *,
    horizon: int,
) -> list[dict[str, Any]]:
    """Per-row paired loss improvement of the model against one baseline."""
    predictions = _series(
        mode, kind, train, test, target, space=space, horizon=horizon
    )
    out: list[dict[str, Any]] = []
    for sample, baseline_prediction in zip(test, predictions, strict=True):
        truth = sample.targets.get(target)
        if truth is None:
            continue
        model_prediction = model.predict_row(sample.features)
        if model_prediction is None:
            continue
        model_loss = lab._row_loss(kind, float(truth), float(model_prediction), space)
        baseline_loss = lab._row_loss(kind, float(truth), float(baseline_prediction), space)
        if model_loss is None or baseline_loss is None:
            continue
        out.append(
            {
                "timestamp": sample.timestamp,
                "improvement": float(baseline_loss - model_loss),
                "model_loss": float(model_loss),
                "baseline_loss": float(baseline_loss),
            }
        )
    return out


def _inner_pick(
    kind: str,
    train: Sequence[Sample],
    target: str,
    *,
    space: str,
    horizon: int,
    modes: Sequence[str],
    holdout_fraction: float = 0.2,
) -> tuple[str, dict[str, float]]:
    """Baseline chosen on a causal inner holdout, over ``modes``."""
    if len(train) < 20:
        return "constant", {}
    cut = max(1, int(len(train) * (1.0 - holdout_fraction)))
    inner_train, inner_test = list(train[:cut]), list(train[cut:])
    if not inner_test:
        return "constant", {}
    losses: dict[str, float] = {}
    for mode in modes:
        predictions = _series(
            mode, kind, inner_train, inner_test, target, space=space, horizon=horizon
        )
        values = [
            lab._row_loss(kind, float(sample.targets[target]), float(prediction), space)
            for sample, prediction in zip(inner_test, predictions, strict=True)
            if sample.targets.get(target) is not None
        ]
        usable = [value for value in values if value is not None]
        losses[mode] = float(np.mean(usable)) if usable else math.inf
    return min(losses, key=lambda name: losses[name]), losses


def _summary(rows_by_ticker: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Pooled paired improvement with the v3 interval, plus point skill."""
    pooled = lab.bootstrap_pooled(
        rows_by_ticker,
        alpha=ALPHA,
        seed=SEED,
        block_bars=BLOCK_BARS,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
    )
    model_losses = [row["model_loss"] for rows in rows_by_ticker.values() for row in rows]
    baseline_losses = [
        row["baseline_loss"] for rows in rows_by_ticker.values() for row in rows
    ]
    mean_model = float(np.mean(model_losses)) if model_losses else None
    mean_baseline = float(np.mean(baseline_losses)) if baseline_losses else None
    relative_skill = (
        (mean_baseline - mean_model) / mean_baseline
        if mean_baseline not in (None, 0.0) and mean_model is not None
        else None
    )
    ci_low, ci_high = pooled.get("ci_low"), pooled.get("ci_high")
    resolved = bool(pooled.get("resolution_ok"))
    edge = bool(
        resolved
        and isinstance(ci_low, (int, float))
        and isinstance(ci_high, (int, float))
        and float(ci_low) > 0.0
    )
    return {
        "relative_skill": relative_skill,
        "mean_model_loss": mean_model,
        "mean_baseline_loss": mean_baseline,
        "mean_improvement": pooled.get("mean_improvement"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": pooled.get("alpha"),
        "interval": pooled.get("interval"),
        "n": pooled.get("n"),
        "eff_n": pooled.get("eff_n"),
        "df": pooled.get("df"),
        "rho": pooled.get("rho"),
        "resolution_ok": resolved,
        "resolution_reason": pooled.get("resolution_reason"),
        "edge": edge,
        "edge_rule": "ci_low > 0 and resolution_ok",
    }


def _record_variant(
    ticker: str,
    variant: str,
    samples: Sequence[Sample],
    *,
    folds: int,
    purge: int,
    pool: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    picks: dict[str, Any],
) -> dict[str, Any]:
    target = "fwd_volatility"
    kind = TARGET_KINDS[target]
    splits = lab.walk_forward(
        list(samples), folds=folds, horizon=1, targets=(target,), purge=purge
    )
    if not splits:
        return {"error": "not enough samples for walk-forward"}
    pick_counts: Counter[str] = Counter()
    pick_counts_lab: Counter[str] = Counter()
    for train, test in splits:
        bundle = train_tier1(
            train,
            horizon=1,
            feature_names=tuple(FEATURE_NAMES),
            targets=(target,),
            model_names={"fwd_volatility": lab.MODEL_SETS["reference"][target]},
            holdout_fraction=0.0,
        )
        model = bundle.targets.get(target)
        if model is None:
            continue
        space = str(getattr(model, "metric_space", "raw"))
        chosen, _ = _inner_pick(
            kind,
            train,
            target,
            space=space,
            horizon=1,
            modes=VOLATILITY_BASELINES,
        )
        chosen_lab = lab.choose_strongest_baseline(
            kind, train, target, space=space, horizon=1
        )
        pick_counts[chosen] += 1
        pick_counts_lab[chosen_lab] += 1
        for mode in VOLATILITY_BASELINES:
            rows = _paired_rows(
                model, kind, space, train, test, target, mode, horizon=1
            )
            pool[variant][mode].setdefault(ticker, []).extend(rows)
    picks[variant] = {
        "with_har_rv": dict(pick_counts),
        "lab_default_set": dict(pick_counts_lab),
        "folds": len(splits),
    }
    return {"folds": len(splits), "purge": purge}


def build_report(
    tickers: Sequence[str], *, folds: int
) -> dict[str, Any]:
    pool: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        variant: {mode: {} for mode in VOLATILITY_BASELINES}
        for variant in ("fwd_volatility", VARIANT_24)
    }
    picks: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for ticker in tickers:
        rows = load_candles(candles_path(ROOT, ticker), ticker=ticker)
        if len(rows) < 200:
            coverage[ticker] = {"error": f"only {len(rows)} candles"}
            continue
        coverage[ticker] = {"bars": len(rows)}
        standard = _variant_samples(rows, window=None)
        windowed = _variant_samples(rows, window=DIAGNOSTIC_VOL_WINDOW)
        for variant, samples, purge in (
            ("fwd_volatility", standard, 2),
            (VARIANT_24, windowed, DIAGNOSTIC_VOL_WINDOW),
        ):
            note = _record_variant(
                ticker,
                variant,
                samples,
                folds=folds,
                purge=purge,
                pool=pool,
                picks=picks,
            )
            coverage[ticker][variant] = note

    report: dict[str, Any] = {
        "task": "T3",
        "kind": "diagnostic",
        "ledger_trial": False,
        "note": (
            "Diagnostic only. No ledger entry, no bundle, no data/ write. "
            "fwd_volatility_24 is not registered in TARGET_KINDS."
        ),
        "alpha": ALPHA,
        "interval": "effective-sample-t (v3)",
        "block_bars": BLOCK_BARS,
        "folds": folds,
        "tickers": list(tickers),
        "baselines": list(VOLATILITY_BASELINES),
        "coverage": coverage,
        "strongest_baseline_picks": picks,
        "variants": {},
    }
    for variant, by_mode in pool.items():
        variant_block: dict[str, Any] = {"pooled": {}, "per_ticker": {}}
        for mode in VOLATILITY_BASELINES:
            rows_by_ticker = by_mode[mode]
            variant_block["pooled"][mode] = _summary(rows_by_ticker)
            variant_block["per_ticker"][mode] = {
                ticker: _summary({ticker: values})
                for ticker, values in rows_by_ticker.items()
            }
        report["variants"][variant] = variant_block
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="comma- or space-separated tickers (default BTC/ETH/SOL)",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args(argv)
    tickers = tuple(
        part for part in str(args.tickers).replace(",", " ").split() if part.strip()
    )
    unknown = [t for t in tickers if t not in TARGET_NAMES and "/" not in t]
    if unknown:
        parser.error(f"unknown tickers: {unknown}")
    report = build_report(tickers, folds=max(2, int(args.folds)))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
