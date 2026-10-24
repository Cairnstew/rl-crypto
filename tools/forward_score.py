#!/usr/bin/env python3
"""Forward score — realised skill of the predictions that were actually served.

A backtest can always be tuned against the data it scores. This cannot: each
live record stores the prediction *before* the outcome existed, so joining it to
the realised bar once the horizon has elapsed is an honest, unfakeable test.

For every ticker it reports, per deployed bundle version:

* resolved predictions vs unresolved (horizon not elapsed) and unjoined (the
  feature bar is no longer in the archive),
* per-target skill against a constant predictor fitted only on *earlier*
  realised outcomes (never on the row being scored),
* after-cost policy PnL against a momentum baseline,
* whether the pre-registered forward window has been met.

Nothing is imputed: a missing prediction or an unresolved outcome is skipped
and counted, not filled.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from rl_crypto.backtest import pnl_metrics
from rl_crypto.cache import candles_path, live_path, load_candles, read_jsonl
from rl_crypto.dataset import parse_iso
from rl_crypto.features import DEFAULT_INTERVAL_SECONDS
from rl_crypto.ml.store import default_bundle_path, load_bundle
from rl_crypto.targets import TARGET_KINDS, TARGET_NAMES, compute_targets

ROOT = Path.cwd().resolve()

#: Phase V defaults: a candidate is confirmed only after this much forward data.
MIN_FORWARD_DAYS = 14
MIN_RESOLVED = 300


# --------------------------------------------------------------------------- inputs


def ticker_for_slug(slug: str) -> str:
    return slug.replace("-", "/").upper()


def resolve_tickers(root: Path, values: Sequence[str] | None) -> tuple[str, ...]:
    out: list[str] = []
    for value in values or ():
        out.extend(part for part in str(value).replace(",", " ").split() if part.strip())
    if out:
        return tuple(dict.fromkeys(out))
    data = root / "data"
    if not data.is_dir():
        return ()
    return tuple(
        ticker_for_slug(path.name)
        for path in sorted(data.iterdir())
        if path.is_dir() and (path / "live.jsonl").is_file()
    )


def load_records(root: Path, ticker: str) -> list[dict[str, Any]]:
    path = live_path(root, ticker)
    if not path.is_file():
        return []
    return [record for record in read_jsonl(path) if isinstance(record, dict)]


def _metadata_ok(predictions: Mapping[str, Any]) -> bool:
    """Whether a record names the bundle's metric space and feature contract.

    Old records predate ``metric_space``; a forward confirmation window built
    from them would mix scales (a log-space volatility prediction scored as a
    raw number) and cannot prove the deployed code produced them. They are
    still scored, but excluded from the confirmation window and counted.
    """
    metric_space = predictions.get("metric_space")
    feature_version = predictions.get("feature_version")
    return (
        isinstance(metric_space, Mapping)
        and bool(metric_space)
        and isinstance(feature_version, str)
        and bool(feature_version)
    )


def _index_rows(rows: Sequence[Any]) -> dict[Any, int]:
    index: dict[Any, int] = {}
    for position, row in enumerate(rows):
        stamp = parse_iso(row.timestamp)
        if stamp is not None:
            index[stamp] = position
    return index


# --------------------------------------------------------------------------- scoring


def _metric(kind: str, actual: Sequence[float], predicted: Sequence[float]) -> float | None:
    if not actual:
        return None
    if kind == "regression":
        errors = [a - p for a, p in zip(actual, predicted, strict=True)]
        return math.sqrt(sum(error * error for error in errors) / len(errors))
    total = 0.0
    for a, p in zip(actual, predicted, strict=True):
        probability = min(max(p, 1e-9), 1.0 - 1e-9)
        total -= a * math.log(probability) + (1.0 - a) * math.log(1.0 - probability)
    return total / len(actual)


def _target_rows(
    target: str,
    observations: Sequence[tuple[float, float, float | None]],
    *,
    space: str = "raw",
) -> dict[str, Any]:
    """Skill of the served prediction against a strictly-prior constant baseline.

    ``observations`` is chronological ``(actual, point, probability)``. The
    baseline for row ``i`` is fitted on rows ``< i`` only, so scoring a row never
    borrows that row's outcome.

    ``space`` comes from the deployed bundle's metadata. A log-space estimator
    (``fwd_volatility``) minimises squared error on ``log(y)``, so both the
    served raw prediction and the realised raw target are scored as logs; a
    real-scale comparison would report a loss the model never optimised and a
    baseline it never had. A non-positive value has no log and is skipped, not
    floored — the same rule the trainer applies.
    """
    if space == "log":
        usable = [
            (truth, point if probability is None else probability)
            for truth, point, probability in observations
            if truth > 0.0 and (point > 0.0 or (probability is not None and probability > 0.0))
        ]
    else:
        usable = [
            (truth, probability if probability is not None else point)
            for truth, point, probability in observations
        ]
    actual: list[float] = []
    predicted: list[float] = []
    baseline: list[float] = []
    for index, (truth, guess) in enumerate(usable):
        prior = [value for value, _ in usable[:index]]
        if not prior:
            continue
        if space == "log":
            actual.append(math.log(truth))
            predicted.append(math.log(guess))
            baseline.append(sum(math.log(value) for value in prior) / len(prior))
        else:
            actual.append(truth)
            predicted.append(guess)
            baseline.append(sum(prior) / len(prior))
    if not actual:
        return {
            "n": 0,
            "kind": TARGET_KINDS[target],
            "metric_space": space,
            "model": None,
            "baseline": None,
            "skill": None,
        }
    kind = TARGET_KINDS[target]
    model_metric = _metric(kind, actual, predicted)
    baseline_metric = _metric(kind, actual, baseline)
    skill = None
    if model_metric is not None and baseline_metric not in (None, 0):
        skill = (baseline_metric - model_metric) / abs(baseline_metric)
    return {
        "n": len(actual),
        "kind": kind,
        "metric_space": space,
        "model": model_metric,
        "baseline": baseline_metric,
        "skill": skill,
    }


def _metric_spaces(root: Path, ticker: str) -> dict[str, str]:
    """Read per-target metric scales from the deployed bundle's metadata.

    The served value is always on the target's natural scale (a log-space
    volatility model predicts ``exp`` back to a positive spread). The bundle
    records the scale its metrics were trained on, and the forward test must use
    the same one so a forward number and a backtest verdict are comparable.
    """
    try:
        bundle = load_bundle(default_bundle_path(ticker, directory=root / "models"))
    except (FileNotFoundError, ValueError, KeyError):
        return {}
    spaces: dict[str, str] = {}
    for name, model in bundle.targets.items():
        spaces[name] = str(getattr(model, "metric_space", "raw") or "raw")
    return spaces


def _position(record: Mapping[str, Any], threshold: float, position: str) -> float | None:
    predictions = record.get("predictions") or {}
    probabilities = predictions.get("probabilities") or {}
    probability = probabilities.get("fwd_direction")
    if isinstance(probability, int | float):
        if probability >= threshold:
            return 1.0
        if position == "long-short" and probability <= 1.0 - threshold:
            return -1.0
        return 0.0
    expected = (predictions.get("predictions") or {}).get("fwd_return")
    if isinstance(expected, int | float):
        return 1.0 if expected > 0 else (-1.0 if position == "long-short" else 0.0)
    return None


def _window_days(stamps: Sequence[Any]) -> float:
    parsed = [parse_iso(stamp) for stamp in stamps]
    parsed = [stamp for stamp in parsed if isinstance(stamp, datetime)]
    if len(parsed) < 2:
        return 0.0
    return (max(parsed) - min(parsed)).total_seconds() / 86_400.0


def score_ticker(
    ticker: str,
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Any],
    *,
    threshold: float = 0.55,
    position: str = "long-only",
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    root: Path | None = None,
) -> dict[str, Any]:
    """Join served predictions to realised targets and score them by bundle."""
    index = _index_rows(rows)
    spaces = _metric_spaces(root or ROOT, ticker)
    observations: dict[str, dict[str, list[tuple[float, float, float | None]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    positions: dict[str, list[float]] = defaultdict(list)
    momentum: dict[str, list[float]] = defaultdict(list)
    returns: dict[str, list[float]] = defaultdict(list)
    stamps: dict[str, list[str]] = defaultdict(list)
    counts = {
        "records": len(records),
        "resolved": 0,
        "unresolved": 0,
        "unjoined": 0,
        "no_prediction": 0,
        "metadata_missing": 0,
    }
    resolved_by_bundle: dict[str, int] = defaultdict(int)
    confirmation_resolved_by_bundle: dict[str, int] = defaultdict(int)
    confirmation_stamps: dict[str, list[str]] = defaultdict(list)
    skipped_positions = 0

    for record in records:
        predictions = record.get("predictions") or {}
        feature_timestamp = predictions.get("feature_timestamp")
        if not isinstance(feature_timestamp, str):
            # Records written before the prediction payload existed (or where
            # every model was withheld) are not a join failure.
            counts["no_prediction"] += 1
            continue
        stamp = parse_iso(feature_timestamp)
        row_index = index.get(stamp)
        if row_index is None:
            counts["unjoined"] += 1
            continue
        horizon = int(predictions.get("horizon") or 1)
        realised = compute_targets(
            rows, row_index, horizon=horizon, interval_seconds=interval_seconds
        )
        if all(value is None for value in realised.values()):
            counts["unresolved"] += 1
            continue

        counts["resolved"] += 1
        bundle = str(predictions.get("bundle_trained_at") or "unknown")
        resolved_by_bundle[bundle] += 1
        stamps[bundle].append(feature_timestamp)
        if _metadata_ok(predictions):
            confirmation_resolved_by_bundle[bundle] += 1
            confirmation_stamps[bundle].append(feature_timestamp)
        else:
            counts["metadata_missing"] += 1
        points = predictions.get("predictions") or {}
        probabilities = predictions.get("probabilities") or {}
        for target in TARGET_NAMES:
            truth = realised.get(target)
            point = points.get(target)
            if truth is None or not isinstance(point, int | float):
                continue
            probability = probabilities.get(target)
            observations[bundle][target].append(
                (
                    float(truth),
                    float(point),
                    float(probability) if isinstance(probability, int | float) else None,
                )
            )

        fwd_return = realised.get("fwd_return")
        if fwd_return is None:
            continue
        held = _position(record, threshold, position)
        if held is None:
            skipped_positions += 1
            continue
        ret1 = (predictions.get("features") or {}).get("ret_1")
        positions[bundle].append(held)
        momentum[bundle].append(
            (1.0 if isinstance(ret1, int | float) and ret1 > 0 else -1.0)
            if position == "long-short"
            else (1.0 if isinstance(ret1, int | float) and ret1 > 0 else 0.0)
        )
        returns[bundle].append(float(fwd_return))

    bundles: list[dict[str, Any]] = []
    for bundle in sorted(set(observations) | set(positions) | set(resolved_by_bundle)):
        targets = {
            target: _target_rows(target, series, space=spaces.get(target, "raw"))
            for target, series in sorted(observations.get(bundle, {}).items())
        }
        days = _window_days(stamps.get(bundle, []))
        resolved = resolved_by_bundle.get(bundle, 0)
        confirmation_days = _window_days(confirmation_stamps.get(bundle, []))
        confirmation_resolved = confirmation_resolved_by_bundle.get(bundle, 0)
        bundles.append(
            {
                "bundle": bundle,
                "resolved": resolved,
                "window_days": round(days, 2),
                "confirmation_resolved": confirmation_resolved,
                "confirmation_window_days": round(confirmation_days, 2),
                "excluded_missing_metadata": resolved - confirmation_resolved,
                "targets": targets,
                "economic": {
                    "model": pnl_metrics(
                        positions.get(bundle, []),
                        returns.get(bundle, []),
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                    ),
                    "momentum": pnl_metrics(
                        momentum.get(bundle, []),
                        returns.get(bundle, []),
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                    ),
                },
                "window_met": (
                    confirmation_days >= MIN_FORWARD_DAYS
                    and confirmation_resolved >= MIN_RESOLVED
                ),
            }
        )

    return {
        "ticker": ticker,
        "counts": counts,
        "skipped_positions": skipped_positions,
        "bundles": bundles,
    }


def forward_score(
    root: Path,
    *,
    tickers: Sequence[str] | None = None,
    threshold: float = 0.55,
    position: str = "long-only",
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    results = []
    for ticker in resolve_tickers(root, tickers):
        records = load_records(root, ticker)
        if not records:
            results.append({"ticker": ticker, "error": "no live records"})
            continue
        try:
            rows = load_candles(candles_path(root, ticker), ticker=ticker)
        except OSError as exc:
            results.append({"ticker": ticker, "error": f"archive unreadable: {exc}"})
            continue
        results.append(
            score_ticker(
                ticker,
                records,
                rows,
                threshold=threshold,
                position=position,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                interval_seconds=interval_seconds,
                root=root,
            )
        )
    payload = {
        "action": "forward-score",
        "threshold": threshold,
        "position": position,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "min_forward_days": MIN_FORWARD_DAYS,
        "min_resolved": MIN_RESOLVED,
        "results": results,
        "summary": {
            "tickers": len(results),
            "errors": sum(1 for result in results if "error" in result),
            "resolved": sum(result.get("counts", {}).get("resolved", 0) for result in results),
            "metadata_missing": sum(
                result.get("counts", {}).get("metadata_missing", 0)
                for result in results
            ),
            "window_met": sum(
                1
                for result in results
                for bundle in result.get("bundles", [])
                if bundle["window_met"]
            ),
        },
    }
    return payload


# --------------------------------------------------------------------------- output


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def render(payload: Mapping[str, Any]) -> None:
    rows = []
    for result in payload["results"]:
        if "error" in result:
            rows.append([result["ticker"], "—", "—", "—", "—", "—", "—", result["error"]])
            continue
        for bundle in result["bundles"]:
            for target, score in bundle["targets"].items():
                economy = bundle["economic"]["model"]
                rows.append(
                    [
                        result["ticker"],
                        bundle["bundle"][:19],
                        target,
                        score["n"],
                        f"{score['model']:.5f}" if score["model"] is not None else "—",
                        f"{score['baseline']:.5f}" if score["baseline"] is not None else "—",
                        f"{100 * score['skill']:+.2f}%" if score["skill"] is not None else "—",
                        f"{economy['trades']} tr "
                        f"{100 * economy['total_return']:+.2f}%"
                        if economy["total_return"] is not None
                        else "—",
                    ]
                )
    if not rows:
        print("no resolved forward predictions yet")
        return
    _print_table(
        ["ticker", "bundle", "target", "n", "model", "baseline", "skill", "after-cost"],
        rows,
    )
    print()
    for result in payload["results"]:
        if "error" in result:
            continue
        counts = result["counts"]
        print(
            f"{result['ticker']}: {counts['resolved']} resolved, "
            f"{counts['unresolved']} pending, {counts['unjoined']} unjoined, "
            f"{counts.get('no_prediction', 0)} without a prediction, "
            f"{counts.get('metadata_missing', 0)} excluded from confirmation "
            "(missing metric_space/feature_version)"
        )
        for bundle in result["bundles"]:
            state = "window met" if bundle["window_met"] else "window not met"
            print(
                f"  {bundle['bundle']}: {bundle['window_days']:.1f} days "
                f"({bundle['confirmation_window_days']:.1f} confirmation days, "
                f"{bundle['confirmation_resolved']} eligible) — {state}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forward-score", description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--position", choices=("long-only", "long-short"), default="long-only")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default=None, help="write the JSON payload here")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root or ROOT).resolve()
    payload = forward_score(
        root,
        tickers=args.tickers,
        threshold=args.threshold,
        position=args.position,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        interval_seconds=args.interval_seconds,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        render(payload)
        if args.output:
            print(f"\nartifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
