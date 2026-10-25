#!/usr/bin/env python
"""Read-only before/after table for the opt-in stricter Tier-1 trust gate.

For every committed bundle this replays the training split from the committed
``data/`` archive, recomputes the strongest simple baseline for each target
(seasonal hour-of-day for ``fwd_volume_ratio``, HAR for ``fwd_volatility``) on
exactly the rows the bundle was scored on, and reports:

* ``before`` — the verdict the shipped bundle carries (constant baseline), and
* ``after``  — the verdict the stricter gate would write.

Nothing is written: not ``models/``, not ``data/``. The gate itself is OFF in
production until a human sets ``RL_CRYPTO_STRONG_BASELINE``; this tool forces the
comparison so the decision can be made from measured numbers.

Usage::

    PYTHONPATH=src python tools/trust_verdict.py [--models models] [--data data]
                                       [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rl_crypto.cache import load_candles
from rl_crypto.features import FEATURE_VERSION
from rl_crypto.ml.store import load_bundle
from rl_crypto.ml.trust_gate import (
    baseline_label,
    compare_target,
    replay_split,
    strong_baseline_enabled,
)
from rl_crypto.samples import build_samples

VERDICT_KEY = {"regression": "rmse", "classification": "log_loss"}


def compare_bundle(bundle, samples) -> list[dict[str, Any]]:
    """Per-target before/after rows for one bundle (read-only)."""
    rows: list[dict[str, Any]] = []
    for target, model in bundle.targets.items():
        train, test = replay_split(bundle, samples, target)
        row = compare_target(
            target,
            model,
            train,
            test,
            expected_holdout=model.holdout.n_samples if model.holdout else None,
        )
        key = VERDICT_KEY.get(model.kind, "rmse")
        row["model"] = (model.holdout.metrics or {}).get(key) if model.holdout else None
        row["before_metric"] = (model.holdout.baseline or {}).get(key) if model.holdout else None
        row["after_metric"] = (
            (row["strong"] or {}).get(key) if row["baseline_used"] == "strong"
            else row["before_metric"]
        )
        rows.append(row)
    return rows


def _fmt(value: Any, digits: int = 5) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _verdict(value: Any) -> str:
    if value is None:
        return "unmeasured"
    return "trusted" if value else "below"


def build_report(models_dir: Path, data_dir: Path) -> dict[str, Any]:
    tickers: list[dict[str, Any]] = []
    for path in sorted(models_dir.glob("*.json")):
        ticker = path.stem
        archive = data_dir / ticker / "candles.jsonl"
        if not archive.exists():
            tickers.append({"ticker": ticker, "error": f"missing archive {archive}"})
            continue
        bundle = load_bundle(path)
        rows = load_candles(archive, ticker=ticker)
        samples = build_samples(rows, horizon=bundle.horizon, interval_seconds=3600)
        tickers.append(
            {
                "ticker": ticker,
                "feature_version": bundle.feature_version,
                "feature_version_current": bundle.feature_version == FEATURE_VERSION,
                "trained_at": bundle.trained_at,
                "rows": compare_bundle(bundle, samples),
            }
        )
    # Summary: trusted counts before/after per target, and overall.
    targets: dict[str, dict[str, int]] = {}
    for entry in tickers:
        for row in entry.get("rows", []):
            target = row["target"]
            bucket = targets.setdefault(
                target, {"measured": 0, "before": 0, "after": 0, "gated": 0}
            )
            if row["before"] is None or not row["replay_ok"]:
                continue
            bucket["measured"] += 1
            bucket["before"] += 1 if row["before"] else 0
            bucket["after"] += 1 if row["after"] else 0
            bucket["gated"] += 1 if row["baseline_used"] == "strong" else 0
    totals = {
        "measured": sum(bucket["measured"] for bucket in targets.values()),
        "before": sum(bucket["before"] for bucket in targets.values()),
        "after": sum(bucket["after"] for bucket in targets.values()),
        "gated": sum(bucket["gated"] for bucket in targets.values()),
    }
    return {
        "flag_enabled_in_this_process": strong_baseline_enabled(),
        "flag": "RL_CRYPTO_STRONG_BASELINE",
        "models_dir": str(models_dir),
        "data_dir": str(data_dir),
        "tickers": tickers,
        "targets": targets,
        "totals": totals,
    }


def print_table(report: dict[str, Any]) -> None:
    print(
        "flag RL_CRYPTO_STRONG_BASELINE = "
        f"{'ON' if report['flag_enabled_in_this_process'] else 'OFF (default)'}"
    )
    print(
        f"{'ticker':10s} {'target':18s} {'before':10s} {'after':10s} "
        f"{'model':>8s} {'const':>8s} {'strong':>8s}  bar"
    )
    for entry in report["tickers"]:
        if entry.get("error"):
            print(f"{entry['ticker']:10s} ERROR {entry['error']}")
            continue
        for row in entry["rows"]:
            strong = (row["strong"] or {}).get(
                VERDICT_KEY.get(row["kind"], "rmse")
            )
            note = f"  [{row['skipped']}]" if row["skipped"] else ""
            print(
                f"{entry['ticker']:10s} {row['target']:18s} "
                f"{_verdict(row['before']):10s} {_verdict(row['after']):10s} "
                f"{_fmt(row['model']):>8s} {_fmt(row['before_metric']):>8s} "
                f"{_fmt(strong):>8s}  "
                f"{baseline_label(row['target']) if row['baseline_used'] == 'strong' else 'constant'}"
                f"{note}"
            )
    print()
    print(
        f"{'target':18s} {'measured':>8s} {'before':>7s} {'after':>6s} "
        f"{'gated':>6s}"
    )
    for target, bucket in sorted(report["targets"].items()):
        print(
            f"{target:18s} {bucket['measured']:8d} {bucket['before']:7d} "
            f"{bucket['after']:6d} {bucket['gated']:6d}"
        )
    totals = report["totals"]
    print(
        f"{'TOTAL':18s} {totals['measured']:8d} {totals['before']:7d} "
        f"{totals['after']:6d} {totals['gated']:6d}"
    )
    print("\nbefore/after = targets whose stored vs stricter verdict is trusted; "
          "gated = rows whose bar changed to the strong baseline")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="models", help="bundle directory (read-only)")
    parser.add_argument("--data", default="data", help="archive directory (read-only)")
    parser.add_argument("--json", help="write the structured report to this path")
    args = parser.parse_args(argv)

    report = build_report(Path(args.models), Path(args.data))
    print_table(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
