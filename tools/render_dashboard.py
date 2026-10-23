#!/usr/bin/env python3
"""Render the live cache dashboard inside ``README.md``.

The committed cache *is* the state of the system: each ticker's run summary, its
last live record and its trained bundle. This tool turns that into the dashboard
readers see, so the README stays a view of the running pipeline instead of a
snapshot that rots.

    python3 tools/render_dashboard.py            # rewrite the block in README.md
    python3 tools/render_dashboard.py --stdout   # print the block only

It is **stdlib-only and offline** because ``production.yml`` runs it in the
merge job, where there is no venv — and a rendering failure must never block a
cache write-back (the workflow warns and continues).

The block lives between ``<!-- dashboard:start -->`` and
``<!-- dashboard:end -->``; everything outside those markers is hand-written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

START = "<!-- dashboard:start -->"
END = "<!-- dashboard:end -->"
ROOT = Path(os.environ.get("RL_CRYPTO_ROOT") or Path.cwd()).resolve()
MISSING = "—"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _last_record(path: Path) -> dict[str, Any]:
    """The newest valid record in a JSONL file (empty when absent/corrupt)."""
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            return record
    return {}


def _instant(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _age(value: Any, now: datetime) -> str:
    stamp = _instant(value)
    if stamp is None:
        return MISSING
    seconds = max(0, int((now - stamp).total_seconds()))
    if seconds < 90 * 60:
        return f"{max(1, seconds // 60)}m ago"
    if seconds < 48 * 3600:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _clock(value: Any) -> str:
    """``HH:MMZ`` for a table cell — absolute, so a stale README cannot lie."""
    stamp = _instant(value)
    return stamp.strftime("%H:%MZ") if stamp else MISSING


def _price(value: Any) -> str:
    if not isinstance(value, int | float):
        return MISSING
    return f"${value:,.2f}" if value >= 1 else f"${value:.4f}"


def _number(value: Any) -> str:
    return f"{value:g}" if isinstance(value, int | float) else MISSING


def _probability(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else MISSING


def _newest(rows: list[dict[str, Any]], key: str) -> tuple[Any, datetime | None]:
    """The newest stamp in a column, as ``(raw, instant)``."""
    best: tuple[Any, datetime | None] = (None, None)
    for row in rows:
        instant = _instant(row.get(key))
        if instant is not None and (best[1] is None or instant > best[1]):
            best = (row[key], instant)
    return best


def _cache_bytes(root: Path) -> int:
    total = 0
    for pattern in ("data/*/*", "models/*.json"):
        for path in root.glob(pattern):
            if path.is_file():
                total += path.stat().st_size
    return total


def _has_verdict(entry: Any) -> bool:
    """Whether a target entry carries an out-of-sample verdict."""
    if not isinstance(entry, dict):
        return False
    if entry.get("trusted") is not None:
        return True
    holdout = entry.get("holdout")
    return isinstance(holdout, dict) and holdout.get("beats_baseline") is not None


def _verdict(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("trusted") is not None:
        return bool(entry["trusted"])
    holdout = entry.get("holdout")
    return isinstance(holdout, dict) and bool(holdout.get("beats_baseline"))


def _beats(summary: dict[str, Any], model: dict[str, Any]) -> tuple[int, int]:
    """``(trusted, total)`` across both cache generations.

    pipeline/3 summaries carry ``targets.*.trusted``; pipeline/2 summaries only
    carry in-sample metrics, so the verdict then comes from the bundle's
    ``holdout.beats_baseline``. Either way the count is the number of targets
    that beat a constant predictor out of sample.
    """
    summary_targets = summary.get("targets")
    summary_targets = summary_targets if isinstance(summary_targets, dict) else {}
    model_targets = model.get("targets")
    model_targets = model_targets if isinstance(model_targets, dict) else {}
    names = list(dict.fromkeys([*summary_targets, *model_targets]))
    if not names:
        return (0, 0)
    trusted = 0
    for name in names:
        entry = summary_targets.get(name)
        if not _has_verdict(entry):
            entry = model_targets.get(name)
        trusted += 1 if _verdict(entry) else 0
    return (trusted, len(names))


def collect_rows(root: Path) -> list[dict[str, Any]]:
    """One row per ticker that has a live stream, newest ticker name first."""
    rows: list[dict[str, Any]] = []
    for live_path in sorted((root / "data").glob("*/live.jsonl")):
        slug = live_path.parent.name
        summary = _load_json(live_path.parent / "summary.json")
        model = _load_json(root / "models" / f"{slug}.json")
        live = _last_record(live_path)
        predictions = live.get("predictions")
        predictions = predictions if isinstance(predictions, dict) else {}
        probabilities = predictions.get("probabilities")
        probabilities = probabilities if isinstance(probabilities, dict) else {}
        market = live.get("market")
        market = market if isinstance(market, dict) else {}
        archive = summary.get("archive")
        archive = archive if isinstance(archive, dict) else {}
        spans = archive.get("spans")
        spans = spans if isinstance(spans, list) else []
        trusted, total = _beats(summary, model)
        summary_live = summary.get("live")
        summary_live = summary_live if isinstance(summary_live, dict) else {}
        rows.append(
            {
                "ticker": summary.get("ticker") or live.get("ticker") or slug.upper(),
                "price": market.get("price", summary_live.get("price")),
                "signal": predictions.get("signal") if predictions else None,
                "confidence": predictions.get("confidence") if predictions else None,
                "p_up": probabilities.get("fwd_direction"),
                "has_predictions": bool(predictions),
                "bars": archive.get("rows"),
                "span": (
                    f"{str(spans[0])[:10]} → {str(spans[-1])[:10]}"
                    if len(spans) >= 2
                    else MISSING
                ),
                "trusted": trusted,
                "targets": total,
                "last_live": live.get("generated_at"),
                # The bundle's training time is the honest freshness of the
                # models; summary.generated_at is only the last pipeline run.
                "trained_at": model.get("trained_at") or summary.get("generated_at"),
                "feature_version": model.get("feature_version"),
            }
        )
    return rows


def render(rows: list[dict[str, Any]], *, now: datetime, cache_bytes: int) -> str:
    """The dashboard body (no markers), deterministic for a fixed ``now``."""
    newest_live, _ = _newest(rows, "last_live")
    newest_train, _ = _newest(rows, "trained_at")
    features = sorted({row["feature_version"] for row in rows if row["feature_version"]})
    contract = f"`{features[0]}`" if len(features) == 1 else MISSING

    lines = [
        "<!-- Generated by tools/render_dashboard.py from the committed cache. -->",
        "",
        f"**{len(rows)} tickers** · **{cache_bytes / 1e6:.1f} MB** cache"
        f" · as of {now.strftime('%Y-%m-%dT%H:%MZ')}",
        "",
        f"Live {newest_live or MISSING} ({_age(newest_live, now)})"
        f" · trained {newest_train or MISSING} ({_age(newest_train, now)})"
        f" · feature contract {contract}",
        "",
        "| ticker | price | signal | conf | p(up) | archive | beats baseline | last live |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        bars = f"{row['bars']:,} · {row['span']}" if isinstance(row["bars"], int) else MISSING
        beats = f"{row['trusted']}/{row['targets']}" if row["targets"] else MISSING
        if not row["has_predictions"]:
            signal, confidence = MISSING, MISSING
        else:
            signal = _number(row["signal"])
            confidence = "n/a" if row["confidence"] is None else _number(row["confidence"])
        lines.append(
            f"| {row['ticker']} | {_price(row['price'])} | {signal} | {confidence}"
            f" | {_probability(row['p_up'])} | {bars} | {beats} | {_clock(row['last_live'])} |"
        )
    lines += [
        "",
        "*`signal` `—` = withheld (no target earned it) · `conf` `n/a` = no confidence"
        " · `p(up)` = model probability of an up move (emitted even when withheld)"
        " · `beats baseline` = targets that beat a constant predictor out of sample.*",
    ]
    return "\n".join(lines)


def replace_block(readme: str, block: str) -> str:
    """Swap the content between the markers, leaving the rest untouched."""
    start = readme.find(START)
    end = readme.find(END, start + len(START))
    if start < 0 or end < 0:
        raise ValueError(f"README is missing the {START} … {END} markers")
    return f"{readme[: start + len(START)]}\n{block.rstrip()}\n{readme[end:]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=str(ROOT), help="repo root holding data/ and models/")
    parser.add_argument(
        "--readme", default=None, help="README to update (default: <root>/README.md)"
    )
    parser.add_argument("--stdout", action="store_true", help="print the block instead of writing")
    parser.add_argument("--now", default=None, help="render instant (ISO-8601; tests use this)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    readme_path = Path(args.readme) if args.readme else root / "README.md"
    now = _instant(args.now) or datetime.now(tz=UTC)

    try:
        block = render(collect_rows(root), now=now, cache_bytes=_cache_bytes(root))
        if args.stdout:
            print(block)
            return 0
        readme_path.write_text(
            replace_block(readme_path.read_text(encoding="utf-8"), block), encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        print(f"dashboard: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
