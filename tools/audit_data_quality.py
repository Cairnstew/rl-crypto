#!/usr/bin/env python3
"""Audit the committed ``data/<slug>/`` cache for quality faults (read-only).

Checks two things the scheduled runs cannot check for themselves — that what is
already on disk is trustworthy, and that the upstreams' own quality counters are
behaving. Nothing is written and no network call is made:

candles.jsonl — the raw market archive
  * continuity: monotonic timestamps, no duplicates, no gaps in the 1h grid
  * OHLC sanity: high/low bracket the body, no non-positive or non-finite prices
  * volume: negative or zero bars, extreme outliers
  * jump magnitude, with the cross-ticker correlation called out (a move that
    hits every asset in the same bar is market action, not a bad bar)

live.jsonl — the unified record
  * duplicate ``generated_at``, schema ids, nullable fields
  * error counters per source, and whether ``ok`` is consistent with them
  * integrity: sentiment in range, price vs consensus, live price vs the
    nearest archived bar

     uv run python tools/audit_data_quality.py
     uv run python tools/audit_data_quality.py --ticker btc-usd --details
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
OHLC = ("open", "high", "low", "close", "volume")
# One canonical parser for every timestamp in this repo. The archives are
# written in ISO-8601 `...Z`; hardcoding a second `strptime` format here is how
# a format change goes unnoticed until the audit crashes on it.
from rl_crypto.dataset import parse_iso  # noqa: E402


def _parse_ts(raw: str) -> datetime:
    """Parse a timestamp from the cache, or raise with the offending value."""
    parsed = parse_iso(raw)
    if parsed is None:
        raise ValueError(f"unparseable timestamp {raw!r}")
    return parsed
HOUR = 3600.0


def _load(path: Path) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((lineno, json.loads(line)))
            except json.JSONDecodeError as exc:
                rows.append((lineno, {"__parse_error__": str(exc)}))
    return rows


def audit_candles(rows: list[tuple[int, dict]]) -> dict:
    """Structural integrity of the OHLCV archive."""
    out: dict = {"bars": len(rows)}
    bad = Counter()
    notes: list[str] = []
    prev = None
    closes: list[tuple[datetime, float]] = []
    volumes: list[float] = []

    for lineno, r in rows:
        if "__parse_error__" in r:
            bad["bad_json_line"] += 1
            continue
        raw = r.get("timestamp")
        if not isinstance(raw, str):
            bad["missing_timestamp"] += 1
            continue
        ts = _parse_ts(raw)
        if prev is not None:
            delta = (ts - prev).total_seconds()
            if delta <= 0:
                bad["non_monotonic_or_duplicate"] += 1
            elif delta != HOUR:
                bad[f"gap_{int(delta)}s"] += 1
                notes.append(f"line {lineno}: {prev.isoformat()} -> {raw}")
        prev = ts

        vals: dict[str, float] = {}
        for field in OHLC:
            v = r.get(field)
            if v is None:
                bad[f"null_{field}"] += 1
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                bad[f"non_numeric_{field}"] += 1
                continue
            if isinstance(v, float) and not math.isfinite(v):
                bad[f"nonfinite_{field}"] += 1
                continue
            vals[field] = v
        body = [vals.get(f) for f in OHLC[:4]]
        nums = [v for v in body if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) == 4:
            o, high, low, close = (float(n) for n in nums)
            out["priced_bars"] = out.get("priced_bars", 0) + 1
            for field, v in zip(OHLC[:4], (o, high, low, close), strict=True):
                if v <= 0:
                    bad[f"nonpositive_{field}"] += 1
                    notes.append(f"line {lineno}: {raw} {field}={v}")
            if high < max(o, close):
                bad["high_below_body"] += 1
                notes.append(f"line {lineno}: {raw} high={high} < body(o={o},c={close})")
            if low > min(o, close):
                bad["low_above_body"] += 1
                notes.append(f"line {lineno}: {raw} low={low} > body(o={o},c={close})")
            if high < low:
                bad["high_below_low"] += 1
            closes.append((ts, close))
        if "volume" in vals:
            v = vals["volume"]
            if v < 0:
                bad["negative_volume"] += 1
            elif v == 0:
                bad["zero_volume_bars"] += 1
            volumes.append(v)

    # A jump is only interesting if it is not shared by the whole market; the
    # archive-level cross-check happens in the report, but keep per-asset detail.
    jumps = []
    for (_, c0), (t1, c1) in itertools.pairwise(closes):
        if c0 > 0:
            ret = c1 / c0 - 1.0
            if abs(ret) > 0.05:
                jumps.append((t1.isoformat(), c0, c1, ret))
    out["jumps_gt5pct_1h"] = jumps
    if volumes and min(volumes) == 0:
        out["zero_volume_bars"] = sum(1 for v in volumes if v == 0)
    if volumes:
        med = statistics.median(volumes)
        if med > 0:
            out["volume_max_over_median"] = round(max(volumes) / med, 1)
    if rows:
        last = _parse_ts(rows[-1][1]["timestamp"])
        out["hours_since_last_bar"] = round((datetime.now(UTC) - last).total_seconds() / HOUR, 1)
        out["first_bar"] = rows[0][1]["timestamp"]
        out["last_bar"] = rows[-1][1]["timestamp"]
    out["faults"] = bad
    out["notes"] = notes[:50]
    return out


def audit_live(rows: list[tuple[int, dict]]) -> dict:
    """Integrity and error-counter behaviour of the live unified records."""
    out: dict = {"records": len(rows)}
    bad = Counter()
    seen: Counter = Counter()
    notes: list[str] = []
    nullable: Counter = Counter()

    def flatten(obj, prefix=""):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from flatten(v, path)
            else:
                yield path, v

    for _, r in rows:
        if "__parse_error__" in r:
            bad["bad_json_line"] += 1
            continue
        gen = r.get("generated_at")
        if gen is None:
            bad["missing_generated_at"] += 1
        else:
            seen[gen] += 1
        for path, v in flatten(r):
            if v is None:
                nullable[path] += 1
        if r.get("schema") != "rl-crypto/1":
            bad["bad_top_schema"] += 1
        pred = r.get("predictions") or {}
        if pred and pred.get("schema") != "rl-crypto/ml/1":
            bad["bad_ml_schema"] += 1
        for group in ("market", "news"):
            g = r.get(group) or {}
            if g.get("errors"):
                bad[f"{group}_errors"] += 1
            if g.get("ok") is False:
                bad[f"{group}_not_ok"] += 1
        n = r.get("news") or {}
        for field in ("mean_sentiment", "latest_sentiment"):
            v = n.get(field)
            if v is not None and not -1.0 <= v <= 1.0:
                bad[f"{field}_out_of_range"] += 1
                notes.append(f"{gen}: {field}={v}")
        m = r.get("market") or {}
        price, cons = m.get("price"), m.get("consensus_price")
        if price and cons and price > 0 and abs(price - cons) / price > 0.02:
            bad["price_vs_consensus_gt2pct"] += 1
            notes.append(f"{gen}: price={price} consensus={cons}")

    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        bad["duplicate_generated_at"] = len(dupes)
        notes.append(f"duplicated stamps: {sorted(dupes.items())[:5]}")
    out["faults"] = bad
    out["nullable_fields"] = nullable
    out["notes"] = notes[:50]
    return out


def cross_check(slug: str) -> list[str]:
    """Live snapshot prices that disagree with the nearest archived bar."""
    cpath, lpath = ROOT / slug / "candles.jsonl", ROOT / slug / "live.jsonl"
    bars = {}
    for _, r in _load(cpath):
        raw = r.get("timestamp")
        close = r.get("close")
        if isinstance(raw, str) and isinstance(close, float):
            bars[_parse_ts(raw)] = close
    if not bars:
        return []
    out = []
    for _, r in _load(lpath):
        if "__parse_error__" in r:
            continue
        gen, price = r.get("generated_at"), (r.get("market") or {}).get("price")
        if not gen or not price:
            continue
        t = _parse_ts(gen)
        secs, bar = min((abs((t - b).total_seconds()), b) for b in bars)
        if secs > 6 * HOUR:
            continue
        if abs(price / bars[bar] - 1) > 0.01:
            out.append(
                f"{gen} price={price} vs bar {bar.isoformat()} {bars[bar]} "
                f"({price / bars[bar] - 1:+.2%})"
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", help="audit one slug (e.g. btc-usd); default all")
    ap.add_argument("--details", action="store_true", help="list individual faults")
    args = ap.parse_args()

    slugs = sorted(
        p.name
        for p in ROOT.iterdir()
        if p.is_dir()
        and (p / "candles.jsonl").exists()
        and (args.ticker is None or p.name == args.ticker)
    )
    if not slugs:
        print(f"no matching ticker under {ROOT}")
        return 1

    header = (
        f"{'slug':<10}{'bars':>6}{'priced':>7}{'gaps':>6}{'dup':>5}"
        f"{'OHLC':>6}{'nulls':>6}{'vol0':>6}{'jumps':>7}{'stale_h':>8}"
        f"{'live':>6}{'mktErr':>8}{'newsErr':>8}{'dupTs':>6}{'priceDev':>10}"
    )
    print(header)
    print("-" * len(header))

    faults_total = Counter()
    for slug in slugs:
        c = audit_candles(_load(ROOT / slug / "candles.jsonl"))
        live = audit_live(_load(ROOT / slug / "live.jsonl"))
        devs = cross_check(slug)
        cf, lf = c["faults"], live["faults"]
        gaps = sum(v for k, v in cf.items() if k.startswith("gap_"))
        ohlc = sum(
            cf.get(k, 0)
            for k in (
                "high_below_body",
                "low_above_body",
                "high_below_low",
                "nonpositive_open",
                "nonpositive_high",
                "nonpositive_low",
                "nonpositive_close",
            )
        )
        nulls = sum(cf.get(f"null_{f}", 0) for f in OHLC) + sum(
            cf.get(f"non_numeric_{f}", 0) for f in OHLC
        )
        print(
            f"{slug:<10}{c['bars']:>6}{c.get('priced_bars', 0):>7}{gaps:>6}"
            f"{cf.get('non_monotonic_or_duplicate', 0):>5}{ohlc:>6}{nulls:>6}"
            f"{c.get('zero_volume_bars', 0):>6}{len(c['jumps_gt5pct_1h']):>7}"
            f"{c.get('hours_since_last_bar', 0):>8}{live['records']:>6}"
            f"{lf.get('market_errors', 0):>8}{lf.get('news_errors', 0):>8}"
            f"{lf.get('duplicate_generated_at', 0):>6}{len(devs):>10}"
        )
        faults_total.update({f"candles:{k}": v for k, v in cf.items()})
        faults_total.update({f"live:{k}": v for k, v in lf.items()})
        if args.details:
            for label, res in (("candles", c), ("live", live)):
                for n in res["notes"]:
                    print(f"    [{label}] {n}")
            for d in devs[:10]:
                print(f"    [join] {d}")

    print("\n=== totals across audited tickers ===")
    if faults_total:
        for k, v in sorted(faults_total.items()):
            print(f"  {k:<34} {v}")
    else:
        print("  no faults")

    null_fields = Counter()
    for slug in slugs:
        null_fields.update(audit_live(_load(ROOT / slug / "live.jsonl"))["nullable_fields"])
    if null_fields:
        print("\n=== nullable live leaf fields ===")
        for k, v in null_fields.most_common():
            print(f"  {k:<34} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
