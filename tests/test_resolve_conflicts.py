"""Offline tests for the cache conflict resolver (union semantics).

The 30-minute timer and the 2-hourly pipeline both append to
``data/<slug>/live.jsonl``, so rebases conflict on those tails. Picking either
side silently drops live records that were genuinely collected, which is the
failure these tests exist to prevent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "resolve_cache_conflicts.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("resolve_cache_conflicts", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rec(ts: str, price: float) -> dict:
    return {"generated_at": ts, "market": {"price": price}, "predictions": {"signal": 1}}


def test_union_keeps_both_sides_new_records():
    tool = _load_tool()
    ours = [_rec("2026-10-12T10:00:00Z", 1.0), _rec("2026-10-12T11:00:00Z", 2.0)]
    theirs = [_rec("2026-10-12T10:00:00Z", 1.0), _rec("2026-10-12T11:30:00Z", 3.0)]

    merged = tool.union_jsonl(ours, theirs)

    stamps = [r["generated_at"] for r in merged]
    assert stamps == [
        "2026-10-12T10:00:00Z",  # shared, not duplicated
        "2026-10-12T11:00:00Z",  # ours only
        "2026-10-12T11:30:00Z",  # theirs only
    ]


def test_union_is_order_independent_and_deterministic():
    """Either side may be "ours"; the result must be identical, and stable."""
    tool = _load_tool()
    a = [_rec("2026-10-12T12:00:00Z", 2.0)]
    b = [_rec("2026-10-12T11:00:00Z", 1.0), _rec("2026-10-12T13:00:00Z", 3.0)]

    forward = tool.union_jsonl(a, b)
    backward = tool.union_jsonl(b, a)
    assert forward == backward
    assert forward == tool.union_jsonl(a, b)  # deterministic


def test_union_falls_back_through_timestamp_key_variants():
    tool = _load_tool()
    records = [
        {"_generated_at": "2026-10-12T11:00:00Z"},
        {"timestamp": "2026-10-12T10:00:00Z"},
        {"ts": "2026-10-12T12:00:00Z"},
        {"no_time": 1},
    ]
    merged = tool.union_jsonl(records, records)
    assert len(merged) == 4
    # the three timed records order oldest -> newest; the untimed one sorts first
    assert merged[0] == {"no_time": 1}
    assert [r.get("timestamp") or r.get("ts") or r.get("_generated_at") for r in merged[1:]] == [
        "2026-10-12T10:00:00Z",
        "2026-10-12T11:00:00Z",
        "2026-10-12T12:00:00Z",
    ]


def test_union_deduplicates_identical_records_only():
    tool = _load_tool()
    rec = _rec("2026-10-12T11:00:00Z", 5.0)
    # same timestamp, different payload => genuinely different data, keep both
    other = _rec("2026-10-12T11:00:00Z", 6.0)
    merged = tool.union_jsonl([rec, other, rec], [rec, other])
    assert len(merged) == 2


def test_union_handles_empty_sides():
    tool = _load_tool()
    rec = _rec("2026-10-12T11:00:00Z", 1.0)
    assert tool.union_jsonl([], []) == []
    assert tool.union_jsonl([rec], []) == [rec]
    assert tool.union_jsonl([], [rec]) == [rec]


def test_unioned_output_is_stable_jsonl_round_trip(tmp_path):
    """A re-serialised union must round-trip, so a replayed commit that changed
    nothing produces no cache diff."""
    tool = _load_tool()
    ours = [_rec("2026-10-12T10:00:00Z", 1.0), _rec("2026-10-12T11:00:00Z", 2.0)]
    theirs = [_rec("2026-10-12T10:00:00Z", 1.0), _rec("2026-10-12T11:30:00Z", 3.0)]

    merged = tool.union_jsonl(ours, theirs)
    body = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in merged)
    reparsed = [json.loads(line) for line in body.splitlines()]
    assert reparsed == merged
