"""Offline tests for the cache conflict resolver (union semantics).

The 30-minute timer and the 2-hourly pipeline both append to
``data/<slug>/live.jsonl``, so rebases conflict on those tails. Picking either
side silently drops live records that were genuinely collected, which is the
failure these tests exist to prevent.
"""

from __future__ import annotations

import contextlib
import json

from rl_crypto.pipeline import conflicts as tool


def _rec(ts: str, price: float) -> dict:
    return {"generated_at": ts, "market": {"price": price}, "predictions": {"signal": 1}}


def test_union_keeps_both_sides_new_records():
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
    a = [_rec("2026-10-12T12:00:00Z", 2.0)]
    b = [_rec("2026-10-12T11:00:00Z", 1.0), _rec("2026-10-12T13:00:00Z", 3.0)]

    forward = tool.union_jsonl(a, b)
    backward = tool.union_jsonl(b, a)
    assert forward == backward
    assert forward == tool.union_jsonl(a, b)  # deterministic


def test_union_falls_back_through_timestamp_key_variants():
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
    rec = _rec("2026-10-12T11:00:00Z", 5.0)
    # same timestamp, different payload => genuinely different data, keep both
    other = _rec("2026-10-12T11:00:00Z", 6.0)
    merged = tool.union_jsonl([rec, other, rec], [rec, other])
    assert len(merged) == 2


def test_union_handles_empty_sides():
    rec = _rec("2026-10-12T11:00:00Z", 1.0)
    assert tool.union_jsonl([], []) == []
    assert tool.union_jsonl([rec], []) == [rec]
    assert tool.union_jsonl([], [rec]) == [rec]


def test_unioned_output_is_stable_jsonl_round_trip(tmp_path):
    """A re-serialised union must round-trip, so a replayed commit that changed
    nothing produces no cache diff."""
    ours = [_rec("2026-10-12T10:00:00Z", 1.0), _rec("2026-10-12T11:00:00Z", 2.0)]
    theirs = [_rec("2026-10-12T10:00:00Z", 1.0), _rec("2026-10-12T11:30:00Z", 3.0)]

    merged = tool.union_jsonl(ours, theirs)
    body = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in merged)
    reparsed = [json.loads(line) for line in body.splitlines()]
    assert reparsed == merged


# --------------------------------------------------------------------------- candles


def _bar(ts: str, close: float) -> dict:
    return {"timestamp": ts, "close": close, "open": close, "high": close, "low": close}


def test_merge_candles_keeps_one_bar_per_timestamp_and_prefers_replayed():
    """The archive is an upsert-by-timestamp series, not append-only: unioning it
    duplicates every shared bar (the bug that once doubled the archive)."""
    ours = [_bar("2026-10-12T10:00:00Z", 1.0), _bar("2026-10-12T11:00:00Z", 2.0)]
    theirs = [_bar("2026-10-12T10:00:00Z", 9.0), _bar("2026-10-12T12:00:00Z", 3.0)]

    merged = tool.merge_candles(ours, theirs)

    stamps = [bar["timestamp"] for bar in merged]
    assert stamps == [
        "2026-10-12T10:00:00Z",  # shared bar, not duplicated
        "2026-10-12T11:00:00Z",  # upstream only
        "2026-10-12T12:00:00Z",  # replayed only
    ]
    # The replayed commit is the newer local run, so its revision wins.
    assert merged[0]["close"] == 9.0


def test_merge_candles_handles_an_absent_side():
    bar = _bar("2026-10-12T10:00:00Z", 1.0)
    assert tool.merge_candles([], [bar]) == [bar]
    assert tool.merge_candles([bar], []) == [bar]


# --------------------------------------------------------------------------- summary.json


def test_pretty_printed_summary_is_kept_whole_not_truncated():
    """summary.json is one pretty-printed document: a line-wise parse finds no
    records and used to leave a 0-byte file behind."""
    theirs = json.dumps({"schema": "rl-crypto/pipeline/3", "generated_at": "b"}, indent=2) + "\n"
    ours = json.dumps({"schema": "rl-crypto/pipeline/2", "generated_at": "a"}, indent=2) + "\n"

    label, body = tool._merge("data/eth-usd/summary.json", ours, theirs)

    assert body == theirs
    assert "replayed" in label
    assert json.loads(body)["generated_at"] == "b"


def test_summary_falls_back_to_upstream_when_the_replayed_side_is_absent():
    ours = json.dumps({"schema": "rl-crypto/pipeline/2"}, indent=2) + "\n"

    _, body = tool._merge("data/eth-usd/summary.json", ours, "")

    assert body == ours


def test_neither_side_valid_json_fails_loudly():
    import pytest

    with pytest.raises(ValueError):
        tool._merge("data/eth-usd/summary.json", "<<<<<<<", "")


# --------------------------------------------------------------------------- end to end


def _git(cwd, *args):
    import os
    import subprocess

    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout


def test_resolve_merges_a_real_rebase_conflict(tmp_path):
    """A rebase over both writers' appends leaves live.jsonl conflicted; the
    resolver unions it, keeps one bar per candle timestamp and prefers the
    replayed summary — then stages everything so the rebase can continue."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    live = repo / "data" / "btc-usd" / "live.jsonl"
    live.parent.mkdir(parents=True)
    live.write_text(json.dumps(_rec("2026-10-12T09:00:00Z", 0)) + "\n")
    summary = repo / "data" / "btc-usd" / "summary.json"
    summary.write_text(json.dumps({"generated_at": "base"}, indent=2) + "\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")

    _git(repo, "checkout", "-q", "-b", "local")
    with live.open("a") as fh:
        fh.write(json.dumps(_rec("2026-10-12T11:00:00Z", 2)) + "\n")
    summary.write_text(json.dumps({"generated_at": "local"}, indent=2) + "\n")
    _git(repo, "commit", "-qam", "local append")

    _git(repo, "checkout", "-q", "main")
    with live.open("a") as fh:
        fh.write(json.dumps(_rec("2026-10-12T10:00:00Z", 1)) + "\n")
    summary.write_text(json.dumps({"generated_at": "upstream"}, indent=2) + "\n")
    _git(repo, "commit", "-qam", "upstream append")

    _git(repo, "checkout", "-q", "local")
    with contextlib.suppress(Exception):
        _git(repo, "rebase", "main")  # expected: the two appends conflict

    assert "live.jsonl" in _git(repo, "diff", "--name-only", "--diff-filter=U")

    resolved = tool.resolve(repo)

    assert resolved == 2
    assert _git(repo, "diff", "--name-only", "--diff-filter=U").strip() == ""
    stamps = [json.loads(line)["generated_at"] for line in live.read_text().splitlines()]
    assert stamps == [
        "2026-10-12T09:00:00Z",
        "2026-10-12T10:00:00Z",
        "2026-10-12T11:00:00Z",
    ]
    assert json.loads(summary.read_text())["generated_at"] == "local"
