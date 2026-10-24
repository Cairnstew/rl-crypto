"""Offline tests for the experiment ledger's trial and holdout accounting."""

from __future__ import annotations

import json

import pytest

from rl_crypto import experiments as ledger


def _preregister(root, *, hypothesis="H1", family="pooled", lineage="H1:BTC/USD", seed=0):
    return ledger.preregister(
        root,
        hypothesis_id=hypothesis,
        family=family,
        lineage_id=lineage,
        mechanism="pooling borrows strength across tickers",
        configs=[{"name": "all+reference+h1", "folds": 4}],
        tickers=["BTC/USD"],
        targets=["fwd_direction"],
        metric="mean fold improvement",
        criteria="beats baseline in 3/4 folds, CI lower bound > 0",
        negative="no consistent fold improvement",
        seed=seed,
    )


def test_preregister_records_the_full_provenance(tmp_path):
    (tmp_path / "data" / "btc-usd").mkdir(parents=True)
    (tmp_path / "data" / "btc-usd" / "candles.jsonl").write_text('{"close": 1}\n')

    entry = _preregister(tmp_path)

    assert entry["kind"] == "preregister"
    assert entry["trial_index"] == 1
    assert entry["schema"] == ledger.SCHEMA
    assert entry["holdout_touched"] is False
    assert entry["seed"] == 0
    assert entry["git_sha"] is None  # tmp_path is not a git repo
    assert entry["data_hash"]["BTC/USD"] is not None
    assert len(entry["data_hash"]["BTC/USD"]) == 64
    assert entry["config_hash"] == ledger.config_hash([{"name": "all+reference+h1", "folds": 4}])


def test_trial_index_increments_per_family_not_per_lineage(tmp_path):
    first = _preregister(tmp_path, family="pooled", lineage="H1:BTC/USD")
    second = _preregister(tmp_path, family="pooled", lineage="H1:ETH/USD")
    other = _preregister(tmp_path, hypothesis="H3", family="cross-asset", lineage="H3:BTC/USD")

    assert first["trial_index"] == 1
    assert second["trial_index"] == 2
    assert other["trial_index"] == 1


def test_result_requires_a_prior_preregistration(tmp_path):
    with pytest.raises(ValueError, match="never pre-registered"):
        ledger.record_result(
            tmp_path,
            hypothesis_id="H1",
            lineage_id="H1:BTC/USD",
            verdict="negative",
        )


def test_result_inherits_the_preregistration_and_rejects_unknown_verdicts(tmp_path):
    pre = _preregister(tmp_path)
    with pytest.raises(ValueError, match="verdict"):
        ledger.record_result(
            tmp_path, hypothesis_id="H1", lineage_id="H1:BTC/USD", verdict="great"
        )

    result = ledger.record_result(
        tmp_path,
        hypothesis_id="H1",
        lineage_id="H1:BTC/USD",
        verdict="negative",
        artifact="~/.local/share/rl-crypto/model-lab/evaluate-x.json",
        metrics={"folds_beating": 1},
    )

    assert result["kind"] == "result"
    assert result["family"] == pre["family"]
    assert result["trial_index"] == pre["trial_index"]
    assert result["config_hash"] == pre["config_hash"]
    assert result["data_hash"] == pre["data_hash"]
    assert result["holdout_touched"] is False


def test_holdout_may_be_spent_once_per_lineage(tmp_path):
    _preregister(tmp_path)
    entries = ledger.read_entries(tmp_path)

    assert (
        ledger.check_promote_allowed(
            entries, lineage_id="H1:BTC/USD", candidate_hash="aaa", apply=False
        )
        is None
    )

    ledger.record_holdout(
        tmp_path,
        hypothesis_id="H1",
        lineage_id="H1:BTC/USD",
        candidate="candidate.json",
        candidate_hash="aaa",
        passed=True,
        apply=False,
    )
    entries = ledger.read_entries(tmp_path)
    assert len(ledger.holdout_touches(entries, "H1:BTC/USD")) == 1

    # The apply that follows the same passing dry run is the same touch.
    assert (
        ledger.check_promote_allowed(
            entries, lineage_id="H1:BTC/USD", candidate_hash="aaa", apply=True
        )
        is None
    )

    # A different candidate in the same lineage must open a new lineage.
    refused = ledger.check_promote_allowed(
        entries, lineage_id="H1:BTC/USD", candidate_hash="bbb", apply=False
    )
    assert refused is not None and "different candidate" in refused


def test_apply_without_a_passing_dry_run_is_refused(tmp_path):
    _preregister(tmp_path)
    ledger.record_holdout(
        tmp_path,
        hypothesis_id="H1",
        lineage_id="H1:BTC/USD",
        candidate="candidate.json",
        candidate_hash="aaa",
        passed=False,
        apply=False,
    )
    entries = ledger.read_entries(tmp_path)
    refused = ledger.check_promote_allowed(
        entries, lineage_id="H1:BTC/USD", candidate_hash="aaa", apply=True
    )
    assert refused is not None and "passing dry run" in refused


def test_a_second_apply_is_refused(tmp_path):
    _preregister(tmp_path)
    for apply in (False, True, True):
        ledger.record_holdout(
            tmp_path,
            hypothesis_id="H1",
            lineage_id="H1:BTC/USD",
            candidate="candidate.json",
            candidate_hash="aaa",
            passed=True,
            apply=apply,
        )
    entries = ledger.read_entries(tmp_path)
    refused = ledger.check_promote_allowed(
        entries, lineage_id="H1:BTC/USD", candidate_hash="aaa", apply=True
    )
    assert refused is not None and "already been applied" in refused


def test_counts_and_scoreboard_render_from_the_ledger(tmp_path):
    _preregister(tmp_path)
    ledger.record_result(
        tmp_path, hypothesis_id="H1", lineage_id="H1:BTC/USD", verdict="negative"
    )
    entries = ledger.read_entries(tmp_path)
    totals = ledger.counts(entries)

    assert totals["entries"] == 2
    assert totals["families"]["pooled"]["trials"] == 1
    assert totals["families"]["pooled"]["results"] == 1
    assert totals["holdout_touches"] == 0

    path = ledger.write_scoreboard(tmp_path)
    text = path.read_text()
    assert "Experiment scoreboard" in text
    assert "| pooled | 1 | 1 | 0 |" in text
    assert "H1:BTC/USD" in text
    # Rendered, not hand-edited: the same ledger always renders the same table.
    assert ledger.render_scoreboard(entries) == ledger.render_scoreboard(entries)


def test_validate_rejects_a_malformed_entry(tmp_path):
    problems = ledger.validate({"schema": ledger.SCHEMA, "kind": "preregister"})
    assert any("missing hypothesis_id" in problem for problem in problems)
    assert any("data_hash" in problem for problem in problems)


def test_ledger_is_append_only_jsonl(tmp_path):
    _preregister(tmp_path)
    _preregister(tmp_path, lineage="H1:ETH/USD")
    lines = ledger.ledger_path(tmp_path).read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["kind"] == "preregister" for line in lines)


def test_file_hash_is_none_for_a_missing_file(tmp_path):
    assert ledger.file_hash(tmp_path / "nope") is None
