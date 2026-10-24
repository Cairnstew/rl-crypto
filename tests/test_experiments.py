"""Offline tests for the append-only experiment ledger and mechanical verdict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_crypto.experiments import (
    DEFAULT_CRITERIA,
    Hypothesis,
    append_entry,
    build_preregistration,
    completed_hypotheses,
    evaluate_criteria,
    holdout_touches,
    load_queue,
    read_ledger,
    render_hypotheses,
    render_scoreboard,
    select_hypothesis,
    sha256_json,
    trial_counts,
)


def _hypothesis(**overrides) -> Hypothesis:
    payload = {
        "id": "H1",
        "title": "test hypothesis",
        "family": "features",
        "order": 1,
        "status": "open",
        "needs": ["code"],
        "mechanism": "because",
        "experiment": {
            "action": "sweep",
            "tickers": ["BTC/USD"],
            "targets": ["fwd_direction"],
            "featureSets": ["all"],
            "modelSets": ["reference"],
            "horizons": [1],
            "maxConfigs": 2,
            "folds": 4,
        },
        "criteria": {
            **DEFAULT_CRITERIA,
            "target": "fwd_direction",
            "tickers_beating_min": 1,
        },
    }
    payload.update(overrides)
    return Hypothesis.from_dict(payload)


def _candles(root: Path, ticker: str = "BTC/USD", rows: int = 5) -> None:
    slug = ticker.replace("/", "-").lower()
    path = root / "data" / slug / "candles.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(
                json.dumps(
                    {
                        "ticker": ticker,
                        "timestamp": f"2026-01-01T{index:02d}:00:00Z",
                        "close": 100.0 + index,
                    }
                )
                + "\n"
            )


# --------------------------------------------------------------------------- ledger


def test_append_entry_assigns_sequential_ids_and_preserves_payload(tmp_path):
    ledger = tmp_path / "experiments" / "ledger.jsonl"
    first = append_entry(ledger, {"kind": "note", "text": "one"})
    second = append_entry(ledger, {"kind": "note", "text": "two"})
    assert first["id"] == "e0001"
    assert second["id"] == "e0002"
    assert first["schema"] == "rl-crypto/experiments/1"
    assert [entry["text"] for entry in read_ledger(ledger)] == ["one", "two"]


def test_trial_counts_are_cumulative_per_family_and_hypothesis(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    for hypothesis_id, family in (("H1", "features"), ("H1", "features"), ("H4", "policy")):
        append_entry(
            ledger,
            {"kind": "preregister", "hypothesis_id": hypothesis_id, "family": family},
        )
    append_entry(ledger, {"kind": "verdict", "hypothesis_id": "H1"})
    counts = trial_counts(read_ledger(ledger))
    assert counts["family"] == {"features": 2, "policy": 1}
    assert counts["hypothesis"] == {"H1": 2, "H4": 1}


def test_holdout_touches_are_counted_per_lineage(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_entry(ledger, {"kind": "holdout", "lineage": "H1/t1", "holdout_touched": True})
    append_entry(ledger, {"kind": "holdout", "lineage": "H1/t1", "holdout_touched": True})
    append_entry(ledger, {"kind": "preregister", "lineage": "H1/t2", "hypothesis_id": "H1"})
    assert holdout_touches(read_ledger(ledger)) == {"H1/t1": 2}


# --------------------------------------------------------------------------- queue


def test_load_queue_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "hypotheses.json"
    path.write_text(
        json.dumps(
            {
                "hypotheses": [
                    {"id": "H1", "title": "a", "experiment": {}, "criteria": {}},
                    {"id": "H1", "title": "b", "experiment": {}, "criteria": {}},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_queue(path)


def test_select_hypothesis_respects_needs_dependencies_and_completion():
    queue = [
        _hypothesis(id="H1", order=1, needs=["code"], experiment={"tickers": ["BTC/USD"], "targets": ["fwd_direction"], "maxConfigs": 1, "folds": 4}),
        _hypothesis(id="H2", order=2, needs=[], experiment={"tickers": ["BTC/USD"], "targets": ["fwd_direction"], "maxConfigs": 1, "folds": 4}),
        _hypothesis(id="H3", order=3, needs=[], depends_on=["H1"], experiment={"tickers": ["BTC/USD"], "targets": ["fwd_direction"], "maxConfigs": 1, "folds": 4}),
    ]
    # With no code capability, a no-needs hypothesis is still runnable.
    mechanical = select_hypothesis(queue, [], capabilities=set())
    assert mechanical is not None and mechanical.id == "H2"
    # A code hypothesis needs the capability.
    coded = select_hypothesis(queue, [], capabilities={"code"})
    assert coded is not None and coded.id == "H1"
    done = [
        {"kind": "verdict", "hypothesis_id": "H1", "outcome": "negative"},
        {"kind": "verdict", "hypothesis_id": "H2", "outcome": "diagnostic"},
    ]
    assert completed_hypotheses(done) == {"H1", "H2"}
    # H3 depends on H1 and is now unblocked.
    third = select_hypothesis(queue, done, capabilities=set())
    assert third is not None and third.id == "H3"


# --------------------------------------------------------------------------- prereg


def test_build_preregistration_freezes_experiment_criteria_and_data(tmp_path):
    _candles(tmp_path, rows=7)
    hypothesis = _hypothesis()
    entries = [
        {
            "kind": "preregister",
            "hypothesis_id": "H1",
            "family": "features",
        }
    ]
    prereg = build_preregistration(
        hypothesis,
        run_id="20260101T000000Z-h1",
        root=tmp_path,
        entries=entries,
        seed=7,
    )
    assert prereg["trial_index"] == 2
    assert prereg["experiment"] == hypothesis.experiment
    assert prereg["criteria"] == hypothesis.criteria
    assert prereg["config_hash"] == sha256_json(hypothesis.experiment)
    assert prereg["criteria_hash"] == sha256_json(hypothesis.criteria)
    fingerprint = prereg["data"]
    assert fingerprint["tickers"]["BTC/USD"]["rows"] == 7
    assert fingerprint["combined"]
    assert prereg["holdout_touched"] is False
    assert prereg["seed"] == 7


# --------------------------------------------------------------------------- verdict


def _lab_row(*, folds_beating: int, mean: float, minimum: float) -> dict:
    return {
        "ticker": "BTC/USD",
        "config": {"name": "all+reference+h1"},
        "targets": {
            "fwd_direction": {
                "folds": 4,
                "folds_beating": folds_beating,
                "beats_ratio": folds_beating / 4,
                "mean_improvement": mean,
                "min_improvement": minimum,
            }
        },
        "economic": {
            "model": {"total_return": 0.4, "sharpe": 1.1, "trades": 150},
            "momentum": {"total_return": 0.1, "sharpe": 0.2, "trades": 150},
        },
    }


def test_evaluate_criteria_passes_only_when_every_bar_is_met():
    prereg = {
        "criteria": {
            **DEFAULT_CRITERIA,
            "target": "fwd_direction",
            "tickers_beating_min": 1,
        }
    }
    good = evaluate_criteria(prereg, {"results": [_lab_row(folds_beating=4, mean=0.02, minimum=0.01)]})
    assert good["passed"] is True
    assert good["outcome"] == "provisional"
    assert good["metrics"]["tickers_beating"] == 1

    bad = evaluate_criteria(
        prereg,
        {"results": [_lab_row(folds_beating=1, mean=0.02, minimum=0.01)]},
    )
    assert bad["passed"] is False
    assert bad["outcome"] == "negative"
    assert any("tickers met the improvement bar" in reason for reason in bad["reasons"])

    negative_mean = evaluate_criteria(
        prereg,
        {"results": [_lab_row(folds_beating=4, mean=-0.01, minimum=-0.02)]},
    )
    assert negative_mean["passed"] is False


def test_evaluate_criteria_rejects_negative_after_cost_return():
    criteria = {
        **DEFAULT_CRITERIA,
        "target": "fwd_direction",
        "tickers_beating_min": 1,
        "economic": {"total_return_min": 0.0, "sharpe_above_momentum": True, "trades_min": 100},
    }
    row = _lab_row(folds_beating=4, mean=0.02, minimum=0.01)
    row["economic"]["model"]["total_return"] = -0.01
    verdict = evaluate_criteria({"criteria": criteria}, {"results": [row]})
    assert verdict["passed"] is False
    assert any("total return" in reason for reason in verdict["reasons"])


def test_diagnostic_criteria_only_needs_rows():
    verdict = evaluate_criteria(
        {"criteria": {"kind": "diagnostic"}},
        {
            "results": [
                {
                    "ticker": "BTC/USD",
                    "targets": {
                        "fwd_direction": {
                            "folds": 4,
                            "folds_beating": 1,
                            "mean_improvement": -0.01,
                            "min_improvement": -0.02,
                        }
                    },
                    "economic": {"model": {"total_return": -0.1, "sharpe": -1.0, "trades": 10}},
                }
            ]
        },
    )
    assert verdict["passed"] is True
    assert verdict["outcome"] == "diagnostic"
    assert verdict["metrics"]["targets"] == ["fwd_direction"]
    stats = verdict["metrics"]["per_ticker"]["BTC/USD"]["targets"]["fwd_direction"]
    assert stats["folds_beating"] == 1
    assert verdict["metrics"]["per_ticker"]["BTC/USD"]["economic"]["trades"] == 10.0
    empty = evaluate_criteria({"criteria": {"kind": "diagnostic"}}, {"results": []})
    assert empty["passed"] is False


def test_missing_lab_result_is_blocked_not_negative():
    verdict = evaluate_criteria({"criteria": DEFAULT_CRITERIA}, None)
    assert verdict["outcome"] == "blocked"
    assert verdict["passed"] is False


# --------------------------------------------------------------------------- rendering


def test_rendering_is_deterministic_and_mentions_negative_results():
    entries = [
        {"id": "e0001", "kind": "preregister", "hypothesis_id": "H1", "family": "features"},
        {
            "id": "e0002",
            "kind": "verdict",
            "hypothesis_id": "H1",
            "family": "features",
            "outcome": "negative",
            "trial_index": 1,
            "metrics": {"target": "fwd_direction", "tickers": ["BTC/USD"], "tickers_beating": 0},
            "reasons": ["no edge"],
        },
    ]
    scoreboard = render_scoreboard(entries)
    assert "negative" in scoreboard
    assert "0/1 tickers" in scoreboard
    assert scoreboard == render_scoreboard(entries)

    markdown = render_hypotheses([_hypothesis()], entries)
    assert "| H1 |" in markdown
    assert "done (negative)" in markdown
