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


def test_infrastructure_blocks_do_not_consume_a_trial(tmp_path):
    """Only a run that reached a lab result (or verdict) spends a trial.

    A credential/role error, a timeout, a stale archive and an unresolvable
    bootstrap CI all leave the hypothesis untested. They must not burn alpha or
    the family cap.
    """
    ledger = tmp_path / "ledger.jsonl"
    cases = [
        ("role", "lead", "role-error"),
        ("timeout", "evaluation", "timeout"),
        ("stale", "staleness", "staleness"),
        ("resolution", "evaluation", "insufficient-resolution"),
    ]
    for run_id, stage, kind in cases:
        append_entry(
            ledger,
            {
                "kind": "preregister",
                "run_id": run_id,
                "hypothesis_id": "H1",
                "family": "features",
            },
        )
        append_entry(
            ledger,
            {
                "kind": "blocked",
                "run_id": run_id,
                "hypothesis_id": "H1",
                "stage": stage,
                "reason": f"{kind}: synthetic",
                "infra": True,
                "infra_kind": kind,
            },
        )
    append_entry(
        ledger,
        {"kind": "preregister", "run_id": "result", "hypothesis_id": "H1", "family": "features"},
    )
    append_entry(
        ledger,
        {"kind": "results", "run_id": "result", "hypothesis_id": "H1", "family": "features"},
    )
    counts = trial_counts(read_ledger(ledger))
    assert counts["family"] == {"features": 1}
    assert counts["hypothesis"] == {"H1": 1}


def test_legacy_infrastructure_blocks_are_classified_from_stage_and_reason():
    from rl_crypto.experiments import is_infra_block

    assert is_infra_block({"kind": "blocked", "stage": "lead", "reason": "x"})
    assert is_infra_block(
        {"kind": "blocked", "stage": "evaluation", "reason": "no lab result was produced"}
    )
    assert is_infra_block(
        {
            "kind": "blocked",
            "stage": "evaluation",
            "reason": "insufficient-resolution: only 8 blocks",
        }
    )
    assert not is_infra_block(
        {"kind": "blocked", "stage": "evaluation", "reason": "bootstrap CI lower bound <= 0"}
    )


def _canary(entries, *, kind, outcome, at, protected_hash, run_id=None):
    return append_entry(
        entries,
        {
            "kind": "verdict",
            "run_id": run_id or f"run-{kind}",
            "hypothesis_id": f"H0-canary-{kind}",
            "canary": True,
            "canary_kind": kind,
            "outcome": outcome,
            "passed": outcome == "provisional",
            "protected_hash": protected_hash,
            "at": at,
        },
    )


def test_binomial_calibration_uses_its_own_uncertainty():
    from rl_crypto.experiments import calibration_verdict, wilson_interval

    low, high = wilson_interval(0, 300)
    assert low == 0.0
    assert 0.0 < high < 0.02
    ok = calibration_verdict(6, 300, 0.025)
    assert ok["rate"] == pytest.approx(0.02)
    assert ok["exceeds"] is False
    failed = calibration_verdict(60, 300, 0.025)
    assert failed["rate"] == pytest.approx(0.2)
    assert failed["exceeds"] is True


def test_v2_contract_violations_are_invalid_not_negative():
    """A leak that the correlation threshold cannot see must still be refused."""
    from rl_crypto.experiments import contract_violation, evaluate_criteria

    prereg = {
        "criteria": {
            "kind": "improvement",
            "criteria_version": 2,
            "target": "fwd_volume_ratio",
            "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
            "alpha": 0.05,
            "fold_rule": "all_positive",
            "require_controls": True,
            "require_multiple_tickers": False,
            "requires_promote": False,
        },
        "trial_index": 1,
    }
    pooled = {
        "mean_improvement": 0.01,
        "ci_low": 0.005,
        "ci_high": 0.02,
        "n": 100,
        "alpha": 0.025,
        "fold_improvements": [0.01] * 4,
        "resolution_ok": True,
    }
    base = {
        "results": [{"ticker": "BTC/USD"}],
        "pooled": pooled,
        "controls": {
            "shuffle": {
                "pooled": {
                    "mean_improvement": -0.01,
                    "ci_low": -0.02,
                    "n": 100,
                    "alpha": 0.025,
                    "fold_improvements": [-0.01] * 4,
                    "resolution_ok": True,
                }
            },
            "shift": {
                "pooled": {
                    "mean_improvement": -0.01,
                    "ci_low": -0.02,
                    "n": 100,
                    "alpha": 0.025,
                    "fold_improvements": [-0.01] * 4,
                    "resolution_ok": True,
                }
            },
        },
        "purge": {"ok": True, "violations": []},
        "structural": {
            "ok": True,
            "truncation": {"ok": True, "violations": []},
            "perturbation": {"ok": True, "violations": []},
        },
        "preprocessing": {"fit_scope": "train"},
        "selection": {"scope": "pre-cutoff"},
    }
    clean = evaluate_criteria(prereg, base)
    assert clean["outcome"] == "provisional"
    assert contract_violation(base) is None

    # Self-reported provenance is documentation: it no longer decides.
    documented = {
        **base,
        "preprocessing": {"fit_scope": "all"},
        "selection": {"scope": "holdout"},
    }
    assert contract_violation(documented) is None
    assert evaluate_criteria(prereg, documented)["outcome"] == "provisional"

    for mutation, needle in (
        (
            {
                "structural": {
                    "ok": False,
                    "truncation": {
                        "ok": False,
                        "violation_count": 1,
                        "violations": [{"kind": "feature", "name": "leak", "row": 3}],
                    },
                    "perturbation": {"ok": True, "violations": []},
                }
            },
            "truncation",
        ),
        ({"purge": {"ok": False, "violations": [{"reason": "overlap"}]}}, "purge"),
        (
            {
                "structural": {
                    "ok": False,
                    "truncation": {"ok": True, "violations": []},
                    "perturbation": {
                        "ok": False,
                        "violations": [{"path": "feature_names"}],
                    },
                }
            },
            "perturbation",
        ),
    ):
        verdict = evaluate_criteria(prereg, {**base, **mutation})
        assert verdict["outcome"] == "invalid"
        assert any(needle in reason for reason in verdict["reasons"])


def test_canary_freshness_requires_both_canaries_at_the_current_hash(tmp_path):
    from rl_crypto.experiments import canary_gate, canary_status

    ledger = tmp_path / "ledger.jsonl"
    _canary(
        ledger,
        kind="null",
        outcome="negative",
        at="2026-10-24T10:00:00Z",
        protected_hash="a" * 64,
    )
    _canary(
        ledger,
        kind="positive",
        outcome="invalid",
        at="2026-10-24T10:01:00Z",
        protected_hash="a" * 64,
    )
    entries = read_ledger(ledger)
    status = canary_status(entries, current_hash="a" * 64)
    assert status["pass"] is True
    assert status["hash"] == "a" * 64
    assert canary_gate(entries, "a" * 64) is None

    # A stale pair proves nothing about the contract now running.
    stale = canary_gate(entries, "b" * 64)
    assert stale is not None and "stale" in stale
    # A positive canary that failed to detect the leak is not a pass.
    ledger2 = tmp_path / "ledger2.jsonl"
    _canary(
        ledger2,
        kind="null",
        outcome="negative",
        at="2026-10-24T10:00:00Z",
        protected_hash="a" * 64,
    )
    _canary(
        ledger2,
        kind="positive",
        outcome="negative",
        at="2026-10-24T10:01:00Z",
        protected_hash="a" * 64,
    )
    assert canary_status(read_ledger(ledger2), current_hash="a" * 64)["pass"] is False


def test_scoreboard_shows_the_canary_hash_and_date():
    entries = [
        {
            "kind": "verdict",
            "canary": True,
            "canary_kind": "null",
            "outcome": "negative",
            "protected_hash": "c" * 64,
            "at": "2026-10-24T09:00:00Z",
        },
        {
            "kind": "verdict",
            "canary": True,
            "canary_kind": "positive",
            "outcome": "invalid",
            "protected_hash": "c" * 64,
            "at": "2026-10-24T09:01:00Z",
        },
    ]
    rendered = render_scoreboard(entries)
    assert f"canary: **pass @ {'c' * 64} 2026-10-24**" in rendered


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


# --------------------------------------------------------------------------- criteria v2


def _paired(mean: float, ci_low: float, ci_high: float, *, n: int = 100, folds=None, alpha=0.025):
    return {
        "mean_improvement": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "alpha": alpha,
        "fold_improvements": folds if folds is not None else [mean] * 4,
    }


def _v2_payload(
    *,
    pooled,
    tickers=("BTC/USD", "ETH/USD"),
    shuffle=None,
    shift=None,
):
    results = []
    for ticker in tickers:
        stat = pooled if len(tickers) == 1 else {
            **_paired(
                pooled["mean_improvement"],
                pooled["ci_low"],
                pooled["ci_high"],
                n=pooled["n"] // len(tickers),
                folds=pooled.get("fold_improvements"),
                alpha=pooled.get("alpha", 0.05),
            )
        }
        results.append({"ticker": ticker, "paired": stat})
    return {
        "results": results,
        "pooled": pooled,
        "pooled_economics": {
            "model": {"total_return": 0.10, "sharpe": 1.5, "trades": 200},
            "momentum": {"total_return": 0.02, "sharpe": 0.4, "trades": 200},
        },
        "controls": {
            "shuffle": {"pooled": shuffle or _paired(-0.01, -0.02, -0.001)},
            "shift": {"pooled": shift or _paired(-0.01, -0.02, -0.001)},
        },
    }


def _v2_criteria(**overrides):
    criteria = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": True,
        "requires_promote": False,
    }
    criteria.update(overrides)
    return criteria


def test_v2_no_signal_fails_and_real_signal_passes():
    no_signal = evaluate_criteria(
        {"criteria": _v2_criteria()},
        _v2_payload(pooled=_paired(0.001, -0.002, 0.004)),
    )
    assert no_signal["outcome"] == "negative"
    assert any("CI lower bound" in reason for reason in no_signal["reasons"])

    signal = evaluate_criteria(
        {"criteria": _v2_criteria()},
        _v2_payload(
            pooled=_paired(0.02, 0.01, 0.03, folds=[0.01, 0.02, 0.03, 0.02])
        ),
    )
    assert signal["outcome"] == "provisional"
    assert signal["passed"] is True
    assert signal["metrics"]["criteria_version"] == 2


def test_v2_control_that_passes_makes_the_run_invalid():
    verdict = evaluate_criteria(
        {"criteria": _v2_criteria()},
        _v2_payload(
            pooled=_paired(0.02, 0.01, 0.03),
            shuffle=_paired(0.02, 0.01, 0.03),
        ),
    )
    assert verdict["outcome"] == "invalid"
    assert any("shuffle control" in reason for reason in verdict["reasons"])


def test_v2_fold_rule_and_single_ticker_flag_are_explicit():
    one_bad_fold = evaluate_criteria(
        {"criteria": _v2_criteria()},
        _v2_payload(
            pooled=_paired(0.02, 0.01, 0.03, folds=[0.01, 0.02, -0.01, 0.02])
        ),
    )
    assert one_bad_fold["outcome"] == "negative"
    assert any("folds are positive" in reason for reason in one_bad_fold["reasons"])

    carried = evaluate_criteria(
        {"criteria": _v2_criteria()},
        _v2_payload(
            pooled=_paired(0.02, 0.01, 0.03),
            tickers=("BTC/USD", "ETH/USD", "SOL/USD"),
        ),
    )
    # The helper gives every ticker the same positive mean, so this passes;
    # a one-ticker result needs an explicit per-ticker distribution.
    payload = _v2_payload(pooled=_paired(0.02, 0.01, 0.03), tickers=("BTC/USD", "ETH/USD"))
    payload["results"][1]["paired"] = _paired(-0.01, -0.02, -0.001)
    single = evaluate_criteria({"criteria": _v2_criteria()}, payload)
    assert single["outcome"] == "negative"
    assert single["metrics"]["carried_by_one_ticker"] is True
    assert carried["outcome"] == "provisional"


def test_v2_requires_controls_and_alpha_matches_the_lab():
    missing = _v2_payload(pooled=_paired(0.02, 0.01, 0.03))
    missing["controls"] = {}
    verdict = evaluate_criteria({"criteria": _v2_criteria()}, missing)
    assert verdict["outcome"] == "blocked"
    assert any("missing control" in reason for reason in verdict["reasons"])

    wrong_alpha = evaluate_criteria(
        {"criteria": _v2_criteria(alpha=0.01)},
        _v2_payload(pooled=_paired(0.02, 0.01, 0.03, alpha=0.05)),
    )
    assert wrong_alpha["outcome"] == "negative"
    assert any("pre-registered alpha" in reason for reason in wrong_alpha["reasons"])


def test_v2_full_archive_diagnostic_cannot_feed_a_verdict():
    payload = _v2_payload(pooled=_paired(0.02, 0.01, 0.03))
    payload["diagnostic_full_archive"] = True
    verdict = evaluate_criteria({"criteria": _v2_criteria()}, payload)
    assert verdict["outcome"] == "blocked"
    assert any("full-archive" in reason for reason in verdict["reasons"])


def test_v2_canary_provisional_is_reported_so_the_driver_can_void_it():
    payload = _v2_payload(pooled=_paired(0.02, 0.01, 0.03))
    payload["canary"] = True
    verdict = evaluate_criteria({"criteria": _v2_criteria()}, payload)
    assert verdict["outcome"] == "provisional"


def test_v1_entries_keep_the_old_interpretation():
    # No criteria_version: the v1 best-row rule still applies.
    prereg = {
        "criteria": {
            **DEFAULT_CRITERIA,
            "target": "fwd_direction",
            "tickers_beating_min": 1,
        }
    }
    verdict = evaluate_criteria(
        prereg, {"results": [_lab_row(folds_beating=4, mean=0.02, minimum=0.01)]}
    )
    assert verdict["outcome"] == "provisional"
    assert "criteria_version" not in verdict["metrics"]


# --------------------------------------------------------------------------- holdout freeze


def test_holdout_cutoff_is_a_frozen_timestamp_and_a_rotate_moves_it_forward(tmp_path):
    from rl_crypto.experiments import holdout_cutoffs, resolve_holdout_start

    ledger = tmp_path / "ledger.jsonl"
    append_entry(
        ledger,
        {
            "kind": "holdout_freeze",
            "lineage": "H1",
            "holdout_start": "2026-06-01T00:00:00Z",
        },
    )
    entries = read_ledger(ledger)
    assert resolve_holdout_start(entries, "H1") == "2026-06-01T00:00:00Z"
    # Appending bars cannot move the recorded cutoff.
    _candles(tmp_path, rows=50)
    assert holdout_cutoffs(read_ledger(ledger)) == {"H1": "2026-06-01T00:00:00Z"}

    append_entry(
        ledger,
        {
            "kind": "holdout_rotate",
            "lineage": "H1",
            "holdout_start": "2026-07-01T00:00:00Z",
        },
    )
    assert resolve_holdout_start(read_ledger(ledger), "H1") == "2026-07-01T00:00:00Z"


def test_holdout_touches_count_runs_once_per_stable_lineage(tmp_path):
    from rl_crypto.experiments import holdout_touches

    ledger = tmp_path / "ledger.jsonl"
    for run in ("r1", "r1", "r2"):
        append_entry(
            ledger,
            {
                "kind": "holdout",
                "lineage": "H1",
                "run_id": run,
                "holdout_touched": True,
            },
        )
    assert holdout_touches(read_ledger(ledger)) == {"H1": 2}


def test_harness_void_is_cleared_by_a_later_human_entry(tmp_path):
    from rl_crypto.experiments import harness_void

    ledger = tmp_path / "ledger.jsonl"
    append_entry(ledger, {"kind": "harness_void", "reason": "canary"})
    assert harness_void(read_ledger(ledger)) is not None
    append_entry(ledger, {"kind": "harness_clear", "reason": "fixed"})
    assert harness_void(read_ledger(ledger)) is None


def test_the_canary_is_the_first_runnable_hypothesis_for_the_schedule():
    root = Path(__file__).resolve().parents[1]
    queue = load_queue(root / "experiments" / "hypotheses.json")
    selected = select_hypothesis(queue, [], capabilities=set())
    assert selected is not None
    assert selected.id == "H0-canary"
    assert selected.canary is True
    assert selected.needs == ()


# --------------------------------------------------------------------------- alpha spending


def test_alpha_spending_schedule_sums_to_the_family_alpha():
    from rl_crypto.experiments import (
        SPENDING_SCHEDULE,
        alpha_for_trial,
        smallest_alpha,
    )

    assert SPENDING_SCHEDULE == "alpha/(i*(i+1))"
    family_alpha = 0.05
    total = sum(alpha_for_trial(family_alpha, index) for index in range(1, 10_000))
    assert total == pytest.approx(family_alpha, abs=1e-4)
    assert alpha_for_trial(family_alpha, 1) == pytest.approx(family_alpha / 2)
    assert alpha_for_trial(family_alpha, 5) == pytest.approx(family_alpha / 30)
    assert alpha_for_trial(family_alpha, 10) == pytest.approx(family_alpha / 110)
    # The capped family never spends more than the declared alpha.
    assert sum(alpha_for_trial(family_alpha, i) for i in range(1, 11)) < family_alpha
    assert smallest_alpha(family_alpha, 10) == pytest.approx(family_alpha / 110)


def test_bootstrap_resolution_requires_the_capped_tail_to_be_resolvable():
    from rl_crypto.experiments import (
        MIN_BOOTSTRAP_BLOCKS,
        bootstrap_resolution,
        required_bootstrap_samples,
        smallest_alpha,
    )

    # alpha / (N(N+1)) at the default cap of 10 leaves a 2.27e-4 tail.
    smallest = smallest_alpha(0.05, 10)
    required = required_bootstrap_samples(smallest)
    assert required >= 8_800
    assert MIN_BOOTSTRAP_BLOCKS >= 24
    assert bootstrap_resolution(0.05, 10, 1_000)["ok"] is False
    assert bootstrap_resolution(0.05, 10, required)["ok"] is True
    # A cap of 3 is reachable with fewer resamples; the check is not fixed.
    assert bootstrap_resolution(0.05, 3, 1_000)["ok"] is True


def test_v2_blocks_when_the_bootstrap_cannot_resolve_the_tail():
    pooled = {
        "mean_improvement": 0.02,
        "ci_low": 0.01,
        "ci_high": 0.03,
        "n": 100,
        "alpha": 0.025,
        "fold_improvements": [0.01, 0.02, 0.01, 0.02],
        "insufficient": True,
        "resolution_reason": "only 10 time blocks (need 24)",
    }
    verdict = evaluate_criteria({"criteria": _v2_criteria()}, _v2_payload(pooled=pooled))
    assert verdict["outcome"] == "blocked"
    assert any(reason.startswith("insufficient-resolution") for reason in verdict["reasons"])


def test_preregistration_records_the_schedule_and_per_trial_alpha(tmp_path):
    from rl_crypto.experiments import SPENDING_SCHEDULE

    _candles(tmp_path, rows=7)
    hypothesis = _hypothesis(
        id="H9",
        family="alpha-test",
        criteria={
            **_v2_criteria(),
            "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        },
    )
    entries = [
        {"kind": "preregister", "hypothesis_id": f"H{index}", "family": "alpha-test"}
        for index in range(4)
    ]
    prereg = build_preregistration(
        hypothesis,
        run_id="20260101T000000Z-h9",
        root=tmp_path,
        entries=entries,
        seed=0,
        holdout_start="2026-01-02T00:00:00Z",
    )
    assert prereg["trial_index"] == 5
    assert prereg["spending_schedule"] == SPENDING_SCHEDULE
    assert prereg["alpha_family"] == pytest.approx(0.05)
    assert prereg["alpha_trial"] == pytest.approx(0.05 / 30)


def test_power_at_trial_1_5_10_declines_with_the_spending_schedule():
    """A fixed effect loses power as the family spends its alpha; the test
    records the three points the schedule is designed around."""
    from statistics import NormalDist

    from rl_crypto.experiments import alpha_for_trial

    normal = NormalDist()
    effect = 0.0026
    se = 0.001  # sigma=0.02, n=400 paired rows

    def power(trial: int) -> float:
        alpha = alpha_for_trial(0.05, trial)
        critical = normal.inv_cdf(1.0 - alpha / 2.0)
        z = effect / se
        return 1.0 - (normal.cdf(critical - z) - normal.cdf(-critical - z))

    powers = {trial: power(trial) for trial in (1, 5, 10)}
    assert powers[1] > powers[5] > powers[10] > 0.05
    # At the cap the required per-trial alpha is tiny, so the effect must be
    # larger to retain the same power; that is the cost of the guarantee.
    assert powers[1] > 0.6


# --------------------------------------------------------------------------- canaries and economics


def test_v2_leaky_feature_is_invalid_not_provisional():
    payload = _v2_payload(pooled=_paired(0.02, 0.01, 0.03))
    payload["leak_detected"] = True
    payload["leak_features"] = [
        {"feature": "leak_next_bar_return", "target": "fwd_return", "correlation": 1.0}
    ]
    verdict = evaluate_criteria({"criteria": _v2_criteria()}, payload)
    assert verdict["outcome"] == "invalid"
    assert any("leak" in reason for reason in verdict["reasons"])


def test_v2_direction_target_requires_pooled_economics():
    payload = _v2_payload(pooled=_paired(0.02, 0.01, 0.03))
    payload.pop("pooled_economics")
    verdict = evaluate_criteria({"criteria": _v2_criteria()}, payload)
    assert verdict["outcome"] == "blocked"
    assert any("requires pooled economics" in reason for reason in verdict["reasons"])


def test_v2_economics_is_the_pooled_portfolio_not_the_first_row():
    payload = _v2_payload(pooled=_paired(0.02, 0.01, 0.03))
    # The first ticker row alone loses money; the pooled portfolio does not.
    payload["results"][0]["economic"] = {
        "model": {"total_return": -0.5, "sharpe": -3.0, "trades": 5},
        "momentum": {"sharpe": 0.0},
    }
    ok = evaluate_criteria({"criteria": _v2_criteria()}, payload)
    assert ok["outcome"] == "provisional"
    assert ok["metrics"]["economic"]["basis"] == "pooled-portfolio"
    assert ok["metrics"]["economic"]["ticker"] is None

    payload["pooled_economics"]["model"] = {
        "total_return": -0.01,
        "sharpe": 1.5,
        "trades": 200,
    }
    bad = evaluate_criteria({"criteria": _v2_criteria()}, payload)
    assert bad["outcome"] == "negative"
    assert any("total return" in reason for reason in bad["reasons"])


def test_canary_kind_is_explicit_and_gated():
    null = Hypothesis.from_dict(
        {
            "id": "C1",
            "title": "null",
            "family": "canary",
            "experiment": {"tickers": ["BTC/USD"], "targets": ["fwd_direction"]},
            "criteria": _v2_criteria(),
            "canary": True,
        }
    )
    assert null.canary_kind == "null"
    positive = Hypothesis.from_dict(
        {
            "id": "C2",
            "title": "positive",
            "family": "canary",
            "experiment": {"tickers": ["BTC/USD"], "targets": ["fwd_direction"]},
            "criteria": _v2_criteria(),
            "canary": True,
            "canary_kind": "positive",
        }
    )
    assert positive.canary_kind == "positive"
    with pytest.raises(ValueError, match="canary_kind"):
        Hypothesis.from_dict(
            {
                "id": "C3",
                "title": "bad",
                "family": "canary",
                "experiment": {"tickers": ["BTC/USD"], "targets": ["fwd_direction"]},
                "criteria": _v2_criteria(),
                "canary_kind": "sideways",
            }
        )


def test_no_new_information_default_is_168_bars_and_configurable():
    payload = json.loads(
        (Path(__file__).parents[1] / "config" / "model-improvement.json").read_text()
    )
    assert payload["noNewInformation"]["minNewBars"] == 168
    assert payload["budgets"]["maxTrialsPerFamily"] == 10
