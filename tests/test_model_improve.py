"""Offline tests for the unattended model-improvement orchestrator.

The real ``opencode`` binary and the real lab are never invoked. A tiny fake
opencode writes the role artifacts (and proves stdin is closed), and a fake lab
emits a deterministic payload, so the tests exercise the orchestration and the
mechanical verdict without spending credits or touching the archive.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "model_improve", Path(__file__).resolve().parents[1] / "tools" / "model_improve.py"
)
assert _SPEC is not None and _SPEC.loader is not None
tool = importlib.util.module_from_spec(_SPEC)
sys.modules["model_improve"] = tool
_SPEC.loader.exec_module(tool)

AGENTS = list(tool.ROLE_AGENTS.values())

FAKE_LAB = '''\
import json, os, sys

action = sys.argv[1]
outcome = os.environ.get("FAKE_LAB_OUTCOME", "pass")
run_dir = os.environ.get("RL_MODEL_IMPROVE_RUN_DIR")
if run_dir:
    with open(os.path.join(run_dir, "lab-argv.json"), "w", encoding="utf-8") as fh:
        json.dump(sys.argv, fh)
    with open(os.path.join(run_dir, f"lab-argv-{action}.json"), "w", encoding="utf-8") as fh:
        json.dump(sys.argv, fh)
folds_beating = 4 if outcome == "pass" else 1
mean = 0.02 if outcome == "pass" else -0.01
minimum = 0.01 if outcome == "pass" else -0.02


def row(ticker="BTC/USD"):
    return {
        "ticker": ticker,
        "config": {"name": "all+reference+h1", "horizon": 1},
        "targets": {
            "fwd_direction": {
                "folds": 4,
                "folds_beating": folds_beating,
                "beats_ratio": folds_beating / 4,
                "mean_improvement": mean,
                "min_improvement": minimum,
            },
            "fwd_volume_ratio": {
                "folds": 4,
                "folds_beating": 4,
                "beats_ratio": 1.0,
                "mean_improvement": 0.1,
                "min_improvement": 0.05,
            },
        },
        "economic": {
            "model": {"total_return": 0.4, "sharpe": 1.1, "trades": 150},
            "momentum": {"total_return": 0.1, "sharpe": 0.2, "trades": 150},
            "always_long": {"total_return": 0.05, "sharpe": 0.1, "trades": 1},
        },
    }


result = row()
pooled = {
    "mean_improvement": 0.02 if outcome == "pass" else -0.01,
    "ci_low": 0.01 if outcome == "pass" else -0.02,
    "ci_high": 0.03 if outcome == "pass" else 0.0,
    "n": 100,
    "alpha": float(sys.argv[sys.argv.index("--alpha") + 1]) if "--alpha" in sys.argv else 0.05,
    "fold_improvements": [0.02, 0.02, 0.02, 0.02] if outcome == "pass" else [-0.01] * 4,
}
null = {
    "mean_improvement": -0.01,
    "ci_low": -0.02,
    "ci_high": 0.0,
    "n": 100,
    "alpha": pooled["alpha"],
    "fold_improvements": [-0.01] * 4,
}
controls = {"shuffle": {"pooled": null}, "shift": {"pooled": null}}
if action == "evaluate":
    payload = {
        "action": "evaluate",
        "primary": "all+reference+h1",
        "results": [result],
        "pooled": pooled,
        "pooled_economics": {
            "model": {"total_return": 0.4, "sharpe": 1.1, "trades": 150},
            "momentum": {"total_return": 0.05, "sharpe": 0.1, "trades": 150},
        },
        "controls": controls,
        "summary": {"runs": 1},
    }
elif action in {"sweep", "promote"}:
    candidate = os.path.join(os.environ.get("RUN_DIR", "."), "candidate.json")
    payload = {
        "action": action,
        "results": [result],
        "top": [result],
        "candidates": [candidate],
        "primary_candidates": [candidate],
        "primary": "all+reference+h1",
        "pooled": pooled,
        "passed": True,
        "reasons": [],
        "summary": {"runs": 1, "gated": 1},
    }
elif action == "status":
    payload = {"action": "status", "tickers": [{"ticker": "BTC/USD", "bars": 150}]}
else:
    payload = {"action": action, "results": []}
print(json.dumps(payload))
'''

FAKE_OPENCODE = '''\
#!/usr/bin/env python3
import json, os, pathlib, sys

args = sys.argv[1:]
print(json.dumps({"type": "step_start", "sessionID": "fake-session"}))
agent = args[args.index("--agent") + 1]
role = agent.replace("model-improve-", "")
run_dir = pathlib.Path(os.environ["RL_MODEL_IMPROVE_RUN_DIR"])
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / f"argv-{role}.json").write_text(json.dumps(args), encoding="utf-8")
(run_dir / f"stdin-{role}.txt").write_text(sys.stdin.read(), encoding="utf-8")
ready = os.environ.get("FAKE_RESEARCHER_READY", "true") == "true"

if role == "researcher":
    (run_dir / "diagnosis.md").write_text("# diagnosis\\n", encoding="utf-8")
    (run_dir / "researcher.json").write_text(
        json.dumps(
            {
                "hypothesis_id": "H0",
                "ready": ready,
                "recommendation": "proceed" if ready else "stop",
                "blockers": [] if ready else ["archive missing"],
                "summary": "fake",
            }
        ),
        encoding="utf-8",
    )
elif role == "implementer":
    (run_dir / "implementation.md").write_text("# implementation\\n", encoding="utf-8")
    (run_dir / "implementation.json").write_text(
        json.dumps({"status": "done", "summary": "fake", "tests": [], "files": []}),
        encoding="utf-8",
    )
elif role == "auditor":
    (run_dir / "audit.json").write_text(
        json.dumps({"blocking": False, "findings": []}), encoding="utf-8"
    )
elif role == "lead":
    # A headless lead can end its first turn before writing lead.json. On the
    # resumed turn (--session present) it produces the full team artifacts.
    if "--session" in args:
        (run_dir / "diagnosis.md").write_text("# diagnosis\\n", encoding="utf-8")
        (run_dir / "researcher.json").write_text(
            json.dumps(
                {
                    "hypothesis_id": "H0",
                    "ready": True,
                    "recommendation": "proceed",
                    "blockers": [],
                    "summary": "fake",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "lead.json").write_text(
            json.dumps({"status": "done", "summary": "fake", "artifacts": {}}),
            encoding="utf-8",
        )
elif role == "reviewer":
    (run_dir / "review.md").write_text("# review\\n", encoding="utf-8")
    (run_dir / "review.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "summary": "fake review",
                "findings": [],
                "config_changes": [],
                "next_actions": [],
            }
        ),
        encoding="utf-8",
    )
'''


def _executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _candles(root: Path, ticker: str = "BTC/USD", rows: int = 150) -> None:
    slug = ticker.replace("/", "-").lower()
    path = root / "data" / slug / "candles.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            stamp = (start + timedelta(hours=index)).strftime("%Y-%m-%dT%H:%M:%SZ")
            handle.write(
                json.dumps(
                    {
                        "ticker": ticker,
                        "timestamp": stamp,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 10.0,
                    }
                )
                + "\n"
            )


def _hypothesis(hypothesis_id: str = "H0", *, needs: list[str] | None = None) -> dict:
    return {
        "id": hypothesis_id,
        "title": "test",
        "family": "diagnostic",
        "order": 0,
        "status": "open",
        "needs": needs or [],
        "mechanism": "because",
        "experiment": {
            "action": "evaluate",
            "tickers": ["BTC/USD"],
            "targets": ["fwd_direction"],
            "featureSet": "all",
            "modelSet": "reference",
            "horizon": 1,
            "folds": 4,
            "maxConfigs": 1,
        },
        "criteria": {"kind": "diagnostic"},
    }


def _repo(tmp_path: Path, hypothesis: dict) -> Path:
    root = tmp_path / "repo"
    for directory in (
        root / "config",
        root / "experiments",
        root / "models",
        root / ".opencode" / "agents",
        root / "tools",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (root / "config" / "model-improvement.json").write_text(
        json.dumps(
            {
                "schema": "rl-crypto/model-improvement/1",
                "promotion": {"apply": False},
                "scope": {"tickers": ["BTC/USD"]},
                "budgets": {
                    "maxConfigs": 1,
                    "folds": 4,
                    "labTimeoutSeconds": 30,
                    # 150-bar fixture: 4-bar blocks still clear the 24-block floor.
                    "bootstrapBlockBars": 4,
                    "bootstrapSamples": 10000,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "experiments" / "hypotheses.json").write_text(
        json.dumps({"hypotheses": [hypothesis]}), encoding="utf-8"
    )
    (root / "experiments" / "ledger.jsonl").write_text("", encoding="utf-8")
    (root / "experiments" / "SCOREBOARD.md").write_text("stale\n", encoding="utf-8")
    for name in AGENTS:
        (root / ".opencode" / "agents" / f"{name}.md").write_text("", encoding="utf-8")
    _candles(root)
    return root


def _seed_canaries(root: Path) -> None:
    """Record both canaries as passing at the fixture's current contract hash."""
    current = tool.protected_digest(root)["hash"]
    for kind, outcome in (("null", "negative"), ("positive", "invalid")):
        tool.append_entry(
            root / "experiments" / "ledger.jsonl",
            {
                "kind": "verdict",
                "run_id": f"canary-{kind}",
                "hypothesis_id": f"H0-canary-{kind}",
                "canary": True,
                "canary_kind": kind,
                "outcome": outcome,
                "passed": False,
                "protected_hash": current,
                "at": "2026-10-24T00:00:00Z",
            },
        )


def _env(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    tmp_path: Path,
    *,
    fake_opencode: bool,
    canaries: bool = True,
):
    fake_lab = tmp_path / "fake_lab.py"
    fake_lab.write_text(FAKE_LAB, encoding="utf-8")
    monkeypatch.setenv("RL_CRYPTO_ROOT", str(root))
    monkeypatch.setenv("RL_MODEL_IMPROVE_RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RL_MODEL_IMPROVE_LAB", f"{sys.executable} {fake_lab}")
    if fake_opencode:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        _executable(bin_dir / "opencode", FAKE_OPENCODE)
        monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    if canaries:
        _seed_canaries(root)


# --------------------------------------------------------------------------- config/preflight


def test_config_deep_merges_over_defaults(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "model-improvement.json").write_text(
        json.dumps(
            {
                "schema": "rl-crypto/model-improvement/1",
                "budgets": {"maxConfigs": 3},
            }
        ),
        encoding="utf-8",
    )
    config = tool.load_config(root)
    assert config["budgets"]["maxConfigs"] == 3
    assert config["budgets"]["folds"] == 4  # inherited default
    assert config["scope"]["tickers"] == ["BTC/USD", "ETH/USD", "SOL/USD"]


def test_config_rejects_unknown_schema(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "model-improvement.json").write_text(
        json.dumps({"schema": "nope"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema"):
        tool.load_config(root)


def test_preflight_blocks_missing_data_and_allows_mechanical_without_agents(tmp_path):
    hypothesis = _hypothesis()
    root = _repo(tmp_path, hypothesis)
    (root / "data" / "btc-usd" / "candles.jsonl").unlink()
    for name in AGENTS:
        (root / ".opencode" / "agents" / f"{name}.md").unlink()
    payload = tool.preflight(root, tool.load_config(root), executor="mechanical")
    assert payload["ok"] is False
    assert "data:BTC/USD" in payload["blocking"]
    assert not any(name == "agents" for name in payload["blocking"])


def test_staleness_parses_iso_timestamps():
    now = datetime.now(UTC)
    recent = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert tool._staleness(recent) == pytest.approx(2.0, abs=0.1)
    assert tool._staleness("not-a-date") is None
    assert tool._staleness(None) is None


# --------------------------------------------------------------------------- plan


def test_plan_shows_frozen_criteria_without_writing_the_ledger(
    tmp_path, monkeypatch, capsys
):
    root = _repo(tmp_path, _hypothesis())
    _env(monkeypatch, root, tmp_path, fake_opencode=False, canaries=False)
    rc = tool.main(["plan", "--executor", "mechanical", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hypothesis"]["id"] == "H0"
    assert payload["preregister"]["criteria"]["kind"] == "diagnostic"
    assert (root / "experiments" / "ledger.jsonl").read_text() == ""


def test_experiment_run_refuses_without_fresh_canaries(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path, _hypothesis())
    _env(monkeypatch, root, tmp_path, fake_opencode=False, canaries=False)
    rc = tool.main(
        [
            "run",
            "--executor",
            "mechanical",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "blocked"
    assert "canary-freshness" in payload["reason"]
    entries = [
        json.loads(line)
        for line in (root / "experiments" / "ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [entry["kind"] for entry in entries] == ["blocked"]
    assert entries[0]["infra"] is True
    assert entries[0]["infra_kind"] == "canary-stale"

    # A harness-change run may proceed while the contract itself is in flux.
    rc = tool.main(
        [
            "run",
            "--executor",
            "mechanical",
            "--run-type",
            "harness-change",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 0


# --------------------------------------------------------------------------- run: mechanical


def test_mechanical_diagnostic_run_records_prereg_results_and_verdict(
    tmp_path, monkeypatch, capsys
):
    root = _repo(tmp_path, _hypothesis())
    _env(monkeypatch, root, tmp_path, fake_opencode=False)
    rc = tool.main(
        [
            "run",
            "--executor",
            "mechanical",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["outcome"] == "diagnostic"
    entries = [
        json.loads(line)
        for line in (root / "experiments" / "ledger.jsonl").read_text().splitlines()
        if line.strip() and not json.loads(line).get("canary")
    ]
    assert [entry["kind"] for entry in entries] == [
        "holdout_freeze",
        "preregister",
        "results",
        "verdict",
        "review",
    ]
    assert entries[1]["status"] == "preregistered"
    assert entries[3]["outcome"] == "diagnostic"
    assert entries[3]["metrics"]["targets"] == ["fwd_direction", "fwd_volume_ratio"]
    assert "BTC/USD" in entries[3]["metrics"]["per_ticker"]
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "report.md").exists()
    assert (run_dir / "preregister.json").exists()
    assert (run_dir / "review.json").exists()
    assert json.loads((run_dir / "review.json").read_text())["status"] == "pass"
    assert json.loads((run_dir / "lab-evaluate.json").read_text())["action"] == "evaluate"
    report = (run_dir / "report.md").read_text()
    assert "diagnostic completed" in report
    assert "all pre-registered bars met" not in report
    assert "diagnostic" in (root / "experiments" / "SCOREBOARD.md").read_text()

    # The same diagnostic on the same config/data is a duplicate unless forced.
    rc2 = tool.main(
        [
            "run",
            "--executor",
            "mechanical",
            "--hypothesis",
            "H0",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc2 == 0
    duplicate = json.loads(capsys.readouterr().out)
    assert duplicate["outcome"] == "duplicate-diagnostic"


def test_failed_improvement_gate_is_a_negative_success(tmp_path, monkeypatch, capsys):
    hypothesis = _hypothesis("D1")
    hypothesis["experiment"]["action"] = "sweep"
    hypothesis["experiment"]["extraLabArgs"] = ["--panel"]
    hypothesis["criteria"] = {
        "kind": "improvement",
        "target": "fwd_direction",
        "tickers_beating_min": 1,
        "folds_beating_min": 3,
        "mean_improvement_min": 0.0,
        "min_improvement_min": 0.0,
        "economic": {
            "total_return_min": 0.0,
            "sharpe_above_momentum": True,
            "trades_min": 100,
        },
    }
    root = _repo(tmp_path, hypothesis)
    _env(monkeypatch, root, tmp_path, fake_opencode=False)
    monkeypatch.setenv("FAKE_LAB_OUTCOME", "fail")
    rc = tool.main(
        [
            "run",
            "--executor",
            "mechanical",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["outcome"] == "negative"
    assert summary["passed"] is False
    assert any("tickers met the improvement bar" in reason for reason in summary["reasons"])
    # The hypothesis's declared interface is actually passed to the lab.
    lab_argv = json.loads((Path(summary["run_dir"]) / "lab-argv.json").read_text())
    assert "--panel" in lab_argv


# --------------------------------------------------------------------------- run: passes


def test_passes_runs_a_closed_stdin_headless_role(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path, _hypothesis())
    _env(monkeypatch, root, tmp_path, fake_opencode=True)
    rc = tool.main(
        [
            "run",
            "--executor",
            "passes",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    run_dir = Path(summary["run_dir"])
    roles = json.loads((run_dir / "roles.json").read_text())
    assert [role["role"] for role in roles] == ["researcher", "reviewer"]
    assert roles[0]["status"] == "done"
    argv = json.loads((run_dir / "argv-researcher.json").read_text())
    assert "--auto" in argv
    assert "--format" in argv and "json" in argv
    assert "--agent" in argv and "model-improve-researcher" in argv
    # stdin is DEVNULL, so the fake's sys.stdin.read() returns immediately.
    assert (run_dir / "stdin-researcher.txt").read_text() == ""


def test_researcher_stop_blocks_with_exit_code_2(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path, _hypothesis())
    _env(monkeypatch, root, tmp_path, fake_opencode=True)
    monkeypatch.setenv("FAKE_RESEARCHER_READY", "false")
    rc = tool.main(
        [
            "run",
            "--executor",
            "passes",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["outcome"] == "blocked"
    entries = [
        json.loads(line)
        for line in (root / "experiments" / "ledger.jsonl").read_text().splitlines()
        if line.strip() and not json.loads(line).get("canary")
    ]
    assert [entry["kind"] for entry in entries] == [
        "holdout_freeze",
        "preregister",
        "blocked",
        "review",
    ]
    assert "researcher" in entries[2]["reason"]
    assert entries[2]["infra"] is True
    assert entries[2]["infra_kind"] == "role-error"
    assert tool.trial_counts(entries)["family"] == {}


# --------------------------------------------------------------------------- git safety


def test_commit_run_refuses_main(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init", "-b", "main", str(root)], check=True, capture_output=True)
    (root / "file.txt").write_text("x", encoding="utf-8")
    subprocess.run(
        ["git", "add", "."], cwd=root, check=True, capture_output=True, env={**os.environ, **env}
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
        env={**os.environ, **env},
    )
    ctx = tool.Context(
        root=root,
        config={"promotion": {"apply": False}},
        executor="mechanical",
        trigger="test",
        run_id="r1",
        run_dir=tmp_path / "runs",
        hypothesis=tool.Hypothesis.from_dict(_hypothesis()),
    )
    with pytest.raises(RuntimeError, match="refusing to commit the run on main"):
        tool.commit_run(ctx, {"outcome": "negative"}, allow_main=False)


def test_executor_auto_falls_back_to_passes_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("RL_MODEL_IMPROVE_EXECUTOR", raising=False)
    assert tool.resolve_executor("auto") == "passes"
    monkeypatch.setenv("RL_MODEL_IMPROVE_EXECUTOR", "mechanical")
    assert tool.resolve_executor("auto") == "mechanical"
    assert tool.resolve_executor("auto", no_llm=True) == "mechanical"


# --------------------------------------------------------------------------- model + review


def test_model_has_one_knob_and_a_per_role_override(monkeypatch):
    monkeypatch.delenv("RL_MODEL_IMPROVE_MODEL", raising=False)
    assert tool.resolve_model({}) == "opencode/deepseek-v4.1-flash-free"
    config = {"model": "provider/default", "models": {"researcher": "provider/special"}}
    assert tool.resolve_model(config, "researcher") == "provider/special"
    assert tool.resolve_model(config, "auditor") == "provider/default"
    # A one-run CLI override wins over a per-role exception.
    tool.apply_model_override(config, "provider/cli")
    assert tool.resolve_model(config, "researcher") == "provider/cli"
    monkeypatch.setenv("RL_MODEL_IMPROVE_MODEL", "provider/env")
    assert tool.resolve_model(config, "researcher") == "provider/env"


def test_mechanical_review_flags_data_or_model_changes(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(
        ["git", "init", "-b", "model-improve/test", str(root)],
        check=True,
        capture_output=True,
    )
    (root / "data").mkdir()
    (root / "data" / "x").write_text("1", encoding="utf-8")
    subprocess.run(
        ["git", "add", "."], cwd=root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )
    (root / "data" / "x").write_text("2", encoding="utf-8")
    (root / "experiments").mkdir()
    ledger = root / "experiments" / "ledger.jsonl"
    tool.append_entry(
        ledger,
        {"kind": "preregister", "run_id": "r1", "hypothesis_id": "H1"},
    )
    tool.append_entry(
        ledger,
        {"kind": "verdict", "run_id": "r1", "hypothesis_id": "H1", "outcome": "diagnostic"},
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "preregister.json").write_text("{}", encoding="utf-8")
    ctx = tool.Context(
        root=root,
        config={"promotion": {"apply": False}, "review": {"enabled": True}},
        executor="mechanical",
        trigger="test",
        run_id="r1",
        run_dir=run_dir,
        prereg={"id": "e0001", "criteria": {}},
        worktree=root,
    )
    review = tool.mechanical_review(ctx, {"outcome": "diagnostic"})
    assert review["status"] == "fail"
    scope = next(check for check in review["checks"] if check["name"] == "scope")
    assert scope["ok"] is False
    assert "data/x" in scope["detail"]


def test_repo_config_has_one_model_knob():
    root = Path(__file__).resolve().parents[1]
    config = tool.load_config(root)
    assert config["model"] == "opencode/deepseek-v4.1-flash-free"
    assert config["models"] == {"auditor": "opencode/claude-haiku-4-5"}
    assert tool.resolve_model(config, "auditor") == "opencode/claude-haiku-4-5"
    assert tool.resolve_model(config, "researcher") == "opencode/deepseek-v4.1-flash-free"
    assert config["review"]["enabled"] is True


def test_fallback_detector_ignores_a_transcript_echo():
    transcript = json.dumps(
        {
            "type": "tool_use",
            "part": {
                "state": {
                    "output": "tools/model_improve.py: is a subagent, not a primary agent"
                }
            },
        }
    )
    assert tool.agent_fallback_banner(transcript + "\n", "") is False
    banner = (
        '! agent "x" is a subagent, not a primary agent. Falling back to default agent\n'
        '{"type":"step_start"}\n'
    )
    assert tool.agent_fallback_banner("", banner) is True


def test_team_lead_resumes_a_turn_that_ended_without_lead_json(
    tmp_path, monkeypatch, capsys
):
    root = _repo(tmp_path, _hypothesis())
    _env(monkeypatch, root, tmp_path, fake_opencode=True)
    rc = tool.main(
        [
            "run",
            "--executor",
            "team",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["review"]["status"] == "pass"
    roles = json.loads((Path(summary["run_dir"]) / "roles.json").read_text())
    lead_turns = [role for role in roles if role["role"] == "lead"]
    assert len(lead_turns) == 2
    assert lead_turns[0]["missing"] == ["lead.json"]
    assert lead_turns[0]["turn"] == 1
    assert lead_turns[1]["turn"] == 2
    assert lead_turns[1]["status"] == "done"


# --------------------------------------------------------------------------- hardening


def _ctx(tmp_path, hypothesis=None, **overrides):
    root = _repo(tmp_path, hypothesis or _hypothesis())
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    config = tool.load_config(root)
    ctx = tool.Context(
        root=root,
        config=config,
        executor="mechanical",
        trigger="test",
        run_id="20260101T000000Z-test",
        run_dir=run_dir,
        hypothesis=tool.Hypothesis.from_dict(hypothesis or _hypothesis()),
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def test_mechanical_review_fails_a_run_that_edits_a_protected_file(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.prereg = {
        "id": "e1",
        "lineage": ctx.lineage,
        "git": {"sha": "deadbeef"},
        "criteria": {"kind": "diagnostic"},
        "protected": {},
    }
    (ctx.run_dir / "preregister.json").write_text(json.dumps(ctx.prereg), encoding="utf-8")
    tool.append_entry(
        ctx.ledger,
        {"kind": "preregister", "run_id": ctx.run_id, "hypothesis_id": "H0"},
    )
    tool.append_entry(
        ctx.ledger,
        {"kind": "verdict", "run_id": ctx.run_id, "outcome": "negative"},
    )
    monkeypatch.setattr(tool, "protected_changes", lambda _ctx: ["tools/model_lab.py"])
    review = tool.mechanical_review(ctx, {"outcome": "negative", "passed": False})
    assert review["status"] == "fail"
    check = next(item for item in review["checks"] if item["name"] == "protected-files")
    assert check["ok"] is False


def test_harness_change_is_accepted_but_never_provisional(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, run_type="harness-change")
    ctx.prereg = {
        "id": "e1",
        "lineage": ctx.lineage,
        "git": {"sha": "deadbeef"},
        "criteria": {"kind": "diagnostic"},
        "protected": {},
    }
    (ctx.run_dir / "preregister.json").write_text(json.dumps(ctx.prereg), encoding="utf-8")
    tool.append_entry(
        ctx.ledger,
        {"kind": "preregister", "run_id": ctx.run_id, "hypothesis_id": "H0"},
    )
    tool.append_entry(
        ctx.ledger,
        {"kind": "verdict", "run_id": ctx.run_id, "outcome": "harness-change"},
    )
    monkeypatch.setattr(tool, "protected_changes", lambda _ctx: ["tools/model_lab.py"])
    review = tool.mechanical_review(ctx, {"outcome": "harness-change", "passed": False})
    assert review["status"] == "warn"
    protected = next(item for item in review["checks"] if item["name"] == "protected-files")
    assert protected["ok"] is True
    approval = next(item for item in review["checks"] if item["name"] == "human-approval")
    assert approval["ok"] is False

    verdict = tool.evaluate_mechanical(ctx, ctx.prereg)
    assert verdict["outcome"] == "harness-change"
    assert verdict["passed"] is False


def test_role_and_wall_clock_ceilings_block_cleanly(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.executor = "passes"
    ctx.config["budgets"]["maxRolesPerRun"] = 0
    record = tool.run_role(ctx, "researcher")
    assert record["status"] == "failed"
    assert "role ceiling" in record["reason"]

    ctx.config["budgets"]["maxWallClockSeconds"] = 1
    ctx.started = 0.0
    reason = tool.ceiling_reason(ctx)
    assert reason is not None and "wall-clock" in reason


def test_no_new_information_blocks_an_identical_family_rerun(tmp_path):
    hypothesis = _hypothesis("H1", needs=["code"])
    hypothesis["family"] = "features"
    hypothesis["experiment"]["action"] = "sweep"
    hypothesis["experiment"]["primary"] = {
        "featureSet": "all",
        "modelSet": "reference",
        "horizon": 1,
    }
    hypothesis["experiment"]["maxConfigs"] = 2
    hypothesis["criteria"] = {"kind": "improvement", "criteria_version": 2, "target": "fwd_direction"}
    ctx = _ctx(tmp_path, hypothesis)
    prior = {
        "kind": "preregister",
        "family": "features",
        "feature_version": tool.FEATURE_VERSION,
        "config_hash": tool.sha_of(hypothesis["experiment"]),
        "data": {"tickers": {"BTC/USD": {"rows": 150}}},
    }
    reason = tool.no_new_information_reason(
        ctx.root, ctx.hypothesis, [prior], min_new_bars=24
    )
    assert reason is not None and "no-new-information" in reason

    changed = dict(hypothesis)
    changed_experiment = dict(hypothesis["experiment"])
    changed_experiment["maxConfigs"] = 3
    changed["experiment"] = changed_experiment
    changed_hypothesis = tool.Hypothesis.from_dict(changed)
    assert (
        tool.no_new_information_reason(
            ctx.root, changed_hypothesis, [prior], min_new_bars=24
        )
        is None
    )


def test_an_infra_blocked_run_does_not_block_a_retry_as_no_new_information(tmp_path):
    ctx = _ctx(tmp_path)
    tool.append_entry(
        ctx.ledger,
        {
            "kind": "preregister",
            "run_id": "infra",
            "hypothesis_id": "H0",
            "family": "diagnostic",
        },
    )
    tool.append_entry(
        ctx.ledger,
        {
            "kind": "blocked",
            "run_id": "infra",
            "hypothesis_id": "H0",
            "stage": "researcher",
            "reason": "credential failure",
            "infra": True,
            "infra_kind": "role-error",
        },
    )
    assert tool.previous_family_prereg(tool.read_ledger(ctx.ledger), "diagnostic") is None

    tool.append_entry(
        ctx.ledger,
        {
            "kind": "preregister",
            "run_id": "real",
            "hypothesis_id": "H0",
            "family": "diagnostic",
        },
    )
    tool.append_entry(
        ctx.ledger,
        {
            "kind": "results",
            "run_id": "real",
            "hypothesis_id": "H0",
            "family": "diagnostic",
        },
    )
    previous = tool.previous_family_prereg(tool.read_ledger(ctx.ledger), "diagnostic")
    assert previous is not None and previous["run_id"] == "real"


def test_canary_provisional_is_converted_to_invalid_and_voids_the_harness(tmp_path, monkeypatch):
    hypothesis = _hypothesis("H0-canary")
    hypothesis["family"] = "canary"
    hypothesis["canary"] = True
    hypothesis["experiment"] = {
        "action": "evaluate",
        "tickers": ["BTC/USD"],
        "targets": ["fwd_direction"],
        "featureSet": "all",
        "modelSet": "reference",
        "horizon": 1,
        "folds": 4,
        "maxConfigs": 1,
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
    }
    hypothesis["criteria"] = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": False,
        "requires_promote": False,
    }
    ctx = _ctx(tmp_path, hypothesis, run_type="experiment")
    ctx.holdout_start = "2026-06-01T00:00:00Z"
    ctx.prereg = {
        "id": "e1",
        "kind": "preregister",
        "run_id": ctx.run_id,
        "hypothesis_id": "H0-canary",
        "family": "canary",
        "lineage": "H0-canary",
        "trial_index": 1,
        "criteria_version": 2,
        "holdout_start": ctx.holdout_start,
        "criteria": hypothesis["criteria"],
        "experiment": hypothesis["experiment"],
    }
    pooled = {
        "mean_improvement": 0.02,
        "ci_low": 0.01,
        "ci_high": 0.03,
        "n": 100,
        "alpha": 0.025,
        "fold_improvements": [0.01, 0.02, 0.01, 0.02],
    }
    monkeypatch.setattr(
        tool,
        "run_primary_evaluate",
        lambda *_args, **_kwargs: {
            "action": "evaluate",
            "controls": {"shuffle": {"pooled": pooled}},
            "results": [{"ticker": "BTC/USD", "paired": pooled}],
        },
    )
    verdict = tool._evaluate_mechanical_v2(ctx, ctx.prereg)
    assert verdict["outcome"] == "invalid"
    from rl_crypto.experiments import (
        CANARY_FILENAME,
        harness_void,
        read_canary,
        read_ledger,
    )

    entries = read_ledger(ctx.ledger)
    assert harness_void(entries) is not None
    canaries = read_canary(ctx.repo / "experiments" / CANARY_FILENAME)
    recorded = canaries[-1]
    assert recorded["canary"] is True
    assert recorded["canary_kind"] == "null"
    # The canary verdict is not a ledger verdict and never counts as a trial.
    assert not [entry for entry in entries if entry.get("kind") == "verdict"]


def test_scoreboard_renders_old_entries_and_adds_new_counts():
    root = Path(__file__).resolve().parents[1]
    from rl_crypto.experiments import read_ledger, render_scoreboard

    entries = read_ledger(root / "experiments" / "ledger.jsonl")
    rendered = render_scoreboard(entries)
    vocabulary = {
        "negative",
        "provisional",
        "confirmed",
        "refuted",
        "invalid",
        "diagnostic",
        "harness-change",
    }
    for entry in entries:
        if entry.get("kind") != "verdict":
            continue
        assert str(entry.get("outcome")) in vocabulary
        assert str(entry.get("id")) in rendered
    assert "provisional:" in rendered
    assert "refuted-to-provisional:" in rendered
    assert "harness:" in rendered


def test_v2_improvement_run_uses_one_primary_config_and_a_frozen_cutoff(tmp_path, monkeypatch, capsys):
    hypothesis = _hypothesis("H1")
    hypothesis["family"] = "features"
    hypothesis["needs"] = []
    hypothesis["experiment"] = {
        "action": "sweep",
        "tickers": ["BTC/USD"],
        "targets": ["fwd_direction"],
        "featureSets": ["all"],
        "modelSets": ["reference"],
        "horizons": [1],
        "maxConfigs": 1,
        "folds": 4,
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
    }
    hypothesis["criteria"] = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": False,
        "requires_promote": False,
    }
    root = _repo(tmp_path, hypothesis)
    _env(monkeypatch, root, tmp_path, fake_opencode=False)
    rc = tool.main(
        [
            "run",
            "--executor",
            "mechanical",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["outcome"] == "provisional"
    assert summary["holdout_start"]
    assert summary["lineage"] == "H1"
    entries = [
        json.loads(line)
        for line in (root / "experiments" / "ledger.jsonl").read_text().splitlines()
        if line.strip() and not json.loads(line).get("canary")
    ]
    kinds = [entry["kind"] for entry in entries]
    assert kinds[0] == "holdout_freeze"
    assert entries[0]["holdout_start"] == summary["holdout_start"]
    assert "results" in kinds and "verdict" in kinds and "review" in kinds
    prereg = json.loads((Path(summary["run_dir"]) / "preregister.json").read_text())
    assert prereg["criteria_version"] == 2
    assert prereg["holdout_start"] == summary["holdout_start"]
    assert prereg["protected"]["hash"]
    argv = json.loads((Path(summary["run_dir"]) / "lab-argv.json").read_text())
    assert "--holdout-start" in argv
    assert "--controls" in argv and argv[argv.index("--controls") + 1] == "both"
    sweep_argv = json.loads((Path(summary["run_dir"]) / "lab-argv-sweep.json").read_text())
    assert "--primary-config" in sweep_argv
    assert sweep_argv[sweep_argv.index("--primary-config") + 1] == "all+reference+h1"


def test_seed_is_passed_to_the_lab(tmp_path, monkeypatch, capsys):
    hypothesis = _hypothesis("H1")
    hypothesis["family"] = "features"
    hypothesis["needs"] = []
    hypothesis["experiment"]["primary"] = {
        "featureSet": "all",
        "modelSet": "reference",
        "horizon": 1,
    }
    hypothesis["criteria"] = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": False,
        "requires_promote": False,
    }
    root = _repo(tmp_path, hypothesis)
    _env(monkeypatch, root, tmp_path, fake_opencode=False)
    rc = tool.main(
        [
            "run",
            "--executor",
            "mechanical",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    argv = json.loads((Path(summary["run_dir"]) / "lab-argv.json").read_text())
    assert "--seed" in argv and argv[argv.index("--seed") + 1] == "0"
    assert summary["seed"] == 0


def test_harness_void_blocks_new_runs_until_cleared(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path, _hypothesis())
    _env(monkeypatch, root, tmp_path, fake_opencode=False)
    tool.append_entry(
        root / "experiments" / "ledger.jsonl",
        {"kind": "harness_void", "reason": "canary produced skill"},
    )
    rc = tool.main(
        [
            "run",
            "--executor",
            "mechanical",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "harness-void"


def test_preflight_warns_when_the_schedule_is_armed_without_forward_score(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    monkeypatch.setenv("RL_MODEL_IMPROVE_ENABLED", "true")
    result = tool.preflight(ctx.root, ctx.config, executor="mechanical")
    assert "schedule-forward-score" in result["warnings"]


def test_team_events_are_persisted_for_audit(tmp_path):
    stdout = "\n".join(
        [
            json.dumps({"type": "tool_use", "part": {"tool": "team_spawn"}}),
            json.dumps({"type": "tool_use", "part": {"tool": "team_results"}}),
            json.dumps({"type": "text", "part": {"text": "not an event"}}),
        ]
    )
    path = tmp_path / "team-events.jsonl"
    count = tool.extract_team_events(stdout, path)
    assert count == 2
    assert tool.team_spawn_count(stdout) == 1
    assert "team_spawn" in path.read_text()


def test_promote_retrains_the_champion_only_on_the_pre_cutoff_slice(tmp_path, monkeypatch):
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location(
        "model_lab_promote",
        Path(__file__).resolve().parents[1] / "tools" / "model_lab.py",
    )
    assert spec is not None and spec.loader is not None
    lab = importlib.util.module_from_spec(spec)
    sys.modules["model_lab_promote"] = lab
    spec.loader.exec_module(lab)

    start = datetime(2026, 1, 1, tzinfo=UTC)
    cutoff = (start + timedelta(hours=100)).strftime("%Y-%m-%dT%H:%M:%SZ")
    samples = [
        type("S", (), {
            "timestamp": (start + timedelta(hours=index)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "features": {},
            "targets": {"fwd_direction": 1.0},
        })()
        for index in range(130)
    ]
    captured = {}

    def fake_train(train, **kwargs):
        captured["train"] = list(train)
        return {"champion": True}

    candidate = types.SimpleNamespace(
        horizon=1,
        feature_version="rl-crypto/features/2",
        targets={
            "fwd_direction": types.SimpleNamespace(
                model=types.SimpleNamespace(to_dict=lambda: {"kind": "logistic"})
            )
        },
        feature_names=["ret_1"],
    )
    current = types.SimpleNamespace(
        horizon=1,
        feature_names=["ret_1"],
        targets={
            "fwd_direction": types.SimpleNamespace(
                model=types.SimpleNamespace(to_dict=lambda: {"kind": "logistic"})
            )
        },
    )
    monkeypatch.setattr(lab, "load_ticker_rows", lambda ticker: samples)
    monkeypatch.setattr(lab, "build_samples", lambda rows, **kwargs: tuple(samples))
    monkeypatch.setattr(
        lab,
        "load_bundle",
        lambda path: candidate if "candidate" in str(path) else current,
    )
    monkeypatch.setattr(lab, "default_bundle_path", lambda ticker, directory: Path("current.json"))
    monkeypatch.setattr(lab, "train_tier1", fake_train)
    monkeypatch.setattr(
        lab,
        "_bundle_view",
        lambda bundle, samples, targets, **kwargs: {
            "targets": {"fwd_direction": {"beats": True, "model": 0.5}},
            "economic": {"model": {"trades": 20, "sharpe": 0.5, "max_drawdown": -0.1}},
        },
    )
    monkeypatch.setattr(lab, "promotion_gate", lambda *args, **kwargs: [])
    monkeypatch.setattr(lab, "_save", lambda action, payload: Path("artifact.json"))
    monkeypatch.setattr(lab, "LAB_DIR", tmp_path / "lab")

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}", encoding="utf-8")
    args = types.SimpleNamespace(
        candidate=str(candidate_path),
        tickers=["BTC/USD"],
        ticker="BTC/USD",
        targets=["fwd_direction"],
        holdout_start=cutoff,
        champion_retrain=True,
        threshold=0.55,
        position="long-only",
        fee_bps=10.0,
        slippage_bps=5.0,
        apply=False,
        json=True,
    )
    assert lab.promote_action(args) == 0
    assert captured["train"]
    # sample_split finds the first post-cutoff sample; champion training is
    # strictly before it.
    assert all(sample.timestamp < cutoff for sample in captured["train"])


def test_second_run_of_a_lineage_cannot_touch_the_holdout(tmp_path, monkeypatch):
    hypothesis = _hypothesis("H1")
    hypothesis["family"] = "features"
    hypothesis["experiment"] = {
        "action": "sweep",
        "tickers": ["BTC/USD"],
        "targets": ["fwd_direction"],
        "featureSets": ["all"],
        "modelSets": ["reference"],
        "horizons": [1],
        "maxConfigs": 1,
        "folds": 4,
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
    }
    hypothesis["criteria"] = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": False,
        "requires_promote": True,
    }
    ctx = _ctx(tmp_path, hypothesis)
    ctx.prereg = {
        "id": "e2",
        "run_id": ctx.run_id,
        "hypothesis_id": "H1",
        "family": "features",
        "lineage": "H1",
        "trial_index": 2,
        "criteria_version": 2,
        "holdout_start": "2026-06-01T00:00:00Z",
        "criteria": hypothesis["criteria"],
        "experiment": hypothesis["experiment"],
    }
    tool.append_entry(
        ctx.ledger,
        {
            "kind": "holdout",
            "run_id": "previous",
            "hypothesis_id": "H1",
            "lineage": "H1",
            "holdout_touched": True,
        },
    )
    pooled = {
        "mean_improvement": 0.02,
        "ci_low": 0.01,
        "ci_high": 0.03,
        "n": 100,
        # trial_index 2 => alpha / (2*3) under the spending schedule.
        "alpha": 0.05 / 6.0,
        "fold_improvements": [0.01, 0.02, 0.01, 0.02],
    }
    monkeypatch.setattr(
        tool,
        "run_primary_evaluate",
        lambda *_args, **_kwargs: {
            "action": "evaluate",
            "primary": "all+reference+h1",
            "pooled": pooled,
            "pooled_economics": {
                "model": {"total_return": 0.10, "sharpe": 1.5, "trades": 200},
                "momentum": {"total_return": 0.02, "sharpe": 0.4, "trades": 200},
            },
            "controls": {
                "shuffle": {"pooled": {**pooled, "mean_improvement": -0.01, "ci_low": -0.02}},
                "shift": {"pooled": {**pooled, "mean_improvement": -0.01, "ci_low": -0.02}},
            },
            "results": [{"ticker": "BTC/USD", "paired": pooled}],
        },
    )
    monkeypatch.setattr(
        tool,
        "run_sweep",
        lambda *_args, **_kwargs: {
            "primary_candidates": ["candidate.json"],
            "top": [{"candidate": "candidate.json", "ticker": "BTC/USD"}],
        },
    )
    monkeypatch.setattr(
        tool,
        "run_promote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not promote")),
    )
    verdict = tool._evaluate_mechanical_v2(ctx, ctx.prereg)
    assert verdict["outcome"] == "blocked"
    assert any("holdout already touched" in reason for reason in verdict["reasons"])


def test_promotion_is_only_attempted_after_the_statistical_rule_passes(tmp_path, monkeypatch):
    hypothesis = _hypothesis("H1")
    hypothesis["family"] = "features"
    hypothesis["experiment"] = {
        "action": "sweep",
        "tickers": ["BTC/USD"],
        "targets": ["fwd_direction"],
        "featureSets": ["all"],
        "modelSets": ["reference"],
        "horizons": [1],
        "maxConfigs": 1,
        "folds": 4,
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
    }
    hypothesis["criteria"] = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": False,
        "requires_promote": True,
    }
    ctx = _ctx(tmp_path, hypothesis)
    ctx.prereg = {
        "id": "e2",
        "run_id": ctx.run_id,
        "hypothesis_id": "H1",
        "family": "features",
        "lineage": "H1",
        "trial_index": 1,
        "criteria_version": 2,
        "holdout_start": "2026-06-01T00:00:00Z",
        "criteria": hypothesis["criteria"],
        "experiment": hypothesis["experiment"],
    }
    pooled = {
        "mean_improvement": 0.02,
        "ci_low": 0.01,
        "ci_high": 0.03,
        "n": 100,
        "alpha": 0.025,
        "fold_improvements": [0.01, 0.02, 0.01, 0.02],
    }
    null = {**pooled, "mean_improvement": -0.01, "ci_low": -0.02, "fold_improvements": [-0.01] * 4}
    monkeypatch.setattr(
        tool,
        "run_primary_evaluate",
        lambda *_args, **_kwargs: {
            "action": "evaluate",
            "primary": "all+reference+h1",
            "pooled": pooled,
            "pooled_economics": {
                "model": {"total_return": 0.10, "sharpe": 1.5, "trades": 200},
                "momentum": {"total_return": 0.02, "sharpe": 0.4, "trades": 200},
            },
            "controls": {"shuffle": {"pooled": null}, "shift": {"pooled": null}},
            "results": [{"ticker": "BTC/USD", "paired": pooled}],
        },
    )
    monkeypatch.setattr(
        tool,
        "run_sweep",
        lambda *_args, **_kwargs: {
            "primary_candidates": ["candidate.json"],
            "top": [{"candidate": "candidate.json", "ticker": "BTC/USD"}],
        },
    )
    calls = []

    def fake_promote(*_args, **_kwargs):
        calls.append(True)
        return {"passed": True, "reasons": []}

    monkeypatch.setattr(tool, "run_promote", fake_promote)
    verdict = tool._evaluate_mechanical_v2(ctx, ctx.prereg)
    assert calls == [True]
    assert verdict["outcome"] == "provisional"


def test_token_and_cost_ceilings_block_from_recorded_step_usage(tmp_path):
    ctx = _ctx(tmp_path)
    logs = ctx.run_dir / "logs"
    logs.mkdir()
    (logs / "researcher.jsonl").write_text(
        json.dumps(
            {
                "type": "step_finish",
                "part": {"tokens": {"total": 500}, "cost": 0.25},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ctx.config["budgets"]["maxTokensPerRun"] = 100
    reason = tool.ceiling_reason(ctx)
    assert reason is not None and "token ceiling" in reason

    ctx.config["budgets"]["maxTokensPerRun"] = 10_000
    ctx.config["budgets"]["maxCostUsdPerRun"] = 0.1
    reason = tool.ceiling_reason(ctx)
    assert reason is not None and "cost ceiling" in reason
    assert ctx.usage["tokens"] == 500
    assert ctx.usage["costUsd"] == 0.25


def test_cli_blocks_a_no_new_information_rerun(tmp_path, monkeypatch, capsys):
    hypothesis = _hypothesis("H1")
    hypothesis["family"] = "features"
    hypothesis["needs"] = []
    hypothesis["experiment"] = {
        "action": "evaluate",
        "tickers": ["BTC/USD"],
        "targets": ["fwd_direction"],
        "featureSet": "all",
        "modelSet": "reference",
        "horizon": 1,
        "folds": 4,
        "maxConfigs": 1,
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
    }
    hypothesis["criteria"] = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": False,
        "requires_promote": False,
    }
    root = _repo(tmp_path, hypothesis)
    _env(monkeypatch, root, tmp_path, fake_opencode=False)
    tool.append_entry(
        root / "experiments" / "ledger.jsonl",
        {
            "kind": "preregister",
            "family": "features",
            "feature_version": tool.FEATURE_VERSION,
            "config_hash": tool.sha_of(hypothesis["experiment"]),
            "data": {"tickers": {"BTC/USD": {"rows": 150}}},
        },
    )
    rc = tool.main(
        [
            "run",
            "--executor",
            "mechanical",
            "--no-worktree",
            "--no-commit",
            "--run-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "blocked"
    assert "no-new-information" in payload["reason"]


def test_family_trial_cap_is_configurable_and_blocks_past_it(tmp_path):
    ctx = _ctx(tmp_path)
    assert tool.family_cap(ctx.config) == 10
    ok = {"family": "features", "trial_index": 10}
    over = {"family": "features", "trial_index": 11}
    assert tool.family_cap_reason(ctx.config, ok) is None
    reason = tool.family_cap_reason(ctx.config, over)
    assert reason and "trial cap (10)" in reason

    ctx.config["budgets"]["maxTrialsPerFamily"] = 4
    assert tool.family_cap_reason(ctx.config, {"family": "features", "trial_index": 4}) is None
    assert tool.family_cap_reason(ctx.config, {"family": "features", "trial_index": 5})


def test_v2_run_is_blocked_when_bootstrap_resolution_is_insufficient(tmp_path, monkeypatch):
    hypothesis = _hypothesis("H1")
    hypothesis["family"] = "features"
    hypothesis["criteria"] = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": False,
        "requires_promote": False,
    }
    ctx = _ctx(tmp_path, hypothesis)
    ctx.config["budgets"]["bootstrapSamples"] = 1000
    ctx.prereg = {
        "id": "e1",
        "kind": "preregister",
        "run_id": ctx.run_id,
        "hypothesis_id": "H1",
        "family": "features",
        "lineage": "H1",
        "trial_index": 1,
        "criteria_version": 2,
        "holdout_start": "2026-06-01T00:00:00Z",
        "criteria": hypothesis["criteria"],
        "experiment": hypothesis["experiment"],
    }
    monkeypatch.setattr(
        tool,
        "run_primary_evaluate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not run the lab before resolution is checked")
        ),
    )
    tool.append_entry(ctx.ledger, ctx.prereg)
    verdict = tool._evaluate_mechanical_v2(ctx, ctx.prereg)
    assert verdict["outcome"] == "blocked"
    assert any("insufficient-resolution" in reason for reason in verdict["reasons"])
    blocked = [entry for entry in tool.read_ledger(ctx.ledger) if entry["kind"] == "blocked"]
    assert blocked and "insufficient-resolution" in blocked[-1]["reason"]
    assert blocked[-1]["infra"] is True
    assert tool.trial_counts(tool.read_ledger(ctx.ledger))["family"] == {}


def test_lab_timeout_is_an_infra_block_and_does_not_consume_a_trial(tmp_path, monkeypatch):
    """A timeout produces no lab result, so it must not append ``results``."""
    hypothesis = _hypothesis("H1")
    hypothesis["family"] = "features"
    hypothesis["criteria"] = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": False,
        "requires_promote": False,
    }
    ctx = _ctx(tmp_path, hypothesis)
    ctx.config["budgets"]["labTimeoutSeconds"] = 0.05
    slow_lab = tmp_path / "slow_lab.py"
    slow_lab.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    monkeypatch.setenv("RL_MODEL_IMPROVE_LAB", f"{sys.executable} {slow_lab}")
    ctx.prereg = {
        "id": "e1",
        "kind": "preregister",
        "run_id": ctx.run_id,
        "hypothesis_id": "H1",
        "family": "features",
        "lineage": "H1",
        "trial_index": 1,
        "criteria_version": 2,
        "holdout_start": "2026-06-01T00:00:00Z",
        "criteria": hypothesis["criteria"],
        "experiment": hypothesis["experiment"],
    }
    tool.append_entry(ctx.ledger, ctx.prereg)

    verdict = tool._evaluate_mechanical_v2(ctx, ctx.prereg)

    assert verdict["outcome"] == "blocked"
    assert verdict["infra"] is True
    entries = tool.read_ledger(ctx.ledger)
    assert not [entry for entry in entries if entry["kind"] == "results"]
    blocked = [entry for entry in entries if entry["kind"] == "blocked"]
    assert blocked[-1]["infra"] is True
    assert blocked[-1]["infra_kind"] == "timeout"
    assert tool.trial_counts(entries)["family"] == {}


FAKE_NULL_LAB = '''\
"""Minimal model_lab surface for the null-calibration test (no numpy)."""


class Config:
    def __init__(self, name, feature_names, model_names, horizon):
        self.name = name
        self.feature_names = feature_names
        self.model_names = model_names
        self.horizon = horizon


def resolve_feature_names(spec):
    return ("ret_1",)


def resolve_model_set(spec):
    return {"fwd_direction": "logistic"}


def evaluate_control_ticker(
    ticker, config, target, mode, *, folds, holdout_start, seed, shift_bars, **kwargs
):
    return {
        "ticker": ticker,
        "paired_rows": [{"timestamp": "t", "improvement": 0.01} for _ in range(100)],
        "fold_improvements": [0.01] * folds,
        "economics": (
            {
                "model": {"total_return": 0.1, "sharpe": 1.0, "trades": 100},
                "momentum": {"sharpe": 0.1},
            }
            if mode is None
            else None
        ),
    }


def bootstrap_pooled(rows_by_ticker, *, alpha, seed, block_bars, bootstrap_samples):
    good = seed % 2 == 0
    return {
        "n": 100,
        "blocks": 30,
        "alpha": alpha,
        "mean_improvement": 0.01 if good else -0.01,
        "ci_low": 0.001 if good else -0.02,
        "ci_high": 0.02,
        "resolution_ok": True,
    }


def pool_fold_improvements(folds_by_ticker, folds):
    return [0.01] * folds
'''


FAKE_POWER_LAB = '''\
"""Minimal model_lab surface for the power-calibration test (no numpy)."""

INTERVAL_SECONDS = 3600
PLANTED_FEATURE = "planted_signal"
MODEL_SETS = {
    "reference": {
        "fwd_volume_ratio": "ridge",
        "fwd_volatility": "log_ridge",
    }
}


class Config:
    def __init__(self, name, feature_names, model_names, horizon):
        self.name = name
        self.feature_names = feature_names
        self.model_names = model_names
        self.horizon = horizon


class Sample:
    def __init__(self, index):
        self.timestamp = "2026-01-01T00:00:00Z"
        self.features = {"ret_1": 0.0}
        self.targets = {
            "fwd_volume_ratio": 1.0 + (index % 10) * 0.01,
            "fwd_volatility": 0.02 + (index % 10) * 0.001,
        }


def resolve_feature_names(spec):
    return ("ret_1",)


def resolve_model_set(spec):
    return dict(MODEL_SETS["reference"])


def load_ticker_rows(ticker):
    return list(range(120))


def filter_rows_before(rows, cutoff):
    return tuple(rows), None


def build_samples(rows, horizon, interval_seconds):
    return [Sample(i) for i in range(100)]


def evaluate_control_ticker(
    ticker, config, target, mode, *, folds, holdout_start, seed, shift_bars,
    leak_feature=None, plant_strength=0.0, plant_space="raw", plant_feature=None,
    **kwargs,
):
    control = mode is not None
    value = -0.001 if control or plant_strength == 0.0 else plant_strength
    return {
        "ticker": ticker,
        "paired_rows": [{"timestamp": "t", "improvement": value} for _ in range(100)],
        "fold_improvements": [value] * folds,
        "planted": {"strength": plant_strength, "effect": plant_strength ** 2},
        "economics": (
            {
                "model": {"total_return": 0.1, "sharpe": 1.0, "trades": 100},
                "momentum": {"sharpe": 0.1},
            }
            if mode is None
            else None
        ),
    }


def bootstrap_pooled(rows_by_ticker, *, alpha, seed, block_bars, bootstrap_samples):
    values = [row["improvement"] for rows in rows_by_ticker.values() for row in rows]
    mean = sum(values) / len(values) if values else None
    return {
        "n": len(values),
        "blocks": 30,
        "alpha": alpha,
        "mean_improvement": mean,
        "ci_low": None if mean is None else mean - 0.001,
        "ci_high": None if mean is None else mean + 0.001,
        "resolution_ok": True,
    }


def pool_fold_improvements(folds_by_ticker, folds):
    return [0.01] * folds
'''


def _null_hypothesis() -> dict:
    hypothesis = _hypothesis("H0-canary")
    hypothesis["family"] = "canary"
    hypothesis["canary"] = True
    hypothesis["canary_kind"] = "null"
    hypothesis["criteria"] = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": False,
        "requires_promote": False,
    }
    return hypothesis


def test_null_calibration_measures_the_empirical_rate_without_touching_the_ledger(
    tmp_path, monkeypatch
):
    root = _repo(tmp_path, _null_hypothesis())
    (root / "tools" / "model_lab.py").write_text(FAKE_NULL_LAB, encoding="utf-8")
    _env(monkeypatch, root, tmp_path, fake_opencode=False, canaries=False)
    ledger_before = (root / "experiments" / "ledger.jsonl").read_text()
    payload = tool.calibrate_null(
        root,
        config=tool.load_config(root),
        queue=tool.load_queue(root / "experiments" / "hypotheses.json"),
        entries=tool.read_ledger(root / "experiments" / "ledger.jsonl"),
        seeds=10,
        trials=(1, 5, 10),
        alpha_family=0.05,
    )
    assert payload["seeds"] == 10
    assert payload["trials"]["1"]["successes"] == 5  # even seeds pass
    assert payload["trials"]["1"]["nominal_alpha"] == pytest.approx(0.025)
    assert payload["trials"]["1"]["rate"] == pytest.approx(0.5)
    assert payload["calibration_failure"] is True
    assert (root / "experiments" / "ledger.jsonl").read_text() == ledger_before


def test_realistic_leak_canaries_all_come_out_invalid():
    alpha = 0.025
    base, scenarios = tool.leak_contract_payloads("fwd_volume_ratio", alpha)
    prereg = tool._leak_prereg("fwd_volume_ratio", 0.05)
    clean = tool.evaluate_criteria(prereg, base)
    assert clean["outcome"] == "provisional"
    assert [name for name, _, _ in scenarios] == [
        "scaler_fit_on_full_data",
        "label_overlap_across_split",
        "selection_on_holdout",
        "lookahead_feature",
    ]
    for name, payload, layer in scenarios:
        verdict = tool.evaluate_criteria(prereg, payload)
        assert verdict["outcome"] == "invalid", name
        assert layer.split()[0] in " ".join(verdict["reasons"])


def test_power_calibration_finds_the_minimum_planted_effect(tmp_path, monkeypatch):
    root = _repo(tmp_path, _null_hypothesis())
    (root / "tools" / "model_lab.py").write_text(FAKE_POWER_LAB, encoding="utf-8")
    _env(monkeypatch, root, tmp_path, fake_opencode=False, canaries=False)
    ledger_before = (root / "experiments" / "ledger.jsonl").read_text()
    payload = tool.calibrate_power(
        root,
        config=tool.load_config(root),
        queue=tool.load_queue(root / "experiments" / "hypotheses.json"),
        entries=tool.read_ledger(root / "experiments" / "ledger.jsonl"),
        targets=["fwd_volume_ratio"],
        reps=4,
        effect_fractions=[0.02, 0.0, 0.01],  # deliberately unsorted
        trials=(1, 5),
        controls=(),
        power=0.8,
    )
    block = payload["results"]["fwd_volume_ratio"]
    assert block["space"] == "raw"
    # The fake passes once the planted strength clears the 0.001 CI width, so
    # the first grid point that does is the 0.01-of-variance effect.
    assert block["mde"]["1"]["effect"] == pytest.approx(0.01 * block["label_variance"])
    assert block["mde"]["1"]["detected"] is True
    assert block["mde"]["1"]["pass_rate"] == pytest.approx(1.0)
    assert block["observed_mean_improvement"] == pytest.approx(-0.001)
    assert (root / "experiments" / "ledger.jsonl").read_text() == ledger_before


def test_calibrate_power_cli_writes_a_table(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path, _null_hypothesis())
    (root / "tools" / "model_lab.py").write_text(FAKE_POWER_LAB, encoding="utf-8")
    _env(monkeypatch, root, tmp_path, fake_opencode=False, canaries=False)
    run_dir = tmp_path / "power"
    rc = tool.main(
        [
            "calibrate-power",
            "--targets",
            "fwd_volume_ratio",
            "--reps",
            "3",
            "--effect-fractions",
            "0.0,0.01",
            "--trials",
            "1,5",
            "--controls",
            "none",
            "--run-dir",
            str(run_dir),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "calibrate-power"
    artifact = Path(payload["run_dir"])
    assert (artifact / "calibrate-power.json").exists()
    markdown = (artifact / "calibrate-power.md").read_text()
    assert "Power calibration" in markdown
    assert "fwd_volume_ratio" in markdown


def test_calibrate_null_cli_writes_under_the_run_dir(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path, _null_hypothesis())
    (root / "tools" / "model_lab.py").write_text(FAKE_NULL_LAB, encoding="utf-8")
    _env(monkeypatch, root, tmp_path, fake_opencode=False, canaries=False)
    run_dir = tmp_path / "calibration"
    rc = tool.main(
        [
            "calibrate-null",
            "--seeds",
            "6",
            "--jobs",
            "1",
            "--run-dir",
            str(run_dir),
            "--json",
        ]
    )
    assert rc == 1  # the fake null is deliberately over-alpha
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "calibrate-null"
    artifact = Path(payload["run_dir"])
    assert artifact.parent == run_dir
    assert (artifact / "calibrate-null.json").exists()
    assert (artifact / "calibrate-null.md").exists()
    assert (root / "experiments" / "ledger.jsonl").read_text() == ""


def test_positive_canary_leak_is_invalid_and_does_not_void_the_harness(tmp_path, monkeypatch):
    hypothesis = _hypothesis("H0-canary-leak")
    hypothesis["family"] = "canary"
    hypothesis["canary"] = True
    hypothesis["canary_kind"] = "positive"
    hypothesis["experiment"] = {
        "action": "evaluate",
        "tickers": ["BTC/USD"],
        "targets": ["fwd_direction"],
        "featureSet": "all",
        "modelSet": "reference",
        "horizon": 1,
        "folds": 4,
        "maxConfigs": 1,
        "extraLabArgs": ["--canary-leak", "leak_next_bar_return"],
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
    }
    hypothesis["criteria"] = {
        "kind": "improvement",
        "criteria_version": 2,
        "target": "fwd_direction",
        "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        "alpha": 0.05,
        "fold_rule": "all_positive",
        "require_controls": True,
        "require_multiple_tickers": False,
        "requires_promote": False,
    }
    ctx = _ctx(tmp_path, hypothesis)
    ctx.holdout_start = "2026-06-01T00:00:00Z"
    ctx.prereg = {
        "id": "e1",
        "kind": "preregister",
        "run_id": ctx.run_id,
        "hypothesis_id": hypothesis["id"],
        "family": "canary",
        "lineage": hypothesis["id"],
        "trial_index": 1,
        "criteria_version": 2,
        "holdout_start": ctx.holdout_start,
        "criteria": hypothesis["criteria"],
        "experiment": hypothesis["experiment"],
    }
    pooled = {
        "mean_improvement": 0.02,
        "ci_low": 0.01,
        "ci_high": 0.03,
        "n": 100,
        "alpha": 0.025,
        "fold_improvements": [0.01, 0.02, 0.01, 0.02],
    }
    monkeypatch.setattr(
        tool,
        "run_primary_evaluate",
        lambda *_args, **_kwargs: {
            "action": "evaluate",
            "primary": "all+reference+h1",
            "except_leak": True,
            "pooled": pooled,
            "pooled_economics": {
                "model": {"total_return": 0.10, "sharpe": 1.5, "trades": 200},
                "momentum": {"total_return": 0.02, "sharpe": 0.4, "trades": 200},
            },
            "leak_detected": True,
            "leak_features": [
                {
                    "feature": "leak_next_bar_return",
                    "target": "fwd_return",
                    "correlation": 1.0,
                }
            ],
            "controls": {
                "shuffle": {"pooled": pooled},
                "shift": {"pooled": pooled},
            },
            "results": [{"ticker": "BTC/USD", "paired": pooled}],
        },
    )
    verdict = tool._evaluate_mechanical_v2(ctx, ctx.prereg)
    assert verdict["outcome"] == "invalid"
    assert any("leak" in reason for reason in verdict["reasons"])
    from rl_crypto.experiments import (
        CANARY_FILENAME,
        harness_void,
        read_canary,
        read_ledger,
    )

    entries = read_ledger(ctx.ledger)
    assert harness_void(entries) is None
    canaries = read_canary(ctx.repo / "experiments" / CANARY_FILENAME)
    recorded = canaries[-1]
    assert recorded["canary"] is True
    assert recorded["canary_kind"] == "positive"


def test_auditor_fixtures_have_the_required_shape():
    fixtures = Path(__file__).resolve().parents[1] / "experiments" / "auditor-fixtures"
    cases = tool._fixture_cases(fixtures)
    leaky = [case for case in cases if case["class"] == "leaky"]
    clean = [case for case in cases if case["class"] == "clean"]
    assert len(leaky) >= 6
    assert len(clean) >= 4
    kinds = {case["kind"] for case in leaky}
    assert "look-ahead feature" in kinds
    assert "fit-on-full-data scaler" in kinds
    assert "label overlap across split" in kinds
    assert "selecting on the holdout" in kinds
    assert "cost omitted on close" in kinds
    assert "protected-file edit" in kinds
    for case in cases:
        assert case["diff"].is_file()


def test_auditor_check_reports_recall_and_false_blocks(tmp_path, monkeypatch):
    import argparse

    fixtures = Path(__file__).resolve().parents[1] / "experiments" / "auditor-fixtures"
    monkeypatch.setattr(tool, "_prepare_fixture_repo", lambda *args, **kwargs: None)

    def fake_audit(config, repo, run_dir, *, timeout):
        blocked = "leak" in repo.name
        return {
            "blocked": blocked,
            "returncode": 0,
            "model": "test-model",
            "findings": [
                {
                    "question": "q",
                    "verdict": "fail" if blocked else "pass",
                    "evidence": "fixture:1",
                }
            ],
            "audit": {"blocking": blocked, "findings": [], "notes": None},
        }

    monkeypatch.setattr(tool, "_audit_fixture", fake_audit)
    args = argparse.Namespace(
        fixtures=str(fixtures),
        run_dir=str(tmp_path / "auditor"),
        timeout=10,
        json=True,
        model=None,
    )
    assert tool.cmd_auditor_check(args) == 0
    payload = json.loads((tmp_path / "auditor" / "auditor-check.json").read_text())
    assert payload["counts"]["leaky"] == 6
    assert payload["counts"]["clean"] == 4
    assert payload["recall"] == 1.0
    assert payload["false_block_rate"] == 0.0


def test_design_scaled_mde_pools_tickers_and_history_with_printed_assumptions():
    pilot = tool.design_scaled_mde(
        0.01,
        pilot_rows=3000,
        block_bars=24,
        tickers=["BTC/USD", "ETH/USD", "SOL/USD"],
        design={
            "kind": "history",
            "tickers": 3,
            "barsPerTicker": 8760,
            "tickerCorrelation": 0.5,
        },
        bars_per_ticker=1000,
    )
    assert pilot["kind"] == "history"
    assert pilot["design_rows"] == pytest.approx(8760 * 1.5)
    assert pilot["design_scaled_mde"] < 0.01
    assert any("cross-ticker correlation" in line for line in pilot["assumptions"])


def test_design_scaling_never_credits_correlated_tickers_as_independent():
    uncorrelated = tool.design_scaled_mde(
        0.01,
        pilot_rows=3000,
        block_bars=24,
        tickers=["BTC/USD"],
        design={"kind": "panel", "tickers": 3, "barsPerTicker": 1000, "tickerCorrelation": 0.0},
        bars_per_ticker=1000,
    )
    correlated = tool.design_scaled_mde(
        0.01,
        pilot_rows=3000,
        block_bars=24,
        tickers=["BTC/USD"],
        design={"kind": "panel", "tickers": 3, "barsPerTicker": 1000, "tickerCorrelation": 0.5},
        bars_per_ticker=1000,
    )
    assert correlated["design_scaled_mde"] > uncorrelated["design_scaled_mde"]


def test_deferred_underpowered_hypotheses_are_not_selected():
    from rl_crypto.experiments import Hypothesis, select_hypothesis

    deferred = Hypothesis.from_dict(
        {
            **_hypothesis("H3"),
            "status": "deferred-underpowered",
            "status_reason": "pilot effect far below the economics MME",
            "reevaluate_when": "a longer horizon lowers the MME",
        }
    )
    open_hypothesis = Hypothesis.from_dict(_hypothesis("H1", needs=[]))
    assert select_hypothesis([deferred, open_hypothesis], []) is open_hypothesis
    assert select_hypothesis([deferred], []) is None


def test_canary_refresh_appends_only_to_the_canary_record(tmp_path, monkeypatch):
    canary_hypotheses = []
    for kind in ("null", "positive"):
        payload = _hypothesis(f"H0-canary-{kind}")
        payload["family"] = "canary"
        payload["canary"] = True
        payload["canary_kind"] = kind
        payload["experiment"] = {
            "action": "evaluate",
            "tickers": ["BTC/USD"],
            "targets": ["fwd_direction"],
            "featureSet": "all",
            "modelSet": "reference",
            "horizon": 1,
            "folds": 4,
            "maxConfigs": 1,
            "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
        }
        payload["criteria"] = {
            "kind": "improvement",
            "criteria_version": 3,
            "target": "fwd_direction",
            "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
            "alpha": 0.05,
            "fold_rule": "all_positive",
            "require_controls": True,
            "require_multiple_tickers": False,
            "requires_promote": False,
        }
        canary_hypotheses.append(payload)
    root = _repo(tmp_path, canary_hypotheses[0])
    (root / "experiments" / "hypotheses.json").write_text(
        json.dumps({"hypotheses": canary_hypotheses}), encoding="utf-8"
    )
    _env(monkeypatch, root, tmp_path, fake_opencode=False, canaries=False)
    ledger_before = (root / "experiments" / "ledger.jsonl").read_text()

    negative = {
        "mean_improvement": -0.01,
        "ci_low": -0.02,
        "ci_high": 0.0,
        "n": 100,
        "alpha": 0.025,
        "fold_improvements": [-0.01] * 4,
        "resolution_ok": True,
    }
    positive_payload = {
        "action": "evaluate",
        "primary": "all+reference+h1",
        "results": [{"ticker": "BTC/USD"}],
        "pooled": negative,
        "controls": {"shuffle": {"pooled": negative}},
        "leak_detected": True,
        "leak_features": [{"feature": "leak_next_bar_return", "target": "fwd_return", "correlation": 1.0}],
        "comparison": "baseline",
    }

    def fake_run(ctx, prereg, **kwargs):
        if ctx.hypothesis.canary_kind == "null":
            return {
                **positive_payload,
                "leak_detected": False,
                "leak_features": [],
            }
        return positive_payload

    monkeypatch.setattr(tool, "run_primary_evaluate", fake_run)
    from rl_crypto.experiments import CANARY_FILENAME, read_canary

    result = tool.canary_refresh(
        root,
        config=tool.load_config(root),
        queue=tool.load_queue(root / "experiments" / "hypotheses.json"),
        run_dir=tmp_path / "canaries",
    )
    assert result["passed"] is True
    assert [item["outcome"] for item in result["outcomes"]] == ["negative", "invalid"]
    assert (root / "experiments" / "ledger.jsonl").read_text() == ledger_before
    canaries = read_canary(root / "experiments" / CANARY_FILENAME)
    assert [entry["canary_kind"] for entry in canaries] == ["null", "positive"]
    assert all(entry["protected_hash"] for entry in canaries)
    assert result["scoreboard_line"].startswith("- canary:")
