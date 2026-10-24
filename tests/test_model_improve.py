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
if action == "evaluate":
    payload = {"action": "evaluate", "results": [result], "summary": {"runs": 1}}
elif action in {"sweep", "promote"}:
    candidate = os.path.join(os.environ.get("RUN_DIR", "."), "candidate.json")
    payload = {
        "action": action,
        "results": [result],
        "top": [result],
        "candidates": [candidate],
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
                "budgets": {"maxConfigs": 1, "folds": 4, "labTimeoutSeconds": 30},
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


def _env(monkeypatch: pytest.MonkeyPatch, root: Path, tmp_path: Path, *, fake_opencode: bool):
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
    _env(monkeypatch, root, tmp_path, fake_opencode=False)
    rc = tool.main(["plan", "--executor", "mechanical", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hypothesis"]["id"] == "H0"
    assert payload["preregister"]["criteria"]["kind"] == "diagnostic"
    assert (root / "experiments" / "ledger.jsonl").read_text() == ""


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
        if line.strip()
    ]
    assert [entry["kind"] for entry in entries] == [
        "preregister",
        "results",
        "verdict",
        "review",
    ]
    assert entries[0]["status"] == "preregistered"
    assert entries[2]["outcome"] == "diagnostic"
    assert entries[2]["metrics"]["targets"] == ["fwd_direction", "fwd_volume_ratio"]
    assert "BTC/USD" in entries[2]["metrics"]["per_ticker"]
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
        if line.strip()
    ]
    assert [entry["kind"] for entry in entries] == [
        "preregister",
        "blocked",
        "review",
    ]
    assert "researcher" in entries[1]["reason"]


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
    assert config["models"] == {}
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
