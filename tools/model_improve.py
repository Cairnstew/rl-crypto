#!/usr/bin/env python3
"""Autonomous model-improvement orchestrator for rl-crypto.

This is the unattended counterpart to ``/model-improve``. It runs the same
phases — preflight, pre-register, diagnose, implement, audit, evaluate, record —
but every decision that can be mechanical is made by code, and every LLM role
is a separate, bounded process with a file-based contract. Nothing is forwarded
from one agent's conversation to another: the handoff is the artifact on disk.

Executors
---------
``team``       one headless lead process that uses OpenCode Ensemble
               (``team_create`` / ``team_spawn``) for the researcher,
               implementer and auditor roles.
``passes``     the same roles run as sequential ``opencode run`` processes with
               fresh context each. This is the CI-safe fallback and the mode
               that needs no plugin.
``mechanical`` no LLM at all. Only hypotheses whose ``needs`` are empty run.
``auto``       ``team`` when the ensemble plugin exists and ``CI`` is unset,
               otherwise ``passes``.

The verdict is never written by an LLM. ``evaluate_criteria`` in
``rl_crypto.experiments`` computes it from the pre-registered criteria and the
``rl-model-lab`` JSON, and the orchestrator records it in the append-only
ledger. An LLM may only add prose to ``evaluation.md``.

Exit codes: 0 success (including a negative result), 2 blocked/preflight
failure, 1 unexpected error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_crypto.experiments import (
    Hypothesis,
    append_entry,
    build_preregistration,
    evaluate_criteria,
    git_state,
    holdout_touches,
    iso,
    load_json,
    load_queue,
    read_ledger,
    render_hypotheses,
    render_scoreboard,
    select_hypothesis,
    trial_counts,
    write_if_changed,
)

SCHEMA = "rl-crypto/model-improvement/1"
ROLE_AGENTS = {
    "lead": "model-improve-lead",
    "researcher": "model-improve-researcher",
    "implementer": "model-improve-implementer",
    "auditor": "model-improve-auditor",
    "evaluator": "model-improve-evaluator",
    "reviewer": "model-improve-reviewer",
}
REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "lead": ("lead.json",),
    "researcher": ("researcher.json", "diagnosis.md"),
    "implementer": ("implementation.json", "implementation.md"),
    "auditor": ("audit.json",),
    "evaluator": ("evaluation.md",),
    "reviewer": ("review.json", "review.md"),
}

DEFAULT_CONFIG: dict[str, Any] = {
    "schema": SCHEMA,
    "executor": "auto",
    # One knob for every role and teammate. Override per role only when a
    # specific role genuinely needs a different model.
    "model": "opencode/deepseek-v4.1-flash-free",
    "models": {},
    "budgets": {
        "maxConfigs": 8,
        "folds": 4,
        "roleTimeoutSeconds": 1800,
        "maxRolesPerRun": 6,
        "maxCostUsdPerRun": 5.0,
    },
    "review": {"enabled": True, "require": False},
    "team": {"maxLeadTurns": 3},
    "promotion": {"apply": False, "requiresPromoteDryRun": True},
    "scope": {
        "tickers": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "targets": [
            "fwd_direction",
            "fwd_return",
            "fwd_volatility",
            "fwd_volume_ratio",
        ],
        "threshold": 0.55,
        "position": "long-only",
        "feeBps": 10.0,
        "slippageBps": 5.0,
    },
    "seed": 0,
    "staleHours": 6,
}


# --------------------------------------------------------------------------- helpers


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def repo_root() -> Path:
    return Path(
        os.environ.get("RL_CRYPTO_ROOT") or Path(__file__).resolve().parents[1]
    ).resolve()


def load_config(root: Path) -> dict[str, Any]:
    path = root / "config" / "model-improvement.json"
    payload = load_json(path, {})
    config = deep_merge(DEFAULT_CONFIG, payload or {})
    if config.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported schema {config.get('schema')!r}")
    return config


def resolve_model(config: Mapping[str, Any], role: str | None = None) -> str | None:
    """Resolve the model for a role.

    Precedence: env override > per-role override > the single ``model`` knob.
    The repo default is ``opencode/deepseek-v4.1-flash-free`` so changing the
    model in one place changes every role and every Ensemble teammate.
    """
    override = os.environ.get("RL_MODEL_IMPROVE_MODEL")
    if override:
        return override
    if config.get("_model_override"):
        model = config.get("model")
        return str(model) if model else None
    models = config.get("models") or {}
    if role and models.get(role):
        return str(models[role])
    model = config.get("model") or DEFAULT_CONFIG.get("model")
    return str(model) if model else None


def apply_model_override(config: dict[str, Any], requested: str | None) -> dict[str, Any]:
    """A one-run --model override wins over per-role exceptions."""
    if requested:
        config["model"] = requested
        config["_model_override"] = True
    return config


def default_run_root() -> Path:
    override = os.environ.get("RL_MODEL_IMPROVE_RUN_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".local" / "share" / "rl-crypto" / "model-improve" / "runs"


def opencode_command() -> str:
    """Binary used for role runs; override to pin a CI install."""
    return os.environ.get("RL_MODEL_IMPROVE_OPENCODE") or "opencode"


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    stdin_null: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=cwd,
        env={**os.environ, **dict(env or {})} if env else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        stdin=subprocess.DEVNULL if stdin_null else None,
    )


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=cwd)


def agent_modes() -> dict[str, str]:
    """Parse ``opencode agent list`` into {name: primary|subagent|all}.

    ``opencode run --agent`` refuses a subagent and silently falls back to the
    default agent, which would bypass the role's permissions. This is how we
    detect that instead of discovering it from a misbehaving run.
    """
    opencode = shutil.which(opencode_command())
    if opencode is None:
        return {}
    proc = _run([opencode, "agent", "list"], timeout=60, stdin_null=True)
    modes: dict[str, str] = {}
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        match = re.match(
            r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s+\((primary|subagent|all)\)\s*$",
            line,
        )
        if match:
            modes[match.group(1)] = match.group(2)
    return modes


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    print("  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _staleness(last: str | None) -> float | None:
    if not last:
        return None
    try:
        stamp = datetime.fromisoformat(last.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return round((datetime.now(UTC) - stamp).total_seconds() / 3600.0, 2)


# --------------------------------------------------------------------------- preflight


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    blocking: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "blocking": self.blocking,
        }


def preflight(
    root: Path,
    config: Mapping[str, Any],
    *,
    executor: str = "auto",
    worktree: bool = False,
) -> dict[str, Any]:
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str, *, blocking: bool = True) -> None:
        checks.append(Check(name=name, ok=ok, detail=detail, blocking=blocking))

    config_path = root / "config" / "model-improvement.json"
    add("config", config_path.exists(), str(config_path))
    add(
        "queue",
        (root / "experiments" / "hypotheses.json").exists(),
        "experiments/hypotheses.json",
    )
    add(
        "ledger",
        (root / "experiments" / "ledger.jsonl").exists(),
        "append-only ledger",
    )
    add(
        "scoreboard",
        (root / "experiments" / "SCOREBOARD.md").exists(),
        "experiments/SCOREBOARD.md",
    )
    # The lab needs numpy/pandas; on NixOS they fail outside `nix develop`
    # unless LD_LIBRARY_PATH is set. Catch that here, not after a long sweep.
    # An explicit external lab command owns its own environment.
    external_lab = os.environ.get("RL_MODEL_IMPROVE_LAB")
    if external_lab:
        add("lab-imports", True, f"external lab: {external_lab}", blocking=False)
    else:
        probe = _run(
            [sys.executable, "-c", "import numpy, pandas"],
            timeout=60,
            stdin_null=True,
        )
        probe_detail = "numpy + pandas import ok"
        if probe.returncode != 0:
            tail = (probe.stderr or "").strip().splitlines()[-1:] or ["import failed"]
            probe_detail = f"{tail[0]} (run inside `nix develop`)"
        add("lab-imports", probe.returncode == 0, probe_detail)
    missing_agents = [
        name
        for name in ROLE_AGENTS.values()
        if not (root / ".opencode" / "agents" / f"{name}.md").exists()
    ]
    add(
        "agents",
        not missing_agents,
        "all role agents present"
        if not missing_agents
        else f"missing: {', '.join(missing_agents)}",
        blocking=executor != "mechanical",
    )
    opencode = shutil.which(opencode_command())
    add(
        "opencode",
        opencode is not None,
        opencode or f"{opencode_command()} not on PATH",
        blocking=executor != "mechanical",
    )
    if executor != "mechanical" and opencode is not None:
        modes = agent_modes()
        subagents = [name for name in ROLE_AGENTS.values() if modes.get(name) == "subagent"]
        unknown = [name for name in ROLE_AGENTS.values() if modes and name not in modes]
        if not modes:
            add(
                "agent-modes",
                True,
                "could not read `opencode agent list`; mode not checked",
                blocking=False,
            )
        else:
            detail = "all role agents are runnable by `opencode run`"
            if subagents:
                detail = f"subagent mode cannot be run headlessly: {', '.join(subagents)}"
            elif unknown:
                detail = f"not loaded: {', '.join(unknown)}"
            add("agent-modes", not subagents and not unknown, detail)

    scope = config.get("scope") or {}
    stale_hours = float(config.get("staleHours") or 6)
    data_rows: list[dict[str, Any]] = []
    for ticker in scope.get("tickers") or []:
        slug = str(ticker).replace("/", "-").replace(" ", "-").lower()
        path = root / "data" / slug / "candles.jsonl"
        if not path.exists():
            data_rows.append({"ticker": ticker, "present": False})
            add(f"data:{ticker}", False, "missing candles.jsonl")
            continue
        last: str | None = None
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:
                    last = json.loads(line).get("timestamp") or last
                except json.JSONDecodeError:
                    continue
        stale = _staleness(last)
        data_rows.append(
            {
                "ticker": ticker,
                "present": True,
                "rows": count,
                "last": last,
                "stale_hours": stale,
            }
        )
        add(f"data:{ticker}", count >= 120, f"{count} bars, last {last}")
        add(
            f"fresh:{ticker}",
            stale is not None and stale <= stale_hours,
            f"{stale}h old (limit {stale_hours}h) — offline experiments allowed",
            blocking=False,
        )
        add(
            f"bundle:{ticker}",
            (root / "models" / f"{slug}.json").exists(),
            f"models/{slug}.json",
            blocking=False,
        )

    # A worktree is created from HEAD. Uncommitted role agents, policy or queue
    # would therefore be invisible to the run (and the role agent would not
    # load), so surface that instead of failing deep inside a role.
    dirty = git_state(root).get("dirty_paths") or []
    inputs = (
        ".opencode/",
        "config/",
        "experiments/",
        "tools/model_improve.py",
        "src/rl_crypto/experiments.py",
    )
    uncommitted = [path for path in dirty if any(token in path for token in inputs)]
    add(
        "committed-inputs",
        not uncommitted,
        "run inputs are committed"
        if not uncommitted
        else f"uncommitted: {', '.join(uncommitted[:6])} — commit before a worktree run",
        # A worktree is created from HEAD, so uncommitted run inputs are
        # invisible to the run and the role agent would not load.
        blocking=worktree,
    )

    ensemble = (
        Path.home() / ".config" / "opencode" / "plugins" / "opencode-ensemble.js"
    )
    add(
        "ensemble",
        ensemble.exists(),
        str(ensemble) if ensemble.exists() else "not installed (passes fallback)",
        blocking=False,
    )
    add(
        "forward-score",
        (root / "tools" / "forward_score.py").exists(),
        "tools/forward_score.py (forward confirmation)",
        blocking=False,
    )

    blocking = [check for check in checks if check.blocking and not check.ok]
    warnings = [check for check in checks if not check.blocking and not check.ok]
    return {
        "root": str(root),
        "executor": executor,
        "data": data_rows,
        "checks": [check.as_dict() for check in checks],
        "blocking": [check.name for check in blocking],
        "warnings": [check.name for check in warnings],
        "ok": not blocking,
        "at": iso(),
    }


# --------------------------------------------------------------------------- run context


@dataclass
class Context:
    root: Path
    config: dict[str, Any]
    executor: str
    trigger: str
    run_id: str
    run_dir: Path
    hypothesis: Hypothesis | None = None
    worktree: Path | None = None
    branch: str | None = None
    prereg: dict[str, Any] | None = None
    roles: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    review: dict[str, Any] | None = None

    @property
    def repo(self) -> Path:
        return self.worktree or self.root

    @property
    def ledger(self) -> Path:
        return self.repo / "experiments" / "ledger.jsonl"

    @property
    def queue_path(self) -> Path:
        return self.repo / "experiments" / "hypotheses.json"

    def artifact(self, name: str) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def make_run_id(hypothesis: Hypothesis | None) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = hypothesis.id.lower() if hypothesis else "plan"
    return f"{stamp}-{suffix}"


def resolve_executor(requested: str, *, no_llm: bool = False) -> str:
    if no_llm:
        return "mechanical"
    if requested != "auto":
        return requested
    if os.environ.get("RL_MODEL_IMPROVE_EXECUTOR"):
        return os.environ["RL_MODEL_IMPROVE_EXECUTOR"]
    ensemble = (
        Path.home() / ".config" / "opencode" / "plugins" / "opencode-ensemble.js"
    )
    if not os.environ.get("CI") and ensemble.exists():
        return "team"
    return "passes"


def create_worktree(ctx: Context) -> None:
    """A dedicated worktree from HEAD so uncommitted main-tree work is untouched."""
    hypothesis = ctx.hypothesis
    assert hypothesis is not None
    date = time.strftime("%Y-%m-%d", time.gmtime())
    base = f"model-improve/{date}-{hypothesis.id.lower()}"
    branch = base
    attempt = 2
    while _git(ctx.root, "rev-parse", "--verify", f"refs/heads/{branch}").returncode == 0:
        branch = f"{base}-{attempt}"
        attempt += 1
    path = ctx.root / ".model-improve" / "worktrees" / ctx.run_id
    path.parent.mkdir(parents=True, exist_ok=True)
    result = _git(ctx.root, "worktree", "add", "-b", branch, str(path), "HEAD")
    if result.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {result.stderr.strip()}")
    ctx.worktree = path
    ctx.branch = branch


def remove_worktree(ctx: Context, *, keep: bool) -> None:
    if ctx.worktree is None or keep:
        return
    _git(ctx.root, "worktree", "remove", "--force", str(ctx.worktree))


# --------------------------------------------------------------------------- role execution


def role_prompt(ctx: Context, role: str) -> str:
    assert ctx.hypothesis is not None
    prereg_path = ctx.artifact("preregister.json")
    contracts = {
        "researcher": (
            "Write RUN_DIR/researcher.json with keys: ready (bool), "
            "recommendation ('proceed' or 'stop'), blockers (list of strings), "
            "summary (string). Also write RUN_DIR/diagnosis.md."
        ),
        "implementer": (
            "Write RUN_DIR/implementation.json with keys: status ('done' or "
            "'blocked'), summary (string), tests (list of command strings and "
            "pass/fail), files (list of changed paths). Also write "
            "RUN_DIR/implementation.md. Run `ruff check` and the offline test "
            "suite before reporting done."
        ),
        "auditor": (
            "Write RUN_DIR/audit.json with keys: blocking (bool), findings "
            "(list of {question, verdict: pass|fail|cannot_tell, evidence}). "
            "Any fail or cannot_tell must set blocking true."
        ),
        "lead": (
            "Orchestrate the team and write RUN_DIR/lead.json with keys: status "
            "('done' or 'blocked'), summary, and artifacts (paths). Ensure "
            "RUN_DIR/researcher.json, diagnosis.md, implementation.json, "
            "implementation.md and audit.json all exist and that merged code is "
            "committed in WORKTREE."
        ),
        "evaluator": (
            "Write RUN_DIR/evaluation.md: interpret the already-computed "
            "mechanical verdict. Never restate it as a different number and "
            "never claim profit."
        ),
        "reviewer": (
            "You are the higher-level reviewer. Read RUN_DIR/review-mechanical.json, "
            "roles.json, logs/*.jsonl, lab-*.json, report.md, preregister.json and "
            "the final worktree. Write RUN_DIR/review.json with keys: status "
            "('pass'|'warn'|'fail'), summary, findings (list of {severity: "
            "info|warn|fail, area, detail, evidence}), config_changes (list of "
            "strings), next_actions. Also write RUN_DIR/review.md. You may improve "
            ".opencode/ (agents, commands, skills) when a concrete defect is "
            "evidenced; record every change in config_changes. Do not edit "
            "config/model-improvement.json, experiments/, data/, or models/."
        ),
    }
    lines = [
        f"You are the {role} in an UNATTENDED model-improvement run.",
        f"RUN_DIR={ctx.run_dir}",
        f"WORKTREE={ctx.repo}",
        f"MODEL={resolve_model(ctx.config, role)}",
        f"HYPOTHESIS={ctx.hypothesis.id} — {ctx.hypothesis.title}",
        f"PREREG={prereg_path}",
        "Read your agent definition, the preregistration, and the repo skills "
        "(model-improvement, model-contract, backtest-protocol).",
        "",
        "Hard rules:",
        "- Do not ask the user anything. Do not wait for approval.",
        "- Do not edit data/, models/, or experiments/ledger.jsonl.",
        "- Do not push, merge to main, or touch another worktree.",
        "- If you cannot proceed, write the blocker to your artifact and stop.",
        "",
    ]
    if role == "lead":
        lines.append(
            "Pass `model=MODEL` to every `team_spawn` call (and to "
            "`team_message(model=...)` when changing a running teammate) so the "
            "whole team uses the configured model."
        )
        if ctx.hypothesis.needs_code:
            lines.append(
                "This hypothesis needs code: spawn the implementer, merge and "
                "test its branch, then spawn the auditor."
            )
        else:
            lines.append(
                "This hypothesis needs no code: spawn only the researcher. Do "
                "not spawn an implementer or auditor."
            )
        lines.append(
            "Do not end your turn until RUN_DIR/lead.json exists. Write it as "
            "your final action."
        )
        lines.append("")
    lines.extend(
        [
            contracts[role],
            "",
            "A parser validates the artifact. Keep it valid JSON; prose belongs "
            "in the .md file.",
        ]
    )
    return "\n".join(lines)


FALLBACK_MARKER = "is a subagent, not a primary agent"


def agent_fallback_banner(stdout: str, stderr: str) -> bool:
    """True only for the startup banner, never for a transcript echo.

    ``--format json`` puts tool output inside JSON events. The original
    detector scanned the whole transcript, so an agent that read this file
    (which contains the marker) was marked as a fallback. The real banner is
    plain text before the first JSON event.
    """
    for text in (stdout[:8192], stderr[:8192]):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("{"):
                break
            if FALLBACK_MARKER in stripped:
                return True
    return False


def _session_id(text: str) -> str | None:
    for line in text.splitlines()[:80]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        session = event.get("sessionID") or (event.get("part") or {}).get("sessionID")
        if session:
            return str(session)
    return None


def run_role(
    ctx: Context,
    role: str,
    *,
    session_id: str | None = None,
    turn: int = 1,
    extra_prompt: str = "",
) -> dict[str, Any]:
    if ctx.executor == "mechanical":
        return {"role": role, "status": "skipped", "reason": "mechanical executor"}
    agent = ROLE_AGENTS[role]
    model = resolve_model(ctx.config, role)
    timeout = float((ctx.config.get("budgets") or {}).get("roleTimeoutSeconds", 1800))
    command = [opencode_command(), "run"]
    if session_id:
        command.extend(["--session", session_id])
    command.extend(
        [
            "--agent",
            agent,
            "--format",
            "json",
            "--auto",
            "--dir",
            str(ctx.repo),
            "--title",
            f"model-improve {ctx.run_id} {role}"
            + (f" turn {turn}" if turn > 1 else ""),
        ]
    )
    if model:
        command.extend(["--model", str(model)])
    prompt = role_prompt(ctx, role)
    if extra_prompt:
        prompt += "\n\n" + extra_prompt
    command.append(prompt)
    started = time.time()
    try:
        proc = _run(
            command,
            cwd=ctx.repo,
            timeout=timeout,
            stdin_null=True,
            env={
                "RL_MODEL_IMPROVE_RUN_DIR": str(ctx.run_dir),
                "RL_MODEL_IMPROVE_WORKTREE": str(ctx.repo),
            },
        )
        returncode = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = str(exc.stdout or "")
        stderr = f"timeout after {timeout}s"
    log_name = f"{role}.jsonl" if turn == 1 else f"{role}-turn{turn}.jsonl"
    log = ctx.artifact(f"logs/{log_name}")
    log.write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")

    missing = [name for name in REQUIRED_ARTIFACTS[role] if not (ctx.run_dir / name).exists()]
    record: dict[str, Any] = {
        "role": role,
        "agent": agent,
        "model": model,
        "status": "done",
        "returncode": returncode,
        "seconds": round(time.time() - started, 1),
        "log": str(log),
        "missing": missing,
        "turn": turn,
        "session": _session_id(stdout) or session_id,
    }
    # A silent fallback to the default agent would bypass the role's
    # permissions, so treat it as a hard failure.
    if agent_fallback_banner(stdout, stderr):
        record["status"] = "failed"
        record["reason"] = (
            f"opencode refused {agent} (subagent) and fell back to the default "
            "agent; set the role agent's mode to `all` or `primary`"
        )
    elif returncode != 0:
        record["status"] = "failed"
        record["reason"] = f"opencode exited {returncode}"
    elif missing:
        record["status"] = "failed"
        record["reason"] = f"missing artifact(s): {', '.join(missing)}"
    ctx.roles.append(record)
    _write_json(ctx.artifact("roles.json"), ctx.roles)
    return record


def _artifact_status(ctx: Context, name: str) -> dict[str, Any] | None:
    return _read_json(ctx.run_dir / name)


def role_blocker(ctx: Context, role: str, record: Mapping[str, Any]) -> str | None:
    """Translate a role result into a stop reason, if any."""
    if record.get("status") != "done":
        return f"{role}: {record.get('reason') or record.get('status')}"
    if role == "researcher":
        payload = _artifact_status(ctx, "researcher.json") or {}
        if payload.get("ready") is False or payload.get("recommendation") == "stop":
            blockers = payload.get("blockers") or ["researcher recommended stop"]
            return "researcher: " + "; ".join(str(item) for item in blockers)
    if role == "implementer":
        payload = _artifact_status(ctx, "implementation.json") or {}
        if payload.get("status") != "done":
            return f"implementer: {payload.get('summary') or payload.get('status')}"
    if role == "auditor":
        payload = _artifact_status(ctx, "audit.json") or {}
        findings = payload.get("findings") or []
        bad = [
            str(item.get("question") or item)
            for item in findings
            if isinstance(item, Mapping)
            and item.get("verdict") in {"fail", "cannot_tell"}
        ]
        if payload.get("blocking") or bad:
            return "leakage audit blocking: " + ("; ".join(bad) or "blocking=true")
    if role == "lead":
        payload = _artifact_status(ctx, "lead.json") or {}
        if payload.get("status") != "done":
            return f"lead: {payload.get('summary') or payload.get('status')}"
    return None


# --------------------------------------------------------------------------- lab evaluation


def lab_command(ctx: Context) -> list[str]:
    override = os.environ.get("RL_MODEL_IMPROVE_LAB")
    if override:
        return shlex.split(override)
    return [sys.executable, str(ctx.repo / "tools" / "model_lab.py")]


def _lab_timeout(ctx: Context) -> float:
    budgets = ctx.config.get("budgets") or {}
    return float(budgets.get("labTimeoutSeconds", 3600))


def _common_lab_args(ctx: Context, experiment: Mapping[str, Any]) -> list[str]:
    scope = ctx.config.get("scope") or {}
    args = [
        "--tickers",
        *[str(t) for t in experiment.get("tickers") or scope.get("tickers") or []],
        "--targets",
        *[str(t) for t in experiment.get("targets") or scope.get("targets") or []],
        "--folds",
        str(int(experiment.get("folds") or (ctx.config.get("budgets") or {}).get("folds", 4))),
        "--threshold",
        str(float(experiment.get("threshold") or scope.get("threshold", 0.55))),
        "--position",
        str(experiment.get("position") or scope.get("position", "long-only")),
        "--fee-bps",
        str(float(experiment.get("feeBps") or scope.get("feeBps", 10.0))),
        "--slippage-bps",
        str(float(experiment.get("slippageBps") or scope.get("slippageBps", 5.0))),
    ]
    return args


def run_lab(ctx: Context, action: str, args: Sequence[str], *, label: str) -> dict[str, Any] | None:
    command = [*lab_command(ctx), action, *args, "--json"]
    env = {
        "RL_CRYPTO_ROOT": str(ctx.repo),
        "RL_MODEL_LAB_DIR": str(ctx.run_dir / "lab"),
        "RL_MODEL_IMPROVE_RUN_DIR": str(ctx.run_dir),
    }
    try:
        proc = _run(
            command,
            cwd=ctx.repo,
            env=env,
            timeout=_lab_timeout(ctx),
            stdin_null=True,
        )
    except subprocess.TimeoutExpired as exc:
        log = ctx.artifact(f"{label}.log")
        log.write_text(f"timeout after {_lab_timeout(ctx)}s\n{exc}\n", encoding="utf-8")
        ctx.artifacts[label] = str(log)
        return None

    stderr_path = ctx.artifact(f"{label}.stderr.log")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    try:
        payload = json.loads(proc.stdout) if proc.stdout.strip() else None
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        # Keep the artifact parseable: the ledger points at this file.
        artifact = ctx.artifact(f"{label}.json")
        _write_json(artifact, payload)
    else:
        artifact = ctx.artifact(f"{label}.log")
        artifact.write_text(
            (proc.stdout or "") + (("\n--- STDERR ---\n" + proc.stderr) if proc.stderr else ""),
            encoding="utf-8",
        )
    ctx.artifacts[label] = str(artifact)
    if proc.returncode != 0:
        return None
    return payload if isinstance(payload, dict) else None


def run_sweep(ctx: Context, prereg: Mapping[str, Any]) -> dict[str, Any] | None:
    experiment = dict(prereg.get("experiment") or {})
    budgets = ctx.config.get("budgets") or {}
    max_configs = min(
        int(experiment.get("maxConfigs") or budgets.get("maxConfigs", 8)),
        int(budgets.get("maxConfigs", 8)),
    )
    action = str(experiment.get("action") or "sweep")
    common = _common_lab_args(ctx, experiment)
    # The hypothesis declares the interface its implementation must add (e.g.
    # `--panel`). If the implementer did not add it, the lab exits nonzero and
    # the run is blocked instead of silently evaluating the old behaviour.
    extra = [str(item) for item in experiment.get("extraLabArgs") or []]
    if action == "evaluate":
        args = [
            *common,
            *extra,
            "--feature-set",
            str(experiment.get("featureSet") or "all"),
            "--model-set",
            str(experiment.get("modelSet") or "reference"),
            "--horizon",
            str(int(experiment.get("horizon") or 1)),
        ]
        return run_lab(ctx, "evaluate", args, label="lab-evaluate")
    args = [
        *common,
        *extra,
        "--feature-sets",
        ",".join(str(s) for s in experiment.get("featureSets") or ["all"]),
        "--model-sets",
        ",".join(str(s) for s in experiment.get("modelSets") or ["reference"]),
        "--horizons",
        ",".join(str(int(h)) for h in experiment.get("horizons") or [1]),
        "--max-configs",
        str(max_configs),
    ]
    return run_lab(ctx, "sweep", args, label="lab-sweep")


def run_promote(
    ctx: Context,
    prereg: Mapping[str, Any],
    sweep: Mapping[str, Any],
    *,
    apply: bool,
) -> dict[str, Any] | None:
    candidates = list(sweep.get("candidates") or [])
    if not candidates:
        return None
    candidate = str(candidates[0])
    ticker = None
    for row in sweep.get("top") or []:
        if row.get("candidate") == candidate:
            ticker = row.get("ticker")
            break
    experiment = dict(prereg.get("experiment") or {})
    extra = [str(item) for item in experiment.get("extraLabArgs") or []]
    args = [
        "--candidate",
        candidate,
        *extra,
        "--targets",
        *[str(t) for t in experiment.get("targets") or []],
        "--threshold",
        str(float(experiment.get("threshold") or 0.55)),
        "--position",
        str(experiment.get("position") or "long-only"),
        "--fee-bps",
        str(float(experiment.get("feeBps") or 10.0)),
        "--slippage-bps",
        str(float(experiment.get("slippageBps") or 5.0)),
    ]
    if ticker:
        args.extend(["--ticker", str(ticker)])
    if apply:
        args.append("--apply")
    return run_lab(ctx, "promote", args, label="lab-promote-apply" if apply else "lab-promote")


def evaluate_mechanical(ctx: Context, prereg: Mapping[str, Any]) -> dict[str, Any]:
    """Run the pre-registered experiment and compute the verdict."""
    sweep = run_sweep(ctx, prereg)
    promote_payload = None
    criteria = prereg.get("criteria") or {}
    if sweep is not None and criteria.get("requires_promote"):
        lineage = str(prereg.get("lineage"))
        touches = holdout_touches(read_ledger(ctx.ledger))
        if touches.get(lineage):
            return {
                "passed": False,
                "outcome": "blocked",
                "reasons": [f"holdout already touched for lineage {lineage}"],
                "metrics": {},
            }
        promote_payload = run_promote(ctx, prereg, sweep, apply=False)
        append_entry(
            ctx.ledger,
            {
                "kind": "holdout",
                "run_id": ctx.run_id,
                "hypothesis_id": prereg.get("hypothesis_id"),
                "lineage": lineage,
                "holdout_touched": True,
                "mode": "dry-run",
                "artifact": ctx.artifacts.get("lab-promote"),
                "passed": (promote_payload or {}).get("passed", False),
            },
        )
    verdict = evaluate_criteria(prereg, sweep, promote_payload=promote_payload)

    if sweep is not None and (
        ctx.artifacts.get("lab-sweep") or ctx.artifacts.get("lab-evaluate")
    ):
        append_entry(
            ctx.ledger,
            {
                "kind": "results",
                "run_id": ctx.run_id,
                "hypothesis_id": prereg.get("hypothesis_id"),
                "family": prereg.get("family"),
                "lineage": prereg.get("lineage"),
                "metrics": verdict.get("metrics") or {},
                "artifact": ctx.artifacts.get("lab-sweep")
                or ctx.artifacts.get("lab-evaluate"),
                "promote_artifact": ctx.artifacts.get("lab-promote"),
            },
        )

    if verdict.get("passed"):
        apply_requested = bool(ctx.config.get("promotion", {}).get("apply"))
        if apply_requested and promote_payload is not None and sweep is not None:
            applied = run_promote(ctx, prereg, sweep, apply=True)
            append_entry(
                ctx.ledger,
                {
                    "kind": "holdout",
                    "run_id": ctx.run_id,
                    "hypothesis_id": prereg.get("hypothesis_id"),
                    "lineage": prereg.get("lineage"),
                    "holdout_touched": True,
                    "mode": "apply",
                    "artifact": ctx.artifacts.get("lab-promote-apply"),
                    "written": (applied or {}).get("written"),
                },
            )
            verdict["metrics"]["applied"] = (applied or {}).get("written")
    if verdict.get("outcome") == "blocked":
        artifact = ctx.artifacts.get("lab-sweep") or ctx.artifacts.get("lab-evaluate")
        detail = ""
        if artifact:
            try:
                detail = " | ".join(Path(artifact).read_text(encoding="utf-8").splitlines()[-6:])
            except OSError:
                detail = ""
        append_entry(
            ctx.ledger,
            {
                "kind": "blocked",
                "run_id": ctx.run_id,
                "hypothesis_id": prereg.get("hypothesis_id"),
                "stage": "evaluation",
                "reason": "; ".join(verdict.get("reasons") or []),
                "artifact": artifact,
                "detail": detail,
            },
        )
    else:
        append_entry(
            ctx.ledger,
            {
                "kind": "verdict",
                "run_id": ctx.run_id,
                "hypothesis_id": prereg.get("hypothesis_id"),
                "title": prereg.get("title"),
                "family": prereg.get("family"),
                "lineage": prereg.get("lineage"),
                "trial_index": prereg.get("trial_index"),
                "outcome": verdict.get("outcome"),
                "passed": verdict.get("passed"),
                "reasons": verdict.get("reasons") or [],
                "metrics": verdict.get("metrics") or {},
                "prereg_id": prereg.get("id"),
                "artifact": ctx.artifacts.get("lab-sweep")
                or ctx.artifacts.get("lab-evaluate"),
            },
        )
    return verdict


# --------------------------------------------------------------------------- post-run review


def _role_done(ctx: Context, role: str) -> bool:
    return any(
        record.get("role") == role and record.get("status") == "done"
        for record in ctx.roles
    )


def mechanical_review(ctx: Context, verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic post-run audit of the session, git state and contract.

    This is the part a human would otherwise check by hand: was the experiment
    pre-registered, did every role complete, did the run touch data/models, is
    it on a non-main branch, and does the ledger contain the verdict it claims?
    The LLM reviewer gets this as its starting evidence and cannot upgrade a
    mechanical failure to a pass.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "fail") -> None:
        checks.append(
            {"name": name, "ok": bool(ok), "detail": detail, "severity": severity}
        )

    prereg = ctx.prereg or {}
    entries = read_ledger(ctx.ledger)
    ours = [entry for entry in entries if entry.get("run_id") == ctx.run_id]
    kinds = {str(entry.get("kind")) for entry in ours}

    add(
        "preregister",
        (ctx.run_dir / "preregister.json").exists() and bool(prereg.get("id")),
        f"pre-registration {prereg.get('id') or 'missing'}",
    )
    add(
        "ledger",
        "preregister" in kinds and bool({"verdict", "blocked"} & kinds),
        f"run ledger entries: {', '.join(sorted(kinds)) or 'none'}",
    )
    add(
        "verdict",
        verdict.get("outcome") in {"negative", "provisional", "diagnostic", "blocked"},
        f"experiment outcome: {verdict.get('outcome')}",
    )

    hypothesis = ctx.hypothesis
    if ctx.executor == "mechanical":
        add("roles", True, "mechanical executor: no LLM roles", severity="info")
    elif ctx.executor == "team":
        add("lead", _role_done(ctx, "lead"), "lead completed")
        required = ["researcher.json", "diagnosis.md"]
        if hypothesis is not None and hypothesis.needs_code:
            required += ["implementation.json", "implementation.md", "audit.json"]
        missing = [name for name in required if not (ctx.run_dir / name).exists()]
        add(
            "role-artifacts",
            not missing,
            f"missing: {', '.join(missing)}" if missing else "all role artifacts present",
        )
    else:
        expected = ["researcher"]
        if hypothesis is not None and hypothesis.needs_code:
            expected += ["implementer", "auditor"]
        missing_roles = [role for role in expected if not _role_done(ctx, role)]
        add(
            "roles",
            not missing_roles,
            f"missing/failed: {', '.join(missing_roles)}"
            if missing_roles
            else "all expected roles completed",
        )

    audit = _read_json(ctx.run_dir / "audit.json")
    if audit is not None:
        add(
            "audit",
            not audit.get("blocking"),
            "audit is not blocking" if not audit.get("blocking") else "audit blocking",
        )
    implementation = _read_json(ctx.run_dir / "implementation.json")
    if implementation is not None:
        add(
            "implementation",
            implementation.get("status") == "done",
            f"implementation status: {implementation.get('status')}",
        )

    git = git_state(ctx.repo)
    branch = git.get("branch")
    on_protected = branch in {None, "", "HEAD", "main", "master"}
    add(
        "branch",
        (not on_protected) or ctx.worktree is None,
        f"branch: {branch}",
        # A worktree is expected to be on its run branch. An in-place run is
        # not committed by this driver unless --commit, which refuses main.
        severity="fail" if ctx.worktree is not None else "info",
    )
    status_text = _git(ctx.repo, "status", "--porcelain").stdout
    changed = [
        line[3:].strip().strip('"')
        for line in status_text.splitlines()
        if line.strip()
    ]
    apply_enabled = bool((ctx.config.get("promotion") or {}).get("apply"))
    forbidden = [
        path
        for path in changed
        if path.startswith("data/") or (path.startswith("models/") and not apply_enabled)
    ]
    add(
        "scope",
        not forbidden,
        f"forbidden changes: {', '.join(forbidden)}"
        if forbidden
        else "data/ untouched; models/ write only when apply is enabled",
    )
    add("dirty", True, f"{len(changed)} uncommitted path(s) to commit", severity="info")

    if (prereg.get("criteria") or {}).get("requires_promote"):
        add(
            "holdout",
            "holdout" in kinds,
            "holdout touch recorded"
            if "holdout" in kinds
            else "holdout was required but not touched",
        )
    else:
        add(
            "holdout",
            True,
            "holdout not required by this experiment",
            severity="info",
        )

    failures = [check for check in checks if check["severity"] == "fail" and not check["ok"]]
    warnings = [check for check in checks if check["severity"] == "warn" and not check["ok"]]
    status = "fail" if failures else ("warn" if warnings else "pass")
    return {
        "status": status,
        "source": "mechanical",
        "checks": checks,
        "git": git,
        "changed": changed,
        "at": iso(),
    }


def run_review(
    ctx: Context,
    verdict: Mapping[str, Any],
    *,
    allow_llm: bool,
) -> dict[str, Any]:
    """Mechanical audit + optional higher-level reviewer, recorded in the ledger."""
    mechanical = mechanical_review(ctx, verdict)
    _write_json(ctx.artifact("review-mechanical.json"), mechanical)
    review: dict[str, Any] = dict(mechanical)

    if allow_llm and ctx.executor != "mechanical":
        record = run_role(ctx, "reviewer")
        payload = _artifact_status(ctx, "review.json") or {}
        if record.get("status") != "done" or not payload:
            review = {
                **mechanical,
                "source": "llm",
                "status": "fail" if mechanical["status"] == "fail" else "warn",
                "summary": (
                    "reviewer did not complete: "
                    f"{record.get('reason') or 'missing review.json'}"
                ),
                # checks are rendered from `checks`; do not copy them into
                # `findings` or the report shows every check twice.
                "findings": [],
            }
        else:
            llm_status = str(payload.get("status", "warn"))
            if mechanical["status"] == "fail" or llm_status == "fail":
                status = "fail"
            elif mechanical["status"] == "warn" or llm_status == "warn":
                status = "warn"
            else:
                status = "pass"
            review = {
                **payload,
                "status": status,
                "source": "llm",
                "mechanical": mechanical,
                "checks": mechanical.get("checks") or [],
                "git": payload.get("git") or mechanical.get("git"),
            }

    ctx.review = review
    # Normalise the artifact: mechanical-only runs and LLM runs both leave an
    # authoritative review.json (with the merged status and checks) for CI.
    _write_json(ctx.artifact("review.json"), review)
    append_entry(
        ctx.ledger,
        {
            "kind": "review",
            "run_id": ctx.run_id,
            "hypothesis_id": (ctx.hypothesis.id if ctx.hypothesis else None),
            "status": review.get("status"),
            "source": review.get("source"),
            "summary": review.get("summary"),
            "checks": review.get("checks") or [],
            "findings": review.get("findings") or [],
            "config_changes": review.get("config_changes") or [],
            "git": review.get("git") or git_state(ctx.repo),
        },
    )
    return review


def review_blocks_run(ctx: Context, review: Mapping[str, Any] | None) -> bool:
    if not review:
        return False
    if review.get("status") == "fail":
        return True
    require_pass = bool((ctx.config.get("review") or {}).get("require"))
    return require_pass and review.get("status") != "pass"


# --------------------------------------------------------------------------- report / git


def final_state(ctx: Context, commit: str | None) -> dict[str, Any]:
    """Post-commit state: what the branch actually contains now."""
    status = _git(ctx.repo, "status", "--porcelain").stdout.strip()
    log = _git(ctx.repo, "log", "--oneline", "origin/main..HEAD")
    if log.returncode != 0:
        log = _git(ctx.repo, "log", "--oneline", "-5")
    return {
        "at": iso(),
        "branch": _git(ctx.repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or None,
        "head": commit or _git(ctx.repo, "rev-parse", "HEAD").stdout.strip() or None,
        "clean": not bool(status),
        "uncommitted": status.splitlines() if status else [],
        "commits": log.stdout.strip().splitlines() if log.stdout.strip() else [],
    }


def build_report(
    ctx: Context,
    verdict: Mapping[str, Any],
    *,
    next_hypothesis: Hypothesis | None,
    final: Mapping[str, Any] | None = None,
) -> str:
    assert ctx.prereg is not None
    prereg = ctx.prereg
    metrics = verdict.get("metrics") or {}
    lines = [
        f"# Model-improvement run `{ctx.run_id}`",
        "",
        f"- **Outcome:** `{verdict.get('outcome')}`"
        f" ({'passed' if verdict.get('passed') else 'did not pass'})",
        f"- **Hypothesis:** {prereg.get('hypothesis_id')} — {prereg.get('title')}",
        f"- **Executor:** `{ctx.executor}` (trigger: `{ctx.trigger}`)",
        f"- **Branch:** `{ctx.branch or git_state(ctx.repo).get('branch') or 'in-place'}`",
        f"- **Pre-registration:** `{prereg.get('id')}`"
        f" (family `{prereg.get('family')}`, trial {prereg.get('trial_index')})",
        f"- **Git:** `{((prereg.get('git') or {}).get('sha') or 'unknown')[:12]}`",
        f"- **Data:** `{((prereg.get('data') or {}).get('combined') or 'absent')[:16]}`",
        "",
        "## Verdict reasons",
        "",
    ]
    reasons = verdict.get("reasons") or []
    if verdict.get("outcome") == "diagnostic":
        lines.append(
            "- diagnostic completed; the criteria are recorded but non-gating "
            "(there is no pass/fail bar for a diagnostic)"
        )
    else:
        lines.extend(
            [f"- {reason}" for reason in reasons]
            or ["- none: all pre-registered bars met"]
        )
    lines.extend(["", "## Per-ticker result", ""])
    per_ticker = metrics.get("per_ticker") or {}
    targets = metrics.get("targets") or []
    if targets and per_ticker:
        # Diagnostic runs cover several targets, so show a matrix rather than
        # pretending one target is the verdict.
        lines.extend(
            [
                "| ticker | target | folds beating | mean improvement | min improvement |",
                "| ------ | ------ | ------------- | ---------------- | --------------- |",
            ]
        )
        for ticker, info in sorted(per_ticker.items()):
            for target in targets:
                stats = (info.get("targets") or {}).get(target)
                if not isinstance(stats, Mapping):
                    continue
                mean = stats.get("mean_improvement")
                minimum = stats.get("min_improvement")
                mean_text = f"{mean:+.3f}" if isinstance(mean, int | float) else "—"
                min_text = f"{minimum:+.3f}" if isinstance(minimum, int | float) else "—"
                lines.append(
                    f"| {ticker} | {target} | "
                    f"{stats.get('folds_beating')}/{stats.get('folds')} | "
                    f"{mean_text} | {min_text} |"
                )
        lines.extend(
            [
                "",
                "### After-cost policy (per ticker)",
                "",
                "| ticker | total return | Sharpe | trades |",
                "| ------ | ------------ | ------ | ------ |",
            ]
        )
        for ticker, info in sorted(per_ticker.items()):
            economic = info.get("economic") or {}
            lines.append(
                f"| {ticker} | {economic.get('total_return')} | "
                f"{economic.get('sharpe')} | {economic.get('trades')} |"
            )
    elif per_ticker:
        lines.extend(
            [
                "| ticker | folds beating | mean improvement | min improvement | beats |",
                "| ------ | ------------- | ---------------- | --------------- | ----- |",
            ]
        )
        for ticker, stats in sorted(per_ticker.items()):
            if stats.get("absent"):
                lines.append(f"| {ticker} | — | — | — | missing |")
                continue
            mean = stats.get("mean_improvement")
            minimum = stats.get("min_improvement")
            if isinstance(mean, int | float) and isinstance(minimum, int | float):
                lines.append(
                    f"| {ticker} | {stats.get('folds_beating')}/{stats.get('folds')} | "
                    f"{mean:+.3f} | {minimum:+.3f} | "
                    f"{'yes' if stats.get('beats') else 'no'} |"
                )
            else:
                lines.append(f"| {ticker} | — | — | — | no |")
    else:
        lines.append("_No per-ticker metrics were produced._")
    if metrics.get("applied"):
        lines.extend(
            [
                "",
                "## Promotion",
                "",
                f"- wrote `{metrics['applied']}` on the run branch.",
                "- the scheduled retrain still overwrites `models/` until "
                "`config/training.json` is read by production; this is not"
                " durable yet.",
            ]
        )
    economic = metrics.get("economic") or {}
    if economic:
        lines.extend(
            [
                "",
                "## Economics (after costs)",
                "",
                f"- ticker: `{economic.get('ticker')}`",
                f"- total return: `{economic.get('total_return')}`",
                f"- Sharpe: `{economic.get('sharpe')}`"
                f" vs momentum `{economic.get('momentum_sharpe')}`",
                f"- trades: `{economic.get('trades')}`",
            ]
        )
    lines.extend(["", "## Roles", ""])
    if ctx.roles:
        lines.extend(
            f"- `{role.get('role')}`: {role.get('status')}"
            f" ({role.get('seconds', 0)}s, {role.get('model') or 'default model'})"
            for role in ctx.roles
        )
    else:
        lines.append("- mechanical executor: no LLM roles ran")
    review = ctx.review or {}
    if review:
        git = review.get("git") or {}
        lines.extend(
            [
                "",
                "## Session review",
                "",
                f"- status: `{review.get('status')}` (source: `{review.get('source')}`)",
                f"- git: `{git.get('branch')}` @ "
                f"`{(git.get('sha') or 'unknown')[:12]}`"
                f" (dirty={git.get('dirty')})",
            ]
        )
        if review.get("summary"):
            lines.append(f"- summary: {review['summary']}")
        for check in review.get("checks") or []:
            if isinstance(check, Mapping):
                state = "ok" if check.get("ok") else "FAIL"
                lines.append(
                    f"- [{state}] {check.get('name', '')}: {check.get('detail', '')}"
                )
        findings = review.get("findings") or []
        for finding in findings:
            if isinstance(finding, Mapping):
                if not finding.get("detail"):
                    continue
                lines.append(
                    f"- [{finding.get('severity', 'info')}] "
                    f"{finding.get('area', '')}: {finding.get('detail', '')}".strip()
                )
            else:
                lines.append(f"- {finding}")
        for change in review.get("config_changes") or []:
            lines.append(f"- config change: {change}")
        for action in review.get("next_actions") or []:
            lines.append(f"- next: {action}")
    if final:
        lines.extend(
            [
                "",
                "## Final state (post-commit)",
                "",
                f"- branch: `{final.get('branch')}`",
                f"- head: `{(final.get('head') or 'unknown')[:12]}`",
                f"- clean: `{final.get('clean')}`",
            ]
        )
        if final.get("uncommitted"):
            lines.append(
                "- uncommitted: " + ", ".join(str(p) for p in final["uncommitted"][:10])
            )
        for commit_line in final.get("commits") or []:
            lines.append(f"- commit: `{commit_line}`")
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- `{path}`" for path in sorted(ctx.artifacts.values()))
    for name in (
        "preflight.json",
        "preregister.json",
        "roles.json",
        "review-mechanical.json",
        "review.json",
        "final-state.json",
        "logs",
    ):
        path = ctx.run_dir / name
        if path.exists():
            lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Next",
            "",
            f"- next runnable hypothesis: "
            f"`{next_hypothesis.id if next_hypothesis else 'none'}`",
            "",
            "This report is generated mechanically from the ledger and lab JSON.",
            "It is not a profit claim; a negative result is a valid completion.",
            "",
        ]
    )
    return "\n".join(lines)


def regenerate_records(ctx: Context) -> None:
    entries = read_ledger(ctx.ledger)
    write_if_changed(ctx.repo / "experiments" / "SCOREBOARD.md", render_scoreboard(entries))
    queue = load_queue(ctx.queue_path)
    write_if_changed(
        ctx.repo / "experiments" / "hypotheses.md",
        render_hypotheses(queue, entries),
    )


def commit_run(
    ctx: Context,
    verdict: Mapping[str, Any],
    *,
    allow_main: bool,
) -> str | None:
    branch = _git(ctx.repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch in {"main", "master"} and not allow_main:
        raise RuntimeError(
            "refusing to commit the run on main; pass --allow-main-commit only "
            "when you mean it"
        )
    paths = [
        "experiments",
        "config",
        "src",
        "tests",
        "tools",
        "notebooks",
        "examples",
        ".opencode",
        ".github",
        ".gitignore",
    ]
    existing = [path for path in paths if (ctx.repo / path).exists()]
    _git(ctx.repo, "add", "-A", "--", *existing)
    if ctx.config.get("promotion", {}).get("apply") and (ctx.repo / "models").exists():
        _git(ctx.repo, "add", "-A", "--", "models")
    staged = _git(ctx.repo, "diff", "--cached", "--quiet")
    if staged.returncode == 0:
        return None
    hypothesis = ctx.hypothesis
    assert hypothesis is not None
    message = (
        f"model-improve({hypothesis.id}): {verdict.get('outcome')} — {hypothesis.title}"
    )
    commit = _git(ctx.repo, "commit", "-m", message)
    if commit.returncode != 0:
        raise RuntimeError(f"git commit failed: {commit.stderr.strip()}")
    return _git(ctx.repo, "rev-parse", "HEAD").stdout.strip()


# --------------------------------------------------------------------------- commands


def _emit(payload: Mapping[str, Any], *, as_json: bool, table: bool = True) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if not table:
        return
    checks = payload.get("checks") or []
    rows = [
        [
            check.get("name"),
            "ok" if check.get("ok") else "FAIL",
            "blocking" if check.get("blocking") else "warn",
            check.get("detail"),
        ]
        for check in checks
    ]
    if rows:
        _print_table(["check", "state", "severity", "detail"], rows)
    if payload.get("blocking"):
        print(f"\nblocking: {', '.join(payload['blocking'])}")
    if payload.get("warnings"):
        print(f"warnings: {', '.join(payload['warnings'])}")


def cmd_preflight(args: argparse.Namespace) -> int:
    root = repo_root()
    config = apply_model_override(load_config(root), getattr(args, "model", None))
    executor = resolve_executor(args.executor, no_llm=args.no_llm)
    payload = preflight(root, config, executor=executor)
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        _write_json(run_dir / "preflight.json", payload)
    _emit(payload, as_json=args.json)
    return 0 if payload["ok"] else 2


def _load_state(
    root: Path,
    *,
    model: str | None = None,
) -> tuple[list[Hypothesis], list[dict[str, Any]], dict[str, Any]]:
    config = apply_model_override(load_config(root), model)
    queue = load_queue(root / "experiments" / "hypotheses.json")
    entries = read_ledger(root / "experiments" / "ledger.jsonl")
    return queue, entries, config


def _pick(
    args: argparse.Namespace,
    queue: Sequence[Hypothesis],
    entries: Sequence[Mapping[str, Any]],
    *,
    executor: str,
) -> Hypothesis | None:
    if getattr(args, "hypothesis", None):
        wanted = args.hypothesis.upper()
        for hypothesis in queue:
            if hypothesis.id.upper() == wanted:
                return hypothesis
        raise SystemExit(f"unknown hypothesis {args.hypothesis!r}")
    capabilities = {"code", "tooling"} if executor != "mechanical" else set()
    return select_hypothesis(queue, entries, capabilities=capabilities)


def cmd_plan(args: argparse.Namespace) -> int:
    root = repo_root()
    executor = resolve_executor(args.executor, no_llm=args.no_llm)
    queue, entries, config = _load_state(root, model=getattr(args, "model", None))
    hypothesis = _pick(args, queue, entries, executor=executor)
    if hypothesis is None:
        print(
            "no runnable hypothesis: every open hypothesis is done or needs a "
            "capability this run lacks"
        )
        return 0
    run_id = make_run_id(hypothesis)
    prereg = build_preregistration(
        hypothesis,
        run_id=run_id,
        root=root,
        entries=entries,
        trigger=args.trigger,
        seed=int(config.get("seed") or 0),
    )
    payload = {
        "run_id": run_id,
        "executor": executor,
        "model": resolve_model(config),
        "models": {role: resolve_model(config, role) for role in ROLE_AGENTS},
        "hypothesis": hypothesis.as_dict(),
        "preregister": prereg,
        "preflight": preflight(root, config, executor=executor),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(f"run:        {run_id}")
        print(f"executor:   {executor}")
        print(f"model:      {payload['model']}")
        print(f"hypothesis: {hypothesis.id} — {hypothesis.title}")
        print(f"family:     {hypothesis.family}")
        print(f"needs:      {', '.join(hypothesis.needs) or 'nothing (mechanical)'}")
        print(f"target:     {hypothesis.criteria.get('target', '—')}")
        print(f"trial:      {prereg.get('trial_index')} in family")
        print(f"experiment: {json.dumps(hypothesis.experiment, sort_keys=True)}")
        print(f"criteria:   {json.dumps(hypothesis.criteria, sort_keys=True)}")
    return 0 if payload["preflight"]["ok"] else 2


def cmd_status(args: argparse.Namespace) -> int:
    root = repo_root()
    executor = resolve_executor(args.executor, no_llm=args.no_llm)
    queue, entries, config = _load_state(root, model=getattr(args, "model", None))
    next_hypothesis = _pick(args, queue, entries, executor=executor)
    counts = trial_counts(entries)
    verdicts = [entry for entry in entries if entry.get("kind") == "verdict"]
    payload = {
        "entries": len(entries),
        "model": resolve_model(config),
        "preregistrations": sum(1 for entry in entries if entry.get("kind") == "preregister"),
        "verdicts": verdicts[-int(args.limit) :],
        "trial_counts": counts,
        "next_hypothesis": next_hypothesis.id if next_hypothesis else None,
        "scoreboard": render_scoreboard(entries),
    }
    if args.render:
        write_if_changed(root / "experiments" / "SCOREBOARD.md", render_scoreboard(entries))
        write_if_changed(
            root / "experiments" / "hypotheses.md",
            render_hypotheses(queue, entries),
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    print(f"ledger:          {root / 'experiments' / 'ledger.jsonl'}")
    print(f"model:           {payload['model']}")
    print(f"entries:         {payload['entries']}")
    print(f"preregistrations:{payload['preregistrations']}")
    print(f"verdicts:        {len(verdicts)}")
    print(f"next hypothesis: {payload['next_hypothesis'] or 'none'}")
    if verdicts:
        print()
        _print_table(
            ["id", "at", "hypothesis", "outcome", "reasons"],
            [
                [
                    entry.get("id"),
                    str(entry.get("at", ""))[:10],
                    entry.get("hypothesis_id"),
                    entry.get("outcome"),
                    "; ".join(entry.get("reasons") or [])[:70],
                ]
                for entry in verdicts[-int(args.limit) :]
            ],
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    args.render = True
    return cmd_status(args)


def cmd_doctor(args: argparse.Namespace) -> int:
    root = repo_root()
    config = apply_model_override(load_config(root), getattr(args, "model", None))
    executor = resolve_executor(args.executor, no_llm=args.no_llm)
    opencode = shutil.which(opencode_command())
    version = None
    if opencode:
        proc = _run([opencode, "--version"], timeout=30)
        output = (proc.stdout or proc.stderr).strip().splitlines()
        version = output[0] if output else None
    agents = {
        name: (root / ".opencode" / "agents" / f"{name}.md").exists()
        for name in ROLE_AGENTS.values()
    }
    modes = agent_modes()
    unusable = [name for name in ROLE_AGENTS.values() if modes.get(name) == "subagent"]
    ensemble = Path.home() / ".config" / "opencode" / "plugins" / "opencode-ensemble.js"
    lab_override = os.environ.get("RL_MODEL_IMPROVE_LAB")
    payload = {
        "root": str(root),
        "executor": executor,
        "model": resolve_model(config),
        "models": {role: resolve_model(config, role) for role in ROLE_AGENTS},
        "opencode": opencode,
        "opencode_version": version,
        "agents": agents,
        "agent_modes": modes,
        "unusable_agents": unusable,
        "ensemble_plugin": str(ensemble) if ensemble.exists() else None,
        "lab_command": lab_override or [sys.executable, str(root / "tools" / "model_lab.py")],
        "config": str(root / "config" / "model-improvement.json"),
        "run_root": str(default_run_root()),
        "env": {
            "RL_MODEL_IMPROVE_EXECUTOR": os.environ.get("RL_MODEL_IMPROVE_EXECUTOR"),
            "RL_MODEL_IMPROVE_MODEL": os.environ.get("RL_MODEL_IMPROVE_MODEL"),
            "RL_MODEL_IMPROVE_RUN_DIR": os.environ.get("RL_MODEL_IMPROVE_RUN_DIR"),
            "RL_MODEL_IMPROVE_LAB": lab_override,
            "CI": os.environ.get("CI"),
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    print(f"root:             {payload['root']}")
    print(f"executor:         {payload['executor']}")
    print(f"model:            {payload['model']}")
    print(f"opencode:         {payload['opencode']} ({payload['opencode_version']})")
    print(f"ensemble plugin:  {payload['ensemble_plugin'] or 'not installed'}")
    print(f"run root:         {payload['run_root']}")
    for name, present in agents.items():
        mode = modes.get(name, "?")
        state = "MISSING" if not present else ("SUBAGENT" if mode == "subagent" else "ok")
        print(f"agent {name:<24} {state:<8} mode={mode}")
    if not opencode and executor != "mechanical":
        print("opencode is not on PATH", file=sys.stderr)
        return 2
    return 0 if all(agents.values()) and not unusable else 2


def cmd_verify(args: argparse.Namespace) -> int:
    root = repo_root()
    config = load_config(root)
    scope = config.get("scope") or {}
    tickers = [str(t) for t in scope.get("tickers") or []]
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else default_run_root() / make_run_id(None)
    )
    ctx = Context(
        root=root,
        config=config,
        executor="mechanical",
        trigger="verify",
        run_id=make_run_id(None),
        run_dir=run_dir,
    )
    status = run_lab(ctx, "status", ["--tickers", *tickers], label="lab-status")
    forward_path = root / "tools" / "forward_score.py"
    forward: dict[str, Any] | None = None
    if forward_path.exists():
        proc = _run(
            [sys.executable, str(forward_path), "--json"],
            cwd=root,
            env={"RL_CRYPTO_ROOT": str(root)},
            timeout=900,
            stdin_null=True,
        )
        (ctx.artifact("forward-score.json")).write_text(
            proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8"
        )
        try:
            forward = json.loads(proc.stdout)
        except json.JSONDecodeError:
            forward = {"error": "forward_score.py did not emit JSON"}
    payload = {
        "status": status,
        "forward_score": forward,
        "forward_available": forward_path.exists(),
        "artifact": str(run_dir),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    if status:
        print("bundles:")
        _print_table(
            ["ticker", "bars", "feature", "trusted", "error"],
            [
                [
                    row.get("ticker"),
                    row.get("bars"),
                    row.get("feature_version") or "—",
                    ",".join(row.get("trusted") or []) or "—",
                    row.get("bundle_error") or "",
                ]
                for row in status.get("tickers") or []
            ],
        )
    else:
        print("could not read lab status")
    forward_state = (
        "ran" if payload["forward_available"] else "not implemented yet (tools/forward_score.py)"
    )
    print(f"\nforward scorer: {forward_state}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = repo_root()
    config = apply_model_override(load_config(root), getattr(args, "model", None))
    if getattr(args, "apply", None) is not None:
        config.setdefault("promotion", {})["apply"] = bool(args.apply)
    executor = resolve_executor(args.executor, no_llm=args.no_llm)
    queue = load_queue(root / "experiments" / "hypotheses.json")
    entries = read_ledger(root / "experiments" / "ledger.jsonl")
    hypothesis = _pick(args, queue, entries, executor=executor)
    if hypothesis is None:
        payload = {"outcome": "no-runnable-hypothesis", "executor": executor}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                "no runnable hypothesis: every open hypothesis is done or needs a "
                "capability this run lacks"
            )
        return 0

    run_id = make_run_id(hypothesis)
    run_root = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else default_run_root()
    )
    ctx = Context(
        root=root,
        config=config,
        executor=executor,
        trigger=args.trigger,
        run_id=run_id,
        run_dir=run_root / run_id,
        hypothesis=hypothesis,
    )
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    pf = preflight(root, config, executor=executor, worktree=not args.no_worktree)
    _write_json(ctx.artifact("preflight.json"), pf)
    if not pf["ok"] and not args.force:
        _emit(pf, as_json=args.json)
        print("\npreflight failed; pass --force to run anyway", file=sys.stderr)
        return 2

    prereg = build_preregistration(
        hypothesis,
        run_id=run_id,
        root=root,
        entries=entries,
        trigger=args.trigger,
        seed=int(config.get("seed") or 0),
    )
    if not args.force and (hypothesis.criteria or {}).get("kind") == "diagnostic":
        prior_preregs = [
            entry
            for entry in entries
            if entry.get("kind") == "preregister"
            and entry.get("hypothesis_id") == hypothesis.id
            and entry.get("config_hash") == prereg["config_hash"]
            and (entry.get("data") or {}).get("combined")
            == (prereg.get("data") or {}).get("combined")
        ]
        prior_runs = {entry.get("run_id") for entry in prior_preregs}
        prior_verdicts = [
            entry
            for entry in entries
            if entry.get("kind") == "verdict" and entry.get("run_id") in prior_runs
        ]
        if prior_verdicts:
            message = (
                "duplicate diagnostic: this hypothesis/config/data already has "
                f"verdict {prior_verdicts[-1].get('id')}; pass --force to repeat "
                "or change the data/features"
            )
            if args.json:
                print(
                    json.dumps(
                        {"outcome": "duplicate-diagnostic", "reason": message},
                        indent=2,
                    )
                )
            else:
                print(message)
            return 0

    ctx.prereg = prereg
    _write_json(ctx.artifact("preregister.json"), prereg)
    _write_json(
        ctx.artifact("run.json"),
        {
            "run_id": run_id,
            "executor": executor,
            "trigger": args.trigger,
            "hypothesis": hypothesis.as_dict(),
            "preregister": prereg,
        },
    )
    if args.dry_run:
        print(f"dry run: {run_id} ({executor}) would pre-register {hypothesis.id}")
        print(json.dumps(prereg, indent=2, sort_keys=True, default=str))
        return 0

    created = False
    if not args.no_worktree:
        try:
            create_worktree(ctx)
            created = True
        except RuntimeError as exc:
            print(f"could not create worktree: {exc}", file=sys.stderr)
            return 1

    record = append_entry(ctx.ledger, prereg)
    prereg["id"] = record["id"]
    _write_json(ctx.artifact("preregister.json"), prereg)
    if args.commit is None:
        args.commit = created

    def stop(reason: str, stage: str) -> int:
        append_entry(
            ctx.ledger,
            {
                "kind": "blocked",
                "run_id": run_id,
                "hypothesis_id": hypothesis.id,
                "stage": stage,
                "reason": reason,
            },
        )
        verdict = {"outcome": "blocked", "passed": False, "reasons": [reason], "metrics": {}}
        pre_review_next = select_hypothesis(
            load_queue(ctx.queue_path),
            read_ledger(ctx.ledger),
            capabilities={"code", "tooling"} if executor != "mechanical" else set(),
        )
        ctx.artifact("report.md").write_text(
            build_report(ctx, verdict, next_hypothesis=pre_review_next),
            encoding="utf-8",
        )
        review_enabled = bool((ctx.config.get("review") or {}).get("enabled", True))
        if review_enabled and not args.no_review:
            run_review(ctx, verdict, allow_llm=executor != "mechanical")
        next_hypothesis = select_hypothesis(
            load_queue(ctx.queue_path),
            read_ledger(ctx.ledger),
            capabilities={"code", "tooling"} if executor != "mechanical" else set(),
        )
        report = build_report(ctx, verdict, next_hypothesis=next_hypothesis)
        ctx.artifact("report.md").write_text(report, encoding="utf-8")
        regenerate_records(ctx)
        commit = None
        if args.commit:
            try:
                commit = commit_run(ctx, verdict, allow_main=args.allow_main_commit)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
        final = final_state(ctx, commit)
        _write_json(ctx.artifact("final-state.json"), final)
        report = build_report(
            ctx, verdict, next_hypothesis=next_hypothesis, final=final
        )
        ctx.artifact("report.md").write_text(report, encoding="utf-8")
        if created:
            remove_worktree(ctx, keep=args.keep_worktree)
        if args.json:
            print(json.dumps({**verdict, "run_id": run_id, "commit": commit}, indent=2))
        else:
            print(report)
        return 2

    if executor == "mechanical":
        pass
    elif executor == "team":
        # A headless lead can end its turn after spawning teammates, before it
        # has their results. Resume the same session (bounded) instead of
        # treating that as a hard failure.
        max_turns = int((ctx.config.get("team") or {}).get("maxLeadTurns", 3))
        record = run_role(ctx, "lead")
        turn = 1
        while (
            turn < max_turns
            and record.get("returncode") == 0
            and "lead.json" in (record.get("missing") or [])
            and record.get("session")
        ):
            turn += 1
            record = run_role(
                ctx,
                "lead",
                session_id=str(record["session"]),
                turn=turn,
                extra_prompt=(
                    "CONTINUE: your previous turn ended before writing "
                    "RUN_DIR/lead.json. Wait for the teammates' messages, merge "
                    "and test the implementer's work, run the auditor, and write "
                    "RUN_DIR/lead.json as your final action. Do not end this "
                    "turn before that file exists."
                ),
            )
        blocker = role_blocker(ctx, "lead", record)
        if blocker:
            return stop(blocker, "lead")
        required = ["researcher.json", "diagnosis.md"]
        if hypothesis.needs_code:
            required += ["implementation.json", "implementation.md", "audit.json"]
        missing = [name for name in required if not (ctx.run_dir / name).exists()]
        if missing:
            return stop(f"lead missing artifact(s): {', '.join(missing)}", "lead")
        if hypothesis.needs_code and not _git(ctx.repo, "status", "--porcelain").stdout.strip():
            return stop("lead reported done but the worktree changed no files", "lead")
    else:
        record = run_role(ctx, "researcher")
        blocker = role_blocker(ctx, "researcher", record)
        if blocker:
            return stop(blocker, "researcher")
        if hypothesis.needs_code:
            record = run_role(ctx, "implementer")
            blocker = role_blocker(ctx, "implementer", record)
            if blocker:
                return stop(blocker, "implementer")
            changed = _git(ctx.repo, "status", "--porcelain").stdout.strip()
            if not changed:
                return stop("implementer reported done but changed no files", "implementer")
            record = run_role(ctx, "auditor")
            blocker = role_blocker(ctx, "auditor", record)
            if blocker:
                return stop(blocker, "auditor")

    verdict = evaluate_mechanical(ctx, prereg)

    if executor != "mechanical" and args.with_evaluator:
        record = run_role(ctx, "evaluator")
        if record.get("status") != "done":
            note = f"evaluator prose pass: {record.get('status')}"
            verdict.setdefault("reasons", []).append(note)

    # Write a provisional report before the reviewer: the reviewer is allowed
    # to read report.md, and it is rebuilt with the review + final state after.
    pre_review_next = select_hypothesis(
        load_queue(ctx.queue_path),
        read_ledger(ctx.ledger),
        capabilities={"code", "tooling"} if executor != "mechanical" else set(),
    )
    ctx.artifact("report.md").write_text(
        build_report(ctx, verdict, next_hypothesis=pre_review_next),
        encoding="utf-8",
    )

    review: dict[str, Any] | None = None
    review_enabled = bool((ctx.config.get("review") or {}).get("enabled", True))
    if review_enabled and not args.no_review:
        review = run_review(ctx, verdict, allow_llm=executor != "mechanical")

    next_hypothesis = select_hypothesis(
        load_queue(ctx.queue_path),
        read_ledger(ctx.ledger),
        capabilities={"code", "tooling"} if executor != "mechanical" else set(),
    )
    report = build_report(ctx, verdict, next_hypothesis=next_hypothesis)
    ctx.artifact("report.md").write_text(report, encoding="utf-8")
    regenerate_records(ctx)

    commit = None
    if args.commit:
        try:
            commit = commit_run(ctx, verdict, allow_main=args.allow_main_commit)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    final = final_state(ctx, commit)
    _write_json(ctx.artifact("final-state.json"), final)
    report = build_report(ctx, verdict, next_hypothesis=next_hypothesis, final=final)
    ctx.artifact("report.md").write_text(report, encoding="utf-8")
    if created:
        remove_worktree(ctx, keep=args.keep_worktree)

    summary = {
        "run_id": run_id,
        "executor": executor,
        "hypothesis": hypothesis.id,
        "outcome": verdict.get("outcome"),
        "passed": verdict.get("passed"),
        "reasons": verdict.get("reasons") or [],
        "review": {
            "status": (review or {}).get("status"),
            "source": (review or {}).get("source"),
        }
        if review
        else None,
        "branch": ctx.branch,
        "commit": commit,
        "run_dir": str(ctx.run_dir),
        "report": str(ctx.artifact("report.md")),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(report)
        print(f"artifact: {ctx.run_dir}")
        if ctx.branch:
            print(f"branch:   {ctx.branch}")
    return 2 if review_blocks_run(ctx, review) else 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-improve",
        description="Unattended model-improvement loop (ledger + lab + bounded LLM roles).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def executor_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--executor",
            choices=("auto", "team", "passes", "mechanical"),
            default="auto",
            help="team = Ensemble, passes = sequential headless roles, mechanical = no LLM",
        )
        p.add_argument("--no-llm", action="store_true", help="force the mechanical executor")
        p.add_argument(
            "--model",
            default=None,
            help="override config.model for every role (provider/model)",
        )
        p.add_argument("--json", action="store_true")

    pre = sub.add_parser("preflight", help="check data, bundles, foundation and tooling")
    executor_args(pre)
    pre.add_argument("--run-dir", default=None, help="write preflight.json here")
    pre.add_argument("--trigger", default="manual")
    pre.set_defaults(func=cmd_preflight)

    plan = sub.add_parser("plan", help="show the next pre-registration without executing it")
    executor_args(plan)
    plan.add_argument("--hypothesis", default=None)
    plan.add_argument("--trigger", default="manual")
    plan.set_defaults(func=cmd_plan)

    status = sub.add_parser("status", help="ledger summary and next hypothesis")
    executor_args(status)
    status.add_argument("--hypothesis", default=None)
    status.add_argument("--limit", type=int, default=10)
    status.add_argument("--render", action="store_true", help="regenerate the markdown records")
    status.set_defaults(func=cmd_status)

    report = sub.add_parser("report", help="status + regenerate the markdown records")
    executor_args(report)
    report.add_argument("--hypothesis", default=None)
    report.add_argument("--limit", type=int, default=20)
    report.add_argument("--render", action="store_true", default=True)
    report.set_defaults(func=cmd_report)

    run = sub.add_parser("run", help="run one pre-registered experiment")
    executor_args(run)
    run.add_argument("--hypothesis", default=None)
    run.add_argument("--trigger", default="manual")
    run.add_argument("--run-dir", default=None)
    run.add_argument("--worktree", dest="no_worktree", action="store_false", default=True)
    run.add_argument("--no-worktree", dest="no_worktree", action="store_true")
    run.add_argument(
        "--commit",
        dest="commit",
        action="store_true",
        default=None,
        help="commit the run; defaults to true when a worktree was created",
    )
    run.add_argument("--no-commit", dest="commit", action="store_false")
    run.add_argument("--keep-worktree", action="store_true")
    run.add_argument("--allow-main-commit", action="store_true")
    run.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        default=None,
        help="write models/<slug>.json after a passing gate (overrides config)",
    )
    run.add_argument("--no-apply", dest="apply", action="store_false")
    run.add_argument("--with-evaluator", action="store_true", help="add the prose evaluator pass")
    run.add_argument(
        "--no-review",
        action="store_true",
        help="skip the post-run reviewer (mechanical checks still run)",
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--force",
        action="store_true",
        help="run even if preflight has blocking failures",
    )
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="forward-test status and bundle freshness")
    verify.add_argument("--run-dir", default=None)
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=cmd_verify)

    doctor = sub.add_parser("doctor", help="tooling, agents and configuration")
    executor_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())


