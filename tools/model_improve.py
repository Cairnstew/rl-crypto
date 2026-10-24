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
import math
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
    CANARY_FILENAME,
    FAMILY_TRIAL_CAP_DEFAULT,
    PROTECTED_FILES,
    SPENDING_SCHEDULE,
    Hypothesis,
    alpha_for_trial,
    append_canary,
    append_entry,
    bootstrap_resolution,
    build_preregistration,
    calibration_verdict,
    canary_gate,
    canary_status,
    consumed_runs,
    contract_violation,
    criteria_version,
    evaluate_criteria,
    git_state,
    harness_void,
    holdout_cutoffs,
    holdout_touches,
    infra_kind,
    is_infra_block,
    iso,
    load_json,
    load_queue,
    paired_rule,
    protected_digest,
    read_canary,
    read_ledger,
    render_hypotheses,
    render_scoreboard,
    resolve_holdout_start,
    select_hypothesis,
    trial_counts,
    write_if_changed,
)
from rl_crypto.features import FEATURE_VERSION

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
    # A per-role exception is deliberate: the auditor gets a different model
    # family so a single model's blind spots cannot approve its own work.
    "models": {"auditor": "opencode/claude-haiku-4-5"},
    "budgets": {
        "maxConfigs": 8,
        "folds": 4,
        "roleTimeoutSeconds": 1800,
        "labTimeoutSeconds": 3600,
        "maxRolesPerRun": 6,
        "maxCostUsdPerRun": 5.0,
        "maxTokensPerRun": 2_000_000,
        "maxWallClockSeconds": 7200,
        # At the default family cap (10 trials) the smallest per-trial alpha is
        # 0.05/110; resolving its 2.27e-4 tail needs ~8,800 draws.
        "bootstrapSamples": 10000,
        "bootstrapBlockBars": 24,
        "maxTrialsPerFamily": 10,
    },
    "review": {"enabled": True, "require": False},
    "team": {"maxLeadTurns": 3},
    "promotion": {"apply": False, "requiresPromoteDryRun": True},
    "holdout": {"start": None},
    "noNewInformation": {"minNewBars": 168},
    "statistics": {"spendingSchedule": SPENDING_SCHEDULE},
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


# --------------------------------------------------------------------------- holdout freeze


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


def archive_timestamps(root: Path, ticker: str) -> list[str]:
    slug = str(ticker).replace("/", "-").replace(" ", "-").lower()
    path = root / "data" / slug / "candles.jsonl"
    if not path.exists():
        return []
    stamps: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                stamp = json.loads(line).get("timestamp")
            except json.JSONDecodeError:
                continue
            if stamp:
                stamps.append(str(stamp))
    return sorted(stamps)


def default_holdout_start(root: Path, tickers: Sequence[str]) -> str | None:
    """A timestamp at each ticker's 80th percentile, minimum across tickers.

    The minimum guarantees every scoped ticker keeps at least 20% of its bars
    for promotion. It is computed once and then frozen in the ledger, so later
    appends never move it.
    """
    candidates: list[str] = []
    for ticker in tickers:
        stamps = archive_timestamps(root, ticker)
        if not stamps:
            continue
        index = min(len(stamps) - 1, int(len(stamps) * 0.8))
        candidates.append(stamps[index])
    return min(candidates) if candidates else None


def resolve_or_compute_holdout(
    root: Path,
    config: Mapping[str, Any],
    hypothesis: Hypothesis,
    entries: Sequence[Mapping[str, Any]],
) -> str | None:
    existing = resolve_holdout_start(entries, hypothesis.stable_lineage)
    if existing:
        return existing
    override = os.environ.get("RL_MODEL_IMPROVE_HOLDOUT_START")
    if override:
        return override
    configured = (config.get("holdout") or {}).get("start")
    if configured:
        return str(configured)
    return default_holdout_start(root, hypothesis.experiment.get("tickers") or [])


# --------------------------------------------------------------------------- no-new-information


def previous_family_prereg(
    entries: Sequence[Mapping[str, Any]], family: str
) -> Mapping[str, Any] | None:
    """The last *consumed* trial in a family, ignoring infrastructure failures.

    An infra-blocked run is not a prior measurement, so it must not trigger the
    no-new-information guard against a retry.
    """
    consumed = consumed_runs(entries)
    infra_runs = {
        str(entry.get("run_id"))
        for entry in entries
        if entry.get("run_id") and is_infra_block(entry)
    }
    for entry in reversed(entries):
        if entry.get("kind") != "preregister" or entry.get("family") != family:
            continue
        run_id = str(entry.get("run_id") or "")
        if run_id and (run_id not in consumed or run_id in infra_runs):
            continue
        return entry
    return None


def no_new_information_reason(
    root: Path,
    hypothesis: Hypothesis,
    entries: Sequence[Mapping[str, Any]],
    *,
    min_new_bars: int,
) -> str | None:
    """Block an improvement rerun with the same data, config and feature set."""
    if (hypothesis.criteria or {}).get("kind") != "improvement":
        return None
    if hypothesis.canary:
        # A canary is a harness check, not a claim about new information; it
        # must be runnable on the same archive immediately after the other one.
        return None
    previous = previous_family_prereg(entries, hypothesis.family)
    if previous is None:
        return None
    changed: list[str] = []
    if previous.get("feature_version") and previous.get("feature_version") != FEATURE_VERSION:
        changed.append("feature_version")
    if previous.get("config_hash") != sha_of(hypothesis.experiment):
        changed.append("config")
    deltas: dict[str, int] = {}
    for ticker in hypothesis.experiment.get("tickers") or []:
        previous_tickers = (previous.get("data") or {}).get("tickers") or {}
        before = int((previous_tickers.get(ticker) or {}).get("rows") or 0)
        after = len(archive_timestamps(root, ticker))
        deltas[str(ticker)] = after - before
    if deltas and min(deltas.values()) >= int(min_new_bars):
        changed.append("bars")
    if changed:
        return None
    return (
        "no-new-information: same feature version, config and fewer than "
        f"{min_new_bars} new bars per ticker since the last {hypothesis.family} trial "
        f"({deltas or 'no data'})"
    )


def sha_of(value: Any) -> str:
    import hashlib

    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- budgets / protected


def usage_from_logs(ctx: Context) -> dict[str, Any]:
    tokens = 0
    cost = 0.0
    with_usage = 0
    logs = ctx.run_dir / "logs"
    if logs.exists():
        for path in sorted(logs.glob("*.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "step_finish":
                    continue
                part = event.get("part") or {}
                with_usage += 1
                tokens += int((part.get("tokens") or {}).get("total") or 0)
                cost += float(part.get("cost") or 0.0)
    return {
        "tokens": tokens,
        "costUsd": round(cost, 6),
        "stepFinishes": with_usage,
        "roles": len(ctx.roles),
        "teamSpawns": int((ctx.usage or {}).get("teamSpawns") or 0),
    }


def ceiling_reason(ctx: Context) -> str | None:
    budgets = ctx.config.get("budgets") or {}
    usage = usage_from_logs(ctx)
    ctx.usage = {**usage, **(ctx.usage or {})}
    elapsed = time.time() - ctx.started
    max_wall = float(budgets.get("maxWallClockSeconds", 7200))
    if elapsed > max_wall:
        return f"wall-clock ceiling exceeded: {elapsed:.0f}s > {max_wall:.0f}s"
    max_tokens = int(budgets.get("maxTokensPerRun", 2_000_000))
    if usage["tokens"] > max_tokens:
        return f"token ceiling exceeded: {usage['tokens']} > {max_tokens}"
    max_cost = float(budgets.get("maxCostUsdPerRun", 5.0))
    if usage["costUsd"] > max_cost:
        return f"cost ceiling exceeded: ${usage['costUsd']:.4f} > ${max_cost:.2f}"
    max_roles = int(budgets.get("maxRolesPerRun", 6))
    total_roles = usage["roles"] + usage["teamSpawns"]
    if total_roles > max_roles:
        return f"role ceiling exceeded: {total_roles} > {max_roles}"
    return None


def protected_changes(ctx: Context) -> list[str]:
    base = str(((ctx.prereg or {}).get("git") or {}).get("sha") or "")
    changed: set[str] = set()
    if base:
        result = _git(ctx.repo, "diff", "--name-only", f"{base}..HEAD")
        if result.returncode == 0:
            changed.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    status = _git(ctx.repo, "status", "--porcelain").stdout
    for line in status.splitlines():
        path = line[3:].strip().strip('"')
        if path:
            changed.add(path)
    return sorted(changed & set(PROTECTED_FILES))


def extract_team_events(stdout: str, path: Path) -> int:
    """Persist Ensemble tool events so a teammate's spawn can be audited later."""
    events: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part") or {}
        tool = str(part.get("tool") or "")
        if tool.startswith("team_"):
            events.append(json.dumps(event, sort_keys=True))
    path.write_text("\n".join(events) + ("\n" if events else ""), encoding="utf-8")
    return len(events)


def team_spawn_count(stdout: str) -> int:
    count = 0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part") or {}
        if str(part.get("tool") or "") == "team_spawn":
            count += 1
    return count


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
    schedule_armed = os.environ.get("RL_MODEL_IMPROVE_ENABLED", "").lower() in {
        "1",
        "true",
        "yes",
    }
    add(
        "schedule-forward-score",
        not (schedule_armed and not (root / "tools" / "forward_score.py").exists()),
        "the schedule is off; forward confirmation is unavailable"
        if not schedule_armed
        else "RL_MODEL_IMPROVE_ENABLED is set but tools/forward_score.py is missing "
        "— keep the weekly schedule off until it lands on main",
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
    #: Lab invocations that produced no usable payload, keyed by artifact
    #: label. A timeout is client-side and returns no output at all, so the
    #: caller needs the distinction to avoid treating a log file as a result.
    lab_failures: dict[str, str] = field(default_factory=dict)
    review: dict[str, Any] | None = None
    run_type: str = "experiment"
    lineage: str = ""
    holdout_start: str | None = None
    seed: int = 0
    started: float = field(default_factory=time.time)
    usage: dict[str, Any] = field(default_factory=dict)

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
    max_roles = int((ctx.config.get("budgets") or {}).get("maxRolesPerRun", 6))
    if len(ctx.roles) >= max_roles:
        record = {
            "role": role,
            "status": "failed",
            "reason": f"role ceiling reached ({len(ctx.roles)} >= {max_roles})",
            "turn": turn,
        }
        ctx.roles.append(record)
        _write_json(ctx.artifact("roles.json"), ctx.roles)
        return record
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
    if role == "lead":
        extract_team_events(stdout, ctx.artifact("team-events.jsonl"))
        ctx.usage["teamSpawns"] = max(
            int(ctx.usage.get("teamSpawns") or 0), team_spawn_count(stdout)
        )

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


def primary_config_name(experiment: Mapping[str, Any]) -> str:
    primary = experiment.get("primary")
    if isinstance(primary, str) and primary:
        return primary
    if isinstance(primary, Mapping):
        return (
            f"{primary.get('featureSet') or 'all'}+"
            f"{primary.get('modelSet') or 'reference'}+"
            f"h{int(primary.get('horizon') or 1)}"
        )
    if experiment.get("featureSet") or experiment.get("modelSet"):
        return (
            f"{experiment.get('featureSet') or 'all'}+"
            f"{experiment.get('modelSet') or 'reference'}+"
            f"h{int(experiment.get('horizon') or 1)}"
        )
    feature_sets = experiment.get("featureSets") or ["all"]
    model_sets = experiment.get("modelSets") or ["reference"]
    horizons = experiment.get("horizons") or [1]
    return f"{feature_sets[0]}+{model_sets[0]}+h{int(horizons[0])}"


def adjusted_alpha(criteria: Mapping[str, Any], trial_index: int) -> float:
    """The pre-hardening rule, kept only for criteria written before v2.

    ``alpha / trial_index`` sums to a divergent series, so a long family spends
    far more than its declared alpha. New criteria use :func:`alpha_for_trial`.
    """
    alpha = float(criteria.get("alpha", 0.05))
    trials = max(1, int(trial_index or 1))
    return alpha / trials


def trial_alpha(criteria: Mapping[str, Any], trial_index: int) -> float:
    """The per-trial alpha an entry is judged at, by its criteria version."""
    alpha = float(criteria.get("alpha", 0.05))
    if criteria_version(criteria) >= 2:
        return alpha_for_trial(alpha, trial_index)
    return adjusted_alpha(criteria, trial_index)


def family_cap(config: Mapping[str, Any]) -> int:
    return int(
        (config.get("budgets") or {}).get(
            "maxTrialsPerFamily", FAMILY_TRIAL_CAP_DEFAULT
        )
    )


def family_cap_reason(config: Mapping[str, Any], prereg: Mapping[str, Any]) -> str | None:
    """Why this preregistration exceeds its family's trial budget, or ``None``."""
    cap = family_cap(config)
    trial = int(prereg.get("trial_index") or 1)
    if trial > cap:
        family = prereg.get("family") or prereg.get("hypothesis_id") or "?"
        return (
            f"family {family!r} is at its trial cap ({cap}); retire it and open "
            "a new family with a fresh alpha budget"
        )
    return None


def _common_lab_args(ctx: Context, experiment: Mapping[str, Any]) -> list[str]:
    scope = ctx.config.get("scope") or {}
    budgets = ctx.config.get("budgets") or {}
    criteria = (ctx.prereg or {}).get("criteria") or {}
    trial_index = int((ctx.prereg or {}).get("trial_index") or 1)
    alpha = trial_alpha(criteria, trial_index) if criteria_version(criteria) >= 2 else 0.05
    args = [
        "--tickers",
        *[str(t) for t in experiment.get("tickers") or scope.get("tickers") or []],
        "--targets",
        *[str(t) for t in experiment.get("targets") or scope.get("targets") or []],
        "--primary-target",
        str(criteria.get("target") or "fwd_direction"),
        "--folds",
        str(int(experiment.get("folds") or budgets.get("folds", 4))),
        "--threshold",
        str(float(experiment.get("threshold") or scope.get("threshold", 0.55))),
        "--position",
        str(experiment.get("position") or scope.get("position", "long-only")),
        "--fee-bps",
        str(float(experiment.get("feeBps") or scope.get("feeBps", 10.0))),
        "--slippage-bps",
        str(float(experiment.get("slippageBps") or scope.get("slippageBps", 5.0))),
        "--seed",
        str(int(ctx.seed or ctx.config.get("seed") or 0)),
        "--alpha",
        str(alpha),
        "--bootstrap-samples",
        str(int(budgets.get("bootstrapSamples", 1000))),
        "--bootstrap-block",
        str(int(budgets.get("bootstrapBlockBars", 24))),
    ]
    if ctx.holdout_start:
        args.extend(["--holdout-start", str(ctx.holdout_start)])
    return args


def run_primary_evaluate(
    ctx: Context,
    prereg: Mapping[str, Any],
    *,
    controls: str = "both",
    full_archive: bool = False,
) -> dict[str, Any] | None:
    experiment = dict(prereg.get("experiment") or {})
    primary = primary_config_name(experiment)
    feature_set, model_set, horizon = primary.split("+")
    criteria = (ctx.prereg or {}).get("criteria") or {}
    control_shifts = criteria.get("control_shifts") or [
        169,
        173,
        181,
        191,
        337,
    ]
    args = [
        *_common_lab_args(ctx, experiment),
        "--feature-set",
        feature_set,
        "--model-set",
        model_set,
        "--horizon",
        horizon.removeprefix("h"),
        "--controls",
        controls,
        "--control-shifts",
        ",".join(str(int(value)) for value in control_shifts),
        "--control-replicates",
        str(int(criteria.get("control_replicates") or 6)),
        "--control-quantile",
        str(float(criteria.get("control_quantile") or 0.95)),
        "--baseline",
        str(criteria.get("baseline") or "strongest"),
    ]
    extra = [str(item) for item in experiment.get("extraLabArgs") or []]
    args.extend(extra)
    if full_archive:
        args.append("--full-archive")
    return run_lab(ctx, "evaluate", args, label="lab-evaluate")


def run_sweep(ctx: Context, prereg: Mapping[str, Any]) -> dict[str, Any] | None:
    experiment = dict(prereg.get("experiment") or {})
    prereg_criteria = prereg.get("criteria") or {}
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
        "--primary-config",
        primary_config_name(experiment),
        "--baseline",
        str((prereg_criteria or {}).get("baseline") or "strongest"),
    ]
    return run_lab(ctx, "sweep", args, label="lab-sweep")


def run_promote(
    ctx: Context,
    prereg: Mapping[str, Any],
    sweep: Mapping[str, Any],
    *,
    apply: bool,
) -> dict[str, Any] | None:
    candidates = list(sweep.get("primary_candidates") or sweep.get("candidates") or [])
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
        "--seed",
        str(int(ctx.seed or ctx.config.get("seed") or 0)),
    ]
    if ctx.holdout_start:
        args.extend(["--holdout-start", str(ctx.holdout_start)])
    if ticker:
        args.extend(["--ticker", str(ticker)])
    if apply:
        args.append("--apply")
    return run_lab(ctx, "promote", args, label="lab-promote-apply" if apply else "lab-promote")


def run_lab(
    ctx: Context, action: str, args: Sequence[str], *, label: str
) -> dict[str, Any] | None:
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
        ctx.lab_failures[label] = "timeout"
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
    if proc.returncode != 0 or not isinstance(payload, dict):
        ctx.lab_failures[label] = "timeout" if proc.returncode == 0 else "nonzero"
        return None
    ctx.lab_failures.pop(label, None)
    return payload


def _record_holdout(
    ctx: Context,
    prereg: Mapping[str, Any],
    *,
    mode: str,
    artifact: str | None,
    passed: bool,
) -> None:
    append_entry(
        ctx.ledger,
        {
            "kind": "holdout",
            "run_id": ctx.run_id,
            "hypothesis_id": prereg.get("hypothesis_id"),
            "lineage": prereg.get("lineage"),
            "holdout_touched": True,
            "mode": mode,
            "artifact": artifact,
            "passed": passed,
        },
    )


def record_verdict(
    ctx: Context,
    prereg: Mapping[str, Any],
    artifact: str | None,
    verdict: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a verdict to the ledger, or to the canary record when it is one.

    Canaries are harness checks, not trials: keeping them out of the ledger is
    what stops them from consuming family alpha or appearing in trial counts.
    The run's pre-registration and results stay in the ledger as provenance.
    """
    payload = {
        "kind": "verdict",
        "run_id": ctx.run_id,
        "hypothesis_id": prereg.get("hypothesis_id"),
        "title": prereg.get("title"),
        "family": prereg.get("family"),
        "lineage": prereg.get("lineage"),
        "trial_index": prereg.get("trial_index"),
        "criteria_version": criteria_version(prereg.get("criteria")),
        "run_type": ctx.run_type,
        "canary": bool(ctx.hypothesis and ctx.hypothesis.canary),
        "canary_kind": (ctx.hypothesis.canary_kind if ctx.hypothesis else "null"),
        "holdout_start": ctx.holdout_start,
        "outcome": verdict.get("outcome"),
        "passed": verdict.get("passed"),
        "reasons": verdict.get("reasons") or [],
        "metrics": verdict.get("metrics") or {},
        "prereg_id": prereg.get("id"),
        "protected_hash": (prereg.get("protected") or {}).get("hash"),
        "artifact": artifact,
    }
    if payload["canary"]:
        return append_canary(ctx.repo / "experiments" / CANARY_FILENAME, payload)
    return append_entry(ctx.ledger, payload)


def _evaluate_harness_change(ctx: Context, prereg: Mapping[str, Any]) -> dict[str, Any]:
    """A protected-file change is recorded, never scoreboard-eligible."""
    return {
        "passed": False,
        "outcome": "harness-change",
        "requires_approval": True,
        "reasons": [
            "harness-change may touch protected measurement files; "
            "human approval is required and no verdict is scored"
        ],
        "metrics": {
            "criteria_version": criteria_version(prereg.get("criteria")),
            "protected": prereg.get("protected") or {},
        },
    }


def _evaluate_mechanical_v2(
    ctx: Context, prereg: Mapping[str, Any]
) -> dict[str, Any]:
    """One primary config, pooled paired improvement, CI, null controls."""
    criteria = dict(prereg.get("criteria") or {})
    experiment = dict(prereg.get("experiment") or {})
    canary = bool(ctx.hypothesis and ctx.hypothesis.canary)
    canary_kind = ctx.hypothesis.canary_kind if ctx.hypothesis else "null"
    null_canary = canary and canary_kind == "null"
    if criteria_version(criteria) >= 2 and criteria.get("kind") != "diagnostic":
        budget = ctx.config.get("budgets") or {}
        resolution = bootstrap_resolution(
            float(criteria.get("alpha", 0.05)),
            family_cap(ctx.config),
            int(budget.get("bootstrapSamples", 1000)),
        )
        if not resolution["ok"]:
            reason = (
                "insufficient-resolution: "
                f"alpha={resolution['alpha_family']:g} at family cap "
                f"{resolution['family_cap']} leaves alpha_trial="
                f"{resolution['smallest_alpha']:.3g}; need "
                f"{resolution['required_samples']} bootstrap resamples, "
                f"configured {resolution['configured_samples']}"
            )
            append_entry(
                ctx.ledger,
                {
                    "kind": "blocked",
                    "run_id": ctx.run_id,
                    "hypothesis_id": prereg.get("hypothesis_id"),
                    "stage": "evaluation",
                    "reason": reason,
                    "infra": True,
                    "infra_kind": "insufficient-resolution",
                },
            )
            return {
                "passed": False,
                "outcome": "blocked",
                "reasons": [reason],
                "metrics": {"resolution": resolution},
                "infra": True,
                "infra_kind": "insufficient-resolution",
            }
    # The statistical rule is judged before the holdout is read; a dry run is
    # only attempted once the candidate has cleared it.
    stat_prereg = prereg
    if criteria.get("requires_promote"):
        stat_prereg = {
            **prereg,
            "criteria": {**criteria, "requires_promote": False},
        }
    sweep: dict[str, Any] | None = None
    if str(experiment.get("action") or "sweep") == "sweep":
        # Grid configs are diagnostics only; they never feed the verdict.
        sweep = run_sweep(ctx, prereg)
    primary = run_primary_evaluate(
        ctx,
        prereg,
        controls="shuffle" if null_canary else "both",
        full_archive=bool(experiment.get("fullArchive"))
        and criteria.get("kind") == "diagnostic",
    )
    produced_result = primary is not None
    if primary is None:
        failure = ctx.lab_failures.get("lab-evaluate", "no-payload")
        reason = (
            f"{failure}: no lab result was produced"
            if failure != "nonzero"
            else "lab exited nonzero: no lab result was produced"
        )
        verdict = {
            "passed": False,
            "outcome": "blocked",
            "reasons": [reason],
            "metrics": {},
            "infra": True,
            "infra_kind": "timeout" if failure == "timeout" else "evaluation",
        }
        primary = {}
    elif null_canary:
        control = (primary.get("controls") or {}).get("shuffle") or {}
        pooled = control.get("pooled") or {}
        if not pooled:
            verdict = {
                "passed": False,
                "outcome": "blocked",
                "reasons": ["canary shuffle control did not run"],
                "metrics": {},
            }
        else:
            primary = {**primary, "canary": True, "pooled": pooled}
            verdict = evaluate_criteria(stat_prereg, primary)
    else:
        verdict = evaluate_criteria(stat_prereg, primary)

    # A full-archive diagnostic sees the reserved tail; it is honest about that
    # and consumes the lineage's holdout budget, but it can never be promoted.
    if primary.get("diagnostic_full_archive"):
        _record_holdout(
            ctx,
            prereg,
            mode="diagnostic-full-archive",
            artifact=ctx.artifacts.get("lab-evaluate"),
            passed=False,
        )

    promote_payload: dict[str, Any] | None = None
    if (
        not canary
        and verdict.get("outcome") != "blocked"
        and criteria.get("requires_promote")
    ):
        lineage = str(prereg.get("lineage"))
        touches = holdout_touches(read_ledger(ctx.ledger))
        if not verdict.get("passed"):
            pass
        elif touches.get(lineage):
            verdict = {
                "passed": False,
                "outcome": "blocked",
                "reasons": [f"holdout already touched for lineage {lineage}"],
                "metrics": verdict.get("metrics") or {},
            }
        elif sweep is None:
            verdict = {
                "passed": False,
                "outcome": "blocked",
                "reasons": ["promotion requires a sweep candidate but the action was evaluate"],
                "metrics": verdict.get("metrics") or {},
            }
        else:
            promote_payload = run_promote(ctx, prereg, sweep, apply=False)
            _record_holdout(
                ctx,
                prereg,
                mode="dry-run",
                artifact=ctx.artifacts.get("lab-promote"),
                passed=bool((promote_payload or {}).get("passed")),
            )
            verdict = evaluate_criteria(
                prereg, primary, promote_payload=promote_payload
            )

    if canary and verdict.get("outcome") == "provisional":
        append_entry(
            ctx.ledger,
            {
                "kind": "harness_void",
                "run_id": ctx.run_id,
                "hypothesis_id": prereg.get("hypothesis_id"),
                "reason": (
                    "H0-canary produced provisional skill from shuffled labels; "
                    "the harness is void until a human clears it"
                ),
                "artifact": ctx.artifacts.get("lab-evaluate"),
            },
        )
        verdict = {
            "passed": False,
            "outcome": "invalid",
            "reasons": ["canary reached provisional: harness void"],
            "metrics": verdict.get("metrics") or {},
        }

    artifact = (
        ctx.artifacts.get("lab-evaluate") or ctx.artifacts.get("lab-sweep")
        if produced_result
        else None
    )
    if produced_result and artifact:
        append_entry(
            ctx.ledger,
            {
                "kind": "results",
                "run_id": ctx.run_id,
                "hypothesis_id": prereg.get("hypothesis_id"),
                "family": prereg.get("family"),
                "lineage": prereg.get("lineage"),
                "metrics": verdict.get("metrics") or {},
                "artifact": artifact,
                "promote_artifact": ctx.artifacts.get("lab-promote"),
                "protected_hash": (prereg.get("protected") or {}).get("hash"),
            },
        )

    if verdict.get("outcome") == "blocked":
        blocked = {
            "kind": "blocked",
            "run_id": ctx.run_id,
            "hypothesis_id": prereg.get("hypothesis_id"),
            "stage": "evaluation",
            "reason": "; ".join(verdict.get("reasons") or []),
            "artifact": artifact,
        }
        if verdict.get("infra"):
            blocked["infra"] = True
            blocked["infra_kind"] = verdict.get("infra_kind") or "evaluation"
        append_entry(ctx.ledger, blocked)
    else:
        record_verdict(ctx, prereg, artifact, verdict)
    if (
        verdict.get("passed")
        and ctx.config.get("promotion", {}).get("apply")
        and promote_payload is not None
        and sweep is not None
    ):
        applied = run_promote(ctx, prereg, sweep, apply=True)
        _record_holdout(
            ctx,
            prereg,
            mode="apply",
            artifact=ctx.artifacts.get("lab-promote-apply"),
            passed=bool((applied or {}).get("written")),
        )
        verdict.setdefault("metrics", {})["applied"] = (applied or {}).get("written")
    return verdict


def evaluate_mechanical(ctx: Context, prereg: Mapping[str, Any]) -> dict[str, Any]:
    """Run the pre-registered experiment and compute the verdict."""
    criteria = prereg.get("criteria") or {}
    if ctx.run_type == "harness-change":
        return _evaluate_harness_change(ctx, prereg)
    if criteria_version(criteria) >= 2:
        return _evaluate_mechanical_v2(ctx, prereg)

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
        blocked = {
            "kind": "blocked",
            "run_id": ctx.run_id,
            "hypothesis_id": prereg.get("hypothesis_id"),
            "stage": "evaluation",
            "reason": "; ".join(verdict.get("reasons") or []),
            "artifact": artifact,
            "detail": detail,
        }
        kind = infra_kind(" ".join(verdict.get("reasons") or []), "evaluation")
        if kind:
            blocked["infra"] = True
            blocked["infra_kind"] = kind
        append_entry(ctx.ledger, blocked)
    else:
        record_verdict(
            ctx,
            prereg,
            ctx.artifacts.get("lab-sweep") or ctx.artifacts.get("lab-evaluate"),
            verdict,
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
        verdict.get("outcome")
        in {
            "negative",
            "provisional",
            "confirmed",
            "refuted",
            "invalid",
            "diagnostic",
            "blocked",
            "harness-change",
        },
        f"experiment outcome: {verdict.get('outcome')}",
    )
    add(
        "run-type",
        ctx.run_type in {"experiment", "harness-change"},
        f"run type: {ctx.run_type}",
        severity="info",
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
    changed = {
        line[3:].strip().strip('"')
        for line in status_text.splitlines()
        if line.strip()
    }
    base = str((prereg.get("git") or {}).get("sha") or "")
    if base:
        diff = _git(ctx.repo, "diff", "--name-only", f"{base}..HEAD")
        if diff.returncode == 0:
            changed.update(line.strip() for line in diff.stdout.splitlines() if line.strip())
    changed = sorted(changed)
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
    protected = protected_changes(ctx)
    may_touch = ctx.run_type == "harness-change"
    add(
        "protected-files",
        not protected or may_touch,
        f"protected files changed: {', '.join(protected)}"
        if protected and not may_touch
        else (
            "harness-change may touch protected files"
            if protected
            else "protected files untouched"
        ),
    )
    recorded_hash = str((prereg.get("protected") or {}).get("hash") or "")
    current_hash = str(protected_digest(ctx.repo).get("hash") or "")
    add(
        "protected-hash",
        not recorded_hash or recorded_hash == current_hash or may_touch,
        "measurement contract hash matches the pre-registration"
        if recorded_hash == current_hash
        else "measurement contract changed after pre-registration",
    )
    add(
        "dirty",
        True,
        f"{len(changed)} uncommitted path(s) to commit",
        severity="info",
    )
    add(
        "usage",
        True,
        f"{ctx.usage.get('tokens', 0)} tokens, ${ctx.usage.get('costUsd', 0)}",
        severity="info",
    )
    ceiling = ceiling_reason(ctx)
    add("ceilings", ceiling is None, ceiling or "within the configured ceilings")
    if may_touch:
        add(
            "human-approval",
            False,
            "harness-change requires human approval before merge",
            severity="warn",
        )

    if criteria_version(prereg.get("criteria")) >= 2:
        add(
            "holdout-freeze",
            bool(ctx.holdout_start)
            and resolve_holdout_start(entries, str(prereg.get("lineage")))
            == ctx.holdout_start,
            f"holdout_start: {ctx.holdout_start or 'missing'}",
        )

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
        f"- **Run type:** `{ctx.run_type}` (lineage `{ctx.lineage}`)",
        f"- **Criteria:** v{criteria_version(prereg.get('criteria'))}"
        f", alpha {((prereg.get('criteria') or {}).get('alpha', 'n/a'))}",
        f"- **Holdout cutoff:** `{ctx.holdout_start or 'none'}`"
        " (frozen timestamp; appends never move it)",
        f"- **Seed:** `{ctx.seed}`",
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
    if ctx.run_type == "harness-change":
        lines.extend(
            [
                "",
                "## Human approval required",
                "",
                "- This run may touch the protected measurement contract. It is "
                "**not** scoreboard-eligible; a human must review the diff and the "
                "mechanical review before merge.",
            ]
        )
    pooled = metrics.get("pooled") or {}
    if pooled:
        lines.extend(
            [
                "",
                "## V2 statistics (pooled paired improvement)",
                "",
                f"- mean: `{pooled.get('mean_improvement')}`",
                f"- CI: `[{pooled.get('ci_low')}, {pooled.get('ci_high')}]` "
                f"(alpha `{pooled.get('alpha')}`, seed `{pooled.get('seed')}`)",
                f"- n: `{pooled.get('n')}` in `{pooled.get('blocks')}` time blocks "
                f"of `{pooled.get('block_bars')}` bars",
                f"- fold improvements: `{pooled.get('fold_improvements')}`",
            ]
        )
    controls = metrics.get("controls") or {}
    for name, block in controls.items():
        control = (block or {}).get("pooled") or {}
        lines.append(
            f"- {name} control: mean `{control.get('mean_improvement')}` "
            f"CI `[{control.get('ci_low')}, {control.get('ci_high')}]`"
            f" (passes={block.get('passed')})"
        )
    if metrics.get("carried_by_one_ticker"):
        lines.append("- **flag:** the result is carried by a single ticker.")
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
    canaries = read_canary(ctx.repo / "experiments" / CANARY_FILENAME)
    write_if_changed(
        ctx.repo / "experiments" / "SCOREBOARD.md",
        render_scoreboard(entries, canary_entries=canaries),
    )
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
    holdout_start = resolve_or_compute_holdout(root, config, hypothesis, entries)
    if criteria_version(hypothesis.criteria) >= 2 and not holdout_start:
        print(
            "cannot freeze a holdout cutoff: no archive timestamps for the scoped tickers",
            file=sys.stderr,
        )
        return 2
    prereg = build_preregistration(
        hypothesis,
        run_id=run_id,
        root=root,
        entries=entries,
        trigger=args.trigger,
        seed=int(config.get("seed") or 0),
        lineage=hypothesis.stable_lineage,
        holdout_start=holdout_start,
        run_type=hypothesis.run_type,
        feature_version=FEATURE_VERSION,
        protected=protected_digest(root),
    )
    cap_reason = family_cap_reason(config, prereg)
    if cap_reason:
        print(cap_reason, file=sys.stderr)
        return 2
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
        "holdout_cutoffs": holdout_cutoffs(entries),
        "harness_void": harness_void(entries),
        "canary": canary_status(
            entries,
            canary_entries=read_canary(root / "experiments" / CANARY_FILENAME),
        ),
        "scoreboard": render_scoreboard(
            entries,
            canary_entries=read_canary(root / "experiments" / CANARY_FILENAME),
        ),
    }
    if args.render:
        canaries = read_canary(root / "experiments" / CANARY_FILENAME)
        write_if_changed(
            root / "experiments" / "SCOREBOARD.md",
            render_scoreboard(entries, canary_entries=canaries),
        )
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


def cmd_holdout_rotate(args: argparse.Namespace) -> int:
    root = repo_root()
    ledger = root / "experiments" / "ledger.jsonl"
    entries = read_ledger(ledger)
    previous = resolve_holdout_start(entries, args.lineage)
    if previous is None:
        raise SystemExit(f"no frozen holdout for lineage {args.lineage!r}")
    previous_stamp = _parse_ts(previous)
    new_stamp = _parse_ts(args.holdout_start)
    if previous_stamp is None or new_stamp is None:
        raise SystemExit("holdout-start must be an ISO timestamp")
    if new_stamp <= previous_stamp:
        raise SystemExit("holdout-rotate may only move the cutoff forward")
    entry = append_entry(
        ledger,
        {
            "kind": "holdout_rotate",
            "lineage": args.lineage,
            "holdout_start": args.holdout_start,
            "rotated_from": previous,
            "reason": args.reason,
            "approved_by": "human",
        },
    )
    payload = {"rotated": True, "entry": entry}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"rotated {args.lineage}: {previous} -> {args.holdout_start}")
    return 0


def cmd_harness_clear(args: argparse.Namespace) -> int:
    root = repo_root()
    ledger = root / "experiments" / "ledger.jsonl"
    entries = read_ledger(ledger)
    if harness_void(entries) is None:
        print("no harness_void marker to clear")
        return 0
    entry = append_entry(
        ledger,
        {
            "kind": "harness_clear",
            "reason": args.reason,
            "approved_by": "human",
        },
    )
    payload = {"cleared": True, "entry": entry}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("harness void cleared")
    return 0


# --------------------------------------------------------------------------- calibration


_MODEL_LAB_CACHE: dict[str, Any] = {}


def load_model_lab(root: Path) -> Any:
    """Load ``tools/model_lab.py`` with its repo root anchored to ``root``.

    Calibration is offline and must not import the lab's ``ROOT`` from a
    different checkout. The module is cached under a private name so a test
    fixture and the real repo cannot shadow each other.
    """
    import importlib.util

    cache_key = str(root.resolve())
    cached = _MODEL_LAB_CACHE.get(cache_key)
    if cached is not None:
        return cached
    name = "rl_crypto_model_lab_calibration"
    path = root / "tools" / "model_lab.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - impossible
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = os.environ.get("RL_CRYPTO_ROOT")
    os.environ["RL_CRYPTO_ROOT"] = str(root)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop("RL_CRYPTO_ROOT", None)
        else:
            os.environ["RL_CRYPTO_ROOT"] = previous
    setattr(module, "ROOT", root)  # noqa: B010 - script module, not a package
    _MODEL_LAB_CACHE[cache_key] = module
    return module


def calibration_hypothesis(queue: Sequence[Hypothesis]) -> Hypothesis | None:
    for hypothesis in queue:
        if hypothesis.canary and hypothesis.canary_kind == "null":
            return hypothesis
    for hypothesis in queue:
        if hypothesis.canary:
            return hypothesis
    return None


def calibration_cutoff(
    root: Path,
    config: Mapping[str, Any],
    hypothesis: Hypothesis,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str]:
    existing = resolve_holdout_start(entries, hypothesis.stable_lineage)
    if existing:
        return existing, "ledger"
    override = os.environ.get("RL_MODEL_IMPROVE_HOLDOUT_START")
    if override:
        return override, "environment"
    configured = (config.get("holdout") or {}).get("start")
    if configured:
        return str(configured), "config"
    return default_holdout_start(root, hypothesis.experiment.get("tickers") or []), "computed"


def _null_seed_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One independent null draw (shuffled labels) and its per-trial verdicts."""
    root = Path(str(payload["root"]))
    lab = load_model_lab(root)
    experiment = payload["experiment"]
    config = lab.Config(
        name=payload["config_name"],
        feature_names=lab.resolve_feature_names(str(experiment.get("featureSet") or "all")),
        model_names=lab.resolve_model_set(str(experiment.get("modelSet") or "reference")),
        horizon=int(experiment.get("horizon") or 1),
    )
    seed = int(payload["seed"])
    target = str(payload["target"])
    folds = int(payload["folds"])
    per_ticker = {
        ticker: lab.evaluate_control_ticker(
            ticker,
            config,
            target,
            "shuffle",
            folds=folds,
            holdout_start=payload.get("holdout_start"),
            seed=seed,
            shift_bars=int(payload.get("shift_bars") or 169),
        )
        for ticker in payload["tickers"]
    }
    rows_by_ticker = {
        ticker: list(row.get("paired_rows") or [])
        for ticker, row in per_ticker.items()
    }
    folds_by_ticker = {
        ticker: list(row.get("fold_improvements") or [])
        for ticker, row in per_ticker.items()
    }
    outcomes: dict[str, Any] = {}
    for trial, alpha in payload["alphas"].items():
        pooled = lab.bootstrap_pooled(
            rows_by_ticker,
            alpha=float(alpha),
            seed=seed,
            block_bars=int(payload["block_bars"]),
            bootstrap_samples=int(payload["bootstrap_samples"]),
        )
        pooled["fold_improvements"] = lab.pool_fold_improvements(
            folds_by_ticker, folds
        )
        pooled["target"] = target
        resolved = pooled.get("resolution_ok") is True
        passed, reasons = paired_rule(
            pooled, alpha=float(alpha), fold_rule=str(payload["fold_rule"])
        )
        outcomes[str(trial)] = {
            "passed": bool(passed and resolved),
            "resolution_ok": resolved,
            "reasons": reasons,
            "mean_improvement": pooled.get("mean_improvement"),
            "ci_low": pooled.get("ci_low"),
            "ci_high": pooled.get("ci_high"),
            "blocks": pooled.get("blocks"),
        }
    return {"seed": seed, "trials": outcomes}


def calibrate_null(
    root: Path,
    *,
    config: Mapping[str, Any],
    queue: Sequence[Hypothesis],
    entries: Sequence[Mapping[str, Any]],
    seeds: int,
    trials: Sequence[int] = (1, 5, 10),
    alpha_family: float = 0.05,
    jobs: int = 1,
    cutoff: str | None = None,
) -> dict[str, Any]:
    """Fitting-noise canary: the v2 rule under shuffled labels.

    This checks that the *whole fitting pipeline* cannot manufacture skill from
    permuted labels. It is deliberately conservative and is **not** the
    calibration of the CI rule: shuffling destroys the serial correlation that
    makes the real paired-difference series hard, so its false-positive rate is
    a lower bound. Use ``calibrate-coverage`` for the CI rule itself.
    """
    hypothesis = calibration_hypothesis(queue)
    if hypothesis is None:
        raise SystemExit("no canary hypothesis in experiments/hypotheses.json")
    experiment = dict(hypothesis.experiment or {})
    tickers = [str(ticker) for ticker in experiment.get("tickers") or []]
    if not tickers:
        raise SystemExit("the null canary hypothesis has no tickers")
    computed_cutoff, cutoff_source = calibration_cutoff(
        root, config, hypothesis, entries
    )
    cutoff = cutoff or computed_cutoff
    if cutoff and cutoff != computed_cutoff:
        cutoff_source = "cli"
    target = str(
        (hypothesis.criteria or {}).get("target")
        or experiment.get("targets", ["fwd_direction"])[0]
    )
    budgets = config.get("budgets") or {}
    alphas = {
        int(trial): alpha_for_trial(alpha_family, int(trial))
        for trial in sorted({int(t) for t in trials})
    }
    payload_base = {
        "root": str(root),
        "experiment": experiment,
        "config_name": primary_config_name(experiment),
        "target": target,
        "folds": int(experiment.get("folds") or budgets.get("folds", 4)),
        "tickers": tickers,
        "holdout_start": cutoff,
        "shift_bars": int(experiment.get("shiftBars") or 169),
        "block_bars": int(budgets.get("bootstrapBlockBars", 24)),
        "bootstrap_samples": int(budgets.get("bootstrapSamples", 10000)),
        "fold_rule": str((hypothesis.criteria or {}).get("fold_rule", "all_positive")),
        "alphas": {str(trial): alpha for trial, alpha in alphas.items()},
    }
    jobs = max(1, min(int(jobs), int(seeds)))
    draws: list[dict[str, Any]] = []
    if jobs == 1:
        draws = [
            _null_seed_job({**payload_base, "seed": seed})
            for seed in range(int(seeds))
        ]
    else:  # pragma: no cover - exercised by the real calibration run
        import concurrent.futures
        import multiprocessing

        context = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs, mp_context=context
        ) as pool:
            draws = list(
                pool.map(
                    _null_seed_job,
                    [{**payload_base, "seed": seed} for seed in range(int(seeds))],
                )
            )
    table: dict[str, Any] = {}
    failure = False
    insufficient = 0
    for trial, alpha in alphas.items():
        key = str(trial)
        passed = sum(1 for draw in draws if draw["trials"][key]["passed"])
        insufficient += sum(
            1 for draw in draws if not draw["trials"][key]["resolution_ok"]
        )
        verdict = calibration_verdict(passed, len(draws), alpha)
        verdict["trial"] = trial
        table[key] = verdict
        failure = failure or bool(verdict["exceeds"])
    protected = protected_digest(root)
    return {
        "action": "calibrate-null",
        "seeds": int(seeds),
        "target": target,
        "tickers": tickers,
        "frozen_cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "alpha_family": float(alpha_family),
        "spending_schedule": SPENDING_SCHEDULE,
        "bootstrap_samples": payload_base["bootstrap_samples"],
        "bootstrap_blocks": payload_base["block_bars"],
        "protected_hash": protected["hash"],
        "date": iso(),
        "trials": table,
        "insufficient_resolution_draws": insufficient,
        "calibration_failure": bool(failure),
        "draws": draws,
    }


def _calibration_dir(args: argparse.Namespace, action: str) -> Path:
    if getattr(args, "run_dir", None):
        base = Path(args.run_dir).expanduser().resolve()
    else:
        base = default_run_root()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = base / f"{stamp}-{action}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_calibrate_null(args: argparse.Namespace) -> int:
    root = repo_root()
    config = load_config(root)
    queue = load_queue(root / "experiments" / "hypotheses.json")
    entries = read_ledger(root / "experiments" / "ledger.jsonl")
    payload = calibrate_null(
        root,
        config=config,
        queue=queue,
        entries=entries,
        seeds=int(args.seeds),
        trials=[int(t) for t in str(args.trials).replace(",", " ").split() if t.strip()],
        alpha_family=float(args.alpha),
        jobs=int(args.jobs),
        cutoff=args.cutoff,
    )
    run_dir = _calibration_dir(args, "calibrate-null")
    payload["run_dir"] = str(run_dir)
    _write_json(run_dir / "calibrate-null.json", payload)
    lines = [
        "# Null calibration",
        "",
        f"- seeds: {payload['seeds']}",
        f"- target: {payload['target']}",
        f"- tickers: {', '.join(payload['tickers'])}",
        f"- frozen cutoff: {payload['frozen_cutoff']} ({payload['cutoff_source']})",
        f"- protected hash: {payload['protected_hash']}",
        f"- date: {payload['date']}",
        "",
        "| trial | nominal alpha | empirical rate | 95% CI | exceeds |",
        "| ----- | ------------- | -------------- | ------ | ------- |",
    ]
    for key, row in payload["trials"].items():
        lines.append(
            f"| {key} | {row['nominal_alpha']:.6f} | {row['rate']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{'YES' if row['exceeds'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        "Calibration: "
        + ("FAILED" if payload["calibration_failure"] else "ok")
        + f" (insufficient-resolution draws: {payload['insufficient_resolution_draws']})"
    )
    (run_dir / "calibrate-null.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["markdown"] = str(run_dir / "calibrate-null.md")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if not payload["calibration_failure"] else 1
    print("\n".join(lines))
    print(f"\nartifact: {run_dir}")
    return 0 if not payload["calibration_failure"] else 1


def target_metric_space(lab: Any, target: str) -> str:
    """Metric space the reference model minimises for ``target``."""
    name = str((lab.MODEL_SETS.get("reference") or {}).get(target) or "ridge")
    return "log" if name == "log_ridge" else "raw"


def label_variance(
    lab: Any,
    root: Path,
    tickers: Sequence[str],
    target: str,
    *,
    cutoff: str | None,
    horizon: int,
    space: str,
) -> float:
    """Variance of the label in the metric space, across the scoped archive."""
    import numpy as np

    values: list[float] = []
    for ticker in tickers:
        rows, _ = lab.filter_rows_before(lab.load_ticker_rows(ticker), cutoff)
        samples = lab.build_samples(
            rows, horizon=horizon, interval_seconds=lab.INTERVAL_SECONDS
        )
        for sample in samples:
            value = sample.targets.get(target)
            if value is None:
                continue
            number = float(value)
            if space == "log":
                if number <= 0.0:
                    continue
                values.append(math.log(number))
            else:
                values.append(number)
    return float(np.var(values)) if len(values) > 1 else 0.0


def _power_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One planted-signal repetition: candidate plus optional null controls."""
    root = Path(str(payload["root"]))
    lab = load_model_lab(root)
    experiment = payload["experiment"]
    plant_feature = str(payload["plant_feature"])
    extra = () if float(payload.get("strength") or 0.0) == 0.0 else (plant_feature,)
    feature_names = tuple(
        dict.fromkeys(
            (
                *lab.resolve_feature_names(str(experiment.get("featureSet") or "all")),
                *extra,
            )
        )
    )
    config = lab.Config(
        name=payload["config_name"],
        feature_names=feature_names,
        model_names=lab.resolve_model_set(str(experiment.get("modelSet") or "reference")),
        horizon=int(experiment.get("horizon") or 1),
    )
    seed = int(payload["seed"])
    target = str(payload["target"])
    folds = int(payload["folds"])
    strength = float(payload["strength"])
    space = str(payload["space"])
    alphas = {str(trial): float(alpha) for trial, alpha in payload["alphas"].items()}
    fold_rule = str(payload["fold_rule"])
    economics = target == "fwd_direction"

    def run(mode: str | None) -> dict[str, Any]:
        per_ticker = {
            ticker: lab.evaluate_control_ticker(
                ticker,
                config,
                target,
                mode,
                folds=folds,
                holdout_start=payload.get("holdout_start"),
                seed=seed,
                shift_bars=int(payload.get("shift_bars") or 169),
                plant_strength=strength,
                plant_space=space,
                plant_feature=plant_feature,
                structural=False,
                economics=economics and mode is None,
                threshold=float(payload.get("threshold") or 0.55),
                position=str(payload.get("position") or "long-only"),
                fee_bps=float(payload.get("fee_bps") or 10.0),
                slippage_bps=float(payload.get("slippage_bps") or 5.0),
            )
            for ticker in payload["tickers"]
        }
        rows_by_ticker = {
            ticker: list(row.get("paired_rows") or [])
            for ticker, row in per_ticker.items()
        }
        folds_by_ticker = {
            ticker: list(row.get("fold_improvements") or [])
            for ticker, row in per_ticker.items()
        }
        pooled_by_trial: dict[str, Any] = {}
        for trial, alpha in alphas.items():
            pooled = lab.bootstrap_pooled(
                rows_by_ticker,
                alpha=alpha,
                seed=seed,
                block_bars=int(payload["block_bars"]),
                bootstrap_samples=int(payload["bootstrap_samples"]),
            )
            pooled["fold_improvements"] = lab.pool_fold_improvements(
                folds_by_ticker, folds
            )
            pooled["target"] = target
            pooled_by_trial[trial] = pooled
        means = {
            str(ticker): float(
                sum(float(item["improvement"]) for item in row["paired_rows"])
                / len(row["paired_rows"])
            )
            if row.get("paired_rows")
            else 0.0
            for ticker, row in per_ticker.items()
        }
        economics_rows = {
            ticker: row.get("economics")
            for ticker, row in per_ticker.items()
            if row.get("economics")
        }
        return {
            "pooled_by_trial": pooled_by_trial,
            "per_ticker_means": means,
            "economics": economics_rows,
        }

    candidate = run(None)
    control_runs = {str(mode): run(str(mode)) for mode in payload.get("controls") or []}
    trials: dict[str, Any] = {}
    for trial, alpha in alphas.items():
        pooled = candidate["pooled_by_trial"][trial]
        resolved = pooled.get("resolution_ok") is True
        passed, reasons = paired_rule(pooled, alpha=alpha, fold_rule=fold_rule)
        positive = [
            name for name, mean in candidate["per_ticker_means"].items() if mean > 0.0
        ]
        if payload.get("require_multiple_tickers", True) and len(
            candidate["per_ticker_means"]
        ) > 1 and len(positive) < 2:
            passed = False
            reasons.append("the planted result is carried by fewer than two tickers")
        control_detail: dict[str, Any] = {}
        for mode, control in control_runs.items():
            control_pooled = control["pooled_by_trial"][trial]
            control_passed, _ = paired_rule(
                control_pooled, alpha=alpha, fold_rule=fold_rule
            )
            control_detail[mode] = {
                "passed": control_passed,
                "mean_improvement": control_pooled.get("mean_improvement"),
            }
            if control_passed:
                passed = False
                reasons.append(f"{mode} control also passed")
        economics_detail: dict[str, Any] | None = None
        economics_ok = True
        if economics:
            blocks = [
                block for block in (candidate.get("economics") or {}).values() if block
            ]
            if blocks:
                mean_return = sum(
                    float((block.get("model") or {}).get("total_return") or 0.0)
                    for block in blocks
                ) / len(blocks)
                mean_sharpe = sum(
                    float((block.get("model") or {}).get("sharpe") or 0.0)
                    for block in blocks
                ) / len(blocks)
                mean_momentum = sum(
                    float((block.get("momentum") or {}).get("sharpe") or 0.0)
                    for block in blocks
                ) / len(blocks)
                economics_ok = mean_return > 0.0 and mean_sharpe > mean_momentum
                economics_detail = {
                    "mean_total_return": mean_return,
                    "mean_sharpe": mean_sharpe,
                    "mean_momentum_sharpe": mean_momentum,
                    "passed": economics_ok,
                }
            else:
                economics_ok = False
                economics_detail = {"passed": False, "reason": "no economics block"}
            if not economics_ok:
                reasons.append("after-cost economics gate failed")
        trials[trial] = {
            "passed": bool(passed and resolved and economics_ok),
            "statistical_passed": bool(passed and resolved),
            "resolution_ok": resolved,
            "mean_improvement": pooled.get("mean_improvement"),
            "ci_low": pooled.get("ci_low"),
            "reasons": reasons,
            "per_ticker_positive": len(positive),
            "controls": control_detail,
            "economics": economics_detail,
        }
    return {
        "seed": seed,
        "strength": strength,
        "effect": float(payload["effect"]),
        "target": target,
        "classification_economics": economics,
        "trials": trials,
    }


def _ledger_relative_improvement(
    entries: Sequence[Mapping[str, Any]], target: str
) -> float | None:
    values: list[float] = []
    for entry in entries:
        if entry.get("kind") != "results":
            continue
        metrics = entry.get("metrics") or {}
        for ticker_stats in (metrics.get("per_ticker") or {}).values():
            stats = (ticker_stats.get("targets") or {}).get(target)
            if isinstance(stats, Mapping) and isinstance(
                stats.get("mean_improvement"), int | float
            ):
                values.append(float(stats["mean_improvement"]))
    return sum(values) / len(values) if values else None


def calibrate_power(
    root: Path,
    *,
    config: Mapping[str, Any],
    queue: Sequence[Hypothesis],
    entries: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    reps: int = 20,
    effect_fractions: Sequence[float] = (0.001, 0.003, 0.01, 0.03, 0.1),
    trials: Sequence[int] = (1, 5, 10),
    alpha_family: float = 0.05,
    power: float = 0.8,
    jobs: int = 1,
    cutoff: str | None = None,
    controls: Sequence[str] = ("shuffle", "shift"),
) -> dict[str, Any]:
    """Minimum planted improvement that clears the v2 rule with the target power.

    The signal is planted into the labels in the metric space the reference
    model minimises (raw for ``fwd_volume_ratio``, log for ``fwd_volatility``)
    and exposed as an independent feature. Its oracle paired improvement is
    ``effect``, so the reported MDE is directly comparable with the pooled
    paired improvements the lab reports.
    """
    lab = load_model_lab(root)
    hypothesis = calibration_hypothesis(queue)
    if hypothesis is None:
        raise SystemExit("no canary hypothesis in experiments/hypotheses.json")
    experiment = dict(hypothesis.experiment or {})
    tickers = [str(ticker) for ticker in experiment.get("tickers") or []]
    computed_cutoff, cutoff_source = calibration_cutoff(
        root, config, hypothesis, entries
    )
    cutoff = cutoff or computed_cutoff
    if cutoff and cutoff != computed_cutoff:
        cutoff_source = "cli"
    budgets = config.get("budgets") or {}
    alphas = {
        int(trial): alpha_for_trial(alpha_family, int(trial))
        for trial in sorted({int(t) for t in trials})
    }
    fold_rule = str((hypothesis.criteria or {}).get("fold_rule", "all_positive"))
    report: dict[str, Any] = {}
    for target in targets:
        space = target_metric_space(lab, target)
        variance = label_variance(
            lab,
            root,
            tickers,
            target,
            cutoff=cutoff,
            horizon=int(experiment.get("horizon") or 1),
            space=space,
        )
        if variance <= 0.0:
            report[target] = {"error": "no usable labels / zero variance"}
            continue
        effects = sorted(
            max(0.0, float(fraction) * variance) for fraction in effect_fractions
        )
        base = {
            "root": str(root),
            "experiment": experiment,
            "config_name": primary_config_name(experiment),
            "plant_feature": lab.PLANTED_FEATURE,
            "target": target,
            "space": space,
            "folds": int(experiment.get("folds") or budgets.get("folds", 4)),
            "tickers": tickers,
            "holdout_start": cutoff,
            "shift_bars": int(experiment.get("shiftBars") or 169),
            "block_bars": int(budgets.get("bootstrapBlockBars", 24)),
            "bootstrap_samples": int(budgets.get("bootstrapSamples", 10000)),
            "fold_rule": fold_rule,
            "require_multiple_tickers": bool(
                (hypothesis.criteria or {}).get("require_multiple_tickers", True)
            ),
            "controls": list(controls),
            "threshold": float(experiment.get("threshold") or 0.55),
            "position": str(experiment.get("position") or "long-only"),
            "fee_bps": float(experiment.get("feeBps") or 10.0),
            "slippage_bps": float(experiment.get("slippageBps") or 5.0),
        }
        alpha_payload = {str(trial): alpha for trial, alpha in alphas.items()}
        # The observed comparison is the unplanted pooled paired improvement on
        # the same archive and cutoff, in the same metric units as the MDE.
        observed_runs = [
            _power_job(
                {
                    **base,
                    "seed": seed,
                    "strength": 0.0,
                    "effect": 0.0,
                    "alphas": alpha_payload,
                }
            )
            for seed in range(max(1, min(3, reps)))
        ]
        observed = [
            float(run["trials"][str(min(alphas))]["mean_improvement"])
            for run in observed_runs
            if isinstance(
                (run.get("trials") or {}).get(str(min(alphas)), {}).get(
                    "mean_improvement"
                ),
                int | float,
            )
        ]
        observed_mean = sum(observed) / len(observed) if observed else None
        rows: list[dict[str, Any]] = []
        for effect in effects:
            strength = math.sqrt(effect)
            jobs_list = [
                {
                    **base,
                    "seed": seed,
                    "strength": strength,
                    "effect": effect,
                    "alphas": alpha_payload,
                }
                for seed in range(int(reps))
            ]
            if jobs > 1:  # pragma: no cover - real calibration path
                import concurrent.futures
                import multiprocessing

                context = multiprocessing.get_context("fork")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max(1, min(int(jobs), len(jobs_list))),
                    mp_context=context,
                ) as pool:
                    draws = list(pool.map(_power_job, jobs_list))
            else:
                draws = [_power_job(job) for job in jobs_list]
            row: dict[str, Any] = {
                "effect": effect,
                "strength": strength,
                "passes": {
                    str(trial): sum(
                        1 for draw in draws if draw["trials"][str(trial)]["passed"]
                    )
                    for trial in alphas
                },
                "achieved_mean": None,
                "resolution_failures": {
                    str(trial): sum(
                        1
                        for draw in draws
                        if not draw["trials"][str(trial)]["resolution_ok"]
                    )
                    for trial in alphas
                },
            }
            achieved = [
                float(draw["trials"][str(min(alphas))]["mean_improvement"])
                for draw in draws
                if isinstance(
                    draw["trials"][str(min(alphas))].get("mean_improvement"),
                    int | float,
                )
            ]
            row["achieved_mean"] = sum(achieved) / len(achieved) if achieved else None
            rows.append(row)
        mde: dict[str, Any] = {}
        for trial in alphas:
            key = str(trial)
            hit = next(
                (
                    row
                    for row in rows
                    if int(reps) and row["passes"][key] / int(reps) >= power
                ),
                None,
            )
            mde[key] = {
                "effect": None if hit is None else hit["effect"],
                "strength": None if hit is None else hit["strength"],
                "pass_rate": None if hit is None else hit["passes"][key] / int(reps),
                "achieved_mean": None if hit is None else hit["achieved_mean"],
                "detected": hit is not None,
            }
        report[target] = {
            "space": space,
            "label_variance": variance,
            "observed_mean_improvement": observed_mean,
            "observed_ledger_relative": _ledger_relative_improvement(entries, target),
            "effect_fractions": list(effect_fractions),
            "rows": rows,
            "mde": mde,
        }
    protected = protected_digest(root)
    return {
        "action": "calibrate-power",
        "targets": list(targets),
        "reps": int(reps),
        "power": float(power),
        "trials": {str(trial): alpha for trial, alpha in alphas.items()},
        "frozen_cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "alpha_family": float(alpha_family),
        "protected_hash": protected["hash"],
        "date": iso(),
        "results": report,
    }


def cmd_calibrate_power(args: argparse.Namespace) -> int:
    root = repo_root()
    config = load_config(root)
    queue = load_queue(root / "experiments" / "hypotheses.json")
    entries = read_ledger(root / "experiments" / "ledger.jsonl")
    targets = [
        part for part in str(args.targets).replace(",", " ").split() if part.strip()
    ]
    fractions = [
        float(part)
        for part in str(args.effect_fractions).replace(",", " ").split()
        if part.strip()
    ]
    payload = calibrate_power(
        root,
        config=config,
        queue=queue,
        entries=entries,
        targets=targets,
        reps=int(args.reps),
        effect_fractions=fractions,
        trials=[int(t) for t in str(args.trials).replace(",", " ").split() if t.strip()],
        alpha_family=float(args.alpha),
        power=float(args.power),
        jobs=int(args.jobs),
        cutoff=args.cutoff,
        controls=[part for part in str(args.controls).split(",") if part and part != "none"],
    )
    run_dir = _calibration_dir(args, "calibrate-power")
    payload["run_dir"] = str(run_dir)
    _write_json(run_dir / "calibrate-power.json", payload)
    lines = [
        "# Power calibration",
        "",
        f"- reps: {payload['reps']} | target power: {payload['power']:.2f}",
        f"- frozen cutoff: {payload['frozen_cutoff']} ({payload['cutoff_source']})",
        f"- protected hash: {payload['protected_hash']}",
        f"- date: {payload['date']}",
        "",
        "| target | trial | nominal alpha | MDE (metric units) | pass rate |"
        " observed (metric units) | ledger relative |",
        "| ------ | ----- | ------------- | ------------------ | --------- |"
        " ----------------------- | --------------- |",
    ]
    for target, block in payload["results"].items():
        if "error" in block:
            lines.append(f"| {target} | — | — | — | — | — | — |")
            continue
        for trial, row in block["mde"].items():
            effect = "none" if row["effect"] is None else f"{row['effect']:.6g}"
            rate = "—" if row["pass_rate"] is None else f"{row['pass_rate']:.2f}"
            observed = (
                "—"
                if block["observed_mean_improvement"] is None
                else f"{block['observed_mean_improvement']:.6g}"
            )
            ledger = (
                "—"
                if block["observed_ledger_relative"] is None
                else f"{block['observed_ledger_relative']:.3f}"
            )
            lines.append(
                f"| {target} | {trial} | {payload['trials'][trial]:.6f} | "
                f"{effect} | {rate} | {observed} | {ledger} |"
            )
    lines.append("")
    lines.append(
        "MDE uses the same planted-signal grid for every trial; a smaller "
        "alpha needs a larger effect."
    )
    (run_dir / "calibrate-power.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["markdown"] = str(run_dir / "calibrate-power.md")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    print("\n".join(lines))
    print(f"\nartifact: {run_dir}")
    return 0


def _leak_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One graded look-ahead scenario: candidate plus shuffle/shift controls.

    ``correlation`` is injected as a real future-dependent feature, not a
    post-hoc transform, so truncation invariance can catch it at any strength.
    """
    root = Path(str(payload["root"]))
    lab = load_model_lab(root)
    experiment = payload["experiment"]
    feature = str(payload["feature"])
    correlation = payload.get("correlation")
    lookahead = payload.get("lookahead", True) and correlation is not None
    extra = () if correlation is None else (feature,)
    feature_names = tuple(
        dict.fromkeys(
            (
                *lab.resolve_feature_names(str(experiment.get("featureSet") or "all")),
                *extra,
            )
        )
    )
    config = lab.Config(
        name=payload["config_name"],
        feature_names=feature_names,
        model_names=lab.resolve_model_set(str(experiment.get("modelSet") or "reference")),
        horizon=int(experiment.get("horizon") or 1),
    )
    target = str(payload["target"])
    folds = int(payload["folds"])
    seed = int(payload["seed"])

    def run(mode: str | None) -> dict[str, Any]:
        per_ticker = {
            ticker: lab.evaluate_control_ticker(
                ticker,
                config,
                target,
                mode,
                folds=folds,
                holdout_start=payload.get("holdout_start"),
                seed=seed,
                shift_bars=int(payload.get("shift_bars") or 169),
                inject_correlation=(
                    None
                    if payload.get("correlation") is None or lookahead
                    else float(payload["correlation"])
                ),
                lookahead_correlation=(
                    float(payload["correlation"]) if lookahead else None
                ),
                inject_feature=feature,
                baseline=str(payload.get("baseline") or "strongest"),
                structural=mode is None,
            )
            for ticker in payload["tickers"]
        }
        rows_by_ticker = {
            ticker: list(row.get("paired_rows") or [])
            for ticker, row in per_ticker.items()
        }
        pooled = lab.bootstrap_pooled(
            rows_by_ticker,
            alpha=float(payload["alpha"]),
            seed=seed,
            block_bars=int(payload["block_bars"]),
            bootstrap_samples=int(payload["bootstrap_samples"]),
        )
        pooled["fold_improvements"] = lab.pool_fold_improvements(
            {ticker: row.get("fold_improvements") or [] for ticker, row in per_ticker.items()},
            folds,
        )
        pooled["target"] = target
        return {"pooled": pooled, "per_ticker": per_ticker}

    candidate = run(None)
    controls = {mode: run(mode) for mode in payload.get("controls") or []}
    structural = {
        ticker: row.get("structural") or {}
        for ticker, row in candidate["per_ticker"].items()
    }
    return {
        "candidate": candidate,
        "controls": controls,
        "purge": lab.aggregate_purge(list(candidate["per_ticker"].values())),
        "structural": {
            "ok": all(bool(block.get("ok", True)) for block in structural.values()),
            "per_ticker": structural,
        },
    }


def _leak_prereg(target: str, alpha: float) -> dict[str, Any]:
    return {
        "run_id": "calibration-leak",
        "hypothesis_id": "calibration-leak",
        "family": "calibration",
        "lineage": "calibration-leak",
        "trial_index": 1,
        "criteria_version": 2,
        "holdout_start": "2026-01-01T00:00:00Z",
        "criteria": {
            "kind": "improvement",
            "criteria_version": 2,
            "target": target,
            "primary": {"featureSet": "all", "modelSet": "reference", "horizon": 1},
            "alpha": alpha,
            "fold_rule": "all_positive",
            "require_controls": True,
            "require_multiple_tickers": False,
            "requires_promote": False,
        },
        "experiment": {"tickers": ["BTC/USD"], "targets": [target]},
    }


def _leak_payload(
    lab_payload: Mapping[str, Any], target: str
) -> dict[str, Any]:
    """Shape a v2 lab payload from a leak job."""
    candidate = lab_payload["candidate"]
    controls = lab_payload["controls"]
    return {
        "results": [
            {
                "ticker": ticker,
                "paired": {
                    "n": len(row.get("paired_rows") or []),
                    "mean_improvement": (
                        sum(float(item["improvement"]) for item in row["paired_rows"])
                        / len(row["paired_rows"])
                        if row.get("paired_rows")
                        else None
                    ),
                    "ci_low": None,
                    "ci_high": None,
                },
            }
            for ticker, row in candidate["per_ticker"].items()
        ],
        "pooled": candidate["pooled"],
        "controls": {
            mode: {"pooled": block["pooled"]}
            for mode, block in controls.items()
        },
        "purge": lab_payload["purge"],
        # The structural result is the gate; the two blocks below are kept only
        # as documentation of what the lab says it did.
        "structural": lab_payload.get("structural") or {"ok": True},
        "preprocessing": {"fit_scope": "train"},
        "selection": {"scope": "pre-cutoff"},
        "target": target,
    }


def leak_contract_payloads(
    target: str, alpha: float
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any], str]]]:
    """A passing v2 payload plus one realistic leak per mechanism."""
    negative = {
        "mean_improvement": -0.01,
        "ci_low": -0.02,
        "ci_high": 0.0,
        "n": 100,
        "alpha": alpha,
        "fold_improvements": [-0.01] * 4,
        "resolution_ok": True,
    }
    base: dict[str, Any] = {
        "results": [
            {
                "ticker": "BTC/USD",
                "paired": {"n": 100, "mean_improvement": 0.01, "ci_low": 0.005},
            }
        ],
        "pooled": {
            "mean_improvement": 0.01,
            "ci_low": 0.005,
            "ci_high": 0.02,
            "n": 100,
            "alpha": alpha,
            "fold_improvements": [0.01, 0.01, 0.01, 0.01],
            "resolution_ok": True,
        },
        "controls": {
            "shuffle": {"pooled": dict(negative)},
            "shift": {"pooled": dict(negative)},
        },
        "purge": {"ok": True, "violations": []},
        "structural": {
            "ok": True,
            "truncation": {"ok": True, "violations": []},
            "perturbation": {"ok": True, "violations": []},
        },
        "preprocessing": {"fit_scope": "train"},
        "selection": {"scope": "pre-cutoff"},
        "target": target,
    }
    # Each scenario carries the structural evidence a real run would produce.
    # The self-reported fit_scope/selection fields stay in place as
    # documentation; they are no longer what makes the verdict invalid.
    scenarios: list[tuple[str, dict[str, Any], str]] = [
        (
            "scaler_fit_on_full_data",
            {
                **base,
                "structural": {
                    "ok": False,
                    "truncation": {"ok": True, "violations": []},
                    "perturbation": {
                        "ok": False,
                        "violations": [{"path": "targets.fwd_volume_ratio.model.scaler.mean"}],
                    },
                },
                "preprocessing": {"fit_scope": "all"},
            },
            "perturbation invariance",
        ),
        (
            "label_overlap_across_split",
            {
                **base,
                "purge": {
                    "ok": False,
                    "violations": [{"fold": 0, "reason": "label reach overlaps"}],
                },
            },
            "purge",
        ),
        (
            "selection_on_holdout",
            {
                **base,
                "structural": {
                    "ok": False,
                    "truncation": {"ok": True, "violations": []},
                    "perturbation": {
                        "ok": False,
                        "violations": [{"path": "feature_names"}],
                    },
                },
                "selection": {"scope": "holdout"},
            },
            "perturbation invariance (selection)",
        ),
        (
            "lookahead_feature",
            {
                **base,
                "structural": {
                    "ok": False,
                    "truncation": {
                        "ok": False,
                        "violation_count": 1,
                        "violations": [{"kind": "feature", "name": "leak", "row": 10}],
                    },
                    "perturbation": {"ok": True, "violations": []},
                },
            },
            "truncation invariance",
        ),
    ]
    return base, scenarios


def calibrate_leaks(
    root: Path,
    *,
    config: Mapping[str, Any],
    queue: Sequence[Hypothesis],
    entries: Sequence[Mapping[str, Any]],
    target: str = "fwd_volume_ratio",
    correlations: Sequence[float] = (0.5, 0.2, 0.1, 0.05),
    cutoff: str | None = None,
) -> dict[str, Any]:
    """Which mechanical layer (if any) catches each realistic leak."""
    lab = load_model_lab(root)
    hypothesis = calibration_hypothesis(queue)
    if hypothesis is None:
        raise SystemExit("no canary hypothesis in experiments/hypotheses.json")
    experiment = dict(hypothesis.experiment or {})
    tickers = [str(ticker) for ticker in experiment.get("tickers") or []]
    computed_cutoff, cutoff_source = calibration_cutoff(
        root, config, hypothesis, entries
    )
    cutoff = cutoff or computed_cutoff
    if cutoff and cutoff != computed_cutoff:
        cutoff_source = "cli"
    budgets = config.get("budgets") or {}
    alpha_family = float((hypothesis.criteria or {}).get("alpha", 0.05))
    alpha = alpha_for_trial(alpha_family, 1)
    prereg = _leak_prereg(target, alpha_family)

    graded: list[dict[str, Any]] = []
    for correlation in correlations:
        job = _leak_job(
            {
                "root": str(root),
                "experiment": experiment,
                "config_name": primary_config_name(experiment),
                "feature": lab.LEAK_CORRELATION_FEATURE,
                "target": target,
                "folds": int(experiment.get("folds") or budgets.get("folds", 4)),
                "tickers": tickers,
                "holdout_start": cutoff,
                "shift_bars": int(experiment.get("shiftBars") or 169),
                "block_bars": int(budgets.get("bootstrapBlockBars", 24)),
                "bootstrap_samples": int(budgets.get("bootstrapSamples", 10000)),
                "alpha": alpha,
                "seed": int(config.get("seed") or 0),
                "correlation": float(correlation),
                "lookahead": True,
                "controls": ["shuffle", "shift"],
            }
        )
        detected: list[dict[str, Any]] = []
        actual: list[float] = []
        for ticker in tickers:
            meta = (
                (job["candidate"]["per_ticker"].get(ticker) or {}).get("lookahead")
                or {}
            )
            if meta.get("actual_correlation") is not None:
                actual.append(float(meta["actual_correlation"]))
            rows, _ = lab.filter_rows_before(lab.load_ticker_rows(ticker), cutoff)
            samples = lab.build_samples(
                rows,
                horizon=int(experiment.get("horizon") or 1),
                interval_seconds=lab.INTERVAL_SECONDS,
            )
            samples, _ = lab.inject_lookahead_feature(
                rows,
                samples,
                target,
                float(correlation),
                horizon=int(experiment.get("horizon") or 1),
                interval_seconds=lab.INTERVAL_SECONDS,
                seed=int(config.get("seed") or 0),
                feature=lab.LEAK_CORRELATION_FEATURE,
            )
            detected.extend(lab.detect_leaks(samples, lab.TARGET_NAMES))
        payload = _leak_payload(job, target)
        payload["leak_detected"] = bool(detected)
        payload["leak_features"] = detected
        verdict = evaluate_criteria(prereg, payload)
        control_detail = {
            mode: bool(
                block["pooled"].get("resolution_ok") is True
                and paired_rule(
                    block["pooled"], alpha=alpha, fold_rule="all_positive"
                )[0]
            )
            for mode, block in job["controls"].items()
        }
        control_flagged = any(control_detail.values())
        structural = job.get("structural") or {}
        truncation_caught = any(
            (block.get("truncation") or {}).get("ok") is False
            for block in (structural.get("per_ticker") or {}).values()
        )
        perturbation_caught = any(
            (block.get("perturbation") or {}).get("ok") is False
            for block in (structural.get("per_ticker") or {}).values()
        )
        layers: list[str] = []
        if detected:
            layers.append("correlation")
        if truncation_caught:
            layers.append("truncation")
        if perturbation_caught:
            layers.append("perturbation")
        if control_flagged:
            layers.append("controls")
        if not job["purge"].get("ok", True):
            layers.append("purge")
        violation = contract_violation(payload)
        if violation:
            layers.append("contract")
        graded.append(
            {
                "correlation_requested": float(correlation),
                "correlation_actual": (
                    sum(actual) / len(actual) if actual else None
                ),
                "outcome": verdict.get("outcome"),
                "flagged_by": layers or ["none"],
                "uncaught": verdict.get("outcome") != "invalid",
                "correlation_detector": bool(detected),
                "truncation_caught": truncation_caught,
                "perturbation_caught": perturbation_caught,
                "controls_flagged": control_flagged,
                "controls": control_detail,
                "purge_ok": bool(job["purge"].get("ok", True)),
                "contract_violation": violation,
                "pooled_mean": job["candidate"]["pooled"].get("mean_improvement"),
                "pooled_ci_low": job["candidate"]["pooled"].get("ci_low"),
            }
        )

    baseline_job = _leak_job(
        {
            "root": str(root),
            "experiment": experiment,
            "config_name": primary_config_name(experiment),
            "feature": lab.LEAK_CORRELATION_FEATURE,
            "target": target,
            "folds": int(experiment.get("folds") or budgets.get("folds", 4)),
            "tickers": tickers,
            "holdout_start": cutoff,
            "shift_bars": int(experiment.get("shiftBars") or 169),
            "block_bars": int(budgets.get("bootstrapBlockBars", 24)),
            "bootstrap_samples": int(budgets.get("bootstrapSamples", 10000)),
            "alpha": alpha,
            "seed": int(config.get("seed") or 0),
            "correlation": None,
            "controls": ["shuffle", "shift"],
        }
    )
    baseline_controls = {
        mode: {
            "passed": bool(
                block["pooled"].get("resolution_ok") is True
                and paired_rule(
                    block["pooled"], alpha=alpha, fold_rule="all_positive"
                )[0]
            ),
            "mean_improvement": block["pooled"].get("mean_improvement"),
            "ci_low": block["pooled"].get("ci_low"),
        }
        for mode, block in baseline_job["controls"].items()
    }
    # Full control distribution on the unplanted archive: several >=1-week
    # shifts plus block permutations. The old single 24-bar shift lived inside
    # the target's own daily seasonality, so it could pass and make an honest
    # fwd_volume_ratio run automatically invalid.
    baseline_config = lab.Config(
        name=primary_config_name(experiment),
        feature_names=lab.resolve_feature_names(
            str(experiment.get("featureSet") or "all")
        ),
        model_names=lab.resolve_model_set(
            str(experiment.get("modelSet") or "reference")
        ),
        horizon=int(experiment.get("horizon") or 1),
    )
    baseline_distribution = lab._controls_for(
        tickers,
        baseline_config,
        target,
        folds=int(experiment.get("folds") or budgets.get("folds", 4)),
        holdout_start=cutoff,
        seed=int(config.get("seed") or 0),
        shift_bars=169,
        alpha=alpha,
        block_bars=int(budgets.get("bootstrapBlockBars", 24)),
        bootstrap_samples=1000,
        modes=["shuffle", "shift", "block"],
        control_shifts=lab.DEFAULT_CONTROL_SHIFTS,
        control_replicates=4,
        control_quantile=0.95,
        baseline="strongest",
    )
    for name, block in baseline_distribution.items():
        if name == "distribution":
            baseline_controls["distribution"] = {
                "median": block.get("median"),
                "quantile": block.get("quantile"),
                "quantile_value": block.get("quantile_value"),
                "n": block.get("n"),
            }
            continue
        for variant in block.get("by_shift") or block.get("by_seed") or [block]:
            pooled = variant.get("pooled") or variant
            label = (
                f"{name}:{variant.get('shift_bars') or variant.get('seed')}"
                if variant is not block
                else name
            )
            baseline_controls[label] = {
                "passed": bool(
                    pooled.get("resolution_ok") is True
                    and paired_rule(
                        pooled, alpha=alpha, fold_rule="all_positive"
                    )[0]
                ),
                "mean_improvement": pooled.get("mean_improvement"),
                "ci_low": pooled.get("ci_low"),
            }
    baseline_flagged = any(
        row.get("passed") for row in baseline_controls.values() if isinstance(row, dict)
    )
    baseline_invalidating = bool(
        (baseline_controls.get("shuffle") or {}).get("passed")
    )
    baseline_structural = baseline_job.get("structural") or {}
    baseline_structural_ok = bool(baseline_structural.get("ok"))
    clean_payload = _leak_payload(baseline_job, target)
    clean_payload["controls"] = baseline_distribution
    clean = evaluate_criteria(prereg, clean_payload)
    for row in graded:
        row["controls_specific"] = bool(row["controls_flagged"] and not baseline_flagged)

    # (b) A scaler fitted on the whole series must be caught by perturbation
    # invariance. This is a direct demonstration, independent of the pipeline.
    import numpy as np

    from rl_crypto.ml.base import Standardizer

    rows, _ = lab.filter_rows_before(lab.load_ticker_rows(tickers[0]), cutoff)
    samples = lab.build_samples(
        rows,
        horizon=int(experiment.get("horizon") or 1),
        interval_seconds=lab.INTERVAL_SECONDS,
    )
    scaler_features = lab.resolve_feature_names(str(experiment.get("featureSet") or "all"))
    complete = [
        sample
        for sample in samples
        if all(sample.features.get(name) is not None for name in scaler_features)
    ]

    def full_data_scaler_fit(values: Sequence[Any]) -> dict[str, Any]:
        usable = [
            sample
            for sample in values
            if all(sample.features.get(name) is not None for name in scaler_features)
        ]
        matrix = np.asarray(
            [[sample.features[name] for name in scaler_features] for sample in usable],
            dtype=float,
        )
        return {"scaler": Standardizer().fit(matrix).to_dict()}

    scaler_canary = lab.perturbation_invariance(
        tuple(complete),
        fit=full_data_scaler_fit,
        targets=[target],
        seed=int(config.get("seed") or 0),
    )

    base, scenarios = leak_contract_payloads(target, alpha)
    canaries: list[dict[str, Any]] = []
    contract_clean = evaluate_criteria(prereg, base)
    for name, payload, layer in scenarios:
        verdict = evaluate_criteria(prereg, payload)
        canaries.append(
            {
                "canary": name,
                "expected_layer": layer,
                "outcome": verdict.get("outcome"),
                "passed": verdict.get("outcome") == "invalid",
                "reasons": verdict.get("reasons") or [],
            }
        )
    protected = protected_digest(root)
    return {
        "action": "calibrate-leaks",
        "target": target,
        "tickers": tickers,
        "frozen_cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "protected_hash": protected["hash"],
        "date": iso(),
        "clean_baseline_outcome": clean.get("outcome"),
        "contract_clean_outcome": contract_clean.get("outcome"),
        "baseline_structural": baseline_structural,
        "baseline_structural_ok": baseline_structural_ok,
        "control_baseline": {
            "any_passed": baseline_flagged,
            "shuffle_passed": baseline_invalidating,
            "controls": baseline_controls,
        },
        "scaler_canary": {
            "caught": not bool(scaler_canary.get("ok")),
            "result": scaler_canary,
        },
        "graded": graded,
        "uncaught": [
            row["correlation_requested"] for row in graded if row["uncaught"]
        ],
        "canaries": canaries,
    }


def cmd_calibrate_leaks(args: argparse.Namespace) -> int:
    root = repo_root()
    config = load_config(root)
    queue = load_queue(root / "experiments" / "hypotheses.json")
    entries = read_ledger(root / "experiments" / "ledger.jsonl")
    correlations = [
        float(part)
        for part in str(args.correlations).replace(",", " ").split()
        if part.strip()
    ]
    payload = calibrate_leaks(
        root,
        config=config,
        queue=queue,
        entries=entries,
        target=args.target,
        correlations=correlations,
        cutoff=args.cutoff,
    )
    run_dir = _calibration_dir(args, "calibrate-leaks")
    payload["run_dir"] = str(run_dir)
    _write_json(run_dir / "calibrate-leaks.json", payload)
    lines = [
        "# Leak sensitivity",
        "",
        f"- target: {payload['target']}",
        f"- frozen cutoff: {payload['frozen_cutoff']} ({payload['cutoff_source']})",
        f"- protected hash: {payload['protected_hash']}",
        f"- clean baseline outcome: {payload['clean_baseline_outcome']}",
        "",
        "| leak | |r| actual | truncation | correlation detector | perturbation |"
        " controls | control baseline | purge | contract | outcome | flagged by |",
        "| ---- | --------- | ---------- | -------------------- | ------------ |"
        " -------- | ---------------- | ----- | -------- | ------- | ---------- |",
    ]
    for row in payload["graded"]:
        actual = (
            "—"
            if row["correlation_actual"] is None
            else f"{row['correlation_actual']:.3f}"
        )
        lines.append(
            f"| corr {row['correlation_requested']:.2f} | {actual} | "
            f"{'YES' if row['truncation_caught'] else 'no'} | "
            f"{'YES' if row['correlation_detector'] else 'no'} | "
            f"{'YES' if row['perturbation_caught'] else 'no'} | "
            f"{'YES' if row['controls_flagged'] else 'no'} | "
            f"{'YES' if payload['control_baseline']['any_passed'] else 'no'} | "
            f"{'no' if not row['purge_ok'] else 'ok'} | "
            f"{'YES' if row['contract_violation'] else 'no'} | "
            f"{row['outcome']} | {', '.join(row['flagged_by'])} |"
        )
    lines.extend(
        [
            "",
            "| realistic canary | expected layer | outcome | caught |",
            "| ---------------- | -------------- | ------- | ------ |",
        ]
    )
    for row in payload["canaries"]:
        lines.append(
            f"| {row['canary']} | {row['expected_layer']} | {row['outcome']} | "
            f"{'yes' if row['passed'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            f"- baseline structural ok: **{payload['baseline_structural_ok']}**",
            f"- full-data scaler caught by perturbation: "
            f"**{payload['scaler_canary']['caught']}**",
            f"- uncaught graded leaks: **{payload['uncaught'] or 'none'}**",
            "",
            "A 'none' row is a finding: no mechanical layer catches that leak "
            "at that strength.",
        ]
    )
    (run_dir / "calibrate-leaks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["markdown"] = str(run_dir / "calibrate-leaks.md")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    print("\n".join(lines))
    print(f"\nartifact: {run_dir}")
    return 0


def calibrate_baselines(
    root: Path,
    *,
    config: Mapping[str, Any],
    queue: Sequence[Hypothesis],
    entries: Sequence[Mapping[str, Any]],
    targets: Sequence[str] = ("fwd_volume_ratio", "fwd_volatility"),
    cutoff: str | None = None,
    folds: int | None = None,
) -> dict[str, Any]:
    """Diagnostic table: model skill against constant, seasonal, persistence.

    No ledger write and no verdict. The v2 criterion uses the strongest of the
    three; this shows the individual bars so a weak constant baseline cannot
    make a persistent target look skilful.
    """
    lab = load_model_lab(root)
    hypothesis = calibration_hypothesis(queue)
    if hypothesis is None:
        raise SystemExit("no canary hypothesis in experiments/hypotheses.json")
    experiment = dict(hypothesis.experiment or {})
    tickers = [str(ticker) for ticker in experiment.get("tickers") or []]
    computed_cutoff, cutoff_source = calibration_cutoff(
        root, config, hypothesis, entries
    )
    cutoff = cutoff or computed_cutoff
    if cutoff and cutoff != computed_cutoff:
        cutoff_source = "cli"
    budgets = config.get("budgets") or {}
    scope = config.get("scope") or {}
    config_obj = lab.Config(
        name=primary_config_name(experiment),
        feature_names=lab.resolve_feature_names(str(experiment.get("featureSet") or "all")),
        model_names=lab.resolve_model_set(str(experiment.get("modelSet") or "reference")),
        horizon=int(experiment.get("horizon") or 1),
    )
    fold_count = int(folds or experiment.get("folds") or budgets.get("folds", 4))
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        result = lab.evaluate_ticker(
            ticker,
            config_obj,
            list(targets),
            folds=fold_count,
            threshold=float(scope.get("threshold", 0.55)),
            position=str(scope.get("position", "long-only")),
            fee_bps=float(scope.get("feeBps", 10.0)),
            slippage_bps=float(scope.get("slippageBps", 5.0)),
            holdout_start=cutoff,
            primary_target=str(targets[0]),
            baseline="strongest",
            seed=int(config.get("seed") or 0),
        )
        if "error" in result:
            rows.append({"ticker": ticker, "error": result["error"]})
            continue
        for target, block in (result.get("targets") or {}).items():
            rows.append(
                {
                    "ticker": ticker,
                    "target": target,
                    "metric": block.get("metric"),
                    "model": block.get("model_mean"),
                    "baseline_mode": block.get("baseline_mode"),
                    "baselines": block.get("baselines"),
                    "skill_vs": block.get("skill_vs"),
                    "improvement_vs_strongest": block.get("improvement"),
                    "beats": block.get("beats"),
                }
            )
    protected = protected_digest(root)
    return {
        "action": "calibrate-baselines",
        "targets": list(targets),
        "tickers": tickers,
        "folds": fold_count,
        "frozen_cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "protected_hash": protected["hash"],
        "date": iso(),
        "rows": rows,
    }


def cmd_calibrate_baselines(args: argparse.Namespace) -> int:
    root = repo_root()
    config = load_config(root)
    queue = load_queue(root / "experiments" / "hypotheses.json")
    entries = read_ledger(root / "experiments" / "ledger.jsonl")
    targets = [
        part for part in str(args.targets).replace(",", " ").split() if part.strip()
    ]
    payload = calibrate_baselines(
        root,
        config=config,
        queue=queue,
        entries=entries,
        targets=targets,
        cutoff=args.cutoff,
        folds=args.folds,
    )
    run_dir = _calibration_dir(args, "calibrate-baselines")
    payload["run_dir"] = str(run_dir)
    _write_json(run_dir / "calibrate-baselines.json", payload)
    lines = [
        "# Baseline diagnostic",
        "",
        f"- tickers: {', '.join(payload['tickers'])}",
        f"- folds: {payload['folds']}",
        f"- frozen cutoff: {payload['frozen_cutoff']} ({payload['cutoff_source']})",
        f"- protected hash: {payload['protected_hash']}",
        "",
        "| target | ticker | metric | model | constant | seasonal | persistence |"
        " strongest | skill vs constant | skill vs seasonal | skill vs persistence |",
        "| ------ | ------ | ------ | ----- | -------- | -------- | ----------- |"
        " --------- | ----------------- | ----------------- | -------------------- |",
    ]
    for row in payload["rows"]:
        if "error" in row:
            lines.append(f"| — | {row['ticker']} | — | error: {row['error']} | | | | | | | |")
            continue
        baselines = row.get("baselines") or {}
        skill = row.get("skill_vs") or {}

        def fmt(value):
            return "—" if value is None else f"{value:+.4f}"

        def fmt_metric(value):
            return "—" if value is None else f"{value:.5f}"

        lines.append(
            f"| {row['target']} | {row['ticker']} | {row['metric']} | "
            f"{fmt_metric(row.get('model'))} | "
            f"{fmt_metric(baselines.get('constant'))} | "
            f"{fmt_metric(baselines.get('seasonal'))} | "
            f"{fmt_metric(baselines.get('persistence'))} | "
            f"{row.get('baseline_mode')} | "
            f"{fmt(skill.get('constant'))} | "
            f"{fmt(skill.get('seasonal'))} | "
            f"{fmt(skill.get('persistence'))} |"
        )
    lines.append("")
    lines.append(
        "Skill is relative improvement; a positive value means the model beat that "
        "baseline. The strongest column is the one the v2 verdict uses."
    )
    (run_dir / "calibrate-baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["markdown"] = str(run_dir / "calibrate-baselines.md")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    print("\n".join(lines))
    print(f"\nartifact: {run_dir}")
    return 0


def calibrate_controls(
    root: Path,
    *,
    config: Mapping[str, Any],
    queue: Sequence[Hypothesis],
    entries: Sequence[Mapping[str, Any]],
    targets: Sequence[str] = ("fwd_volume_ratio", "fwd_volatility"),
    shifts: Sequence[int] | None = None,
    folds: int | None = None,
    cutoff: str | None = None,
    block_bars: int = 24,
    seed: int = 0,
) -> dict[str, Any]:
    """Control improvement as a function of shift length (a diagnosis).

    The old control shifted labels by 24 bars, a multiple of both a day and a
    week. For a target with daily/weekly seasonality or strong persistence that
    shift preserves the very structure the control is supposed to destroy, so
    the "control" can look like skill. This table shows where the effect lives;
    it never writes the ledger.
    """
    lab = load_model_lab(root)
    hypothesis = calibration_hypothesis(queue)
    if hypothesis is None:
        raise SystemExit("no canary hypothesis in experiments/hypotheses.json")
    experiment = dict(hypothesis.experiment or {})
    tickers = [str(ticker) for ticker in experiment.get("tickers") or []]
    computed_cutoff, cutoff_source = calibration_cutoff(
        root, config, hypothesis, entries
    )
    cutoff = cutoff or computed_cutoff
    if cutoff and cutoff != computed_cutoff:
        cutoff_source = "cli"
    budgets = config.get("budgets") or {}
    fold_count = int(folds or experiment.get("folds") or budgets.get("folds", 4))
    shift_list = list(shifts or (1, 6, 12, 24, 48, 72, 96, 120, 144, 168, 192,
                                 216, 240, 264, 288, 312, 336, 169, 173, 181, 191))
    config_obj = lab.Config(
        name=primary_config_name(experiment),
        feature_names=lab.resolve_feature_names(str(experiment.get("featureSet") or "all")),
        model_names=lab.resolve_model_set(str(experiment.get("modelSet") or "reference")),
        horizon=int(experiment.get("horizon") or 1),
    )
    report: dict[str, Any] = {}
    for target in targets:
        rows: list[dict[str, Any]] = []
        for shift in shift_list:
            per_ticker = {
                ticker: lab.evaluate_control_ticker(
                    ticker,
                    config_obj,
                    target,
                    "shift",
                    folds=fold_count,
                    holdout_start=cutoff,
                    seed=int(seed),
                    shift_bars=int(shift),
                    block_bars=int(block_bars),
                    structural=False,
                )
                for ticker in tickers
            }
            pooled = lab.bootstrap_pooled(
                {ticker: row["paired_rows"] for ticker, row in per_ticker.items()},
                alpha=float((hypothesis.criteria or {}).get("alpha", 0.05)),
                seed=int(seed),
                block_bars=int((config.get("budgets") or {}).get("bootstrapBlockBars", 24)),
                bootstrap_samples=1000,
            )
            rows.append(
                {
                    "shift_bars": int(shift),
                    "multiple_of_24": int(shift) % 24 == 0,
                    "multiple_of_168": int(shift) % 168 == 0,
                    "mean_improvement": pooled.get("mean_improvement"),
                    "ci_low": pooled.get("ci_low"),
                    "ci_high": pooled.get("ci_high"),
                }
            )
        report[target] = rows
    protected = protected_digest(root)
    return {
        "action": "calibrate-controls",
        "targets": list(targets),
        "tickers": tickers,
        "folds": fold_count,
        "block_bars": int(block_bars),
        "frozen_cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "protected_hash": protected["hash"],
        "date": iso(),
        "results": report,
    }


def cmd_calibrate_controls(args: argparse.Namespace) -> int:
    root = repo_root()
    config = load_config(root)
    queue = load_queue(root / "experiments" / "hypotheses.json")
    entries = read_ledger(root / "experiments" / "ledger.jsonl")
    targets = [
        part for part in str(args.targets).replace(",", " ").split() if part.strip()
    ]
    shifts = [
        int(part)
        for part in str(args.shifts).replace(",", " ").split()
        if part.strip()
    ] or None
    payload = calibrate_controls(
        root,
        config=config,
        queue=queue,
        entries=entries,
        targets=targets,
        shifts=shifts,
        folds=args.folds,
        cutoff=args.cutoff,
        block_bars=int(args.block_bars),
        seed=int(config.get("seed") or 0),
    )
    run_dir = _calibration_dir(args, "calibrate-controls")
    payload["run_dir"] = str(run_dir)
    _write_json(run_dir / "calibrate-controls.json", payload)
    lines = [
        "# Control diagnosis: improvement vs shift length",
        "",
        f"- tickers: {', '.join(payload['tickers'])}",
        f"- folds: {payload['folds']}",
        f"- frozen cutoff: {payload['frozen_cutoff']} ({payload['cutoff_source']})",
        f"- protected hash: {payload['protected_hash']}",
        "",
        "| target | shift | x24 | x168 | control improvement | CI low | CI high |",
        "| ------ | ----- | --- | ---- | ------------------- | ------ | ------- |",
    ]
    for target, rows in payload["results"].items():
        for row in rows:
            mean = (
                "—"
                if row["mean_improvement"] is None
                else f"{row['mean_improvement']:+.6f}"
            )
            low = "—" if row["ci_low"] is None else f"{row['ci_low']:+.6f}"
            high = "—" if row["ci_high"] is None else f"{row['ci_high']:+.6f}"
            lines.append(
                f"| {target} | {row['shift_bars']} | "
                f"{'yes' if row['multiple_of_24'] else ''} | "
                f"{'yes' if row['multiple_of_168'] else ''} | {mean} | {low} | {high} |"
            )
    lines.append("")
    lines.append(
        "A control improvement that rises at multiples of 24 or 168 supports the "
        "seasonality/persistence explanation for the old 24-bar control."
    )
    (run_dir / "calibrate-controls.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["markdown"] = str(run_dir / "calibrate-controls.md")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    print("\n".join(lines))
    print(f"\nartifact: {run_dir}")
    return 0


def cmd_calibrate_coverage(args: argparse.Namespace) -> int:
    from rl_crypto.power import calibrate_coverage

    alphas = [
        float(part) for part in str(args.alphas).replace(",", " ").split() if part.strip()
    ]
    result = calibrate_coverage(
        n_rows=int(args.rows),
        block_bars=int(args.block_bars),
        alphas=alphas,
        draws=int(args.draws),
        simulations=int(args.simulations),
        autocorrelation=float(args.autocorrelation),
        effect=float(args.effect),
        seed=int(args.seed),
    )
    root = repo_root()
    protected = protected_digest(root)
    payload = {
        "action": "calibrate-coverage",
        "protected_hash": protected["hash"],
        "date": iso(),
        **result,
    }
    run_dir = _calibration_dir(args, "calibrate-coverage")
    payload["run_dir"] = str(run_dir)
    _write_json(run_dir / "calibrate-coverage.json", payload)
    lines = [
        "# CI coverage calibration",
        "",
        f"- protected hash: {payload['protected_hash']}",
        f"- rows: {payload['n_rows']} | blocks: {payload['n_blocks']}"
        f" | block bars: {payload['block_bars']}",
        f"- block autocorrelation: {payload['autocorrelation']}"
        f" | true effect: {payload['effect']}",
        f"- simulations: {args.simulations} | bootstrap draws: {args.draws}",
        "",
        "| alpha | coverage | excludes zero |",
        "| ----- | -------- | ------------- |",
    ]
    for row in payload["alphas"].values():
        lines.append(
            f"| {row['alpha']:.4g} | {row['coverage']:.4f} | "
            f"{row['excludes_zero']:.4f} |"
        )
    uniformity = payload.get("p_value_uniformity") or {}
    lines.extend(
        [
            "",
            f"p-value uniformity: n={uniformity.get('n')} "
            f"mean={uniformity.get('mean')} "
            f"chi-square={uniformity.get('chi_square')}",
            f"bins: {uniformity.get('counts')}",
            "",
            "Coverage should be at least 1 - alpha; at a zero effect the "
            "exclude-zero rate should be at or below alpha.",
        ]
    )
    (run_dir / "calibrate-coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["markdown"] = str(run_dir / "calibrate-coverage.md")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    print("\n".join(lines))
    print(f"\nartifact: {run_dir}")
    return 0


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


def estimate_power(
    root: Path,
    config: Mapping[str, Any],
    hypothesis: Hypothesis,
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate the design's MDE at the pre-registered trial alpha.

    A cheap causal pilot: train the pre-registered primary config on each
    walk-forward fold of the pre-cutoff slice and collect the paired
    improvement series. The time-block bootstrap's standard error gives the
    smallest effect the design can detect at the target power. For a
    ``comparison: champion`` hypothesis the series is candidate-vs-retrained-
    champion instead of candidate-vs-baseline.
    """
    criteria = dict(prereg.get("criteria") or {})
    experiment = dict(prereg.get("experiment") or {})
    minimal = prereg.get("minimal_meaningful_effect")
    if minimal is None:
        return {"checked": False, "reason": "no minimal_meaningful_effect declared"}
    if hypothesis.canary or str(criteria.get("kind")) == "diagnostic":
        return {"checked": False, "reason": "canary/diagnostic does not need power"}
    target = str(criteria.get("target") or "fwd_direction")
    try:
        feature_set, model_set, horizon_text = primary_config_name(experiment).split("+")
        horizon = int(horizon_text.removeprefix("h"))
    except (AttributeError, ValueError):
        return {"checked": False, "reason": "primary config is not a feature+model+hN name"}
    try:
        import numpy as np

        from rl_crypto.power import minimum_detectable_effect
    except ImportError as exc:  # pragma: no cover - environment failure
        return {"checked": False, "reason": f"power module unavailable: {exc}"}
    lab = load_model_lab(root)
    config_obj = lab.Config(
        name=primary_config_name(experiment),
        feature_names=lab.resolve_feature_names(feature_set),
        model_names=lab.resolve_model_set(model_set),
        horizon=horizon,
    )
    budgets = config.get("budgets") or {}
    tickers = [str(ticker) for ticker in experiment.get("tickers") or []]
    folds = int(experiment.get("folds") or budgets.get("folds", 4))
    block_bars = int(budgets.get("bootstrapBlockBars", 24))
    comparison = str(prereg.get("comparison") or "baseline")
    alpha = alpha_for_trial(
        float(criteria.get("alpha", 0.05)), int(prereg.get("trial_index") or 1)
    )
    baseline_values: list[float] = []
    champion_values: list[float] = []
    per_ticker: dict[str, Any] = {}
    for ticker in tickers:
        rows, _ = lab.filter_rows_before(
            lab.load_ticker_rows(ticker), prereg.get("holdout_start")
        )
        samples = lab.build_samples(
            rows, horizon=horizon, interval_seconds=lab.INTERVAL_SECONDS
        )
        splits = lab.walk_forward(
            samples, folds=folds, horizon=horizon, targets=[target]
        )
        baseline_rows: list[dict[str, Any]] = []
        champion_rows: list[dict[str, Any]] = []
        champion = None
        if comparison == "champion":
            try:
                champion = lab.load_bundle(
                    lab.default_bundle_path(ticker, directory=root / "models")
                )
            except FileNotFoundError:
                champion = None
        for train, test in splits:
            bundle = lab.train_tier1(
                train,
                horizon=horizon,
                feature_names=config_obj.feature_names,
                targets=[target],
                model_names=config_obj.model_names,
                holdout_fraction=0.0,
            )
            baseline_rows.extend(
                lab.paired_target_rows(
                    bundle, target, train, test, baseline="strongest", horizon=horizon
                )
            )
            if champion is not None and target in champion.targets:
                champion_model_names = {
                    target: str(
                        champion.targets[target].model.to_dict().get("kind")
                        or "ridge"
                    )
                }
                retrained = lab.train_tier1(
                    train,
                    horizon=champion.horizon,
                    feature_names=champion.feature_names,
                    targets=[target],
                    model_names=champion_model_names,
                    holdout_fraction=0.0,
                )
                champion_rows.extend(
                    lab.paired_between_bundles(
                        bundle, retrained, target, test, horizon=horizon
                    )
                )
        baseline_values.extend(float(row["improvement"]) for row in baseline_rows)
        champion_values.extend(float(row["improvement"]) for row in champion_rows)
        per_ticker[ticker] = {
            "n": len(baseline_rows),
            "mean": (
                float(np.mean([row["improvement"] for row in baseline_rows]))
                if baseline_rows
                else None
            ),
            "champion_n": len(champion_rows),
            "champion_mean": (
                float(np.mean([row["improvement"] for row in champion_rows]))
                if champion_rows
                else None
            ),
        }
    baseline_power = minimum_detectable_effect(
        baseline_values, alpha=alpha, block_bars=block_bars
    )
    champion_power = (
        minimum_detectable_effect(champion_values, alpha=alpha, block_bars=block_bars)
        if comparison == "champion" and champion_values
        else None
    )
    chosen = champion_power if comparison == "champion" else baseline_power
    mde = chosen.get("mde") if chosen else None
    return {
        "checked": True,
        "comparison": comparison,
        "target": target,
        "alpha": alpha,
        "minimal_meaningful_effect": float(minimal),
        "baseline": baseline_power,
        "champion": champion_power,
        "mde": mde,
        "underpowered": bool(
            mde is not None and float(mde) > float(minimal)
        ),
        "per_ticker": per_ticker,
    }


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

    run_type = str(getattr(args, "run_type", None) or hypothesis.run_type)
    if run_type not in {"experiment", "harness-change"}:
        raise SystemExit(f"unknown run type {run_type!r}")
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
        run_type=run_type,
        lineage=hypothesis.stable_lineage,
        seed=int(config.get("seed") or 0),
    )
    ctx.run_dir.mkdir(parents=True, exist_ok=True)

    def record_infra(stage: str, reason: str, kind: str | None = None) -> None:
        """Record an infrastructure block that never earned a pre-registration."""
        append_entry(
            ctx.ledger,
            {
                "kind": "blocked",
                "run_id": run_id,
                "hypothesis_id": hypothesis.id,
                "stage": stage,
                "reason": reason,
                "infra": True,
                "infra_kind": kind or infra_kind(reason, stage) or "infra",
            },
        )

    pf = preflight(root, config, executor=executor, worktree=not args.no_worktree)
    _write_json(ctx.artifact("preflight.json"), pf)
    if not pf["ok"] and not args.force:
        _emit(pf, as_json=args.json)
        record_infra("preflight", "preflight failed: " + ", ".join(pf.get("blocking") or []))
        print("\npreflight failed; pass --force to run anyway", file=sys.stderr)
        return 2

    void = harness_void(entries)
    if void is not None and not args.force:
        message = (
            "harness-void: the canary produced shuffled-label skill; a human must "
            "run `harness-clear` before further runs"
        )
        record_infra("harness", message, "harness-void")
        if args.json:
            print(json.dumps({"outcome": "harness-void", "reason": message}, indent=2))
        else:
            print(message)
        return 2

    if ctx.run_type != "harness-change" and not args.force:
        current_hash = protected_digest(ctx.repo)["hash"]
        reason = canary_gate(
            entries,
            current_hash,
            canary_entries=read_canary(root / "experiments" / CANARY_FILENAME),
        )
        if reason:
            record_infra("canary", reason, "canary-stale")
            if args.json:
                print(json.dumps({"outcome": "blocked", "reason": reason}, indent=2))
            else:
                print(reason)
            return 2

    ctx.holdout_start = resolve_or_compute_holdout(root, config, hypothesis, entries)
    if criteria_version(hypothesis.criteria) >= 2 and not ctx.holdout_start:
        message = (
            "cannot freeze a holdout cutoff: no archive timestamps for the scoped "
            "tickers; fix the data or pass RL_MODEL_IMPROVE_HOLDOUT_START"
        )
        record_infra("staleness", message, "staleness")
        print(message, file=sys.stderr)
        return 2
    new_info_reason = no_new_information_reason(
        root,
        hypothesis,
        entries,
        min_new_bars=int((config.get("noNewInformation") or {}).get("minNewBars", 168)),
    )
    if new_info_reason and not args.force:
        record_infra("staleness", new_info_reason, "staleness")
        if args.json:
            print(json.dumps({"outcome": "blocked", "reason": new_info_reason}, indent=2))
        else:
            print(new_info_reason)
        return 2

    prereg = build_preregistration(
        hypothesis,
        run_id=run_id,
        root=root,
        entries=entries,
        trigger=args.trigger,
        seed=ctx.seed,
        lineage=ctx.lineage,
        holdout_start=ctx.holdout_start,
        run_type=ctx.run_type,
        feature_version=FEATURE_VERSION,
        protected=protected_digest(root),
    )
    if run_type != "harness-change" and not args.force:
        power_report = estimate_power(root, config, hypothesis, prereg)
        prereg["power"] = power_report
        _write_json(ctx.artifact("power.json"), power_report)
        if power_report.get("underpowered"):
            reason = (
                "underpowered: the design's minimum detectable effect "
                f"{power_report.get('mde')} exceeds the minimal meaningful effect "
                f"{power_report.get('minimal_meaningful_effect')} at "
                f"alpha={power_report.get('alpha')}; no trial is consumed"
            )
            append_entry(
                ctx.ledger,
                {
                    "kind": "blocked",
                    "run_id": run_id,
                    "hypothesis_id": hypothesis.id,
                    "stage": "preregister",
                    "reason": reason,
                    "underpowered": True,
                    "power": power_report,
                },
            )
            if args.json:
                print(json.dumps({"outcome": "underpowered", "reason": reason}, indent=2))
            else:
                print(reason)
            return 2
    cap_reason = family_cap_reason(config, prereg)
    if cap_reason and not args.force:
        reason = cap_reason
        append_entry(
            ctx.ledger,
            {
                "kind": "blocked",
                "run_id": run_id,
                "hypothesis_id": hypothesis.id,
                "stage": "preregister",
                "reason": reason,
                "infra": True,
                "infra_kind": "family-cap",
            },
        )
        if args.json:
            print(json.dumps({"outcome": "blocked", "reason": reason}, indent=2))
        else:
            print(reason)
        return 2

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

    if ctx.holdout_start and not resolve_holdout_start(
        read_ledger(ctx.ledger), ctx.lineage
    ):
        append_entry(
            ctx.ledger,
            {
                "kind": "holdout_freeze",
                "lineage": ctx.lineage,
                "hypothesis_id": hypothesis.id,
                "holdout_start": ctx.holdout_start,
                "tickers": list(hypothesis.experiment.get("tickers") or []),
                "frozen_by": "model_improve",
                "note": "timestamp cutoff; appending bars never moves it",
            },
        )

    record = append_entry(ctx.ledger, prereg)
    prereg["id"] = record["id"]
    _write_json(ctx.artifact("preregister.json"), prereg)
    if args.commit is None:
        args.commit = created

    def stop(reason: str, stage: str) -> int:
        kind = infra_kind(reason, stage) or "infra"
        append_entry(
            ctx.ledger,
            {
                "kind": "blocked",
                "run_id": run_id,
                "hypothesis_id": hypothesis.id,
                "stage": stage,
                "reason": reason,
                "infra": True,
                "infra_kind": kind,
            },
        )
        verdict = {
            "outcome": "blocked",
            "passed": False,
            "reasons": [reason],
            "metrics": {},
            "infra": True,
            "infra_kind": kind,
        }
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

    budget_blocker = ceiling_reason(ctx)
    if budget_blocker:
        return stop(budget_blocker, "budget")

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
        "run_type": ctx.run_type,
        "lineage": ctx.lineage,
        "holdout_start": ctx.holdout_start,
        "seed": ctx.seed,
        "outcome": verdict.get("outcome"),
        "passed": verdict.get("passed"),
        "reasons": verdict.get("reasons") or [],
        "usage": ctx.usage,
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


# --------------------------------------------------------------------------- auditor red team


AUDITOR_PROMPT = """\
You are the adversarial leakage auditor. This directory is a git repository with
exactly one uncommitted change.

1. Read `PREREG.json` first; it is the frozen contract.
2. Inspect the change with `git diff` and read any changed files and tests.
3. Answer the seven audit questions in your agent instructions. Prefer concrete
   `file:line` evidence over generalities.
4. Write `audit.json` in this directory, and nothing else:

{
  "blocking": false,
  "findings": [
    {"question": "...", "verdict": "pass", "evidence": "src/...:12 ..."}
  ],
  "notes": "anything a reader should not misinterpret"
}

Set `blocking: true` if any verdict is `fail` or `cannot_tell`. Do not commit,
do not fix code, and do not edit any file other than `audit.json`.
"""


def _fixture_cases(fixtures: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for cls in ("leaky", "clean"):
        directory = fixtures / cls
        if not directory.is_dir():
            continue
        for diff in sorted(directory.glob("*.diff")):
            sidecar = diff.with_suffix(".json")
            expectation = load_json(sidecar, {}) or {}
            cases.append(
                {
                    "fixture": diff.stem,
                    "class": cls,
                    "kind": expectation.get("kind") or "",
                    "rationale": expectation.get("rationale") or "",
                    "diff": diff,
                }
            )
    return cases


def _prepare_fixture_repo(baseline: Path, diff: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(baseline, destination)
    agent_dir = destination / ".opencode" / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    # The auditor agent must be discoverable from the fixture repo.
    source_agent = (
        repo_root() / ".opencode" / "agents" / "model-improve-auditor.md"
    )
    if source_agent.is_file():
        shutil.copy2(source_agent, agent_dir / "model-improve-auditor.md")
    identity = ["-c", "user.email=auditor-check@example.invalid", "-c", "user.name=auditor-check"]
    for args in (["init", "-q"], ["add", "-A"], [*identity, "commit", "-qm", "baseline"]):
        result = _git(destination, *args)
        if result.returncode != 0:
            raise RuntimeError(f"fixture git {args[0]} failed: {result.stderr.strip()}")
    applied = _git(destination, "apply", "--whitespace=nowarn", str(diff))
    if applied.returncode != 0:
        raise RuntimeError(f"fixture diff did not apply: {applied.stderr.strip()}")


def _audit_fixture(
    config: Mapping[str, Any],
    repo: Path,
    run_dir: Path,
    *,
    timeout: float,
) -> dict[str, Any]:
    command = [
        opencode_command(),
        "run",
        "--agent",
        "model-improve-auditor",
        "--format",
        "json",
        "--auto",
        "--dir",
        str(repo),
        "--title",
        f"auditor-check {repo.name}",
    ]
    model = resolve_model(config, "auditor")
    if model:
        command.extend(["--model", str(model)])
    command.append(AUDITOR_PROMPT)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        proc = _run(
            command,
            cwd=repo,
            timeout=timeout,
            stdin_null=True,
            env={"RL_MODEL_IMPROVE_RUN_DIR": str(run_dir)},
        )
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = str(exc.stdout or "")
        stderr = f"timeout after {timeout}s"
    (run_dir / "auditor.jsonl").write_text(
        stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8"
    )
    audit = load_json(run_dir / "audit.json", None)
    findings = list((audit or {}).get("findings") or []) if isinstance(audit, dict) else []
    verdicts = [str(item.get("verdict") or "").lower() for item in findings]
    error = None
    if not isinstance(audit, dict):
        error = (
            f"auditor produced no audit.json (returncode {returncode}); "
            "treat this fixture as unmeasured, not as a pass"
        )
    blocked = bool((audit or {}).get("blocking")) or any(
        verdict in {"fail", "cannot_tell"} for verdict in verdicts
    )
    return {
        "returncode": returncode,
        "seconds": round(time.time() - started, 1),
        "model": model,
        "audit": audit,
        "findings": findings,
        "blocked": blocked,
        "error": error,
        "stdout": stdout,
    }


def cmd_auditor_check(args: argparse.Namespace) -> int:
    root = repo_root()
    config = apply_model_override(load_config(root), getattr(args, "model", None))
    fixtures = (
        Path(args.fixtures).expanduser().resolve()
        if args.fixtures
        else root / "experiments" / "auditor-fixtures"
    )
    baseline = fixtures / "baseline"
    if not baseline.is_dir():
        raise SystemExit(f"fixture baseline not found: {baseline}")
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = default_run_root() / f"auditor-check-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    timeout = float(args.timeout or (config.get("budgets") or {}).get("roleTimeoutSeconds", 300))
    cases = _fixture_cases(fixtures)
    results: list[dict[str, Any]] = []
    for case in cases:
        repo = run_dir / "repos" / case["fixture"]
        _prepare_fixture_repo(baseline, case["diff"], repo)
        outcome = _audit_fixture(config, repo, run_dir / "logs" / case["fixture"], timeout=timeout)
        results.append(
            {
                "fixture": case["fixture"],
                "class": case["class"],
                "kind": case["kind"],
                "rationale": case["rationale"],
                "blocked": outcome["blocked"],
                "error": outcome.get("error"),
                "returncode": outcome["returncode"],
                "model": outcome["model"],
                "findings": [
                    {
                        "question": item.get("question"),
                        "verdict": item.get("verdict"),
                        "evidence": item.get("evidence"),
                    }
                    for item in outcome["findings"]
                    if isinstance(item, Mapping)
                ],
                "notes": (outcome["audit"] or {}).get("notes") if outcome["audit"] else None,
            }
        )
    measured = [row for row in results if not row.get("error")]
    leaky = [row for row in measured if row["class"] == "leaky"]
    clean = [row for row in measured if row["class"] == "clean"]
    blocked_leaky = sum(1 for row in leaky if row["blocked"])
    blocked_clean = sum(1 for row in clean if row["blocked"])
    payload = {
        "action": "auditor-check",
        "fixtures": str(fixtures),
        "model": resolve_model(config, "auditor"),
        "recall": (blocked_leaky / len(leaky)) if leaky else None,
        "false_block_rate": (blocked_clean / len(clean)) if clean else None,
        "counts": {
            "leaky": sum(1 for row in results if row["class"] == "leaky"),
            "clean": sum(1 for row in results if row["class"] == "clean"),
            "measured": len(measured),
            "errors": len(results) - len(measured),
            "leaky_blocked": blocked_leaky,
            "clean_blocked": blocked_clean,
        },
        "results": results,
    }
    (run_dir / "auditor-check.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"model: {payload['model']}")
    print(f"recall: {payload['recall']}  false-block: {payload['false_block_rate']}")
    _print_table(
        ["fixture", "class", "blocked", "kind"],
        [
            [
                row["fixture"],
                row["class"],
                "error" if row.get("error") else ("yes" if row["blocked"] else "no"),
                row["kind"],
            ]
            for row in results
        ],
    )
    print(f"artifact: {run_dir / 'auditor-check.json'}")
    return 0


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
    run.add_argument(
        "--run-type",
        choices=("experiment", "harness-change"),
        default=None,
        help="harness-change may touch protected measurement files; needs human approval",
    )
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

    auditor = sub.add_parser(
        "auditor-check",
        help="red-team the auditor on leaky and clean fixture diffs",
    )
    auditor.add_argument("--fixtures", default=None)
    auditor.add_argument("--run-dir", default=None)
    auditor.add_argument("--timeout", type=float, default=None)
    auditor.add_argument("--json", action="store_true")
    auditor.set_defaults(func=cmd_auditor_check)

    cal_coverage = sub.add_parser(
        "calibrate-coverage",
        help="offline: CI coverage under known-zero block-autocorrelated draws",
    )
    cal_coverage.add_argument("--rows", type=int, default=600)
    cal_coverage.add_argument("--block-bars", type=int, default=24)
    cal_coverage.add_argument("--alphas", default="0.01,0.05,0.1,0.2")
    cal_coverage.add_argument("--draws", type=int, default=1000)
    cal_coverage.add_argument("--simulations", type=int, default=400)
    cal_coverage.add_argument("--autocorrelation", type=float, default=0.8)
    cal_coverage.add_argument("--effect", type=float, default=0.0)
    cal_coverage.add_argument("--seed", type=int, default=0)
    cal_coverage.add_argument("--run-dir", default=None)
    cal_coverage.add_argument("--json", action="store_true")
    cal_coverage.set_defaults(func=cmd_calibrate_coverage)

    cal_null = sub.add_parser(
        "calibrate-null",
        help="offline: empirical false-positive rate of the v2 rule under shuffled labels",
    )
    cal_null.add_argument("--seeds", type=int, default=300)
    cal_null.add_argument("--trials", default="1,5,10")
    cal_null.add_argument("--alpha", type=float, default=0.05)
    cal_null.add_argument("--jobs", type=int, default=1)
    cal_null.add_argument(
        "--cutoff", default=None, help="pin the holdout cutoff instead of computing it"
    )
    cal_null.add_argument("--run-dir", default=None)
    cal_null.add_argument("--json", action="store_true")
    cal_null.set_defaults(func=cmd_calibrate_null)

    cal_power = sub.add_parser(
        "calibrate-power",
        help="offline: minimum detectable planted improvement at a target power",
    )
    cal_power.add_argument(
        "--targets", default="fwd_volume_ratio fwd_volatility", help="regression targets"
    )
    cal_power.add_argument("--reps", type=int, default=20)
    cal_power.add_argument(
        "--effect-fractions",
        default="0.001,0.003,0.01,0.03,0.1",
        help="planted effect as a fraction of label variance per grid point",
    )
    cal_power.add_argument("--trials", default="1,5,10")
    cal_power.add_argument("--alpha", type=float, default=0.05)
    cal_power.add_argument("--power", type=float, default=0.8)
    cal_power.add_argument("--jobs", type=int, default=1)
    cal_power.add_argument(
        "--controls", default="shuffle,shift", help="control modes, or 'none'"
    )
    cal_power.add_argument(
        "--cutoff", default=None, help="pin the holdout cutoff instead of computing it"
    )
    cal_power.add_argument("--run-dir", default=None)
    cal_power.add_argument("--json", action="store_true")
    cal_power.set_defaults(func=cmd_calibrate_power)

    cal_baselines = sub.add_parser(
        "calibrate-baselines",
        help="offline: skill vs constant/seasonal/persistence (diagnostic only)",
    )
    cal_baselines.add_argument(
        "--targets", default="fwd_volume_ratio,fwd_volatility"
    )
    cal_baselines.add_argument("--folds", type=int, default=None)
    cal_baselines.add_argument("--cutoff", default=None)
    cal_baselines.add_argument("--run-dir", default=None)
    cal_baselines.add_argument("--json", action="store_true")
    cal_baselines.set_defaults(func=cmd_calibrate_baselines)

    cal_controls = sub.add_parser(
        "calibrate-controls",
        help="offline: control improvement vs shift length (diagnosis only)",
    )
    cal_controls.add_argument(
        "--targets", default="fwd_volume_ratio,fwd_volatility"
    )
    cal_controls.add_argument("--shifts", default="")
    cal_controls.add_argument("--folds", type=int, default=None)
    cal_controls.add_argument("--block-bars", type=int, default=24)
    cal_controls.add_argument("--cutoff", default=None)
    cal_controls.add_argument("--run-dir", default=None)
    cal_controls.add_argument("--json", action="store_true")
    cal_controls.set_defaults(func=cmd_calibrate_controls)

    cal_leaks = sub.add_parser(
        "calibrate-leaks",
        help="offline: which mechanical layer catches each injected leak",
    )
    cal_leaks.add_argument("--target", default="fwd_volume_ratio")
    cal_leaks.add_argument("--correlations", default="0.5,0.2,0.1,0.05")
    cal_leaks.add_argument("--cutoff", default=None)
    cal_leaks.add_argument("--run-dir", default=None)
    cal_leaks.add_argument("--json", action="store_true")
    cal_leaks.set_defaults(func=cmd_calibrate_leaks)

    verify = sub.add_parser("verify", help="forward-test status and bundle freshness")
    verify.add_argument("--run-dir", default=None)
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=cmd_verify)

    doctor = sub.add_parser("doctor", help="tooling, agents and configuration")
    executor_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    rotate = sub.add_parser(
        "holdout-rotate",
        help="human-approved: move a lineage's holdout cutoff forward",
    )
    rotate.add_argument("--lineage", required=True)
    rotate.add_argument("--holdout-start", required=True)
    rotate.add_argument("--reason", required=True)
    rotate.add_argument("--json", action="store_true")
    rotate.set_defaults(func=cmd_holdout_rotate)

    clear = sub.add_parser(
        "harness-clear", help="human-approved: clear a harness_void marker"
    )
    clear.add_argument("--reason", required=True)
    clear.add_argument("--json", action="store_true")
    clear.set_defaults(func=cmd_harness_clear)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())


