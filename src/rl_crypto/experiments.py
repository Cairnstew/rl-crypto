"""Append-only experiment records for the model-improvement loop.

This module is the *mechanical* half of ``/model-improve`` and the
``tools/model_improve.py`` orchestrator. It knows nothing about LLMs: it
pre-registers an experiment before it runs, fingerprints the data and git
state, counts trials per hypothesis family, enforces holdout reuse, and turns a
pre-registered criterion plus a ``rl-model-lab`` payload into a pass/fail
verdict with reasons.

The rules encoded here are the ones a human would otherwise have to remember:

* a criterion is written **before** the result exists and may not change after;
* a hypothesis lineage may touch the final holdout once;
* a family's trial count is cumulative across runs and never resets;
* an improvement must hold per ticker, not only in a pooled mean;
* a negative result is a completed experiment, not a failure.

Everything is JSONL/JSON so ``git log`` is the audit trail and a CI run can
render the same scoreboard as a human.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "rl-crypto/experiments/1"
HYPOTHESIS_STATUSES = frozenset({"open", "blocked", "done", "retired"})
ENTRY_KINDS = frozenset(
    {
        "preregister",
        "results",
        "holdout",
        "holdout_freeze",
        "holdout_rotate",
        "verdict",
        "blocked",
        "note",
        "harness_void",
        "harness_clear",
    }
)
CAPABILITY_NEEDS = frozenset({"code", "data", "approval", "tooling"})
RUN_TYPES = frozenset({"experiment", "harness-change"})

#: Criteria written before this hardening keep the old interpretation forever.
CRITERIA_VERSION = 2

#: Files whose content defines the measurement contract. A normal experiment run
#: may not edit them: doing so would let an implementer move the goalposts it is
#: being judged against. ``run_type: harness-change`` is the only escape hatch,
#: and it cannot produce a scoreboard-eligible verdict.
PROTECTED_FILES: tuple[str, ...] = (
    "src/rl_crypto/experiments.py",
    "src/rl_crypto/targets.py",
    "src/rl_crypto/samples.py",
    "src/rl_crypto/ml/tier1.py",
    "tools/model_lab.py",
    "tools/model_improve.py",
    "tools/forward_score.py",
    "tests/test_experiments.py",
    "tests/test_model_lab.py",
    "tests/test_ml_measurement.py",
    "tests/test_model_improve.py",
)

DEFAULT_CRITERIA: dict[str, Any] = {
    "kind": "improvement",
    "target": "fwd_direction",
    "tickers_beating_min": 2,
    "folds_beating_min": 3,
    "mean_improvement_min": 0.0,
    "min_improvement_min": 0.0,
    "economic": {
        "total_return_min": 0.0,
        "sharpe_above_momentum": True,
        "trades_min": 100,
    },
    "requires_promote": False,
}

#: Criteria version 2 replaces ticker votes with one pooled paired improvement
#: and a time-block bootstrap CI. ``fold_rule`` is explicit because the old
#: combination of ``folds_beating_min=3`` and ``min_improvement_min=0`` was
#: contradictory (the latter already required every fold to be positive).
DEFAULT_CRITERIA_V2: dict[str, Any] = {
    "kind": "improvement",
    "criteria_version": CRITERIA_VERSION,
    "target": "fwd_direction",
    "primary": None,
    "alpha": 0.05,
    "fold_rule": "all_positive",
    "require_controls": True,
    "require_multiple_tickers": True,
    "economic": {
        "total_return_min": 0.0,
        "sharpe_above_momentum": True,
        "trades_min": 100,
    },
    "requires_promote": False,
}


def criteria_version(criteria: Mapping[str, Any] | None) -> int:
    """The interpretation version of a criteria block.

    Absent means version 1: every ledger entry written before the hardening is
    judged by exactly the rules that produced it.
    """
    if not criteria:
        return 1
    try:
        return int(criteria.get("criteria_version") or 1)
    except (TypeError, ValueError):
        return 1


def protected_digest(root: Path) -> dict[str, Any]:
    """Hash the measurement contract so a run can prove it was not edited."""
    files: dict[str, str | None] = {}
    for name in PROTECTED_FILES:
        path = root / name
        files[name] = sha256_file(path) if path.exists() else None
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((files[name] or "missing").encode("utf-8"))
        digest.update(b"\n")
    return {"hash": digest.hexdigest(), "files": files}


# --------------------------------------------------------------------------- time / hashing


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(moment: datetime | None = None) -> str:
    return (moment or utcnow()).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug_for_ticker(ticker: str) -> str:
    return ticker.replace("/", "-").replace(" ", "-").lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def combined_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# --------------------------------------------------------------------------- git / data


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state(root: Path) -> dict[str, Any]:
    """Commit SHA, branch and dirtiness at the moment the experiment starts."""
    sha = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(root, "status", "--porcelain")
    return {
        "sha": sha,
        "branch": branch,
        "dirty": bool(status),
        "dirty_paths": status.splitlines() if status else [],
    }


def data_fingerprint(root: Path, tickers: Sequence[str]) -> dict[str, Any]:
    """Content hash + span of the archive actually used by an experiment.

    Only ``candles.jsonl`` is hashed: it is the input both the lab and the
    production trainer read. A missing file is recorded as ``absent`` instead
    of silently hashing an empty string, so a stale/missing ticker is visible
    in the ledger.
    """
    per_ticker: dict[str, Any] = {}
    hashes: list[str] = []
    for ticker in tickers:
        path = root / "data" / slug_for_ticker(ticker) / "candles.jsonl"
        if not path.exists():
            per_ticker[ticker] = {"absent": True}
            continue
        digest = sha256_file(path)
        rows = 0
        last: str | None = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                try:
                    last = json.loads(line).get("timestamp") or last
                except json.JSONDecodeError:
                    continue
        per_ticker[ticker] = {"sha256": digest, "rows": rows, "last": last}
        hashes.append(digest)
    return {"tickers": per_ticker, "combined": combined_hash(hashes) if hashes else None}


# --------------------------------------------------------------------------- queue


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One pre-registered research question from ``experiments/hypotheses.json``."""

    id: str
    title: str
    family: str
    order: int
    status: str
    needs: tuple[str, ...]
    mechanism: str
    experiment: dict[str, Any]
    criteria: dict[str, Any]
    depends_on: tuple[str, ...] = ()
    evidence: str = ""
    notes: str = ""
    run_type: str = "experiment"
    lineage: str = ""
    canary: bool = False

    @property
    def needs_code(self) -> bool:
        return "code" in self.needs

    @property
    def stable_lineage(self) -> str:
        """The durable lineage id; a new candidate chain needs a new id."""
        return self.lineage or self.id

    @property
    def needs_approval(self) -> bool:
        return "approval" in self.needs

    @property
    def needs_data(self) -> bool:
        return "data" in self.needs

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "family": self.family,
            "order": self.order,
            "status": self.status,
            "needs": list(self.needs),
            "mechanism": self.mechanism,
            "experiment": self.experiment,
            "criteria": self.criteria,
            "depends_on": list(self.depends_on),
            "evidence": self.evidence,
            "notes": self.notes,
            "run_type": self.run_type,
            "lineage": self.stable_lineage,
            "canary": self.canary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Hypothesis:
        unknown = set(payload.get("needs") or ()) - CAPABILITY_NEEDS
        if unknown:
            raise ValueError(f"{payload.get('id')}: unknown needs {sorted(unknown)}")
        status = str(payload.get("status", "open"))
        if status not in HYPOTHESIS_STATUSES:
            raise ValueError(f"{payload.get('id')}: unknown status {status!r}")
        run_type = str(payload.get("run_type", "experiment"))
        if run_type not in RUN_TYPES:
            raise ValueError(f"{payload.get('id')}: unknown run_type {run_type!r}")
        raw_criteria = dict(payload.get("criteria") or {})
        version = criteria_version(raw_criteria)
        base = DEFAULT_CRITERIA_V2 if version >= 2 else DEFAULT_CRITERIA
        return cls(
            id=str(payload["id"]),
            title=str(payload["title"]),
            family=str(payload.get("family", payload["id"])),
            order=int(payload.get("order", 0)),
            status=status,
            needs=tuple(str(n) for n in payload.get("needs") or ()),
            mechanism=str(payload.get("mechanism", "")),
            experiment=dict(payload.get("experiment") or {}),
            criteria={**base, **raw_criteria, "criteria_version": version},
            depends_on=tuple(str(d) for d in payload.get("depends_on") or ()),
            evidence=str(payload.get("evidence", "")),
            notes=str(payload.get("notes", "")),
            run_type=run_type,
            lineage=str(payload.get("lineage") or ""),
            canary=bool(payload.get("canary", False)),
        )


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc


def load_queue(path: Path) -> list[Hypothesis]:
    payload = load_json(path, {"hypotheses": []})
    rows = payload.get("hypotheses") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list or {{'hypotheses': [...]}}")
    queue = [Hypothesis.from_dict(row) for row in rows]
    ids = [h.id for h in queue]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate hypothesis ids")
    return sorted(queue, key=lambda h: (h.order, h.id))


# --------------------------------------------------------------------------- ledger


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON ({exc})") from exc
            entries.append(entry)
    return entries


def _next_id(entries: Sequence[Mapping[str, Any]]) -> str:
    highest = 0
    for entry in entries:
        value = str(entry.get("id", ""))
        if value.startswith("e") and value[1:].isdigit():
            highest = max(highest, int(value[1:]))
    return f"e{highest + 1:04d}"


def append_entry(path: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Append one immutable record, assigning a sequential id under a lock.

    The ledger is append-only by contract: records are never edited or deleted.
    A correction is a new ``note`` entry that cites the superseded id.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing = [json.loads(line) for line in handle if line.strip()]
            record = {
                "id": _next_id(existing),
                "schema": SCHEMA,
                "at": iso(),
                **dict(entry),
            }
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


# --------------------------------------------------------------------------- accounting


def trial_counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Cumulative preregistrations per family and per hypothesis."""
    by_family: dict[str, int] = {}
    by_hypothesis: dict[str, int] = {}
    for entry in entries:
        if entry.get("kind") != "preregister":
            continue
        family = str(entry.get("family") or entry.get("hypothesis_id") or "unknown")
        hid = str(entry.get("hypothesis_id") or "unknown")
        by_family[family] = by_family.get(family, 0) + 1
        by_hypothesis[hid] = by_hypothesis.get(hid, 0) + 1
    return {"family": by_family, "hypothesis": by_hypothesis}


def holdout_touches(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Distinct holdout-touching runs per durable lineage.

    Old entries used ``<H>-<run_id>`` as the lineage, so the count is one per
    entry; new entries use a stable lineage and count each run once even when a
    dry run and an apply are recorded for the same run.
    """
    seen: dict[str, set[str]] = {}
    for entry in entries:
        if not entry.get("holdout_touched"):
            continue
        lineage = str(entry.get("lineage") or entry.get("hypothesis_id") or "unknown")
        run = str(entry.get("run_id") or entry.get("id") or "")
        seen.setdefault(lineage, set()).add(run)
    return {lineage: len(runs) for lineage, runs in seen.items()}


def holdout_cutoffs(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Latest frozen holdout cutoff per lineage.

    ``holdout_rotate`` is a human-approved forward move; the newest entry wins.
    """
    cutoffs: dict[str, str] = {}
    for entry in entries:
        if entry.get("kind") not in {"holdout_freeze", "holdout_rotate"}:
            continue
        lineage = str(entry.get("lineage") or entry.get("hypothesis_id") or "")
        start = entry.get("holdout_start")
        if lineage and start:
            cutoffs[lineage] = str(start)
    return cutoffs


def resolve_holdout_start(
    entries: Sequence[Mapping[str, Any]], lineage: str
) -> str | None:
    return holdout_cutoffs(entries).get(lineage)


def harness_void(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """An uncleared void marker, if the canary has failed the harness."""
    void: dict[str, Any] | None = None
    for entry in entries:
        if entry.get("kind") == "harness_void":
            void = entry
        elif entry.get("kind") == "harness_clear":
            void = None
    return void


def completed_hypotheses(entries: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for entry in entries:
        if entry.get("kind") == "verdict" and entry.get("outcome") in {
            "negative",
            "provisional",
            "confirmed",
            "refuted",
            "diagnostic",
            "invalid",
            "harness-change",
        }:
            out.add(str(entry.get("hypothesis_id")))
    return out


def select_hypothesis(
    queue: Sequence[Hypothesis],
    entries: Sequence[Mapping[str, Any]],
    *,
    capabilities: set[str] | None = None,
) -> Hypothesis | None:
    """First runnable, not-yet-completed hypothesis in queue order.

    ``capabilities`` defaults to every capability, i.e. code/data/approval are
    assumed available. The orchestrator passes its real set so a
    ``--no-llm`` run can only select hypotheses that need no code.
    """
    if capabilities is None:
        capabilities = set(CAPABILITY_NEEDS)
    done = completed_hypotheses(entries)
    for hypothesis in queue:
        if hypothesis.status != "open" or hypothesis.id in done:
            continue
        if any(need not in capabilities for need in hypothesis.needs):
            continue
        if hypothesis.depends_on and not set(hypothesis.depends_on) <= done:
            continue
        return hypothesis
    return None


def build_preregistration(
    hypothesis: Hypothesis,
    *,
    run_id: str,
    root: Path,
    entries: Sequence[Mapping[str, Any]],
    trigger: str = "manual",
    seed: int = 0,
    lineage: str | None = None,
    holdout_start: str | None = None,
    run_type: str | None = None,
    feature_version: str | None = None,
    protected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the experiment definition before anything is executed."""
    counts = trial_counts(entries)
    tickers = hypothesis.experiment.get("tickers") or []
    git = git_state(root)
    prereg = {
        "kind": "preregister",
        "run_id": run_id,
        "hypothesis_id": hypothesis.id,
        "title": hypothesis.title,
        "family": hypothesis.family,
        "lineage": lineage or hypothesis.stable_lineage,
        "trial_index": counts["family"].get(hypothesis.family, 0) + 1,
        "hypothesis_trial_index": counts["hypothesis"].get(hypothesis.id, 0) + 1,
        "trigger": trigger,
        "seed": seed,
        "run_type": run_type or hypothesis.run_type,
        "criteria_version": criteria_version(hypothesis.criteria),
        "holdout_start": holdout_start,
        "feature_version": feature_version,
        "protected": dict(protected or {}),
        "git": git,
        "data": data_fingerprint(root, tickers),
        "experiment": hypothesis.experiment,
        "criteria": hypothesis.criteria,
        "config_hash": sha256_json(hypothesis.experiment),
        "criteria_hash": sha256_json(hypothesis.criteria),
        "mechanism": hypothesis.mechanism,
        "holdout_touched": False,
        "status": "preregistered",
    }
    problems = validate_preregistration(prereg)
    if problems:
        raise ValueError("invalid preregistration: " + "; ".join(problems))
    return prereg


def validate_preregistration(entry: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    for key in (
        "run_id",
        "hypothesis_id",
        "family",
        "lineage",
        "trial_index",
        "experiment",
        "criteria",
        "config_hash",
        "criteria_hash",
    ):
        if key not in entry or entry[key] in (None, "", {}, []):
            problems.append(f"missing {key}")
    experiment = entry.get("experiment") or {}
    for key in ("tickers", "targets", "maxConfigs", "folds"):
        if not experiment.get(key):
            problems.append(f"experiment missing {key}")
    criteria = entry.get("criteria") or {}
    if criteria.get("kind", "improvement") == "improvement":
        if "target" not in criteria:
            problems.append("criteria missing target")
        if criteria_version(criteria) >= 2:
            if not criteria.get("primary"):
                problems.append("criteria v2 missing primary config")
            if not entry.get("holdout_start"):
                problems.append("criteria v2 missing holdout_start")
            if not 0.0 < float(criteria.get("alpha", 0.05)) < 1.0:
                problems.append("criteria v2 alpha must be in (0, 1)")
            if criteria.get("fold_rule") not in {"all_positive", "majority_positive", "none"}:
                problems.append("criteria v2 has an unknown fold_rule")
        elif int(criteria.get("tickers_beating_min", 0)) < 1:
            problems.append("criteria missing tickers_beating_min")
    return problems


# --------------------------------------------------------------------------- verdict


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _best_rows_by_ticker(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Best scored row per ticker.

    Prefer the lab's gated ``top`` list; otherwise rank every row so a negative
    result still reports the numbers that made it negative.
    """
    rows = list(payload.get("top") or [])
    if not rows:
        rows = list(payload.get("results") or [])
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "unknown")

        def rank_key(item: Mapping[str, Any]) -> tuple[float, float, float]:
            targets = item.get("targets") or {}
            values = [
                (target or {}).get("folds_beating") or 0
                for target in targets.values()
                if isinstance(target, Mapping)
            ]
            means = [
                (target or {}).get("mean_improvement")
                for target in targets.values()
                if isinstance(target, Mapping)
            ]
            numeric_means = [float(v) for v in means if isinstance(v, int | float)]
            return (
                float(max(values or [0])),
                max(numeric_means or [-1.0]),
                float(item.get("rank", (0,))[0]) if item.get("rank") else 0.0,
            )

        current = best.get(ticker)
        if current is None or rank_key(row) > rank_key(current):
            best[ticker] = row
    return best


def evaluate_criteria(
    prereg: Mapping[str, Any],
    lab_payload: Mapping[str, Any] | None,
    *,
    promote_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Mechanical verdict. No discretion, no LLM, no post-hoc criteria edits."""
    criteria = dict(prereg.get("criteria") or {})
    kind = str(criteria.get("kind", "improvement"))
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    if lab_payload is None:
        return {
            "passed": False,
            "outcome": "blocked",
            "reasons": ["no lab result was produced"],
            "metrics": metrics,
        }

    rows = list(lab_payload.get("results") or [])
    metrics["rows"] = len(rows)
    metrics["candidates"] = len(lab_payload.get("candidates") or [])
    metrics["summary"] = lab_payload.get("summary") or {}

    if criteria_version(criteria) >= 2:
        return _evaluate_criteria_v2(prereg, lab_payload, promote_payload=promote_payload)

    if kind == "diagnostic":
        diagnostic_per_ticker: dict[str, Any] = {}
        targets_seen: set[str] = set()
        for row in rows:
            ticker = str(row.get("ticker") or "unknown")
            target_stats: dict[str, Any] = {}
            for name, value in (row.get("targets") or {}).items():
                if not isinstance(value, Mapping):
                    continue
                targets_seen.add(name)
                target_stats[name] = {
                    "folds_beating": value.get("folds_beating"),
                    "folds": value.get("folds"),
                    "mean_improvement": _number(value.get("mean_improvement")),
                    "min_improvement": _number(value.get("min_improvement")),
                    "metric": value.get("metric"),
                }
            economic = (row.get("economic") or {}).get("model") or {}
            if not isinstance(economic, Mapping):
                economic = {}
            diagnostic_per_ticker[ticker] = {
                "targets": target_stats,
                "economic": {
                    "total_return": _number(economic.get("total_return")),
                    "sharpe": _number(economic.get("sharpe")),
                    "trades": _number(economic.get("trades")),
                },
            }
        metrics["per_ticker"] = diagnostic_per_ticker
        metrics["targets"] = sorted(targets_seen)
        passed = bool(rows)
        if not passed:
            reasons.append("diagnostic produced no rows")
        return {
            "passed": passed,
            "outcome": "diagnostic" if passed else "blocked",
            "reasons": reasons,
            "metrics": metrics,
        }

    target = str(criteria.get("target", "fwd_direction"))
    best = _best_rows_by_ticker(lab_payload)
    per_ticker: dict[str, Any] = {}
    beating: list[str] = []
    for ticker, row in sorted(best.items()):
        stats = (row.get("targets") or {}).get(target)
        if not isinstance(stats, Mapping):
            per_ticker[ticker] = {"absent": True}
            reasons.append(f"{ticker}: target {target} missing from the lab result")
            continue
        folds_beating = int(stats.get("folds_beating") or 0)
        folds = int(stats.get("folds") or criteria.get("folds", 0) or 0)
        mean_improvement = _number(stats.get("mean_improvement"))
        min_improvement = _number(stats.get("min_improvement"))
        ok = (
            folds_beating >= int(criteria.get("folds_beating_min", 3))
            and mean_improvement is not None
            and mean_improvement >= float(criteria.get("mean_improvement_min", 0.0))
            and min_improvement is not None
            and min_improvement >= float(criteria.get("min_improvement_min", 0.0))
        )
        per_ticker[ticker] = {
            "folds_beating": folds_beating,
            "folds": folds,
            "mean_improvement": mean_improvement,
            "min_improvement": min_improvement,
            "beats": ok,
        }
        if ok:
            beating.append(ticker)

    metrics["target"] = target
    metrics["per_ticker"] = per_ticker
    metrics["tickers_beating"] = len(beating)
    metrics["tickers"] = sorted(best)
    need = int(criteria.get("tickers_beating_min", 2))
    if len(beating) < need:
        reasons.append(
            f"only {len(beating)}/{len(best)} tickers met the improvement bar "
            f"(need {need}): {', '.join(sorted(beating)) or 'none'}"
        )

    economic = criteria.get("economic")
    if economic:
        row = next(iter(lab_payload.get("top") or []), None) or _first_row(best)
        model = ((row or {}).get("economic") or {}).get("model") or {}
        momentum = ((row or {}).get("economic") or {}).get("momentum") or {}
        total_return = _number(model.get("total_return"))
        sharpe = _number(model.get("sharpe"))
        momentum_sharpe = _number(momentum.get("sharpe"))
        trades = _number(model.get("trades"))
        metrics["economic"] = {
            "total_return": total_return,
            "sharpe": sharpe,
            "momentum_sharpe": momentum_sharpe,
            "trades": trades,
            "ticker": (row or {}).get("ticker"),
        }
        if total_return is None or total_return <= float(economic.get("total_return_min", 0.0)):
            reasons.append("after-cost total return is not positive")
        if economic.get("sharpe_above_momentum") and (
            sharpe is None or momentum_sharpe is None or sharpe <= momentum_sharpe
        ):
            reasons.append("Sharpe does not beat the momentum baseline")
        if trades is None or trades < float(economic.get("trades_min", 100)):
            reasons.append(
                f"only {int(trades or 0)} trades (need {economic.get('trades_min', 100)})"
            )

    if criteria.get("requires_promote"):
        if promote_payload is None:
            reasons.append("promotion dry run was required but not run")
        elif not promote_payload.get("passed"):
            gate = promote_payload.get("reasons") or ["promotion gate failed"]
            reasons.append("promotion gate failed: " + "; ".join(str(r) for r in gate))
        else:
            metrics["promotion"] = {
                "passed": True,
                "candidate": promote_payload.get("candidate"),
            }

    passed = not reasons
    return {
        "passed": passed,
        "outcome": "provisional" if passed else "negative",
        "reasons": reasons,
        "metrics": metrics,
    }


def _paired_rule(
    pooled: Mapping[str, Any] | None,
    *,
    alpha: float,
    fold_rule: str,
) -> tuple[bool, list[str]]:
    """The v2 statistical rule, applied identically to the candidate and controls."""
    reasons: list[str] = []
    if not isinstance(pooled, Mapping):
        return False, ["no pooled paired result"]
    mean = _number(pooled.get("mean_improvement"))
    ci_low = _number(pooled.get("ci_low"))
    n = int(pooled.get("n") or 0)
    if n < 2:
        reasons.append("fewer than 2 paired observations")
    if mean is None or mean <= 0.0:
        reasons.append("pooled paired improvement is not positive")
    if ci_low is None:
        reasons.append("bootstrap CI is unavailable")
    elif ci_low <= 0.0:
        reasons.append(f"bootstrap CI lower bound {ci_low:.6f} is not above zero")
    folds = pooled.get("fold_improvements") or []
    positive = [
        float(value)
        for value in folds
        if isinstance(value, (int, float)) and float(value) > 0.0
    ]
    if fold_rule == "all_positive" and folds and len(positive) != len(folds):
        reasons.append(f"only {len(positive)}/{len(folds)} folds are positive")
    if fold_rule == "majority_positive" and folds and len(positive) <= len(folds) / 2:
        reasons.append(f"only {len(positive)}/{len(folds)} folds are positive")
    if pooled.get("alpha") is not None and float(pooled["alpha"]) > alpha + 1e-12:
        reasons.append("the lab CI was not computed at the pre-registered alpha")
    return not reasons, reasons


def _config_name(primary: Any) -> str | None:
    if isinstance(primary, str) and primary:
        return primary
    if isinstance(primary, Mapping):
        return (
            f"{primary.get('featureSet') or 'all'}+"
            f"{primary.get('modelSet') or 'reference'}+"
            f"h{int(primary.get('horizon') or 1)}"
        )
    return None


def _pooled_control(payload: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    controls = payload.get("controls") or {}
    block = controls.get(name)
    return block if isinstance(block, Mapping) else None


def _evaluate_criteria_v2(
    prereg: Mapping[str, Any],
    lab_payload: Mapping[str, Any],
    *,
    promote_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pooled paired improvement + time-block bootstrap + null controls.

    Version 1 selected the best row per ticker and counted tickers as votes.
    Version 2 judges one pre-registered config on the pooled paired
    improvement; the per-ticker distribution is reported but never becomes a
    vote. The shuffled/shifted-label controls are the leakage detector: if a
    null also clears the same rule, the run is ``invalid``, not negative.
    """
    criteria = dict(prereg.get("criteria") or {})
    kind = str(criteria.get("kind", "improvement"))
    alpha = float(criteria.get("alpha", 0.05))
    fold_rule = str(criteria.get("fold_rule", "all_positive"))
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "criteria_version": 2,
        "alpha": alpha,
        "trials_in_family": int(prereg.get("trial_index") or 1),
        "fold_rule": fold_rule,
        "rows": len(lab_payload.get("results") or []),
        "summary": lab_payload.get("summary") or {},
    }

    expected_primary = _config_name(criteria.get("primary"))
    actual_primary = lab_payload.get("primary")
    if expected_primary and actual_primary and str(actual_primary) != expected_primary:
        return {
            "passed": False,
            "outcome": "blocked",
            "reasons": [
                f"lab evaluated {actual_primary!r}, not the pre-registered "
                f"primary {expected_primary!r}"
            ],
            "metrics": metrics,
        }

    if lab_payload.get("diagnostic_full_archive") and kind != "diagnostic":
        return {
            "passed": False,
            "outcome": "blocked",
            "reasons": ["a full-archive diagnostic cannot feed a verdict"],
            "metrics": metrics,
        }

    if kind == "diagnostic":
        rows = list(lab_payload.get("results") or [])
        metrics["per_ticker"] = {
            str(row.get("ticker") or "unknown"): {
                "targets": row.get("targets") or {},
                "economic": (row.get("economic") or {}).get("model") or {},
            }
            for row in rows
        }
        metrics["targets"] = sorted(
            {
                name
                for row in rows
                for name in (row.get("targets") or {})
            }
        )
        passed = bool(rows)
        if not passed:
            reasons.append("diagnostic produced no rows")
        return {
            "passed": passed,
            "outcome": "diagnostic" if passed else "blocked",
            "reasons": reasons,
            "metrics": metrics,
        }

    if lab_payload.get("canary"):
        # The canary is deliberately run with shuffled labels: it must fail.
        pooled = lab_payload.get("pooled")
        passed, rule_reasons = _paired_rule(pooled, alpha=alpha, fold_rule=fold_rule)
        metrics["canary"] = True
        metrics["pooled"] = pooled
        if passed:
            return {
                "passed": True,
                "outcome": "provisional",
                "reasons": ["canary reached provisional: shuffled labels produced skill"],
                "metrics": metrics,
            }
        return {
            "passed": False,
            "outcome": "negative",
            "reasons": ["canary negative as required"],
            "metrics": metrics,
        }

    pooled = lab_payload.get("pooled")
    metrics["pooled"] = pooled
    passed, rule_reasons = _paired_rule(pooled, alpha=alpha, fold_rule=fold_rule)
    reasons.extend(rule_reasons)

    # Controls are mandatory for an improvement claim. A passing null means the
    # pipeline can manufacture skill from noise, so the run is void.
    if criteria.get("require_controls", True):
        missing = [
            name for name in ("shuffle", "shift") if _pooled_control(lab_payload, name) is None
        ]
        if missing:
            return {
                "passed": False,
                "outcome": "blocked",
                "reasons": [f"missing control(s): {', '.join(missing)}"],
                "metrics": metrics,
            }
        nulls: dict[str, Any] = {}
        for name in ("shuffle", "shift"):
            block = _pooled_control(lab_payload, name) or {}
            control_pooled = block.get("pooled") or block
            control_passed, _ = _paired_rule(
                control_pooled, alpha=alpha, fold_rule=fold_rule
            )
            nulls[name] = {
                "pooled": control_pooled,
                "passed": control_passed,
                "shift_bars": block.get("shift_bars"),
            }
            if control_passed:
                return {
                    "passed": False,
                    "outcome": "invalid",
                    "reasons": [
                        f"{name} control beat the same rule: the measurement is leaking"
                    ],
                    "metrics": {**metrics, "controls": nulls},
                }
        metrics["controls"] = nulls

    # Per-ticker distribution: reported, but not counted as independent votes.
    per_ticker: dict[str, Any] = {}
    for row in lab_payload.get("results") or []:
        ticker = str(row.get("ticker") or "unknown")
        paired = (row.get("paired") or {}) if isinstance(row, Mapping) else {}
        per_ticker[ticker] = {
            "n": paired.get("n"),
            "mean_improvement": _number(paired.get("mean_improvement")),
            "ci_low": _number(paired.get("ci_low")),
            "ci_high": _number(paired.get("ci_high")),
        }
    metrics["per_ticker"] = per_ticker
    positive_tickers = [
        name
        for name, stats in per_ticker.items()
        if (stats.get("mean_improvement") or 0.0) > 0.0
    ]
    metrics["tickers_positive"] = sorted(positive_tickers)
    single = len(per_ticker) > 1 and len(positive_tickers) < 2
    metrics["carried_by_one_ticker"] = single
    if single and criteria.get("require_multiple_tickers", True):
        reasons.append(
            "the result is carried by a single ticker (correlated assets are not votes)"
        )

    economic = criteria.get("economic")
    if economic:
        row = next(iter(lab_payload.get("results") or []), None) or {}
        model = ((row.get("economic") or {}).get("model") or {}) if isinstance(row, Mapping) else {}
        momentum = (
            ((row.get("economic") or {}).get("momentum") or {})
            if isinstance(row, Mapping)
            else {}
        )
        total_return = _number(model.get("total_return"))
        sharpe = _number(model.get("sharpe"))
        momentum_sharpe = _number(momentum.get("sharpe"))
        trades = _number(model.get("trades"))
        metrics["economic"] = {
            "total_return": total_return,
            "sharpe": sharpe,
            "momentum_sharpe": momentum_sharpe,
            "trades": trades,
            "ticker": (row or {}).get("ticker"),
        }
        if total_return is None or total_return <= float(economic.get("total_return_min", 0.0)):
            reasons.append("after-cost total return is not positive")
        if economic.get("sharpe_above_momentum") and (
            sharpe is None or momentum_sharpe is None or sharpe <= momentum_sharpe
        ):
            reasons.append("Sharpe does not beat the momentum baseline")
        if trades is None or trades < float(economic.get("trades_min", 100)):
            reasons.append(
                f"only {int(trades or 0)} trades (need {economic.get('trades_min', 100)})"
            )

    if criteria.get("requires_promote"):
        if promote_payload is None:
            reasons.append("promotion dry run was required but not run")
        elif not promote_payload.get("passed"):
            gate = promote_payload.get("reasons") or ["promotion gate failed"]
            reasons.append("promotion gate failed: " + "; ".join(str(r) for r in gate))
        else:
            metrics["promotion"] = {
                "passed": True,
                "candidate": promote_payload.get("candidate"),
            }

    passed = not reasons
    return {
        "passed": passed,
        "outcome": "provisional" if passed else "negative",
        "reasons": reasons,
        "metrics": metrics,
    }


def _first_row(best: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next(iter(best.values()), None)


# --------------------------------------------------------------------------- rendering


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:+.3f}"
    return str(value)


def render_hypotheses(
    queue: Sequence[Hypothesis],
    entries: Sequence[Mapping[str, Any]],
) -> str:
    done = completed_hypotheses(entries)
    verdicts = {
        str(e.get("hypothesis_id")): e
        for e in entries
        if e.get("kind") == "verdict"
    }
    lines = [
        "# Hypothesis queue",
        "",
        "<!-- Generated by `tools/model_improve.py report --render`; edit hypotheses.json. -->",
        "",
        "| id | order | family | status | needs | trials | last verdict | mechanism |",
        "| -- | ----- | ------ | ------ | ----- | ------ | ------------ | --------- |",
    ]
    counts = trial_counts(entries)["hypothesis"]
    for hypothesis in queue:
        verdict = verdicts.get(hypothesis.id)
        status = hypothesis.status
        if hypothesis.id in done and status == "open":
            status = f"done ({verdict.get('outcome')})" if verdict else "done"
        template = (
            "| {id} | {order} | {family} | {status} | {needs} | {trials} "
            "| {verdict} | {mech} |"
        )
        lines.append(
            template.format(
                id=hypothesis.id,
                order=hypothesis.order,
                family=hypothesis.family,
                status=status,
                needs=",".join(hypothesis.needs) or "—",
                trials=counts.get(hypothesis.id, 0),
                verdict=(verdict or {}).get("outcome", "—"),
                mech=hypothesis.mechanism.replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_scoreboard(entries: Sequence[Mapping[str, Any]]) -> str:
    verdicts = [e for e in entries if e.get("kind") == "verdict"]
    prereg = [e for e in entries if e.get("kind") == "preregister"]
    outcomes = [str(entry.get("outcome") or "") for entry in verdicts]
    provisional = outcomes.count("provisional")
    confirmed = outcomes.count("confirmed")
    refuted = outcomes.count("refuted")
    invalid = outcomes.count("invalid")
    void = harness_void(entries)
    lines = [
        "# Model-improvement scoreboard",
        "",
        "<!-- Generated from experiments/ledger.jsonl; never hand-edit. -->",
        "",
        f"- ledger entries: **{len(entries)}**",
        f"- preregistrations: **{len(prereg)}**",
        f"- verdicts: **{len(verdicts)}**",
        f"- provisional: **{provisional}** | confirmed: **{confirmed}**"
        f" | refuted: **{refuted}** | invalid: **{invalid}**",
        f"- refuted-to-provisional: **"
        f"{('n/a' if provisional == 0 else f'{refuted / provisional:.2f}')}**",
        f"- harness: **{'VOID' if void else 'ok'}**",
        "",
        "A negative verdict is a completed experiment: it closes that hypothesis",
        "lineage. Re-running it needs new data, a new feature class, or a fixed",
        "measurement bug, and counts as a new trial in the same family.",
        "",
        "| id | at | hypothesis | family | outcome | trial | evidence |",
        "| -- | -- | ---------- | ------ | ------- | ----- | -------- |",
    ]
    for entry in verdicts:
        metrics = entry.get("metrics") or {}
        if metrics.get("target"):
            evidence = (
                f"{metrics.get('tickers_beating', 0)}/{len(metrics.get('tickers') or [])} "
                f"tickers, target {metrics['target']}"
            )
        elif metrics.get("rows") is not None:
            evidence = f"{metrics['rows']} rows"
        else:
            evidence = "; ".join(entry.get("reasons") or [])[:80] or "—"
        lines.append(
            "| {id} | {at} | {hid} | {family} | {outcome} | {trial} | {evidence} |".format(
                id=entry.get("id", ""),
                at=str(entry.get("at", ""))[:10],
                hid=entry.get("hypothesis_id", ""),
                family=entry.get("family", ""),
                outcome=entry.get("outcome", ""),
                trial=entry.get("trial_index", ""),
                evidence=evidence.replace("|", "/"),
            )
        )
    if not verdicts:
        lines.append("| — | — | — | — | — | — | no verdicts yet |")
    lines.append("")
    return "\n".join(lines)


def write_if_changed(path: Path, content: str) -> bool:
    """Write generated docs only when the content changed (clean CI diffs)."""
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True
