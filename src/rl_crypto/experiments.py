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
import math
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "rl-crypto/experiments/1"
HYPOTHESIS_STATUSES = frozenset({
    "open",
    "blocked",
    "done",
    "retired",
    "deferred-underpowered",
})
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

#: Canary verdicts live in their own append-only file, keyed by protected hash.
#: They are harness checks, never trials, so they do not belong in the ledger's
#: trial accounting.
CANARY_FILENAME = "canary.jsonl"
RUN_TYPES = frozenset({"experiment", "harness-change"})

#: Criteria interpretation version. Version 2 keeps the percentile block
#: bootstrap it was written against; version 3 uses the corrected
#: effective-sample t interval. Ledger entries carry their own version, so an
#: old entry is never retroactively re-judged.
CRITERIA_VERSION = 3

#: Alpha spending across a hypothesis family. Trial ``i`` spends
#: ``alpha / (i * (i + 1))``; the series sums to ``alpha`` over every trial, so a
#: family of any length cannot exceed its declared false-positive budget. The
#: old rule (``alpha / trial_index``) has a divergent sum and a family that runs
#: ten trials silently spends ~2.9x alpha.
SPENDING_SCHEDULE = "alpha/(i*(i+1))"

#: How many trials a family may run before it must be retired and a new family
#: (with its own error budget) opened. The cap bounds the smallest per-trial
#: alpha, which in turn bounds the bootstrap resolution required below.
FAMILY_TRIAL_CAP_DEFAULT = 10

#: Stages whose failure is infrastructure, not a measurement. A run that only
#: produced a pre-registration and one of these blocks never evaluated the
#: hypothesis, so it must not spend alpha or a family trial.
INFRA_BLOCK_STAGES = frozenset(
    {
        "preflight",
        "lead",
        "researcher",
        "implementer",
        "auditor",
        "budget",
        "staleness",
        "timeout",
    }
)

#: Reason fragments that identify an infrastructure block in older ledger
#: entries, which predate the explicit ``infra`` flag.
INFRA_REASON_MARKERS = (
    "no lab result was produced",
    "insufficient-resolution",
    "insufficient-blocks",
    "timeout",
    "timed out",
    "credential",
    "opencode refused",
    "harness-void",
    "role ",
)

#: Minimum number of time blocks for a percentile CI. Eight blocks made the
#: interval anti-conservative: the effective sample size of a block bootstrap is
#: the number of blocks, and eight is too few for a 2.5% tail. Twenty-four is
#: six blocks per fold at the default four folds, and matches the default
#: 24-bar block width used by the lab.
MIN_BOOTSTRAP_BLOCKS = 24

#: Files whose content defines the measurement contract. A normal experiment run
#: may not edit them: doing so would let an implementer move the goalposts it is
#: being judged against. ``run_type: harness-change`` is the only escape hatch,
#: and it cannot produce a scoreboard-eligible verdict.
#:
#: The *definition* of this list is pinned in ``experiments/protected-files.pin``
#: and checked by ``tests/test_protected_pin.py``. Adding or removing a file is
#: therefore a two-file change (list + pin) that a human must review: it is the
#: moment the measurement contract itself changes.
PROTECTED_FILES: tuple[str, ...] = (
    "src/rl_crypto/experiments.py",
    "src/rl_crypto/power.py",
    "src/rl_crypto/structural.py",
    "src/rl_crypto/targets.py",
    "src/rl_crypto/samples.py",
    "src/rl_crypto/backtest.py",
    "src/rl_crypto/ml/tier1.py",
    "src/rl_crypto/ml/training_config.py",
    "tools/model_lab.py",
    "tools/model_improve.py",
    "tools/forward_score.py",
    "tests/test_experiments.py",
    "tests/test_power.py",
    "tests/test_structural.py",
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
#: A v2 claim about direction or return must survive after-cost economics on the
#: pooled portfolio. These are the defaults; a criteria block may tighten them.
POOLED_ECONOMIC_DEFAULT: dict[str, Any] = {
    "total_return_min": 0.0,
    "sharpe_above_momentum": True,
    "trades_min": 100,
}

#: Targets whose v2 verdict always includes the pooled economic gate.
ECONOMIC_TARGETS = frozenset({"fwd_direction", "fwd_return"})

#: Pre-registered control distribution. A real pooled improvement must exceed
#: this quantile of the label-destroying controls; the median is reported.
CONTROL_QUANTILE_DEFAULT = 0.95
CONTROL_SHIFTS_DEFAULT: tuple[int, ...] = (169, 173, 181, 191, 337)
CONTROL_BLOCK_BARS_DEFAULT = 24
CONTROL_REPLICATES_DEFAULT = 6

DEFAULT_CRITERIA_V2: dict[str, Any] = {
    "kind": "improvement",
    "criteria_version": CRITERIA_VERSION,
    "target": "fwd_direction",
    "primary": None,
    "alpha": 0.05,
    "fold_rule": "all_positive",
    "require_controls": True,
    "control_quantile": CONTROL_QUANTILE_DEFAULT,
    "control_shifts": list(CONTROL_SHIFTS_DEFAULT),
    "control_block_bars": CONTROL_BLOCK_BARS_DEFAULT,
    "control_replicates": CONTROL_REPLICATES_DEFAULT,
    "baseline": "strongest",
    "require_multiple_tickers": True,
    # Economics is not a default v2 gate: a pooled paired CI is the verdict,
    # and promotion has its own after-cost gate. A hypothesis may opt in.
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


def alpha_for_trial(alpha: float, trial_index: int) -> float:
    """Per-trial alpha under the declared family spending schedule."""
    index = max(1, int(trial_index or 1))
    return float(alpha) / (index * (index + 1))


def smallest_alpha(alpha: float, family_cap: int) -> float:
    """The smallest per-trial alpha reachable before the family-size cap."""
    return alpha_for_trial(alpha, max(1, int(family_cap)))


def required_bootstrap_samples(alpha_trial: float) -> int:
    """Resamples needed to resolve the two-sided ``alpha_trial`` tail quantile.

    A percentile bootstrap can only resolve quantiles at ``1 / B`` resolution, so
    a two-sided tail of ``alpha/2`` needs at least ``B * alpha/2 >= 2`` order
    statistics to place both the lower and upper bound. Below that the CI is a
    smoothed guess, not a measured quantile. The floor keeps ordinary CIs stable.
    """
    tail = max(float(alpha_trial), 1e-12) / 2.0
    return max(1000, math.ceil(2.0 / tail))


def bootstrap_resolution(
    alpha: float, family_cap: int, bootstrap_samples: int
) -> dict[str, Any]:
    """Whether the configured resample count can resolve the legacy tail.

    The active verdict no longer uses a bootstrap quantile (it uses the
    effective-sample t interval), so this check is informational: the legacy
    percentile comparison still needs enough resamples, while the gate itself
    blocks on ``insufficient-blocks`` from the lab.
    """
    from .power import MIN_EFFECTIVE_BLOCKS

    smallest = smallest_alpha(alpha, family_cap)
    required = required_bootstrap_samples(smallest)
    return {
        "schedule": SPENDING_SCHEDULE,
        "family_cap": int(family_cap),
        "alpha_family": float(alpha),
        "smallest_alpha": smallest,
        "required_samples": required,
        "configured_samples": int(bootstrap_samples),
        "ok": int(bootstrap_samples) >= required,
        "min_blocks": MIN_EFFECTIVE_BLOCKS,
        "legacy_min_blocks": MIN_BOOTSTRAP_BLOCKS,
    }


def protected_list_digest() -> str:
    """SHA-256 of the protected-file list definition (the pin's value)."""
    return hashlib.sha256(
        "\n".join(sorted(PROTECTED_FILES)).encode("utf-8")
    ).hexdigest()


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
    #: "null" is the shuffled-label canary; "positive" injects a leaky feature.
    canary_kind: str = "null"
    #: The smallest pooled improvement worth acting on, in criterion units.
    minimal_meaningful_effect: float | None = None
    #: "baseline" compares against the strongest baseline; "champion" against
    #: the retrained deployed bundle.
    comparison: str = "baseline"
    #: Why a hypothesis is deferred or blocked in plain language.
    status_reason: str = ""
    #: The condition under which a deferred hypothesis is re-evaluated.
    reevaluate_when: str = ""
    #: A one-line justification for a hand-set non-economic MME.
    minimal_meaningful_effect_justification: str = ""
    #: Design scaling assumptions for a panel/history hypothesis.
    design: dict[str, Any] = field(default_factory=dict)

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
            "canary_kind": self.canary_kind,
            "minimal_meaningful_effect": self.minimal_meaningful_effect,
            "minimal_meaningful_effect_justification": (
                self.minimal_meaningful_effect_justification
            ),
            "comparison": self.comparison,
            "status_reason": self.status_reason,
            "reevaluate_when": self.reevaluate_when,
            "design": dict(self.design),
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
        canary_kind = str(payload.get("canary_kind") or "null")
        if canary_kind not in {"null", "positive"}:
            raise ValueError(f"{payload.get('id')}: unknown canary_kind {canary_kind!r}")
        comparison = str(payload.get("comparison") or "baseline")
        if comparison not in {"baseline", "champion"}:
            raise ValueError(f"{payload.get('id')}: unknown comparison {comparison!r}")
        raw_effect = payload.get("minimal_meaningful_effect")
        try:
            minimal_effect = None if raw_effect is None else float(raw_effect)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{payload.get('id')}: minimal_meaningful_effect must be numeric"
            ) from exc
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
            canary_kind=canary_kind,
            minimal_meaningful_effect=minimal_effect,
            comparison=comparison,
            status_reason=str(payload.get("status_reason") or payload.get("reason") or ""),
            reevaluate_when=str(payload.get("reevaluate_when") or ""),
            minimal_meaningful_effect_justification=str(
                payload.get("minimal_meaningful_effect_justification") or ""
            ),
            design=dict(payload.get("design") or {}),
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


def read_canary(path: Path) -> list[dict[str, Any]]:
    """Read the append-only canary record (same JSONL shape as the ledger)."""
    return read_ledger(path)


def append_canary(path: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Append one canary verdict; canaries are never trials."""
    return append_entry(path, {"kind": "verdict", **dict(entry)})


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


def infra_kind(entry: Mapping[str, Any] | str | None, stage: str | None = None) -> str | None:
    """Classify an infrastructure block by its reason, or ``None``.

    The distinction matters for accounting: a credential failure, a role
    timeout, a stale archive and an unresolvable bootstrap CI all leave the
    hypothesis untested. Only a run that reached a lab ``results`` entry spends a
    trial. New entries carry ``infra: true`` and ``infra_kind`` explicitly;
    older entries are classified from their stage and reason so the historical
    ledger is interpreted consistently without being rewritten.
    """
    if isinstance(entry, Mapping):
        if entry.get("kind") != "blocked":
            return None
        if "infra_kind" in entry:
            return str(entry["infra_kind"] or "infra")
        if "infra" in entry:
            return "infra" if entry.get("infra") else None
        stage = str(entry.get("stage") or "") or stage
        reason = str(entry.get("reason") or "")
    else:
        reason = str(entry or "")
    text = reason.lower()
    if "insufficient-resolution" in text:
        return "insufficient-resolution"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "stale" in text or "no-new-information" in text:
        return "staleness"
    if (
        "credential" in text
        or "opencode refused" in text
        or "harness-void" in text
    ):
        return "role-error"
    if stage and str(stage) in INFRA_BLOCK_STAGES:
        roles = {"lead", "researcher", "implementer", "auditor"}
        return "role-error" if str(stage) in roles else str(stage)
    if any(marker in text for marker in INFRA_REASON_MARKERS):
        return "infra"
    return None


def is_infra_block(entry: Mapping[str, Any]) -> bool:
    """Whether a ledger entry records an infrastructure failure, not a result."""
    return infra_kind(entry) is not None


def consumed_runs(entries: Sequence[Mapping[str, Any]]) -> set[str]:
    """Runs that reached a lab result or a scored verdict (public wrapper)."""
    return _consumed_runs(entries)


def _consumed_runs(entries: Sequence[Mapping[str, Any]]) -> set[str]:
    """Runs that reached a lab result or a scored verdict."""
    runs: set[str] = set()
    for entry in entries:
        kind = entry.get("kind")
        run_id = str(entry.get("run_id") or "")
        if not run_id:
            continue
        if (kind == "results" and entry.get("lab", True) is not False) or kind == "verdict":
            runs.add(run_id)
    return runs


def trial_counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Trials actually consumed per family and per hypothesis.

    A pre-registration is a promise, not a trial. It consumes one only when the
    run reached a lab ``results`` entry (or a scored verdict). A run stopped by
    an infrastructure failure — a role/credential error, a timeout, a stale or
    unchanged archive, or an unresolvable CI — is recorded as a blocked entry
    with ``infra: true`` and does not consume the family's alpha budget.
    """
    consumed = _consumed_runs(entries)
    infra_runs = {
        str(entry.get("run_id"))
        for entry in entries
        if entry.get("run_id") and is_infra_block(entry)
    }
    by_family: dict[str, int] = {}
    by_hypothesis: dict[str, int] = {}
    for entry in entries:
        if entry.get("kind") != "preregister":
            continue
        # A canary is a harness check, not a trial: it never spends family alpha.
        if entry.get("canary") or str(entry.get("family") or "") == "canary":
            continue
        run_id = str(entry.get("run_id") or "")
        # Very old entries predate ``run_id``. Without a run to match, the safe
        # assumption is that the pre-registration did consume a trial.
        if run_id and (run_id not in consumed or run_id in infra_runs):
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
            void = dict(entry)
        elif entry.get("kind") == "harness_clear":
            void = None
    return void


def _canary_hash(entry: Mapping[str, Any]) -> str | None:
    direct = entry.get("protected_hash")
    if direct:
        return str(direct)
    protected = entry.get("protected") or {}
    if isinstance(protected, Mapping) and protected.get("hash"):
        return str(protected["hash"])
    return None


def canary_status(
    entries: Sequence[Mapping[str, Any]],
    *,
    current_hash: str | None = None,
    canary_entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Freshness of the two harness canaries against the current contract.

    A canary result is only meaningful for the measurement code that produced
    it: a pass at an older ``PROTECTED_FILES`` hash proves nothing about the
    code now running. ``null`` must come out ``negative`` (shuffled labels did
    not produce skill) and ``positive`` must come out ``invalid`` (the leak was
    detected). Both must be at ``current_hash`` when one is supplied.
    """
    kinds: dict[str, dict[str, Any]] = {}
    latest_at: str | None = None
    combined = list(entries) + list(canary_entries or [])
    for entry in combined:
        kind_value = entry.get("kind")
        if kind_value not in (None, "verdict") or not entry.get("canary"):
            continue
        kind = str(entry.get("canary_kind") or "null")
        at = str(entry.get("at") or "")
        if latest_at is None or at >= latest_at:
            latest_at = at
        kinds[kind] = {
            "outcome": entry.get("outcome"),
            "passed": entry.get("passed"),
            "at": at,
            "hash": _canary_hash(entry),
            "run_id": entry.get("run_id"),
        }

    def ok(name: str) -> bool:
        block = kinds.get(name)
        if not block:
            return False
        if current_hash and block.get("hash") != current_hash:
            return False
        if not current_hash and not block.get("hash"):
            return False
        if name == "null":
            return block.get("outcome") == "negative" and not block.get("passed")
        return block.get("outcome") == "invalid"

    hashes = {block.get("hash") for block in kinds.values() if block.get("hash")}
    hash_value = current_hash or (next(iter(hashes)) if len(hashes) == 1 else None)
    passed = ok("null") and ok("positive") and bool(hash_value)
    return {
        "pass": passed,
        "hash": hash_value,
        "date": (latest_at or "")[:10],
        "at": latest_at,
        "kinds": kinds,
        "missing": [name for name in ("null", "positive") if name not in kinds],
        "stale": [
            name
            for name, block in kinds.items()
            if current_hash and block.get("hash") != current_hash
        ],
    }


def canary_gate(
    entries: Sequence[Mapping[str, Any]],
    current_hash: str | None,
    canary_entries: Sequence[Mapping[str, Any]] | None = None,
) -> str | None:
    """Why an experiment run may not proceed, or ``None`` when fresh."""
    if not current_hash:
        return "cannot hash the protected measurement files; refusing to run"
    status = canary_status(
        entries, current_hash=current_hash, canary_entries=canary_entries
    )
    if status["pass"]:
        return None
    missing = status.get("missing") or []
    stale = status.get("stale") or []
    detail = []
    if missing:
        detail.append("missing " + ", ".join(missing))
    if stale:
        detail.append("stale at current hash: " + ", ".join(stale))
    if not detail:
        detail.append("a canary did not produce its required outcome")
    return (
        "canary-freshness: "
        + "; ".join(detail)
        + f" (current protected hash {current_hash[:12]})"
    )


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
            "inconclusive",
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
    power: Mapping[str, Any] | None = None,
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
        "spending_schedule": (
            SPENDING_SCHEDULE
            if criteria_version(hypothesis.criteria) >= 2
            else "fixed-alpha"
        ),
        "alpha_family": float(hypothesis.criteria.get("alpha", 0.05)),
        "alpha_trial": (
            alpha_for_trial(
                float(hypothesis.criteria.get("alpha", 0.05)),
                counts["family"].get(hypothesis.family, 0) + 1,
            )
            if criteria_version(hypothesis.criteria) >= 2
            else float(hypothesis.criteria.get("alpha", 0.05))
        ),
        "trigger": trigger,
        "seed": seed,
        "run_type": run_type or hypothesis.run_type,
        "canary": bool(hypothesis.canary),
        "canary_kind": hypothesis.canary_kind,
        "minimal_meaningful_effect": hypothesis.minimal_meaningful_effect,
        "minimal_meaningful_effect_justification": (
            hypothesis.minimal_meaningful_effect_justification
        ),
        "design": dict(hypothesis.design),
        "comparison": hypothesis.comparison,
        "power": dict(power or {}),
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
            if criteria_version(criteria) >= 3:
                target = str(criteria.get("target") or "")
                declared = entry.get("minimal_meaningful_effect")
                justification = str(
                    entry.get("minimal_meaningful_effect_justification") or ""
                ).strip()
                if target not in ECONOMIC_TARGETS:
                    if declared is None:
                        problems.append(
                            "criteria v3 non-economic target needs a hand-set "
                            "minimal_meaningful_effect"
                        )
                    elif not justification:
                        problems.append(
                            "criteria v3 hand-set minimal_meaningful_effect needs "
                            "a one-line justification"
                        )
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
        elif not (promote_payload.get("champion_comparison") or {}).get("passed"):
            reasons.append(
                "promotion requires a passing pooled paired comparison over the "
                "champion retrained on the pre-cutoff slice"
            )
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


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial rate.

    Used by the calibration commands: with 300 null draws, the empirical
    false-positive rate must be compared with the nominal alpha with its own
    uncertainty, not as a point estimate.
    """
    n = max(0, int(trials))
    if n == 0:
        return (0.0, 1.0)
    k = min(max(0, int(successes)), n)
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    low = max(0.0, centre - half)
    high = min(1.0, centre + half)
    return (0.0 if low < 1e-12 else low, 1.0 if high > 1.0 - 1e-12 else high)


def calibration_verdict(
    successes: int, trials: int, nominal_alpha: float
) -> dict[str, Any]:
    """Compare an empirical rate with a nominal alpha using its own CI."""
    low, high = wilson_interval(successes, trials)
    rate = (successes / trials) if trials else 0.0
    return {
        "successes": int(successes),
        "trials": int(trials),
        "rate": rate,
        "ci_low": low,
        "ci_high": high,
        "nominal_alpha": float(nominal_alpha),
        "exceeds": low > float(nominal_alpha),
    }


def paired_rule(
    pooled: Mapping[str, Any] | None,
    *,
    alpha: float,
    fold_rule: str,
    version: int = CRITERIA_VERSION,
) -> tuple[bool, list[str]]:
    """Public alias for the paired rule (used by calibration commands)."""
    return _paired_rule(
        pooled, alpha=alpha, fold_rule=fold_rule, version=version
    )


def contract_violation(lab_payload: Mapping[str, Any] | None) -> str | None:
    """A deterministic provenance failure, or ``None`` when the contracts hold.

    The gate is the lab's own **structural** result, not its self-description:
    truncation invariance (no feature/target reads a bar it could not have
    known) and perturbation invariance (holdout rows cannot move the fitted
    scaler, feature selection or coefficients). ``preprocessing.fit_scope`` and
    ``selection.scope`` are retained as documentation only; they no longer
    decide anything, because a payload can claim ``"train"`` while fitting on
    everything.

    A label reach that crosses the split (``purge``) is still a hard failure:
    that is a mechanical fact about the split, not a self-report.
    """
    if not isinstance(lab_payload, Mapping):
        return None
    purge = lab_payload.get("purge")
    if isinstance(purge, Mapping) and purge.get("ok") is False:
        violations = purge.get("violations") or []
        detail = ""
        if violations:
            first = violations[0]
            if isinstance(first, Mapping):
                detail = f" ({first.get('reason') or first})"
        return "label reach crosses the train/test split (purge violation)" + detail
    structural = lab_payload.get("structural")
    if isinstance(structural, Mapping) and structural.get("ok") is False:
        details: list[str] = []
        truncation = structural.get("truncation")
        if isinstance(truncation, Mapping) and truncation.get("ok") is False:
            count = truncation.get("violation_count") or len(
                truncation.get("violations") or []
            )
            first = (truncation.get("violations") or [None])[0]
            hint = ""
            if isinstance(first, Mapping):
                what = first.get("name") or first.get("kind")
                hint = f" ({what} at row {first.get('row')})"
            details.append(f"truncation invariance failed on {count} row(s){hint}")
        perturbation = structural.get("perturbation")
        if isinstance(perturbation, Mapping) and perturbation.get("ok") is False:
            first = (perturbation.get("violations") or [None])[0]
            path = first.get("path") if isinstance(first, Mapping) else None
            details.append(
                "perturbation invariance failed: the holdout changed fitted "
                f"parameters{(' at ' + str(path)) if path else ''}"
            )
        return "structural causality check failed: " + "; ".join(details)
    return None


def pooled_interval_for_version(
    pooled: Mapping[str, Any] | None, version: int
) -> Mapping[str, Any] | None:
    """The CI a criteria version was written against.

    Version 2 keeps the percentile block-bootstrap interval it was
    pre-registered with. Version 3+ uses the corrected effective-sample t
    interval. The lab computes both so the record is comparable.
    """
    if not isinstance(pooled, Mapping):
        return pooled
    if int(version) >= 3:
        return pooled
    low = pooled.get("legacy_percentile_ci_low")
    high = pooled.get("legacy_percentile_ci_high")
    if low is None and high is None:
        return pooled
    return {**pooled, "ci_low": low, "ci_high": high, "interval": "percentile"}


def _resolution_block(
    pooled: Mapping[str, Any] | None, *, version: int = CRITERIA_VERSION
) -> str | None:
    """A blocked reason when the CI cannot be certified for this design."""
    if not isinstance(pooled, Mapping):
        return None
    if int(version) >= 3:
        if pooled.get("resolution_ok") is True:
            return None
        if pooled.get("insufficient_blocks"):
            detail = pooled.get("resolution_reason") or "too few effective blocks"
            return f"insufficient-blocks: {detail}"
        if pooled.get("insufficient"):
            detail = pooled.get("resolution_reason") or "too few rows"
            return f"insufficient-resolution: {detail}"
        return None
    # v2 keeps the percentile rule's own sufficiency test.
    legacy = pooled.get("legacy_resolution_ok")
    if legacy is True:
        return None
    if legacy is False:
        detail = pooled.get("legacy_resolution_reason") or "legacy interval unresolved"
        return f"insufficient-resolution: {detail}"
    if legacy is None and pooled.get("insufficient"):
        detail = pooled.get("resolution_reason") or "interval unresolved"
        return f"insufficient-resolution: {detail}"
    return None


def _paired_rule(
    pooled: Mapping[str, Any] | None,
    *,
    alpha: float,
    fold_rule: str,
    version: int = CRITERIA_VERSION,
) -> tuple[bool, list[str]]:
    pooled = pooled_interval_for_version(pooled, version)
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
    version = criteria_version(criteria)
    alpha_family = float(criteria.get("alpha", 0.05))
    trial_index = int(prereg.get("trial_index") or 1)
    alpha = alpha_for_trial(alpha_family, trial_index)
    fold_rule = str(criteria.get("fold_rule", "all_positive"))
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "criteria_version": version,
        "alpha": alpha,
        "alpha_family": alpha_family,
        "alpha_trial": alpha,
        "spending_schedule": SPENDING_SCHEDULE,
        "trials_in_family": trial_index,
        "fold_rule": fold_rule,
        "minimal_meaningful_effect": _number(
            prereg.get("minimal_meaningful_effect")
            if prereg.get("minimal_meaningful_effect") is not None
            else criteria.get("minimal_meaningful_effect")
        ),
        "comparison": str(prereg.get("comparison") or "baseline"),
        "power": dict(prereg.get("power") or {}),
        "rows": len(lab_payload.get("results") or []),
        "summary": lab_payload.get("summary") or {},
        # Self-reported provenance is kept for the record, not for the gate:
        # the structural result below is what decides.
        "documented_contracts": {
            "preprocessing": lab_payload.get("preprocessing"),
            "selection": lab_payload.get("selection"),
        },
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

    violation = contract_violation(lab_payload)
    if violation is not None:
        return {
            "passed": False,
            "outcome": "invalid",
            "reasons": [violation],
            "metrics": metrics,
        }

    if lab_payload.get("leak_detected"):
        leaks = lab_payload.get("leak_features") or []
        detail = ", ".join(
            f"{item.get('feature')}~{item.get('target')} r={item.get('correlation'):.3f}"
            for item in leaks
            if isinstance(item, Mapping)
        )
        return {
            "passed": False,
            "outcome": "invalid",
            "reasons": [
                "a feature is a deterministic function of a target (leak)"
                + (f": {detail}" if detail else "")
            ],
            "metrics": {**metrics, "leak_features": leaks},
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
        blocked = _resolution_block(pooled, version=version)
        if blocked is not None:
            return {
                "passed": False,
                "outcome": "blocked",
                "reasons": [blocked],
                "metrics": {**metrics, "pooled": pooled},
            }
        passed, rule_reasons = _paired_rule(
            pooled, alpha=alpha, fold_rule=fold_rule, version=version
        )
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

    comparison = str(prereg.get("comparison") or "baseline")
    if comparison == "champion" and str(lab_payload.get("comparison") or "baseline") != "champion":
        return {
            "passed": False,
            "outcome": "blocked",
            "reasons": [
                "the hypothesis requires a champion comparison but the lab "
                "reported a baseline comparison"
            ],
            "metrics": metrics,
        }
    pooled = lab_payload.get("pooled")
    metrics["pooled"] = pooled
    metrics["comparison"] = comparison
    blocked = _resolution_block(pooled, version=version)
    if blocked is not None:
        return {
            "passed": False,
            "outcome": "blocked",
            "reasons": [blocked],
            "metrics": metrics,
        }
    passed, rule_reasons = _paired_rule(
        pooled, alpha=alpha, fold_rule=fold_rule, version=version
    )
    reasons.extend(rule_reasons)

    # Controls are mandatory for an improvement claim. A passing *shuffle* is
    # the one result that can only mean the pipeline manufactures skill from
    # noise; shift/permute controls define the null distribution below.
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
        for name in ("shuffle", "shift", "permute"):
            block = _pooled_control(lab_payload, name)
            if block is None:
                continue
            variants = block.get("by_shift") or block.get("by_seed") or [block]
            for variant in variants:
                control_pooled = variant.get("pooled") or variant
                control_passed, _ = _paired_rule(
                    control_pooled,
                    alpha=alpha,
                    fold_rule=fold_rule,
                    version=version,
                )
                label = (
                    f"{name}:{variant.get('shift_bars') or variant.get('seed')}"
                    if variant is not block
                    else name
                )
                nulls[label] = {
                    "pooled": control_pooled,
                    "passed": control_passed,
                    "shift_bars": variant.get("shift_bars"),
                    "seed": variant.get("seed"),
                }
                # A globally shuffled label is the one control that can only
                # pass if the measurement manufactures skill from noise. Shift
                # and block-permutation controls preserve the target's own
                # autocorrelation, so for a persistent target (fwd_volume_ratio)
                # they legitimately carry skill; they form the null
                # distribution the candidate must clear, not an automatic void.
                if control_passed and name == "shuffle":
                    return {
                        "passed": False,
                        "outcome": "invalid",
                        "reasons": [
                            f"{label} control beat the same rule: the measurement "
                            "is leaking"
                        ],
                        "metrics": {**metrics, "controls": nulls},
                    }
        metrics["controls"] = nulls

        # The control distribution is the real null: the candidate must clear a
        # pre-registered quantile of it, not merely any single control. A daily
        # or weekly-multiple shift used to echo the target's own seasonality, so
        # one weak control could not certify anything.
        distribution = _pooled_control(lab_payload, "distribution")
        if isinstance(distribution, Mapping):
            quantile = float(criteria.get("control_quantile", CONTROL_QUANTILE_DEFAULT))
            quantile_value = _number(distribution.get("quantile_value"))
            median = _number(distribution.get("median"))
            metrics["control_distribution"] = {
                "n": distribution.get("n"),
                "median": median,
                "quantile": quantile,
                "quantile_value": quantile_value,
                "max": _number(distribution.get("max")),
            }
            candidate_mean = (
                _number(pooled.get("mean_improvement"))
                if isinstance(pooled, Mapping)
                else None
            )
            if candidate_mean is None or quantile_value is None:
                reasons.append("control distribution is incomplete")
            elif candidate_mean <= quantile_value:
                reasons.append(
                    f"pooled improvement {candidate_mean:.6f} does not exceed the "
                    f"{quantile:.2f} control quantile {quantile_value:.6f} "
                    f"(control median {median:.6f})"
                )

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

    target_name = str(criteria.get("target", "fwd_direction"))
    mandatory_economics = target_name in ECONOMIC_TARGETS
    economic = dict(criteria.get("economic") or {})
    if mandatory_economics:
        economic = {**POOLED_ECONOMIC_DEFAULT, **economic}
    pooled_economics = lab_payload.get("pooled_economics")
    if mandatory_economics and not isinstance(pooled_economics, Mapping):
        return {
            "passed": False,
            "outcome": "blocked",
            "reasons": [
                f"v2 target {target_name!r} requires pooled economics, but the lab "
                "did not report a pooled portfolio"
            ],
            "metrics": metrics,
        }
    if economic:
        if isinstance(pooled_economics, Mapping):
            model = pooled_economics.get("model") or {}
            momentum = pooled_economics.get("momentum") or {}
            basis = "pooled-portfolio"
            ticker = None
        else:
            row = next(iter(lab_payload.get("results") or []), None) or {}
            model = (
                ((row.get("economic") or {}).get("model") or {})
                if isinstance(row, Mapping)
                else {}
            )
            momentum = (
                ((row.get("economic") or {}).get("momentum") or {})
                if isinstance(row, Mapping)
                else {}
            )
            basis = "first-row"
            ticker = (row or {}).get("ticker")
        total_return = _number(model.get("total_return"))
        sharpe = _number(model.get("sharpe"))
        momentum_sharpe = _number(momentum.get("sharpe"))
        trades = _number(model.get("trades"))
        metrics["economic"] = {
            "basis": basis,
            "total_return": total_return,
            "sharpe": sharpe,
            "momentum_sharpe": momentum_sharpe,
            "trades": trades,
            "ticker": ticker,
            "mandatory": mandatory_economics,
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

    # The MME turns "the CI cleared zero" into "the CI cleared an effect worth
    # acting on". A CI whose upper bound is below the MME is genuinely negative;
    # anything that still overlaps both zero and the MME is inconclusive.
    mme = metrics.get("minimal_meaningful_effect")
    if isinstance(pooled, Mapping):
        metrics["ci_low"] = _number(pooled.get("ci_low"))
        metrics["ci_high"] = _number(pooled.get("ci_high"))
        metrics["mean_improvement"] = _number(pooled.get("mean_improvement"))
    passed = not reasons
    if passed and mme is not None:
        ci_low = metrics.get("ci_low")
        if ci_low is None or ci_low < float(mme):
            reasons.append(
                f"the CI lower bound {ci_low} does not certify the minimal "
                f"meaningful effect {float(mme):.6f}"
            )
            passed = False
    outcome = _outcome_from_interval(
        metrics.get("ci_low"), metrics.get("ci_high"), mme, passed
    )
    return {
        "passed": passed,
        "outcome": outcome,
        "reasons": reasons,
        "metrics": metrics,
    }


def _outcome_from_interval(
    ci_low: Any, ci_high: Any, minimal_meaningful_effect: Any, passed: bool
) -> str:
    """Classify a result into ``provisional`` / ``inconclusive`` / ``negative``.

    ``negative`` is only claimed when the CI upper bound is below the minimal
    meaningful effect (or below zero when none is declared). A CI that still
    overlaps zero or the MME cannot support either a positive or a negative
    claim, so it is ``inconclusive``.
    """
    high = _number(ci_high)
    low = _number(ci_low)
    if high is None:
        return "inconclusive"
    threshold = (
        float(minimal_meaningful_effect)
        if isinstance(minimal_meaningful_effect, int | float)
        else 0.0
    )
    if high < threshold:
        return "negative"
    if passed and low is not None and low >= threshold:
        return "provisional"
    return "inconclusive"


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


def render_scoreboard(
    entries: Sequence[Mapping[str, Any]],
    canary_entries: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    verdicts = [e for e in entries if e.get("kind") == "verdict"]
    prereg = [e for e in entries if e.get("kind") == "preregister"]
    outcomes = [str(entry.get("outcome") or "") for entry in verdicts]
    provisional = outcomes.count("provisional")
    confirmed = outcomes.count("confirmed")
    refuted = outcomes.count("refuted")
    invalid = outcomes.count("invalid")
    inconclusive = outcomes.count("inconclusive")
    void = harness_void(entries)
    canary = canary_status(entries, canary_entries=canary_entries)
    if canary["pass"]:
        canary_line = f"- canary: **pass @ {canary['hash']} {canary['date']}**"
    elif canary["hash"]:
        canary_line = (
            f"- canary: **MISSING/STALE @ {canary['hash']} {canary['date']}**"
        )
    else:
        canary_line = "- canary: **MISSING**"
    lines = [
        "# Model-improvement scoreboard",
        "",
        "<!-- Generated from experiments/ledger.jsonl; never hand-edit. -->",
        "",
        f"- ledger entries: **{len(entries)}**",
        f"- preregistrations: **{len(prereg)}**",
        f"- verdicts: **{len(verdicts)}**",
        f"- provisional: **{provisional}** | confirmed: **{confirmed}**"
        f" | refuted: **{refuted}** | invalid: **{invalid}**"
        f" | inconclusive: **{inconclusive}**",
        f"- refuted-to-provisional: **"
        f"{('n/a' if provisional == 0 else f'{refuted / provisional:.2f}')}**",
        f"- harness: **{'VOID' if void else 'ok'}**",
        canary_line,
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
