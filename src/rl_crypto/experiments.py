"""Append-only experiment ledger with trial and holdout accounting.

The lab can measure, but a measurement is not evidence unless it is tied to a
pre-registered hypothesis, the code that ran, and the exact data it read. This
module owns that record:

* ``experiments/ledger.jsonl`` — one JSON object per line, append-only. It is
  committed to the repo, so an experiment is reproducible from a commit plus
  the hashes in its entry.
* every entry carries the hypothesis, its family and lineage, the trial index
  within the family, the git SHA, a per-ticker hash of the archive used, a hash
  of the config, and the seed.
* ``holdout_touched`` is set only by ``promote``, and
  :func:`check_promote_allowed` refuses a lineage whose final holdout has
  already been spent on a different candidate.

The rules live here rather than in the TypeScript wrapper so they cannot be
bypassed by calling the Python engine directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "rl-crypto/experiment/1"

#: Entry kinds. ``preregister`` states the test before it runs; ``result``
#: records what happened; ``holdout`` records a final-holdout touch.
KINDS: tuple[str, ...] = ("preregister", "result", "holdout")

#: Verdict lifecycle from the model-improvement command.
VERDICTS: tuple[str, ...] = ("negative", "provisional", "confirmed", "refuted", "blocked")

_CORE_FIELDS: tuple[str, ...] = (
    "schema",
    "at",
    "kind",
    "hypothesis_id",
    "family",
    "lineage_id",
    "trial_index",
    "git_sha",
    "data_hash",
    "config_hash",
    "seed",
    "holdout_touched",
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ledger_path(root: str | Path) -> Path:
    """The committed ledger path for a repo root."""
    return Path(root) / "experiments" / "ledger.jsonl"


def scoreboard_path(root: str | Path) -> Path:
    return Path(root) / "experiments" / "SCOREBOARD.md"


# --------------------------------------------------------------------------- hashing


def canonical_hash(payload: Any) -> str:
    """Stable SHA-256 of a JSON-serialisable payload (sorted keys, no whitespace)."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def config_hash(configs: Any) -> str:
    return canonical_hash(configs)


def file_hash(path: str | Path) -> str | None:
    """SHA-256 of a file's bytes, or ``None`` when it does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_hashes(root: str | Path, tickers: Iterable[str]) -> dict[str, str | None]:
    """The archive hash per ticker, so an entry names the data it actually read."""
    base = Path(root) / "data"
    out: dict[str, str | None] = {}
    for ticker in tickers:
        slug = ticker.strip().replace("/", "-").replace(" ", "-").lower()
        out[ticker] = file_hash(base / slug / "candles.jsonl")
    return out


def git_sha(root: str | Path) -> str | None:
    """HEAD of the repo containing ``root``, or ``None`` outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else None


# --------------------------------------------------------------------------- ledger io


def read_entries(root: str | Path) -> list[dict[str, Any]]:
    """All ledger entries in order; blank and malformed lines are skipped."""
    path = ledger_path(root)
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def validate(entry: Mapping[str, Any]) -> list[str]:
    """Field-level problems with ``entry``; empty list means it is well formed."""
    problems = [f"missing {name}" for name in _CORE_FIELDS if name not in entry]
    if entry.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    if entry.get("kind") not in KINDS:
        problems.append(f"kind must be one of {', '.join(KINDS)}")
    if not isinstance(entry.get("data_hash"), dict):
        problems.append("data_hash must be a mapping of ticker -> hash")
    if not isinstance(entry.get("seed"), int):
        problems.append("seed must be an integer")
    if not isinstance(entry.get("trial_index"), int) or entry.get("trial_index", 0) < 1:
        problems.append("trial_index must be an integer >= 1")
    return problems


def append(root: str | Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and append one entry. Raises ``ValueError`` on a malformed entry."""
    payload = dict(entry)
    payload.setdefault("schema", SCHEMA)
    payload.setdefault("at", _now())
    payload.setdefault("git_sha", git_sha(root))
    payload.setdefault("seed", 0)
    payload.setdefault("holdout_touched", False)
    problems = validate(payload)
    if problems:
        raise ValueError("invalid ledger entry: " + "; ".join(problems))
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    return payload


# --------------------------------------------------------------------------- trials


def trial_index(entries: Sequence[Mapping[str, Any]], family: str) -> int:
    """1-based trial number for the next pre-registration in ``family``."""
    prior = sum(
        1
        for entry in entries
        if entry.get("kind") == "preregister" and entry.get("family") == family
    )
    return prior + 1


def lineage_trial_index(entries: Sequence[Mapping[str, Any]], lineage_id: str) -> int:
    """The trial index recorded for a lineage's pre-registration, defaulting to 1."""
    for entry in entries:
        if entry.get("kind") == "preregister" and entry.get("lineage_id") == lineage_id:
            value = entry.get("trial_index")
            if isinstance(value, int):
                return value
    return 1


def holdout_touches(
    entries: Sequence[Mapping[str, Any]], lineage_id: str
) -> list[Mapping[str, Any]]:
    """Entries that spent the final holdout on ``lineage_id``."""
    return [
        entry
        for entry in entries
        if entry.get("kind") == "holdout"
        and entry.get("lineage_id") == lineage_id
        and entry.get("holdout_touched") is True
    ]


def check_promote_allowed(
    entries: Sequence[Mapping[str, Any]],
    *,
    lineage_id: str,
    candidate_hash: str,
    apply: bool,
) -> str | None:
    """Why ``promote`` must refuse, or ``None`` when it may proceed.

    The holdout is spent at most once per lineage. A dry run spends it; an
    ``--apply`` for the *same candidate* after a passing dry run is the same
    touch and is allowed. A different candidate in the same lineage is refused —
    iterating after a failure requires a new lineage (and a new trial).
    """
    earlier = [
        entry
        for entry in entries
        if entry.get("kind") == "holdout" and entry.get("lineage_id") == lineage_id
    ]
    if not earlier:
        return None
    if any(entry.get("candidate_hash") != candidate_hash for entry in earlier):
        return (
            f"lineage {lineage_id!r} already spent its final holdout on a different "
            "candidate; open a new lineage (new trial) instead"
        )
    if apply:
        if any(entry.get("apply") and entry.get("passed") for entry in earlier):
            return f"lineage {lineage_id!r}: this candidate has already been applied"
        if not any(entry.get("apply") is False and entry.get("passed") for entry in earlier):
            return (
                f"lineage {lineage_id!r}: --apply requires a prior passing dry run "
                "of the same candidate"
            )
    return None


# --------------------------------------------------------------------------- record builders


def preregister(
    root: str | Path,
    *,
    hypothesis_id: str,
    family: str,
    lineage_id: str,
    mechanism: str,
    configs: Sequence[Mapping[str, Any]],
    tickers: Sequence[str],
    targets: Sequence[str],
    metric: str,
    criteria: str,
    negative: str,
    seed: int = 0,
    folds: int = 4,
) -> dict[str, Any]:
    """Write the pre-registration for one hypothesis before any experiment runs."""
    entries = read_entries(root)
    entry = {
        "schema": SCHEMA,
        "at": _now(),
        "kind": "preregister",
        "hypothesis_id": hypothesis_id,
        "family": family,
        "lineage_id": lineage_id,
        "trial_index": trial_index(entries, family),
        "git_sha": git_sha(root),
        "data_hash": data_hashes(root, tickers),
        "config_hash": config_hash(configs),
        "seed": int(seed),
        "holdout_touched": False,
        "mechanism": mechanism,
        "configs": [dict(config) for config in configs],
        "tickers": list(tickers),
        "targets": list(targets),
        "metric": metric,
        "criteria": criteria,
        "negative": negative,
        "folds": int(folds),
    }
    return append(root, entry)


def record_result(
    root: str | Path,
    *,
    hypothesis_id: str,
    lineage_id: str,
    verdict: str,
    artifact: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Record the outcome of a pre-registered run and its verdict."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
    entries = read_entries(root)
    prev = next(
        (e for e in reversed(entries) if e.get("lineage_id") == lineage_id),
        None,
    )
    if prev is None:
        raise ValueError(f"lineage {lineage_id!r} was never pre-registered")
    entry = {
        "schema": SCHEMA,
        "at": _now(),
        "kind": "result",
        "hypothesis_id": hypothesis_id,
        "family": prev.get("family"),
        "lineage_id": lineage_id,
        "trial_index": prev.get("trial_index", 1),
        "git_sha": git_sha(root),
        "data_hash": prev.get("data_hash", {}),
        "config_hash": prev.get("config_hash"),
        "seed": prev.get("seed", 0),
        "holdout_touched": False,
        "verdict": verdict,
        "artifact": artifact,
        "metrics": dict(metrics or {}),
        "notes": list(notes),
    }
    return append(root, entry)


def record_holdout(
    root: str | Path,
    *,
    hypothesis_id: str,
    lineage_id: str,
    candidate: str,
    candidate_hash: str,
    passed: bool,
    apply: bool,
    reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Record a final-holdout touch, after :func:`check_promote_allowed` passes."""
    entries = read_entries(root)
    prev = next(
        (e for e in reversed(entries) if e.get("lineage_id") == lineage_id),
        None,
    )
    first_touch = not holdout_touches(entries, lineage_id)
    entry = {
        "schema": SCHEMA,
        "at": _now(),
        "kind": "holdout",
        "hypothesis_id": hypothesis_id,
        "family": (prev or {}).get("family", "adhoc"),
        "lineage_id": lineage_id,
        "trial_index": lineage_trial_index(entries, lineage_id),
        "git_sha": git_sha(root),
        "data_hash": (prev or {}).get("data_hash", {}),
        "config_hash": (prev or {}).get("config_hash"),
        "seed": (prev or {}).get("seed", 0),
        # A dry run after a dry run is not a second touch; only the first is.
        "holdout_touched": first_touch,
        "candidate": candidate,
        "candidate_hash": candidate_hash,
        "passed": bool(passed),
        "apply": bool(apply),
        "reasons": list(reasons),
    }
    return append(root, entry)


# --------------------------------------------------------------------------- reporting


def counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Trial, lineage and holdout totals, per family."""
    families: dict[str, dict[str, int]] = {}
    for entry in entries:
        family = str(entry.get("family") or "adhoc")
        bucket = families.setdefault(
            family,
            {"trials": 0, "results": 0, "holdouts": 0},
        )
        kind = entry.get("kind")
        if kind == "preregister":
            bucket["trials"] += 1
        elif kind == "result":
            bucket["results"] += 1
        elif kind == "holdout" and entry.get("holdout_touched"):
            bucket["holdouts"] += 1
    return {
        "entries": len(entries),
        "families": families,
        "holdout_touches": sum(
            1
            for entry in entries
            if entry.get("kind") == "holdout" and entry.get("holdout_touched")
        ),
        "lineages": len({entry.get("lineage_id") for entry in entries}),
    }


def _short(value: Any, length: int = 12) -> str:
    if value is None:
        return "—"
    text = str(value)
    return text[:length]


def verdict_for_lineage(
    entries: Sequence[Mapping[str, Any]], lineage_id: str
) -> str | None:
    """The newest recorded verdict for a lineage."""
    for entry in reversed(entries):
        if entry.get("lineage_id") == lineage_id and entry.get("kind") == "result":
            verdict = entry.get("verdict")
            if isinstance(verdict, str):
                return verdict
    return None


def render_scoreboard(entries: Sequence[Mapping[str, Any]]) -> str:
    """Render ``experiments/SCOREBOARD.md`` from the ledger (never hand-edited)."""
    totals = counts(entries)
    lines = [
        "# Experiment scoreboard",
        "",
        "Generated from `experiments/ledger.jsonl` by "
        "`python tools/model_lab.py scoreboard`. Do not edit by hand.",
        "",
        f"- entries: {totals['entries']}",
        f"- lineages: {totals['lineages']}",
        f"- final-holdout touches: {totals['holdout_touches']}",
        "",
    ]
    if not entries:
        lines += ["No experiments recorded yet.", ""]
        return "\n".join(lines)

    lines += [
        "## Families",
        "",
        "| family | trials | results | holdout touches |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, bucket in sorted(totals["families"].items()):
        lines.append(
            f"| {family} | {bucket['trials']} | {bucket['results']} | {bucket['holdouts']} |"
        )
    lines += [
        "",
        "## Lineages",
        "",
        "| lineage | hypothesis | trial | verdict | holdout | git | data |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    seen: set[str] = set()
    for entry in entries:
        lineage = str(entry.get("lineage_id") or "adhoc")
        if lineage in seen:
            continue
        seen.add(lineage)
        holdout = holdout_touches(entries, lineage)
        data = entry.get("data_hash") or {}
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                lineage,
                entry.get("hypothesis_id", "—"),
                entry.get("trial_index", "—"),
                verdict_for_lineage(entries, lineage) or "preregistered",
                "used" if holdout else "unused",
                _short(entry.get("git_sha"), 8),
                _short(next(iter(data.values()), None), 12),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_scoreboard(root: str | Path, entries: Sequence[Mapping[str, Any]] | None = None) -> Path:
    """Regenerate ``experiments/SCOREBOARD.md`` from the ledger."""
    path = scoreboard_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_scoreboard(entries or read_entries(root)), encoding="utf-8")
    return path


def resolve_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Repo root from an explicit value or the current directory."""
    return Path(value or os.environ.get("RL_CRYPTO_ROOT") or Path.cwd()).resolve()


__all__ = [
    "KINDS",
    "SCHEMA",
    "VERDICTS",
    "append",
    "canonical_hash",
    "check_promote_allowed",
    "config_hash",
    "counts",
    "data_hashes",
    "file_hash",
    "git_sha",
    "holdout_touches",
    "ledger_path",
    "lineage_trial_index",
    "preregister",
    "read_entries",
    "record_holdout",
    "record_result",
    "render_scoreboard",
    "resolve_root",
    "scoreboard_path",
    "trial_index",
    "validate",
    "verdict_for_lineage",
    "write_scoreboard",
]
