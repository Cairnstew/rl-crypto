---
name: pipeline-hardening
description: >-
  Use when auditing or improving the rl-crypto production pipeline — its
  modularity, cleanliness, or hardening. Triggers: "audit the pipeline",
  "pipeline modularity", "pipeline cleanup", "harden the pipeline",
  "rl-pipeline", "pipeline refactor", "run.py", "live.py", "conflicts.py",
  "sync.py", "why did the pipeline exit 0", "partial failure", "cache write
  safety", "pipeline CI gaps". Runs an offline architecture/cleanliness/
  hardening audit, classifies each finding, maps it to a fix, and appends
  durable findings to this file so the next review starts from accumulated
  knowledge instead of zero.
---

# Pipeline modularity, cleanliness & hardening

This skill audits the **code that writes `data/` and `models/`**, not the data
itself. It is two things:

1. a repeatable review of the production pipeline's architecture, boundaries,
   contracts and failure modes; and
2. the **living memory** — the `Memory` sections hold what previous reviews
   proved. Read them first and update them after. A fresh review that ignores
   them is a regression.

Companion skill: **`data-quality-audit`** audits the resulting cache. Use that
one when the symptom is bad data; use this one when the symptom is code that
writes it badly, hides a failure, or is hard to change.

## 0. Contract

- **Read-only first.** The default review runs tests/lint/type-check, reads the
  code, and makes no network or `data/` change.
- **Behaviour before style.** A refactor that changes an exit code, a cache
  file, a workflow invariant, or a public CLI is a behaviour change and needs
  its own finding, test, and commit.
- **Do not break the cache contracts.** The archive is an upsert series, live
  records are append-capped, `summary.json` is overwritten, models are
  committed JSON. See `data/README.md` and `models/README.md`.
- **Do not break the single-writer invariant.** Matrix jobs stage artifacts;
  only the merge job pushes. `concurrency: rl-crypto-cache` is load-bearing.
- **Do not remove `os._exit` without solving its reason.** It exists because a
  timed-out fetch can leave a worker thread behind.
- **No fabricated evidence.** Every finding carries a dated command or a
  `path:line`; "feels tangled" is not a finding.
- **A fix is not part of an audit-only run.** When asked to improve, make the
  smallest durable fix, test it, and record it. Otherwise stop at the mapping.

## 1. Architecture map

| Layer | Modules | Owns | Must not |
| --- | --- | --- | --- |
| **CLI / dispatch** | `src/rl_crypto/pipeline/cli.py`, `__main__.py` | argparse tree, subcommand wiring, exit-code normalisation, hard exit | import heavy domain code at module scope; contain runner logic |
| **Runners** | `pipeline/run.py`, `pipeline/live.py` | fetch → archive → train → live; per-ticker loop; summary | do raw git/cache I/O (delegate to `cache.py`) |
| **Maintenance** | `pipeline/conflicts.py`, `pipeline/sync.py` | rebase conflict resolution; fast-forward sync | parse market/news data; train |
| **Cache I/O** | `src/rl_crypto/cache.py`, `export.py` | paths, JSONL read/write, candle upsert, live cap, timeout wrapper, `sync_cache` | know about argparse or ML |
| **Domain / analysis** | `dataset.py`, `unify.py`, `features.py`, `targets.py`, `samples.py`, `models.py`, `tickers.py`, `config.py` | fetching, fusion, features/labels, schema types, watchlist | import `rl_crypto.pipeline` |
| **ML** | `ml/{tier1,tier2,serve,store,types,base,linear}.py` | train, outlook, bundle load/save | import pipeline |
| **Compatibility** | `tools/{run_pipeline,record_live,resolve_cache_conflicts}.py` | deprecated one-release shims | gain new behaviour |
| **Operator shell** | `tools/record_live.sh` | local live timer (legacy/local) | be the production path |
| **CI / workflows** | `.github/workflows/{ci,production,watchdog}.yml` | offline gates; scheduler; single-writer merge; watchdog | duplicate runner logic in YAML |

**Dependency direction:** `cli → pipeline.{run,live,conflicts,sync} → cache/domain/ml`.
Domain and ML modules must never import `pipeline`. A function-local import
that closes a cycle is debt, not a design (see PL-006).

**Command protocol (today, informal):**

| Command | Handler | Returns |
| --- | --- | --- |
| `run` | `run.run` | `int` exit status |
| `live` | `live.run` | `int` exit status |
| `conflicts` | `conflicts.run` | **number of resolved paths** (not a status) |
| `sync` | `sync.run` | `int` exit status |
| `version` | `cli._cmd_version` | `0` |

The documented convention is `0` ok, `1` partial/failed, `2` usage
(`cli.py:11-18`). `conflicts` violates it; `run` violates it on a partial
multi-ticker failure. These are PL-001/PL-002, not style.

**Data contracts that must survive any refactor:**

- `data/<slug>/candles.jsonl` — one bar per canonical `…Z` timestamp, upsert,
  never shrinks, never duplicates.
- `data/<slug>/live.jsonl` — append-only, capped at `LIVE_CAP_DEFAULT = 1000`.
- `data/<slug>/summary.json` — schema `rl-crypto/pipeline/3`, overwritten.
- `models/<slug>.json` — schema `rl-crypto/ml/1`, `feature_version` checked on
  load; an unservable bundle loses a rebase conflict to a servable one.
- `--root` / `RL_CRYPTO_ROOT`, defaulting to the caller's cwd (never
  `__file__`: hermetic installs live in the Nix store).

## 2. Environment

Run everything offline inside the dev shell:

```bash
nix develop --command bash -c 'uv run python -m pytest -m "not live" -q'
nix develop --command bash -c 'uv run ruff check src/ tests/ examples/ tools/'
nix develop --command bash -c 'uv run basedpyright'
nix develop --command bash -c 'uv run rl-pipeline --help'
nix develop --command bash -c 'uv run rl-pipeline <command> --help'
```

`uv` is not on the host `PATH`. Prefer the dev shell; do not bake store paths
into this file. If `.venv/bin/python` is used directly, its numpy/pandas wheels
need ELF64 `libz`/`libstdc++` (the store also holds ELF32) — another reason to
use `nix develop`.

## 3. Procedure

### 3.1 Pin the run

```bash
date -u +%FT%TZ
git rev-parse --short HEAD
git log -10 --oneline -- src/rl_crypto/pipeline .github/workflows tools/record_live.sh
```

Read the `Memory` sections. Record the baseline even when nothing changed.

### 3.2 Baseline

Record:

- test count and result (`pytest`), `ruff` result, `basedpyright` result;
- the command list from `rl-pipeline --help`;
- module count and the pipeline module sizes;
- the import graph and any cycles;
- the workflow invariants you can see (single schedule, matrix, merge writer,
  concurrency group, retry count);
- the current cache footprint (cross-check `data-quality-audit` for the numbers).

**Import-graph probe** (stdlib only; run from the repo root):

```bash
python3 - <<'PY'
import ast, collections, pathlib

root = pathlib.Path("src/rl_crypto")
mods = {
    str(p.relative_to(root))[:-3].replace("/", "."): p
    for p in root.rglob("*.py")
    if "__pycache__" not in p.parts
}
edges = collections.defaultdict(set)
for name, path in mods.items():
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.ImportFrom):
            if node.level:
                # Resolve a relative import against this module's package.
                pkg = name.split(".")[:-1]
                keep = len(pkg) - (node.level - 1)
                base = ".".join(pkg[: max(keep, 0)])
                target = f"{base}.{node.module}" if node.module else base
                target = target.strip(".")
            else:
                target = node.module or ""
                if target.startswith("rl_crypto."):
                    target = target.removeprefix("rl_crypto.")
            targets = [target] if target else []
        elif isinstance(node, ast.Import):
            targets = [
                a.name.removeprefix("rl_crypto.")
                for a in node.names
                if a.name.startswith("rl_crypto.")
            ]
        for t in targets:
            if t in mods:
                edges[name].add(t)

def cycles(start, cur, path, seen):
    for nxt in edges.get(cur, ()):
        if nxt == start:
            return path + [nxt]
        if nxt not in seen:
            seen.add(nxt)
            found = cycles(start, nxt, path + [nxt], seen)
            if found:
                return found
    return None

found = []
for node in edges:
    cycle = cycles(node, node, [node], {node})
    if cycle:
        found.append(cycle)

print("modules:", len(mods))
print("cycles:", found or "none")
for name in sorted(edges):
    if name.startswith("pipeline.") or name in ("cache", "dataset", "unify"):
        print(f"{name} -> {', '.join(sorted(edges[name]))}")
PY
```

Note: an import inside a function still appears here. A cycle closed by a
function-local import is the exact PL-006 shape; report it and name the lazy
import.

### 3.3 Modularity audit

Ask, in order:

1. **Does the layer graph point one way?** `cli → pipeline → cache/domain/ml`.
   Any domain/ML module importing `pipeline` is a leak. Any cycle is debt.
2. **Is a command a thin adapter?** Each command should own argument parsing and
   one orchestration call. If a command parses, fetches, writes, trains, and
   prints, it is doing too much.
3. **Is there one command protocol?** Compare return types and exit semantics
   across `run`/`live`/`conflicts`/`sync`. Divergence is a contract bug, not a
   preference.
4. **Is duplication mechanical or semantic?** Four copies of `ROOT`,
   `add_arguments`, `parse_args`, `main`, and `--root` are mechanical (PL-007).
   A second copy of `slug_for` in YAML is semantic and is already pinned by a
   test (`test_workflows.py:69-78`); do not remove the guard.
5. **Can a new stage be added without editing five files?** A new pipeline stage
   should be one module + one `cli.py` wiring + one test, not a spread across
   runners and shims.
6. **Are test seams honest?** The suite monkeypatches module-level functions
   (`runner.build_dataset`, `runner.build_unified`, `tool.fetch_prices`,
   `tool.build_unified_many`). A refactor that moves those names breaks tests
   for no behavioural reason; either keep the seam or update the tests in the
   same commit with a clear reason.
7. **Is compatibility code isolated?** `tools/*.py` must only delegate. If a
   shim grows logic, the real fix is in `pipeline/`.

### 3.4 Cleanliness audit

- **Dead or legacy paths:** deprecated shims still in use by
  `tools/record_live.sh`; README still mentions the old local timer while
  `data/README.md` says collection is remote. Flag docs drift.
- **Constants single-source:** `LIVE_CAP_DEFAULT` lives in `cache.py`;
  `TAIL_HOURS_DEFAULT` in `live.py`; fetch timeouts are argparse defaults. A
  new default must not be re-typed in a workflow or shell script.
- **Schema/field consistency:** `run` records and `live` records must carry the
  same timestamp field. Today `live` adds `_generated_at` (PL-008); the resolver
  treats it as a fallback key (`conflicts.py:39`), the audit checks
  `generated_at` only.
- **Error/notes conventions:** partial feedback belongs in `notes` (a tuple of
  human strings) and in the summary; a silent `except` is never acceptable.
  Broad `except Exception` is allowed only at an isolation boundary, and must
  say what it caught.
- **Console vs data:** stdout is for humans; the summary is the machine record.
  Do not make a workflow parse a human line.
- **Naming:** `run`/`run_pipeline`, `resolve`/`conflicts`, `--ticker`/`--tickers`
  are stable public surfaces. Rename only with a compatibility shim and a test.
- **Docs:** every schema/exit-code/behaviour change updates `README.md`,
  `data/README.md`, or the module docstring it contradicts. A stale doc is a
  finding, not a cosmetic.

### 3.5 Hardening audit

Check each row; a fail is a finding, not an automatic fix.

| Area | Question | Where to look |
| --- | --- | --- |
| **Exit codes** | Does every command return 0/1/2 consistently? Does partial failure return nonzero? | `cli.py` docstring; `run.py:322-343`; `conflicts.py:216-236,261-262` |
| **Exception boundary** | Is a failure isolated per item, or does one batch kill the run? Does the CLI print a clean error or a traceback? | `run.run`, `live.record_live`, `cli.main` |
| **Subprocess** | Timeout? `check`? `env`? `shell=False`? prompt disabled? | `cache._git` (hardened) vs `conflicts._git` (not) |
| **Writes** | Temp-file + `os.replace`? `fsync`? Ordered so no artifact references a missing one? | `cache.append_capped`, `cache.merge_candles`, `export.write_json(_jsonl)`, `conflicts.resolve` |
| **Concurrency** | Single writer? Lock? Rebase/retry? Is the local path also a writer? | `production.yml` merge; `tools/record_live.sh` |
| **Timeouts** | Per-fetch cap? Per-git cap? Abandoned thread? Hard exit only where intended? | `cache.call_with_timeout`, `cli.HARD_EXIT_COMMANDS` |
| **Input validation** | Ticker format, empty list, negative days/cap, root outside repo, path traversal | `tickers.parse_tickers`, `cache.slug_for`, argparse types |
| **Bounds** | Matrix parallelism, history days, live cap, model/archive size | `production.yml`, argparse defaults |
| **Idempotency** | Re-running a tick safe? Upsert vs append vs overwrite? | `merge_candles`, `append_capped`, `write_json` |
| **Observability** | Are per-ticker failures in the summary? Is there a structured status? | `run_pipeline` summary; `record_live` return |
| **CI** | ruff, pytest, basedpyright, shellcheck, actionlint; upstream import resolution | `ci.yml`, `pyproject.toml:90-95` |
| **Security** | Token in `git remote set-url`; secrets in logs; shell quoting; artifact path | `production.yml` merge; `record_live.sh` |

Two recurring traps in this repo:

- **A return value is not an exit code.** `conflicts.resolve` returns a count;
  the CLI turns it into an exit status. That is why a successful 2-path
  resolution exits 2 (PL-002).
- **A per-item try does not cover the batch.** `record_live` isolates the
  per-ticker loop, but `build_unified_many` (one batch fetch) is outside it
  (PL-009).

### 3.6 Classify

| Level | Meaning | Example |
| --- | --- | --- |
| **P0** | Silent data corruption/loss, or a cache commit silently skipped after work | partial live records never committed |
| **P1** | Automation contract broken; a workflow step fails after successful work | `conflicts` exits nonzero after resolving |
| **P2** | Availability/robustness: one failure kills the run, unbounded call, traceback, inconsistent artifact | batch fetch kills the watchlist |
| **P3** | Maintainability/cleanliness: cycle, duplication, schema inconsistency, CI gap | `dataset ↔ cache` lazy import |

Origin: repo code / workflow YAML / shell script / tests / upstream interface /
deliberate design. A deliberate design gets `EXPECTED`, not a "fix" — see
`os._exit`, `max-parallel: 4`, and the single-writer split.

Reproduce before filing:

- Exit-code claims: run the command in a throwaway repo and print `$?`
  (see PL-001/PL-002 for the reproductions used here).
- Concurrency claims: build a temporary git repo, create the rebase, run the
  resolver; never experiment on the real `data/`.
- Write-safety claims: inspect the write path; a crash test is only worth
  building if the failure can be committed.

### 3.7 Improvement / mapping step (required)

For every P0-P3 finding, add or update a row in the **Improvement map**.

| Field | Content |
| --- | --- |
| ID | `IM-N` |
| Finding | `PL-N` it addresses |
| Layer | cli / runner / cache / workflow / shell / test / docs |
| Action | concrete change, not a wish |
| Target | `path:line` or test name |
| Risk | behaviour change, cache migration, workflow |
| Status | proposed / in progress / monitoring / done |

Action ladder — smallest durable rung:

1. **Prove it** with a failing test or a throwaway-repo reproduction.
2. **Fix the boundary** (exit code, return type, timeout) before refactoring
   internals.
3. **Add the regression test** next to the existing suite.
4. **Add the CI gate** if the class of bug can recur (type check, shellcheck,
   actionlint).
5. **Refactor** only after behaviour is pinned.
6. **Retire legacy** only after every caller and doc is migrated.

Rules:

- Separate behaviour changes from refactors in the commit history.
- Preserve the client contracts in §1; if one must change, bump a schema and
  update `data/README.md` / `models/README.md`.
- Do not replace `os._exit` with normal exit until the abandoned-worker
  problem is solved another way.
- Do not add a file lock that fights the git merge model; prefer atomic
  replacement plus the semantic resolver.
- Every row names a `path:line` or a test. If it cannot, it is not ready.
- A P3 that is deliberate gets `closed: expected` with the reason, so it is not
  re-reported next review.

### 3.7.1 Upstream interface changes

If the finding is actually an upstream `tickerkit` / `ticker-news-signals`
interface defect (a renamed field, a changed error shape, a missing batch), do
**not** patch around it here. Use the `data-quality-audit` skill's
**§3.7.1 Upstream changes: spawn into an Ensemble space** workflow: check the
registered spaces, `team_spawn(space=..., worktree=false)` when one exists, and
re-pin the flake/uv input after the upstream push. This skill owns the local
pipeline code; that one owns the upstream boundary.

### 3.8 Update this file (the self-expansion step)

Do this **before** the final summary. The skill is the memory; a review that
does not write back is wasted work.

1. **Baseline block** — update numbers/timestamp in place; never append a
   second baseline.
2. **New finding** — append under `Memory — active findings` with the template
   below. One entry per root cause, not per symptom.
3. **Existing finding** — update the `Evidence` line's date/values in place.
4. **Status change** — append a dated line; move fixed/expected entries to
   `Memory — resolved & expected` with how they were verified.
5. **Mapping** — add rows; change `Status` for existing rows.
6. **Open questions** — add what you could not prove; remove what you answered.
7. **Audit log** — append exactly one row.
8. **Long-form report** — only for a substantial review, add a dated section to
   the repo's docs (or a design note) and link it from the finding.

Bounds:

- Active findings: **≤ 12**.
- Audit log: **≤ 20 rows**; collapse the oldest into one summary line.
- Each finding: **≤ 5 lines** after the heading.
- No narrative, no duplicated tables, no store paths, no undated numbers.

Template:

```markdown
### PL-0NN · <short claim> · <OPEN|MONITORING|FIXED|EXPECTED> (since YYYY-MM-DD)
- **Layer:** cli | runner | cache | workflow | shell | test | docs
- **Evidence (YYYY-MM-DD):** <path:line + reproduction/result>
- **Read:** <why it matters; the condition that makes it a real fault>
- **Map:** IM-N, IM-M
- **Status:** <last change + date>
```

### 3.9 Verify and commit

- Re-run pytest + ruff; run basedpyright if you touched src; run shellcheck if
  you touched `record_live.sh`.
- `git diff --stat`; commit behaviour fixes and refactors separately.
- Do not commit a `data/` change from a code review; if the review proves a
  cache fault, hand it to `data-quality-audit`.
- Report: verdict, P0/P1 exceptions, what changed in the ledger.

## Frequency and scope

- **Quick review** (default): baseline + import graph + exit-code/exception
  checks + memory update.
- **Full review**: all of §3.3-§3.5, per-command help parity, workflow invariants,
  and a fresh long-form report for any P0/P1.
- **Scoped review**: one module/command; still update the shared ledger but do
  not touch other findings.

## Do not break (refactor guardrails)

- `candles.jsonl` upsert and canonical timestamp keying.
- `live.jsonl` append + cap semantics.
- `summary.json` schema `rl-crypto/pipeline/3` and the `targets.*.holdout`
  block.
- `models/<slug>.json` schema/feature-version check and servable-bundle
  preference during conflicts.
- Single-writer workflow, `concurrency: rl-crypto-cache`, matrix jobs never
  pushing.
- `--root` / `RL_CRYPTO_ROOT` cwd anchoring.
- `os._exit` for `run`/`live` only.
- The `data/<slug>` naming (`slug_for`) and artifact layout in the workflow.

## Memory — baseline

**As of 2026-10-23T23:18Z** (`HEAD` `213d17b`):

| Metric | Value |
| --- | --- |
| Commands | 5: `run`, `live`, `conflicts`, `sync`, `version` |
| Pipeline modules | `run.py` 395, `live.py` 282, `conflicts.py` 270, `cli.py` 107, `sync.py` 65, `__init__` 22, `__main__` 8 lines |
| Shared I/O | `cache.py` 431, `export.py` 56, `config.py` 73, `models.py` 232 |
| Tests | **181 passed** (offline, `-m "not live"`) |
| Lint | `ruff check src/ tests/ examples/ tools/` — **clean** |
| Types | `basedpyright` — **8 errors**, all `reportMissingImports` for `ticker_news_signals` (not in CI) |
| Import graph | 28 modules; **cycle `dataset ↔ cache`** via a function-local import |
| Workflow shape | one schedule `*/30`; matrix `max-parallel: 4`; merge is the only pusher; `concurrency: rl-crypto-cache`; 3 push attempts |
| Cache | 15 tickers, 25,810 bars, 3,277 live records (see `data-quality-audit`) |

Verdict: **core architecture is sound and tested; two exit-code contract bugs
and several hardening gaps.** Fix PL-001/PL-002 before cosmetic refactors.

## Memory — active findings

### PL-001 · `rl-pipeline run` exits 0 on partial failure · OPEN (since 2026-10-23)
- **Layer:** runner (`src/rl_crypto/pipeline/run.py:322-343,379-380`)
- **Evidence (2026-10-23):** patched `run_pipeline` to raise for BTC and succeed for ETH over two tickers: printed `pipeline: 1/2 ticker(s) ok, failed: BTC/USD` and returned **0**. `exit_code` becomes 0 when any ticker succeeds; only "no summaries at all" resets it to 1. Tests cover all-fail, not partial.
- **Read:** contradicts `cli.py:11-18` ("0 ok, 1 partial or failed, 2 usage") and hides a local multi-ticker failure from `set -e`/CI. A production matrix runs one ticker per job, so this is local/dispatch-facing, not the scheduled matrix.
- **Map:** IM-1, IM-10
- **Status:** open; decide the contract, then test it.

### PL-002 · `rl-pipeline conflicts` returns the resolution count as its exit code · OPEN (since 2026-10-23)
- **Layer:** maintenance (`src/rl_crypto/pipeline/conflicts.py:216-236,261-262`)
- **Evidence (2026-10-23):** a real 2-path rebase conflict resolved successfully, then `python conflicts.py --root <tmp>` exited **2**. `resolve()` returns `len(paths)`; `run()` returns it directly.
- **Read:** under `bash -e` (GitHub Actions `run:` default) and `if ! command` (`tools/record_live.sh:75`), a **successful** resolution looks like a failure. The production merge step runs `python3 .../conflicts.py` after a failed rebase, so this can fail the merge after the cache was correctly merged.
- **Map:** IM-2, IM-3
- **Status:** open; return 0 on success and print the count.

### PL-003 · `conflicts._git` is unbounded and unhardened · OPEN (since 2026-10-23)
- **Layer:** maintenance (`src/rl_crypto/pipeline/conflicts.py:44-47`)
- **Evidence (2026-10-23):** `subprocess.run(..., check=True)` with no `timeout`, no `GIT_TERMINAL_PROMPT=0`, no `env`. Compare the hardened `cache._git` (`cache.py:262-275`: timeout, `check=False`, catches `OSError`/`TimeoutExpired`).
- **Read:** a credential prompt, stuck hook, or slow FS can hang the merge/resolver indefinitely; a non-git root raises `CalledProcessError` with a traceback. The rebase path is exactly where a hang is most expensive.
- **Map:** IM-3, IM-4
- **Status:** open.

### PL-004 · `cli.main` has no command-level error boundary · OPEN (since 2026-10-23)
- **Layer:** cli (`src/rl_crypto/pipeline/cli.py:83-93`)
- **Evidence (2026-10-23):** only `ImportError` and `KeyboardInterrupt` are caught. `conflicts.resolve` can raise `CalledProcessError` (`_git`, non-git root) or `ValueError` (`conflicts.py:213` when neither side is valid JSON); `rl-pipeline conflicts` then tracebacks. `run`/`live` catch internally; `sync` uses `sync_cache`, which catches.
- **Read:** the maintenance surface leaks Python tracebacks where the rest of the CLI prints `error: …` and returns 1/2. A traceback is not a contract.
- **Map:** IM-4
- **Status:** open.

### PL-005 · Cache writes are not atomic; summary can precede the bundle it describes · OPEN (since 2026-10-23)
- **Layer:** cache (`cache.py:102-113,156-176`; `export.py:22-51`; `run.py:252-256`; `conflicts.py:232`)
- **Evidence (2026-10-23):** `append_capped` reads the whole file and `write_text`s it; `merge_candles` → `write_jsonl(..., append=False)` truncates then writes; `write_json` truncates; the resolver `write_text`s. No temp + `os.replace`, no `fsync`. `run_pipeline` writes `summary.json` **before** `save_bundle`, so a bundle failure leaves a summary claiming a model that is not on disk.
- **Read:** a crash mid-write can leave a truncated `live.jsonl`/`candles.jsonl`/`summary.json`; git makes it recoverable only if the corruption is not committed. The summary/bundle order is a consistency violation the data-quality audit can see (`archive.rows` vs candles, summary vs model).
- **Map:** IM-5, IM-12
- **Status:** open; measure whether a committed corruption is reachable before adding a lock.

### PL-006 · `dataset ↔ cache` import cycle, closed by a function-local import · OPEN (since 2026-10-23)
- **Layer:** core (`src/rl_crypto/dataset.py:30`; `src/rl_crypto/cache.py:129-137`)
- **Evidence (2026-10-23):** AST graph finds `dataset ↔ cache`. `dataset` imports `cache.call_with_timeout`; `cache._canonical_timestamp` imports `dataset.parse_iso` inside the function to avoid a top-level cycle (the comment says so).
- **Read:** the lazy import is load-bearing; moving either function, or importing `parse_iso` at module scope, re-creates the cycle. It makes `cache`/`dataset` harder to test and reason about, but it is **not** a runtime bug today.
- **Map:** IM-6
- **Status:** open; low risk, do it as a pure refactor with tests.

### PL-007 · CLI scaffolding and `--root` are duplicated across four command modules · OPEN (since 2026-10-23)
- **Layer:** cli/runners (`run.py:60,273,306,391`; `live.py:50,210,248,277`; `sync.py:22,25,38,60`; `conflicts.py:41,239,251,265`)
- **Evidence (2026-10-23):** each module defines `ROOT = Path(os.environ.get("RL_CRYPTO_ROOT") or Path.cwd()).resolve()`, `add_arguments`, `parse_args`, `main`, and a `--root` flag. The four `main()` bodies are identical (`run(parse_args(argv))`).
- **Read:** a new global flag or a cwd/root change must be edited four times; divergence is how the exit-code bugs survived. Mechanical duplication, low risk to extract.
- **Map:** IM-7
- **Status:** open; refactor only, no behaviour change.

### PL-008 · `_generated_at` exists only on `live` records · OPEN (since 2026-10-23)
- **Layer:** runner/schema (`src/rl_crypto/pipeline/live.py:179-181`; `run.py` writes `record.to_dict()`)
- **Evidence (2026-10-23):** `live` payloads gain `_generated_at`; `run` payloads do not. The resolver treats it as a fallback time key (`conflicts.py:39`); the audit checks `generated_at` only (`tools/audit_data_quality.py:178`); the test asserts it exists (`test_record_live.py:76`).
- **Read:** a private-looking field is persisted on one path only. It may be intentional (a shared batch timeline), but the schema does not say so and the resolver/audit disagree about which field is authoritative.
- **Map:** IM-8
- **Status:** open; decide "batch timestamp" (emit on both paths) or "drop it" (one field).

### PL-009 · A batch market failure kills the whole live run · OPEN (since 2026-10-23)
- **Layer:** runner (`src/rl_crypto/pipeline/live.py:151-157,203-207`)
- **Evidence (2026-10-23):** `build_unified_many(tickers, ...)` (one `fetch_markets` batch) runs **before** the per-ticker `try`; if it raises, `record_live` returns 1 with zero payloads and no per-ticker records. The docstring's "one bad ticker cannot take the rest down" holds only after a successful batch. `test_record_live_isolates_a_failing_ticker` fails inside `fetch_prices`, not the batch.
- **Read:** the batch is the shared-budget optimization; a total batch failure (network/upstream down) is a legitimate whole-run failure, but the current behaviour is undocumented and untested. Decide: fail fast with a clear error, or emit per-ticker records with notes and a fallback fetch.
- **Map:** IM-9
- **Status:** open; document the contract first.

### PL-010 · CI does not run the configured type checker, shellcheck, or actionlint · OPEN (since 2026-10-23)
- **Layer:** ci (`.github/workflows/ci.yml:34-44`; `pyproject.toml:90-95`)
- **Evidence (2026-10-23):** CI runs ruff, notebook build, and pytest. `basedpyright` is configured and installed but not run; running it yields **8 errors**, all `reportMissingImports` for `ticker_news_signals` (path-source resolution). No shellcheck for `tools/record_live.sh`, no actionlint for the workflows.
- **Read:** the type config is aspirational; the shell/workflow bugs (PL-002/PL-011) are outside the current gates. Adding basedpyright without fixing upstream path resolution would make CI red for the wrong reason.
- **Map:** IM-10
- **Status:** open; configure `extraPaths`/ignore for the upstream, then add the gate.

### PL-011 · `tools/record_live.sh` is fragile and can drop a partial success · MONITORING (since 2026-10-23)
- **Layer:** shell (`tools/record_live.sh:13,15,28,44,50,69-79`)
- **Evidence (2026-10-23):** hardcodes `/etc/profiles/per-user/seanc/bin` (15), `origin main` (28), GNU `mktemp --suffix` (44), and runs `rl-pipeline live` under `set -e` (13, 50). `live` returns 1 when **any** ticker fails, so a partial success aborts before `git add`/commit; the successful records sit uncommitted until the next run. The conflict branch (75) uses the exit-code-buggy resolver (PL-002).
- **Read:** README says collection is now fully remote (`README.md:330`), so this is likely legacy/local. If it stays, it needs a partial-success policy and portable shell; if it goes, delete it and the shims together.
- **Map:** IM-11
- **Status:** monitoring; confirm whether the timer is still installed before hardening it.

## Memory — resolved & expected

- **R-001 · Single-writer workflow** — matrix jobs stage artifacts; only `merge` pushes; `concurrency: rl-crypto-cache`; 3 push attempts. Locked by `test_workflows.py:98-146`. Do not add a second pusher.
- **R-002 · Hard `os._exit` for `run`/`live`** — deliberate: a timed-out fetch can leave a worker thread. `HARD_EXIT_COMMANDS` is pinned by `test_pipeline_cli.py:83-85`. Keep it until the worker problem is solved another way.
- **R-003 · `call_with_timeout` abandons a worker thread** — accepted library-style partial failure (`cache.py:36-54`); the CLI hard exit is the containment boundary. Do not "fix" it by waiting, which reintroduces the hang.
- **R-004 · `max-parallel: 4`** — the ceiling is upstream RSS/provider tolerance, not CPU. Raising it changes the request budget; treat as a product decision.
- **R-005 · Deprecated `tools/*.py` shims** — one-release compatibility, tested to delegate (`test_pipeline_cli.py:88-113`). Remove only with the timer/caller migration.
- **R-006 · `slug_for` is traversal-safe** — `/` and spaces become `-` (`cache.py:60-62`), so `..` cannot escape `data/`. A stricter ticker regex is fine, but changing the slug rule needs a data migration.
- **R-007 · Model-bundle conflict preference** — a servable bundle beats a newer unservable one (`conflicts.py:124-140,199-208`); tested. Do not reduce this to "newest wins".

## Memory — improvement map

| ID | Finding | Layer | Action | Target | Risk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| IM-1 | PL-001 | runner | Make any ticker failure return 1; include the failure list in the summary/return; test partial success | `src/rl_crypto/pipeline/run.py:322-343`; `tests/test_run_pipeline.py` | exit-code behaviour; workflow dispatch sees it | proposed |
| IM-2 | PL-002 | maintenance | Separate the resolved count from the exit status; return 0 on success and print the count | `src/rl_crypto/pipeline/conflicts.py:216-236,261-262`; `tests/test_resolve_conflicts.py` | workflow/shell now see success correctly | proposed |
| IM-3 | PL-002, PL-003 | maintenance | Reuse one bounded git helper (timeout, `GIT_TERMINAL_PROMPT=0`, `check=False`, typed error) | `conflicts.py:44-47`; `cache.py:262-275` | low; behaviour on git failure changes | proposed |
| IM-4 | PL-003, PL-004 | cli | Catch expected command failures in `cli.main`, print `error: …`, return 1; add `--traceback` for debugging | `src/rl_crypto/pipeline/cli.py:83-93` | no traceback on operator errors | proposed |
| IM-5 | PL-005 | cache | Add a temp-file + `os.replace` + `fsync` writer for JSON/JSONL; write the bundle before the summary; test interruption | `cache.py:102-113,156-176`; `export.py:22-51`; `run.py:252-256` | atomicity on every write; small overhead | proposed |
| IM-6 | PL-006 | core | Extract `parse_iso` or `call_with_timeout` into a leaf module (no cycle), then drop the function-local import | `dataset.py:30`; `cache.py:129-137` | pure refactor; test import graph | proposed |
| IM-7 | PL-007 | cli | Add a shared command base/helper for `ROOT`, `--root`, `add_arguments`, `parse_args`, `main`; keep per-command flags explicit | `pipeline/{run,live,sync,conflicts}.py` | refactor; keep CLI help/contracts | proposed |
| IM-8 | PL-008 | schema | Decide: emit `_generated_at` on both paths as the batch timestamp, or drop it and use `generated_at`; document in `data/README.md` | `live.py:179-181`; `run.py`; `data/README.md` | schema/doc; resolver/audit consistency | proposed |
| IM-9 | PL-009 | runner | Make the batch-failure contract explicit: fail fast with a clear error, or emit per-ticker notes and fall back to individual fetches; test both | `live.py:151-207` | availability vs quota; choose deliberately | proposed |
| IM-10 | PL-010 | ci | Configure basedpyright for the upstream path (or ignore `reportMissingImports` for it), then add it; add shellcheck/actionlint | `ci.yml:34-44`; `pyproject.toml:90-95` | CI time; new red on real type errors | proposed |
| IM-11 | PL-011 | shell | Confirm the timer is still used; if yes, make partial success commit-safe, derive the branch, remove the hardcoded PATH, use a portable temp file; if no, retire the script and shims | `tools/record_live.sh` | data commit behaviour if hardened | monitoring |
| IM-12 | PL-005 | cross-skill | After any cache-schema/ordering change, run `data-quality-audit` and update its baseline; do not duplicate the audit here | `data-quality-audit` §3 | none | proposed |

## Memory — audit log

| Run (UTC) | HEAD | Tests | Lint | Types | Modules | Cycles | Findings | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-23T23:18Z | `213d17b` | 181 pass | clean | 8 upstream-import errors | 28 | `dataset↔cache` | PL-001…011 | core sound; fix the two exit-code bugs first |

## Memory — open questions

- **Q-1** What is the intended `rl-pipeline run` contract on a partial multi-ticker failure — 0 (current) or 1 (docstring)? Confirm before changing it.
- **Q-2** Is `_generated_at` consumed by anything outside `live.py`, the resolver fallback, and its test? If not, drop it rather than spread it.
- **Q-3** Is a committed mid-write corruption actually reachable, or does git + the next run always repair it? Measure before adding write locks.
- **Q-4** Should `conflicts` be a library function plus a thin CLI with a separate `(count, status)` result? That would remove the count/exit-code ambiguity permanently.
- **Q-5** Does basedpyright need `extraPaths` to the sibling upstreams, or is `reportMissingImports` suppression for `ticker_news_signals` the right call?
- **Q-6** Is `tools/record_live.sh` still installed as a systemd timer? README says collection is fully remote; confirm before spending effort on it.
- **Q-7** Should `run_pipeline` persist per-ticker failure details in `summary.json` (a `failures` block), or is stderr the contract?
- **Q-8** On a total market-batch failure, is failing the whole live run preferable to emitting per-ticker records with notes? This is a product/quota decision, not just code.




