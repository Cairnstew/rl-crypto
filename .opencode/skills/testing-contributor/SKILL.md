---
name: testing-contributor
description: >-
  Use when adding or fixing tests in rl-crypto, reproducing a bug offline, or
  deciding which test level a change needs. Triggers: "add a test", "write a
  test", "offline test", "mock upstream", "fixture", "pytest", "live marker",
  "workflow test", "why is the test flaky". Encodes the offline test pyramid,
  the fake-upstream and hermetic-git patterns, and the gaps the suite does not
  cover yet.
---

# Testing for contributors

The suite is already strong: offline, deterministic, and fast. This skill is
about extending it **at the right level** and not accidentally introducing a
network dependency, a real `data/` write, or a test that only proves the
implementation.

Companions: **`pipeline-hardening`** (the contract bugs the suite should pin),
**`model-contract`** (the ML test levels), **`data-quality-audit`** (cache
assertions).

## 0. Contract

- **Offline by default.** CI runs `pytest -m "not live"`; no test may hit a real
  upstream unless it is explicitly marked `live`.
- **No writes outside `tmp_path`.** Tests must never mutate the repo's real
  `data/`, `models/`, or git state.
- **Hermetic git.** Any git test sets `GIT_CONFIG_GLOBAL=/dev/null`,
  `GIT_CONFIG_SYSTEM=/dev/null`, `GIT_TERMINAL_PROMPT=0`, and a throwaway
  identity via `-c user.name=... -c user.email=...`.
- **Deterministic.** No wall-clock, random, network, or ordering assumptions;
  use the fixed rows in `tests/conftest.py` or a fixed seed.
- **Behaviour, not implementation.** Assert the artifact/summary/exit code; do
  not assert on a human console line unless the line is the contract.
- **Reproduce first.** Add the failing test before the fix when possible.

## 1. Ground truth

| Item | Value |
| --- | --- |
| Suite | 181 tests, offline, ~17 s (2026-10-23) |
| Marker | `live` is opt-in; default `addopts = -m 'not live'` (`pyproject.toml`) |
| CI | `.github/workflows/ci.yml`: ruff, notebook build, offline suite |
| Fake upstreams | `tests/conftest.py`: `make_snapshot`, `make_vector`, `make_signal_result`, `make_dataset_rows` |
| Hermetic git | `tests/test_cache_sync.py`, `tests/test_resolve_conflicts.py` |
| Workflow tests | `tests/test_workflows.py` parses YAML and executes the `prepare` heredoc |
| ML tests | `test_features`, `test_targets`, `test_samples`, `test_ml_linear`, `test_ml_measurement`, `test_ml_tier1`, `test_ml_serve`, `test_ml_store` |
| Pipeline tests | `test_run_pipeline`, `test_record_live`, `test_resolve_conflicts`, `test_cache_sync`, `test_pipeline_cli` |

## 2. Test pyramid

| Level | Use for | Examples | Cost |
| --- | --- | --- | --- |
| **Unit** | Pure functions: features, targets, model math, slug/ticker parsing | `test_features.py`, `test_targets.py`, `test_ml_linear.py` | ms |
| **Integration (tmp dir)** | A runner against a temp repo/cache | `test_run_pipeline.py`, `test_record_live.py`, `test_cache_sync.py` | <1 s |
| **Git/rebase** | Real conflict semantics in a throwaway repo | `test_resolve_conflicts.py` | <1 s |
| **Workflow structural** | YAML invariants, matrix source, merge path, prepare script | `test_workflows.py` | ms |
| **Live (opt-in)** | Real upstream smoke tests | `pytest -m live` | seconds/minutes, network |

Choose the **lowest level that can fail for the reason you care about**. A
workflow invariant belongs in `test_workflows.py`, not in a live run. A cache
merge belongs in a temp git repo, not in the real checkout.

## 3. Patterns

### 3.1 Fake the upstream objects, not HTTP

The builders read only a handful of attributes. Use the `SimpleNamespace`
doubles in `tests/conftest.py`:

```python
from tests.conftest import make_snapshot, make_signal_result, make_vector

snapshot = make_snapshot(price=2500.0, errors=1)
assert snapshot.market.price == 2500.0
```

Do not import tickerkit or ticker-news-signals in an offline test, and do not
spin up `respx` unless you are testing HTTP behavior that the doubles cannot
express.

### 3.2 Deterministic rows

`make_dataset_rows(n, wiggle=..., news_every=...)` builds a fixed hourly path
with hand-checkable returns. Use it for features/targets/ML; do not call
`utcnow()` in the assertion.

### 3.3 Monkeypatch the seam, not the internals

The runners expose module-level functions specifically as test seams:

```python
monkeypatch.setattr(runner, "build_dataset", lambda *a, **k: Dataset(...))
monkeypatch.setattr(runner, "build_unified", lambda *a, **k: UnifiedRecord(...))
monkeypatch.setattr(tool, "fetch_prices", lambda *a, **k: (candles, ()))
monkeypatch.setattr(tool, "build_unified_many", lambda ts, **k: [...])
```

If a refactor moves these names, update the tests **and** keep the seam, or
explain why the seam moved. Do not mock deep private helpers.

### 3.4 Temp directories and hermetic git

```python
def _git(cwd, *args):
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=cwd, check=True, capture_output=True, text=True, env=env,
    ).stdout
```

Use `tmp_path`; never `Path.cwd()` for writes. Build a local remote with
`git init` + `git clone` when testing sync; no network.

### 3.5 Workflow tests

`test_workflows.py` is the right place for invariants that YAML alone cannot
express:

- parse the workflow with `yaml.safe_load` (remember YAML 1.1 parses `on:` as
  `True`);
- assert the matrix source, single-writer rule, `max-parallel`, concurrency
  group, retry loop, artifact path;
- extract the `prepare` heredoc and execute it with a temp `GITHUB_OUTPUT` to
  test the real script, as the existing tests do.

A workflow test may assert a substring like `"rl-pipeline live"`; keep those
assertions tied to a documented invariant, not incidental formatting.

### 3.6 Assert the artifact, not the console

Prefer:

- `summary["schema"]`, `summary["archive"]["rows"]`, `summary["targets"][...]`;
- the lines in `data/<slug>/candles.jsonl` / `live.jsonl`;
- the exit code;
- the existence/contents of `models/<slug>.json`.

Only assert stdout when stdout is the public interface (e.g. `rl-pipeline
version`).

### 3.7 Live tests

Mark with `@pytest.mark.live` and keep them out of the default run. They are
for real upstream smoke checks, never for a logic test that can be faked.

## 4. Procedure

1. Read `Memory`; run the baseline (`pytest -m "not live"`, ruff).
2. Reproduce the bug with the smallest failing test.
3. Put the test at the lowest level that can catch it.
4. Reuse `conftest.py` fixtures; add one only if it is reused.
5. Assert the contract (artifact/exit code), not a private call.
6. Run the full offline suite and ruff.
7. If the test needs a new marker/dependency/CI step, update `pyproject.toml`
   and `ci.yml` in the same commit.
8. Update the memory with any new fixture pattern or gap.

## 5. Update this file

- Add a new fixture/pattern to §3 only if it is reusable.
- Add a coverage gap to `Memory — active findings` (not every missing test is a
  finding; the contract gaps are).
- Update the baseline test count/time after a significant suite change.

## Memory — baseline

**As of 2026-10-23T23:18Z** (`HEAD` `213d17b`):

| Metric | Value |
| --- | --- |
| Tests | **181 passed**, 0 failed |
| Runtime | ~16.9 s offline |
| Warnings | 10 `DeprecationWarning`s from `aiohttp` (upstream; not ours) |
| Lint | ruff clean |
| Types | `basedpyright` 8 `reportMissingImports` errors for `ticker_news_signals` (not in CI) |
| Marker | `live` opt-in; default `-m 'not live'` |
| CI | ruff + notebook build + offline suite |

## Memory — active findings

### TC-001 · No test covers the conflict resolver's CLI exit code · OPEN (since 2026-10-23)
- **Layer:** test gap (`tests/test_resolve_conflicts.py`)
- **Evidence (2026-10-23):** the suite asserts `resolve(repo) == 2` for a 2-path conflict, but never calls `run()`/`main()` and checks the **exit status**. A real resolution exits 2 (pipeline-hardening PL-002).
- **Read:** the missing test is why the count-as-exit-code bug survived. Add a CLI-level test that asserts 0 on success and prints the count.
- **Map:** IM-1
- **Status:** open.

### TC-002 · No test covers `run`'s partial-failure exit code · OPEN (since 2026-10-23)
- **Layer:** test gap (`tests/test_run_pipeline.py`)
- **Evidence (2026-10-23):** `test_run_pipeline_exit_nonzero_when_everything_fails` covers all-fail; a two-ticker run with one failure returns 0 (pipeline-hardening PL-001).
- **Read:** add a test that patches `run_pipeline` to raise for one ticker and succeed for another, asserts the documented status, and pins the failure list.
- **Map:** IM-1
- **Status:** open.

### TC-003 · No test covers cache-write atomicity or ordering · OPEN (since 2026-10-23)
- **Layer:** test gap (`tests/test_cache.py`, `tests/test_run_pipeline.py`)
- **Evidence (2026-10-23):** writes go through `Path.write_text`/`write_jsonl` with truncate-then-write; no test interrupts a write or asserts `summary` is written after the bundle (pipeline-hardening PL-005).
- **Read:** a crash test is only worth building if the failure can be committed; at minimum assert the artifact order and that a failed bundle write does not leave a summary claiming a model.
- **Map:** IM-2
- **Status:** open.

### TC-004 · Workflow tests extract the `prepare` heredoc · EXPECTED (since 2026-10-23)
- **Layer:** tests (`tests/test_workflows.py:164-203`)
- **Evidence (2026-10-23):** the test pulls the `python3 - <<'PY'` block out of `production.yml` and executes it with a temp `GITHUB_OUTPUT`.
- **Read:** this is deliberately stronger than string assertions, but it couples to the heredoc shape. If the workflow changes, update the extractor; do not weaken it to a substring check.
- **Map:** IM-6
- **Status:** expected.

### TC-005 · The type checker is configured but not gated · OPEN (since 2026-10-23)
- **Layer:** CI (`pyproject.toml:90-95`; `.github/workflows/ci.yml`)
- **Evidence (2026-10-23):** `basedpyright` is installed/configured, but CI never runs it; a direct run gives 8 upstream-import errors.
- **Read:** add it only after configuring the upstream path or suppressing `reportMissingImports` for `ticker_news_signals`, otherwise CI is red for the wrong reason.
- **Map:** IM-3
- **Status:** open.

### TC-006 · Live tests are opt-in · EXPECTED (since 2026-10-23)
- **Layer:** tests (`pyproject.toml` marker)
- **Evidence (2026-10-23):** `addopts = -m 'not live'`; CI runs exactly that; live tests hit real upstreams by hand.
- **Read:** keep it that way. A live test in the default suite makes CI flaky and quota-dependent.
- **Map:** none
- **Status:** expected.

### TC-007 · No coverage measurement · OPEN (since 2026-10-23)
- **Layer:** tooling (`pyproject.toml`)
- **Evidence (2026-10-23):** `pytest-cov` is not a dependency and CI reports only pass/fail, not line coverage.
- **Read:** coverage is not a goal by itself, but the exit-code/atomicity gaps are exactly the kind a coverage report would highlight. Add it as a report, not a hard threshold, or it will encourage low-value tests.
- **Map:** IM-4
- **Status:** open.

### TC-008 · Tests depend on module-level monkeypatch seams · EXPECTED (since 2026-10-23)
- **Layer:** tests/runners
- **Evidence (2026-10-23):** `runner.build_dataset`, `runner.build_unified`, `tool.fetch_prices`, `tool.build_unified_many` are patched by name.
- **Read:** these are intentional seams; a refactor that moves them breaks tests for no behavioural reason. Keep or update them deliberately, and note it in the refactor commit.
- **Map:** IM-6
- **Status:** expected.

## Memory — resolved & expected

- **R-001 · The suite is fully offline by default** — upstream results are `SimpleNamespace` doubles; no HTTP mocking is needed for most tests.
- **R-002 · Git tests use throwaway repos** — `git init`/`clone` under `tmp_path`, hermetic config, no network, no real `data/`.
- **R-003 · The workflow tests execute the real prepare script** — stronger than YAML shape assertions; keep the extractor aligned with the heredoc.
- **R-004 · ML tests use deterministic rows** — `make_dataset_rows` gives hand-checkable returns; do not use random data for a contract test.
- **R-005 · `live` is a marker, not a separate suite** — one command excludes it; do not create a second test tree.

## Memory — improvement map

| ID | Finding | Layer | Action | Target | Risk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| IM-1 | TC-001, TC-002 | tests | Add CLI-level exit-code tests for `conflicts` and `run` (including partial failure) | `tests/test_resolve_conflicts.py`, `tests/test_run_pipeline.py` | low; pins the fix | proposed |
| IM-2 | TC-003 | tests | Add an artifact-order/atomic-write regression test; assert no summary without a bundle | `tests/test_run_pipeline.py`, `tests/test_cache.py` | low | proposed |
| IM-3 | TC-005 | CI | Configure basedpyright for the upstream imports, then add it to CI; add shellcheck/actionlint | `pyproject.toml`, `ci.yml` | new red on real type errors | proposed |
| IM-4 | TC-007 | tooling | Add `pytest-cov` as a report (not a threshold) and record the baseline | `pyproject.toml`, `ci.yml` | low | proposed |
| IM-5 | TC-004, TC-008 | tests | Factor the hermetic git helper into a shared fixture; keep the prepare-script extractor | `tests/conftest.py` | low | proposed |
| IM-6 | all | docs | Keep this test map linked from the README/testing section | `README.md` | none | proposed |

## Memory — audit log

| Run (UTC) | HEAD | Tests | Runtime | Lint | Types | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-23T23:18Z | `213d17b` | 181 pass | ~16.9 s | clean | 8 upstream-import errors | strong offline pyramid; exit-code and atomicity gaps are untested |

## Memory — open questions

- **Q-1** Should coverage be a report or a gate? A hard threshold on this codebase would reward shallow tests; start with a report.
- **Q-2** Should the hermetic git helper live in `conftest.py` or a small `tests/gitutil.py`? It is duplicated in `test_cache_sync.py` and `test_resolve_conflicts.py`.
- **Q-3** How should a test assert the `_generated_at` schema decision (pipeline-hardening PL-008) once it is made?
- **Q-4** Is `notebooks/tests/quick_test.ipynb` part of the test surface? It is not executed by `ci.yml`; decide whether it is a smoke test or dead weight.
- **Q-5** Should a `--cov` baseline be recorded in this skill so a future change can compare rather than guess?

