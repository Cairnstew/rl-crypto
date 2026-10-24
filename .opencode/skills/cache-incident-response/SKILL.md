---
name: cache-incident-response
description: >-
  Use when the rl-crypto production cache looks stale, missing, partially
  written, or stuck: no new data/ commit, a failed production or watchdog run,
  a rejected push, a rebase conflict, a dirty/diverged checkout, a missing
  model for a new ticker, or a cache that stops updating. Triggers: "cache is
  stale", "no cache commit", "watchdog", "production red", "merge failed",
  "rebase conflict", "conflicts", "sync refused", "missing bundle", "pipeline
  stopped". A safe operator runbook with a decision tree, exact commands, and
  an incident memory.
---

# Cache incident response

This is the **operator** skill for the committed `data/` + `models/` cache. It
answers "what do I do now?" during a live incident. It does not diagnose data
correctness or redesign the code:

- **`data-quality-audit`** — is the data that landed correct?
- **`pipeline-hardening`** — is the code that wrote it robust?
- **this skill** — the cache is stale/broken; restore the writer safely.

## 0. Contract

- **Diagnose before mutating.** Read the run, the commit, and `git status`
  before touching anything.
- **Never force-push.** Never `git push --force`, never rewrite `data/` history.
- **Never hand-edit the archive.** `candles.jsonl` is an upsert series; a manual
  rewrite is a data incident, not a fix.
- **Never discard a rebase before capturing it.** The conflicted sides may be
  the only copy of collected records.
- **The merge job is the only scheduled writer.** Do not start a second writer
  against the same branch while a run is in flight.
- **A nonzero `rl-pipeline conflicts` may be the count, not a failure**
  (pipeline-hardening PL-002). Verify by reading stdout and checking that no
  unmerged paths remain; do not blindly treat exit 2 as success or failure.
- **Record the incident.** Every response ends with an entry in this skill's
  memory: symptom, cause, command, resolution, and what would have caught it.

## 1. Ground truth

| Fact | Value / source |
| --- | --- |
| Scheduler | `.github/workflows/production.yml`, one cron `*/30 * * * *` |
| Modes | `live` on odd UTC hours; `train` on even UTC hours (archive + retrain) |
| Watchdog | `.github/workflows/watchdog.yml`, hourly at `:17`; dispatch `live` if newest `data/` commit is older than `RL_CRYPTO_MAX_AGE_MINUTES` (default **75**) |
| Matrix | one job per ticker, `max-parallel: 4`, `fail-fast: false` |
| Writer | only the `merge` job pushes; `concurrency: rl-crypto-cache`; 3 push attempts |
| Artifacts | matrix jobs stage `cache-<slug>`; merge downloads to `rl-crypto/staging`, then copies into `data/`/`models/` |
| Cache contracts | `candles.jsonl` upsert, `live.jsonl` append/capped, `summary.json` overwritten, `models/<slug>.json` committed |
| Current baseline | newest `data/` commit `5724fa0` at 2026-10-23T21:06Z; 15 tickers (re-check) |

## 2. Decision tree

### 2.1 Is it actually stale?

```bash
git log -1 --format='%h %cI %s' -- data/
git log -1 --format='%h %cI %s'
gh run list --workflow production.yml --limit 5
```

- Newest `data/` commit **< 75 min** → not stale; a tick may still be running.
- **75 min–2 h** → watchdog should dispatch within the hour; check the watchdog
  run before intervening.
- **> 2 h** → treat as an incident. Continue to §2.2.

### 2.2 Did the last run fail?

```bash
gh run list --workflow production.yml --limit 10
gh run view <run-id> --log-failed
gh run view <run-id>
```

Classify:

| Symptom | Likely cause | Go to |
| --- | --- | --- |
| `prepare` failed | bad watchlist/env/tickers file | §3.1 |
| one `live`/`train` matrix job failed | one ticker/upstream; other artifacts still stage | §3.2 |
| all matrix jobs failed | upstream/network/dependency outage | §3.2, then §3.6 |
| `merge` failed before commit | no artifacts, copy failure, disk | §3.3 |
| `merge` failed on push | concurrent writer/rebase conflict | §3.4 |
| `merge` succeeded but no commit | "no cache changes to commit" (genuinely unchanged) | §3.5 |
| watchdog failing | `gh`/permissions/API | §3.6 |

### 2.3 Is the local checkout part of the problem?

```bash
git status --short
git branch --show-current
git log --oneline -3
git rev-parse --abbrev-ref --symbolic-full-name '@{u}'
```

- Detached HEAD → reattach before any commit/push (`git switch main` or
  `git switch -C main origin/main`; this is what `record_live.sh` does).
- Dirty only under `data/`/`models/` → a local run left work uncommitted; decide
  whether it is wanted before syncing.
- Diverged (ahead and behind) → do **not** force; use `rl-pipeline sync` and
  read its notes.

## 3. Scenario runbooks

### 3.1 `prepare` failed

`prepare` is the single source of the matrix and mode. If it fails, there is no
run to merge.

1. Read the failed step log. Common causes: a bad `RL_CRYPTO_TICKERS` value, a
   `tickers.py` import error, an invalid `GITHUB_OUTPUT` line.
2. Fix the variable or code; do not bypass `prepare` by hand-editing the matrix.
3. Re-run: `gh workflow run production.yml --ref main -f mode=live` (or
   `mode=train`). For a one-off narrowed run, pass `-f tickers="BTC/USD"`.

### 3.2 One or more matrix jobs failed

The matrix is `fail-fast: false`, and staging runs `if: always()`. A failed
ticker therefore does **not** block the others.

1. Identify the failed tickers from `gh run view`.
2. For one ticker, reproduce locally:

   ```bash
   nix develop --command bash -c 'uv run rl-pipeline live --tickers BTC/USD --fetch-timeout 120'
   ```

   A live failure usually means no bundle (new ticker) → §3.5, or an upstream
   outage → §3.6.
3. If the merge still ran, let it land the surviving artifacts. Do not re-run
   the whole matrix to "fill in" one ticker unless the merge did not commit.
4. If all jobs failed, treat it as an upstream/network incident; check the
   sibling upstreams and `data-quality-audit` before retrying.

### 3.3 `merge` failed before commit

1. Check whether artifacts arrived: `gh run view <id>` → the download step.
   The workflow already fails loudly if `staging/data` and `staging/models` are
   both absent; that is intentional, not a no-op.
2. Check the copy step for a path/permission error. The download path must be
   `rl-crypto/staging` (workspace-root-relative), not `staging`.
3. If a partial copy happened, the step aborted before `git commit`; the
   checkout is disposable. Re-run the merge from a clean run rather than
   committing a partial cache by hand.
4. If a hand commit is unavoidable, verify with `data-quality-audit` first and
   commit only complete per-ticker directories.

### 3.4 `merge` failed on push / rebase conflict

The workflow already retries three times, rebases, resolves `data/` conflicts,
re-renders the dashboard, and continues the rebase.

Manual path if it is stuck:

```bash
git status --short
git diff --name-only --diff-filter=U
nix develop --command bash -c 'uv run rl-pipeline conflicts --dry-run'
nix develop --command bash -c 'uv run rl-pipeline conflicts'
git diff --name-only --diff-filter=U   # must now be empty
GIT_EDITOR=true git rebase --continue
```

Resolution semantics (why this is safe):

- `live.jsonl` — union by record identity, ordered by timestamp; same-timestamp
  different-payload records both survive.
- `candles.jsonl` — one bar per timestamp; the newer archive wins a collision.
- `summary.json` / `models/<slug>.json` — keep the newer run whole; a model the
  running code cannot serve loses to one it can.

**Do not** `git checkout --theirs` or `--ours` on a whole `data/` tree. **Do
not** `git rebase --abort` until you have captured both sides; if you must
abort, save the conflicted files first.

If `conflicts` exits nonzero, read its stdout. On the current code a successful
resolution exits with the **count** (e.g. 2), not 0 — verify the unmerged-path
list is empty before continuing (pipeline-hardening PL-002).

### 3.5 Missing bundle / new ticker

A ticker with no `models/<slug>.json` cannot record a live outlook. The first
train is mandatory:

```bash
nix develop --command bash -c 'uv run rl-pipeline run --tickers NEW/USD --history-days 60 --fetch-timeout 600'
```

Then verify `data/new-usd/candles.jsonl`, `data/new-usd/summary.json`, and
`models/new-usd.json`; run `live` once. Do not "fix" this by copying another
ticker's bundle.

### 3.6 Upstream / network outage

- Check the two sibling checkouts' own tests/smoke commands.
- A partial outage is expected to land as notes, not a whole-run failure. If all
  tickers fail, wait and re-dispatch; do not add credentials or key-gate a
  provider.
- If a specific upstream interface changed, stop patching here and use
  `data-quality-audit` §3.7.1 (Ensemble space) for the upstream fix.

### 3.7 Sync refused / dirty checkout

```bash
nix develop --command bash -c 'uv run rl-pipeline sync'
```

Read the notes. `sync_cache` refuses to overwrite a dirty cache or a diverged
branch; that is correct. Do not force. Decide whether the local changes are
wanted; if not, stash/discard them with explicit approval, then sync again.

## 4. Safe vs unsafe commands

| Safe | Why |
| --- | --- |
| `gh run list/view --log-failed` | read-only |
| `git log`, `git status`, `git diff --name-only --diff-filter=U` | read-only |
| `rl-pipeline sync` | ff-only, refuses safely |
| `rl-pipeline conflicts --dry-run` | no writes |
| `rl-pipeline conflicts` | semantic merge; verify the exit code |
| `rl-pipeline run --retrain-only` | no network, no live append |

| Unsafe | Why |
| --- | --- |
| `git push --force` | destroys concurrent collected records |
| `git checkout --ours/--theirs` on `data/` | silently drops one writer's records |
| `git rebase --abort` before capture | may discard the replayed records |
| hand-editing `candles.jsonl`/`live.jsonl` | breaks the upsert/append contracts |
| starting a local run while a merge is in flight | two writers against one branch |
| replacing a bundle with another ticker's | wrong feature contract/weights |

## 5. Escalation

- **Data looks wrong after recovery** → `data-quality-audit`.
- **The same failure recurs / code is fragile** → `pipeline-hardening`.
- **Upstream interface changed** → `data-quality-audit` §3.7.1.
- **Model/feature contract was violated** → `model-contract`.

## 6. Update this file

After every incident:

1. Append one row to `Memory — incident log` (symptom, cause, command, outcome).
2. Add/update a finding if the cause is systemic; mark `FIXED` with the commit.
3. Update the improvement map.
4. Leave the human-facing incident timeline in the issue/PR if one exists; keep
   this ledger short.

## Memory — baseline

**As of 2026-10-23T23:18Z** (`HEAD` `213d17b`):

| Metric | Value |
| --- | --- |
| Active incident | none |
| Newest `data/` commit | `5724fa0` at 2026-10-23T21:06Z |
| Watchdog limit | 75 minutes (repository variable `RL_CRYPTO_MAX_AGE_MINUTES`) |
| Cadence | live every 30 min; train on even UTC hours |
| Writer | merge job only; `concurrency: rl-crypto-cache`; 3 push attempts |
| Tracked tickers | 15 (re-check `src/rl_crypto/tickers.py`) |
| Known latent hazards | resolver exit code = count (PL-002); watchdog path scope (IR-003); local timer partial success (PL-011) |

## Memory — incident log

| Date (UTC) | Symptom | Cause | Command / outcome |
| --- | --- | --- | --- |
| 2026-10-23 | audit only; no incident | — | Verified cadence/watchdog/merge shape; recorded latent PL-002 and IR-003. |

## Memory — active findings

### IR-001 · A successful conflict resolution can exit nonzero · OPEN (since 2026-10-23)
- **Layer:** workflow/CLI (`src/rl_crypto/pipeline/conflicts.py:216-236,261-262`; `production.yml` merge)
- **Evidence (2026-10-23):** resolving 2 real conflicted paths exited **2**; the verifier confirmed no unmerged paths remained. Cross-ref `pipeline-hardening` PL-002.
- **Read:** the merge step runs under `bash -e`, and `record_live.sh` uses `if ! …`; either can treat a correct resolution as a failure. Until fixed, always verify the unmerged list after a nonzero exit.
- **Map:** IM-1
- **Status:** open; fix belongs to pipeline-hardening.

### IR-002 · The local timer can drop a partial success · MONITORING (since 2026-10-23)
- **Layer:** shell (`tools/record_live.sh:13,50`)
- **Evidence (2026-10-23):** `set -e` + `rl-pipeline live` returning 1 on **any** ticker failure aborts before `git add`/commit; successful records remain uncommitted until the next run. Cross-ref `pipeline-hardening` PL-011.
- **Read:** production is fully remote (`README.md:330`), so this is local-only. Confirm the timer is still installed before hardening it; retiring it may be the better fix.
- **Map:** IM-4
- **Status:** monitoring.

### IR-003 · The watchdog counts docs commits under `data/` as fresh cache · OPEN (since 2026-10-23)
- **Layer:** workflow (`.github/workflows/watchdog.yml:33-45`)
- **Evidence (2026-10-23):** `gh api "repos/${GITHUB_REPOSITORY}/commits?path=data&per_page=1"` returns the newest commit touching **any** path under `data/`, including `data/AUDIT.md` and `data/README.md`.
- **Read:** a docs-only edit under `data/` resets the age and suppresses the heal dispatch while the actual cache may be stale. The data-quality audit itself writes `data/AUDIT.md`, so this is reachable.
- **Map:** IM-2
- **Status:** open; narrow the path to `data/*/live.jsonl` or `data/*/summary.json`, or filter to `live:`/`pipeline:` commits.

### IR-004 · A failed production run has no explicit alert · OPEN (since 2026-10-23)
- **Layer:** workflow (`.github/workflows/production.yml`)
- **Evidence (2026-10-23):** a failed matrix/merge job is visible in the Actions UI and GitHub's default notifications, but nothing opens an issue or pages; the watchdog only heals **staleness**, not a failed merge that leaves the cache stale for <75 min.
- **Read:** a partial failure usually still commits (staging `if: always()`), so the important alert is "merge failed" or "no commit after a run", not every ticker failure.
- **Map:** IM-3
- **Status:** open; decide the alert channel before adding a workflow.

### IR-005 · Manual triage assumes `gh` is authenticated · EXPECTED (since 2026-10-23)
- **Layer:** tooling
- **Evidence (2026-10-23):** all run inspection uses `gh run list/view`, `gh api`, and `gh workflow run`; an unauthenticated shell fails before any diagnosis.
- **Read:** check `gh auth status` first. This is an operator prerequisite, not a cache fault.
- **Map:** none
- **Status:** expected.

### IR-006 · A new ticker cannot record live until it has a bundle · EXPECTED (since 2026-10-23)
- **Layer:** runner/bundle (`src/rl_crypto/pipeline/live.py:168`)
- **Evidence (2026-10-23):** `load_bundle(default_bundle_path(...))` is required before the outlook; a ticker added to `tickers.py` has no `models/<slug>.json` until a train run.
- **Read:** dispatch `train` for the new ticker first (`rl-pipeline run --tickers NEW/USD`), then `live`. This is documented in `README.md` and is not a bug.
- **Map:** none
- **Status:** expected.

## Memory — resolved & expected

- **R-001 · Matrix jobs never push; the merge is the single writer** — `production.yml` + `concurrency: rl-crypto-cache`. This is what makes conflict resolution a merge problem, not a lost-update problem.
- **R-002 · `if: always()` staging preserves partial success** — a failed ticker still uploads whatever it produced; the merge runs `if: always() && prepare == success`.
- **R-003 · Three push attempts, then rebase + semantic resolve + retry** — expected behavior; a single rejection is normal under concurrent writers.
- **R-004 · `rl-pipeline sync` refuses dirty/diverged state** — `cache.sync_cache` reports the reason instead of forcing; do not "fix" that with `reset --hard`.
- **R-005 · Dashboard conflict resolution keeps upstream text and re-renders** — the workflow checks out `--ours` for README, re-renders from the merged cache, and continues the rebase.
- **R-006 · Watchdog dispatch is the safety net, not the primary scheduler** — 75 min allows two missed 30-minute ticks; a longer outage is an incident.

## Memory — improvement map

| ID | Finding | Layer | Action | Target | Risk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| IM-1 | IR-001 | CLI | Return 0 on a successful resolution; print the count; add a CLI exit-code test | `pipeline/conflicts.py`, `tests/test_resolve_conflicts.py` | workflow now sees success | proposed |
| IM-2 | IR-003 | workflow | Narrow the watchdog path to actual cache artifacts (`data/*/live.jsonl` or `data/*/summary.json`) or filter `live:`/`pipeline:` subjects | `.github/workflows/watchdog.yml` | a docs-only edit no longer suppresses healing | proposed |
| IM-3 | IR-004 | workflow | Add an explicit failure alert (issue/notification) for `merge` failure or "run succeeded but no cache commit" | `.github/workflows/production.yml` | notification noise; choose a channel | proposed |
| IM-4 | IR-002 | shell | Confirm the local timer is still used; if yes, make partial success commit-safe; if no, retire it and the shims | `tools/record_live.sh` | local commit behaviour | monitoring |
| IM-5 | all | tooling | Add a read-only `rl-pipeline status` (newest commit age, last run result, per-ticker bundle/summary freshness) | `pipeline/cli.py` | new surface; keep it read-only | proposed |
| IM-6 | IR-001, IR-003 | docs | Keep this runbook linked from the README/ops section | `README.md` | none | proposed |

## Memory — open questions

- **Q-1** Should the watchdog ignore docs-only commits under `data/`? Narrowing to `data/*/summary.json` is the smallest fix; confirm GitHub's path filter supports the glob reliably.
- **Q-2** What is the acceptable recovery time? Current worst case is ~75 min detection + hourly watchdog + one 30-min run.
- **Q-3** Should a failed `merge` open an issue automatically, or is the GitHub Actions notification enough?
- **Q-4** Is `tools/record_live.sh` still installed as a timer? If not, retire it rather than fix its partial-success behavior.
- **Q-5** Should `rl-pipeline conflicts` be fixed before the next scheduled conflict window, or is the manual verify step sufficient?
- **Q-6** Does `sync_cache.up_to_date` ignoring `ahead` mislead an operator? It is cache-oriented; document it if it causes confusion.

