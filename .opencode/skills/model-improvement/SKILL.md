---
name: model-improvement
description: Use when the user asks to improve rl-crypto model performance, find an edge, retrain, sweep, diagnose a weak model, or promote a candidate. Orchestrates status → evaluate → diagnose → sweep → promote → wire-in, using rl-model-lab and the repo's data/model/pipeline skills. Never claims profit and stops when the data has no edge.
---

# Model Improvement Loop

This is the **orchestrator**. It decides what to do; `rl-model-lab` measures.
Do not hardcode a sweep-everything loop. Diagnose first, then choose the
smallest experiment that tests a real hypothesis.

Companions: **`model-contract`** (features/targets/holdout/bundle contract),
**`backtest-protocol`** (evaluation contract), **`data-quality-audit`**
(input data), **`pipeline-hardening`** (runner/cache),
**`cache-incident-response`** (stale/broken cache),
**`ticker-onboarding`** (watchlist changes).

## 0. Guardrails

- **No profit claims.** Report out-of-sample metrics, after-cost policy, and
  gate reasons. "No edge" is a valid, useful result.
- **Never train or tune on the final 20%.** Only `promote` may read it.
- **No `--apply` without explicit human approval.** Dry-run first.
- **No feature/target change without bumping `FEATURE_VERSION`** and following
  `model-contract`.
- **Do not sweep stale or broken data.** Fix the input first.
- **Budget:** default one sweep per request, `--max-configs 8`, `--folds 4`.
  Ask before exceeding.
- **Do not promote** a candidate that wins one fold, loses after costs, or only
  beats a weaker previous bundle.

## 1. Scope

Establish: tickers (default `BTC/USD ETH/USD SOL/USD`), targets (default all
four; directional work uses `fwd_direction` / `fwd_return`), time budget, and
whether the user asked for diagnosis, sweep, or promotion.

## 2. Observe

Run `rl-model-lab` `status`, then `evaluate` on the scoped tickers.

Read:

- **Target table** — model vs constant baseline, mean/min improvement, folds
  beating. Directional needs a consistent majority; one lucky fold is noise.
- **Policy table** — after-cost Sharpe/return/drawdown vs momentum and
  always-long. A gross edge smaller than fees + slippage is not an edge.
- **Bundle state** — feature version, trust, holdout, freshness.
- **Read `trusted` and `holdout beats` together.** `trusted` lists the targets
  whose *own* holdout beat the baseline; `holdout beats` shows each target
  (`ret`, `dir`, `vol`, `vratio`). Do not treat the `dir` row as the whole
  bundle: directional failure does **not** mean the activity targets failed.

## 3. Diagnose before acting

Choose exactly one branch:

- **Stale/missing cache, failed pipeline, feature-version mismatch** → load
  `cache-incident-response` / `data-quality-audit` / `pipeline-hardening`.
  Fix that; do not sweep.
- **Directional targets at/below baseline, policy negative after costs** →
  price-only hourly data may simply have no directional edge. Check cost
  sensitivity and horizon before spending a sweep. Do not force a direction.
- **Volatility/volume targets beat, direction does not** → the real skill is in
  activity/volatility. Do not promote a directional model just because another
  target works.
- **Current and candidate horizons differ** → align horizons; promotion will
  reject a mismatch.
- **Plausible hypothesis** (feature group, horizon, model set, ticker) → run a
  small sweep that tests it, not a blind grid.

## 4. Sweep

Use `rl-model-lab` `sweep` with an explicit budget:

```text
action: sweep
tickers: BTC/USD
featureSets: market,core
modelSets: reference
horizons: 1
maxConfigs: 8
folds: 4
```

The sweep evaluates the first 80%; the final 20% is reserved. A sweep winner is
**not** promoted yet. If no config beats the baseline consistently, stop and
report the negative result with the artifact path.

## 5. Promote

For a candidate that passed the sweep gate:

```text
action: promote
candidate: <path from the sweep artifact>
ticker: BTC/USD
```

Dry-run by default. If the gate fails, report every reason and stop. If it
passes, ask the human before `apply: true`.

## 6. Make it durable

`--apply` writes `models/<slug>.json`, but the scheduled train overwrites it.
To keep the win:

1. Read the promotion artifact's `recommendation`.
2. Update `pipeline/run.py` so the default training uses the same
   `feature_names`, `model_names`, and `horizon`.
3. Re-run the pipeline / retrain.
4. Verify `models/<slug>.json`, `models/README.md`, and the dashboard.
5. Run the offline tests and `ruff`.

## 7. Record

Report: scope, before/after metrics, policy numbers, gate reasons, artifact
paths, and what changed. Append durable findings to the relevant skill's RUN LOG
or the memory graph. Do not commit or push without explicit direction.

## 8. Unattended mode

When no human is in the loop, do not improvise the orchestration. Use the
driver, which implements phases 0-7 with a file-based handoff:

Run these inside `nix develop`: the venv's numpy/pandas need the devshell's
`LD_LIBRARY_PATH`, and preflight fails loudly without it.

```bash
uv run python tools/model_improve.py plan            # frozen prereg, no execution
uv run python tools/model_improve.py run             # one hypothesis
uv run python tools/model_improve.py run --no-llm    # mechanical only
uv run python tools/model_improve.py status --render # scoreboard + queue
```

- The **criteria and configs come from `experiments/hypotheses.json`**, not
  from the agent. The driver writes them to the append-only ledger before the
  lab runs; the verdict is computed by `rl_crypto.experiments`, never by an
  LLM.
- Roles are separate processes with fresh context (`--executor passes`) or an
  Ensemble lead (`--executor team`). They must not ask questions or wait for
  approval; a blocker goes into the role artifact and stops the run.
- The final 20% is still reserved for `promote`. `promotion.apply` defaults to
  false; a passing gate writes `models/` on the run branch only, and CI
  delivers that as a PR. Never set it true to "see what happens".
- `experiments/` is committed, including negative results. A blocked run is a
  ledger entry, not a silent retry.
- The forward scorer does not exist yet, so nothing from this loop is
  `confirmed`; a backtest pass is at most `provisional`. Say that explicitly.
- **Model:** one knob, `config/model-improvement.json` → `model`
  (default `opencode/deepseek-v4.1-flash-free`). The role agents have no
  hardcoded model; `--model` / `RL_MODEL_IMPROVE_MODEL` override it, and the
  lead passes it to every `team_spawn`.
- **Reviewer:** after the verdict, a mechanical audit plus
  `model-improve-reviewer` check the ensemble session, report honesty and final
  git state. A `fail` blocks the run; the reviewer may fix a concrete defect in
  `.opencode/` and must record it in `review.json`'s `config_changes`. It may
  not change the model policy or the hypothesis queue.

## RUN LOG

### 2026-10-24 — unattended orchestrator + mechanical verdict

- Built `tools/model_improve.py` and `src/rl_crypto/experiments.py`: the
  pre-registration, trial counter, holdout accounting, criteria evaluation and
  ledger rendering are now code, not agent judgment. The five role contracts
  live in `.opencode/agents/model-improve-*.md`.
- The evaluator is intentionally the weakest role: it may only write prose.
  Promotion is gated by `promotion_gate`, and `apply` defaults off because the
  scheduled retrain still overwrites `models/` (no `config/training.json` yet).
- CI (`model-improve.yml`) publishes a branch + PR and never pushes main; the
  weekly schedule stays off behind `RL_MODEL_IMPROVE_ENABLED`.
- **Trap found by running it, not by reading docs:** `opencode run --agent X`
  silently falls back to the default agent when `X` is `mode: subagent`, so a
  role would run with the wrong permissions. All role agents are now
  `mode: all`; preflight's `agent-modes` check and `run_role`'s log scan both
  treat a fallback as a hard failure.
- Local runs need `nix develop` (the venv's numpy/pandas need the devshell's
  `LD_LIBRARY_PATH`); preflight's `lab-imports` probe fails early instead of
  after a long sweep.
- Added a post-run `model-improve-reviewer` (mechanical checks + an adversarial
  session/git/report audit; may fix concrete `.opencode/` defects and must
  record them) and a single `model` knob
  (`opencode/deepseek-v4.1-flash-free`). The role agents no longer hardcode a
  model: an agent-file `model:` overrides the CLI `--model`, so a single knob
  requires removing it. `--model` / `RL_MODEL_IMPROVE_MODEL` now changes the
  whole team, and the lead passes `model=MODEL` to every `team_spawn`.

### 2026-10-24 — first real run (negative directional, activity trusted)

- A bare `/rl-model-lab` run diagnosed BTC/ETH/SOL: directional/return `0/4`
  folds, policy negative even at zero cost — no gross directional edge.
- The first `status` table had one `holdout` column that was actually
  `fwd_direction` only; the run misread it as bundle-wide and wrongly concluded
  the activity targets failed too. The bundle holdouts say otherwise:
  `fwd_volume_ratio` 15/15 and `fwd_volatility` 10/15 pass.
- Fix: `status` now prints per-target `holdout beats`
  (`ret/D dir/D vol/D vratio/D`) and `--json` carries `target_holdouts`; the
  single `holdout` field is labelled `fwd_direction`.

### 2026-10-24 — H1 run blocked by a false-positive fallback detector

- `20261024T155909Z-h1` (H1 pooled panel) preregistered `e0011`, then blocked
  at the lead stage (`e0012`, exit 2). The recorded reason was
  "opencode refused model-improve-lead (subagent)", but `model-improve-lead`
  is `mode: primary` and preflight's `agent-modes` check passed minutes before.
  The real cause is `run_role`'s detector: it greps the whole role transcript
  for the literal string `is a subagent, not a primary agent`, and the lead
  read `tools/model_improve.py` (which contains that literal on line 697), so
  the detector matched its own source echo. Fix belongs in
  `tools/model_improve.py`: inspect only the stderr banner / a structured
  field, or anchor the match to the `opencode` error line, never arbitrary
  file output the role printed.
- The same run exposed a prompt bug now fixed in
  `.opencode/agents/model-improve-lead.md`: the lead was told it was "already
  inside a dedicated worktree", but CI and `--no-worktree` runs execute in the
  repo itself. The lead worked around it by creating branches and
  `git reset`-ing staged run inputs. The lead now checks the layout and, for
  in-place runs, leaves the commit to the orchestrator.
- Separate branch `model-improve/2026-10-24-h1` (worktree
  `../rl-crypto-h1`) already holds a completed H1 pooled-panel trial
  (preregister 14:28Z, result 14:40Z) whose primary target `fwd_volume_ratio`
  was **negative** (panel-vs-control paired CI95 includes 0) and whose
  `fwd_direction` / `fwd_return` improvements were negative. That result is
  not merged into `main`, so `e0011` still counts as trial 1. A future H1 run
  must cite it instead of silently re-testing the same panel idea; if the
  panel code is re-landed, the trial counter has to advance.

### 2026-10-24 — H0 smoke run (`team` executor): diagnostic semantics and reviewer inputs

- `20261024T161936Z-h0` ran H0 a **fourth** time (`e0014`→`e0016`) on the same
  `config_hash` and data hashes; `e0016` reproduces `e0009` to the digit
  (BTC/ETH/SOL direction 0/4 folds, Sharpe -18.46/-17.98/-9.24). Treat a
  duplicate diagnostic as a control only. Do not spend a fifth trial on the
  same archive: if a new H0 is queued, either cite the existing verdict or
  require new data/features first.
- `kind: diagnostic` is **non-gating**. `evaluate_criteria` returns
  `passed = bool(rows)` for a diagnostic and does not apply
  `folds_beating_min` / `tickers_beating_min` / `mean_improvement_min`, while
  `build_report` prints "all pre-registered bars met" when `reasons` is empty.
  Read the per-ticker matrix, not the `passed` flag, and never quote a
  diagnostic `passed: true` as evidence of directional edge.
- `evaluate` uses `reserve_final = 0.0`, so its walk-forward test windows cover
  the newest rows that `promote` reserves. That is fine for a diagnostic that
  selects nothing, but it means the tail has been *seen*; a later promote on
  the same archive is not a clean first look. Keep saying `holdout_touched:
  false` only when no selection or promotion used it.
- `report.md` is written by `run_review()`'s caller **after** the reviewer
  finishes, so the reviewer cannot read it. Review honesty from `lead.json`,
  `diagnosis.md`, the lab payload and the ledger. Also: the root
  `lab-*.json` file is a stdout capture ending in `--- STDERR ---` and is not
  valid JSON; the parseable artifact is `RUN_DIR/lab/evaluate-*.json`.
- No-code runs: the orchestrator requires only `researcher.json` +
  `diagnosis.md`. The lead may record no-op `implementation.json` / `audit.json`
  itself, but they are not an independent audit and must be labelled
  lead-recorded. `roles.json` records only the lead in team mode; the spawned
  teammate's transcript is not captured under `logs/`, so verify it ran from
  the spawn event, the artifact mtimes and `team_results`.

### 2026-10-24 — harness hardening: the real false-positive sources

- **The holdout guard never fired.** `build_preregistration` set
  `lineage = "<H>-<run_id>"`, so every run was a new lineage and
  `evaluate_mechanical`'s `holdout_touches` check was always empty. The
  hardening uses a stable lineage (`hypothesis.stable_lineage`), a one-time
  `holdout_freeze` entry and distinct-run touch counting; a new lineage needs a
  new hypothesis id or a human `holdout-rotate`.
- **The reserved tail moved with the archive.** `evaluate` used
  `reserve_final=0.0` and `sweep` used `0.2` (a fraction of the current bars),
  so "the final 20%" was a different timestamp after every append. New runs
  freeze a `holdout_start` timestamp and all lab actions filter on it.
- **`fold_rule` was contradictory.** `folds_beating_min: 3` with
  `min_improvement_min: 0` means every fold must be positive; the 3 was
  decorative. Criteria v2 states `fold_rule: all_positive` explicitly and keeps
  v1 entries on the old interpretation.
- **Diagnostics are not verdicts.** A diagnostic `passed: true` only means rows
  were produced. Read the per-ticker matrix, never quote it as skill.
- **The measurement code was editable by the roles.** The implementer and lead
  could edit `experiments.py`, `model_lab.py` and their tests. Those files are
  now protected, hashed in the pre-registration, and a protected diff fails the
  mechanical review; only a human-approved `harness-change` may touch them.
- **The reviewer prompt was stale.** `run_review()` is preceded by a provisional
  `report.md`; the reviewer can read it. The final report is rebuilt afterwards.
