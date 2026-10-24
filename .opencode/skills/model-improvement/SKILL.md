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

## RUN LOG

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
