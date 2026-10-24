---
name: backtest-protocol
description: >-
  Use when designing, reviewing, or planning a backtest or strategy evaluation
  for rl-crypto, especially as Tier 3 / RL lands. Triggers: "backtest",
  "strategy evaluation", "walk-forward", "purged split", "transaction costs",
  "RL environment", "evaluate a signal", "Sharpe", "out-of-sample". Defines a
  leakage-safe, reproducible evaluation contract before implementation.
  Status: forward-looking — no backtest code exists in the repo yet.
---

# Backtest protocol (forward-looking)

**Status gate: this repo has no backtest implementation today.** Tier 3 is
explicitly out of scope in the README; there is no `backtest/` module, no
execution simulator, no cost model, and no experiment registry. This skill
defines the contract to build against so the first implementation does not bake
in leakage or irreproducibility.

Do **not** claim a backtest result from this skill. It is a design and review
checklist plus a memory of decisions made when Tier 3 starts.

Companions: **`model-contract`** (holdout/feature contract),
**`data-quality-audit`** (point-in-time archive), **`pipeline-hardening`**
(runner/cache robustness).

The first harness to extend is **`rl-model-lab`** (`tools/model_lab.py`): it
already has walk-forward splits, per-side costs, baselines, a reserved final
holdout, and an experiment ledger. Add the registry/cost config there rather
than starting a second harness.

## 0. Contract

- **No lookahead.** A decision at time `t` may use only data available at `t`.
  That includes the target horizon: a label spanning `t..t+h` is not known at
  `t`.
- **Chronological evaluation.** Train on the past, evaluate on the future.
  Never shuffle rows across time.
- **Purge/embargo overlapping horizons.** If labels overlap (a 1-hour label at
  every 5 minutes), adjacent train/test rows leak. Purge the overlap and add an
  embargo.
- **Costs are part of the result.** Fees, spread, slippage, funding, and
  turnover change the sign of a marginal strategy. A zero-cost backtest is a
  diagnostic, not a result.
- **Baselines are mandatory.** Buy-and-hold, always-up/always-down, and a random
  policy. A strategy that does not beat the relevant baseline has no claim.
- **Reproducibility is a requirement.** Record the code SHA, data snapshot
  SHA/date range, config, seed, and artifact paths. A result without those is
  an anecdote.
- **Multiple testing is real.** Trying many signals/windows and reporting the
  best one is fitting the test set. Pre-register or correct.
- **Absence of evidence is not evidence.** A withheld signal / `None` prediction
  stays `None`; do not impute a trade.

## 1. What already exists

| Piece | State | Source |
| --- | --- | --- |
| Causal T0 features | implemented | `src/rl_crypto/features.py` |
| Forward targets | implemented | `src/rl_crypto/targets.py` |
| T1 models + holdout | implemented | `src/rl_crypto/ml/tier1.py` |
| T2 outlook + signal gate | implemented | `src/rl_crypto/ml/tier2.py` |
| RL observation seam | implemented as a flattening helper | `Outlook.as_features(prefix="pred_")`, `ml/types.py:73-89` |
| Dataset schema | implemented | `src/rl_crypto/dataset.py` (OHLCV + sentiment + has_price/has_news) |
| Backtest engine / costs / execution | **absent** | — |
| Experiment registry / report | **absent** | — |
| Tier 3 RL environment | **absent** | README says out of scope today |

**Current evidence to start from:** the directional models do not beat a
constant baseline out of sample (`models/README.md`; `fwd_return` 0/15,
`fwd_direction` 0/15). A backtest of those signals should therefore be expected
to fail; the first job is to prove the harness can detect that, not to tune until
it passes.

## 2. Design checklist

### 2.1 Data

- **Point-in-time:** use the record as it existed at that timestamp, not a later
  corrected version. The committed archive is the closest available; document
  when a bar was revised.
- **Survivorship:** the watchlist is curated and changes. A backtest over the
  current 15 tickers is not the same universe as the live history; record the
  universe and its effective dates.
- **Timestamps:** all UTC; align decision time to bar close, not bar open. A
  live snapshot at `21:05` cannot use the `21:00` bar's close as if it were
  known at `21:00:00` unless the bar is complete at that instant.
- **Gaps:** `None` is a gap. Do not forward-fill a price for a missed bar and
  pretend it was tradeable.
- **Latency:** a live decision has fetch/processing latency. The pipeline's live
  record is not an exchange-timestamped tick; state the assumption.

### 2.2 Splits

- **Walk-forward / expanding window:** train on `[0, t)`, evaluate on `[t, t+w)`,
  roll forward. No random split.
- **Purge:** drop training rows whose label horizon overlaps the test window.
- **Embargo:** drop a small buffer after the test window before the next train
  window to avoid serial correlation.
- **Final holdout:** keep a completely untouched recent block for the final
  report; do not use it for model selection.
- **Regime split:** report bull/bear/high-vol slices separately; an aggregate
  Sharpe can hide a strategy that only works in one regime.

### 2.3 Signal and execution

- **Decision rule:** map the model output to a position deterministically
  (threshold, sizing, max position). Record it.
- **Costs:** maker/taker fee, spread, slippage model, funding for perps, and
  turnover. State whether costs are per-trade or per-notional.
- **Execution:** fill at the next available price, not the signal bar's close,
  unless the latency assumption explicitly allows it.
- **Constraints:** no shorting, max leverage, min notional, borrow availability.
  A backtest that ignores them is an upper bound, not a strategy.
- **Turnover:** report it; a high-Sharpe strategy that trades every bar may be
  untradeable after costs.

### 2.4 Metrics

Report at least:

- cumulative return / equity curve, with and without costs;
- annualised Sharpe (state the risk-free rate), Sortino, max drawdown, Calmar;
- hit rate and average win/loss;
- calibration for directional probabilities (Brier/log loss), not just accuracy;
- turnover and exposure;
- per-regime and per-ticker breakdown;
- confidence intervals / bootstrap, not a single point estimate.

### 2.5 Baselines

- **Buy-and-hold** the same universe.
- **Always-up / always-down** for directional signals.
- **Random policy** with the same turnover.
- **The T2 withheld signal** (do nothing) — the honest default today.

### 2.6 Reproducibility

Record in one artifact:

- code SHA and branch;
- `data/` snapshot commit SHA and date range;
- universe/watchlist and effective dates;
- feature version, model bundle SHA/`trained_at`;
- split windows, purge/embargo, seed;
- cost/fee/slippage config;
- metrics JSON and the report path;
- the command that produced it.

### 2.7 Leakage checklist

- [ ] No feature reads a row after its own timestamp.
- [ ] No label's future is known at decision time.
- [ ] No train row's label overlaps the test window (purge/embargo).
- [ ] No model selection on the final holdout.
- [ ] No backfilled/revised bar used as if it were live.
- [ ] No `feature_version` mismatch served.
- [ ] No `None` prediction imputed to a trade.
- [ ] No target-derived column in the features.

## 3. When Tier 3 starts

1. Add a `backtest/` module (or a clearly named package) with:
   - a splitter (walk-forward + purge/embargo);
   - an execution/cost model;
   - a metrics/report layer;
   - a baseline runner;
   - an experiment registry.
2. Add a CLI command (e.g. `rl-crypto backtest`) with a config file, not a pile
   of flags. Keep the read-only default; require an explicit output directory.
3. Add fixtures: a deterministic tiny price path with a known edge and a known
   no-edge case. The harness must find the edge and reject the noise.
4. Add leakage tests: shuffle the future, assert the score collapses; shift the
   timestamps, assert the split purges.
5. Update this skill with the real commands, the first experiments, and the
   results (including negative ones).

Do not build the RL environment before the evaluation harness can prove a
trivial baseline and reject a known-leaky strategy.

## 4. Update this file

When the harness exists:

- Replace the status gate with the real entry point and command.
- Add the experiment registry/baseline results to the memory.
- Keep the negative results; they are the most valuable record.

## Memory — baseline

**As of 2026-10-23T23:18Z** (`HEAD` `213d17b`):

| Item | Value |
| --- | --- |
| Backtest code | **none** |
| Experiment registry | **none** |
| Cost/execution model | **none** |
| Tier 3 / RL environment | **out of scope today** |
| Directional skill | `fwd_return` 0/15, `fwd_direction` 0/15 beat baseline |
| Trusted targets | `fwd_volume_ratio` 15/15, `fwd_volatility` 10/15 |
| Observation seam | `Outlook.as_features(prefix="pred_")` |
| Status | design contract only; no results |

**Rule:** until a harness exists, any "backtest" claim is a hypothesis, not a
measurement.

## Memory — active findings

### BT-001 · No backtest implementation exists · EXPECTED (since 2026-10-23)
- **Layer:** repo/status (README Tier 3; no `backtest/`)
- **Evidence (2026-10-23):** no backtest module/CLI/experiment registry in `src/`, `tools/`, or `pyproject.toml`; README says Tier 3 is deliberately out of scope.
- **Read:** this skill is a forward-looking contract. Do not present its checklist as an implemented result; build the harness before claiming one.
- **Map:** IM-1
- **Status:** expected until Tier 3.

### BT-002 · `Outlook.as_features` is the RL observation seam · EXPECTED (since 2026-10-23)
- **Layer:** T2 (`src/rl_crypto/ml/types.py:73-89`)
- **Evidence (2026-10-23):** the method flattens `pred_<target>`, `pred_prob_<target>`, `pred_signal`, `pred_confidence`, preserving `None`.
- **Read:** changing/removing/reordering keys changes the RL observation width. Version the key set when Tier 3 lands; do not silently change it in a model tweak.
- **Map:** IM-7
- **Status:** expected.

### BT-003 · Directional models have no out-of-sample skill · EXPECTED (since 2026-10-23)
- **Layer:** model evidence (`models/README.md`; `data/*/summary.json`)
- **Evidence (2026-10-23):** `fwd_return` 0/15 and `fwd_direction` 0/15 beat the constant baseline; T2 withholds the signal.
- **Read:** a backtest of a withheld signal should do nothing, not trade. The first harness test is a **negative control**: it must show that the current directional signal adds no edge.
- **Map:** IM-5
- **Status:** expected.

### BT-004 · Overlapping horizons require purge/embargo · PROPOSED (since 2026-10-23)
- **Layer:** evaluation design
- **Evidence (2026-10-23):** targets are forward-shifted by `horizon` intervals and the pipeline can produce rows at a finer cadence than the horizon; a naive row split leaks adjacent labels.
- **Read:** a 1-bar horizon at hourly rows has no overlap; a shorter decision cadence or a longer horizon does. The splitter must purge the overlap and add an embargo.
- **Map:** IM-3
- **Status:** proposed; implement with the splitter.

### BT-005 · No cost, slippage, or funding model · PROPOSED (since 2026-10-23)
- **Layer:** execution design
- **Evidence (2026-10-23):** no cost/fee/slippage configuration exists in the repo; the pipeline records market prices, not fills.
- **Read:** a zero-cost backtest is an upper bound. At 30-minute/hourly crypto cadence, fees and spread can dominate a small edge.
- **Map:** IM-2
- **Status:** proposed.

### BT-006 · No experiment registry or reproducibility artifact · PROPOSED (since 2026-10-23)
- **Layer:** tooling design
- **Evidence (2026-10-23):** no config/seed/SHA/results record exists for a backtest; model summaries are per-run, not per-experiment.
- **Read:** without a registry, results cannot be compared or reproduced. The first implementation must write one JSON record per run.
- **Map:** IM-4
- **Status:** proposed.

### BT-007 · No multiple-testing discipline · PROPOSED (since 2026-10-23)
- **Layer:** evaluation design
- **Evidence (2026-10-23):** no pre-registration/log of attempted signals/windows; the repo has 15 tickers and 4 targets, which is already a search space.
- **Read:** reporting the best of many attempts is fitting the evaluation set. Pre-register the hypothesis or report the number of attempts and a corrected interval.
- **Map:** IM-4, IM-5
- **Status:** proposed.

## Memory — resolved & expected

- **R-001 · The existing holdout is the right prior art** — chronological, newest rows, constant baseline, 1% margin (`models/README.md`). A backtest must not be weaker than that.
- **R-002 · `None` propagates through the stack** — a missing feature/prediction is `None`, never a fabricated `0`; the backtest must not impute a trade.
- **R-003 · The signal is withheld while untrusted** — "do nothing" is a valid policy and the current default; the harness must be able to score it.
- **R-004 · The archive is the point-in-time record** — `data/<slug>/candles.jsonl` is the committed, upserted history; use it (and its git history for revisions), not a re-fetch.
- **R-005 · The feature contract is versioned** — a backtest must record `feature_version` and the bundle's `trained_at`; a mismatch is not a valid experiment.

## Memory — improvement map

| ID | Finding | Layer | Action | Target | Risk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| IM-1 | BT-001 | repo | Build a `backtest/` package + CLI when Tier 3 starts: splitter, execution, metrics, registry, baselines | new module; `pyproject.toml` | scope; keep read-only default | proposed |
| IM-2 | BT-005 | execution | Add a cost/fee/slippage/funding config; report gross and net | backtest config | wrong defaults hide edge | proposed |
| IM-3 | BT-004 | evaluation | Implement a purged/embargoed walk-forward splitter | backtest module | leakage if omitted | proposed |
| IM-4 | BT-006, BT-007 | tooling | Write one experiment record per run: code SHA, data SHA, config, seed, attempts, metrics | backtest registry | none | proposed |
| IM-5 | BT-003, BT-007 | evaluation | Ship mandatory baselines (buy-and-hold, always-up/down, random, do-nothing) and negative controls | backtest module/tests | none | proposed |
| IM-6 | all | tests | Add leakage tests: shuffled future collapses, shifted timestamps purge, `None` never trades | backtest tests | none | proposed |
| IM-7 | BT-002 | T3 | Version the `as_features` key set and add a width/schema test before RL training | `ml/types.py` | observation-width break | proposed |

## Memory — audit log

| Run (UTC) | HEAD | Backtest code | Experiments | Verdict |
| --- | --- | --- | --- | --- |
| 2026-10-23T23:18Z | `213d17b` | none | 0 | contract defined; no implementation or results |

## Memory — open questions

- **Q-1** Which venue/cost model is the reference — spot fees + spread, or perps with funding? The answer changes the strategy class.
- **Q-2** What is the decision cadence and horizon? The pipeline supports 1h and a 1-bar horizon; a backtest at a finer cadence needs a different label alignment.
- **Q-3** What walk-forward window sizes and embargo length are defensible for ~1,700 bars per ticker? Document the choice, not just the default.
- **Q-4** Is the first goal supervised signal evaluation or a full RL environment? Build the evaluator first; it is the ground truth for both.
- **Q-5** How is the universe handled over time (15 curated tickers, survivorship, listing dates)? Record effective dates per asset.
- **Q-6** What is the success metric beyond Sharpe — calibration, drawdown, turnover-adjusted return, or capacity?
- **Q-7** How does the policy treat a withheld signal or a `None` prediction — flat, hold, or skip? Define it before measuring.

