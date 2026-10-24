---
description: Verifiable model-improvement loop for rl-crypto. One pre-registered hypothesis per run, judged out of sample, logged forever.
---

# /model-improve $ARGUMENTS

You are running the rl-crypto model-improvement loop. The goal is a model
stack that gets measurably better over time, and measurably means: every claim
is backed by a ledger entry, a git SHA, a data hash, and an out-of-sample
number that was committed to BEFORE the experiment ran. A negative result is a
valid, logged outcome. Producing noise that looks like progress is the only
failure.

## Modes (from $ARGUMENTS)

| Argument      | Behaviour                                                                                                                     |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| empty         | Run one full loop (Phases 0-7) on the next untried hypothesis in the queue.                                                   |
| `foundation`  | Build the missing measurement infrastructure listed in Phase F. Nothing else.                                                 |
| `status`      | Read-only. Print the scoreboard and the last 10 ledger entries. Stop.                                                         |
| `verify`      | Read-only. Run only the forward test (Phase V) for the current champion and any provisional candidates. Stop.                 |
| `H<n>`        | Run one full loop on that specific hypothesis, if it is untried or has new data / a new feature class since its last attempt. |
| anything else | Treat as a free-text goal, but still map it to a hypothesis ID and pre-register it.                                           |

## Hard rules

1. **Isolation.** Work ONLY in a dedicated git worktree on a branch named
   `model-improve/<date>-<hypothesis-id>`. Never edit the main working tree,
   never touch uncommitted or untracked files you did not create, never touch
   branches, worktrees or spaces used for upstream development. Never merge,
   push to main, or write `models/` on main. Your output is a branch plus a
   report; the human merges.
2. **Pre-registration.** Before any experiment executes, write the hypothesis,
   the exact configs, the metric of record, and the pass/fail criteria into the
   ledger as a `preregister` entry. Criteria may not change after results are
   seen. If they must change, that is a new hypothesis with a new entry and an
   incremented trial count.
3. **Holdout discipline.** The final holdout is touched at most once per
   candidate lineage, by `promote` only. All tuning, selection and debugging
   use walk-forward folds on data strictly before the holdout (with the purge
   from Phase F). If a candidate fails the holdout, that lineage is closed;
   iterating on it afterwards means a new lineage that counts as a new trial
   against the same holdout and needs the stricter threshold in Phase 5.
4. **Trial budget.** Max 12 configs per run, max 1 hypothesis per run, and a
   cumulative per-family trial counter in the ledger. Never retry a hypothesis
   family without new data, a new feature class, or a fixed measurement bug.
5. **Honest costs.** Report every economic result at the default 10 + 5 bps and
   at zero cost (as a diagnostic only). Never promote on a zero-cost result.
   Never lower costs to make something pass.
6. **No pipeline surgery without a gate.** Changes to `.github/workflows/`,
   the scheduler, the live recorder, secrets, or the committed `data/` are out
   of scope. Propose them in the report; do not make them.
7. **Evidence.** Every claim in your report cites a ledger id, an artifact
   path, or a `file:line`. Unverified items are labelled unverified.
8. **Stop rules.** Stop and report if: tests fail on your branch and you
   cannot attribute it to your change; the archive fails the audit
   (`tools/audit_data_quality.py` reports gaps/dupes/OHLC violations); you
   catch yourself adjusting criteria after seeing results.
9. **Staleness is scoped, not a blanket stop.** Check the newest archive bar
   in Phase 0 and treat it as follows:
    - Offline walk-forward experiments (Phases 1-6 on historical bars): a stale
      tail is NOT a blocker. Run them, and record the newest bar timestamp and
      staleness in the preregister and results entries.
    - Forward-test claims (Phase V) and `promote --apply`: blocked while the
      archive is more than 6 hours stale, because new predictions are not
      being recorded and the forward clock is not advancing. Report the gap
      instead of claiming forward results.
    - A stale archive is a **pipeline incident**: report it as its own finding
      (what is stale, since when, what it blocks, what you did and did not
      verify) with a proposed investigation. Do not edit workflows, dispatch
      runs, or attempt a fix (hard rule 6).

## Team (optional)

If opencode's team/space orchestration is available, create a NEW space for
this run, separate from any space used for upstream development, and never
share branches with it. Use these roles. If not available, run the same roles
as sequential, separate passes and do not let one pass see another's
reasoning, only its written output.

- **Lead** (you): owns the ledger, the pre-registration, and the final report.
- **Researcher**: Phase 1-2. Diagnoses, reads the ledger, proposes the
  hypothesis and its criteria. Read-only.
- **Implementer**: Phase 3. Writes code and tests on the branch. May not run
  `promote` and may not edit the ledger.
- **Leakage auditor** (adversarial, fresh context): Phase 4. Gets the diff and
  the pre-registration only. Tries to break the experiment: look-ahead in new
  features, fit-on-test preprocessing, label overlap across the split, selection
  on the holdout, cost model errors. Its findings are blocking.
- **Evaluator**: Phase 5-6. Runs the pre-registered configs, computes the
  verdict mechanically from the criteria, and writes the artifact. May not
  change criteria.

## Unattended mode

If no human is in the loop, do not hand-drive Phases 0-7. Use
`/model-improve-auto` or `tools/model_improve.py`:

```bash
uv run python tools/model_improve.py run --executor team   # or passes/mechanical
```

That driver owns pre-registration, the role handoffs, the mechanical verdict
and the ledger commit. It cannot change criteria after seeing a result, cannot
let an LLM decide pass/fail, and writes `models/` only when
`config/model-improvement.json`'s `promotion.apply` is true and the gate passes.
This command remains the human-in-the-loop review path.

## Phase 0: Preflight (read-only)

1. `git status --short`, current branch, list worktrees. Record any unrelated
   changes and leave them alone.
2. Read the ledger (`experiments/ledger.jsonl`) and scoreboard
   (`experiments/SCOREBOARD.md`) if they exist.
3. Run `model_lab.py status` and the data audit. Confirm: all tickers present,
   feature version matches bundles, no audit violations. Record the newest
   bar timestamp and staleness in hours, and apply hard rule 9 to decide what
   it does and does not block.
4. Confirm each Phase F prerequisite exists on the branch you will build from
   (check by presence and by running its tests; cite evidence). Prerequisites
   are: `label_purge` shared by trainer and lab, `rl_crypto.experiments`
   holdout accounting enforced in Python, lab/production model parity,
   `config/training.json` read by production, `tools/forward_score.py`,
   cost-grid batching, tracked lab files, and `experiments/` records. If the
   foundation branch is unmerged, build from it, not from main, and say so.
   If any item is missing, switch to `foundation` mode and build only the
   missing items.
5. Create the worktree and branch.

## Phase F: Foundation (`foundation` mode, or automatically when Phase 0.4 finds gaps)

No approval is needed to build this, because it happens on a branch and
nothing merges. Human approval is required only to merge the branch. Build
only items that are missing, in priority order, each as its own small commit
with tests. These are the known measurement gaps; until they are closed,
improvement claims are not verifiable. When foundation work is done, stop;
do not start an experiment in the same run.

1. **Close the volatility boundary leak.** The purge between train and holdout
   must remove `max(horizon, 2)` rows for `fwd_volatility` (and the trainer's
   split must not let a training label read holdout bars). Add a test that
   covers `fwd_volatility`, not just return/direction.
2. **Trial counter and holdout-reuse tracking.** Ledger entries carry:
   hypothesis id, family, lineage id, trial index within family, git SHA, data
   hash (hash of the archive files used, per ticker), config hash, seed, and
   whether the holdout was touched. `promote` refuses a lineage whose holdout
   was already used, and `status`/`report` surface the counts. Enforce this in
   the Python engine, not only in the TypeScript wrapper.
3. **Lab/production parity.** The lab's `reference` model set must be able to
   evaluate exactly what production deploys (including `log_ridge` for
   volatility, log-space baselines, and per-target defaults), or the lab
   verdict is not comparable to the bundle.
4. **Durable promotion.** Production training must read its feature set, model
   set and per-target model choice from a committed config file that `promote
--apply` edits on the branch, so a promoted candidate survives the next
   scheduled retrain. Without this, promotion is cosmetic.
5. **Forward scorer.** `tools/forward_score.py`: for each ticker, join the
   predictions recorded in `data/<slug>/live.jsonl` with realized outcomes once
   the horizon has elapsed, and report per-target skill vs the constant
   baseline and after-cost policy PnL, per champion-bundle version. This is
   the one test that cannot be overfit, because the predictions were made
   before the outcomes existed.
6. **Cost-grid batching.** `evaluate` accepts a list of thresholds and cost
   levels and fits each fold once.
7. **Commit the lab.** `tools/model_lab.py`, its tests and `.opencode/` are
   untracked; they must be tracked before experiments are reproducible.
8. **Experiment records in the repo.** Create `experiments/ledger.jsonl`
   (append-only), `experiments/SCOREBOARD.md` (rendered from the ledger by a
   script, never hand-edited), and `experiments/hypotheses.md` (the queue
   below, with status).

Report what was built, what tests cover it, and what remains. Stop.

## Phase 1: Diagnose (Researcher, read-only)

1. Current champion per target, from the scoreboard (not from memory).
2. Baseline evaluate across all tickers with default costs, 4 folds, horizon 1,
   via the lab. Note fold-level and ticker-level spread, not just means.
3. Forward-test the champion (Phase V) so the diagnosis includes real
   out-of-sample behaviour, not only backtests.
4. Write a short diagnosis: which targets have skill, which do not, how large
   the gap to breakeven is after costs, and what the sample size limits you to
   (about 1,700 hourly bars per ticker; most per-ticker conclusions have wide
   confidence intervals, say so).

## Phase 2: Choose and pre-register one hypothesis

Pick the first untried or unblocked hypothesis from the queue below, unless
Phase 1 gives a specific evidence-based reason to reorder (cite it). Write the
`preregister` ledger entry containing:

- hypothesis id, family, lineage id, trial index
- the mechanism: why should this help, in two sentences
- exact configs to run (cap 12) and which tickers
- metric of record and the control it must beat (champion, constant baseline,
  and for policies the momentum baseline)
- pass/fail criteria (see Phase 5), written as numbers
- what result would count as a negative finding

### Hypothesis queue (ordered by expected payoff; update statuses in hypotheses.md)

- **H1 Pooled panel model.** One model trained on all 15 tickers (per-ticker
  normalisation, ticker-time purged folds), evaluated per ticker. Biggest lever
  against the 1,700-bar sample limit. Needs a panel mode in training and
  evaluation.
- **H2 Longer history.** Backfill more bars per ticker (paginated fetch, as far
  as the upstream allows) and re-run the current best config. Isolates the data
  effect from the model effect. Verify upstream limits before planning.
- **H3 Cross-asset features from the existing archive.** BTC/ETH lead-lag
  returns, relative strength vs a market proxy, cross-sectional rank of
  momentum and volume. No new data needed, only a feature-version bump.
- **H4 Cost-aware policy.** Hysteresis (enter above, exit below a lower
  level), minimum holding period, no-trade band, expected-move > round-trip
  cost filter. Policy-only change, cheap to test, uses existing models.
- **H5 Regime conditioning.** Trade direction only when the (already skilful)
  volume-ratio and volatility predictions indicate favourable conditions;
  size by inverse predicted volatility.
- **H6 Cross-sectional relative direction.** Rank tickers by predicted
  direction, long top / short bottom, which removes the market factor.
- **H7 Lower-frequency horizons (4h, daily).** Requires non-overlapping
  holding periods in the evaluator and interval metadata in the archive.
- **H8 New data classes (funding rates, open interest, order-book imbalance).**
  Requires an upstream capability check first; archive schema change; larger
  scope, needs human approval.
- **H9 Stronger model classes** (boosted trees, small ensembles). Only after H1
  or H2, and with the trial counter.
- **H10 Improve the skilful targets directly** (`fwd_volume_ratio`,
  `fwd_volatility`): feature sets and model class for these targets. Their
  gains are real and verifiable even though they are not directly tradable;
  report them separately from direction results.

## Phase 3: Implement (Implementer, on the branch)

- Smallest change that tests the hypothesis. New feature or schema change:
  bump the feature version and extend tests.
- All new features must use the existing exact-spacing guards and return `None`
  on gaps. No fabricated values.
- Add or update tests that would fail if the new code looked ahead.
- Run `ruff` and the offline test suite on the branch. Do not proceed on red.

## Phase 4: Leakage audit (Auditor, blocking)

Give the auditor the diff and the pre-registration only. It must answer, with
evidence, each of:

1. Does any feature at row i use data after row i?
2. Is any preprocessing (scaling, feature selection, imputation) fit on data
   that includes the evaluation rows?
3. Do training labels overlap the evaluation window across the split?
4. Was anything about the configs chosen after seeing evaluation results?
5. Does the policy PnL charge costs on every position change, including the
   close, and apply the position to the correct bar?
6. Could this run have touched the final holdout before `promote`?

Any "yes" or "cannot tell" blocks Phase 5 until fixed and re-audited.

## Phase 5: Evaluate and decide (Evaluator, mechanical)

Run the pre-registered configs only, with the lab, same folds, same costs.
Decide from the criteria without discretion.

**Minimum bar to be a promotable candidate** (all must hold, per target
claimed):

- Beats the constant baseline AND the current champion in at least 3 of 4
  walk-forward folds on the pooled result, with the paired mean improvement
  positive.
- A bootstrap (block bootstrap over time) confidence interval for the paired
  improvement whose lower bound is above zero at a significance level tightened
  for the cumulative number of trials in this family (Bonferroni: alpha /
  trials-in-family, with alpha = 0.05). State the level used.
- For any economic claim: positive after-cost return AND Sharpe above the
  momentum baseline AND at least 100 trades per ticker pooled (below that, the
  estimate is not meaningful), at default costs.
- No more than a stated fraction of tickers carry the result (report the
  per-ticker distribution; a result driven by one or two tickers is flagged,
  not promoted).
- Sanity: shuffled-label and shifted-label controls on the candidate's best
  config must score near baseline. If the control also "wins", the pipeline is
  leaking and the run is void.

If the bar is not met: record a `negative` entry and stop. Do not run more
configs to find one that passes.

If met: run `promote` as a DRY RUN against the final holdout (once, recorded).
Only a passing dry run makes the candidate **provisional**.

## Phase 6: Record

Append ledger entries (preregister, results with artifact paths, holdout use,
verdict) and regenerate the scoreboard from the ledger. On the branch, commit
code, tests, `experiments/` updates, and (for provisional candidates only) the
config change that makes the candidate durable. Do not commit lab artifacts or
`data/`; record their paths and hashes instead.

## Phase V: Forward test (the verification that cannot be gamed)

Run `forward_score.py` over `live.jsonl` since each bundle version was
deployed. Before trusting any number, check two things and report them:

- **Attribution.** Resolved records under bundle `unknown` (predating
  `bundle_trained_at`) cannot be attributed to a deployed bundle. They are
  context only and never count toward a confirmation window.
- **Baseline parity.** The forward scorer must use the same metric space and
  baseline as the backtest verdict for that target (log-space for
  `fwd_volatility`, with a causal expanding baseline). If the spaces differ,
  forward and backtest skill are not comparable; fix the scorer before
  reading results. A forward volatility result that contradicts a trusted
  backtest verdict is first checked for this mismatch, and only then treated
  as evidence against the verdict.

A candidate is **confirmed** only when, after it has been deployed
for a pre-registered minimum forward window (default 14 days and at least 300
resolved predictions per ticker pooled), its realized skill vs the baseline
meets the same sign and rough magnitude as the backtest. Until then it is
**provisional**. If forward results contradict the backtest, mark the lineage
**refuted**, and treat that as evidence the evaluation protocol has a hole;
file it as a risk, not just a failed run.

Status lifecycle: `preregistered -> negative | provisional -> confirmed |
refuted`. Only `confirmed` changes the champion on the scoreboard.

## Phase 7: Report (single reply)

1. **Outcome**: one line (negative / provisional / confirmed / refuted /
   blocked) for the hypothesis id.
2. **Table**: candidate vs champion vs constant baseline vs momentum, per
   target, with fold counts, CI, and the significance level used.
3. **Economics**: after-cost and zero-cost, trades per ticker, per-ticker
   spread.
4. **Controls**: shuffled-label and shifted-label results.
5. **Verification status**: forward-test state of the champion and every
   provisional candidate.
6. **Trial accounting**: trials this run, cumulative in family, holdout
   touches.
7. **Branch and artifacts**: branch name, commit SHAs, ledger ids, artifact
   paths.
8. **Pipeline status**: newest bar, staleness, what it blocks under rule 9,
   and any incident found, reported separately from the modelling outcome.
9. **Next**: the next hypothesis in the queue and what it needs (data,
   approval, tooling). If something needs approval, say exactly what,
   including whether the foundation branch still needs merging.
10. **Unverified / risks found**: anything you could not confirm, and any new
    measurement hole you found.

Be concise. Do not claim an improvement the ledger does not show. Do not
describe a backtest as a result of the deployed model.
