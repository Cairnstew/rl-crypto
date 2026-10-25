# Honest-gate round — task blocks

Base branch: `coverage-power/2026-10-25` (09247f1).
Integration branch: `honest-gate/2026-10-25` in `/home/seanc/Projects/rl-crypto-honest-gate`.
No experiments, no promote, no holdout reads, no changes to `.github/`, `data/`,
`models/`, or the real `experiments/ledger.jsonl`. Protected-file changes follow the
harness-change process and update the pin. Tests must pass at every commit.

<!-- T1 -->
## T1: Separate statistical power from economic feasibility
a. Split minimal_meaningful_effect into (i) `quality_mme`, a pre-registered
   model-quality improvement the design must be able to detect, and (ii) an
   `economic_feasibility` check reporting the hit rate / correlation needed to
   cover round-trip costs at the declared horizon and threshold.
b. If the required hit rate exceeds a configurable ceiling (default 0.60),
   block direction/return hypotheses with `economically-infeasible` (no trial
   consumed), reporting the horizon at which the ceiling is first met.
c. Provisional/negative/inconclusive for direction use quality_mme; economic
   success is a separate gate reported alongside.
d. Show H1 and H2 as currently declared are blocked or re-declared, and say
   what they test.
Report: feasibility table by horizon (1h, 4h, 24h) and H1/H2 status.
<!-- /T1 -->

<!-- T2 -->
## T2: Re-triage by target track (after T1 is merged)
Add `track` (activity | direction) to hypotheses. Activity = fwd_volume_ratio
and fwd_volatility, compared against the strongest baseline. Re-evaluate H3
and H9 for the activity track; keep direction-only hypotheses deferred with a
re-evaluation condition (horizon feasibility from T1). Record via ledger note.
Mark H7 (lower frequency) as the direction track's unblocker and state its
prerequisites (interval metadata, non-overlapping holding evaluation).
Report: queue table (id | track | status | reason).
<!-- /T2 -->

<!-- T3 -->
## T3: Baselines and intervals
Add fitted HAR-RV (OLS on 1h/24h/168h realised-volatility components, fit on
the training slice) to the strongest-baseline set. Add a 24-bar-window
volatility target variant as a diagnostic only (no ledger trial). Report skill
vs constant/seasonal/persistence/EWMA/HAR/fitted-HAR for both target variants
on BTC, ETH, SOL, each with the v3 interval. No "edge" label unless the
interval excludes zero.
Report: baseline table with intervals.
<!-- /T3 -->

<!-- T4 -->
## T4: Calibration under realistic noise
Re-run calibrate_coverage with t-distributed innovations (df 3 and 5) and
GARCH-style volatility clustering across the previous grid (rows 300/600/1700,
rho 0.5/0.8/0.95, alphas 0.01/0.05/0.1/0.2). Confirm the one-sided tail used by
the decision rule. Report failing cells and refuse the claim wherever
coverage is anti-conservative. Add code in new modules/tests where possible.
Use a quick mode during development and the full grid once at the end.
Report: coverage table (rows | rho | noise | alpha | excess | status).
<!-- /T4 -->

<!-- T5 -->
## T5: Production trust criterion behind a flag
Implement, default OFF, a Tier-1 verdict that uses the strongest baseline
(seasonal for volume_ratio, HAR for volatility). Produce a before/after table
of trusted counts per ticker and target from the committed bundles WITHOUT
writing models/ or data/. Human approval is needed to enable it.
Report: trusted-count before/after table.
<!-- /T5 -->

<!-- T6 -->
## T6: Upstream history probe (read-only)
Determine, without writing data/, the maximum hourly history tickerkit can
return per ticker and any pagination or rate-limit constraints. State what H2's
design-scaled assumptions should be given what is actually available. Edit no
code; write findings only to the results directory.
Report: per-ticker max history and constraints.
<!-- /T6 -->

<!-- T7 -->
## T7: Records and canary refresh (after all merges)
Fix the empty positive-canary reason text. Update README/docs for the track
model and the feasibility gate. Run canary-refresh at the final protected hash
and report both outcomes.
Report: canary outcomes and the scoreboard canary line.
<!-- /T7 -->

<!-- V -->
## V: Verifier (fresh context, read-only except its own worktree)
For the integration branch, check out each teammate commit and each merge
point in a scratch worktree and record `pytest -q | tail -n 1` and ruff status
per commit. Confirm `experiments/ledger.jsonl` is unchanged apart from
intended notes, and `git diff --name-only <base>..HEAD` touches nothing under
.github/, data/, or models/. Write a table to results/V/report.md.
<!-- /V -->

<!-- A -->
## A: Auditor (fresh context)
Read only `git diff --stat` and the diffs of protected files on the
integration branch. Check: any look-ahead, any fit-on-evaluation-data, any
self-reported contract replacing a structural check, any weakened gate, the
pin matching the protected list. List blocking and non-blocking findings with
file:line. Write to results/A/report.md.
<!-- /A -->
