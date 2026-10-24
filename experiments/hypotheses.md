# Hypothesis queue

The ordered backlog for the model-improvement loop. Each hypothesis is
pre-registered in `experiments/ledger.jsonl` (with numeric pass/fail criteria)
*before* its experiment runs; this file only tracks status and what blocks the
next step. Update the status here when the ledger records a verdict.

Status values: `untried`, `blocked`, `running`, `negative`, `provisional`,
`confirmed`, `refuted`.

| id | hypothesis | status | needs |
| --- | --- | --- | --- |
| H1 | Pooled panel model: one model across all 15 tickers, per-ticker normalisation, ticker-time purged folds | negative | retry only after the shifted-label null is fixed (see below) |
| H2 | Longer history: backfill paginated bars, re-run the current best config | blocked | upstream limit check; fresh archive |
| H3 | Cross-asset features from the existing archive (BTC/ETH lead-lag, relative strength, cross-sectional rank) | untried | feature-version bump + tests |
| H4 | Cost-aware policy (hysteresis, min holding period, no-trade band, expected-move > round-trip cost) | untried | policy-only change in the lab |
| H5 | Regime conditioning on the skilful activity targets; size by inverse predicted volatility | untried | H1 or H4 policy plumbing |
| H6 | Cross-sectional relative direction: long top / short bottom by predicted direction | untried | panel mode (H1) |
| H7 | Lower-frequency horizons (4h, daily) | blocked | non-overlapping evaluator + interval metadata in the archive |
| H8 | New data classes (funding, open interest, order-book imbalance) | blocked | upstream capability check + schema change + human approval |
| H9 | Stronger model classes (boosted trees, small ensembles) | blocked | H1 or H2 first, plus the trial counter |
| H10 | Improve the skilful targets directly (`fwd_volume_ratio`, `fwd_volatility`) | untried | feature sets / model class for those targets |

## Rules that apply to every row

- One hypothesis per run; at most 12 configs.
- A family is never retried without new data, a new feature class, or a fixed
  measurement bug.
- The final holdout is spent at most once per lineage, by `promote` only.
- A negative result is a valid outcome and is recorded as one.

## Measurement hole found by H1 (blocking H1 retries)

The pre-registered H1 verdict is `negative` (ledger lineage `h1-panel-v1`), but
the interesting result is the control: a 24-bar shifted label beats the constant
baseline in 14/15 tickers, and a post-hoc 7-bar shift wins on
`fwd_volume_ratio`, `fwd_direction` *and* `fwd_return`. The shifted-label null
is therefore not a valid no-signal baseline for this target family: re-pairing
labels against a train-fold constant can manufacture apparent skill. Fix the
control (e.g. pre-specify a non-overlapping shift, score against a
test-window/expanding baseline, or use a within-fold label-permutation null)
before spending another trial on H1.

## Foundation (precondition)

Measurement gaps were closed before the queue was opened; see the
`model-improve/2026-10-24-foundation` branch. Until the archive is fresh again
(production train runs resume), H1-H10 are blocked on data freshness, not on
modelling.
