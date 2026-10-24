# Trained model bundles

Tier-1 models trained by `rl-crypto train` are committed here as **JSON
bundles** — one file per ticker (`models/eth-usd.json` for `ETH/USD`). A bundle
is a single self-describing document (schema `rl-crypto/ml/1`):

```json
{
  "schema": "rl-crypto/ml/1",
  "feature_version": "rl-crypto/features/2",
  "trained_at": "…",
  "horizon": 1,
  "n_samples": 1201,
  "holdout_n_samples": 299,
  "feature_names": ["ret_1", "ret_3", "…"],
  "dropped_features": ["sentiment", "has_news", "…"],
  "targets": {
    "fwd_return": {
      "kind": "regression",
      "model": { "kind": "ridge", "alpha": 1.0, "scaler": {}, "coef": [], "intercept": 0.0 },
      "metrics": { "rmse": 0.012, "mae": 0.009, "r2": 0.31 },
      "holdout": {
        "n_samples": 299,
        "metrics":  { "rmse": 0.0046, "mae": 0.0032, "r2": -0.05 },
        "baseline": { "rmse": 0.0046, "mae": 0.0033, "r2": -0.02 },
        "beats_baseline": false
      },
      "n_samples": 1201
    },
    "fwd_direction": { "kind": "classification", "model": { "kind": "logistic", "…": "…" } }
  },
  "notes": ["…"]
}
```

## Read `holdout`, not `metrics`

`metrics` is computed on the rows the model was **fitted** on. It measures how
much of the training noise the model memorised, which is why it is nearly always
flattering: on this data `fwd_direction` scores ~0.55 in-sample and ~0.49 on
unseen rows.

`holdout` is the out-of-sample verdict, and it is the number that matters. Three
things make it trustworthy:

- **The shipped weights are the ones evaluated.** Models are fitted on the
  training split only, so `holdout` describes exactly the weights in this file —
  there is no "we refit on everything afterwards" gap between the score and the
  artifact.
- **The split is chronological, newest rows reserved.** The withheld tail is the
  most recent data, which is also what the live path predicts forward *into*, so
  the evaluation asks the same question deployment does. A random split would
  let a model learn the future's mean from the past.
- **`baseline` is a constant predictor.** Regression compares against always
  predicting the training mean; classification against the better of
  always-up/always-down. `beats_baseline` requires a **1% relative** margin, not
  merely "better" — on a ~300-row holdout a 0.1% log-loss difference is noise,
  and treating it as skill is how a model earns the right to emit a signal it
  has not earned.

The withheld rows are not discarded: the pipeline retrains over the whole growing
archive every two hours, so each run folds them back in.

## What is trusted today

Across all 15 tracked tickers, on the current archive:

| target | beats baseline | verdict |
| ------ | -------------- | ------- |
| `fwd_return` | 0/15 | **no directional skill** — worse than a constant on every ticker |
| `fwd_direction` | 0/15 | **no directional skill** — at or below always-up |
| `fwd_volatility` | 5/15 | marginal — log-space verdict, inconsistent across tickers |
| `fwd_volume_ratio` | 15/15 | solidly real (14–24% better than baseline) |

*Re-derived from `data/*/summary.json` on 2026-10-24; re-derive rather than copy.*
The volatility count is a **log-space** verdict (see below); the earlier 10/15
was raw-scale RMSE, which is not the loss the log-space estimator minimises.

`Tier1Bundle.trusted(target)` returns this verdict, and Tier 2 withholds
`signal`/`confidence` for any target that is not trusted. Point predictions are
still emitted either way — they are inputs for a future RL observation vector,
not claims about the market.

A point prediction is only emitted when it is inside the target's domain.
`fwd_volatility` is a standard deviation, so a negative fit is impossible
physically. Two layers enforce that now:

- **The estimator (`log_ridge`).** `fwd_volatility` is fitted in log space and
  exponentiated back, so the served value is positive by construction. A plain
  `ridge` is unbounded, and on 2026-10-24 11 of 15 tickers served a negative
  spread — a number no market can produce. The log predictor is clipped to a
  safe range so an extreme out-of-range row cannot underflow to `0.0` either.
- **The serve guard (`tier2._clip_domain`).** A negative value from any future
  estimator is withheld as `None` with a note, rather than clipped to `0.0`,
  which would fabricate a spread no asset has.

Because the model now minimises squared error in log space, its metrics and the
constant baseline are computed there too (`metric_space: "log"` in the bundle
and summary, versus `"raw"` for the other targets). Scoring a log-space model by
raw RMSE against a raw-mean baseline would report a loss it never optimised and
wrongly fail it. Rows with a zero realised spread have no log and are excluded
from log-space scoring. The target definition itself is unchanged: it is still a
realised standard deviation.

`fwd_return` scoring *below* the baseline out of sample is the expected result
for 1-hour-ahead returns on price-only features: the model is fitting noise, and
the out-of-sample numbers say so where the in-sample numbers did not.

## `dropped_features`

Features that never vary across the training rows are dropped at training time
and listed here. Today that is the whole news group: the archive is market-only,
so `sentiment`, `has_news` and friends are constant — yet the live path *does*
have news. Keeping them would hand the model six columns it learned nothing
from and then feed it out-of-distribution values at serve time. They light up on
their own once news history is ingested; nothing needs to change here when it is.

## `feature_version`

The feature contract the bundle was trained against, re-checked on load. A
bundle whose `feature_version` no longer matches the running code is **flagged,
not loaded as trusted**: it predicts nothing and says why in `notes`, rather
than silently serving numbers computed against a different feature definition.
This is why a bundle is rebuilt with `--retrain-only` after any feature change.

## Why JSON, and why committed

- **Reviewable.** Weights, feature lists and metrics diff like any source change;
  there is no opaque binary blob.
- **Hermetic.** Loading requires only the python+numpy this repo already pins —
  no pickle, no serialised lambda, no version skew surprise.
- **Auditable.** `git log models/` is the full training history. Retraining,
  changing a horizon, or swapping a model shows up as a clean diff.

Live prediction with a bundle:

```bash
rl-crypto predict ETH/USD --model models/eth-usd.json --json
```

## Regenerating

```bash
# inside `nix develop`:
rl-crypto train ETH/USD --from 2026-01-01 --to 2026-07-01 --timeframe 1h

# rebuild every bundle from the committed archive, spending no upstream quota:
uv run rl-pipeline run --retrain-only
```

`--retrain-only` is the right tool after a feature-contract change: the bars have
not moved, so re-fetching them would buy nothing, but features and labels must be
recomputed against the new contract.