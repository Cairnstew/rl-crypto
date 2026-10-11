# Trained model bundles

Tier-1 models trained by `rl-crypto train` are committed here as **JSON
bundles** — one file per ticker (`models/eth-usd.json` for `ETH/USD`). A bundle
is a single self-describing document (schema `rl-crypto/ml/1`):

```json
{
  "schema": "rl-crypto/ml/1",
  "trained_at": "…",
  "horizon": 1,
  "n_samples": 200,
  "feature_names": ["ret_1", "ret_3", "…"],
  "targets": {
    "fwd_return": {
      "kind": "regression",
      "model": { "kind": "ridge", "alpha": 1.0, "scaler": {}, "coef": [], "intercept": 0.0 },
      "metrics": { "rmse": 0.012, "mae": 0.009, "r2": 0.31 },
      "n_samples": 200
    },
    "fwd_direction": { "kind": "classification", "model": { "kind": "logistic", "…": "…" } }
  },
  "notes": []
}
```

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
```