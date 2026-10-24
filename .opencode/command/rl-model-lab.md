---
description: "Model lab: bare = agent-driven improvement loop; direct = status | evaluate | sweep | preregister | promote | report | scoreboard | help. Flags: tickers, targets, folds, threshold(s), position, feeBps, slippageBps, costLevels, featureSet(s), modelSet(s), horizon(s), maxConfigs, candidate, hypothesis, lineage, apply, limit."
agent: build
---

Decide the mode from `$ARGUMENTS`:

- **Empty, or a goal/ticker list** (e.g. `BTC/USD`, `improve BTC/USD`,
  `all tickers`) → load the `model-improvement` skill and follow its loop.
  Start read-only with `status` then `evaluate`; do not sweep, apply, or edit
  files before diagnosing. Ask before spending more than one sweep or 8 configs.
- **Starts with a known action** (`status`, `evaluate`, `sweep`, `preregister`,
  `promote`, `report`, `scoreboard`, `help`) → call the `rl-model-lab` tool
  directly with those arguments and report the result.

Direct-mode reference:

- `status` — deployed bundles/trust/freshness. Flags: `tickers`, `json`.
- `evaluate` — walk-forward metrics + after-cost policy. Flags: `tickers`,
  `targets`, `folds`, `threshold`, `position`, `feeBps`, `slippageBps`,
  `thresholds`, `costLevels`, `featureSet`, `modelSet`, `horizon`, `limit`,
  `json`. A thresholds/cost grid fits each fold once.
- `sweep` — controlled feature/model/horizon grid. Flags: `tickers`, `targets`,
  `folds`, `threshold`, `position`, `feeBps`, `slippageBps`, `featureSets`,
  `modelSets`, `horizons`, `maxConfigs`, `limit`, `json`.
- `preregister` — record a hypothesis and numeric criteria before it runs.
  Flags: `hypothesis`, `family`, `lineage`, `mechanism`, `configs`,
  `tickers`, `targets`, `metric`, `criteria`, `negative`, `seed`, `folds`, `json`.
- `promote` — final-holdout gate. Flags: `candidate` (required), `ticker`,
  `hypothesis`, `lineage`, `targets`, `threshold`, `position`, `feeBps`,
  `slippageBps`, `apply`, `json`. The holdout is spent once per lineage.
- `report` — experiment ledger with trial/holdout counts. Flags: `limit`, `json`.
- `scoreboard` — regenerate `experiments/SCOREBOARD.md` from the ledger. Flags: `json`.
- `help` — full reference.

Examples: `status`; `evaluate tickers=BTC/USD folds=3`;
`sweep tickers=BTC/USD maxConfigs=4`; `promote candidate=/path/to.json ticker=BTC/USD`;
`report`; bare `/rl-model-lab BTC/USD`.

Rules:

- Report out-of-sample numbers and after-cost results; never claim the model is profitable.
- For `promote`, dry-run unless the user explicitly says to apply.
- Include the gate reasons and the artifact path in the summary.
