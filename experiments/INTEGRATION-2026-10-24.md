# Integration 2026-10-24: foundation + harness hardening

Branch `integration/2026-10-24`, based on `harness-hardening/2026-10-24`
(`2fc570d`). Never merged to `main`, never pushed.

## Conflict table

| symbol | foundation version | hardening version | kept |
| --- | --- | --- | --- |
| `src/rl_crypto/experiments.py` (whole module) | `rl-crypto/experiment/1`; `preregister`, `record_result`, `record_holdout`, `check_promote_allowed`, `trial_index`, `lineage_trial_index`, `counts`, `verdict_for_lineage` | `rl-crypto/experiments/1`; `Hypothesis`, `build_preregistration`, `evaluate_criteria`, `append_entry`, `holdout_cutoffs`, `harness_void`, `criteria_version` | **hardening** (17 committed ledger entries are immutable) |
| `holdout_touches` | list of entries for one `lineage_id` | counts per `lineage` | **hardening** (stable lineage + frozen cutoff) |
| `render_scoreboard` | renders the foundation schema | renders the hardening schema | **hardening** |
| purge / label reach | `targets.label_purge`, `ml.tier1.chronological_split`, lab `walk_forward(targets=...)` | hardcoded `max(0, horizon, 2)` in `walk_forward` | **foundation `label_purge`**; hardening's duplicate deleted |
| lineage | `lineage_id` + `lineage_trial_index` | `Hypothesis.stable_lineage` + `resolve_holdout_start` + `trial_counts` | **hardening** |
| promote gating | `experiments.check_promote_allowed` (one holdout per lineage, dry run before apply) + `model_lab.promotion_gate` | `promote_action` requires `--holdout-start` for apply; champion-retrain comparison; driver holdout-touch gate | **hardening**; apply now also writes `config/training.json` |
| lab arguments | `--thresholds`, `--cost-levels`, `preregister`/`scoreboard` actions | `--alpha`, `--bootstrap-*`, `--holdout-start`, `--controls`, `--primary-config`, `--champion-retrain` | **hardening parser** + ported cost-grid flags; duplicate preregister/scoreboard actions deleted |
| `_pnl_metrics` | thin wrapper over `rl_crypto.backtest.pnl_metrics` | inline copy | **one definition**: `rl_crypto.backtest.pnl_metrics` |
| `evaluate_target` / `paired_target_rows` | raw-scale metrics | raw-scale metrics | **hardening** + bundle `metric_space` (log for `fwd_volatility`) in both |
| `walk_forward` | label-reach purge | horizon-only purge | **hardening signature + foundation reach** |

Files changed by both branches: 19 (11 `.opencode` docs/tools, `experiments/SCOREBOARD.md`,
`hypotheses.md`, `ledger.jsonl`, `src/rl_crypto/experiments.py`,
`tests/test_experiments.py`, `tests/test_model_lab.py`, `tools/model_lab.py`).
`.opencode` content and existing ledger entries were kept from hardening; the TS
lab wrapper was migrated to the reconciled Python CLI.

## Commits

| commit | task |
| --- | --- |
| `e2886f7` | 1 — reconcile purge, training config, cost grid, forward scorer |
| `f017b83` | 1c — durable promote + cutoff purge |
| `da25177` | 2 — alpha spending + bootstrap resolution + family cap |
| `676404f` | 3 — canaries + pooled economics + 168-bar no-new-information |
| `5dfd993` | 4 — auditor fixtures + `auditor-check` |
| `0c879f8` | 5 — protected-list pin |
| `8b2dad6` | docs / TS caller migration |
| this commit | 1d — pair `fwd_volatility` improvements in log space |

Every commit was checked with `ruff check` (clean) and the full offline suite:
273, 302, 303, 310, 317, 319, 321, 321, 322 passed at the successive tips.

## Canary outcomes (mechanical, real archive)

- `H0-canary` (shuffled labels): **negative / passed=false**, reason
  `canary negative as required`. Pooled mean −0.00832, CI [−0.01387, −0.00304],
  n=2740, blocks=115, resolution_ok=true. No `harness_void` entry.
- `H0-canary-leak` (positive): **invalid / passed=false**, reason
  `a feature is a deterministic function of a target (leak)` with
  `leak_next_bar_return~fwd_return r=1.000`. No `harness_void` entry.
- Artifacts: `/tmp/opencode/integration-runs/<run-id>/` (not committed; the
  ledger was restored so harness checks are not experiment trials).

## Auditor red-team run

`python tools/model_improve.py auditor-check` ran all 10 fixtures with the
configured auditor model `opencode/claude-haiku-4-5`. Every call returned
`402 Insufficient account funds`, so the harness recorded 10 `error` fixtures and
**no** measured recall or false-block rate. The model was not changed. Artifact:
`/tmp/opencode/auditor-check2/auditor-check.json`.

## Residual risks / human approval before merge

1. The auditor model could not be exercised; recall/false-block are unmeasured.
2. `PROTECTED_FILES` now includes `src/rl_crypto/backtest.py` and
   `src/rl_crypto/ml/training_config.py`; the pin is a new contract change and
   needs the explicit human review documented in `experiments/README.md`.
3. `--alpha` spending changes the meaning of a v2 trial. Old entries keep
   `criteria_version`-1 rules; no historical verdict was recomputed.
4. The family cap (10) and 10,000 resamples make a full family expensive; a lab
   run over three tickers takes longer than the old 1,000-draw default.
5. `.github/`, `data/` and `models/` were not changed by integration commits; the
   branch inherits the hardening workflow file from its base.
