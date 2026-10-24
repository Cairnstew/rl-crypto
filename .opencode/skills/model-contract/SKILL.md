---
name: model-contract
description: >-
  Use when adding or changing a feature, target, model, feature version, or
  training protocol in rl-crypto. Triggers: "add a feature", "change a target",
  "feature_version", "retrain", "retrain-only", "holdout", "beats baseline",
  "model not trusted", "signal withheld", "bundle schema", "train/serve
  parity", "why is the prediction missing". Encodes the causal T0→T2 contract,
  the out-of-sample trust gate, and the bundle migration path, and appends
  durable findings to this file so each change builds on the last.
---

# Model contract — features, targets, holdout, bundles

This skill governs the **learning contract**, not the data and not the runner.
Use it for any change that alters what a model sees, what it predicts, how it is
scored, or how a bundle is served.

Companions:

- **`data-quality-audit`** — is the input data sound?
- **`pipeline-hardening`** — is the writer/runner robust?
- **this skill** — is the model/feature contract honest and reproducible?

**Tooling:** use the `rl-model-lab` opencode tool for the measurement loop
(`status`, `evaluate`, `sweep`, `promote`). It is offline, reserves a final
holdout, charges fees/slippage, and writes its ledger outside the repo. It
measures; it does not guarantee profit.

## 0. Contract

- **Causality is non-negotiable.** A feature may read only rows at or before its
  own timestamp. A target may read only future rows, exactly `horizon`
  intervals ahead.
- **`None` is an honest gap, never a fabricated zero.** The only encoded
  absence is the documented news pair (`sentiment=0.0`, `has_news=0`).
- **In-sample is not a result.** The headline is the chronological holdout
  versus a constant baseline. `beats_baseline` is the gate; `trusted` is the
  published verdict.
- **A feature change bumps `FEATURE_VERSION`.** A bundle trained against an
  older contract must be rebuilt, not silently served.
- **Train and serve use one feature definition.** No second implementation for
  the live path.
- **No labels are cached.** Forward targets are recomputed from the archive
  every run; a bar's label exists only once its future is archived.
- **Do not change the protocol and the model in one step.** Isolate the
  contract change, retrain, verify, then tune.

## 1. Architecture map

| Tier | Module | Owns | Contract |
| --- | --- | --- | --- |
| **T0** | `src/rl_crypto/features.py` | Causal features, `FEATURE_GROUPS`, `FEATURE_NAMES`, `FEATURE_VERSION` | Time-based windows; `None` on gaps; train/serve parity |
| **T1 labels** | `src/rl_crypto/targets.py` | `fwd_return`, `fwd_direction`, `fwd_volatility`, `fwd_volume_ratio` | Forward-only, exact-horizon spacing, `None` on a gap |
| **T1 samples** | `src/rl_crypto/samples.py` | Align features + targets; drop incomplete rows | No lookahead; horizon-aware |
| **T1 model** | `src/rl_crypto/ml/tier1.py` | Estimators, chronological holdout, baseline comparison, `Tier1Bundle.trusted` | Fit on train split only; holdout describes shipped weights |
| **T2** | `src/rl_crypto/ml/tier2.py` | Fuse predictions into `Outlook`; withhold unearned signal | `signal`/`confidence` only for trusted targets |
| **Serve** | `src/rl_crypto/ml/serve.py` | Build the live feature row; attach the outlook | Same feature definition as training; cadence-aware |
| **Store** | `src/rl_crypto/ml/store.py` | JSON bundle load/save; feature-version check | `rl-crypto/ml/1`; mismatch flags, does not silently predict |
| **Conflict** | `src/rl_crypto/pipeline/conflicts.py` | Prefer a servable bundle during a rebase | `_servable` compares `feature_version` |

**The Tier-3 seam:** `Outlook.as_features(prefix="pred_")`
(`src/rl_crypto/ml/types.py:73-89`) flattens predictions, probabilities,
`signal`, and `confidence` into scalar columns. Changing that shape changes the
future RL observation vector; treat it as a schema change.

## 2. Ground truth (as of 2026-10-23)

| Item | Value | Source |
| --- | --- | --- |
| Feature contract | `rl-crypto/features/2` | `features.py:53` |
| Bundle schema | `rl-crypto/ml/1` | `ml/types.py` / `store.py` |
| Summary schema | `rl-crypto/pipeline/3` | `pipeline/run.py:187-251` |
| Targets | `fwd_return` regression, `fwd_direction` classification, `fwd_volatility` regression, `fwd_volume_ratio` regression | `targets.py` `TARGET_KINDS` |
| Holdout | newest ~20%, chronological, never fitted | `ml/tier1.py` |
| Baseline | constant predictor; regression = training mean, classification = better of up/down | `ml/tier1.py` |
| Baseline margin | 1% relative, not merely "better" | `ml/tier1.py` |
| Trusted today | `fwd_volume_ratio` 15/15; `fwd_volatility` 5/15 (log-space); `fwd_return` 0/15; `fwd_direction` 0/15 | re-derive from `data/*/summary.json` |
| Signal | withheld while no directional target is trusted | `ml/tier2.py:45-71` |
| Dropped features today | `sentiment`, `sentiment_delta`, `sentiment_ewma_6`, `articles_log`, `novelty`, `has_news` | archive is market-only; trainer drops constants |

**Do not copy the trusted counts.** They move as the archive grows. The
procedure below re-derives them.

## 3. Procedure

### 3.1 Pin the run

```bash
date -u +%FT%TZ
git rev-parse --short HEAD
git log -5 --oneline -- src/rl_crypto/features.py src/rl_crypto/targets.py src/rl_crypto/ml models/
```

Read the `Memory` sections. Record the baseline even if the change is small.

### 3.2 Baseline

```bash
nix develop --command bash -c 'uv run python -m pytest -m "not live" -q'
nix develop --command bash -c 'uv run ruff check src/ tests/ examples/ tools/'
nix develop --command bash -c 'uv run python tools/audit_data_quality.py'
```

Then re-derive the model state:

```bash
python3 - <<'PY'
import glob, json, collections
beats = collections.Counter()
versions = collections.Counter()
for path in sorted(glob.glob("data/*/summary.json")):
    d = json.load(open(path))
    versions[d.get("feature_version")] += 1
    for name, block in (d.get("targets") or {}).items():
        if (block.get("holdout") or {}).get("beats_baseline"):
            beats[name] += 1
print("feature_versions:", dict(versions))
print("beats_baseline:", dict(beats))
PY
```

Also check one bundle's `feature_version` against `FEATURE_VERSION` and confirm
the summary's `archive.rows` matches the candle line count. If either drifts,
stop and fix the pipeline before changing the model.

### 3.3 Feature change

For every proposed feature, answer these in order:

1. **Causality:** is it a pure function of rows `<= i`? If it needs a future row,
   it is a target, not a feature.
2. **Time, not rows:** do the windows measure real intervals? A gap must yield
   `None`, not a retimed window. Use `interval_seconds` and `_span_ok`.
3. **Missingness:** is `None` propagated? The only encoded absence is the news
   pair; do not impute anything else to `0`.
4. **Constant columns:** will it vary in the current archive? Constant features
   are dropped by `usable_features`; a feature that is constant *only*
   in-sample is a different (interesting) case — document it.
5. **Order:** `FEATURE_NAMES` is the model's input order. Adding in the middle
   or reordering without a version bump is silent corruption.
6. **Version:** bump `FEATURE_VERSION` for any change to names, order, meaning,
   or window semantics. Additive constants still need a version decision.
7. **Parity:** the live builder in `ml/serve.py` must produce the same row from
   the archive + tail. Add a train/serve parity test.
8. **Docs/tests:** update `FEATURE_DESCRIPTIONS`, `models/README.md` if the
   contract changes, and the feature tests.

Minimum test set: `tests/test_features.py` (values, gaps, `None`, window
spacing), `tests/test_ml_measurement.py` (holdout/baseline), `tests/test_ml_serve.py`
(parity), plus `test_samples.py` for alignment.

### 3.4 Target change

1. **Forward-only:** the label reads `i + horizon`; features never do.
2. **Exact horizon:** `_ahead_by` must confirm the future row is exactly
   `horizon` intervals away. A gap yields `None`; never relabel a longer move.
3. **Kind:** regression vs classification must match the estimator and the
   metric/baseline (`TARGET_KINDS`).
4. **Minimum history:** volatility needs `MIN_VOL_SPAN`; a shorter span is
   `None`.
5. **Baseline:** every new target needs a defensible constant baseline and a
   `beats_baseline` verdict before it can be trusted.
6. **Docs/tests:** `TARGET_DESCRIPTIONS`, `tests/test_targets.py`,
   `test_ml_measurement.py`.

### 3.5 Model / protocol change

- **Split:** chronological, newest holdout, fitted on the training split only.
  Never refit on the holdout after scoring.
- **Baseline:** keep the constant predictor and the 1% relative margin. A
  "better by 0.1%" result on a ~300-row holdout is noise.
- **Trust:** `Tier1Bundle.trusted(target)` is the only publication gate. A
  missing holdout is not trusted.
- **Bundle:** bump the schema for a shape change; keep `feature_version` in the
  bundle; `load_bundle` flags a mismatch and suppresses predictions.
- **Serve:** `outlook_for_rows` must pass the real cadence
  (`interval_seconds`); the wrong cadence withholds every windowed feature.
- **Conflict:** `_servable` must continue to prefer the bundle the running code
  can serve, even if the other is newer.
- **Tests:** `test_ml_linear`, `test_ml_tier1`, `test_ml_measurement`,
  `test_ml_serve`, `test_ml_store`.

### 3.6 Retrain and verify

After a feature/target contract change, retrain from the committed archive
without spending quota:

```bash
nix develop --command bash -c 'uv run rl-pipeline run --retrain-only'
```

Then verify:

- every summary's `feature_version` equals the code's `FEATURE_VERSION`;
- every bundle's `feature_version` matches and `feature_version_mismatch` is
  false;
- `archive.rows` equals the candle line count;
- the holdout block exists and the trusted counts moved for a defensible
  reason;
- the dashboard still renders (`uv run python tools/render_dashboard.py`);
- the offline suite and ruff are clean.

If a feature change needs new data (a new column that does not exist in the
archive), it is not a retrain-only change. Backfill/seed first, then retrain.

### 3.7 Classify

| Level | Meaning | Example |
| --- | --- | --- |
| **P0** | Leakage or silent corruption: future data in a feature, retimed window, mismatched bundle served | feature reads `i+1`; reordered `FEATURE_NAMES` without a version bump |
| **P1** | Contract broken: holdout refit, missing baseline, feature version not bumped | new feature shipped under `features/2` |
| **P2** | Reproducibility/quality: holdout too small, missing parity test, stale docs | live path builds a different row |
| **P3** | Maintainability: description drift, confusing name, missing metric | a target with no baseline documentation |

Origin: feature code / target code / trainer / bundle store / serve path /
conflict resolver / docs. A deliberate design (`signal` withheld, `has_news=0`)
is `EXPECTED`, not a bug.

### 3.8 Improvement / mapping step

Every P0-P2 gets an `IM-N` row with a concrete `path:line`, a test, and a
status. Action ladder:

1. **Prove** with a failing value/leakage test.
2. **Fix the contract** (causality, horizon, version) before tuning the model.
3. **Add the parity/regression test.**
4. **Retrain** with `--retrain-only` (or seed data first if the archive lacks
   the input).
5. **Document** the contract in `models/README.md` / `data/README.md`.
6. **Re-derive** the trusted counts and update the dashboard.

Rules:

- Never edit `notebooks/unified.ipynb` or `README.md`'s generated dashboard to
  "show" a result; regenerate them.
- Never treat an in-sample metric as evidence.
- Never serve a bundle whose `feature_version` differs.
- A model change that only improves in-sample is not an improvement.

### 3.9 Update this file

Before the final summary:

1. Update the baseline numbers/version in place.
2. Add a new finding under `Memory — active findings` (one per root cause).
3. Update existing evidence in place; move fixed/expected entries to
   `Memory — resolved & expected`.
4. Update the improvement map and the audit log (one row).
5. Add/answer open questions.

Bounds: ≤12 active findings, ≤20 log rows, ≤5 lines per finding.

Template:

```markdown
### MC-0NN · <short claim> · <OPEN|MONITORING|FIXED|EXPECTED> (since YYYY-MM-DD)
- **Layer:** feature | target | trainer | bundle | serve | conflict | docs
- **Evidence (YYYY-MM-DD):** <path:line + measured result>
- **Read:** <why it matters; the condition that makes it a real fault>
- **Map:** IM-N
- **Status:** <last change + date>
```

### 3.10 Verify and commit

- Offline tests + ruff; `basedpyright` if you touched src.
- `git diff --stat`; separate the contract change, the retrain artifacts, and
  the docs.
- Update the data-quality audit baseline if the cache schema changed.
- Report: the verdict, the trusted deltas, and what changed in the ledger.

## Memory — baseline

**As of 2026-10-24T02:36Z** (`HEAD` `213d17b`):

| Metric | Value |
| --- | --- |
| Feature contract | `rl-crypto/features/2` |
| Bundle schema | `rl-crypto/ml/1` |
| Summary schema | `rl-crypto/pipeline/3` |
| Feature names | 21 (return 3, volatility 2, bar 2, trend 2, volume 2, news 6, calendar 4) |
| Targets | 4 (`fwd_return`, `fwd_direction`, `fwd_volatility`, `fwd_volume_ratio`) |
| Trusted | `fwd_volume_ratio` 15/15, `fwd_volatility` 5/15 (log-space), directional 0/15 (re-derived 2026-10-24) |
| Dropped in training | 6 news features (constant in the market-only archive) |
| Tests | 203 offline pass; ruff clean |
| Types | `basedpyright` 8 upstream-import errors (not in CI) |

Verdict: **the contract is coherent and well tested; the main risk is a silent
feature/target change without a version bump, and stale trusted counts.** The
directional models are honestly untrusted — do not "fix" that by publishing a
signal.

## Memory — active findings

### MC-001 · Directional targets have no out-of-sample skill · EXPECTED (since 2026-10-23)
- **Layer:** trainer/T2 (`src/rl_crypto/ml/tier1.py:301-312`; `ml/tier2.py:45-71`)
- **Evidence (2026-10-23):** `fwd_return` 0/15 and `fwd_direction` 0/15 beat baseline; `Tier1Bundle.trusted` returns false; `compose_outlook` emits `signal=None` with a note.
- **Read:** withholding is the correct behaviour, not missing data. A future change that starts publishing a signal must show a holdout `beats_baseline=True` on the directional target first.
- **Map:** none
- **Status:** expected; re-derive the counts, never copy them.

### MC-002 · In-sample metrics are not a result · EXPECTED (since 2026-10-23)
- **Layer:** trainer (`src/rl_crypto/ml/tier1.py:209-240`)
- **Evidence (2026-10-23):** every summary carries `in_sample` and `holdout`; the README and `models/README.md` make `holdout` the headline; directional in-sample accuracy (~0.54) exceeds its holdout (~0.49).
- **Read:** a change that improves only `in_sample` has not improved the model. Gate on `holdout.beats_baseline` and the 1% margin.
- **Map:** IM-2
- **Status:** expected.

### MC-003 · A feature-version mismatch suppresses predictions · EXPECTED (since 2026-10-13)
- **Layer:** bundle/serve (`src/rl_crypto/ml/store.py:43-62`; `ml/tier1.py:301-312`)
- **Evidence (2026-10-23):** `load_bundle` sets `feature_version_mismatch` and appends a note; `trusted()` returns false; `_composite` withholds the signal. `conflicts._servable` prefers the servable bundle even if the other is newer.
- **Read:** a stale bundle is flagged, not silently served. After a feature change, rebuild with `--retrain-only`; do not hand-edit `feature_version`.
- **Map:** IM-4
- **Status:** expected.

### MC-004 · News features are dropped as constant in the current archive · EXPECTED (since 2026-10-23)
- **Layer:** samples/trainer (`src/rl_crypto/features.py:70-77`; `ml/tier1.py` `usable_features`)
- **Evidence (2026-10-23):** summaries list `dropped_constant: sentiment, sentiment_delta, sentiment_ewma_6, articles_log, novelty, has_news`; the archive is market-only, live news is still recorded.
- **Read:** `has_news=0` is an explicit encoding, not a missing measurement. The six features light up once news history is ingested; keep the encoding until then.
- **Map:** none
- **Status:** expected.

### MC-005 · Windows are measured in time; a gap yields `None` · EXPECTED (since 2026-10-23)
- **Layer:** features/targets (`src/rl_crypto/features.py` `_span_ok`; `targets.py` `_ahead_by`)
- **Evidence (2026-10-23):** `ret_1`/`ma_gap_24` return `None` when the span is not exactly the expected interval; `fwd_return`/volatility do the same across a gap.
- **Read:** this is the leakage/horizon guard. Do not "fix" a gap by retiming the window or by filling the bar; that is a different feature/target wearing the same name.
- **Map:** IM-3
- **Status:** expected.

### MC-006 · Trusted counts drift with the archive · MONITORING (since 2026-10-23)
- **Layer:** docs/summary (`data/*/summary.json`; `models/README.md`)
- **Evidence (2026-10-23):** `fwd_volatility` is 10/15 in the summaries while `models/README.md` records 9/15 from an earlier snapshot; `fwd_volume_ratio` is 15/15.
- **Read:** any table copied from the README goes stale. Re-derive before quoting; update the docs only when the change is durable.
- **Map:** IM-2
- **Status:** monitoring.

### MC-007 · `Outlook.as_features` is the future RL observation contract · EXPECTED (since 2026-10-23)
- **Layer:** T2/serve (`src/rl_crypto/ml/types.py:73-89`)
- **Evidence (2026-10-23):** the method emits `pred_<target>`, `pred_prob_<target>`, `pred_signal`, `pred_confidence`, preserving `None`.
- **Read:** it is the explicit Tier-3 seam. Adding/removing/renaming a key changes the RL observation width; treat it as a schema change and version it when Tier 3 exists.
- **Map:** IM-6
- **Status:** expected until Tier 3.

### MC-008 · `FEATURE_NAMES` order is the model input contract · EXPECTED (since 2026-10-23)
- **Layer:** features/bundle (`features.py:79-83`; `ml/tier1.py` `feature_names`)
- **Evidence (2026-10-23):** the bundle stores `feature_names`; `predict_row` maps by name, but a reorder without a version bump changes the contract the bundle was trained under and is not caught by names alone.
- **Read:** reordering is a **P1** change unless `FEATURE_VERSION` is bumped and the bundle is retrained. Prefer append-only additions.
- **Map:** IM-1
- **Status:** expected.

### MC-009 · `fwd_volatility` was served outside its domain · FIXED (since 2026-10-24)
- **Layer:** trainer/serve (`ml/linear.py` `LogRidge`; `ml/tier1.py` `DEFAULT_MODEL_BY_TARGET`, `_regression_metrics_in_space`; `ml/tier2.py` `_clip_domain`)
- **Evidence (2026-10-24):** 11 of 15 bundles served a negative `fwd_volatility` (e.g. XRP `-0.006564`); a stdev cannot be negative. After switching the target to `log_ridge` and retraining, 0/15 are negative and 15/15 serve a positive spread; 5/15 beat the log-space constant baseline.
- **Read:** two layers. `log_ridge` fits `log(y)` and exponentiates back (positive by construction, clipped against underflow); `_clip_domain` withholds any future out-of-domain value as `None`, never a fabricated `0.0`. A log-space estimator must be **scored in log space**: raw RMSE against a raw-mean baseline reports a loss it never optimised (it scored 0/15 that way vs 5/15 on log loss). Zero-spread rows are excluded from log-space scoring.
- **Map:** IM-7
- **Status:** fixed 2026-10-24 (log_ridge + guard + `--retrain-only`); 203 tests and ruff clean.

### MC-010 · `metric_space` travels with the model · EXPECTED (since 2026-10-24)
- **Layer:** trainer/bundle/summary (`ml/tier1.py` `Tier1Model.metric_space`; `pipeline/run.py`)
- **Evidence (2026-10-24):** every bundle target and summary target block now records `metric_space` (`"log"` for `fwd_volatility`, `"raw"` otherwise), so a reader can tell which scale `rmse` is on.
- **Read:** a metric whose space is unlabelled is ambiguous — the same number means different things for a log-space estimator. Do not compare log-space and raw-scale RMSE across targets.
- **Map:** IM-8
- **Status:** expected; schema is additive under `ml/1`.

## Memory — resolved & expected

- **R-001 · Holdout is chronological and covers the shipped weights** — the newest ~20% is withheld and never refit; `models/README.md` documents the protocol. Tested by `test_ml_measurement.py`.
- **R-002 · Baseline is a constant predictor with a 1% relative margin** — regression vs training mean; classification vs the better of always-up/always-down. Do not weaken the margin to make a model "pass".
- **R-003 · `signal` is withheld, not faked** — `Tier1Bundle.trusted` gates `compose_outlook`; a `None` signal is the correct output while no directional target beats baseline.
- **R-004 · Bundle load checks the feature contract and schema** — `load_bundle` rejects an unknown schema and flags a feature-version mismatch.
- **R-005 · Conflict resolution prefers a servable bundle** — `conflicts._servable` compares `feature_version`; a newer but unservable bundle loses. Tested.
- **R-006 · Labels are recomputed, never cached** — training reads the whole archive and recomputes forward targets; a bar's label is undefined until its future exists.

## Memory — improvement map

| ID | Finding | Layer | Action | Target | Risk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| IM-1 | MC-008 | feature | Add a test that pins `FEATURE_NAMES` order and fails on a reorder without a `FEATURE_VERSION` bump | `tests/test_features.py` | low | proposed |
| IM-2 | MC-002, MC-006 | docs/tests | Add a small trusted-count probe (reuse §3.2) and use it in reviews; stop copying the table | `tools/` or the skill probe | none | proposed |
| IM-3 | MC-005 | tests | Add parity/leakage tests for every feature group and every target horizon, including a gap | `tests/test_features.py`, `tests/test_targets.py`, `tests/test_ml_serve.py` | low | proposed |
| IM-4 | MC-003 | docs | Document the retrain-only migration path and the "no hand-edited feature_version" rule in `models/README.md` | `models/README.md` | none | proposed |
| IM-5 | MC-001, MC-002 | docs | Keep the "what is trusted today" table dated and auto-derivable; do not present in-sample as a result | `models/README.md` | none | proposed |
| IM-7 | MC-009 | model | Fit `fwd_volatility` with `log_ridge` (log-space fit, exponentiated back) and score it in log space; keep `_clip_domain` as the safety net | `ml/linear.py`, `ml/tier1.py` | target unchanged; retrain done | **done 2026-10-24** |
| IM-6 | MC-007 | T2/T3 | When Tier 3 lands, version the `as_features` key set and add a width/schema test | `ml/types.py`, Tier-3 tests | future schema change | proposed |
| IM-8 | MC-010 | trainer/docs | Record `metric_space` on every model and summary target so RMSE cannot be read on the wrong scale | `ml/tier1.py`, `pipeline/run.py` | none | **done 2026-10-24** |

## Memory — audit log

| Run (UTC) | HEAD | Contract | Trusted | Tests | Verdict |
| --- | --- | --- | --- | --- | --- |
| 2026-10-23T23:18Z | `213d17b` | features/2, ml/1, pipeline/3 | vol 10/15, volume 15/15, directional 0/15 | 181 pass | coherent; no silent-skill claim; main risk is an unversioned feature change |
| 2026-10-24T02:36Z | `213d17b` | features/2, ml/1, pipeline/3 | vol 10/15, volume 15/15, directional 0/15 | 196 pass | dashboard now surfaces the earned activity targets; vol domain guard added; directional work closed |
| 2026-10-24T12:05Z | `81c0468` | features/2, ml/1, pipeline/3 | vol 5/15 (log), volume 15/15, directional 0/15 | 203 pass | vol root cause fixed via `log_ridge` + log-space scoring; 15/15 spreads positive (11/15 were negative); narrower but honest verdict |

## Memory — open questions

- **Q-1** Should a feature addition always bump `FEATURE_VERSION`, even when all existing names/order/meaning are unchanged? Current rule: yes for names/order/meaning; document the additive case.
- **Q-2** Is `holdout_n_samples` the max across targets or per-target? Per-target `holdout.n_samples` is authoritative; the bundle field is a summary.
- **Q-3** What is the minimum archive size for a meaningful volatility holdout? `MIN_HOLDOUT_SAMPLES` sets the floor; record when a ticker crosses it.
- **Q-4** Should `models/README.md`'s trusted table be generated from the summaries instead of hand-maintained? That would remove MC-006 permanently.
- **Q-5** When Tier 3 lands, is `Outlook.as_features` the only observation seam, or will RL also consume raw T0 features? Define the width contract before training.

