---
name: ticker-onboarding
description: >-
  Use when adding, screening, or retiring a ticker in the rl-crypto watchlist.
  Triggers: "add a ticker", "retire a ticker", "new asset", "watchlist",
  "probe_tickers", "RL_CRYPTO_TICKERS", "why is this ticker not recording",
  "backfill a ticker". Covers evidence-based screening, the stablecoin and
  correlation policy, the mandatory first train, the archive/model artifacts,
  and the fork override, with a screening ledger as memory.
---

# Ticker onboarding and retirement

The watchlist is one tuple in `src/rl_crypto/tickers.py`, but changing it fans
out to the archive, models, workflow matrix, dashboard, docs, and the first-train
requirement. This skill makes that change safe and records why each candidate
was accepted or rejected.

Companions: **`data-quality-audit`** (verify the new archive after seeding),
**`model-contract`** (bundle/feature-version checks), **`pipeline-hardening`**
(the matrix/runners if onboarding exposes a code bug).

## 0. Contract

- **Screen before adding.** A ticker is evidence, not a hunch: a full hourly
  history window and a populated news window.
- **No stablecoins.** Near-zero variance makes them poor training targets
  (`tickers.py` explains this).
- **No correlation duplicates.** A series that tracks an already-tracked asset
  buys little signal at the same cost.
- **First train before live.** `live` needs `models/<slug>.json`; a new ticker
  has none until a `train` run.
- **The archive is append-only.** Retiring a ticker does not delete its history
  by default; deletion is a separate, explicit decision.
- **`tickers.py` is the single source of truth.** The workflow matrix reads it;
  do not add a second list in YAML.
- **Forks use the variable, not code.** `RL_CRYPTO_TICKERS` narrows the
  watchlist without editing `tickers.py`.

## 1. Ground truth

| Item | Value | Source |
| --- | --- | --- |
| Watchlist | 15 tickers | `src/rl_crypto/tickers.py:33-50` |
| Screening tool | `tools/probe_tickers.py` | read-only; no `data/` writes |
| Workflow matrix | built from `tickers.py` in `prepare` | `.github/workflows/production.yml` |
| Fork override | `RL_CRYPTO_TICKERS` repository variable | `production.yml` `prepare` |
| First-train requirement | a ticker with no bundle cannot record live | `pipeline/live.py:168`; README |
| Per-ticker cache | `data/<slug>/{candles,live}.jsonl`, `summary.json`; `models/<slug>.json` | `data/README.md` |
| Current baseline | ETH, BTC, SOL, XRP, BNB, ADA, AVAX, LINK, LTC, DOGE, ATOM, NEAR, FIL, SUI, APT | `tickers.py` |

## 2. Procedure

### 2.1 Screen the candidate

```bash
nix develop --command bash -c \
  'uv run python tools/probe_tickers.py --tickers NEW/USD --history-days 60 --lookback-hours 3'
```

The probe is **read-only** and exits with `os._exit(0)` to avoid ccxt socket
teardown; that is deliberate, not a crash. Read:

- `market.candles` — a full 60-day hourly window is ~1,440 bars. A thin result
  is a thin ticker, not a pipeline bug.
- `market.first` / `market.last` — the actual span, not just the count.
- `market.snapshot_price` / `resolved_symbol` — does the live snapshot resolve?
- `news.terms` — did the symbol resolve to real search terms?
- `news.hour_vectors` / `populated_hours` / `articles` — what actually reaches
  the feature join. Empty hourly buckets do not help.
- `news.news_errors` and any `*_error` — surface the failure; do not infer
  success from a blank field.

Record the result in the **screening ledger** (memory) with the date and the
window used.

### 2.2 Apply the policy

- **Stablecoins:** exclude (`USDT/USD`, `USDC/USD`, …). Near-zero variance is
  not a training target.
- **Correlation duplicates:** compare daily/hourly returns with the existing
  watchlist. If the correlation is very high (the repo cites BCH↔BTC ~0.9,
  ETC↔ETH ~0.95 as rejections), exclude it and record the reason.
- **Liquidity/news:** a full candle fill with no populated news is still a
  market-only ticker; decide explicitly whether that is acceptable. The current
  archive is market-only anyway, but live news is part of the record.
- **Already tracked:** normalize the symbol (`ETH/USD`); do not add a case or
  separator variant.

There is no automated correlation tool today (TO-002); a quick pandas/numpy
correlation over the committed archives is enough for a decision. Record the
method and the number.

### 2.3 Add to the watchlist

Edit `src/rl_crypto/tickers.py`:

```python
TRACKED_TICKERS: tuple[str, ...] = (
    ...
    "NEW/USD",
)
```

Why this is enough:

- `pipeline` imports `TRACKED_TICKERS`; the workflow `prepare` step loads
  `tickers.py` directly (it deliberately does not import the package).
- The matrix, slug, artifacts, and dashboard all follow.
- `tests/test_workflows.py` asserts the matrix comes from `tickers.py` and that
  the slug rule agrees.

Run the offline suite before any network work:

```bash
nix develop --command bash -c 'uv run python -m pytest -m "not live" -q'
nix develop --command bash -c 'uv run ruff check src/ tests/ examples/ tools/'
```

### 2.4 Seed, train, and verify

The first run is the slow one; later runs refresh.

```bash
nix develop --command bash -c \
  'uv run rl-pipeline run --tickers NEW/USD --history-days 60 --fetch-timeout 600'
```

Verify:

- `data/new-usd/candles.jsonl` exists and grows (upsert, sorted, canonical `…Z`);
- `data/new-usd/summary.json` has `schema: rl-crypto/pipeline/3`,
  `archive.rows` matches the candle line count, and a `holdout` block;
- `models/new-usd.json` exists, its `feature_version` matches
  `rl-crypto/features/2`, and `feature_version_mismatch` is false;
- the holdout/trusted counts are plausible (directional may be 0 — that is
  expected, not a failure).

Then record one live point:

```bash
nix develop --command bash -c 'uv run rl-pipeline live --tickers NEW/USD'
```

Check the appended record has `predictions`, `market`, and `news` blocks.

### 2.5 Dashboard and docs

```bash
nix develop --command bash -c 'uv run python tools/render_dashboard.py'
```

The dashboard is generated from the cache; do not hand-edit the block between
the markers. If the README watchlist prose is out of date, update the
hand-written text — not the generated table.

### 2.6 CI and the scheduled matrix

- `ci.yml` runs the offline suite; `test_workflows.py` checks the matrix source.
- `production.yml` will include the new ticker automatically on the next run.
- If the ticker is added to a fork, prefer `RL_CRYPTO_TICKERS` over a code edit;
  a dispatch can override it per run.

### 2.7 Retire a ticker

1. Remove it from `TRACKED_TICKERS` (and any fork variable).
2. Decide the data/model retention explicitly:
   - keep `data/<slug>/` and `models/<slug>.json` as historical evidence, or
   - remove them with a dated note (this is a data change; run
     `data-quality-audit` first and commit it separately).
3. Update the dashboard (regenerate) and any hand-written watchlist prose.
4. Run the offline suite; confirm the workflow matrix no longer includes it.
5. Record the retirement and reason in the screening ledger.

Do not delete the archive silently. A retired ticker's history may be needed to
reproduce a model or a backtest.

### 2.8 Fork onboarding

Read `README.md` §Forking. The short version:

1. Enable Actions in the fork (scheduled runs only fire on the default branch).
2. Set `RL_CRYPTO_TICKERS` (e.g. `BTC/USD`); the scheduled `prepare` reads it.
3. Optionally tune `RL_CRYPTO_MAX_AGE_MINUTES`.
4. Dispatch a **train** run first; then the schedule maintains it.
5. The upstreams are public sibling checkouts; no secrets are required.

A fork's `data/` and `models/` diverge from upstream by design.

## 3. Update this file

After an onboarding/retirement:

1. Add the candidate to the **screening ledger** with accept/reject and reason.
2. Update the baseline watchlist if it changed.
3. Add a finding if a screening or workflow rule was missing.
4. Update the improvement map.

## Memory — baseline watchlist

**As of 2026-10-23** (`HEAD` `213d17b`): 15 tickers, in `tickers.py` order:

`ETH/USD, BTC/USD, SOL/USD, XRP/USD, BNB/USD, ADA/USD, AVAX/USD, LINK/USD, LTC/USD, DOGE/USD, ATOM/USD, NEAR/USD, FIL/USD, SUI/USD, APT/USD`

Policy from the file: stablecoins and high-correlation duplicates are
deliberately excluded; membership is evidence-based (60-day 1h fill, populated
hourly news).

## Memory — screening ledger

| Date (UTC) | Candidate | Decision | Evidence / reason |
| --- | --- | --- | --- |
| 2026-10-23 | existing 15 | accepted baseline | recorded in `tickers.py`; re-probe only if the archive looks thin |

Add a row for every new candidate. Keep rejected candidates and the reason —
they are the expensive knowledge.

## Memory — active findings

### TO-001 · Screening is manual and not gated · OPEN (since 2026-10-23)
- **Layer:** process/tooling (`tools/probe_tickers.py`)
- **Evidence (2026-10-23):** the probe prints a report, but nothing persists the result or fails CI on a thin ticker; `tickers.py` is edited by hand.
- **Read:** the policy is sound, but a future addition can claim "probed" without evidence. Record the probe output in the ledger (this skill) at minimum.
- **Map:** IM-1
- **Status:** open.

### TO-002 · No cross-asset correlation helper · OPEN (since 2026-10-23)
- **Layer:** tooling (`tools/`)
- **Evidence (2026-10-23):** `tickers.py` documents BCH↔BTC ~0.9 and ETC↔ETH ~0.95 as rejections, but there is no script that computes the correlation from the committed archive.
- **Read:** the rejection rule is only as good as the measurement. A small offline helper over `candles.jsonl` would make it reproducible.
- **Map:** IM-2
- **Status:** open.

### TO-003 · A new ticker cannot record live until trained · EXPECTED (since 2026-10-23)
- **Layer:** runner/bundle (`pipeline/live.py:168`)
- **Evidence (2026-10-23):** `load_bundle(default_bundle_path(...))` is mandatory; a new ticker has no `models/<slug>.json` until a `train` run.
- **Read:** dispatch `train` first. This is documented in the README and is not a bug; a live-only matrix job for an untrained ticker will fail.
- **Map:** none
- **Status:** expected.

### TO-004 · Stablecoin/correlation exclusions live only in a docstring · EXPECTED (since 2026-10-23)
- **Layer:** policy (`tickers.py:19-32`)
- **Evidence (2026-10-23):** the rationale is prose; `parse_tickers` accepts any string and there is no code-level reject list.
- **Read:** policy-by-docstring is acceptable for a human-curated watchlist; do not add a hard-coded blocklist that prevents a deliberate exception. The ledger is the enforcement.
- **Map:** IM-1
- **Status:** expected.

### TO-005 · Retirement does not define data/model retention · OPEN (since 2026-10-23)
- **Layer:** policy/data (`data/README.md`, `models/README.md`)
- **Evidence (2026-10-23):** removing a ticker from `tickers.py` stops new runs, but the existing `data/<slug>/` and `models/<slug>.json` are untouched and undocumented.
- **Read:** keeping history is the safe default; deleting it is a data change that must be explicit, dated, and audited. Decide the default before the next retirement.
- **Map:** IM-3
- **Status:** open.

### TO-006 · The fork watchlist is not validated · OPEN (since 2026-10-23)
- **Layer:** workflow (`production.yml` `prepare`)
- **Evidence (2026-10-23):** `RL_CRYPTO_TICKERS` is parsed by `parse_tickers`; an empty/blank value falls back to the tracked set, and an invalid symbol becomes a matrix entry that fails at fetch time.
- **Read:** a fork typo costs a matrix job, not a wrong cache. Document the accepted forms and the empty-value behavior.
- **Map:** IM-4
- **Status:** open.

### TO-007 · The probe exits with `os._exit(0)` · EXPECTED (since 2026-10-23)
- **Layer:** tooling (`tools/probe_tickers.py:106`)
- **Evidence (2026-10-23):** `os._exit(0)` skips atexit/thread teardown because ccxt keeps sockets alive.
- **Read:** do not wrap the probe in a process that expects normal cleanup; the exit code is still 0. It is read-only and does not touch `data/`.
- **Map:** none
- **Status:** expected.

## Memory — resolved & expected

- **R-001 · Adding a ticker to `tickers.py` updates the matrix** — the `prepare` step loads that one file; `test_workflows.py` pins it. Do not add a second list in YAML.
- **R-002 · A ticker's first run is a backfill; later runs are refresh** — `--history-days 60` seeds the archive; the normal cadence tops it up.
- **R-003 · The archive is upsert-by-timestamp** — re-running a backfill does not duplicate bars; a still-open bar is corrected on the next run.
- **R-004 · A fork narrows with `RL_CRYPTO_TICKERS`** — no code edit; the dispatch input can override per run.
- **R-005 · The watchlist policy is human-curated on purpose** — stablecoins and highly correlated assets are excluded for training value, not because they cannot be fetched.

## Memory — improvement map

| ID | Finding | Layer | Action | Target | Risk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| IM-1 | TO-001, TO-004 | process | Persist a probe report in the screening ledger; add a documented checklist (no CI gate needed until the list grows) | this skill | none | proposed |
| IM-2 | TO-002 | tooling | Add a small offline correlation helper over `data/*/candles.jsonl`, with the same reject threshold documented in `tickers.py` | `tools/` | low | proposed |
| IM-3 | TO-005 | policy | Document the retention default: keep archive/model history on retirement; deletion is a separate dated data change | `data/README.md`, `models/README.md` | none | proposed |
| IM-4 | TO-006 | workflow | Validate the fork watchlist (trim, reject blanks, report invalid symbols) in `prepare` | `production.yml` | low; workflow change | proposed |
| IM-5 | all | docs | Link this onboarding checklist from the README watchlist section | `README.md` | none | proposed |

## Memory — audit log

| Run (UTC) | HEAD | Watchlist | Probe | Verdict |
| --- | --- | --- | --- | --- |
| 2026-10-23T23:18Z | `213d17b` | 15 | not re-run this session (read-only review) | policy sound; screening/retention are manual and need a ledger |

## Memory — open questions

- **Q-1** What is the minimum candle fill for acceptance — 1,440/1,440, or a percentage with a stated tolerance? The probe reports the count; the policy is prose.
- **Q-2** What correlation threshold is disqualifying? The docstring cites ~0.9/0.95; record the exact cutoff if a helper is added.
- **Q-3** On retirement, is keeping the archive the default forever, or is there a retention window? This affects repo size and audit scope.
- **Q-4** Should `RL_CRYPTO_TICKERS` accept a ticker not in `TRACKED_TICKERS`? Today it does; a fork may want that, but it bypasses the screening ledger.
- **Q-5** Should the probe write a machine-readable report (JSON) so the ledger is reproducible rather than transcribed?

