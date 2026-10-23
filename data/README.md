# Committed pipeline cache

`data/` is the **git-committed, growing cache** written by the scheduled
workflow (`.github/workflows/production.yml` → `rl-pipeline live` / `rl-pipeline run`)
and by hand:

Quality audits of this directory live in [`AUDIT.md`](AUDIT.md); re-run one with
`uv run python tools/audit_data_quality.py` (read-only, no network).

```bash
uv run rl-pipeline run --history-days 60           # whole watchlist
uv run rl-pipeline run --retrain-only              # models only, no network
uv run rl-pipeline live                            # frequent live points
uv run rl-pipeline live --tickers BTC/USD ETH/USD  # just these
```

**The tracked watchlist** lives in [`src/rl_crypto/tickers.py`](../src/rl_crypto/tickers.py)
and is the single source of truth for every runner — the scheduled workflow and
manual dispatch (a fork can narrow it with the `RL_CRYPTO_TICKERS` repository
variable). Every ticker in it is recorded on **every** run (no stagger, no
rotation). Add or retire a ticker there and all paths follow. Screening
new candidates:

```bash
uv run python tools/probe_tickers.py --tickers BTC/USD SOL/USD --history-days 60
```

Per ticker, under `data/<slug>/` (e.g. `data/eth-usd/` for `ETH/USD`):

| file | contents | growth |
| ---- | -------- | ------ |
| `candles.jsonl` | **raw market archive** — one clean OHLCV bar per interval, ticker+timestamp+OHLCV, sorted, timestamps canonically `…Z` | **only grows**: upsert by timestamp, never shrinks, never duplicates |
| `live.jsonl` | the live unified record (market **and** live news) + attached Tier-2 outlook | appended, capped at recent records (default 1000 — a rolling ~21 days at 30-minute ticks) |
| `summary.json` | run metadata, archive span, **out-of-sample** model quality, outlook, notes | overwritten every pipeline run |

Timestamps are canonicalised to ISO-8601 `…Z` on both read and write, and the
archive is keyed on that canonical form. Two spellings of the same hour would
otherwise file as two bars and silently duplicate the archive.

### What `summary.json` reports

`targets.<name>` carries **two** metric blocks plus a verdict:

```json
"fwd_volume_ratio": {
  "in_sample":  { "rmse": 0.55, "mae": 0.42, "r2": 0.44 },
  "holdout":    { "n_samples": 299,
                  "metrics":  { "rmse": 0.61, "mae": 0.47, "r2": 0.31 },
                  "baseline": { "rmse": 0.73, "…": "…" },
                  "beats_baseline": true },
  "trusted": true
}
```

`in_sample` is fit quality on the rows the model was fitted on. **`holdout` is the
number to read**: it is measured on the newest ~20% of rows, which were withheld
from training, and compared against a constant predictor. See
[`models/README.md`](../models/README.md) for the full protocol and what is
trusted today (short version: nothing directional is).

The trained Tier-1 bundles land in [`models/`](../models/README.md). Training
reads the **whole** `candles.jsonl` archive every run and recomputes the
forward-shifted targets on the fly, so a bar's labels only become defined once
its future is archived — the archive is the single source of truth, and no
derived rows are ever cached stale.

## Historic live records have no outlook

`live.jsonl` rows recorded before the out-of-sample gate was added carried a
`predictions` block with a `signal` and `confidence` produced by an in-sample
model. Those numbers were not merely noisy, they asserted skill that did not
exist, so the `predictions` block has been removed from every historical record
and each row carries a note saying so. The market and news halves of those
records are untouched and remain the best record of what each interval looked
like.

## Rebuilding models without spending quota

```bash
uv run rl-pipeline run --retrain-only
```

Recomputes features, labels and bundles from the committed archive — the right
move after a feature-contract change, where the bars have not moved and only the
model is stale. Takes no network and appends no live records.

> **News history is intentionally absent.** The runners fetch market history
> only; news features appear as `has_news=0` until RL training lands. Because a
> constant column teaches a model nothing and would arrive out-of-distribution at
> serve time, the trainer **drops** those features from every bundle and records
> them under `dropped_features` rather than fitting them to zero. Live news still
> shows up in `live.jsonl` via the unified record.

## Cadence

The scheduled path records **every tracked ticker on every run** — nothing is
staggered or rotated, so every asset gets an equal-resolution series.

- **production.yml** (GitHub Actions) — one scheduler, two modes. Every
  **30 minutes** a `live` run appends one record per ticker (market snapshot +
  live news + outlook); on **even UTC hours** a `train` run also grows the
  archive, rebuilds features/labels and retrains the Tier-1 bundle. Both fan out
  as a matrix (`max-parallel: 4`) and stage each ticker's cache as an artifact;
  a single `merge` job is the only writer and makes **one commit per run**, so N
  tickers never race to push `data/`.
- **watchdog.yml** (GitHub Actions) — hourly: if the newest `data/` commit is
  older than `RL_CRYPTO_MAX_AGE_MINUTES` (default 75), it dispatches a `live`
  run. This is the safety net for GitHub's best-effort schedule delivery, and
  replaces the old local systemd timer — collection is fully remote.
- `workflow_dispatch` on **production.yml** runs either mode on demand, on the
  whole watchlist or a narrowed ticker list.

### Upstream request budget

Both runners respect the free-tier limits of the upstreams:

- **Market snapshots** are fetched for the whole watchlist in **one**
  `aget_tickers()` batch, so adding tickers adds concurrency, not requests per
  minute — one shared rate limiter, circuit breaker and connection pool.
- **Feature windows** are built from the committed archive plus a short live
  tail (`--tail-hours`, default 6h), so a 30-minute tick costs one small fetch
  per ticker instead of a deep history pull.
- **Deep history** stays single-asset and serial, as upstream tickerkit intends.

A full sequential watchlist tick measured ~23 minutes at 15 tickers — about 2
minutes for the shared market batch and ~90 s per ticker of live news
(ticker-news-signals has no batch API and re-pulls its feeds per ticker). The
workflow does not pay that sum: it fans out as a matrix, so wall clock is the
slowest single ticker (~2 min) and the 30-minute schedule has ample slack.
`max-parallel: 4` is the deliberate ceiling — the limit is the public RSS hosts'
per-IP tolerance, not our CPU, and concurrent pulls of the same hosts would cost
far more data than the extra tickers are worth.

The workflow owns both `data/` and `models/`. Only the `merge` job pushes, it
holds the `rl-crypto-cache` concurrency group, and a rejected push is rebased and
resolved (`rl-pipeline conflicts`) before retrying — so a hand push or a
manual local run can never strand it. A local `rl-pipeline live`/`run` is
optional (seeding, spot checks); it is not part of the schedule.

## Seeding locally (recommended before scheduling)

Run it **locally first** to populate the archive, then let the scheduled
workflows refresh on top:

```bash
uv run rl-pipeline run --history-days 60 --fetch-timeout 600
```

Narrow it to seed just one asset:

```bash
uv run rl-pipeline run --tickers ETH/USD --history-days 60 --fetch-timeout 600
```

Local runs reuse the sibling projects' on-disk caches
(`../tickerkit/tickerkit_cache.sqlite3`,
`../ticker-news-signals/ticker_news_signals_cache.sqlite3`), so heavy backfills
can complete here and are then cheap refresh work for the GitHub runner. Commit
the result — the scheduler only ever appends to it.