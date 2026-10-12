# Committed pipeline cache

`data/` is the **git-committed, growing cache** written by the scheduled
workflows (`.github/workflows/pipeline.yml` + `.github/workflows/live.yml` →
`tools/run_pipeline.py`, `tools/record_live.py`) and by hand:

```bash
uv run python tools/run_pipeline.py --history-days 60          # whole watchlist
uv run python tools/record_live.py                             # frequent live points
uv run python tools/record_live.py --tickers BTC/USD ETH/USD   # just these
```

**The tracked watchlist** lives in [`src/rl_crypto/tickers.py`](../src/rl_crypto/tickers.py)
and is the single source of truth for every runner — pipeline, local timer, and
manual dispatch. Every ticker in it is recorded on **every** run (no stagger, no
rotation). Add or retire a ticker there and all three paths follow. Screening
new candidates:

```bash
uv run python tools/probe_tickers.py --tickers BTC/USD SOL/USD --history-days 60
```

Per ticker, under `data/<slug>/` (e.g. `data/eth-usd/` for `ETH/USD`):

| file | contents | growth |
| ---- | -------- | ------ |
| `candles.jsonl` | **raw market archive** — one clean OHLCV bar per interval, ticker+timestamp+OHLCV, sorted | **only grows**: upsert by timestamp, never shrinks, never duplicates |
| `live.jsonl` | the live unified record (market **and** live news) + attached Tier-2 outlook | appended, capped at recent records (default 1000; more frequent via live.yml) |
| `summary.json` | run metadata, archive span, model metrics, outlook, notes | overwritten every pipeline run |

The trained Tier-1 bundles land in [`models/`](../models/README.md). Training
reads the **whole** `candles.jsonl` archive every run and recomputes the
forward-shifted targets on the fly, so a bar's labels only become defined once
its future is archived — the archive is the single source of truth, and no
derived rows are ever cached stale.

> **News history is intentionally absent.** The runners fetch market history
> only; news features appear as `has_news=0` until RL training lands. Live news
> still shows up in `live.jsonl` via the unified record.

## Cadence

Both scheduled paths record **every tracked ticker on every run** — nothing is
staggered or rotated, so every asset gets an equal-resolution series.

- **pipeline.yml** (GitHub Actions) — every 2h (and `workflow_dispatch`): grow
  each ticker's archive, retrain its bundle, append one live record, commit
  `models/ + data/` **in a single commit**. Tickers are processed one at a time
  on purpose (a deep history backfill spends many upstream calls per series),
  and in one job rather than a matrix, so N tickers never race to push `data/`.
- **Local systemd timer** — every 30 min on this machine, the *primary* frequent
  live path: `tools/record_live.sh` syncs, records one live point per ticker
  (market + live news + outlook), and pushes `data/`. Install with:

  ```bash
  mkdir -p ~/.config/systemd/user
  # units rl-crypto-live.service / rl-crypto-live.timer (OnCalendar=*:10/30)
  systemctl --user daemon-reload
  systemctl --user enable --now rl-crypto-live.timer
  ```

- **live.yml** (GitHub Actions) — `workflow_dispatch` only: the on-demand/backup
  live recorder (its cron was dropped because Actions didn't deliver schedule
  events for a second scheduled workflow on the young repo).

### Upstream request budget

Both runners respect the free-tier limits of the upstreams:

- **Market snapshots** are fetched for the whole watchlist in **one**
  `aget_tickers()` batch, so adding tickers adds concurrency, not requests per
  minute — one shared rate limiter, circuit breaker and connection pool.
- **Feature windows** are built from the committed archive plus a short live
  tail (`--tail-hours`, default 6h), so a 30-minute tick costs one small fetch
  per ticker instead of a deep history pull.
- **Deep history** stays single-asset and serial, as upstream tickerkit intends.

**A full watchlist tick costs ~23 minutes** at 15 tickers — about 2 minutes for
the shared market batch and ~90 s per ticker of live news (ticker-news-signals
has no batch API and re-pulls its feeds per ticker). That fits inside the
30-minute cadence, but the margin is now the binding constraint on how large the
watchlist can get:

| watchlist | measured tick | headroom in 30 min |
| --------- | ------------- | ------------------- |
| 10        | ~13 min       | ~17 min             |
| **15** (current) | **~23 min** | ~7 min |
| ~19       | ~30 min       | **none — ticks get coalesced** |

When the timer overruns, systemd coalesces rather than running two ticks at
once, so the series loses resolution rather than corrupting. Growing the
watchlist further means parallelising the per-ticker news pulls (bounded, e.g.
3–4 workers) — deliberately not done yet, since concurrent pulls of the same
public RSS hosts risk the IP-level throttling that would cost far more data than
the extra tickers are worth. The timer unit is a `Type=oneshot` service, whose
start timeout is `infinity`, so a long tick is never killed mid-run.

The Actions pipeline has no such ceiling: it fans out as a matrix
(`prepare` → `produce` per ticker → `merge`), so wall clock tracks the slowest
single ticker rather than the sum of all of them.

The Actions pipeline and the local timer both write `data/`; the pipeline also
owns `models/`. Write-backs from Actions serialize via a shared concurrency
group, the merge job is the only Actions-side writer to `main`, and the timer
does a `--ff-only` pull before pushing plus a rebase-and-retry if a pipeline
commit lands mid-run.

## Seeding locally (recommended before scheduling)

Run it **locally first** to populate the archive, then let the scheduled
workflows refresh on top:

```bash
uv run python tools/run_pipeline.py --history-days 60 --fetch-timeout 600
```

Narrow it to seed just one asset:

```bash
uv run python tools/run_pipeline.py --tickers ETH/USD --history-days 60 --fetch-timeout 600
```

Local runs reuse the sibling projects' on-disk caches
(`../tickerkit/tickerkit_cache.sqlite3`,
`../ticker-news-signals/ticker_news_signals_cache.sqlite3`), so heavy backfills
can complete here and are then cheap refresh work for the GitHub runner. Commit
the result — the scheduler only ever appends to it.