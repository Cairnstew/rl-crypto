# Committed pipeline cache

`data/` is the **git-committed, growing cache** written by the scheduled
workflows (`.github/workflows/pipeline.yml` + `.github/workflows/live.yml` →
`tools/run_pipeline.py`, `tools/record_live.py`) and by hand:

```bash
uv run python tools/run_pipeline.py --ticker ETH/USD --history-days 60
uv run python tools/record_live.py   --ticker ETH/USD   # frequent live point
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

- **pipeline.yml** — every 2h (and `workflow_dispatch`): grow the archive,
  retrain, append one live record, commit `models/ + data/`.
- **live.yml** — every 30 min (and `workflow_dispatch`): load the latest
  committed bundle, fetch a short market window for features, append one live
  record (with live news), commit `data/` only. Each run is cheap: no training.

The two workflows share a concurrency group so their write-backs never race.

## Seeding locally (recommended before scheduling)

Run it **locally first** to populate the archive, then let the scheduled
workflows refresh on top:

```bash
uv run python tools/run_pipeline.py --ticker ETH/USD --history-days 60 --fetch-timeout 600
```

Local runs reuse the sibling projects' on-disk caches
(`../tickerkit/tickerkit_cache.sqlite3`,
`../ticker-news-signals/ticker_news_signals_cache.sqlite3`), so heavy backfills
can complete here and are then cheap refresh work for the GitHub runner. Commit
the result — the scheduler only ever appends to it.