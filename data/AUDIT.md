# Data-quality audit — 2026-10-12

Audit of the committed cache under `data/`, plus upstream reproduction runs
against the sibling checkouts `../tickerkit` and `../ticker-news-signals`.

Re-run any time (read-only, no network):

```bash
uv run python tools/audit_data_quality.py              # summary table
uv run python tools/audit_data_quality.py --details    # + per-fault lines
```

## Verdict

The **archive is sound**. Every fault worth reporting is in the **live** records,
and every one of those traces to an upstream defect, not to anything this repo
wrote. Three issues were filed:

| upstream | issue | defect |
| -------- | ----- | ------ |
| `Cairnstew/tickerkit` | [#1](https://github.com/Cairnstew/tickerkit/issues/1) | `coinpaprika` keyless API now returns `402 Payment Required` — every snapshot records 3 `ProviderError`s |
| `Cairnstew/tickerkit` | [#2](https://github.com/Cairnstew/tickerkit/issues/2) | CoinGecko keyless 25 rpm trips the circuit breaker mid-batch, nulling the four aggregate market fields |
| `Cairnstew/ticker-news-signals` | [#1](https://github.com/Cairnstew/ticker-news-signals/issues/1) | `rss:cryptoslate` 429s on every pull — 1 of 19 feeds missing from most sentiment vectors |

## `candles.jsonl` — clean

15 tickers × 1440–1457 hourly bars, all spanning `2026-08-13 → 2026-10-12 13:00Z`:

| check | result |
| ----- | ------ |
| hourly continuity | 0 gaps, 0 duplicate/non-monotonic timestamps in 21,750 bars |
| OHLC consistency | 0 violations (`high` always brackets `max(open,close)`, `low` always brackets `min(open,close)`, no `high < low`) |
| non-positive / non-finite / null prices | 0 |
| volume | 0 negative, 0 zero-volume bars |
| freshness | every ticker's last bar is `13:00Z` (the pipeline cadence, not staleness) |

The `>5%` per-hour moves (12 in `near-usd`, 10 in `fil-usd`, 0 in BTC/SOL) are
**market action, not bad bars**: they hit the *same* bars across nearly every
asset — `2026-08-22 04:00→05:00Z` and `2026-10-08 14:00→15:00Z` — each accompanied
by a volume spike (e.g. DOGE 145M → 230M, XRP 100M) and no immediate reversion.
Avenue-specific corruption would not correlate across assets that way.

## `live.jsonl` — the real findings

541 records over 15 tickers:

| metric | count | share |
| ------ | ----- | ----- |
| `market.errors >= 1` | 445 | 82% |
| `market.errors >= 3` | 74 | 14% |
| `aggregated_price` / `change_24h_pct` / `volume_24h` / `market_cap` **all null** | 40 | 7% |
| `news.errors >= 1` | 341 | 63% |
| — of which `news.errors == 1` | 314 | — |
| — of which `news.errors == 2` | 27 | — |

Other checks that passed: no duplicate `generated_at`, no malformed schema ids,
all sentiments within `[-1, 1]`, live `price` always within 2% of its own
`consensus_price`, and `latest_sentiment_at` never later than `generated_at`
(p50 0.5h, max 1.4h behind).

### 1. CoinPaprika is dead keyless (tickerkit #1)

`https://api.coinpaprika.com/v1/search/...` and `/v1/tickers/...` both return
`402 Payment Required` from a clean IP with tickerkit's own UA, sequential and
concurrent. Onset observed mid-sweep on 2026-10-12 ~19:25Z (the first 13 tickers
of a sequential run returned 0 errors, the last two returned 3 each).

Every batch snapshot therefore carries 3 `ProviderError`s for capabilities
`resolve`, `meta`, `agg_market`, and `DataQuality.ok` is `False` even though
`market.price` is perfect. The on-disk `tickerkit_cache.sqlite3` holds
`coinpaprika:search:*` entries, which is why this looks intermittent — but
`SQLiteCache.get` already returns `MISS` for them (stored ~74h ago, `ttl=86400`).

### 2. Aggregate market fields silently null (tickerkit #2)

40 records have all four aggregate fields null, every one inside
**2026-10-12 13:17–14:20Z** with `market.errors` 3–6. Reproduction:
`aget_tickers(TRACKED_TICKERS)` gives one `coingecko` suppression per ticker
(`rate limit still in effect; request suppressed (retry after 57s)`) and 12 of
15 tickers come back with `cap=None`.

`_blocked_until` is only set after a *real* 429, so CoinGecko genuinely 429s at
tickerkit's assumed keyless 25 rpm (~45 requests in ~2 minutes for a 15-ticker
sweep). A standalone request returns `200`, which is why single-ticker runs look
healthy. Combined with #1 there is no aggregator left at all — the price still
comes from ccxt venue consensus, so nothing alarms.

### 3. `rss:cryptoslate` 429s on every pull (tns #1)

```
x14: ('rss:cryptoslate', 'news',     'rss:cryptoslate returned HTTP 429 after 4 attempt(s)')
x4 : ('rss:cryptoslate', 'trending', 'rss:cryptoslate returned HTTP 429 after 4 attempt(s)')
```

from a fresh 15-ticker sweep; no other feed failed once. The feed itself is
**up** — `curl -A "Mozilla/5.0 …"` returns `200 application/rss+xml` — so this
is a Cloudflare-managed 429 that the pull pattern/rate runs into. The retry
budget against a 429 with no `Retry-After` is ~2 s (0.5/1/2 s backoff,
`max_retries=3`), which is not enough to clear the challenge.

The 27 records with `news.errors == 2` are cryptoslate plus one intermittent
feed; the stored count alone cannot name it, which is why the issue asks
upstream to record *which* source failed rather than only how many.

## Checks that came back clean (deliberately, not by luck)

- **No silent corruption in the archive** — the joins, upserts and 2h pipeline
  cadence are doing their job across 21,750 bars.
- **`signal_hours` is not an off-by-one.** 499 of 541 records show
  `lookback_hours=1` with `signal_hours=2`. That is correct: a 1-hour window
  starting at, say, `13:47` crosses the `13:00` and `12:00` bucket boundaries.
  The 32 records with `signal_hours=1` are exactly those generated within ~15
  minutes of the hour. Only inspected because it *looks* wrong.
- **`has_news=0` in the features is by design** (see `data/README.md`): news
  history is not ingested, so `ret_*`/`sentiment` features are 0 for archived
  rows while live `news.articles` is populated.

## Not filed upstream (rl-crypto side)

Two things this repo should own, no upstream issue for them:

1. **`market.ok` is misleading.** `unify.py` sets `market_ok = True` on any
   non-raising call, so `live.jsonl` shows `ok: true` next to `errors: 1`.
   tickerkit's own `DataQuality.ok` is `not self.errors` — the opposite
   meaning. Any consumer reading `ok` as "clean" is being told the wrong thing.
2. **Only the error *count* is stored.** `market_errors = len(snap.data_quality.errors)`
   discards the provider, capability and message, which is why pinning the above
   faults to their causes required re-running live sweeps rather than reading
   `data/`. Persisting the error detail (or at least provider + capability)
   would make the next audit a file read.

## Related

- `data/README.md` — what each file is, cadence, upstream budget
- `tools/audit_data_quality.py` — the audit itself
