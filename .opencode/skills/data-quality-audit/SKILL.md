---
name: data-quality-audit
description: >-
  Use when auditing or improving data quality in rl-crypto: the committed
  data/ cache (candles.jsonl, live.jsonl, summary.json) or models/. Triggers:
  "audit data quality", "data/ looks wrong", "missing bars", "null market
  fields", "stale cache", "news errors", "market errors", "is the archive
  clean", "verify the cache", "improve data quality", "upstream tickerkit",
  "ticker-news-signals upstream". Runs the read-only audit, classifies each
  fault by origin, maps it to a fix, and appends durable findings to this file
  so the next audit starts from accumulated knowledge instead of zero. Upstream
  fixes are delegated to an OpenCode Ensemble agent space, not patched here.
---

# Data-quality audit & improvement loop

This file is two things:

1. the procedure to audit `data/` and `models/` **read-only, offline**; and
2. the **living memory** — the `Memory` sections at the bottom hold what
   previous audits proved. Read them before measuring, and update them after.
   A fresh audit that ignores the memory is a regression.

`data/AUDIT.md` is the long-form historical narrative (2026-10-12 and the
2026-10-13 follow-up). Read it for deep context; do **not** quote its counts as
current — every number there is dated. This skill's ledger is the current state.

Companion skill: **`pipeline-hardening`** audits the production pipeline that
writes this cache (exit codes, exception boundaries, atomic writes, modularity,
CI gates). Use it when the data is sound but the writer is fragile; use this
skill when the symptom is the data itself.

## 0. Contract

- **Audit first, mutate never (by default).** The audit is read-only and makes
  no network call. Reading committed files and `git log` is the whole job.
- **Classify before fixing.** A fault is only a fault after you know its layer:
  repo write path, upstream `tickerkit`, upstream `ticker-news-signals`, already
  committed history, market action, or design.
- **The archive only grows.** `candles.jsonl` is an upsert series. Never delete
  bars, never rewrite history, never "repair" a bar because it looks odd. A
  suspected bad bar is a finding, not an instruction to edit data.
- **No fabricated evidence.** Every memory entry carries a date and a number
  from a command you ran. "Seems noisy" is not a finding; put it in Open
  questions instead.
- **Memory is append-mostly, not append-always.** Update an existing finding in
  place; append only genuinely new findings, mapping rows and audit-log rows.
- **A fix is not part of an audit-only run.** When the user asks to *improve*,
  make the smallest durable fix, test it, and record it. Otherwise produce the
  audit plus the mapping and stop.

## 1. Ground truth and upstreams

| Path | What it is | Growth / owner |
| --- | --- | --- |
| `data/<slug>/candles.jsonl` | Raw OHLCV archive, one bar per hour, ticker + canonical ISO-8601 `…Z` timestamp | **Only grows**, upsert by timestamp. `src/rl_crypto/pipeline/run.py`; merged by `src/rl_crypto/pipeline/conflicts.py` |
| `data/<slug>/live.jsonl` | Unified market + news record plus the Tier-2 outlook | Append-only, capped at **1000** records (~21 days at 30-min ticks). `run.py` / `live.py` |
| `data/<slug>/summary.json` | Per-run metadata, archive span, out-of-sample model quality, outlook, notes | Overwritten each run; schema `rl-crypto/pipeline/3` |
| `models/<slug>.json` | Committed Tier-1 bundle | JSON, schema `rl-crypto/ml/1`; `feature_version` must match `FEATURE_VERSION` in `src/rl_crypto/features.py:53` |
| `src/rl_crypto/tickers.py` | The watchlist — one source of truth for every runner | 15 tickers as of 2026-10-23 |
| `data/AUDIT.md` | Dated audit narratives | Historical; not the current ledger |

### 1.1 External upstream repositories

rl-crypto owns **no collection code**. Everything under `data/` is the output
of two external repos, consumed as sibling checkouts and as flake inputs:

| Upstream | GitHub | Local checkout | Owns | Wired in |
| --- | --- | --- | --- | --- |
| `tickerkit` | `Cairnstew/tickerkit` | `../tickerkit` | Market snapshots, venue consensus price, OHLCV history, provider provenance and `ProviderError` (`DataQuality.ok`) | `pyproject.toml` `[tool.uv.sources]` (`tickerkit = { path = "../tickerkit", editable = true }`); `flake.nix:26` (`git+file:../tickerkit?ref=main`) |
| `ticker-news-signals` | `Cairnstew/ticker-news-signals` | `../ticker-news-signals` | Per-hour VADER sentiment vectors, feed provenance, tolerated source errors (`SourceErrorRecord`) | `pyproject.toml` `[tool.uv.sources]`; `flake.nix:27` (`git+file:../ticker-news-signals?ref=main`) |

They are declared as runtime dependencies (`tickerkit>=0.1`,
`ticker-news-signals>=0.2`) and overridden to editable sibling paths by
`[tool.uv.sources]`, so a local edit to either upstream is live in this venv.
Each also keeps its own on-disk TTL cache
(`../tickerkit/tickerkit_cache.sqlite3`,
`../ticker-news-signals/ticker_news_signals_cache.sqlite3`); a cached success
can hide a currently failing provider, so reproduce against a fresh/expired
cache before blaming the join.

**Ownership boundary.** rl-crypto joins, stores and evaluates their output. It
does not vendor or patch either upstream. A defect that reproduces against the
upstream's own code belongs upstream; a defect in the join, the stored record,
or the schema belongs here.

**Attribution is already available upstream.** tickerkit's `ProviderError`
carries `provider`, `capability`, `exception_type`, `message`
(`../tickerkit/src/tickerkit/models.py:648`); tns's `SourceErrorRecord` carries
`source`, `capability`, `exception_type`, `message`
(`../ticker-news-signals/ticker_news_signals/models.py:278`). This repo's
`unify.py` stores only `len(...)` (`src/rl_crypto/unify.py:126,144`), so losing
the provider/source is a **repo-side** fix (IM-1), not an upstream gap.
`data/AUDIT.md`'s older wording that upstream should "record which source
failed" predates the current upstream types; do not file that request again.

**Before calling a finding upstream:**

```bash
git -C ../tickerkit log -3 --oneline
git -C ../ticker-news-signals log -3 --oneline
gh issue list --repo Cairnstew/tickerkit
gh issue list --repo Cairnstew/ticker-news-signals
```

Reproduce with the smallest call that still fails. A 15-ticker batch can trip a
rate limit that a single-ticker call never sees — that is the upstream's shared
limiter, not the join.

When the upstream change is needed, use the Ensemble-space workflow in §3.7.1 —
do not edit the sibling checkout as the lead and do not paper over it here.

### 1.2 Cadence, and therefore what "stale" means

- `production.yml` runs **every 30 minutes** in `live` mode, and on **even UTC
  hours** in `train` mode (archive + retrain + one live record).
- The archive is **hourly**, so `hours_since_last_bar` around 1-2 h is the
  cadence, not staleness. The watchdog dispatches a run when the newest `data/`
  commit is older than **75 minutes** (`RL_CRYPTO_MAX_AGE_MINUTES`), so a
  commit-age breach is an Actions incident, not a data-quality fault.
- `live.jsonl` is capped, so trend questions older than the cap must read
  **git history** (`git log -p -- data/<slug>/live.jsonl`), not the working
  copy.

## 2. Environment

Run everything inside the dev shell — `uv` is not on the host `PATH`:

```bash
nix develop --command bash -c 'uv run python tools/audit_data_quality.py'
nix develop --command bash -c 'uv run python tools/audit_data_quality.py --details'
nix develop --command bash -c 'uv run python tools/audit_data_quality.py --ticker btc-usd --details'
```

Do not copy `/nix/store/...` paths into this file; they rot. If the dev shell is
unavailable and `.venv/bin/python` is used directly, its numpy/pandas wheels
need an **ELF64** `libz.so.1` and `libstdc++.so.6` on `LD_LIBRARY_PATH` (the
store also holds ELF32 builds — check `readelf -h`). Prefer the dev shell.

## 3. Procedure

### 3.1 Pin the run

```bash
date -u +%FT%TZ
git rev-parse --short HEAD
git log -1 --format='%h %cI %s' -- data/
```

Read the `Memory` sections. Then take the baseline; record it in the audit log
even when nothing changed.

### 3.2 Baseline

Run the audit tool (summary table, then `--details` for a ticker). Also run the
aggregate/error probe below — the tool's table is necessary but not sufficient.

```bash
python3 - <<'PY'
import json, glob, collections

rows = 0
null_agg = []
errs = collections.Counter()
news_errs = collections.Counter()
ok_with_errors = 0
for path in glob.glob("data/*/live.jsonl"):
    slug = path.split("/")[1]
    for line in open(path):
        r = json.loads(line)
        m, n = r.get("market") or {}, r.get("news") or {}
        rows += 1
        agg = [m.get(k) for k in ("aggregated_price", "change_24h_pct",
                                  "volume_24h", "market_cap")]
        if all(v is None for v in agg):
            null_agg.append((slug, r.get("generated_at"), m.get("errors")))
        errs[m.get("errors")] += 1
        news_errs[n.get("errors")] += 1
        if m.get("ok") is True and (m.get("errors") or 0) >= 1:
            ok_with_errors += 1

print("records:", rows)
print("market.errors distribution:", errs.most_common())
print("news.errors distribution:", news_errs.most_common())
print("ok:true with errors>=1:", ok_with_errors)
print("all-null aggregate records:", len(null_agg))
if null_agg:
    stamps = sorted(str(x[1]) for x in null_agg)
    print("  earliest:", stamps[0])
    print("  latest:  ", stamps[-1])
    print("  by date: ", collections.Counter(s[:10] for s in stamps).most_common())
PY
```

Record in the audit log: tickers, archive bars, live records, records with
`market.errors >= 1` and `news.errors >= 1`, all-null aggregate count and date
range, `stale_h`, and `priceDev` total.

### 3.3 Archive — `candles.jsonl`

A clean run looks like this (verified 2026-10-23):

| Check | Pass condition | How |
| --- | --- | --- |
| Continuity | 0 gaps, 0 duplicate or non-monotonic timestamps on the 1 h grid | audit table columns `gaps`, `dup` |
| OHLC shape | `high >= max(open, close)`, `low <= min(open, close)`, `high >= low` | `OHLC` column = 0 |
| Prices | all four finite and `> 0` | `nulls` = 0; per-ticker `--details` |
| Volume | 0 negative, 0 zero-volume bars | `vol0` column |
| Jumps | individually 0 or shared across assets at the same bar with a volume spike | `jumps` column; inspect timestamps |
| Freshness | `stale_h` ≈ cadence (≤ ~2 h) | `stale_h` column |
| Span | all tickers start/end within a few bars of each other | `first_bar` / `last_bar` from `--details` |

Per-ticker bar counts differ by a handful (1725 vs 1712, ETH 1729) because
fetch windows/listings differ. That is not a gap. A jump that hits every asset
at the same timestamp is **market action**; do not file it as corruption.

Any continuity/OHLC violation is **P1**: preserve the line numbers, map it to
the remediation policy (IM-5), and do not touch the file without a written plan.

### 3.4 Live records — `live.jsonl`

Checks:

- No duplicate `generated_at`; top-level `schema == "rl-crypto/1"`;
  `predictions.schema == "rl-crypto/ml/1"` when present.
- `mean_sentiment` / `latest_sentiment` in `[-1, 1]`;
  `latest_sentiment_at <= generated_at`.
- `market.price` within 2% of `market.consensus_price` (currently 0 faults).
- All-null aggregate block = **data loss** (P0/P1 depending on duration). Record
  the count, per-ticker spread and date range.
- `market.errors` / `news.errors` counts; cross-check `ok` against them. Note
  that `ok` is **not** `not errors` today — see DQ-001.
- `predictions.signal` / `confidence` null is **by design** while no directional
  target is trusted — see DQ-004.
- `priceDev` > 1% between a 30-minute live price and the nearest hourly bar is
  **expected** intra-hour movement — see DQ-006. Do not report it as a fault
  unless it is systematic (same timestamp across tickers, same direction) or
  exceeds 5% with no matching bar jump.

### 3.5 Summaries and models

```bash
python3 - <<'PY'
import json, glob, collections
schema, feat, beats = collections.Counter(), collections.Counter(), collections.Counter()
for path in sorted(glob.glob("data/*/summary.json")):
    d = json.load(open(path))
    schema[d.get("schema")] += 1
    feat[d.get("feature_version")] += 1
    for name, block in (d.get("targets") or {}).items():
        if (block.get("holdout") or {}).get("beats_baseline"):
            beats[name] += 1
print("summaries:", sum(schema.values()), "schema:", dict(schema))
print("feature_version:", dict(feat))
print("targets beating baseline:", dict(beats))
PY
```

Then check:

- Every summary is `rl-crypto/pipeline/3` and carries a `holdout` block.
- `feature_version` matches `FEATURE_VERSION` (`src/rl_crypto/features.py:53`).
- `archive.rows` equals the line count of the matching `candles.jsonl`
  (summary is written after the archive upsert; a mismatch means a merge or
  write-order bug).
- `samples.total` equals `archive.rows`; `samples.holdout` is > 0.
- Recompute the trusted counts; never copy the table from `models/README.md`.
  Directional targets (`fwd_return`, `fwd_direction`) beating baseline is the
  exception, not the norm. `fwd_volatility` and `fwd_volume_ratio` normally do.
- `notes` explains any withheld signal and any dropped features.
- Bundle `feature_version` matches the summaries; a mismatch is a stale bundle
  (run `rl-pipeline run --retrain-only`).

### 3.6 Classify

First map every check to a data-quality dimension, so "nothing parsed wrong" is
not mistaken for "the data is good":

| Dimension | Question | Check that answers it |
| --- | --- | --- |
| Completeness | Is a measurement missing? | all-null aggregates; null leaf fields |
| Validity | Is the value legal? | OHLC shape, prices > 0, sentiment in range, schema ids |
| Uniqueness | Is a row duplicated? | duplicate bar timestamps, duplicate `generated_at` |
| Consistency | Do independent views agree? | price vs consensus, summary rows vs candles, bundle vs feature version, `ok` vs errors |
| Timeliness | Is it current? | `stale_h` vs cadence, newest `data/` commit vs 75 min |
| Accuracy / join | Does live agree with the archive? | `priceDev`, with the intra-hour movement caveat |
| Provenance | Can a fault be attributed? | error counters carry provider/capability detail (currently no — IM-1) |
| Coverage | Is every asset/feed represented? | watchlist count, per-ticker records, news feeds |

Missing provenance or coverage is a mapping input (IM-1/IM-4), not a P1.

Severity:

| Level | Meaning | Example |
| --- | --- | --- |
| **P0** | Data loss: a measurement is null for a whole ticker/period | all four market aggregates null |
| **P1** | Integrity: the archive or join is actually wrong | gap, duplicate timestamp, OHLC violation |
| **P2** | Noise: counters fire but the data is present | `market.errors = 1`, news 429s |
| **P3** | Expected/design | withheld signal, `has_news = 0`, `priceDev`, cadence staleness |

Origin:

| Origin | How to confirm |
| --- | --- |
| Repo write path | The value is produced by `unify.py` / `models.py` / `pipeline/*`; reproduce from committed inputs |
| `tickerkit` | Reproduce one live sweep; check `git -C ../tickerkit log --oneline -5` and open issues |
| `ticker-news-signals` | Same, in `../ticker-news-signals`; check the per-feed failure |
| Committed history | Fault is confined to a dated range already explained by an upstream fix |
| Market action | Same bar across many tickers, volume spike, no reversion |
| Design | Documented in `data/README.md`, `models/README.md`, or this skill |

Rules:

- Reproduce once before blaming an upstream. A clean single-ticker call can look
  healthy while a 15-ticker batch trips a rate limit.
- Check whether an upstream fix landed before re-reporting a finding. A fixed
  cause is `FIXED — monitor`, not `OPEN`.
- A fix commit that says `Refs #N` does **not** close issue `#N`. Check both
  `git log` and `gh issue list`; a fixed code path with a still-open issue is
  `FIXED — monitor`, not `OPEN`.
- A count of 1 error with complete data is P2 noise; a count of 0 with nulls is
  worse than a count of 3 with complete data. Judge the fields, not the counter.

### 3.7 Improvement / mapping step (required)

For every P0-P2 finding, add or update a row in the **Improvement map**. The map
is the bridge from observation to durable improvement.

| Field | Content |
| --- | --- |
| ID | `IM-N` |
| Finding | `DQ-N` it addresses |
| Layer | repo / tickerkit / tns / data / policy |
| Action | concrete change, not a wish |
| Target | `path:line`, command, or upstream issue URL |
| Risk | schema break, migration, test gap |
| Status | proposed / in progress / upstream / monitoring / done |

Action ladder — take the **smallest durable rung** that removes the recurrence:

1. **Observe** — record the finding and its evidence.
2. **Document** — if it is expected, write why (so it is not re-investigated).
3. **Add a regression check** — a check in `tools/audit_data_quality.py` that
   would have caught it, plus a test. This is almost always the right first fix.
4. **Fix the write path** — `src/rl_crypto/unify.py`, `models.py`, or
   `pipeline/*`, with a schema decision if the record changes.
5. **File upstream** — every upstream defect goes to the sibling repo
   (`gh issue` there), never silently worked around here.
6. **Remediate data** — only for provable P1 corruption, with a dated plan, a
   backup, and a test. The archive is append-only.

Rules:

- Every mapping row names a real file/line or URL. If you cannot name one, the
  row is not ready.
- Prefer an **additive** record field over changing an existing meaning. If a
  meaning must change, bump the schema and audit consumers with
  `grep -rn '"ok"\|market_ok\|market_errors' src tests tools notebooks`.
- A P3 finding gets `closed: expected` with the reason; it must not reappear as
  a new finding next run.
- A schema or semantic change also updates `data/README.md` and/or
  `models/README.md`; an undocumented record change is an incomplete fix.
- Default to audit + map. Only implement when the user asked to improve, and
  keep the read-only audit separate from the mutation.

### 3.7.1 Upstream changes: spawn into an Ensemble space

When a mapping row says the fix belongs in `tickerkit` or
`ticker-news-signals`, the lead does **not** edit the sibling checkout. Use an
OpenCode Ensemble **agent space** — a full clone of a different repo, isolated
from this worktree.

1. **Load the `opencode-ensemble` skill** before creating a team or spawning. It
   is the only place the `team_*` tools and space semantics are documented.
2. **Check what is registered** in `spaces` in
   `~/.config/opencode/ensemble.json` (this environment:
   `/home/seanc/.config/opencode/ensemble.json`). A space exists only if it is
   listed; never invent a name — `team_spawn(space=...)` fails if it is not
   registered.
3. **As of 2026-10-23 neither upstream has a space.** Registered are
   `ensemble`, `agenix-manager`, `kraken-python`, `project-zomboid-servers` and
   `spotify-playlist-manager`. The correct outcome today is to report the
   missing space and ask to add it — not to silently fall back to editing
   `../tickerkit`.
4. **If the space exists:** create the team (`team_create`), add one task
   (`team_tasks_add`), then spawn a narrow builder, one at a time:
   `team_spawn(name="tickerkit-fix", space="tickerkit", worktree=false, prompt="…")`.
   If the upstream cause is not yet proven, spawn a read-only `scout` in the
   space first (`worktree=false`), then the builder. Spaces and worktrees are
   mutually exclusive, so `worktree=false` is required. The prompt names: the DQ
   id and failing check, the exact upstream file, the required behaviour, the
   test to add, the upstream's own test/lint command, and that the teammate
   commits and pushes upstream (nothing is merged back into rl-crypto). Wait for
   the teammate's message; do not poll `team_status`.
5. **To register a space in this NixOS-managed setup**, edit the authoritative
   source — `nixos-config` `modules/nixos/homeManager/config.nix`, the `spaces`
   block around line 290 — not the rendered symlink. Entries for these upstreams
   would be:
   ```nix
   tickerkit = {
     path = "/home/seanc/Projects/tickerkit";
     agent = "build";
     description = "Market data / OHLCV upstream for rl-crypto";
     flakeInput = "tickerkit";          # optional: re-pin after a clean push
   };
   ticker-news-signals = {
     path = "/home/seanc/Projects/ticker-news-signals";
     agent = "build";
     description = "News-sentiment upstream for rl-crypto";
     flakeInput = "ticker-news-signals";
   };
   ```
   Both sibling checkouts are git repos with `origin` set to their GitHub
   remotes, so they pass the plugin's `.git` validation. Rebuild/restart
   opencode; the next `team_spawn(space="tickerkit")` clones or fast-forwards
   under `~/.config/opencode/ensemble-spaces/`.
   A **project-local** `.opencode/ensemble.json` is not a merge: the plugin does
   a shallow `{...global, ...project}` spread, so a project `spaces` object
   replaces the global one wholesale. If you use one, repeat the existing
   spaces or they disappear from the registered list. Prefer the NixOS source.
6. **After the upstream fix lands and is pushed**, refresh the pin here:
   `nix flake lock --update-input tickerkit` (or the tns input; `uv lock` covers
   the path deps), then re-run the audit and record the upstream commit SHA in
   the finding. Shut the teammate down and clean up the team; the space's work
   is already in its own repo, so there is nothing to merge into rl-crypto.
7. **Only if the user explicitly declines the space** may the lead work directly
   in the sibling checkout. Record that exception in the audit log and keep the
   change minimal.

### 3.8 Update this file (the self-expansion step)

Do this **before** the final summary. The skill is the memory; an audit that
does not write back is wasted work.

1. **Baseline block** — update its numbers and timestamp in place. Never append
   a second baseline.
2. **New finding** — append under `Memory — active findings` using the template
   below. One entry per root cause, not per ticker.
3. **Existing finding** — update the `Evidence` line's date and counts in
   place. Do not duplicate it.
4. **Status change** — append a dated status line inside the entry. When it is
   fixed or proven expected, move the entry to `Memory — resolved & expected`
   as a one-liner (with how it was verified) and update the map.
5. **Mapping** — add rows for new actions; change `Status` for existing rows.
6. **Open questions** — add what you could not prove; remove what you answered.
7. **Audit log** — append exactly one row for this run.
8. **Long-form report** — only for a substantial audit, append a dated section
   to `data/AUDIT.md` and link it from the finding. Small audits stay here.

Bounds, so the file does not grow without limit:

- Active findings: **≤ 12**. Merge related ones or resolve them.
- Audit log: **≤ 20 rows**. Collapse the oldest rows into a one-line summary.
- Each finding: **≤ 5 lines** after the heading.
- No narrative, no duplicated tables, no store paths, no undated numbers.

Finding template:

```markdown
### DQ-0NN · <short claim> · <OPEN|MONITORING|FIXED|EXPECTED> (since YYYY-MM-DD)
- **Layer:** repo | tickerkit | tns | data | market | design
- **Evidence (YYYY-MM-DD):** <number + command/section that produced it>
- **Read:** <what it means; the condition that would make it a real fault>
- **Map:** IM-N, IM-M
- **Status:** <last change + date>
```

### 3.9 Verify and commit

- Re-run `tools/audit_data_quality.py`; confirm no regression and that the
  memory numbers match the tool.
- If code or the tool changed:
  `nix develop --command bash -c 'uv run pytest -m "not live"'` and
  `nix develop --command bash -c 'ruff check src/ tests/ examples/ tools/'`.
- `git diff --stat`; commit the skill/memory change separately, e.g.
  `docs(skill): <what the audit learned>`. Never commit `data/` changes from an
  audit-only run.
- An upstream fix lives in the upstream repo. Here it is only a re-pinned
  `flake.lock`/`uv.lock` plus a memory entry; do not copy upstream source into
  rl-crypto.
- Report the verdict, the P0/P1 exceptions, and what changed in the ledger.

## Frequency and scope

- **Quick audit** (default): baseline + archive table + live probe + update log.
- **Full audit**: all of the above, per-ticker details, model/summary checks,
  and a fresh `data/AUDIT.md` section when a finding is P0/P1 or a new upstream
  defect.
- **Ticker-scoped audit**: `--ticker <slug>`; still update the shared ledger but
  do not touch other tickers' findings.

## Memory — baseline

<!-- Update in place. Numbers are from the audit tool + probe, not estimates. -->

**As of 2026-10-23T22:26Z** (`HEAD` `213d17b`; newest `data/` commit `5724fa0`,
2026-10-23T21:06Z):

| Metric | Value |
| --- | --- |
| Tracked tickers | 15 |
| Archive bars | 25,810 (1,712-1,729 per ticker) |
| Live records | 3,277 (198-245 per ticker) |
| Archive faults | 0 gaps, 0 duplicates, 0 OHLC violations, 0 null/non-finite prices, 0 zero-volume bars |
| >5% hourly jumps | 69; prior audit found them clustered on the same bars across assets — re-verify clusters |
| `stale_h` | 1.4 (hourly archive, 30-min live cadence) |
| Records with `market.errors >= 1` | 2,210 / 3,277 (67%) |
| Records with `news.errors >= 1` | 2,244 / 3,277 (68%) |
| All-null market aggregates | 90, all dated 2026-10-12/13 (last `2026-10-13T23:14:10Z`); 0 since |
| `priceDev` > 1% | 674 (expected — 30-min live vs hourly bar) |
| Summaries | 15 × `rl-crypto/pipeline/3`, feature version `rl-crypto/features/2` |
| Targets beating baseline | `fwd_volume_ratio` 15/15, `fwd_volatility` 10/15, directional 0/15 |

Verdict: **the archive is sound; live error counters are noisy but no current
data loss.** The 2026-10-12 loss window is closed.

## Memory — upstream state

| Upstream | Local HEAD (2026-10-23) | Relevant state | Re-check |
| --- | --- | --- | --- |
| `tickerkit` | `c24d4f4` | Fix landed in `3196b24` (retrying 402 classified as throttle, keyless CoinGecko eased), but issues #1/#2 are **still OPEN on GitHub** (2026-10-23); residual `errors=1` is DQ-002 | `git -C ../tickerkit log -3 --oneline`; `gh issue list --repo Cairnstew/tickerkit` |
| `ticker-news-signals` | `23bf760` | Issue #1 (`rss:cryptoslate` 429) still open in the 2026-10-13 follow-up; no commits since 2026-10-10 | `git -C ../ticker-news-signals log -3 --oneline`; `gh issue list --repo Cairnstew/ticker-news-signals` |
| Ensemble spaces | — | **No** `tickerkit` or `ticker-news-signals` space registered as of 2026-10-23; §3.7.1 covers registering one | read `spaces` in `~/.config/opencode/ensemble.json` |

## Memory — active findings

### DQ-001 · `market.ok` contradicts `market.errors` · OPEN (since 2026-10-12)
- **Layer:** repo (`src/rl_crypto/unify.py:116`, `src/rl_crypto/models.py:117`)
- **Evidence (2026-10-23):** 2,210/3,277 records (67%) have `market.ok: true` with `market.errors >= 1`; `market_ok` is set on any non-raising fetch, while tickerkit's `DataQuality.ok` is `not errors`.
- **Read:** `ok` cannot distinguish a clean record from a noisy-but-complete one; consumers reading it as "clean" are misled. The fields are complete, so this is P3/P2, not data loss.
- **Map:** IM-2 (semantics), IM-1 (persist detail)
- **Status:** open; changing `ok` is a breaking meaning change, so prefer an additive `market_clean` unless consumers are audited.

### DQ-002 · Residual market errors are noise; data loss is closed · MONITORING (since 2026-10-13)
- **Layer:** upstream `tickerkit` (CoinGecko `meta` suppression) / committed history
- **Evidence (2026-10-23):** errors distribution `0:1067, 1:2035, 2:14, 3:24, 4:133, 5:2, 6:1, 7:1`; all-null aggregates 90 (70 on 2026-10-12, 20 on 2026-10-13), none after 2026-10-13T23:14:10Z.
- **Read:** a count of 1 is residual provider suppression, not lost data. The alarm condition is **all-null aggregates returning**, not the count.
- **Map:** IM-1, IM-3
- **Status:** monitor; `tickerkit` fixed the loss in `3196b24` (2026-10-13, refs #1/#2; issues still OPEN on GitHub).

### DQ-003 · News errors are high; this repo discards the attribution · OPEN (since 2026-10-12)
- **Layer:** upstream `ticker-news-signals` (the 429) + repo record shape (detail discarded)
- **Evidence (2026-10-23):** 2,244/3,277 records (68%) have `news.errors >= 1`; `unify.py:144` persists only `len(...)`, although tns already exposes `SourceErrorRecord(source, capability, exception_type, message)` (`../ticker-news-signals/ticker_news_signals/models.py:278`). 2026-10-13 follow-up: `rss:cryptoslate` 429 on 3 of 4 tickers; tns issue #1 still open then.
- **Read:** news sentiment is present on most records; the counter is mostly one feed. The 429 is upstream; the inability to name the feed from `data/` is repo-side. Check whether the upstream fix landed before calling the 429 open.
- **Map:** IM-1 (persist detail), IM-4 (upstream 429)
- **Status:** open; upstream issue `Cairnstew/ticker-news-signals#1`.

## Memory — resolved & expected

- **R-001 · tickerkit keyless 402 / CoinGecko throttling (#1, #2)** — code fixed upstream 2026-10-13 in `3196b24` (issues still OPEN on GitHub); all-null aggregates stop after 2026-10-13T23:14:10Z. The error-counter noise remains (DQ-002). Do not key-gate or drop a provider on the old report's say-so.
- **R-002 · `signal_hours` "off-by-one"** — not a bug: a 1-hour lookback crosses two hourly buckets; `signal_hours=1` occurs within ~15 min of the hour (from `data/AUDIT.md`).
- **R-003 · `has_news=0` in archived features** — by design: news history is not ingested. Live `news.articles` is populated; the trainer drops the constant news features (`models/README.md`).
  - **Measured 2026-10-24 — backfill is infeasible, not merely unbuilt.** tns v0.2 ships a Wayback history provider; probing it (`fetch_history`, BTC/USD) gave: hour granularity `2026-10-17..19` exceeded 400 s (killed), day granularity `2026-10-16..19` completed in 201 s with 3 vectors / 206 articles / 0 errors, and `2026-10-21..23` returned 0 vectors because Wayback captures stop at `2026-10-19` (≈5-day lag). At hour granularity 72 days × 15 tickers is out of reach on the production path; do not re-propose a news backfill without a different history source.
- **R-004 · Per-ticker bar count differences** — fetch-window start differs by up to a day (ETH starts 2026-08-12, the rest 2026-08-13); 1,712-1,729 bars is normal, not a gap.
- **R-005 · `priceDev`** — 30-minute live prices compared with the nearest hourly bar; >1% is ordinary intra-hour movement. Not a join fault (DQ-006).
- **R-006 · `stale_h` 1-2 h** — hourly archive + 30-minute live + even-hour training; only >2.5 h or a `data/` commit older than 75 min is an incident.

### DQ-004 · Withheld directional signal is by design · EXPECTED (since 2026-10-13)
- **Layer:** design (`src/rl_crypto/ml/tier2.py`, `models/README.md`)
- **Evidence (2026-10-23):** `fwd_return` 0/15 and `fwd_direction` 0/15 beat baseline; `fwd_volatility` 10/15 and `fwd_volume_ratio` 15/15 do; null `signal`/`confidence` accompanies an explicit note.
- **Read:** null `signal`/`confidence` is **not** missing data — Tier 2 withholds a signal no target earned. Point predictions and `p(up)` are still emitted.
- **Map:** none
- **Status:** expected; re-derive the counts, never copy them.

### DQ-005 · The archive is clean · EXPECTED (since 2026-10-23)
- **Layer:** data (`data/<slug>/candles.jsonl`)
- **Evidence (2026-10-23):** 25,810 bars across 15 tickers; 0 gaps, 0 duplicate/non-monotonic stamps, 0 OHLC violations, 0 null/non-finite/non-positive prices, 0 zero-volume bars; 69 >5% jumps (prior audit found the large moves cluster on the same bars across assets — re-verify, do not assume).
- **Read:** this is the pass condition, not an absence of checks. A continuity or OHLC violation is a **new P1**; per-ticker bar-count differences and cross-asset jumps are not.
- **Map:** IM-3 (keep it as a regression check)
- **Status:** expected; re-derive every run.

### DQ-006 · `priceDev` is expected, not a join fault · EXPECTED (since 2026-10-23)
- **Layer:** measurement method (`tools/audit_data_quality.py:cross_check`)
- **Evidence (2026-10-23):** 674 records >1% from the nearest bar; deviations are isolated in time/ticker (e.g. BTC 7, ADA 51), not a same-timestamp cross-asset event.
- **Read:** a 30-min live snapshot will differ from an hourly bar by intra-hour movement. Investigate only if >5% with no matching bar jump, or if the same timestamp moves every ticker in the same direction.
- **Map:** IM-3 (make the metric volatility-aware) — proposal only.
- **Status:** expected; do not report as P1.

### DQ-007 · `stale_h` reflects cadence · EXPECTED (since 2026-10-23)
- **Layer:** scheduler/cadence
- **Evidence (2026-10-23):** 1.4 h on all 15 tickers at 22:26Z; newest `data/` commit 21:06Z.
- **Read:** the archive is hourly and trains on even UTC hours. Check the commit age against the 75-minute watchdog threshold before calling it stale.
- **Map:** IM-6 (make commit-age explicit in the audit)
- **Status:** expected.

## Memory — improvement map

| ID | Finding | Layer | Action | Target | Risk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| IM-1 | DQ-002, DQ-003 | repo | Persist upstream detail (tickerkit `ProviderError`, tns `SourceErrorRecord`) as an additive field, not only the count — **the detail already exists upstream** | `src/rl_crypto/unify.py:126,144`; `src/rl_crypto/models.py:125,136`; schema bump | record schema; consumers | proposed |
| IM-2 | DQ-001 | repo | Make `ok` mean "no errors" or add `market_clean`/`news_clean` | `src/rl_crypto/unify.py:116`; `src/rl_crypto/models.py:117` | breaking meaning; audit consumers first | proposed |
| IM-3 | DQ-002, DQ-006 | repo | Add regressions: `ok && errors`, null-aggregate date range, priceDev volatility-aware, summary vs candles, bundle feature version | `tools/audit_data_quality.py`; new `tests/test_audit_data_quality.py` | low; test-only | proposed |
| IM-4 | DQ-003 | upstream | Fix/limit the `rss:cryptoslate` 429 (attribution already exists; IM-1 persists it) | `Cairnstew/ticker-news-signals#1` | upstream scope | upstream |
| IM-5 | all P1 | policy | Data-remediation protocol: only provable corruption, dated plan + backup + test; archive stays append-only | `data/AUDIT.md`; this skill | irreversible edits if skipped | policy |
| IM-6 | DQ-007 | repo/skill | Check newest `data/` commit age against 75 min in the audit (shell step; not inside the offline tool) | §3.1 + audit log | none | proposed |

## Memory — audit log

| Run (UTC) | Tickers | Bars | Live | Archive | `market.errors>=1` | `news.errors>=1` | All-null agg | `priceDev` | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-23T22:26Z | 15 | 25,810 | 3,277 | clean | 2,210 (67%) | 2,244 (68%) | 90 (≤ 2026-10-13) | 674 (expected) | archive sound; data loss closed; counter noise unchanged |

## Memory — open questions

- **Q-1** `live.jsonl` is capped at 1000; how far back can a trend question see
  before it must read git history? Confirm the cap in the running code, not just
  the README.
- **Q-2** Are records with `market.errors == 1` ever all-null now? The current
  answer is no after 2026-10-13; make that a standing check (IM-3).
- **Q-3** `data/AUDIT.md` counted 70 all-null records before the fix; this audit
  counts 90 including 2026-10-13. Split the count by date before citing either
  number.
- **Q-4** Should `priceDev` be normalized by realised volatility or bar age
  instead of a flat >1%? Proposal only (IM-3).
- **Q-5** Does `summary.json` `archive.rows` ever differ from the candle line
  count after a rebase/merge? Confirm on the next `train` run.
- **Q-6** Should `tickerkit` and `ticker-news-signals` be registered as Ensemble
  spaces in the NixOS config? Today neither is, so the upstream-change workflow
  (§3.7.1) stops at "ask the user".
