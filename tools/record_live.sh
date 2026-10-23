#!/usr/bin/env bash
# Frequent live recorder used by the local systemd timer (and manually).
#
# Syncs the repo, records one live point (market + live news + outlook) for
# EVERY tracked ticker via `rl-pipeline live`, and commits/pushes the data
# cache if anything changed. The watchlist lives in src/rl_crypto/tickers.py,
# so this script deliberately does not hardcode a ticker: no stagger, every
# ticker on every 30-minute tick.
#
# The frequent live path lives here because GitHub Actions' schedule delivery
# proved unreliable for a second cron workflow on this young repo; the Actions
# pipeline (2-hourly) still owns the history archive + retraining.
set -euo pipefail

export PATH="/etc/profiles/per-user/seanc/bin:$PATH"

cd "$(dirname "$0")/.."

# Override the watchlist for a one-off run: LIVE_TICKERS="BTC/USD ETH/USD"
# ./tools/record_live.sh  (default = the tracked set in the runner)
TICKER_ARGS=()
if [[ -n "${LIVE_TICKERS:-}" ]]; then
  # shellcheck disable=SC2206 # deliberate word-splitting into separate args
  TICKER_ARGS=(--tickers ${LIVE_TICKERS})
fi

# Sync with the last pipeline/timer commit before recording (best-effort).
git pull --ff-only -q origin main || true

# Never record from a detached HEAD. Committing off a detached head and pushing
# `HEAD:main` looks fine, but a concurrent-update retry below then runs
# `git pull --rebase` in that state, which can strand a half-finished rebase in
# .git and silently block every later branch operation. Reattach first.
if ! git symbolic-ref -q HEAD >/dev/null; then
  echo "detached HEAD detected — reattaching to main"
  git switch main || git switch -C main origin/main
  git merge --ff-only -q origin/main || true
fi

# A rebase retry below stops on commits from before the rl-pipeline move, so the
# working tree at that moment may not contain the resolver at all (and any older
# copy it does contain predates the summary/candle fixes). Capture the current
# tip's resolver now, before a rebase can move the checkout, and run that copy.
RESOLVER="$(mktemp --suffix=.py)"
trap 'rm -f "$RESOLVER"' EXIT
if ! git show HEAD:src/rl_crypto/pipeline/conflicts.py > "$RESOLVER" 2>/dev/null; then
  RESOLVER="$PWD/tools/resolve_cache_conflicts.py"
fi

nix develop -c uv run rl-pipeline live \
  "${TICKER_ARGS[@]}" \
  --live-cap "${LIVE_CAP:-1000}"

git add data
if git diff --cached --quiet; then
  echo "no live cache change to commit"
  exit 0
fi
git commit -m "live: record $(date -u +%Y-%m-%dT%H:%MZ)"

# A full watchlist run takes minutes, so the 2-hourly Actions pipeline can land a
# cache commit while this run is mid-flight. Both writers APPEND to
# data/<slug>/live.jsonl, so rebasing conflicts on those tails — and `set -e`
# would abort the run, stranding the rebase and losing a whole tick.
#
# Resolution is a merge, not a side: live.jsonl is append-only, so every record
# collected by either writer is kept, and summary.json keeps the newer local run.
# `rl-pipeline conflicts` does that (candles merge by timestamp, never duplicating).
if ! git push origin HEAD:main; then
  echo "push rejected (concurrent cache commit?) — rebasing and retrying"
  if ! git pull --rebase -q origin main; then
    echo "rebase conflicted — resolving data/ conflicts"
    # Run the captured copy (see RESOLVER above), never a path in the tree: the
    # conflicted checkout can be an old commit with no rl-pipeline entry point.
    if ! nix develop -c uv run python "$RESOLVER" --root "$PWD"; then
      echo "conflict resolution failed — leaving the rebase in place to inspect" >&2
      exit 1
    fi
    GIT_EDITOR=true git rebase --continue
  fi
  git push origin HEAD:main
fi