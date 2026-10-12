#!/usr/bin/env bash
# Frequent live recorder used by the local systemd timer (and manually).
#
# Syncs the repo, records one live point (market + live news + outlook) for
# EVERY tracked ticker via tools/record_live.py, and commits/pushes the data
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

nix develop -c uv run python tools/record_live.py \
  "${TICKER_ARGS[@]}" \
  --live-cap "${LIVE_CAP:-1000}"

git add data
if git diff --cached --quiet; then
  echo "no live cache change to commit"
  exit 0
fi
git commit -m "live: record $(date -u +%Y-%m-%dT%H:%MZ)"

# A full watchlist run takes minutes, so the 2-hourly Actions pipeline can land a
# cache commit while this run is mid-flight. Rebase our data-only commit on top
# and retry once; the next tick would recover anyway, but retrying here keeps the
# live series dense.
if ! git push origin HEAD:main; then
  echo "push rejected (concurrent cache commit?) — rebasing and retrying"
  git pull --rebase -q origin main
  git push origin HEAD:main
fi