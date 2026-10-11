#!/usr/bin/env bash
# Frequent live recorder used by the local systemd timer (and manually).
#
# Syncs the repo, records one live point (market + live news + outlook) via
# tools/record_live.py, and commits/pushes the data cache if anything changed.
# The frequent live path lives here because GitHub Actions' schedule delivery
# proved unreliable for a second cron workflow on this young repo; the Actions
# pipeline (2-hourly) still owns the history archive + retraining.
set -euo pipefail

export PATH="/etc/profiles/per-user/seanc/bin:$PATH"

cd "$(dirname "$0")/.."

# Sync with the last pipeline/timer commit before recording (best-effort).
git pull --ff-only -q origin main || true

nix develop -c uv run python tools/record_live.py \
  --ticker "${LIVE_TICKER:-ETH/USD}" --live-cap "${LIVE_CAP:-1000}"

git add data
if git diff --cached --quiet; then
  echo "no live cache change to commit"
  exit 0
fi
git commit -m "live: record $(date -u +%Y-%m-%dT%H:%MZ)"
git push origin HEAD:main