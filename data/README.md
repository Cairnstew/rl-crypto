# Committed pipeline cache

`data/` is the **bounded, git-committed cache** written by the scheduled
production run (`.github/workflows/pipeline.yml` → `tools/run_pipeline.py`) and
by hand with:

```bash
uv run python tools/run_pipeline.py --ticker ETH/USD --train-days 21
```

Per ticker, under `data/<slug>/` (e.g. `data/eth-usd/` for `ETH/USD`):

| file | contents | growth |
| ---- | -------- | ------ |
| `history.jsonl` | one flat feature+target row per interval of the training window | overwritten every run |
| `live.jsonl` | the live unified record + attached Tier-2 outlook per run | appended, capped at recent records |
| `summary.json` | run metadata, dataset coverage, model metrics, outlook, notes | overwritten every run |

The trained Tier-1 models land in [`models/`](../models/README.md) (also
overwritten per run). Nothing here needs to be edited by hand — `git log data/`
and `git log models/` are the machine-readable run history.