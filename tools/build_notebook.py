#!/usr/bin/env python3
"""Generate ``notebooks/unified.ipynb`` from an in-repo template.

A generated notebook stays in sync with the code instead of drifting: rerun with

    uv run python tools/build_notebook.py

and commit the result. The cells are ordinary markdown/code strings so a diff
reads like the notebook it produces.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "unified.ipynb"

MD_INTRO = """\
# rl-crypto — the committed cache

Read the **git-committed cache** the scheduled runners write: the raw candle
archive, the live unified-record stream, and the run summary under
``data/<slug>/``, plus the trained Tier-1 bundle under ``models/``.

The first cell syncs the checkout with the tracking remote (fetch plus a
``--ff-only`` pull), so what you read is the remote's latest committed cache —
the runners push every couple of hours. Every other cell is **offline**. The
live fetching that produced these files lives in ``rl-pipeline run``
(archive + train) and ``rl-pipeline live`` (frequent live points); this
notebook consumes their output the way any downstream analysis would.
"""

MD_SYNC = """\
## Sync with the remote

The cache is only as fresh as the checkout. This cell fetches the tracking
remote and fast-forwards when it can; when local changes or local commits are
in the way, ``--ff-only`` **refuses** rather than overwrite, and the reason is
printed (offline does the same). Every later cell then reads the newest
committed cache — or, when a sync was impossible, the files on disk with the
reason stated.
"""

MD_ARCHIVE = """\
## The raw candle archive

``data/<slug>/candles.jsonl`` is the **only-growing** market archive: candles
are upserted by timestamp, so the history never shrinks and never duplicates.
Tier-1 training reads the whole archive every run and recomputes the
forward-shifted targets on the fly, so a bar's labels only become defined once
its future is archived.
"""

CODE_ROOT = """\
import json
from pathlib import Path

from rl_crypto import __version__, build_samples
from rl_crypto.cache import (
    candles_path,
    live_path,
    load_candles,
    read_jsonl,
    summary_path,
    sync_cache,
)
from rl_crypto.ml import load_bundle, train_tier1
from rl_crypto.ml.serve import outlook_for_rows
from rl_crypto.ml.store import default_bundle_path


def find_root() -> Path:
    \"\"\"The repo root — the nearest ancestor holding the committed cache.\"\"\"
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "data").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise SystemExit(f"no data/ cache found above {Path.cwd()}")


ROOT = find_root()
TICKER = "ETH/USD"

print("rl-crypto", __version__, "| cache root:", ROOT)
"""

CODE_SYNC = """\
sync = sync_cache(ROOT)

print(sync.summary())
for note in sync.notes:
    print(" -", note)
"""

CODE_ARCHIVE = """\
rows = load_candles(candles_path(ROOT, TICKER), ticker=TICKER)

print(TICKER, "archive:", len(rows), "candles")
print("span:", rows[0].timestamp, "->", rows[-1].timestamp)
print(f"first: close={rows[0].close} volume={rows[0].volume}")
print(f"last:  close={rows[-1].close} volume={rows[-1].volume}")
"""

MD_LIVE = """\
## The live record stream

``data/<slug>/live.jsonl`` is the **capped, append-only** stream of live
unified records: one market + live-news snapshot per run, each carrying the
Tier-2 outlook attached at write time. News history is intentionally absent
from the archive, so archive rows are news-free (``has_news=False``) while
these live records still carry sentiment.
"""

CODE_LIVE = """\
live = read_jsonl(live_path(ROOT, TICKER))
newest = live[-1]

print("live records:", len(live), "| newest generated at:", newest["generated_at"])
print("market:", newest["market"]["price"], "| news:", newest["news"]["latest_sentiment"])

# A record may carry no outlook at all (older ones were stripped, and a
# retrain-only run writes none), and confidence is None when no model has
# earned a signal. Both are honest states, so report them rather than crash.
pred = newest.get("predictions") or {}
conf = pred.get("confidence")
print("attached outlook: signal", pred.get("signal"),
      "| confidence", "n/a" if conf is None else round(conf, 4))
print("notes:", newest["notes"])
"""

MD_SUMMARY = """\
## The run summary

``data/<slug>/summary.json`` is overwritten every pipeline run: the archive
span, the fitted per-target metrics, and the outlook that was attached to the
live record at the time.
"""

CODE_SUMMARY = """\
summary = json.loads(summary_path(ROOT, TICKER).read_text(encoding="utf-8"))

print("generated:", summary["generated_at"], "| archive rows:", summary["archive"]["rows"])
print("samples:", summary["samples"])
for name, metrics in summary["targets"].items():
    print(f"  {name}: {metrics}")
"""

MD_ML = """\
## Tiered ML over the archive

Tier 0 builds **causal** features (no future information), Tier 1 fits one
model per forward target (return/direction/volatility/volume), Tier 2 fuses
them into an ``Outlook``. Train a fresh bundle over the **whole archive**, then
serve with the **committed** bundle from ``models/<slug>.json`` — the artifact
the pipeline commits and the live recorder loads.
"""

CODE_ML_TRAIN = """\
samples = build_samples(rows, horizon=1)
bundle = train_tier1(samples, horizon=1)

print("fresh bundle over the archive:", bundle.n_samples, "complete samples")
for name, model in bundle.targets.items():
    print(f"  {name}: {model.metrics}")
"""

CODE_ML_OUTLOOK = """\
committed = load_bundle(default_bundle_path(TICKER, directory=ROOT / "models"))
outlook = outlook_for_rows(rows, committed, ticker=TICKER)

# The gate withholds signal/confidence unless the backing model beat its
# baseline out of sample, so both are legitimately None here.
print("committed bundle -> feature timestamp:", outlook.feature_timestamp)
print("signal:", outlook.signal, "| confidence:",
      "n/a" if outlook.confidence is None else round(outlook.confidence, 4))
print(outlook.predictions)
for note in outlook.notes:
    print(" -", note)
"""


def build() -> nbformat.NotebookNode:
    notebook = new_notebook()
    notebook.cells = [
        new_markdown_cell(MD_INTRO),
        new_code_cell(CODE_ROOT),
        new_markdown_cell(MD_SYNC),
        new_code_cell(CODE_SYNC),
        new_markdown_cell(MD_ARCHIVE),
        new_code_cell(CODE_ARCHIVE),
        new_markdown_cell(MD_LIVE),
        new_code_cell(CODE_LIVE),
        new_markdown_cell(MD_SUMMARY),
        new_code_cell(CODE_SUMMARY),
        new_markdown_cell(MD_ML),
        new_code_cell(CODE_ML_TRAIN),
        new_code_cell(CODE_ML_OUTLOOK),
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "rl-crypto",
            "language": "python",
            "name": "rl-crypto",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return notebook


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build(), OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
