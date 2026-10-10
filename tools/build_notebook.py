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
# rl-crypto — unified dataset

Fuse **tickerkit** (market) with **ticker-news-signals** (news sentiment) for a
single ticker. Both sides are partial-failure tolerant: if one upstream is
unreachable its fields stay `None` and the failure lands in `notes`.
"""

MD_UNIFIED = "## One ticker, both pipelines\n\n`build_unified` returns a point-in-time record."

MD_DATASET = """\
## Hourly dataset

`build_dataset` joins tickerkit's OHLCV history with ticker-news-signals'
Wayback-reconstructed sentiment vectors on the UTC interval. Rows are the
**union** of both sides — an interval known to only one side is kept and
flagged (`has_price` / `has_news`).
"""

CODE_INTRO = """\
from rl_crypto import build_unified, build_dataset, __version__

print("rl-crypto", __version__)
"""

CODE_UNIFIED = """\
record = build_unified("ETH/USD", lookback_hours=3)
print(record.price, record.latest_sentiment)
print(record.to_dict()["notes"])
"""

CODE_DATASET = """\
dataset = build_dataset("ETH/USD", start="2026-01-01", end="2026-01-03", timeframe="1h")
print(dataset.to_json(indent=2)[:800])
"""


def build() -> nbformat.NotebookNode:
    notebook = new_notebook()
    notebook.cells = [
        new_markdown_cell(MD_INTRO),
        new_code_cell(CODE_INTRO),
        new_markdown_cell(MD_UNIFIED),
        new_code_cell(CODE_UNIFIED),
        new_markdown_cell(MD_DATASET),
        new_code_cell(CODE_DATASET),
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
