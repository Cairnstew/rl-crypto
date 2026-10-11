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

MD_ML = """\
## Tiered ML — features, targets, outlook

The fused dataset feeds the tiered pipeline. Tier 0 builds **causal** features
(no future information), Tier 1 fits one model per forward target
(return/direction/volatility/volume), Tier 2 fuses them into an `Outlook`.
This section runs **offline on synthetic rows**, so it is deterministic and
repeatable — swap `synthetic_rows()` for a real `build_dataset` call to train
on live history.
"""

CODE_ML_SETUP = """\
from datetime import UTC, datetime, timedelta
from math import exp, sin

from rl_crypto import build_samples
from rl_crypto.ml import train_tier1
from rl_crypto.ml.serve import outlook_for_rows
from rl_crypto.models import DatasetRow, to_iso


def synthetic_rows(n: int = 120) -> tuple[DatasetRow, ...]:
    rows: list[DatasetRow] = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(n):
        close = 100.0 * exp(0.01 * i + 0.04 * sin(i))
        rows.append(
            DatasetRow(
                ticker="ETH/USD",
                timestamp=to_iso(start + timedelta(hours=i)) or "",
                open=close * 0.999, high=close * 1.01, low=close * 0.99,
                volume=1000.0 + 50.0 * i, close=close,
                sentiment_score=0.1 + 0.02 * sin(i),
                article_count=(i % 3) + 1, novelty_flag=(i % 7 == 0),
                has_price=True, has_news=True,
            )
        )
    return tuple(rows)


print("synthetic rows:", len(synthetic_rows()))
"""

CODE_ML_TRAIN = """\
rows = synthetic_rows()
bundle = train_tier1(build_samples(rows, horizon=1), horizon=1)
print("fitted targets:", {name: model.metrics for name, model in bundle.targets.items()})
"""

CODE_ML_OUTLOOK = """\
outlook = outlook_for_rows(rows, bundle, ticker="ETH/USD")
print("signal:", outlook.signal, "| confidence:", outlook.confidence)
print(outlook.as_features())
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
        new_markdown_cell(MD_ML),
        new_code_cell(CODE_ML_SETUP),
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
