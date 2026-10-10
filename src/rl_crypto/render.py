"""Human- and pipeline-facing formatters for the unified records.

Two audiences, same contract as the upstreams:

* **Humans** — aligned plain-text blocks (no colour, no pandas) that paste
  cleanly into a terminal or an issue.
* **Pipelines** — :func:`unified_flat_metrics` returns one flat ``dict`` of
  scalars per record, keyed the same way the dataset columns are, so it drops
  straight into a DataFrame.

A missing measurement renders as ``-`` and stays ``None`` in the flat metrics —
never ``0``.
"""

from __future__ import annotations

from typing import Any

from .models import Dataset, UnifiedRecord

NA = "-"


def _num(value: Any, fmt: str = ".6g") -> str:
    return NA if value is None else format(value, fmt)


def _pct(value: float | None) -> str:
    return NA if value is None else f"{value:+.2f}%"


def render_unified(record: UnifiedRecord) -> str:
    """A short human-readable block for one :class:`UnifiedRecord`."""
    title = record.resolved_name or record.resolved_symbol or record.ticker
    lines = [
        f"{title} ({record.ticker})",
        f"  generated {record.generated_at or NA}",
        "",
        "  market (tickerkit)"
        + ("" if record.market_ok else "  [unavailable]"),
        f"    price        {_num(record.price)}",
        f"    consensus    {_num(record.consensus_price)}"
        f"   aggregator {_num(record.aggregated_price)}",
        f"    change 24h   {_pct(record.change_24h_pct)}",
        f"    volume 24h   {_num(record.volume_24h)}",
        f"    market cap   {_num(record.market_cap)}",
        f"    venues       {record.venue_count}"
        + (f"   errors {record.market_errors}" if record.market_errors else ""),
        "",
        "  news (ticker-news-signals)"
        + ("" if record.news_ok else "  [unavailable]"),
        f"    latest sent. {_num(record.latest_sentiment)}"
        f"  @ {record.latest_sentiment_at or NA}",
        f"    mean sent.   {_num(record.mean_sentiment)}"
        f"   over {record.lookback_hours}h",
        f"    hours/arts   {record.signal_hours}h / {record.articles} articles",
        f"    novelty      {'yes' if record.novelty_flag else 'no'}",
    ]
    if record.notes:
        lines.append("")
        lines.append("  notes")
        lines.extend(f"    - {note}" for note in record.notes)
    return "\n".join(lines)


def render_unified_markdown(record: UnifiedRecord) -> str:
    """The unified record as a fenced markdown block."""
    return "```text\n" + render_unified(record) + "\n```"


def unified_flat_metrics(record: UnifiedRecord) -> dict[str, Any]:
    """One flat row of scalars, keyed like the dataset columns."""
    return {
        "ticker": record.ticker,
        "resolved_symbol": record.resolved_symbol,
        "resolved_name": record.resolved_name,
        "generated_at": record.generated_at,
        "price": record.price,
        "consensus_price": record.consensus_price,
        "aggregated_price": record.aggregated_price,
        "change_24h_pct": record.change_24h_pct,
        "volume_24h": record.volume_24h,
        "market_cap": record.market_cap,
        "venue_count": record.venue_count,
        "market_ok": record.market_ok,
        "market_errors": record.market_errors,
        "latest_sentiment": record.latest_sentiment,
        "latest_sentiment_at": record.latest_sentiment_at,
        "mean_sentiment": record.mean_sentiment,
        "articles": record.articles,
        "signal_hours": record.signal_hours,
        "novelty_flag": record.novelty_flag,
        "news_ok": record.news_ok,
        "news_errors": record.news_errors,
    }


def dataset_flat_metrics(dataset: Dataset) -> dict[str, Any]:
    """Coverage scalars for a whole :class:`Dataset` (one row per dataset)."""
    return {
        "ticker": dataset.ticker,
        "interval": dataset.interval,
        "start": dataset.start,
        "end": dataset.end,
        "generated_at": dataset.generated_at,
        "rows": len(dataset.rows),
        "price_rows": dataset.price_rows,
        "news_rows": dataset.news_rows,
        "joined_rows": dataset.joined_rows,
        "notes": dataset.notes,
    }


def render_dataset(dataset: Dataset, *, max_rows: int = 24) -> str:
    """A coverage summary plus a preview of the joined rows."""
    lines = [
        f"{dataset.ticker} — {dataset.interval} dataset",
        f"  window     {dataset.start or NA} .. {dataset.end or NA}",
        f"  rows       {len(dataset.rows)}"
        f"   price {dataset.price_rows}"
        f"   news {dataset.news_rows}"
        f"   joined {dataset.joined_rows}",
    ]
    if dataset.notes:
        lines.append("  notes")
        lines.extend(f"    - {note}" for note in dataset.notes)
    lines.append("")
    header = f"  {'timestamp':<20} {'close':>12} {'sent':>8} {'n':>4} {'nov':>4}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for row in dataset.rows[:max_rows]:
        lines.append(
            f"  {row.timestamp:<20} {_num(row.close, '.8g'):>12}"
            f" {_num(row.sentiment_score, '+.4f'):>8}"
            f" {row.article_count:>4}"
            f" {'yes' if row.novelty_flag else '':>4}"
        )
    if len(dataset.rows) > max_rows:
        lines.append(f"  … {len(dataset.rows) - max_rows} more rows")
    return "\n".join(lines)


__all__ = [
    "dataset_flat_metrics",
    "render_dataset",
    "render_unified",
    "render_unified_markdown",
    "unified_flat_metrics",
]
