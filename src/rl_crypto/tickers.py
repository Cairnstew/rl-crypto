"""The tracked ticker set — the one place the pipeline's watchlist is defined.

Every scheduled run (the 2-hourly Actions pipeline, the 30-minute local live
timer, and the manual dispatch paths) records this same list, so a ticker is
added or retired in exactly one place and the ``data/<slug>/`` caches stay in
lockstep with it.

Membership is evidence-based: every ticker here was screened with
``tools/probe_tickers.py`` for a full 60-day 1h candle fill and a populated
hourly news-signal window. Stablecoins (``USDT/USD``, ``USDC/USD``) resolve and
fetch fine but are deliberately excluded — near-zero variance makes them poor
training targets.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Canonical watchlist, ordered roughly by liquidity and news volume.
TRACKED_TICKERS: tuple[str, ...] = (
    "ETH/USD",
    "BTC/USD",
    "SOL/USD",
    "XRP/USD",
    "BNB/USD",
    "ADA/USD",
    "AVAX/USD",
    "LINK/USD",
    "LTC/USD",
    "DOGE/USD",
)


def parse_tickers(values: Iterable[str] | str | None) -> tuple[str, ...]:
    """Normalise ticker arguments into a de-duplicated, order-preserving tuple.

    Accepts ``None`` (→ the tracked set), a single string that may be
    comma- or whitespace-separated (``"BTC/USD,ETH/USD"``, ``"BTC/USD ETH/USD"``),
    or any iterable of either. Blank entries are dropped and duplicates are
    folded, so a watchlist pasted twice costs one fetch.
    """
    if values is None:
        return TRACKED_TICKERS
    if isinstance(values, str):
        raw: list[str] = values.replace(",", " ").split()
    else:
        raw = [part for value in values for part in str(value).replace(",", " ").split()]

    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        ticker = item.strip()
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return tuple(out)


__all__ = ["TRACKED_TICKERS", "parse_tickers"]
