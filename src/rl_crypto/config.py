"""Unified configuration.

rl-crypto owns almost no configuration of its own: the two upstreams each read
their environment (``TICKERKIT_*`` for tickerkit, ``TNS_*`` for
ticker-news-signals). This module only holds the knobs that are specific to
*fusing* the two — the default lookback and timeframe, the output format, and
the dataset interval floor — plus thin builders that hand each fetch its
upstream config unchanged.

Environment variables (all optional)::

    RL_CRYPTO_LOOKBACK_HOURS   default news lookback window   (default: 1)
    RL_CRYPTO_TIMEFRAME        default market history interval (default: 1d)
    RL_CRYPTO_GRANULARITY      news history bucket, hour|day   (default: hour)
    RL_CRYPTO_FORMAT           default dataset format jsonl|csv (default: jsonl)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class Config:
    """The fusion-level defaults only; per-fetch options come from the CLI."""

    lookback_hours: int = 1
    timeframe: str = "1d"
    granularity: str = "hour"
    format: str = "jsonl"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        env = os.environ if env is None else env
        return cls(
            lookback_hours=_int(env.get("RL_CRYPTO_LOOKBACK_HOURS"), 1),
            timeframe=env.get("RL_CRYPTO_TIMEFRAME") or "1d",
            granularity=env.get("RL_CRYPTO_GRANULARITY") or "hour",
            format=env.get("RL_CRYPTO_FORMAT") or "jsonl",
        )

    def with_overrides(self, **kw: Any) -> Config:
        clean = {k: v for k, v in kw.items() if v is not None}
        return replace(self, **clean)

    # -- upstream config builders -------------------------------------------------

    def market_config(self) -> Any:
        """A tickerkit ``Config`` from the environment (or ``None`` to default)."""
        from tickerkit import Config as MarketConfig

        return MarketConfig.from_env()

    def news_config(self) -> Any:
        """A ticker-news-signals ``Config`` from the environment."""
        from ticker_news_signals.config import Config as NewsConfig

        return NewsConfig.from_env()


def _int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


__all__ = ["Config"]
