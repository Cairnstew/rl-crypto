"""Protected label helpers (fixture placeholder)."""

MIN_VOL_SPAN = 2


def label_purge(target: str, horizon: int) -> int:
    if target == "fwd_volatility":
        return max(horizon, MIN_VOL_SPAN)
    return max(0, horizon)
