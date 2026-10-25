from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NoiseSpec:
    kind: str
    df: float | None = None
    garch: bool = False
    rho: float = 0.0


def generate_ar1_innovations(
    rng: np.random.Generator,
    shape: tuple[int, int],
    noise: NoiseSpec,
    rho: float,
    garch_params: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Generate AR(1) series with different innovation types and optional GARCH clustering."""
    sims, n_rows = shape
    if garch_params is None:
        omega, alpha_g, beta_g = 0.01, 0.1, 0.85
    else:
        omega, alpha_g, beta_g = garch_params
    series = np.zeros((sims, n_rows), dtype=float)
    if not noise.garch:
        if noise.kind == "normal":
            innovations = rng.standard_normal((sims, n_rows))
        elif noise.kind == "t":
            df = float(noise.df or 5.0)
            innovations = rng.standard_t(df, size=(sims, n_rows))
        else:
            innovations = rng.standard_normal((sims, n_rows))
        variance = np.sqrt(max(0.0, 1.0 - rho * rho))
        for index in range(n_rows):
            previous = series[:, index - 1] if index else 0.0
            series[:, index] = rho * previous + variance * innovations[:, index]
        return series
    # GARCH(1,1) style conditional volatility clustering over rows
    for s in range(sims):
        sig2 = omega / max(1e-12, (1.0 - alpha_g - beta_g)) if (alpha_g + beta_g) < 1.0 else omega
        for index in range(n_rows):
            if noise.kind == "t":
                df = float(noise.df or 5.0)
                z = rng.standard_t(df)
            else:
                z = rng.standard_normal()
            # scale t to have unit variance approximately
            if noise.kind == "t" and df > 2.0:
                scale = np.sqrt((df - 2.0) / df)
                z = z * scale
            eps = np.sqrt(max(1e-18, sig2)) * z
            previous = series[s, index - 1] if index else 0.0
            # AR(1) component with GARCH innovations
            series[s, index] = rho * previous + eps
            sig2 = omega + alpha_g * (eps * eps) + beta_g * sig2
    return series
