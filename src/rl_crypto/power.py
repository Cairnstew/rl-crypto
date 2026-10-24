"""Power calculations and calibration for the measurement contract.

The verdict used to have only "did the CI clear zero?" as its question. That is
the wrong question for an incremental research programme: a design can have a
positive point estimate while being far too noisy to certify an effect worth
acting on. This module supplies the two numbers the lifecycle needs:

* **Minimum detectable effect (MDE)** — the smallest true paired improvement a
  design can distinguish from zero at the pre-registered alpha and power. It is
  estimated from the observed paired-difference series with a time-block
  bootstrap, so serial correlation is charged against the design instead of
  being laundered into extra independent rows.
* **Coverage / uniformity** — the calibration of the CI rule under a known zero
  mean with realistic block autocorrelation. A rule that is anti-conservative
  is a measurement bug, not a discovery.

Everything is deterministic given a seed and pure Python + numpy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np

#: Default power target for the MDE.
DEFAULT_POWER = 0.8


def _z(alpha: float, power: float, *, two_sided: bool = True) -> float:
    distribution = NormalDist()
    tail = float(alpha) / 2.0 if two_sided else float(alpha)
    return distribution.inv_cdf(1.0 - tail) + distribution.inv_cdf(float(power))


def block_bootstrap_means(
    values: Sequence[float],
    *,
    block_bars: int,
    draws: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap distribution of the mean, resampling whole time blocks."""
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return np.asarray([], dtype=float)
    width = max(1, int(block_bars))
    blocks = [data[start : start + width] for start in range(0, data.size, width)]
    blocks = [block for block in blocks if block.size]
    if not blocks:
        return np.asarray([], dtype=float)
    sums = np.asarray([float(np.sum(block)) for block in blocks], dtype=float)
    counts = np.asarray([float(block.size) for block in blocks], dtype=float)
    rng = np.random.default_rng(int(seed))
    picks = rng.integers(0, len(blocks), size=(int(draws), len(blocks)))
    return sums[picks].sum(axis=1) / counts[picks].sum(axis=1)


def minimum_detectable_effect(
    values: Sequence[float],
    *,
    alpha: float,
    power: float = DEFAULT_POWER,
    block_bars: int = 24,
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """The smallest true mean a paired design can detect at ``alpha``/``power``.

    ``values`` is the per-row paired improvement series (baseline loss minus
    model loss). Its block-bootstrap standard error is the design noise; the
    MDE is ``(z_{1-alpha/2} + z_power) * SE`` in the same metric units as the
    pooled improvement.
    """
    data = [float(value) for value in values]
    if len(data) < 2:
        return {
            "mde": None,
            "se": None,
            "sd": None,
            "n": len(data),
            "n_eff": 0,
            "alpha": float(alpha),
            "power": float(power),
            "block_bars": int(block_bars),
        }
    means = block_bootstrap_means(
        data, block_bars=block_bars, draws=draws, seed=seed
    )
    se = float(np.std(means, ddof=1)) if means.size > 1 else None
    sd = float(np.std(np.asarray(data, dtype=float), ddof=1))
    n_eff = max(1, math.ceil(len(data) / max(1, int(block_bars))))
    multiplier = _z(alpha, power)
    return {
        "mde": None if se is None else float(multiplier * se),
        "se": se,
        "sd": sd,
        "n": len(data),
        "n_eff": n_eff,
        "alpha": float(alpha),
        "power": float(power),
        "block_bars": int(block_bars),
    }


def calibrate_coverage(
    *,
    n_rows: int = 600,
    block_bars: int = 24,
    alphas: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    draws: int = 1000,
    simulations: int = 400,
    autocorrelation: float = 0.8,
    effect: float = 0.0,
    seed: int = 0,
) -> dict[str, object]:
    """Simulate paired differences with known mean and block autocorrelation.

    Each simulation draws a fresh serially-correlated series (row-level AR(1)),
    applies the same time-block bootstrap CI rule the lab uses, and records
    whether the CI covers the true ``effect`` and whether it excludes zero. Coverage should
    be at least ``1 - alpha`` and the false-positive rate at a zero effect
    should sit at or below ``alpha`` for every nominal level.
    """
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2")
    width = max(1, int(block_bars))
    n_blocks = max(2, math.ceil(n_rows / width))
    rng = np.random.default_rng(int(seed))
    rho = max(-0.999, min(0.999, float(autocorrelation)))
    results: dict[str, dict[str, float | int | None]] = {
        f"{float(alpha):.4g}": {
            "alpha": float(alpha),
            "coverage": 0,
            "excludes_zero": 0,
            "simulations": int(simulations),
        }
        for alpha in alphas
    }
    p_values: list[float] = []
    for simulation in range(int(simulations)):
        # Row-level AR(1): realistic serial correlation inside a block. The
        # block bootstrap is supposed to absorb this by resampling contiguous
        # blocks; a block shorter than the correlation length would undercover.
        innovations = rng.standard_normal(n_rows)
        series = np.zeros(n_rows, dtype=float)
        for index in range(n_rows):
            previous = series[index - 1] if index else 0.0
            series[index] = (
                rho * previous
                + math.sqrt(max(0.0, 1.0 - rho * rho)) * innovations[index]
            )
        # Do not force the sample mean to the true effect: the finite-sample
        # mean must vary for coverage and p-values to mean anything.
        sample = series + float(effect)
        means = block_bootstrap_means(
            sample,
            block_bars=width,
            draws=int(draws),
            seed=int(seed) + simulation + 1,
        )
        if means.size < 2:
            continue
        # A two-sided percentile p-value: the mass of the bootstrap mean at or
        # beyond zero, doubled. At a zero effect this should be uniform.
        below = float(np.mean(means <= 0.0))
        above = float(np.mean(means >= 0.0))
        p_values.append(min(1.0, 2.0 * min(below, above)))
        for alpha in alphas:
            key = f"{float(alpha):.4g}"
            low = float(np.quantile(means, float(alpha) / 2.0))
            high = float(np.quantile(means, 1.0 - float(alpha) / 2.0))
            if low <= float(effect) <= high:
                results[key]["coverage"] = float(results[key]["coverage"]) + 1
            if low > 0.0 or high < 0.0:
                results[key]["excludes_zero"] = (
                    float(results[key]["excludes_zero"]) + 1
                )
    for row in results.values():
        row["coverage"] = float(row["coverage"]) / float(simulations)
        row["excludes_zero"] = float(row["excludes_zero"]) / float(simulations)
    return {
        "n_rows": int(n_rows),
        "block_bars": width,
        "n_blocks": n_blocks,
        "autocorrelation": float(autocorrelation),
        "effect": float(effect),
        "seed": int(seed),
        "alphas": results,
        "p_value_uniformity": p_value_uniformity(p_values),
    }


def p_value_uniformity(
    p_values: Sequence[float],
    *,
    bins: int = 10,
) -> dict[str, object]:
    """A bin-count uniformity report for a set of p-values."""
    values = [min(max(float(value), 0.0), 1.0) for value in p_values]
    counts = [0] * int(bins)
    for value in values:
        index = min(int(bins) - 1, int(value * bins))
        counts[index] += 1
    expected = len(values) / float(bins) if bins else 0.0
    chi_square = (
        sum((count - expected) ** 2 / expected for count in counts)
        if expected > 0.0
        else None
    )
    return {
        "n": len(values),
        "bins": int(bins),
        "counts": counts,
        "expected": expected,
        "chi_square": chi_square,
        "mean": (sum(values) / len(values)) if values else None,
    }


__all__ = [
    "DEFAULT_POWER",
    "block_bootstrap_means",
    "calibrate_coverage",
    "minimum_detectable_effect",
    "p_value_uniformity",
]
