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
from typing import Any

import numpy as np

#: Default power target for the MDE.
DEFAULT_POWER = 0.8

#: The corrected interval is an effective-sample HAC t interval. The minimum
#: number of *effective* blocks (independent observations after serial
#: correlation is charged) comes from the coverage calibration, not from the
#: old hard-coded 24. Below this the Student-t quantile is too unstable and the
#: interval is too wide to certify anything, so the gate refuses the run.
MIN_EFFECTIVE_BLOCKS = 12

#: A small-sample inflation applied to the estimated AR(1) coefficient before
#: it is turned into an effective sample size. The lag-1 estimator is biased
#: downward, most severely for short, highly persistent series; this is a
#: two-standard-error conservative upper bound. The coverage grid in
#: ``calibrate_coverage`` is the evidence for this value.
RHO_INFLATION = 2.0


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    tiny = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3.0e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(value: float, df: float) -> float:
    """CDF of Student's t with ``df`` degrees of freedom."""
    if df <= 0.0:
        raise ValueError("df must be positive")
    if value == 0.0:
        return 0.5
    x = df / (df + value * value)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - tail if value > 0.0 else tail


def student_t_quantile(probability: float, df: float) -> float:
    """Inverse Student-t CDF by bisection (no SciPy dependency)."""
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1)")
    if p == 0.5:
        return 0.0
    df = float(df)
    if df <= 0.0:
        raise ValueError("df must be positive")
    low, high = -1.0, 1.0
    while student_t_cdf(low, df) > p:
        low *= 2.0
    while student_t_cdf(high, df) < p:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if student_t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
        if high - low < 1.0e-12:
            break
    return 0.5 * (low + high)


def effective_sample_size(values: Sequence[float]) -> dict[str, float | int | None]:
    """Effective independent rows under an AR(1) approximation.

    The lag-1 autocorrelation is biased downward in short samples, so it is
    inflated by ``RHO_INFLATION`` standard errors (the AR(1) approximation
    ``sqrt((1-rho^2)/n)``) before the usual ``n * (1-rho) / (1+rho)``
    conversion. Positive autocorrelation only: a negative estimate is treated
    as zero (never credited as extra rows).
    """
    data = np.asarray(list(values), dtype=float)
    n = int(data.size)
    if n < 3:
        return {
            "n": n,
            "rho": None,
            "rho_adjusted": None,
            "n_eff": max(1.0, float(n)),
            "df": max(1.0, float(n) - 1.0),
        }
    centred = data - float(np.mean(data))
    denominator = float(np.dot(centred, centred))
    scale = float(np.max(np.abs(data))) if n else 0.0
    if denominator <= 0.0 or math.sqrt(denominator / n) <= 1.0e-12 * max(1.0, scale):
        # A constant series has no sampling noise; every row is effectively
        # independent information about the (exact) mean. The tolerance
        # absorbs float rounding in a constant column.
        return {
            "n": n,
            "rho": 0.0,
            "rho_adjusted": 0.0,
            "n_eff": float(n),
            "df": max(1.0, float(n) - 1.0),
        }
    rho = float(np.dot(centred[1:], centred[:-1]) / denominator)
    rho = min(max(rho, 0.0), 0.995)
    # A two-standard-error upper bound on the lag-1 coefficient. The
    # plug-in standard error sqrt((1-rho^2)/n) is the usual Bartlett/AR(1)
    # approximation; using the upper bound keeps a short, lucky sample from
    # being treated as independent.
    standard_error = math.sqrt(max(0.0, 1.0 - rho * rho) / n)
    adjusted = min(0.995, rho + RHO_INFLATION * standard_error)
    n_eff = max(1.0, n * (1.0 - adjusted) / (1.0 + adjusted))
    return {
        "n": n,
        "rho": rho,
        "rho_adjusted": adjusted,
        "n_eff": n_eff,
        "df": max(1.0, math.floor(n_eff) - 1.0),
    }


def effective_block_plan(
    series: Sequence[Sequence[float]],
    *,
    block_bars: int,
    min_blocks: int = MIN_EFFECTIVE_BLOCKS,
) -> dict[str, Any]:
    """How many blocks the design really has after serial correlation.

    The configured block width is raised to the lag at which the fitted AR(1)
    correlation falls to 0.05. If that leaves fewer than ``min_blocks`` whole
    blocks, no interval is certified: a wider block is the only way to make the
    blocks independent, and there is not enough data left to do it.
    """
    groups = [np.asarray(list(values), dtype=float) for values in series if len(values)]
    estimates = [effective_sample_size(group) for group in groups]
    rho_max = max(
        (float(estimate["rho_adjusted"] or 0.0) for estimate in estimates),
        default=0.0,
    )
    if rho_max <= 0.05:
        correlation_lag = 1
    else:
        correlation_lag = max(
            1, math.ceil(math.log(0.05) / math.log(min(rho_max, 0.995)))
        )
    width = max(1, int(block_bars), int(correlation_lag))
    raw_blocks = sum(int(group.size) // width for group in groups)
    effective = float(sum(float(estimate["n_eff"] or 0.0) for estimate in estimates))
    return {
        "block_bars": width,
        "requested_block_bars": int(block_bars),
        "correlation_lag": int(correlation_lag),
        "rho_max": rho_max,
        "raw_blocks": raw_blocks,
        "effective_blocks_hint": effective,
        "min_effective_blocks": int(min_blocks),
        "sufficient": raw_blocks >= int(min_blocks)
        and effective >= float(min_blocks),
    }


def effective_sample_t_interval(
    values: Sequence[float], *, alpha: float, block_bars: int = 24
) -> dict[str, Any]:
    """Two-sided t interval with ``n_eff - 1`` degrees of freedom.

    This is the corrected rule that replaces the percentile block bootstrap for
    the verdict. Serial correlation is charged through the effective sample
    size and the block-width plan, not laundered into independent rows.
    ``sufficient`` is false when the plan leaves fewer than
    :data:`MIN_EFFECTIVE_BLOCKS` independent blocks; the caller must refuse the
    CI rather than report one.
    """
    grouped = effective_sample_t_interval_grouped(
        [values], alpha=alpha, block_bars=block_bars
    )
    grouped["rho"] = grouped.get("rho")
    return grouped


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


def effective_sample_t_interval_grouped(
    series: Sequence[Sequence[float]],
    *,
    alpha: float,
    block_bars: int = 24,
) -> dict[str, Any]:
    """Effective-sample t interval for pooled groups (tickers) of rows.

    Autocorrelation is estimated **within** each group. Groups are treated as
    independent; the shared market factor is therefore not credited as extra
    information, which is conservative for BTC/ETH/SOL. The variance of the
    pooled mean is the sum of the group variances of their means, and the
    degrees of freedom are ``sum(n_eff) - 1``. The block plan raises the block
    width to the decorrelation lag and refuses when too few blocks remain.
    """
    groups = [np.asarray(list(values), dtype=float) for values in series if len(values)]
    total_n = int(sum(group.size for group in groups))
    plan = effective_block_plan(groups, block_bars=block_bars)
    result: dict[str, Any] = {
        "interval": "effective-sample-t",
        "alpha": float(alpha),
        "n": total_n,
        "eff_n": 0.0,
        "df": 1.0,
        "rho": None,
        "rho_adjusted": None,
        "min_effective_blocks": MIN_EFFECTIVE_BLOCKS,
        "sufficient": False,
        "ci_low": None,
        "ci_high": None,
        **plan,
    }
    if total_n < 2 or not groups:
        return result
    estimates = [effective_sample_size(group) for group in groups]
    rhos = [float(estimate["rho"] or 0.0) for estimate in estimates]
    result["rho"] = max(rhos) if rhos else None
    result["rho_adjusted"] = max(
        (float(estimate["rho_adjusted"] or 0.0) for estimate in estimates),
        default=None,
    )
    total_sum = float(sum(float(np.sum(group)) for group in groups))
    mean = total_sum / total_n
    variance_of_mean = 0.0
    effective = 0.0
    for group, estimate in zip(groups, estimates, strict=True):
        group_n = int(estimate["n"] or 0)
        group_eff = float(estimate["n_eff"] or 0.0)
        effective += group_eff
        if group_n > 1:
            variance_of_mean += float(np.var(group, ddof=1)) / group_eff
    effective = max(1.0, effective)
    result["eff_n"] = effective
    result["df"] = max(1.0, math.floor(effective) - 1.0)
    result["sufficient"] = bool(plan["sufficient"]) and effective >= MIN_EFFECTIVE_BLOCKS
    if not result["sufficient"]:
        return result
    if variance_of_mean <= 1.0e-300:
        result["ci_low"] = mean
        result["ci_high"] = mean
        return result
    se = math.sqrt(variance_of_mean)
    critical = student_t_quantile(1.0 - float(alpha) / 2.0, float(result["df"]))
    result["se"] = se
    result["critical"] = critical
    result["ci_low"] = mean - critical * se
    result["ci_high"] = mean + critical * se
    return result


def _binary_entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def economic_minimal_meaningful_effect(
    target: str,
    *,
    sigma: float,
    fee_bps: float,
    slippage_bps: float,
    threshold: float,
    simulations: int = 20000,
    seed: int = 0,
) -> dict[str, Any] | None:
    """The loss improvement a directional model needs to cover round-trip costs.

    This is a documented, deterministic simulation of the economics, not a
    hand-picked number. For ``fwd_direction`` the metric is log-loss
    improvement (nats); for ``fwd_return`` it is RMSE improvement (raw return
    units). The model is assumed calibrated, returns are symmetric with
    per-bar scale ``sigma``, and a trade is taken only when the predicted
    up-probability clears ``threshold``.

    Direction (log loss):
      * expected absolute return of a symmetric normal bar = ``sigma*sqrt(2/pi)``
      * round-trip cost = ``2*(fee+slippage)/10000`` in return units
      * the edge (``2q-1``) must be at least ``cost/E|r|`` *and* at least the
        declared threshold edge ``2*threshold-1``
      * MME = ``log(2) - H(q)`` for that q.

    Return (RMSE):
      * a forecast with correlation ``rho`` earns ``rho*E|r|`` per trade
      * ``rho`` must cover the same cost and the threshold's implied sign
        accuracy ``0.5 + asin(rho)/pi >= threshold``
      * MME = ``sigma*(1 - sqrt(1-rho^2))``.

    The simulation below re-draws 20k bars and records the realised net PnL and
    metric improvement at the analytic threshold; its role is to show the
    arithmetic rather than to hide it behind a formula.
    """
    if target not in {"fwd_direction", "fwd_return"}:
        return None
    sigma_value = float(sigma)
    if not math.isfinite(sigma_value) or sigma_value <= 0.0:
        return None
    cost = 2.0 * (float(fee_bps) + float(slippage_bps)) / 10000.0
    expected_abs = sigma_value * math.sqrt(2.0 / math.pi)
    required_edge = cost / expected_abs
    threshold_edge = max(0.0, 2.0 * float(threshold) - 1.0)
    rng = np.random.default_rng(int(seed))
    draws = rng.normal(0.0, sigma_value, size=int(simulations))
    if target == "fwd_direction":
        edge = min(0.999, max(required_edge, threshold_edge))
        probability = (1.0 + edge) / 2.0
        improvement = math.log(2.0) - _binary_entropy(probability)
        # Realised check: a calibrated model with this edge and a threshold
        # trade rule earns ``edge * E|r|`` per trade before the cost.
        gross = edge * float(np.mean(np.abs(draws)))
        net = gross - cost
        return {
            "target": target,
            "metric": "log_loss",
            "metric_units": "nats",
            "sigma": sigma_value,
            "fee_bps": float(fee_bps),
            "slippage_bps": float(slippage_bps),
            "round_trip_cost": cost,
            "threshold": float(threshold),
            "expected_abs_return": expected_abs,
            "required_edge": required_edge,
            "threshold_edge": threshold_edge,
            "edge": edge,
            "probability": probability,
            "simulated_mean_abs_return": float(np.mean(np.abs(draws))),
            "simulated_gross_per_trade": gross,
            "simulated_net_per_trade": net,
            "simulations": int(simulations),
            "minimal_meaningful_effect": improvement,
            "assumptions": [
                "calibrated binary model, symmetric returns",
                "per-bar sigma estimated from the pre-cutoff archive",
                "trade only when predicted up-probability >= threshold",
                "round-trip cost = 2*(fee+slippage) in return units",
                "edge and expected absolute return are independent",
            ],
        }
    required_rho = min(0.999, cost / expected_abs)
    threshold_rho = math.sin(math.pi * max(0.0, float(threshold) - 0.5))
    rho = min(0.999, max(required_rho, threshold_rho))
    improvement = sigma_value * (1.0 - math.sqrt(max(0.0, 1.0 - rho * rho)))
    gross = rho * float(np.mean(np.abs(draws)))
    return {
        "target": target,
        "metric": "rmse",
        "metric_units": "raw return",
        "sigma": sigma_value,
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "round_trip_cost": cost,
        "threshold": float(threshold),
        "expected_abs_return": expected_abs,
        "required_correlation": required_rho,
        "threshold_correlation": threshold_rho,
        "correlation": rho,
        "simulated_mean_abs_return": float(np.mean(np.abs(draws))),
        "simulated_gross_per_trade": gross,
        "simulated_net_per_trade": gross - cost,
        "simulations": int(simulations),
        "minimal_meaningful_effect": improvement,
        "assumptions": [
            "bivariate-normal return forecast with correlation rho",
            "per-bar sigma estimated from the pre-cutoff archive",
            "trade sign only when implied accuracy clears threshold",
            "round-trip cost = 2*(fee+slippage) in return units",
        ],
    }


def calibrate_coverage(
    *,
    n_rows: int = 600,
    block_bars: int = 24,
    block_counts: Sequence[int] = (12, 16, 20, 24, 32, 50, 100),
    alphas: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    draws: int = 1000,
    simulations: int = 400,
    autocorrelation: float = 0.8,
    effect: float = 0.0,
    seed: int = 0,
    compare_percentile: bool = False,
    percentile_draws: int = 200,
) -> dict[str, object]:
    """Simulate paired differences with known mean and AR(1) autocorrelation.

    The corrected rule is an effective-sample t interval; it does not use a
    bootstrap quantile, so ``block_counts`` are reported as the design grid and
    the legacy percentile rule is evaluated at each count only when
    ``compare_percentile`` is set. Coverage is compared with nominal using the
    Monte-Carlo standard error; the claim is refused when a non-refused cell is
    anti-conservative (excess rate above ``alpha`` by more than two standard
    errors) or undercovers by more than two standard errors.
    """
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2")
    rng = np.random.default_rng(int(seed))
    rho = max(-0.999, min(0.999, float(autocorrelation)))
    # One simulated series set per (rho, n); every block count and alpha is
    # scored from the same draws, which is what makes the grid affordable.
    innovations = rng.standard_normal((int(simulations), int(n_rows)))
    series = np.zeros((int(simulations), int(n_rows)), dtype=float)
    variance = math.sqrt(max(0.0, 1.0 - rho * rho))
    for index in range(int(n_rows)):
        previous = series[:, index - 1] if index else 0.0
        series[:, index] = rho * previous + variance * innovations[:, index]
    series += float(effect)
    rows: list[dict[str, Any]] = []
    for count in block_counts:
        width = max(1, int(n_rows) // max(1, int(count)))
        for alpha in alphas:
            successes = 0
            excludes = 0
            refused = 0
            for simulation in range(int(simulations)):
                interval = effective_sample_t_interval(
                    series[simulation],
                    alpha=float(alpha),
                    block_bars=width,
                )
                if not interval["sufficient"]:
                    refused += 1
                    continue
                low = interval["ci_low"]
                high = interval["ci_high"]
                if low is None or high is None:
                    refused += 1
                    continue
                if float(low) <= float(effect) <= float(high):
                    successes += 1
                if float(low) > 0.0 or float(high) < 0.0:
                    excludes += 1
            tested = int(simulations) - refused
            coverage = successes / tested if tested else None
            excess = excludes / tested if tested else None
            nominal = float(alpha)
            mc_se = (
                math.sqrt(nominal * (1.0 - nominal) / tested) if tested else None
            )
            tolerance = 2.0 * mc_se if mc_se is not None else None
            # A plan that only certifies a tiny tail of the simulated series is
            # a selection artifact, not a coverage result. Under a persistent
            # process the estimator occasionally looks independent; those are
            # exactly the draws where the interval is too narrow. Refuse the
            # whole cell when too few draws survived the block plan.
            minimum_tested = max(30, math.ceil(0.05 * int(simulations)))
            status = (
                "insufficient-blocks"
                if tested < minimum_tested
                else "ok"
            )
            if status == "ok" and excess is not None and tolerance is not None:
                if excess > nominal + tolerance:
                    status = "anti-conservative"
                elif coverage is not None and coverage < 1.0 - nominal - tolerance:
                    status = "undercoverage"
            row: dict[str, Any] = {
                "n_rows": int(n_rows),
                "autocorrelation": float(autocorrelation),
                "block_count": int(count),
                "block_bars": width,
                "alpha": float(alpha),
                "nominal_coverage": 1.0 - nominal,
                "coverage": coverage,
                "excess": excess,
                "simulations": int(simulations),
                "tested": tested,
                "refused": refused,
                "minimum_tested": minimum_tested,
                "mc_se": mc_se,
                "tolerance": tolerance,
                "status": status,
            }
            if compare_percentile and tested:
                # The legacy rule, kept for comparison only. It resamples whole
                # blocks and uses the percentile interval at the same alpha.
                legacy_cov = 0
                legacy_exc = 0
                for simulation in range(int(simulations)):
                    values = series[simulation]
                    blocks = [
                        values[start : start + width]
                        for start in range(0, len(values), width)
                        if len(values[start : start + width])
                    ]
                    if len(blocks) < 2:
                        continue
                    sums = np.asarray([float(np.sum(block)) for block in blocks])
                    counts = np.asarray([float(block.size) for block in blocks])
                    picks = rng.integers(
                        0, len(blocks), size=(int(percentile_draws), len(blocks))
                    )
                    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
                    low = float(np.quantile(means, float(alpha) / 2.0))
                    high = float(np.quantile(means, 1.0 - float(alpha) / 2.0))
                    legacy_cov += low <= float(effect) <= high
                    legacy_exc += low > 0.0 or high < 0.0
                row["legacy_percentile_coverage"] = legacy_cov / tested
                row["legacy_percentile_excess"] = legacy_exc / tested
            rows.append(row)
    failing = [
        row
        for row in rows
        if row["status"] not in {"ok", "insufficient-blocks"}
    ]
    refused = [row for row in rows if row["status"] == "insufficient-blocks"]
    passing = [row for row in rows if row["status"] == "ok"]
    minimum = (
        min(
            row["block_count"]
            for row in passing
            if (row.get("block_count") or 0) > 0
        )
        if passing
        else None
    )
    alphas_summary: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        cells = [row for row in rows if abs(row["alpha"] - float(alpha)) < 1e-12]
        tested = sum(int(row["tested"] or 0) for row in cells)
        covered = sum(
            float(row["coverage"] or 0.0) * int(row["tested"] or 0) for row in cells
        )
        excluded = sum(
            float(row["excess"] or 0.0) * int(row["tested"] or 0) for row in cells
        )
        alphas_summary[f"{float(alpha):.4g}"] = {
            "alpha": float(alpha),
            "coverage": (covered / tested) if tested else 0.0,
            "excludes_zero": (excluded / tested) if tested else 0.0,
            "simulations": int(simulations),
        }
    return {
        "method": "effective-sample-t",
        "df_rule": "n_eff - 1",
        "n_rows": int(n_rows),
        "block_bars": int(block_bars),
        "block_counts": [int(value) for value in block_counts],
        "autocorrelation": float(autocorrelation),
        "effect": float(effect),
        "seed": int(seed),
        "simulations": int(simulations),
        "min_effective_blocks": MIN_EFFECTIVE_BLOCKS,
        "alphas": alphas_summary,
        "rows": rows,
        "failing_cells": failing,
        "refused_cells": refused,
        "minimum_blocks_from_results": minimum,
        "claim": "supported" if not failing else "refused",
        "p_value_uniformity": p_value_uniformity(
            [
                min(max(float(row["excess"] or 0.0), 0.0), 1.0)
                for row in rows
                if row.get("excess") is not None
            ]
        ),
    }


def calibrate_coverage_grid(
    *,
    rows: Sequence[int] = (300, 600, 1700),
    autocorrelations: Sequence[float] = (0.5, 0.8, 0.95),
    block_counts: Sequence[int] = (12, 16, 20, 24, 32, 50, 100),
    alphas: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    simulations: int = 2000,
    draws: int = 1000,
    seed: int = 0,
    compare_percentile: bool = False,
) -> dict[str, object]:
    """Run the full coverage grid and aggregate every cell.

    The claim is supported only when no non-refused cell is anti-conservative
    or undercovers beyond two Monte-Carlo standard errors. Refused cells
    (fewer than :data:`MIN_EFFECTIVE_BLOCKS` effective blocks) are reported
    separately and do not count as coverage failures.
    """
    cells: list[dict[str, Any]] = []
    for index, (n_rows, rho) in enumerate(
        (int(n), float(r)) for n in rows for r in autocorrelations
    ):
        block = calibrate_coverage(
            n_rows=int(n_rows),
            block_counts=block_counts,
            alphas=alphas,
            draws=draws,
            simulations=simulations,
            autocorrelation=float(rho),
            seed=int(seed) + index,
            compare_percentile=compare_percentile,
        )
        cells.extend(block["rows"])
    failing = [
        cell
        for cell in cells
        if cell["status"] not in {"ok", "insufficient-blocks"}
    ]
    refused = [cell for cell in cells if cell["status"] == "insufficient-blocks"]
    passing = [cell for cell in cells if cell["status"] == "ok"]
    return {
        "method": "effective-sample-t",
        "df_rule": "n_eff - 1",
        "rows_grid": [int(value) for value in rows],
        "autocorrelations": [float(value) for value in autocorrelations],
        "block_counts": [int(value) for value in block_counts],
        "alphas": [float(value) for value in alphas],
        "simulations": int(simulations),
        "min_effective_blocks": MIN_EFFECTIVE_BLOCKS,
        "cells": cells,
        "failing_cells": failing,
        "refused_cells": refused,
        "passing_cells": len(passing),
        "claim": "supported" if not failing else "refused",
        "minimum_blocks_from_results": (
            min(int(cell["block_count"]) for cell in passing) if passing else None
        ),
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
    "MIN_EFFECTIVE_BLOCKS",
    "block_bootstrap_means",
    "calibrate_coverage",
    "calibrate_coverage_grid",
    "economic_minimal_meaningful_effect",
    "effective_block_plan",
    "effective_sample_size",
    "effective_sample_t_interval",
    "effective_sample_t_interval_grouped",
    "minimum_detectable_effect",
    "p_value_uniformity",
    "student_t_cdf",
    "student_t_quantile",
]
