from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from rl_crypto.calibration.tails import NoiseSpec, generate_ar1_innovations
from rl_crypto.power import calibrate_coverage, calibrate_coverage_grid


def calibrate_coverage_tails(
    *,
    n_rows: int = 600,
    block_bars: int = 24,
    block_counts: tuple[int, ...] = (12, 16, 20, 24, 32, 50, 100),
    alphas: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
    draws: int = 1000,
    simulations: int = 400,
    autocorrelation: float = 0.8,
    effect: float = 0.0,
    seed: int = 0,
    compare_percentile: bool = False,
    percentile_draws: int = 200,
    noise_kind: str = "normal",
    df: float | None = None,
    garch: bool = False,
    garch_params: tuple[float, float, float] = (0.01, 0.1, 0.85),
) -> dict[str, Any]:
    """Extended coverage calibration with t innovations and GARCH clustering."""
    import math
    from rl_crypto.power import effective_sample_t_interval, p_value_uniformity

    if n_rows < 2:
        raise ValueError("n_rows must be at least 2")
    rng = np.random.default_rng(int(seed))
    rho = max(-0.999, min(0.999, float(autocorrelation)))
    noise = NoiseSpec(kind=noise_kind, df=df, garch=garch)
    series = generate_ar1_innovations(
        rng, (int(simulations), int(n_rows)), noise, rho, garch_params=garch_params
    )
    series = series + float(effect)
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
            mc_se = math.sqrt(nominal * (1.0 - nominal) / tested) if tested else None
            tolerance = 2.0 * mc_se if mc_se is not None else None
            minimum_tested = max(30, math.ceil(0.05 * int(simulations)))
            status = "insufficient-blocks" if tested < minimum_tested else "ok"
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
                "noise_kind": noise_kind,
                "df": df,
                "garch": bool(garch),
            }
            rows.append(row)
    failing = [row for row in rows if row["status"] not in {"ok", "insufficient-blocks"}]
    refused = [row for row in rows if row["status"] == "insufficient-blocks"]
    passing = [row for row in rows if row["status"] == "ok"]
    minimum = min((row["block_count"] for row in passing if (row.get("block_count") or 0) > 0), default=None)
    alphas_summary: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        cells = [row for row in rows if abs(row["alpha"] - float(alpha)) < 1e-12]
        tested = sum(int(row["tested"] or 0) for row in cells)
        covered = sum(float(row["coverage"] or 0.0) * int(row["tested"] or 0) for row in cells)
        excluded = sum(float(row["excess"] or 0.0) * int(row["tested"] or 0) for row in cells)
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
        "min_effective_blocks": 12,
        "alphas": alphas_summary,
        "rows": rows,
        "failing_cells": failing,
        "refused_cells": refused,
        "minimum_blocks_from_results": minimum,
        "claim": "supported" if not failing else "refused",
        "p_value_uniformity": p_value_uniformity(
            [min(max(float(row["excess"] or 0.0), 0.0), 1.0) for row in rows if row.get("excess") is not None]
        ),
        "noise": {"kind": noise_kind, "df": df, "garch": bool(garch)},
    }


def run_t4_grid(
    rows_list: tuple[int, ...] = (300, 600, 1700),
    rhos: tuple[float, ...] = (0.5, 0.8, 0.95),
    alphas: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2),
    block_counts: tuple[int, ...] = (12, 16, 20, 24, 32, 50, 100),
    simulations: int = 400,
    draws: int = 1000,
    seed: int = 12345,
    compare_percentile: bool = False,
) -> dict[str, Any]:
    """Run full T4 grid over t innovations (df 3,5) + GARCH clustering."""
    all_results = []
    noise_configs = [
        {"noise_kind": "t", "df": 3.0, "garch": False},
        {"noise_kind": "t", "df": 5.0, "garch": False},
        {"noise_kind": "normal", "df": None, "garch": True},  # GARCH-style clustering over rows
        {"noise_kind": "t", "df": 3.0, "garch": True},
        {"noise_kind": "t", "df": 5.0, "garch": True},
    ]
    for ni, nc in enumerate(noise_configs):
        cells = []
        for ri, n_rows in enumerate(rows_list):
            for rj, rho in enumerate(rhos):
                idx = ni * 1000 + ri * 100 + rj
                res = calibrate_coverage_tails(
                    n_rows=int(n_rows),
                    block_counts=block_counts,
                    alphas=alphas,
                    draws=draws,
                    simulations=simulations,
                    autocorrelation=float(rho),
                    seed=int(seed) + idx,
                    compare_percentile=compare_percentile,
                    noise_kind=nc["noise_kind"],
                    df=nc["df"],
                    garch=bool(nc["garch"]),
                )
                cells.extend(res["rows"])
        failing = [c for c in cells if c["status"] not in {"ok", "insufficient-blocks"}]
        refused = [c for c in cells if c["status"] == "insufficient-blocks"]
        all_results.append({
            "noise_config": nc,
            "cells": cells,
            "failing_cells": failing,
            "refused_cells": refused,
            "claim": "supported" if not failing else "refused",
        })
    return {
        "grid": {
            "rows": list(rows_list),
            "rho": list(rhos),
            "alphas": list(alphas),
            "block_counts": list(block_counts),
        },
        "simulations": simulations,
        "noise_configs": noise_configs,
        "results": all_results,
    }
