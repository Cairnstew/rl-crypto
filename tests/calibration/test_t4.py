from __future__ import annotations

from rl_crypto.calibration.t4 import calibrate_coverage_tails, run_t4_grid


def test_t4_tails_df3_df5_quick():
    res = calibrate_coverage_tails(
        n_rows=300,
        block_counts=(12, 24),
        alphas=(0.05,),
        simulations=60,
        autocorrelation=0.8,
        seed=1,
        noise_kind="t",
        df=3.0,
        garch=False,
    )
    assert "rows" in res
    assert res["noise"]["df"] == 3.0


def test_t4_garch_quick():
    res = calibrate_coverage_tails(
        n_rows=300,
        block_counts=(12, 24),
        alphas=(0.05,),
        simulations=60,
        autocorrelation=0.5,
        seed=2,
        noise_kind="normal",
        df=None,
        garch=True,
    )
    assert res["noise"]["garch"] is True


def test_t4_grid_quick():
    res = run_t4_grid(
        rows_list=(300,),
        rhos=(0.8,),
        alphas=(0.05,),
        block_counts=(12, 24),
        simulations=40,
        seed=3,
    )
    assert len(res["results"]) >= 1
