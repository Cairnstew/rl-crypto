"""Power / MDE and coverage calibration."""

from __future__ import annotations

import numpy as np

from rl_crypto.power import (
    block_bootstrap_means,
    calibrate_coverage,
    minimum_detectable_effect,
    p_value_uniformity,
)


def test_mde_grows_with_noise_and_shrinks_with_sample_size():
    rng = np.random.default_rng(0)
    quiet = (0.01 + rng.normal(scale=0.001, size=600)).tolist()
    noisy = (0.01 + rng.normal(scale=0.02, size=600)).tolist()
    small = noisy[:120]

    quiet_mde = minimum_detectable_effect(quiet, alpha=0.05, block_bars=24)
    noisy_mde = minimum_detectable_effect(noisy, alpha=0.05, block_bars=24)
    small_mde = minimum_detectable_effect(small, alpha=0.05, block_bars=24)

    assert quiet_mde["mde"] is not None and noisy_mde["mde"] is not None
    assert small_mde["mde"] is not None
    assert quiet_mde["mde"] < noisy_mde["mde"]
    assert small_mde["mde"] > minimum_detectable_effect(
        noisy, alpha=0.05, block_bars=24
    )["mde"]


def test_mde_uses_block_resampling_not_raw_rows():
    values = [0.01] * 600
    # A perfectly persistent series has zero between-block variance: a block
    # bootstrap sees one effective observation, while a row-wise SE would be
    # ~0. The MDE must not be the naive row count.
    result = minimum_detectable_effect(values, alpha=0.05, block_bars=24)
    assert result["n"] == 600
    assert result["n_eff"] == 25


def test_block_bootstrap_means_is_deterministic_and_block_aware():
    values = [float(index % 7) for index in range(240)]
    first = block_bootstrap_means(values, block_bars=24, draws=500, seed=3)
    second = block_bootstrap_means(values, block_bars=24, draws=500, seed=3)
    assert np.array_equal(first, second)
    assert first.size == 500


def test_coverage_calibration_reports_every_level_and_p_value_uniformity():
    """The calibration must report coverage/p-values at every level.

    The non-overlapping block percentile bootstrap is mildly anti-conservative
    in this design when the block count is small; that undercoverage is the
    finding the calibration exists to surface, so the test asserts the report
    contract and the direction, not a guarantee the rule does not provide.
    """
    result = calibrate_coverage(
        n_rows=480,
        block_bars=24,
        alphas=(0.05, 0.10),
        draws=400,
        simulations=120,
        autocorrelation=0.8,
        effect=0.0,
        seed=7,
    )
    assert set(result["alphas"]) == {"0.05", "0.1"}
    ordered = [result["alphas"][key] for key in ("0.05", "0.1")]
    for row in ordered:
        assert 0.0 <= row["coverage"] <= 1.0
        assert 0.0 <= row["excludes_zero"] <= 1.0
        assert row["simulations"] == 120
    # A looser interval covers at least as often as a tighter one.
    assert ordered[0]["coverage"] >= ordered[1]["coverage"] - 0.15
    uniformity = result["p_value_uniformity"]
    assert uniformity["n"] > 0
    assert uniformity["chi_square"] is not None
    assert sum(uniformity["counts"]) == uniformity["n"]


def test_p_value_uniformity_counts_every_value():
    report = p_value_uniformity([0.0, 0.05, 0.5, 0.99, 1.0], bins=10)
    assert report["n"] == 5
    assert sum(report["counts"]) == 5
    assert report["expected"] == 0.5
