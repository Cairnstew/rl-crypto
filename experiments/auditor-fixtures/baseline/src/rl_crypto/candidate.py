"""A small candidate measurement module used by the auditor fixtures.

Self-contained on purpose: the auditor sees only the diff, so every leak or
omission must be visible in the changed lines.
"""

from __future__ import annotations

import numpy as np

from rl_crypto.targets import label_purge

FEATURE_NAMES = ("ret_1", "vol_6")
RIDGE_ALPHAS = (0.1, 1.0, 10.0)


def build_features(rows, index):
    """Only rows <= index may be read."""
    if index < 6:
        return None
    window = [rows[j].close for j in range(index - 5, index + 1)]
    ret_1 = window[-1] / window[-2] - 1.0
    vol_6 = float(np.std([window[j + 1] / window[j] - 1.0 for j in range(5)]))
    return {"ret_1": ret_1, "vol_6": vol_6}


def chronological_split(samples, target, horizon):
    split = int(len(samples) * 0.8)
    purge = label_purge(target, horizon)
    return samples[: max(0, split - purge)], samples[split:]


def standardize(train, test):
    mean = float(np.mean([s.features["ret_1"] for s in train]))
    std = float(np.std([s.features["ret_1"] for s in train])) or 1.0
    return (
        [((s.features["ret_1"] - mean) / std) for s in train],
        [((s.features["ret_1"] - mean) / std) for s in test],
    )


def fit_ridge(xs, ys, alpha):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    centered = x - x.mean()
    coef = float(np.dot(centered, y - y.mean()) / (np.dot(centered, centered) + alpha))
    intercept = float(y.mean() - coef * x.mean())
    return coef, intercept


def evaluate(samples, target="fwd_direction", horizon=1):
    train, test = chronological_split(samples, target, horizon)
    xs_train, xs_test = standardize(train, test)
    ys = [s.targets[target] for s in train]
    coef, intercept = fit_ridge(xs_train, ys, alpha=1.0)
    predictions = [coef * x + intercept for x in xs_test]
    truth = [s.targets[target] for s in test]
    return {"n": len(truth), "predictions": predictions, "truth": truth}


def policy_pnl(positions, returns, *, fee_bps, slippage_bps):
    cost = (fee_bps + slippage_bps) / 10_000.0
    pnl = []
    previous = 0.0
    for position, realised in zip(positions, returns, strict=True):
        pnl.append(position * realised - cost * abs(position - previous))
        previous = position
    pnl[-1] -= cost * abs(previous)  # charge the close
    return float(sum(pnl))
