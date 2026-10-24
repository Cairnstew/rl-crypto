"""Structural causality checks: truncation and perturbation invariance."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_crypto.experiments import contract_violation
from rl_crypto.features import FEATURE_NAMES
from rl_crypto.ml.base import Standardizer
from rl_crypto.ml.tier1 import train_tier1
from rl_crypto.samples import Sample, build_samples
from rl_crypto.structural import perturbation_invariance, truncation_invariance
from rl_crypto.targets import TARGET_NAMES

from .conftest import make_dataset_rows


def _rows(n: int = 220, *, wiggle: float = 0.03):
    return make_dataset_rows(n, wiggle=wiggle)


def _tier1_params(samples, *, feature_names=None, targets=TARGET_NAMES):
    bundle = train_tier1(
        samples,
        horizon=1,
        feature_names=feature_names or FEATURE_NAMES,
        targets=targets,
        holdout_fraction=0.2,
    )
    return {
        "feature_names": list(bundle.feature_names),
        "dropped": list(bundle.dropped_features),
        "targets": {
            name: {
                "feature_names": list(model.feature_names),
                "model": model.model.to_dict(),
            }
            for name, model in sorted(bundle.targets.items())
        },
    }


def test_truncation_invariance_holds_for_the_real_pipeline():
    rows = _rows(240)
    result = truncation_invariance(
        rows,
        horizon=1,
        targets=TARGET_NAMES,
        sample_size=24,
        seed=7,
    )
    assert result["ok"] is True, result["violations"]
    assert result["n"] == 24


def test_truncation_catches_a_lookahead_feature_at_every_strength():
    rows = _rows(240)
    for correlation in (0.5, 0.2, 0.1, 0.05):
        def build(prefix, rho=correlation):
            samples = build_samples(prefix, horizon=1)
            # The realised forward return is only available after row i.
            values = [sample.targets.get("fwd_return") for sample in samples]
            return tuple(
                Sample(
                    timestamp=sample.timestamp,
                    features={**sample.features, "lookahead": values[index]},
                    targets=dict(sample.targets),
                )
                for index, sample in enumerate(samples)
            )

        result = truncation_invariance(
            rows,
            horizon=1,
            targets=("fwd_return",),
            extra_features=("lookahead",),
            sample_size=16,
            seed=3,
            build=build,
        )
        assert result["ok"] is False, correlation
        assert any(
            violation.get("name") == "lookahead" for violation in result["violations"]
        ), (correlation, result["violations"])


def test_perturbation_invariance_holds_for_the_causal_trainer():
    samples = build_samples(_rows(260), horizon=1)
    result = perturbation_invariance(
        samples,
        fit=lambda values: _tier1_params(values),
        targets=TARGET_NAMES,
        seed=11,
        feature_names=FEATURE_NAMES,
    )
    assert result["ok"] is True, result["violations"]
    assert result["baseline_digest"] == result["perturbed_digest"]


def test_perturbation_catches_a_scaler_fit_on_the_whole_series():
    from rl_crypto.features import FEATURE_NAMES

    samples = build_samples(_rows(240), horizon=1)

    def full_data_scaler(values):
        matrix = np.asarray(
            [
                [sample.features[name] for name in FEATURE_NAMES]
                for sample in values
                if all(sample.features.get(name) is not None for name in FEATURE_NAMES)
            ],
            dtype=float,
        )
        return {"scaler": Standardizer().fit(matrix).to_dict()}

    result = perturbation_invariance(
        samples,
        fit=full_data_scaler,
        targets=TARGET_NAMES,
        seed=11,
        feature_names=FEATURE_NAMES,
    )
    assert result["ok"] is False
    assert "scaler.mean" in result["violations"][0]["path"]


def test_train_tier1_selects_features_on_the_training_slice_only():
    """A feature that is constant in train and varies only in the holdout is dropped."""
    rows = _rows(120)
    samples = build_samples(rows, horizon=1)
    n = len(samples)
    cut = int(n * 0.9)
    patched = []
    for index, sample in enumerate(samples):
        features = dict(sample.features)
        features["ret_6"] = 0.0 if index < cut else float(index)
        patched.append(
            Sample(
                timestamp=sample.timestamp,
                features=features,
                targets=dict(sample.targets),
            )
        )
    bundle = train_tier1(
        patched,
        horizon=1,
        feature_names=FEATURE_NAMES,
        targets=("fwd_return",),
        holdout_fraction=0.2,
    )
    model = bundle.targets.get("fwd_return")
    assert model is not None
    assert "ret_6" not in model.feature_names
    assert "ret_6" in bundle.dropped_features


def test_contract_violation_ignores_self_reported_fit_scope():
    payload = {
        "preprocessing": {"fit_scope": "all"},
        "selection": {"scope": "holdout"},
        "structural": {
            "ok": True,
            "truncation": {"ok": True, "violations": []},
            "perturbation": {"ok": True, "violations": []},
        },
    }
    assert contract_violation(payload) is None


def test_contract_violation_uses_the_structural_result():
    payload = {
        "preprocessing": {"fit_scope": "train"},
        "selection": {"scope": "pre-cutoff"},
        "structural": {
            "ok": False,
            "truncation": {
                "ok": False,
                "violation_count": 2,
                "violations": [{"kind": "feature", "name": "leak", "row": 9}],
            },
            "perturbation": {"ok": True, "violations": []},
        },
    }
    reason = contract_violation(payload)
    assert reason is not None
    assert "truncation invariance" in reason
    assert "leak" in reason


def test_contract_violation_reports_perturbation_failure():
    payload = {
        "structural": {
            "ok": False,
            "truncation": {"ok": True, "violations": []},
            "perturbation": {
                "ok": False,
                "violations": [{"path": "targets.fwd_return.model.scaler.mean"}],
            },
        }
    }
    reason = contract_violation(payload)
    assert reason is not None
    assert "perturbation invariance" in reason
    assert "scaler.mean" in reason


def test_the_structural_module_is_part_of_the_protected_contract():
    from rl_crypto.experiments import PROTECTED_FILES

    assert "src/rl_crypto/structural.py" in PROTECTED_FILES
    assert (Path(__file__).resolve().parents[1] / "src/rl_crypto/structural.py").exists()
