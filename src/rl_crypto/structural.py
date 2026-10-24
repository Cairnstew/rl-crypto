"""Structural causality checks for the measurement contract.

The verdict used to trust what a lab payload *said* about itself: a
``preprocessing.fit_scope`` of ``"train"`` and a ``selection.scope`` of
``"pre-cutoff"``. Those are self-reports. This module replaces them with checks
that exercise the pipeline instead of asking it:

* **Truncation invariance** — a feature anchored at row ``i`` must be computable
  from the rows available at ``i``, and a target anchored at ``i`` from the rows
  available when its label completes. Recomputing both from a truncated prefix
  must reproduce the full-series value exactly. A look-ahead feature cannot: it
  needs a bar the prefix does not have, so the two values differ (or the
  truncated value is unavailable).
* **Perturbation invariance** — re-fitting with the newest holdout rows
  perturbed must leave the fitted preprocessing statistics, the selected
  features and the model coefficients byte-identical. A scaler fitted on the
  full data, or feature selection that looked at the holdout, changes when the
  holdout changes.

Both checks are pure, deterministic and offline. A failure is a measurement
fault, not a modelling result, so the verdict reports ``invalid``.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .features import DEFAULT_INTERVAL_SECONDS, FEATURE_NAMES
from .models import DatasetRow
from .samples import Sample, build_samples
from .targets import TARGET_KINDS, label_purge

#: Features injected by a control or a calibration. They are deliberately not
#: part of the causal feature contract, so the automatic scan skips them; a
#: caller that wants to audit one passes it as ``extra_features``.
SYNTHETIC_FEATURES: frozenset[str] = frozenset(
    {"planted_signal", "leak_correlation", "leak_next_bar_return"}
)

SampleBuilder = Callable[[Sequence[DatasetRow]], Sequence[Sample]]
Fit = Callable[[Sequence[Sample]], Any]


def _pack(value: Any) -> bytes | None:
    """A byte representation of one scalar, for exact comparison.

    ``struct.pack`` is used rather than ``==`` so ``-0.0`` and ``0.0`` are
    distinguished and a NaN compares equal to itself (same payload), matching
    the "byte-identical" requirement.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return b"bool:1" if value else b"bool:0"
    try:
        return struct.pack(">d", float(value))
    except (TypeError, ValueError):
        return repr(value).encode("utf-8")


def _same(left: Any, right: Any) -> bool:
    return _pack(left) == _pack(right)


def _default_build(
    rows: Sequence[DatasetRow], *, horizon: int, interval_seconds: int
) -> tuple[Sample, ...]:
    return build_samples(rows, horizon=horizon, interval_seconds=interval_seconds)


def _reach(targets: Sequence[str], horizon: int) -> int:
    return max((label_purge(target, horizon) for target in targets), default=max(0, horizon))


def truncation_invariance(
    rows: Sequence[DatasetRow],
    *,
    horizon: int,
    targets: Sequence[str],
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    feature_names: Sequence[str] | None = None,
    extra_features: Sequence[str] = (),
    sample_size: int = 32,
    seed: int = 0,
    build: SampleBuilder | None = None,
) -> dict[str, Any]:
    """Check that no feature or target reads a bar it could not have known.

    For each sampled row ``i`` the full-series sample is compared with one
    rebuilt from ``rows[: i + 1]`` (features) and ``rows[: i + reach + 1]``
    (targets). Any difference — including a full value where the truncated
    build has ``None`` — is a violation.

    ``build`` lets a caller audit a transformed pipeline (a planted signal, an
    injected control feature). The default is the production
    :func:`rl_crypto.samples.build_samples`.
    """
    builder = build or (
        lambda prefix: _default_build(
            prefix, horizon=horizon, interval_seconds=interval_seconds
        )
    )
    full = builder(rows)
    if not full:
        return {
            "ok": True,
            "checked_rows": [],
            "n": 0,
            "features": [],
            "targets": list(targets),
            "violations": [],
        }
    # Audit the declared names *and* any feature the pipeline actually emits
    # but did not declare, so a run that adds a feature cannot hide it from the
    # check by forgetting to update its own list.
    present = sorted({name for sample in full for name in sample.features})
    names = tuple(
        dict.fromkeys(
            [
                *(feature_names if feature_names is not None else FEATURE_NAMES),
                *present,
                *extra_features,
            ]
        )
    )
    # Exclude synthetic control features unless the caller explicitly audits
    # them (extra_features), so a planted power signal is not a false positive.
    audited = tuple(
        name for name in names if name in extra_features or name not in SYNTHETIC_FEATURES
    )
    n = len(rows)
    # Leave enough history for the longest feature window; a prefix that cannot
    # produce the row is not a meaningful comparison.
    minimum = min(24, max(0, n - 1))
    candidates = list(range(minimum, n))
    if not candidates:
        candidates = list(range(n))
    rng = np.random.default_rng(int(seed))
    take = min(int(sample_size), len(candidates))
    chosen = sorted(int(index) for index in rng.choice(candidates, size=take, replace=False))
    reach = _reach(targets, horizon)
    violations: list[dict[str, Any]] = []
    for index in chosen:
        prefix = rows[: index + 1]
        try:
            rebuilt = builder(prefix)
        except (IndexError, ValueError) as exc:  # a pipeline that cannot build a prefix
            violations.append(
                {
                    "kind": "feature",
                    "row": index,
                    "timestamp": rows[index].timestamp,
                    "reason": f"prefix build failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        if index >= len(rebuilt):
            violations.append(
                {
                    "kind": "feature",
                    "row": index,
                    "timestamp": rows[index].timestamp,
                    "reason": "prefix build produced too few rows",
                }
            )
            continue
        for name in audited:
            if not _same(full[index].features.get(name), rebuilt[index].features.get(name)):
                violations.append(
                    {
                        "kind": "feature",
                        "name": name,
                        "row": index,
                        "timestamp": rows[index].timestamp,
                        "full": full[index].features.get(name),
                        "truncated": rebuilt[index].features.get(name),
                    }
                )
        # A target anchored at i is available once its widest label reach has
        # completed; truncate exactly there.
        target_prefix = rows[: min(n, index + reach + 1)]
        try:
            rebuilt_targets = builder(target_prefix)
        except (IndexError, ValueError) as exc:
            violations.append(
                {
                    "kind": "target",
                    "row": index,
                    "timestamp": rows[index].timestamp,
                    "reason": f"target prefix build failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        if index >= len(rebuilt_targets):
            violations.append(
                {
                    "kind": "target",
                    "row": index,
                    "timestamp": rows[index].timestamp,
                    "reason": "target prefix build produced too few rows",
                }
            )
            continue
        for name in targets:
            if not _same(
                full[index].targets.get(name), rebuilt_targets[index].targets.get(name)
            ):
                violations.append(
                    {
                        "kind": "target",
                        "name": name,
                        "row": index,
                        "timestamp": rows[index].timestamp,
                        "full": full[index].targets.get(name),
                        "truncated": rebuilt_targets[index].targets.get(name),
                    }
                )
    return {
        "ok": not violations,
        "checked_rows": chosen,
        "n": len(chosen),
        "features": list(audited),
        "targets": list(targets),
        "reach": reach,
        "violations": violations[:20],
        "violation_count": len(violations),
    }


def holdout_start_index(
    samples: Sequence[Sample],
    *,
    targets: Sequence[str],
    feature_names: Sequence[str] | None,
    holdout_fraction: float,
) -> int:
    """First row that every target's chronological holdout contains.

    ``train_tier1`` reserves the newest ``holdout_fraction`` of each target's
    *usable* rows (complete features and a non-``None`` label), which is later
    than ``len(samples) * (1 - holdout_fraction)`` because early rows lack long
    windows and the final rows lack labels. Perturbing a row that is still in
    any target's training slice would make the check meaningless, so the common
    holdout starts at the **latest** per-target start.
    """
    def complete(sample: Sample) -> bool:
        if not feature_names:
            return True
        return all(sample.features.get(name) is not None for name in feature_names)

    starts: list[int] = []
    for target in targets:
        usable = [
            index
            for index, sample in enumerate(samples)
            if complete(sample) and sample.targets.get(target) is not None
        ]
        if len(usable) < 2:
            continue
        n_holdout = int(len(usable) * float(holdout_fraction))
        if n_holdout <= 0:
            continue
        starts.append(usable[len(usable) - n_holdout])
    return max(starts) if starts else len(samples)


def _perturb_holdout(
    samples: Sequence[Sample],
    *,
    cut: int,
    seed: int,
    noise: float,
) -> tuple[Sample, ...]:
    """Perturb rows from ``cut`` onwards with noise and replacement.

    Non-``None`` feature and regression-target values get Gaussian noise scaled
    by the training slice's own spread; classification labels are replaced at
    random. Every third row is replaced wholesale by another holdout row's
    values. Completeness (``None`` stays ``None``) is preserved so the check
    isolates preprocessing/selection from a change in which rows are usable.
    """
    n = len(samples)
    if cut <= 0 or cut >= n:
        return tuple(samples)
    train = samples[:cut]
    holdout = samples[cut:]
    rng = np.random.default_rng(int(seed))

    feature_scale: dict[str, float] = {}
    for name in {key for sample in train for key in sample.features}:
        values = [
            float(sample.features[name])
            for sample in train
            if sample.features.get(name) is not None
        ]
        feature_scale[name] = float(np.std(values)) if len(values) > 1 else 0.0
    target_scale: dict[str, float] = {}
    for name in {key for sample in train for key in sample.targets}:
        values = [
            float(sample.targets[name])
            for sample in train
            if sample.targets.get(name) is not None
        ]
        target_scale[name] = float(np.std(values)) if len(values) > 1 else 0.0

    # A deterministic replacement order over the holdout rows.
    order = list(range(len(holdout)))
    rng.shuffle(order)
    perturbed: list[Sample] = list(samples[:cut])
    for position, sample in enumerate(holdout):
        replace = position % 3 == 2
        source = holdout[order[position]] if replace else sample
        features: dict[str, float | None] = {}
        for name, value in sample.features.items():
            if value is None:
                features[name] = None
                continue
            raw = source.features.get(name) if replace else value
            base = float(raw) if raw is not None else float(value)
            scale = feature_scale.get(name, 0.0) or 0.0
            features[name] = base + float(rng.normal(scale=noise * scale)) if scale else base
        new_targets: dict[str, float | None] = {}
        for name, value in sample.targets.items():
            if value is None:
                new_targets[name] = None
                continue
            kind = TARGET_KINDS.get(name, "regression")
            if kind == "classification":
                # Replace with a random class; this cannot introduce a new
                # category and cannot change the row count.
                new_targets[name] = 0.0 if rng.random() < 0.5 else 1.0
                continue
            raw = source.targets.get(name) if replace else value
            base = float(raw) if raw is not None else float(value)
            scale = target_scale.get(name, 0.0) or 0.0
            new_targets[name] = base + float(rng.normal(scale=noise * scale)) if scale else base
        perturbed.append(
            Sample(timestamp=sample.timestamp, features=features, targets=new_targets)
        )
    return tuple(perturbed)


def _first_difference(left: Any, right: Any, path: str = "") -> dict[str, Any] | None:
    """The first structural difference between two JSON-like parameter trees."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key not in left or key not in right:
                return {
                    "path": f"{path}.{key}".lstrip("."),
                    "left": left.get(key),
                    "right": right.get(key),
                }
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {"path": f"{path}.length", "left": len(left), "right": len(right)}
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if _pack(left) != _pack(right):
        return {"path": path, "left": left, "right": right}
    return None


def perturbation_invariance(
    samples: Sequence[Sample],
    *,
    fit: Fit,
    targets: Sequence[str],
    holdout_fraction: float = 0.2,
    seed: int = 0,
    noise: float = 0.5,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Re-fit with the holdout perturbed and require byte-identical parameters.

    ``fit`` returns a canonical, JSON-serialisable parameter tree (scaler
    statistics, selected feature names, coefficients). It must be a pure
    function of the samples it is given. A pipeline that fits preprocessing or
    feature selection on the holdout will return different parameters once the
    holdout is perturbed.
    """
    baseline = fit(samples)
    cut = holdout_start_index(
        samples,
        targets=targets,
        feature_names=feature_names,
        holdout_fraction=holdout_fraction,
    )
    perturbed_samples = _perturb_holdout(samples, cut=cut, seed=seed, noise=noise)
    perturbed = fit(perturbed_samples)
    difference = _first_difference(baseline, perturbed)
    baseline_json = json.dumps(baseline, sort_keys=True, default=str)
    perturbed_json = json.dumps(perturbed, sort_keys=True, default=str)
    return {
        "ok": difference is None,
        "n_samples": len(samples),
        "n_holdout": max(0, len(samples) - cut),
        "holdout_start_index": cut,
        "seed": int(seed),
        "baseline_digest": _digest(baseline_json),
        "perturbed_digest": _digest(perturbed_json),
        "violations": [] if difference is None else [difference],
    }


def _digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "SYNTHETIC_FEATURES",
    "SampleBuilder",
    "perturbation_invariance",
    "truncation_invariance",
]
