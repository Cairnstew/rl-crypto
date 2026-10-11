"""rl-crypto's tiered ML pipeline.

The pipeline is deliberately tiered so each layer can be reasoned about (and
replaced) on its own:

* **Tier 0 — features** (:mod:`rl_crypto.features`): deterministic, causal
  transforms of the joined dataset. No learning, one definition for train and
  serve.
* **Tier 1 — base predictors** (:mod:`rl_crypto.ml.tier1`): one structured model
  per target metric, fit on historical samples, serialised to JSON.
* **Tier 2 — composition** (:mod:`rl_crypto.ml.tier2`): fuses the Tier-1 outputs
  into a single :class:`~rl_crypto.ml.types.Outlook`.
* **Tier 3 — RL (future)**: the observation vector is the Tier-0 features plus
  the Tier-2 outlook; :meth:`~rl_crypto.ml.types.Outlook.as_features` is the
  flat seam for it. No RL code lives here yet by design.

Typical flow::

    from rl_crypto.ml import train_tier1, save_bundle, outlook_for_dataset

    bundle = train_tier1(build_dataset_samples(dataset, horizon=1), horizon=1)
    save_bundle(bundle, "models/eth-usd.json")

    outlook = outlook_for_dataset(recent_dataset, bundle)
    print(outlook.signal, outlook.predictions, outlook.as_features())

Importing this package pulls in numpy; the core :mod:`rl_crypto` package does
not, so ``import rl_crypto`` stays cheap.
"""

from __future__ import annotations

from . import linear as _linear  # noqa: F401  (register the reference estimators)
from .base import (
    DEFAULT_MODEL,
    Model,
    Standardizer,
    TaskKind,
    make_model,
    model_from_dict,
    model_names,
    models_for,
)
from .serve import (
    DEFAULT_WINDOW,
    attach_predictions,
    load_bundle_for,
    outlook_for_dataset,
    outlook_for_rows,
    predict_live,
)
from .store import DEFAULT_MODEL_DIR, default_bundle_path, load_bundle, save_bundle
from .tier1 import MIN_TRAIN_SAMPLES, Tier1Bundle, Tier1Model, train_tier1
from .tier2 import compose_outlook
from .types import ML_SCHEMA, Outlook, Prediction

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_WINDOW",
    "MIN_TRAIN_SAMPLES",
    "ML_SCHEMA",
    "Model",
    "Outlook",
    "Prediction",
    "Standardizer",
    "TaskKind",
    "Tier1Bundle",
    "Tier1Model",
    "attach_predictions",
    "compose_outlook",
    "default_bundle_path",
    "load_bundle",
    "load_bundle_for",
    "make_model",
    "model_from_dict",
    "model_names",
    "models_for",
    "outlook_for_dataset",
    "outlook_for_rows",
    "predict_live",
    "save_bundle",
    "train_tier1",
]
