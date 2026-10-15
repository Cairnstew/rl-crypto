"""Model-bundle persistence — JSON in, JSON out.

A bundle is a single JSON document (schema ``rl-crypto/ml/1``) holding the
feature contract, the horizon, and every fitted Tier-1 model with its metrics.
It is written with :func:`rl_crypto.export.write_json` so it stays diffable and
reviewable in git, and it round-trips with no bespoke encoder — there is no
pickle to go stale.

The bundle also records the :data:`~rl_crypto.features.FEATURE_VERSION` it was
trained against. :func:`load_bundle` re-checks it on the way in and, on a
mismatch, flags the bundle instead of raising: a bundle fitted against one
feature definition and served against another produces numbers that look fine
and are meaningless, but the live timer records every 30 minutes and must not
crash on a stale artifact. A flagged bundle has every target marked untrusted,
so it predicts nothing and says why.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..export import write_json
from ..features import FEATURE_VERSION
from .tier1 import Tier1Bundle
from .types import ML_SCHEMA

#: Where trained bundles live by default (committed under the repo's ``models/``).
DEFAULT_MODEL_DIR = Path("models")


def default_bundle_path(ticker: str, *, directory: str | Path = DEFAULT_MODEL_DIR) -> Path:
    """The conventional bundle path for ``ticker`` (``ETH/USD`` -> ``eth-usd.json``)."""
    slug = ticker.strip().replace("/", "-").replace(" ", "-").lower()
    return Path(directory) / f"{slug}.json"


def save_bundle(bundle: Tier1Bundle, path: str | Path = DEFAULT_MODEL_DIR / "bundle.json") -> Path:
    """Write ``bundle`` to ``path`` as pretty JSON."""
    return write_json(path, bundle.to_dict())


def load_bundle(path: str | Path) -> Tier1Bundle:
    """Read a bundle back, rejecting an unknown schema up front.

    A ``feature_version`` that no longer matches the running code marks the
    bundle rather than raising — see the module docstring.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != ML_SCHEMA:
        raise ValueError(f"unsupported bundle schema {schema!r} (expected {ML_SCHEMA!r})")
    bundle = Tier1Bundle.from_dict(payload)
    trained_version = bundle.feature_version
    if trained_version != FEATURE_VERSION:
        bundle.feature_version_mismatch = True
        bundle.notes = (
            *bundle.notes,
            f"feature version mismatch: bundle trained against {trained_version or 'unknown'}, "
            f"code is {FEATURE_VERSION}; predictions are suppressed until retrained",
        )
    return bundle


__all__ = ["DEFAULT_MODEL_DIR", "default_bundle_path", "load_bundle", "save_bundle"]
