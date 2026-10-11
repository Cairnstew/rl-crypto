"""Model-bundle persistence — JSON in, JSON out.

A bundle is a single JSON document (schema ``rl-crypto/ml/1``) holding the
feature contract, the horizon, and every fitted Tier-1 model with its metrics.
It is written with :func:`rl_crypto.export.write_json` so it stays diffable and
reviewable in git, and it round-trips with no bespoke encoder — there is no
pickle to go stale.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..export import write_json
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
    """Read a bundle back, rejecting an unknown schema up front."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != ML_SCHEMA:
        raise ValueError(f"unsupported bundle schema {schema!r} (expected {ML_SCHEMA!r})")
    return Tier1Bundle.from_dict(payload)


__all__ = ["DEFAULT_MODEL_DIR", "default_bundle_path", "load_bundle", "save_bundle"]
