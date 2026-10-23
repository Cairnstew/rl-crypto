"""Production runners for the committed cache, behind the ``rl-pipeline`` command.

``rl_crypto`` itself is the analysis/query surface (fuse, dataset, features,
train, predict). This subpackage holds the scheduled machinery instead — the
runners that grow ``data/`` and ``models/``, plus the tooling that keeps
concurrent cache writers from clobbering each other:

* :func:`~rl_crypto.pipeline.run.run_pipeline` — archive → train → live record
* :func:`~rl_crypto.pipeline.live.record_live` — frequent live points only
* :func:`~rl_crypto.pipeline.conflicts.resolve` — union ``data/`` rebase conflicts
* :mod:`rl_crypto.pipeline.sync` — fast-forward the checkout to the remote

Everything is also reachable from the shell as ``rl-pipeline <command>``.
"""

from __future__ import annotations

from .conflicts import resolve
from .live import record_live
from .run import run_pipeline

__all__ = ["record_live", "resolve", "run_pipeline"]
