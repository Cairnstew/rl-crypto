#!/usr/bin/env python3
"""Deprecated shim — the production runner now lives behind ``rl-pipeline run``.

Kept for one release so an existing invocation (the local live timer, a shell
history, a pinned workflow) keeps working after the move to
``src/rl_crypto/pipeline/``.
"""

from __future__ import annotations

import sys

from rl_crypto.pipeline.cli import main as pipeline_main

_DEPRECATION = "deprecated: use `rl-pipeline run …` (was tools/run_pipeline.py)"


def main(argv: list[str] | None = None) -> int:
    print(_DEPRECATION, file=sys.stderr)
    rest = sys.argv[1:] if argv is None else argv
    return pipeline_main(["run", *rest])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
