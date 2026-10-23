#!/usr/bin/env python3
"""Deprecated shim — conflict resolution now lives behind ``rl-pipeline conflicts``.

Kept for one release because ``tools/record_live.sh`` invokes this exact path
after a rejected push.
"""

from __future__ import annotations

import sys

from rl_crypto.pipeline.cli import main as pipeline_main

_DEPRECATION = (
    "deprecated: use `rl-pipeline conflicts` (was tools/resolve_cache_conflicts.py)"
)


def main(argv: list[str] | None = None) -> int:
    print(_DEPRECATION, file=sys.stderr)
    rest = sys.argv[1:] if argv is None else argv
    return pipeline_main(["conflicts", *rest])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
