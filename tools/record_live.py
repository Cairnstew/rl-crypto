#!/usr/bin/env python3
"""Deprecated shim — the live recorder now lives behind ``rl-pipeline live``.

Kept for one release because the local systemd timer invokes this exact path
(``tools/record_live.sh`` -> ``tools/record_live.py``).
"""

from __future__ import annotations

import sys

from rl_crypto.pipeline.cli import main as pipeline_main

_DEPRECATION = "deprecated: use `rl-pipeline live …` (was tools/record_live.py)"


def main(argv: list[str] | None = None) -> int:
    print(_DEPRECATION, file=sys.stderr)
    rest = sys.argv[1:] if argv is None else argv
    return pipeline_main(["live", *rest])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
