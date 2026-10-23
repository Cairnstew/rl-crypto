#!/usr/bin/env python3
"""Fast-forward the cache checkout to the tracking remote.

The committed ``data/``/``models/`` cache is only as fresh as the checkout, and
the scheduled runners push every couple of hours. This command does exactly what
the notebook's first cell does — ``git fetch`` plus a ``--ff-only`` pull — but
from the CLI, so a runner (or a human) can refresh before reading or recording.

It never forces: a busy working tree or a diverged branch refuses and reports
the reason, leaving the checkout untouched::

    rl-pipeline sync
    rl-pipeline sync --no-pull     # fetch and report, do not fast-forward
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(os.environ.get("RL_CRYPTO_ROOT") or Path.cwd()).resolve()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the cache-sync flags (shared by the CLI and the legacy shim)."""
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="fetch and report only; do not fast-forward the checkout",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="per-git-call wall-clock cap (s)"
    )
    parser.add_argument("--root", default=None, help="repo root to sync (default: the cwd)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rl-pipeline sync",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Sync the cache and return 0 when it is current afterwards."""
    from ..cache import sync_cache

    sync = sync_cache(Path(args.root or ROOT), pull=not args.no_pull, timeout=args.timeout)
    print(sync.summary())
    for note in sync.notes:
        print(f" - {note}")
    # Non-zero only when the cache is still behind: dirty is not an error.
    return 1 if sync.cache_stale else 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
