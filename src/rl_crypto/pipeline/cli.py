"""The ``rl-pipeline`` command line: the production runners for the cache.

One entry point over two runners and their maintenance commands::

    rl-pipeline run        grow each ticker's archive, retrain, append a live record
    rl-pipeline live       append one live record per ticker (no training)
    rl-pipeline conflicts  resolve data/ rebase conflicts by merging both sides
    rl-pipeline sync       fast-forward the checkout to the tracking remote
    rl-pipeline version

``rl-crypto`` stays the analysis/query CLI (unified record, dataset, features,
train, predict); ``rl-pipeline`` is what the scheduled runners invoke. Both share
the same library, the same ``--root`` semantics (default: the current working
directory, overridable with ``RL_CRYPTO_ROOT``) and the same exit codes.

``run`` and ``live`` finish with ``os._exit``: a per-fetch timeout can leave a
worker thread behind, and the production runners must not be held alive by it
after the work is flushed. The maintenance commands exit normally.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from .. import __version__
from . import conflicts, live, run, sync

#: Commands whose abandoned fetch workers must not keep the process alive.
HARD_EXIT_COMMANDS = frozenset({"run", "live"})


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"rl-pipeline {__version__}")
    return 0


def _add_command(
    sub: argparse._SubParsersAction,
    name: str,
    summary: str,
    module: Any,
) -> argparse.ArgumentParser:
    """Wire one subcommand: its own flags, its module docstring as help text."""
    parser = sub.add_parser(
        name,
        help=summary,
        description=getattr(module, "__doc__", None),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    module.add_arguments(parser)
    parser.set_defaults(func=module.run)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl-pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"rl-pipeline {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_command(
        sub, "run", "grow the archive, retrain, and append one live record", run
    )
    _add_command(sub, "live", "append one live record per ticker (no training)", live)
    _add_command(sub, "conflicts", "resolve data/ rebase conflicts by merging", conflicts)
    _add_command(sub, "sync", "fast-forward the checkout to the tracking remote", sync)

    p = sub.add_parser("version", help="print the command version")
    p.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = int(args.func(args) or 0)
    except ImportError as exc:  # a missing upstream dependency
        print(
            f"error: could not import an upstream pipeline ({exc}). "
            "Run inside the dev shell: `nix develop` (or `uv sync`).",
            file=sys.stderr,
        )
        code = 1
    except KeyboardInterrupt:  # pragma: no cover
        code = 130

    if args.command in HARD_EXIT_COMMANDS:
        # Flushed work is done; abandoned fetch workers must not keep us alive.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__: list[str] = ["build_parser", "main"]
