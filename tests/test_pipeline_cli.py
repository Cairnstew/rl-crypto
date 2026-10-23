"""The ``rl-pipeline`` command surface: shape of the CLI, not the runners.

The runners themselves are covered by ``test_run_pipeline``/``test_record_live``/
``test_resolve_conflicts``; these tests lock the entry point the workflows and the
timer script depend on — the command names, the parity of every documented flag,
and the hard-exit contract that keeps abandoned fetch workers from holding the
process alive.
"""

from __future__ import annotations

from typing import Any

import pytest

from rl_crypto.pipeline import cli


def _subcommands(parser: Any) -> set[str]:
    """Every registered subcommand name (argparse hides them in a _SubParsersAction)."""
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and "run" in choices:
            return set(choices)
    raise AssertionError("no subcommands registered")


def test_commands_are_exposed():
    assert _subcommands(cli.build_parser()) == {
        "run",
        "live",
        "conflicts",
        "sync",
        "version",
    }


def test_run_flags_match_the_runner():
    args = cli.build_parser().parse_args(
        ["run", "--ticker", "ETH/USD", "--history-days", "7", "--retrain-only"]
    )
    assert args.command == "run"
    assert args.ticker == "ETH/USD"
    assert args.history_days == 7
    assert args.retrain_only is True
    # Defaults the workflows rely on must not drift.
    assert args.timeframe == "1h"
    assert args.horizon == 1
    assert args.fetch_timeout == 300.0


def test_live_flags_match_the_recorder():
    args = cli.build_parser().parse_args(
        ["live", "--tickers", "BTC/USD", "ETH/USD", "--live-cap", "5", "--tail-hours", "2"]
    )
    assert args.tickers == ["BTC/USD", "ETH/USD"]
    assert args.live_cap == 5
    assert args.tail_hours == 2
    assert args.window == 48
    assert args.fetch_timeout == 120.0


def test_maintenance_flags():
    conflicts = cli.build_parser().parse_args(["conflicts", "--dry-run"])
    assert conflicts.dry_run is True
    sync = cli.build_parser().parse_args(["sync", "--no-pull"])
    assert sync.no_pull is True
    assert sync.timeout == 30.0


def test_root_is_optional_and_honoured_when_given():
    """``--root`` defaults to the working directory at run time, not to a path
    baked from ``__file__`` (which is a store path under a hermetic install)."""
    assert cli.build_parser().parse_args(["conflicts"]).root is None
    assert cli.build_parser().parse_args(["conflicts", "--root", "/tmp/x"]).root == "/tmp/x"


def test_version_prints_and_returns_zero(capsys):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.startswith("rl-pipeline ")


def test_hard_exit_is_limited_to_the_runners():
    """Only the runners may hard-exit; maintenance commands must return."""
    assert {"run", "live"} == cli.HARD_EXIT_COMMANDS


def test_shims_delegate_to_the_right_subcommand(monkeypatch):
    """The deprecated tools/*.py shims must keep the timer working."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "tools"
    cases = {
        "run_pipeline.py": "run",
        "record_live.py": "live",
        "resolve_cache_conflicts.py": "conflicts",
    }
    for filename, command in cases.items():
        spec = importlib.util.spec_from_file_location(f"shim_{command}", root / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        calls: list[list[str]] = []

        def _record(argv: list[str], sink: list[list[str]] = calls) -> int:
            sink.append(argv)
            return 0

        monkeypatch.setattr(module, "pipeline_main", _record)
        assert module.main(["--flag"]) == 0
        assert calls == [[command, "--flag"]]


def test_unknown_command_exits_with_a_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["nope"])
    assert excinfo.value.code == 2
