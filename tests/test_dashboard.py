"""Offline tests for the README dashboard renderer.

The dashboard is generated from the committed cache and written by the merge job
on every cache commit, so these tests pin the two things that matter: the table
reflects the cache (including the legacy pipeline/2 summaries that carry no
verdict), and a regeneration is idempotent outside the markers.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TOOL = ROOT / "tools" / "render_dashboard.py"
NOW = datetime(2026, 10, 23, 20, 30, tzinfo=UTC)


def _load():
    spec = importlib.util.spec_from_file_location("render_dashboard", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cache(root: Path, *, withheld: bool = False, verdicts: bool = False) -> None:
    """A one-ticker cache: legacy summary, bundle with holdout verdicts, live row."""
    data = root / "data" / "eth-usd"
    data.mkdir(parents=True)
    # pipeline/2 summary: in-sample metrics only, no `trusted` key.
    targets = {
        "fwd_return": {"mae": 1.0, "r2": 0.0, "rmse": 1.0},
        "fwd_direction": {"accuracy": 0.5, "base_rate": 0.5, "log_loss": 0.7},
        "fwd_volatility": {"mae": 1.0, "r2": 0.1, "rmse": 1.0},
        "fwd_volume_ratio": {"mae": 1.0, "r2": 0.2, "rmse": 1.0},
    }
    if verdicts:
        targets = {
            "fwd_return": {"trusted": False},
            "fwd_direction": {"trusted": False},
            "fwd_volatility": {"trusted": True},
            "fwd_volume_ratio": {"trusted": True},
        }
    (data / "summary.json").write_text(
        json.dumps(
            {
                "schema": "rl-crypto/pipeline/3" if verdicts else "rl-crypto/pipeline/2",
                "ticker": "ETH/USD",
                "generated_at": "2026-10-23T20:20:00Z",
                "archive": {"rows": 1725, "spans": ["2026-08-12", "2026-10-23"]},
                "live": {"price": 2392.11},
                "targets": targets,
            }
        ),
        encoding="utf-8",
    )
    live = {
        "generated_at": "2026-10-23T20:15:00Z",
        "market": {"price": 2392.11},
        "predictions": {
            "signal": None if withheld else 1,
            "confidence": None if withheld else 0.4234,
            "probabilities": {"fwd_direction": 0.5123},
        },
    }
    (data / "live.jsonl").write_text(
        json.dumps({"generated_at": "2026-10-23T19:45:00Z"}) + "\n" + json.dumps(live) + "\n",
        encoding="utf-8",
    )
    models = root / "models"
    models.mkdir()
    (models / "eth-usd.json").write_text(
        json.dumps(
            {
                "feature_version": "rl-crypto/features/2",
                "trained_at": "2026-10-23T18:00:00Z",
                "targets": {
                    "fwd_return": {"holdout": {"beats_baseline": False}},
                    "fwd_direction": {"holdout": {"beats_baseline": False}},
                    "fwd_volatility": {"holdout": {"beats_baseline": True}},
                    "fwd_volume_ratio": {"holdout": {"beats_baseline": True}},
                },
            }
        ),
        encoding="utf-8",
    )


def _block(root: Path) -> str:
    tool = _load()
    return tool.render(tool.collect_rows(root), now=NOW, cache_bytes=tool._cache_bytes(root))


def test_dashboard_has_a_row_with_the_cache_state(tmp_path):
    _write_cache(tmp_path)
    block = _block(tmp_path)

    assert "**1 tickers**" in block
    assert "as of 2026-10-23T20:30Z" in block
    assert "| ETH/USD | $2,392.11 | 1 | 0.4234 | 0.512 | 1,725 · 2026-08-12 → 2026-10-23 | 2/4 | 20:15Z |" in block
    # freshness comes from the bundle's trained_at, not the summary run
    assert "trained 2026-10-23T18:00:00Z (2h ago)" in block


def test_legacy_summary_falls_back_to_the_bundle_holdout(tmp_path):
    """pipeline/2 summaries have no `trusted`; the bundle still knows."""
    _write_cache(tmp_path)
    assert "| 2/4 |" in _block(tmp_path)


def test_summary_verdicts_win_when_present(tmp_path):
    """pipeline/3 summaries carry the verdict; use it rather than the bundle."""
    _write_cache(tmp_path, verdicts=True)
    tool = _load()
    rows = tool.collect_rows(tmp_path)
    assert rows[0]["trusted"] == 2
    assert rows[0]["targets"] == 4


def test_withheld_signal_renders_as_dash_and_na(tmp_path):
    _write_cache(tmp_path, withheld=True)
    assert "| — | n/a |" in _block(tmp_path)


def test_replacement_is_idempotent_and_preserves_handwritten_text(tmp_path):
    tool = _load()
    readme = f"before\n{tool.START}\nold\n{tool.END}\nafter\n"

    once = tool.replace_block(readme, "one\ntwo")
    twice = tool.replace_block(once, "one\ntwo")

    assert once == twice
    assert once.startswith("before\n" + tool.START + "\none\ntwo\n")
    assert once.endswith(tool.END + "\nafter\n")
    assert "old" not in once


def test_missing_markers_fail_loudly():
    tool = _load()
    try:
        tool.replace_block("no dashboard here", "x")
    except ValueError as exc:
        assert "markers" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing markers must raise")


def test_cli_writes_the_readme_block(tmp_path):
    tool = _load()
    _write_cache(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(f"# t\n\n{tool.START}\n{tool.END}\n\nfooter\n", encoding="utf-8")

    assert tool.main(["--root", str(tmp_path), "--now", "2026-10-23T20:30:00Z"]) == 0
    first = readme.read_text(encoding="utf-8")
    assert "| ETH/USD |" in first
    assert first.endswith("footer\n")

    # Re-running at the same instant must not churn the file.
    assert tool.main(["--root", str(tmp_path), "--now", "2026-10-23T20:30:00Z"]) == 0
    assert readme.read_text(encoding="utf-8") == first


def test_the_real_readme_has_the_dashboard_markers():
    """The production workflow rewrites this block; losing the markers would make
    every merge job warn and the README go stale."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count("<!-- dashboard:start -->") == 1
    assert readme.count("<!-- dashboard:end -->") == 1
    assert readme.index("<!-- dashboard:start -->") < readme.index("<!-- dashboard:end -->")
