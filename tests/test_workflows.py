"""Structural checks on the production workflows.

These lock down the invariants the single-scheduler, single-writer design
depends on: both cadences share one workflow (a second cron workflow proved
unreliable here), the matrix comes from one source of truth, matrix jobs never
push, the merge job is the only writer and retries a rejected push, and a fork
can narrow the watchlist with a repository variable instead of a code edit.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
from pathlib import Path

import yaml

from rl_crypto.cache import slug_for
from rl_crypto.tickers import TRACKED_TICKERS

_WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
_PRODUCTION = _WORKFLOWS / "production.yml"
_WATCHDOG = _WORKFLOWS / "watchdog.yml"


def _workflow(path: Path = _PRODUCTION) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(doc: dict) -> dict:
    # PyYAML parses the bare `on:` key as boolean True (YAML 1.1).
    return doc.get(True) or doc.get("on") or {}


def _runs(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


# --------------------------------------------------------------------------- shape


def test_one_workflow_covers_both_cadences():
    jobs = _workflow()["jobs"]
    assert set(jobs) >= {"prepare", "live", "train", "merge"}
    assert "mode == 'live'" in jobs["live"]["if"]
    assert "mode == 'train'" in jobs["train"]["if"]


def test_there_is_a_single_schedule():
    """Two scheduled workflows were unreliable on this repo; one scheduler only."""
    crons = [entry["cron"] for entry in _triggers(_workflow())["schedule"]]
    assert crons == ["*/30 * * * *"]


def test_matrix_comes_from_the_tracked_watchlist():
    """The matrix is built from tickers.py, never a second copy in the YAML."""
    jobs = _workflow()["jobs"]
    for name in ("live", "train"):
        assert jobs[name]["needs"] == "prepare"
        matrix = jobs[name]["strategy"]["matrix"]
        assert "entry" in matrix
        assert "needs.prepare.outputs" in matrix["entry"]
    assert "tickers.py" in _runs(jobs["prepare"])


def test_prepare_emits_artifact_safe_slugs():
    """A ticker like BTC/USD cannot appear in an artifact NAME (slash is
    rejected by upload-artifact), so the slug travels alongside it."""
    runs = _runs(_workflow()["jobs"]["prepare"])
    assert '"slug"' in runs
    # It must not import the real slug_for (that would drag in the package); it
    # reimplements the rule locally — so that local copy has to agree with it.
    assert "from rl_crypto.cache import" not in runs
    for ticker in TRACKED_TICKERS:
        assert slug_for(ticker) == ticker.replace("/", "-").replace(" ", "-").lower()


def test_artifacts_download_inside_the_checkout_dir():
    """download-artifact resolves `path` against the WORKSPACE ROOT, while the
    merge step runs with `working-directory: <checkout>`. If the download is not
    prefixed with the checkout dir, the copy silently finds nothing and the run
    reports "no cache changes" having done all the work.
    """
    steps = {s.get("name"): s for s in _workflow()["jobs"]["merge"]["steps"]}
    download = steps["download cache artifacts"]
    assert download["with"]["path"].startswith("rl-crypto/")

    merge = steps["merge caches into the checkout"]
    assert merge["working-directory"] == "rl-crypto"
    # ...and it must fail loudly rather than no-op if nothing arrived.
    assert "::error::" in merge["run"]
    assert "exit 1" in merge["run"]


def test_matrix_jobs_never_push():
    """Single-writer invariant: only the merge job may push."""
    jobs = _workflow()["jobs"]
    for name in ("prepare", "live", "train"):
        assert "git push" not in _runs(jobs[name]), f"{name} must not push"
    assert "git push" in _runs(jobs["merge"]), "merge must be the writer"
    for name in ("live", "train"):
        assert jobs[name]["strategy"].get("fail-fast") is False


def test_bounded_parallelism():
    jobs = _workflow()["jobs"]
    for name in ("live", "train"):
        assert jobs[name]["strategy"].get("max-parallel"), "unbounded parallelism is unsafe"


def test_concurrency_serialises_only_the_writers():
    """Queueing whole runs would drop live ticks behind a 30-minute train run."""
    jobs = _workflow()["jobs"]
    holders = {name: job.get("concurrency") for name, job in jobs.items() if job.get("concurrency")}
    assert set(holders) == {"merge"}
    assert holders["merge"]["group"] == "rl-crypto-cache"


def test_merge_runs_on_partial_failure():
    """If some tickers fail, the surviving caches must still be committed."""
    merge = _workflow()["jobs"]["merge"]
    assert set(merge["needs"]) == {"prepare", "live", "train"}
    assert "always()" in str(merge.get("if"))


def test_merge_push_retries_after_rebase():
    """A hand push or a leftover writer can land concurrently, so a rejected
    push must rebase, resolve the cache conflict and retry."""
    merge_runs = _runs(_workflow()["jobs"]["merge"])
    assert "git pull --rebase" in merge_runs
    assert "for attempt in" in merge_runs
    assert "pipeline/conflicts.py --root ." in merge_runs


def test_production_uses_the_pipeline_entry_point():
    """The runners live behind ``rl-pipeline``; the workflow must not drift back
    to the old script paths (which are only compatibility shims)."""
    text = _PRODUCTION.read_text(encoding="utf-8")
    assert "rl-pipeline live" in text
    assert "rl-pipeline run" in text
    assert "tools/run_pipeline.py" not in text
    assert "tools/record_live.py" not in text


# --------------------------------------------------------------------------- watchdog


def test_watchdog_dispatches_production_when_stale():
    doc = _workflow(_WATCHDOG)
    assert "actions" in (doc.get("permissions") or {})
    assert doc["permissions"]["actions"] == "write"
    runs = _runs(doc["jobs"]["heal"])
    assert "gh workflow run production.yml" in runs
    assert "MAX_AGE_MINUTES" in runs
    assert "commits?path=data" in runs


# --------------------------------------------------------------------------- prepare


def _run_prepare_script(
    requested: str = "",
    *,
    mode: str = "",
    watchlist: str = "",
    hour: str = "20",
    tmp_path: Path,
) -> tuple[str, str]:
    """Execute the real prepare step's script; return (GITHUB_OUTPUT, stdout)."""
    steps = _workflow()["jobs"]["prepare"]["steps"]
    run = next(step["run"] for step in steps if step.get("id") == "plan")
    source = re.search(r"python3 - <<'PY'\n(.*?)\nPY", run, re.S)
    assert source, "prepare step no longer uses the expected heredoc"
    script = source.group(1)

    out_file = tmp_path / "github_output"
    out_file.write_text("", encoding="utf-8")
    env = {
        "REQUESTED": requested,
        "REQUESTED_MODE": mode,
        "WATCHLIST": watchlist,
        "SCHEDULE_HOUR": hour,
        "GITHUB_OUTPUT": str(out_file),
    }
    old_env = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    stdout = io.StringIO()
    cwd = Path.cwd()
    try:
        os.chdir(Path(__file__).resolve().parents[1])
        with contextlib.redirect_stdout(stdout):
            exec(compile(script, "<prepare>", "exec"), {"__name__": "__main__"})
    finally:
        os.chdir(cwd)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return out_file.read_text(encoding="utf-8"), stdout.getvalue()


def _entries(output: str) -> list[dict]:
    return json.loads(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("matrix="))
    )


def _mode(output: str) -> str:
    return next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("mode="))


def test_prepare_writes_only_key_value_pairs_to_github_output(tmp_path):
    """GITHUB_OUTPUT accepts only key=value lines.

    A `>> "$GITHUB_OUTPUT"` redirect around the whole script also captures the
    `::notice::` line, which fails the run with "Invalid format" — so assert on
    the real script's output rather than trusting the redirect to stay absent.
    """
    output, stdout = _run_prepare_script("", tmp_path=tmp_path)

    for line in output.splitlines():
        if not line.strip():
            continue
        assert "=" in line, f"non key=value line in GITHUB_OUTPUT: {line!r}"
        assert not line.startswith("::"), f"workflow command in GITHUB_OUTPUT: {line!r}"

    assert [entry["ticker"] for entry in _entries(output)] == list(TRACKED_TICKERS)
    # the notice belongs on stdout, where the runner interprets it
    assert "::notice::production " in stdout


def test_prepare_honours_a_narrowed_dispatch_list(tmp_path):
    output, _ = _run_prepare_script("BTC/USD, BNB/USD", tmp_path=tmp_path)
    entries = _entries(output)
    assert [entry["ticker"] for entry in entries] == ["BTC/USD", "BNB/USD"]
    assert all("/" not in entry["slug"] for entry in entries)


def test_prepare_honours_the_fork_watchlist_variable(tmp_path):
    """A fork narrows the watchlist with RL_CRYPTO_TICKERS, no code edit."""
    output, _ = _run_prepare_script(watchlist="SOL/USD ETH/USD", tmp_path=tmp_path)
    assert [entry["ticker"] for entry in _entries(output)] == ["SOL/USD", "ETH/USD"]


def test_scheduled_mode_follows_the_clock(tmp_path):
    """Even UTC hours retrain; odd hours only append live records."""
    assert _mode(_run_prepare_script(hour="20", tmp_path=tmp_path)[0]) == "train"
    assert _mode(_run_prepare_script(hour="21", tmp_path=tmp_path)[0]) == "live"
    assert _mode(_run_prepare_script(hour="0", tmp_path=tmp_path)[0]) == "train"


def test_dispatch_mode_wins_over_the_clock(tmp_path):
    train = _run_prepare_script(mode="train", hour="21", tmp_path=tmp_path)[0]
    live = _run_prepare_script(mode="live", hour="20", tmp_path=tmp_path)[0]
    assert _mode(train) == "train"
    assert _mode(live) == "live"


def test_dispatch_list_beats_the_fork_variable(tmp_path):
    output, _ = _run_prepare_script("ADA/USD", watchlist="BTC/USD", tmp_path=tmp_path)
    assert [entry["ticker"] for entry in _entries(output)] == ["ADA/USD"]
