"""Structural checks on the pipeline workflow.

These lock down the shape the matrix design depends on. The important one is
``test_matrix_jobs_never_push``: GitHub's ``concurrency`` group serialises
workflow *runs*, not jobs inside a run, so if every matrix job pushed to ``main``
they would race and fail as non-fast-forwards. That is a silent, intermittent
failure in CI, so it is asserted here rather than left to review.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from rl_crypto.cache import slug_for
from rl_crypto.tickers import TRACKED_TICKERS

_PIPELINE = Path(__file__).resolve().parents[1] / ".github/workflows/pipeline.yml"


def _workflow() -> dict:
    return yaml.safe_load(_PIPELINE.read_text(encoding="utf-8"))


def _runs(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


def test_pipeline_has_prepare_produce_merge():
    jobs = _workflow()["jobs"]
    assert set(jobs) >= {"prepare", "produce", "merge"}


def test_matrix_comes_from_the_tracked_watchlist():
    """The matrix is built from tickers.py, never a second copy in the YAML."""
    jobs = _workflow()["jobs"]
    assert jobs["produce"]["needs"] == "prepare"
    matrix = jobs["produce"]["strategy"]["matrix"]
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


def test_matrix_jobs_never_push():
    """Single-writer invariant: only the merge job may push."""
    jobs = _workflow()["jobs"]
    assert "git push" not in _runs(jobs["produce"]), "matrix jobs must not push"
    assert "git push" in _runs(jobs["merge"]), "merge job must be the writer"
    assert jobs["produce"]["strategy"].get("fail-fast") is False


def test_bounded_parallelism():
    strategy = _workflow()["jobs"]["produce"]["strategy"]
    assert strategy.get("max-parallel"), "unbounded matrix parallelism is unsafe"


def test_merge_runs_on_partial_failure():
    """If some tickers fail, the surviving caches must still be committed."""
    merge = _workflow()["jobs"]["merge"]
    assert merge["needs"] == "produce"
    assert "cancelled()" in str(merge.get("if"))
    assert "if-no-files-found" not in _runs(merge)  # unrelated, keeps intent clear


def test_merge_push_retries_after_rebase():
    """The local 30-minute timer pushes data/ concurrently, so a rejected push
    must rebase and retry rather than failing the run."""
    merge_runs = _runs(_workflow()["jobs"]["merge"])
    assert "git pull --rebase" in merge_runs
    assert "for attempt in" in merge_runs


def _run_prepare_script(requested: str, tmp_path: Path) -> tuple[str, str]:
    """Execute the real prepare step's script; return (GITHUB_OUTPUT, stdout)."""
    import contextlib
    import io
    import os
    import re

    run = _workflow()["jobs"]["prepare"]["steps"][1]["run"]
    source = re.search(r"python3 - <<'PY'\n(.*?)\nPY", run, re.S)
    assert source, "prepare step no longer uses the expected heredoc"
    script = source.group(1)

    out_file = tmp_path / "github_output"
    out_file.write_text("", encoding="utf-8")
    env = {"REQUESTED": requested, "GITHUB_OUTPUT": str(out_file)}
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    stdout = io.StringIO()
    cwd = Path.cwd()
    try:
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


def test_prepare_writes_only_key_value_pairs_to_github_output(tmp_path):
    """GITHUB_OUTPUT accepts only key=value lines.

    A `>> "$GITHUB_OUTPUT"` redirect around the whole script also captures the
    `::notice::` line, which fails the run with "Invalid format" — so assert on
    the real script's output rather than trusting the redirect to stay absent.
    """
    output, stdout = _run_prepare_script("", tmp_path)

    for line in output.splitlines():
        if not line.strip():
            continue
        assert "=" in line, f"non key=value line in GITHUB_OUTPUT: {line!r}"
        assert not line.startswith("::"), f"workflow command in GITHUB_OUTPUT: {line!r}"

    entries = json.loads(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("matrix="))
    )
    assert [e["ticker"] for e in entries] == list(TRACKED_TICKERS)
    # the notice belongs on stdout, where the runner interprets it
    assert "::notice::pipeline matrix:" in stdout


def test_prepare_honours_a_narrowed_dispatch_list(tmp_path):
    output, _ = _run_prepare_script("BTC/USD, BNB/USD", tmp_path)
    entries = json.loads(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("matrix="))
    )
    assert [e["ticker"] for e in entries] == ["BTC/USD", "BNB/USD"]
    assert all("/" not in e["slug"] for e in entries)
