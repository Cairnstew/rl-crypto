# Efficiency Proposals

Proposals are **records only** — intentionally not built in the session that
produced the evidence. Each entry names an idea, cites the quantitative
evidence, and says what it would save. Nothing here is scoped work.

---

## 2026-10-12 — Use the `ci-monitoring` skill when debugging rl-crypto Actions runs

**Evidence.** During the matrix-pipeline bug-fix cycle, this session made 57
`gh`-related calls (watches, run listings, job log greps) across ~3 iterations
of *change → push → dispatch → wait → inspect → fix*. Query against
`~/.local/share/opencode/opencode.db`:
`part WHERE data LIKE '%gh%watch%'` for `ses_ed7da3a3` → 57 rows (304 total
bash tool calls, 582 tool parts in the session).

**Idea.** Load the already-available `ci-monitoring` skill (and its `ci-monitor`
tool) when a workflow run needs watching. It is a single blocking call that
waits for completion and returns structured JSON with the failing job, instead
of the manual `gh run watch` → `gh run view --json jobs` → `--log` + `grep`
dance. That collapses ~3-5 calls per iteration to ~1.

**Savings.** Roughly 2-3× fewer playback/inspection calls (≈20-30 of the 57 in
this session), and each blocking watch also removes the sleep-based polling that
dominated the wall time between fixes.

**Why proposal-only.** It requires no code change — it is purely a workflow
habit for the agent. Coded elsewhere as a skill, it just needs to be *used*
here; nothing to build in this repo.
---

## 2026-10-24 — Wrap the dev-shell interpreter instead of re-typing `LD_LIBRARY_PATH`

**Evidence.** In the foundation session (`ses_e91f3d653ffeim0RNc8kC7j529`,
130 bash tool calls, cost $0.227), 44 parts carried an explicit
`LD_LIBRARY_PATH=...` prefix and 26 parts invoked `nix develop` — roughly a
third of the shell calls paid the dev-shell ritual just to import numpy (the
uv venv links `libz.so.1` by soname and cannot import it outside the flake's
environment). Query:
`select count(*) from part where session_id='ses_e91f3d653ffeim0RNc8kC7j529' and data like '%LD_LIBRARY_PATH%'`.

**Idea.** Add a tiny `tools/dev-python` shim (or a `direnv`/`nix develop -c`
alias checked into the repo) that resolves the flake's `LD_LIBRARY_PATH` and
execs `.venv/bin/python`, so a run writes `tools/dev-python tools/model_lab.py
...` instead of a two-store-path prefix. A `--references`-style cached export
file would also work.

**Savings.** ~30-40 shell commands per session stop carrying a 120-character
environment prefix and the associated copy/paste risk; the failure mode
(`ImportError: libz.so.1`) becomes impossible rather than documented.

**Why proposal-only.** It is a developer-ergonomics change, not measurement
infrastructure; it was not built in the session that produced the evidence.
