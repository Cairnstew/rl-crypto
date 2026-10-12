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