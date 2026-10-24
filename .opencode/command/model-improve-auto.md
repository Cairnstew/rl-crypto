---
description: Unattended model-improvement loop. Runs the pre-registered experiment with bounded agents, records the verdict in the ledger, and never asks the user a question.
---

# /model-improve-auto $ARGUMENTS

Run the unattended loop. **Do not ask questions, do not edit files yourself,
and do not change a criterion after seeing a result.** `tools/model_improve.py`
owns the run; you only invoke it and report what it wrote.

## Argument mapping

| `$ARGUMENTS` | Command |
| ------------ | ------- |
| empty or `run` | `run` |
| `status` | `status --render` |
| `plan` | `plan` |
| `verify` | `verify` |
| `doctor` | `doctor` |
| `mechanical` | `run --no-llm` |
| `H<n>` (e.g. `H4`) | `run --hypothesis H<n>` |
| anything else | treat as a free-text goal, map it to a queue id, and pass `--hypothesis`; if no id fits, run `plan` and report the queue |

Use the project interpreter inside `nix develop` (the venv's numpy/pandas need
the devshell's `LD_LIBRARY_PATH`). Prefer
`uv run python tools/model_improve.py …`; fall back to
`.venv/bin/python tools/model_improve.py …` when `uv` is absent. Preflight
checks the import and tells you if the environment is wrong.

Add `--executor team` only if the Ensemble plugin is loaded; otherwise let
`auto` choose (it falls back to sequential passes). Add `--json` when you need a
machine-readable summary.

The model is the single `model` knob in `config/model-improvement.json`
(`opencode/deepseek-v4.1-flash-free` by default). Override it for one run with
`--model provider/model` or `RL_MODEL_IMPROVE_MODEL`. Do not edit the model in
several agent files; there are no hardcoded agent models.

The run ends with a post-run reviewer. Do not pass `--no-review` unless the
user explicitly asks; a `fail` review is a real result and must be reported.

## What to do

1. Run `doctor`. If it reports a missing agent or `opencode`, stop and report
   the exact missing item. Do not try to work around a missing role.
2. Run the mapped command. Use a long timeout: a `run` may take up to an hour
   (the lab sweeps four folds per ticker).
3. Read `RUN_DIR/report.md` and print it. Also print the run directory and the
   branch. Do not summarise the numbers from memory — quote the report.
4. Exit code `2` means **blocked**, not failed: report the blocker verbatim and
   stop. Do not retry with different criteria or a bigger grid. Exit code `1`
   is an unexpected error: report the command and stderr.
5. A negative verdict is a successful run. Say so plainly. Never call a
   backtest a deployed result and never claim profit.

## Hard rules (mirroring the command)

- The final 20% of the archive is reserved for `promote`; only the driver may
  run it.
- `--apply` writes `models/` only when `config/model-improvement.json` (or the
  explicit flag) allows it and the mechanical gate passes; otherwise the run
  stays a branch + PR.
- If you are running inside an interactive session, you may additionally run
  `/model-improve` afterward to review the branch by hand. The unattended run
  itself is complete when `report.md` exists.
