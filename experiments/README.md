# Unattended model improvement

This directory is the durable record of the model-improvement loop. The manual
contract lives in `.opencode/command/model-improve.md`; this is the
**unattended** implementation. Run it inside `nix develop` — the venv's
numpy/pandas need the devshell's `LD_LIBRARY_PATH`, and preflight checks this
explicitly:

```bash
uv run python tools/model_improve.py status     # scoreboard + next hypothesis
uv run python tools/model_improve.py plan       # show the frozen experiment
uv run python tools/model_improve.py run        # execute one hypothesis
uv run python tools/model_improve.py verify     # forward/bundle check
uv run python tools/model_improve.py doctor     # tooling and role agents
```

The loop is designed to run with **no user interaction and no forwarding
between agents**. Every handoff is a file on disk, and the pass/fail decision is
computed by Python, not by a model.

## Model

There is one knob:

```json
// config/model-improvement.json
"model": "opencode/deepseek-v4.1-flash-free"
```

It applies to every role and to every Ensemble teammate. Precedence:

1. `RL_MODEL_IMPROVE_MODEL` environment variable
2. the `--model` CLI flag
3. `models.<role>` (a deliberate per-role exception)
4. `model` (the repo default)

The role agents carry no hardcoded model; `opencode run --agent` and
`team_spawn(model=...)` take it from this setting. In CI, set the repository
variable `RL_MODEL_IMPROVE_MODEL` or pass the `model` dispatch input; leave the
repo default alone unless you want to change it for everyone.

## Authority model

There are three layers, deliberately separate:

| Layer | Owns | Cannot |
| ----- | ---- | ------ |
| `config/model-improvement.json` | executor, models, budgets, scope, whether `models/` may be written | change a criterion after seeing a result |
| `experiments/hypotheses.json` | the queue, the exact experiment, the frozen criteria | be edited by a role agent |
| `tools/model_improve.py` | preflight, pre-registration, role orchestration, the mechanical verdict, the ledger commit | promote without a passing gate |

The LLM roles (researcher, implementer, auditor, evaluator) only produce
artifacts. They cannot edit `experiments/ledger.jsonl`, `data/`, `models/`, the
role agents, or the workflows.

`promotion.apply` defaults to **false**. Setting it to `true` is a one-time
standing decision, not a per-run prompt: the mechanical gate still has to pass,
and the write happens on the run branch (CI delivers it as a PR). A negative
verdict never writes anything.

## Files

| Path | Meaning |
| ---- | ------- |
| `experiments/ledger.jsonl` | append-only JSONL; pre-registrations, results, holdout touches, verdicts, blocks |
| `experiments/hypotheses.json` | source of truth for the queue; edit this, not the markdown |
| `experiments/hypotheses.md` | generated from the queue + ledger |
| `experiments/SCOREBOARD.md` | generated from the ledger; never hand-edit |
| `config/model-improvement.json` | policy (models, budgets, scope, apply) |
| `tools/model_improve.py` | the orchestrator |
| `src/rl_crypto/experiments.py` | pure ledger/criteria/rendering logic |
| `.opencode/agents/model-improve-*.md` | the six role contracts (including the post-run reviewer) |

Run artifacts live outside the repo by default:
`~/.local/share/rl-crypto/model-improve/runs/<run-id>/`. Override with
`--run-dir` or `RL_MODEL_IMPROVE_RUN_DIR`. Worktrees live under
`.model-improve/worktrees/` (gitignored).

## One run, step by step

1. **Preflight** — config, queue, ledger, agents, `opencode`, per-ticker bars,
   staleness and bundles. Staleness is a warning: offline walk-forward work is
   allowed on a stale tail, but forward claims and `promote --apply` are not.
2. **Select** — first `status: open` hypothesis in queue order whose `needs`
   are available and whose `depends_on` are completed. `--hypothesis H4` forces
   one.
3. **Pre-register** — the driver writes the experiment, the criteria, the data
   hash and the git SHA to the ledger *before* executing anything. Criteria are
   frozen; changing them is a new trial.
4. **Diagnose** — researcher (read-only) writes `diagnosis.md` and
   `researcher.json`. `ready: false` or `recommendation: stop` blocks the run.
5. **Implement** — implementer writes the smallest code/tests change, runs
   `ruff` and the offline suite, and records both.
6. **Audit** — the adversarial auditor inspects the diff for look-ahead,
   fit-on-test preprocessing, label overlap, post-hoc criteria, cost errors and
   holdout leakage. Any `fail` or `cannot_tell` blocks.
7. **Evaluate** — the driver runs the pre-registered `rl-model-lab` configs and
   computes the verdict from the criteria. The evaluator may only write prose.
8. **Review** — the mechanical reviewer checks the ledger, roles, audit, branch
   and scope; the higher-level reviewer then audits the whole ensemble session,
   the report honesty and the final git state. It may fix a concrete defect in
   `.opencode/` (agent permissions/prompts/commands) and must list every change.
   A `fail` blocks the run; `review.require: true` also blocks on `warn`.
9. **Record** — ledger entries, regenerated scoreboard/hypotheses, `report.md`,
   and one commit on the run branch. In CI that branch becomes a PR.

## Executors

| Executor | What happens | Used when |
| -------- | ------------ | --------- |
| `team` | one headless lead uses `team_create` / `team_spawn` for researcher, implementer and auditor | local runs with the Ensemble plugin |
| `passes` | the same roles as separate `opencode run --agent … --auto` processes with fresh context | CI (no plugin needed) and any machine without Ensemble |
| `mechanical` | no LLM; only hypotheses with `needs: []` run | smoke tests, cheap scheduled diagnosis, CI without credentials |

`auto` chooses `team` when the plugin exists and `CI` is unset, otherwise
`passes`. Every role is invoked with `--auto` and stdin closed: an approval
prompt would hang the run, so the agents are permissioned to be autonomous and
denied the dangerous commands (push, main checkout, `sudo`, `rm -rf /`).

## Adding or retiring a hypothesis

Edit `experiments/hypotheses.json`, then:

```bash
uv run python tools/model_improve.py report --render
```

Each entry needs:

- `needs`: subset of `code`, `data`, `approval`, `tooling`. A blocked external
  dependency is `status: blocked`, not `needs`.
- `experiment`: `action` (`evaluate` or `sweep`), tickers, targets, feature
  sets, model sets, horizons, `maxConfigs`, `folds`, and optional
  `extraLabArgs` — the explicit flags the implementation must add (e.g.
  `["--panel"]`). If the flag is missing, the lab exits nonzero and the run is
  blocked rather than silently evaluating the old behaviour.
- `criteria`: the metric of record and the numeric bar. `kind: diagnostic`
  means "ran and produced rows" and is the only kind that needs no target.
- `depends_on`: hypothesis ids that must have a completed verdict first.

A verdict closes the hypothesis (`select_hypothesis` skips it). To re-open it
after new data or a fixed measurement bug, add a new entry (or a new `family`
trial); do not rewrite the old one.

## CI/CD

`.github/workflows/model-improve.yml` runs the loop as a dispatch, or weekly
once enabled. It:

1. checks out the repo plus the two upstream siblings, syncs `uv`;
2. installs a pinned `opencode-ai` and writes `OPENCODE_AUTH_JSON` to the
   runner's opencode data dir (mode 600, never logged);
3. creates a `model-improve/<date>-<hid>` branch and runs the driver with
   `--no-worktree --commit`;
4. pushes the branch and opens a PR; it never pushes `main` and never merges;
5. uploads the run directory (report, lab JSON, role logs) for 30 days.

Setup:

- Add a repository secret `OPENCODE_AUTH_JSON` containing a CI-only
  `auth.json` (the same shape as `~/.local/share/opencode/auth.json`). Do not
  reuse a personal multi-provider credential if you can mint a scoped one.
  It must be able to use the configured provider/model (`opencode` /
  `deepseek-v4.1-flash-free` by default).
- Dispatch once with `mode=plan` (no secret needed; installs opencode but makes
  no model call) to smoke-test the plumbing and see the next pre-registration.
  `mode=status` checks the ledger without installing anything.
- Only then set the repository variable `RL_MODEL_IMPROVE_ENABLED=true` to arm
  the weekly schedule. Without it, the schedule job is skipped.

`executor=team` is available for CI but needs the Ensemble plugin installed on
the runner; `passes` is the supported CI default.

## Honest limits

The orchestration is complete; the measurement foundation is not. These gaps
still prevent a **confirmed** claim and are reported as warnings by preflight:

- `tools/forward_score.py` does not exist, so forward confirmation (Phase V)
  is unavailable; a passing backtest is at most **provisional**.
- Production training does not yet read `config/training.json`, so a promoted
  bundle is overwritten by the next scheduled retrain. Promotion is therefore
  branch-only until that lands.
- The volatility purge test (`max(horizon, 2)` rows) and lab/production parity
  are still tracked in `/model-improve` Phase F.

The loop refuses to pretend otherwise: `promotion.apply` is off, and every
report states the verdict and the reasons, never a profit claim.

## Troubleshooting

| Symptom | Cause | Action |
| ------- | ----- | ------ |
| `preflight failed` with `opencode` | headless agent executor but no binary | install `opencode-ai`, or run `--executor mechanical` |
| role `failed: missing artifact` | the agent stopped without writing its contract | read `runs/<id>/logs/<role>.jsonl`; fix the prompt/agent, do not hand-write the artifact |
| `opencode refused … (subagent) and fell back` | a role agent is `mode: subagent` | set `mode: all` in `.opencode/agents/model-improve-*.md`; preflight's `agent-modes` blocks this |
| reviewer `status: fail` | contract violation, dishonest report or unrelated git change | read `review.json`/`review.md`; fix the evidence, then re-run |
| reviewer model is wrong | `model` changed without checking the provider credential | edit `config/model-improvement.json`'s `model`, or set `RL_MODEL_IMPROVE_MODEL` |
| `blocked` at audit | leakage or `cannot_tell` | fix the code or the test; a re-run is a new trial |
| two runs at once | ledger/worktree race | use the workflow's `concurrency` group; locally, one run at a time |
| no PR in CI | dry run, or no commits beyond `main` | check `steps.loop.outputs.rc` and the console artifact |
