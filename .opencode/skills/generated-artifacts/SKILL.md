---
name: generated-artifacts
description: >-
  Use when editing or regenerating the rl-crypto notebook, README dashboard,
  Jupyter kernel, or any file produced by tools/. Triggers: "update the
  notebook", "unified.ipynb", "dashboard", "README dashboard", "render
  dashboard", "kernel", "no kernel", "validate notebook", "build_notebook",
  "install_kernel". Explains which files are generated, the source to edit
  instead, and how to verify the output, with a memory of kernel/CI quirks.
---

# Generated artifacts

Several user-facing artifacts in this repo are **generated from code or the
committed cache**. Editing the output directly is the classic trap: the next
regeneration overwrites it, or CI disagrees with the working tree.

Companions: **`data-quality-audit`** (the dashboard reads the cache),
**`pipeline-hardening`** (the merge job renders the dashboard),
**`model-contract`** (the bundle fields shown in the table).

## 0. Contract

- **Edit the source, regenerate the output.** Never hand-edit a generated file
  except to fix a one-off merge conflict that the generator cannot express.
- **Commit both.** A source change without its regenerated artifact (or vice
  versa) is a drift finding.
- **Respect the generated markers.** In `README.md`, only the block between
  `<!-- dashboard:start -->` and `<!-- dashboard:end -->` is generated.
- **The notebook is not the source.** Cells live in `tools/build_notebook.py`.
- **The kernel is pinned by absolute path.** Re-run the installer after
  recreating `.venv`.
- **Offline first.** The notebook reads the committed cache; it should not need
  network. The live workflow is opt-in/scheduled.

## 1. Ground truth

| Artifact | Source | Regenerate | Verify |
| --- | --- | --- | --- |
| `notebooks/unified.ipynb` | `tools/build_notebook.py` | `uv run python tools/build_notebook.py` | `uv run python tools/validate_notebook.py` |
| README dashboard block | `tools/render_dashboard.py` + `data/` + `models/` | `uv run python tools/render_dashboard.py` | `--stdout`; `tests/test_dashboard.py` |
| Jupyter kernelspec `rl-crypto` | `tools/install_kernel.py` | `uv run python tools/install_kernel.py` | `jupyter kernelspec list`; Zed selects it via `.zed/settings.json` |
| README hand-written prose | humans | edit directly | outside the markers |
| `data/AUDIT.md` | `data-quality-audit` runs | edit/append directly | not a generated file |

Relevant workflows:

- `ci.yml` — runs ruff, **builds** the notebook offline, and runs the tests.
  It does not execute the notebook.
- `notebook.yml` — weekly (Mondays 04:23 UTC) and on dispatch; installs the
  `rl-crypto` kernel and **executes** the notebook end-to-end.
- `production.yml` — renders the dashboard in the merge job on every cache
  commit; a render failure warns and continues.

## 2. Procedure

### 2.1 Notebook change

1. Edit the markdown/code cell strings in `tools/build_notebook.py`. The cells
   are ordinary strings, so the diff reads like the notebook.
2. Regenerate:

   ```bash
   nix develop --command bash -c 'uv run python tools/build_notebook.py'
   ```

3. Validate offline (it reads the committed cache; no live upstream calls):

   ```bash
   nix develop --command bash -c 'uv run python tools/validate_notebook.py'
   ```

   For the exact kernel/CI path:

   ```bash
   nix develop --command bash -c 'uv run python tools/validate_notebook.py --kernel rl-crypto --timeout 180'
   ```

4. Commit `tools/build_notebook.py` **and** `notebooks/unified.ipynb`.
5. If the notebook needs a new dependency, add it to `pyproject.toml` and
   `uv.lock`; CI and `notebook.yml` run `uv sync --group dev`.

Do not edit `notebooks/unified.ipynb` directly unless you are resolving a merge
conflict and will regenerate immediately.

### 2.2 Kernel

The `rl-crypto` kernelspec pins the project interpreter by absolute path:

- `.venv/bin/python` when the venv exists, else the interpreter running the
  installer (the `nix develop` Python).

```bash
nix develop --command bash -c 'uv run python tools/install_kernel.py'
jupyter kernelspec list
```

Re-run after recreating `.venv` or changing the dev shell. `.zed/settings.json`
selects the `rl-crypto` kernel for Python; if the spec is missing, the notebook
reports "no kernel" rather than silently using another interpreter.

The installer uses `JUPYTER_DATA_DIR` when set, else a project-local
`.jupyter-data` if it exists, else `~/.local/share/jupyter/kernels`. The
`.gitignore` excludes `.jupyter-data/`.

### 2.3 README dashboard

```bash
nix develop --command bash -c 'uv run python tools/render_dashboard.py'
nix develop --command bash -c 'uv run python tools/render_dashboard.py --stdout'
```

- It is **stdlib-only and offline** because the merge job runs it without a
  venv.
- It rewrites only the block between the markers; everything outside is
  hand-written.
- The table is derived from `data/*/summary.json`, the newest
  `data/*/live.jsonl` record, and `models/*.json`; an unreadable file renders as
  a dash rather than failing the whole dashboard.
- A render failure in `production.yml` is a warning, never a reason to drop a
  cache commit.

After a cache/schema change, regenerate and include the README diff with the
cache change. If only docs changed, do not invent a cache diff.

### 2.4 Docs and the generated block

- Hand-written README text: edit directly.
- Dashboard block: regenerate, do not hand-edit.
- Data/models docs: `data/README.md` and `models/README.md` are hand-written
  contracts; update them when the schema or protocol changes.
- A stale generated artifact is a finding; a stale hand-written doc is also a
  finding (see `pipeline-hardening` cleanliness).

### 2.5 Verify and commit

```bash
nix develop --command bash -c 'uv run python -m pytest -m "not live" -q'
nix develop --command bash -c 'uv run ruff check src/ tests/ examples/ tools/'
git diff --stat
```

- If you regenerated the notebook, confirm the `.ipynb` diff is only the
  intended cell changes, not a wholesale rewrite (nbformat can reorder metadata).
- If you regenerated the dashboard, confirm the diff is confined to the marker
  block.
- Commit the source and the generated artifact together.

## 3. Update this file

Record a new quirk (kernel path, nbclient timeout, a generated-file conflict
rule) in `Memory — active findings`; keep operational commands in §2. Update the
baseline when the tool set changes.

## Memory — baseline

**As of 2026-10-23T23:18Z** (`HEAD` `213d17b`):

| Item | Value |
| --- | --- |
| Generated notebook | `notebooks/unified.ipynb` (13 cells, kernel `rl-crypto`) |
| Notebook source | `tools/build_notebook.py` |
| Dashboard block | `README.md` between `<!-- dashboard:start -->` and `<!-- dashboard:end -->`; last rendered 2026-10-23T21:07Z |
| Kernel installer | `tools/install_kernel.py` (`rl-crypto`, absolute interpreter path) |
| CI | `ci.yml` builds the notebook offline; `notebook.yml` executes it weekly (Mon 04:23 UTC) |
| Merge behavior | `production.yml` renders the dashboard on every cache commit; failure warns, never blocks |
| Tests | `test_dashboard.py`, `test_workflows.py` (merge renders the dashboard), 181 offline tests |

## Memory — active findings

### GA-001 · The notebook is generated from `build_notebook.py` · EXPECTED (since 2026-10-23)
- **Layer:** notebook (`notebooks/unified.ipynb`; `tools/build_notebook.py`)
- **Evidence (2026-10-23):** the file header says it is generated; `build_notebook.py` writes all cells from strings; `notebooks/unified.ipynb` has 13 cells.
- **Read:** a direct edit is overwritten on the next build. Edit the source and regenerate; commit both.
- **Map:** IM-1
- **Status:** expected.

### GA-002 · The README dashboard is a generated block · EXPECTED (since 2026-10-23)
- **Layer:** docs (`README.md`; `tools/render_dashboard.py`)
- **Evidence (2026-10-23):** the block is delimited by `<!-- dashboard:start/end -->`; the renderer rewrites only that block; the production merge job calls it on every cache commit.
- **Read:** hand-editing the table is overwritten; hand-written prose outside the markers is safe. Regenerate after a cache/schema change.
- **Map:** IM-1
- **Status:** expected.

### GA-003 · The kernel pins an absolute interpreter path · EXPECTED (since 2026-10-23)
- **Layer:** kernel (`tools/install_kernel.py:target_python`)
- **Evidence (2026-10-23):** `argv[0]` is `.venv/bin/python` when present, else `sys.executable`; recreating `.venv` leaves a stale spec until the installer is re-run.
- **Read:** "no kernel" or a wrong interpreter in Zed is almost always a stale kernelspec. Re-run `install_kernel.py`; `.zed/settings.json` already selects `rl-crypto`.
- **Map:** IM-4
- **Status:** expected.

### GA-004 · Notebook execution is weekly, not per-change · OPEN (since 2026-10-23)
- **Layer:** CI (`.github/workflows/ci.yml`; `notebook.yml`)
- **Evidence (2026-10-23):** `ci.yml` builds the notebook but does not execute it; `notebook.yml` executes it only on a weekly cron or dispatch.
- **Read:** a broken notebook can land and stay broken for up to a week. A cheap fix is executing the offline notebook in PR CI, but it adds runner minutes and a kernel install.
- **Map:** IM-2
- **Status:** open; decide the cost/benefit.

### GA-005 · The dashboard can lag a code/docs change · MONITORING (since 2026-10-23)
- **Layer:** workflow (`production.yml` merge)
- **Evidence (2026-10-23):** the dashboard renders only when there is a cache write; a code/docs-only commit does not refresh it, by design ("a run that collected nothing must not produce a README-only commit").
- **Read:** the table describes cache state, so it is correctly tied to cache commits. A stale dashboard during a quiet period is expected; do not force a README-only commit just to refresh it.
- **Map:** none
- **Status:** monitoring.

### GA-006 · No `--check` mode for the generators · OPEN (since 2026-10-23)
- **Layer:** tooling (`tools/build_notebook.py`, `tools/render_dashboard.py`)
- **Evidence (2026-10-23):** both tools rewrite their output; neither can compare and exit nonzero on drift. CI therefore cannot tell "generated artifact is stale" from "no change".
- **Read:** a `--check` mode (render to memory, compare, exit 1 on diff) would let CI enforce "source and artifact committed together" without rewriting files.
- **Map:** IM-3
- **Status:** open.

## Memory — resolved & expected

- **R-001 · The notebook needs no network** — it syncs the checkout, then reads the committed cache; live fetching lives in `rl-pipeline`.
- **R-002 · The dashboard renderer is stdlib-only** — the production merge job has no venv; import only the standard library there.
- **R-003 · A render failure warns, never drops the cache** — the cache commit is the priority; the dashboard is a view.
- **R-004 · README conflicts keep upstream text and re-render** — the merge job checks out `--ours`, then renders from the merged cache; do not hand-merge the table.
- **R-005 · The kernel is re-installed in the notebook workflow** — `notebook.yml` installs it before executing, so CI does not depend on a developer's kernelspec.

## Memory — improvement map

| ID | Finding | Layer | Action | Target | Risk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| IM-1 | GA-001, GA-002 | docs | Keep source + generated artifact in the same commit; add a short "generated files" note to the README layout | `README.md`, `tools/` | none | proposed |
| IM-2 | GA-004 | CI | Execute the offline notebook in PR CI (or a cheaper smoke cell) instead of waiting for the weekly run | `.github/workflows/ci.yml` | runner minutes; kernel install | proposed |
| IM-3 | GA-006 | tooling | Add `--check` to `build_notebook.py` and `render_dashboard.py`; fail on drift | `tools/` | low | proposed |
| IM-4 | GA-003 | docs | Mention the kernel reinstall after `.venv` recreation in the README notebook section | `README.md` | none | proposed |
| IM-5 | GA-005 | workflow | Leave dashboard rendering tied to cache commits; document why a quiet README is expected | `README.md` / this skill | none | proposed |

## Memory — audit log

| Run (UTC) | HEAD | Notebook | Dashboard | Kernel | Verdict |
| --- | --- | --- | --- | --- | --- |
| 2026-10-23T23:18Z | `213d17b` | 13 cells, generated | rendered 2026-10-23T21:07Z | `rl-crypto` absolute | generated sources are clear; drift checks are missing |

## Memory — open questions

- **Q-1** Is executing the notebook in PR CI worth the runner minutes, or is the weekly smoke test enough?
- **Q-2** Should `--check` compare byte-for-byte, or normalize `.ipynb` metadata first? nbformat can reorder metadata and create noisy diffs.
- **Q-3** Is `notebooks/tests/quick_test.ipynb` generated or hand-written? It is not mentioned by `build_notebook.py`; confirm before adding it to the generated set.
- **Q-4** Should the dashboard renderer be run after a model-only change (`--retrain-only`) even without a cache commit? It reads models; decide whether that is a "cache" change.
- **Q-5** Should the README dashboard show the model's `trained_at` age as a warning when it exceeds the retrain cadence? That is a data-quality concern; keep it in the audit skill unless it becomes an operational alert.

