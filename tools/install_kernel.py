#!/usr/bin/env python3
"""Install a Jupyter kernelspec pinned to this project's interpreter.

Zed (and Jupyter) discover notebook kernels via ``jupyter kernelspec list``; a
bare ``python3`` spec resolves ``python`` through ``PATH``, so a notebook cell
can quietly run a different interpreter than the one under test. This writes a
spec whose ``argv`` names the project interpreter by **absolute path**:

* ``.venv/bin/python`` when a virtualenv exists (the uv/pip workflow), else
* the interpreter running this script (the ``nix develop`` workflow).

    python tools/install_kernel.py

Re-run after recreating the virtualenv.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

KERNEL_NAME = "rl-crypto"
DISPLAY_NAME = "rl-crypto"

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def target_python() -> Path:
    if VENV_PYTHON.exists():
        return VENV_PYTHON
    return Path(sys.executable)


def user_data_dir() -> Path:
    """The Jupyter data dir whose ``kernels/`` subdir Jupyter and Zed scan."""
    override = os.environ.get("JUPYTER_DATA_DIR")
    if override:
        return Path(override) / "kernels"
    local = ROOT / ".jupyter-data"
    if local.is_dir():
        return local / "kernels"
    return Path.home() / ".local" / "share" / "jupyter" / "kernels"


def main() -> int:
    python = target_python()
    if not python.exists():
        print(f"interpreter not found: {python}", file=sys.stderr)
        return 1

    target = user_data_dir() / KERNEL_NAME
    target.mkdir(parents=True, exist_ok=True)
    spec = {
        "argv": [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": DISPLAY_NAME,
        "language": "python",
        "metadata": {"debugger": True},
    }
    (target / "kernel.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(f"installed kernelspec {KERNEL_NAME!r} -> {python}")
    print(f"  {target / 'kernel.json'}")

    import importlib.util

    if importlib.util.find_spec("ipykernel") is None:
        print(
            "warning: ipykernel is not importable here; "
            'install it (uv sync --group dev, or `nix develop`)',
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
