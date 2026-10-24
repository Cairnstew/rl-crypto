"""The protected-file list is pinned so changing it needs a reviewable edit.

``PROTECTED_FILES`` is the measurement contract: a normal run may not edit those
files, and ``run_type: harness-change`` is the only escape hatch. Editing the
list itself is the moment the contract changes, so the list's hash is committed
to ``experiments/protected-files.pin``. A change to one without the other fails
this test and must be reviewed by a human.
"""

from __future__ import annotations

from pathlib import Path

from rl_crypto.experiments import PROTECTED_FILES, protected_list_digest

REPO_ROOT = Path(__file__).resolve().parents[1]
PIN = REPO_ROOT / "experiments" / "protected-files.pin"


def _pinned_digest() -> str:
    tokens = [
        line.strip()
        for line in PIN.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(tokens) == 1, "the pin file must contain exactly one hash"
    return tokens[0]


def test_protected_file_list_matches_the_committed_pin():
    assert _pinned_digest() == protected_list_digest(), (
        "PROTECTED_FILES changed without updating experiments/protected-files.pin; "
        "this is a measurement-contract change and needs an explicit human review"
    )


def test_the_pin_is_not_trivially_stale_when_the_list_moves():
    # Guard against a pin that no longer describes the real list at all.
    assert len(PROTECTED_FILES) == len(set(PROTECTED_FILES))
    assert "src/rl_crypto/experiments.py" in PROTECTED_FILES
    assert "tools/model_lab.py" in PROTECTED_FILES
