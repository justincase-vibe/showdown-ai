"""A5 — relative user-supplied paths resolve against the PROJECT ROOT, not the
GUI's CWD.

The checkpoint-path validators already did this via ``_resolve_ckpt_path``
(eval/play/train/initial-ckpt). The BC *fixture* path was the one straggler:
it checked ``Path(fixture).expanduser().is_file()`` (CWD-relative) while its
error message claimed "relative to the project root" — so a valid relative
fixture was wrongly 400'd if the GUI was ever started from elsewhere (e.g. a
Desktop shortcut with a different working dir).

This pins:
  * a relative fixture that exists under ROOT is accepted even when CWD != ROOT;
  * a genuinely-missing relative fixture still 400s (loud, not a crash).

Run from the project root:

    python -m pytest gui/tests/test_paths_resolve_root.py -v
"""
from __future__ import annotations

import pytest

from gui import launcher


def _bc_cfg(fixture: str) -> dict:
    return {"fixture": fixture, "n_battles": 1, "max_decisions": 4,
            "heldout_frac": 0.5, "epochs": 1, "batch_size": 2, "lr": 1e-3,
            "device": "cpu", "seed": 0}


def test_bc_fixture_relative_resolves_against_root(tmp_path, monkeypatch):
    # Project root = a temp dir; the fixture lives at <root>/replays/gen9ou.jsonl.
    root = tmp_path / "projroot"
    rel_fixture = "replays/gen9ou.jsonl"
    (root / "replays").mkdir(parents=True)
    (root / rel_fixture).write_text('{"x": 1}\n', encoding="utf-8")
    monkeypatch.setattr(launcher, "ROOT", root)

    # Simulate the GUI being started from a DIFFERENT working directory.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # CWD != ROOT

    # A relative path that exists under ROOT must be accepted (not 400'd).
    out = launcher.validate_bc_config(_bc_cfg(rel_fixture))
    assert out["fixture"] == rel_fixture


def test_bc_fixture_missing_relative_still_400(tmp_path, monkeypatch):
    root = tmp_path / "projroot"
    root.mkdir()
    monkeypatch.setattr(launcher, "ROOT", root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with pytest.raises(ValueError, match="fixture"):
        launcher.validate_bc_config(_bc_cfg("nope/missing.jsonl"))
