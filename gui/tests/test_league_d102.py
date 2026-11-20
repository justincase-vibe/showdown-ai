"""D102 — league mode in the GUI + format dropdowns + champions prior data.

Three contracts, all fast + CPU-only:

  1. ``bot.teams.available_formats()`` is the single source of truth for the
     format dropdowns (pool-gated — a format is selectable iff teams can be
     built for it), and includes the formats the user asked for (gen8ou,
     gen9championsou).
  2. ``launcher.validate_league_config`` — the loud contract (every problem
     listed) INCLUDING the roster: <2 bots or a missing checkpoint is a
     config error before spawn; the payload carries ``_rounds_planned``
     (pairings x best_of) and pins the registry path.
  3. The desktop app builds the league fields headlessly (shared_root
     fixture — one Tk interpreter per module) and ``_build_config`` reads
     them into the right shape.

Run from the project root:

    python -m pytest gui/tests/test_league_d102.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def roster(tmp_path: Path):
    """A throwaway registry with 2 bots + fake-but-present checkpoints."""
    from league.registry import BotRegistry
    regp = tmp_path / "bots.json"
    ck_a = tmp_path / "a.pt"; ck_a.write_bytes(b"fake")
    ck_b = tmp_path / "b.pt"; ck_b.write_bytes(b"fake")
    reg = BotRegistry(regp)
    reg.add("alpha", str(ck_a))
    reg.add("beta", str(ck_b))
    reg.save()
    return regp


# ── 1. available_formats (the dropdown's single source of truth) ────────────

def test_available_formats_is_pool_gated():
    from bot import teams
    vals = teams.available_formats()
    assert "gen9ou" in vals
    assert "gen8ou" in vals          # the user explicitly asked for gen 8 OU
    assert "gen9championsou" in vals  # the champions pool (Smogon sample teams)
    assert vals == sorted(vals)      # stable order for the dropdown


# ── 2. validate_league_config ────────────────────────────────────────────────

def test_league_happy_path_rounds_planned(roster):
    from gui import launcher
    cfg = launcher.validate_league_config(
        {"registry_path": str(roster), "pairing": "round_robin", "best_of": 2})
    # 2 bots -> 1 pairing (alpha-beta); best_of 2 -> that pairing twice
    assert cfg["_rounds_planned"] == 2
    assert cfg["pairing"] == "round_robin"
    assert Path(cfg["registry_path"]).name == "bots.json"  # pinned
    # the payload round-trips through LeagueConfig.from_dict (unknown keys dropped)
    from league.engine import LeagueConfig
    lc = LeagueConfig.from_dict(cfg)
    assert lc.pairing == "round_robin" and lc.best_of == 2


def test_league_elo_weighted_rounds_planned(roster):
    from gui import launcher
    cfg = launcher.validate_league_config(
        {"registry_path": str(roster), "pairing": "elo_weighted"})
    assert cfg["_rounds_planned"] == 1  # top vs bottom, once


def test_league_roster_too_small(tmp_path):
    from league.registry import BotRegistry
    from gui import launcher
    regp = tmp_path / "bots.json"
    reg = BotRegistry(regp); reg.add("lone", str(tmp_path / "x.pt")); reg.save()
    with pytest.raises(ValueError) as exc:
        launcher.validate_league_config({"registry_path": str(regp)})
    assert "roster" in str(exc.value) and "lone" in str(exc.value)


def test_league_missing_checkpoint_is_a_config_error(tmp_path):
    from league.registry import BotRegistry
    from gui import launcher
    regp = tmp_path / "bots.json"
    ck = tmp_path / "present.pt"; ck.write_bytes(b"fake")
    reg = BotRegistry(regp)
    reg.add("a", str(ck))
    reg.add("b", str(tmp_path / "MISSING.pt"))   # does not exist
    reg.save()
    with pytest.raises(ValueError) as exc:
        launcher.validate_league_config({"registry_path": str(regp)})
    msg = str(exc.value)
    assert "checkpoint" in msg and "'b'" in msg


def test_league_lists_every_problem(roster):
    from gui import launcher
    bad = {"registry_path": str(roster), "pairing": "swiss", "device": "gpu",
           "best_of": 0, "seed": -1}
    with pytest.raises(ValueError) as exc:
        launcher.validate_league_config(bad)
    for needle in ("pairing", "device", "best_of", "seed"):
        assert needle in str(exc.value), f"expected {needle!r} in {exc.value}"


# ── 3. desktop app: league fields + format dropdown ─────────────────────────

def test_gui_league_fields_build_and_config(shared_root):
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    app._mode_var.set("League (bot vs bot)")
    app._on_mode_change(None)
    assert app._current_mode == "league"
    # every field exists
    for attr in ("_league_fmt", "_league_pairing", "_league_best_of",
                 "_league_greedy", "_league_device", "_league_n_servers",
                 "_league_per_server", "_league_port_base",
                 "_league_max_steps", "_league_seed", "_league_tree"):
        assert getattr(app, attr) is not None, attr
    # the config builder reads them into the right shape
    cfg = app._build_config()
    assert cfg["mode"] == "league"
    assert cfg["battle_format"] == "gen9ou"      # pool default
    assert cfg["pairing"] in ("round_robin", "elo_weighted")
    app.close()


def test_gui_format_dropdown_is_pool_fed(shared_root):
    """The collect-mode Format field is a readonly combobox whose values are
    exactly the pool's formats (single source of truth, D102)."""
    import tkinter as tk
    from tkinter import ttk
    from bot import teams
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    app._mode_var.set("Self-play (collect replays)")
    app._on_mode_change(None)
    # find the combobox bound to _collect_fmt among the fields frame children
    boxes = [w for w in app._fields_frame.winfo_children() if isinstance(w, ttk.Combobox)]
    bound = [w for w in boxes if w.cget("textvariable") == str(app._collect_fmt)]
    assert bound, "expected a readonly format combobox in the collect fields"
    values = list(bound[0].cget("values"))
    expected = teams.available_formats()
    assert values == expected
    # ttk.Combobox cget('state') returns a Tcl index object on this Tk build —
    # compare its str form (the value is 'readonly', per the object's repr).
    assert str(bound[0].cget("state")) == "readonly"
    assert app._collect_fmt.get() == "gen9ou"
    app.close()


def test_start_trainer_rejects_unknown_mode_including_league_wording():
    from gui import launcher
    with pytest.raises(ValueError) as exc:
        launcher.start_trainer({"mode": "nope"})
    assert "'league'" in str(exc.value)  # the error lists league as a valid mode
