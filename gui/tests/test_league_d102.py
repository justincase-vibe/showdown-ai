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


def test_league_learning_mode_rejects_shared_checkpoint(tmp_path):
    """D103: a learning league mutates bots -> ONE FILE PER BOT is a hard
    contract. Two roster bots on one checkpoint must be a config error
    BEFORE the pool boots (the first bot to save would clobber the
    other's weights)."""
    from league.registry import BotRegistry
    from gui import launcher
    regp = tmp_path / "bots.json"
    shared = tmp_path / "shared.pt"; shared.write_bytes(b"fake")
    reg = BotRegistry(regp)
    reg.add("alpha", str(shared))
    reg.add("beta", str(shared))
    reg.save()
    with pytest.raises(ValueError) as exc:
        launcher.validate_league_config({"registry_path": str(regp),
                                         "learning": True})
    msg = str(exc.value)
    assert "ONE FILE PER BOT" in msg
    assert "'alpha'" in msg and "'beta'" in msg


def test_league_eval_mode_allows_shared_checkpoint(tmp_path):
    """learning=False (the legacy eval-only run) never writes -> a shared
    checkpoint file is still legal there (the pre-D103 contract)."""
    from league.registry import BotRegistry
    from gui import launcher
    regp = tmp_path / "bots.json"
    shared = tmp_path / "shared.pt"; shared.write_bytes(b"fake")
    reg = BotRegistry(regp)
    reg.add("alpha", str(shared))
    reg.add("beta", str(shared))
    reg.save()
    cfg = launcher.validate_league_config({"registry_path": str(regp),
                                           "learning": False})
    assert cfg["learning"] is False
    assert cfg["_rounds_planned"] == 1


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
                 "_league_greedy", "_league_learning", "_league_device",
                 "_league_n_servers", "_league_per_server", "_league_port_base",
                 "_league_max_steps", "_league_seed", "_league_tree"):
        assert getattr(app, attr) is not None, attr
    # the config builder reads them into the right shape
    cfg = app._build_config()
    assert cfg["mode"] == "league"
    assert cfg["battle_format"] == "gen9ou"      # pool default
    assert cfg["pairing"] in ("round_robin", "elo_weighted")
    assert cfg["learning"] is True               # D103: league learns by default
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


# ── 4. D103 step 3 (GUI): PPO knobs + copy-on-add + delete ─────────────────

def test_gui_league_ppo_fields_in_config(shared_root):
    """D103 step 3: the Rounds / PPO epochs / PPO lr fields exist with
    TrainConfig-mirroring defaults and flow through ``_build_config`` into
    the league payload (LeagueConfig carries the same key names)."""
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    app._mode_var.set("League (bot vs bot)")
    app._on_mode_change(None)
    assert app._league_n_rounds.get() == 1        # mirrors net.trainer defaults
    assert app._league_epochs.get() == 2
    assert app._league_lr.get() == pytest.approx(3e-4)
    cfg = app._build_config()
    assert cfg["n_rounds"] == 1
    assert cfg["epochs"] == 2
    assert cfg["lr"] == pytest.approx(3e-4)
    app.close()


def test_validate_league_config_checks_ppo_knobs(tmp_path):
    """The launcher contract: bad n_rounds / epochs / lr are loud config
    errors before spawn (same style as best_of / seed)."""
    from gui import launcher
    # roster of two bots with present files (so ONLY the knob problems fire)
    from league.registry import BotRegistry
    regp = tmp_path / "bots.json"
    ck_a = tmp_path / "a.pt"; ck_a.write_bytes(b"fake")
    ck_b = tmp_path / "b.pt"; ck_b.write_bytes(b"fake")
    reg = BotRegistry(regp)
    reg.add("alpha", str(ck_a)); reg.add("beta", str(ck_b)); reg.save()
    bad = {"registry_path": str(regp), "n_rounds": 0, "epochs": -1, "lr": 0.0}
    with pytest.raises(ValueError) as exc:
        launcher.validate_league_config(bad)
    for needle in ("n_rounds", "epochs", "lr"):
        assert needle in str(exc.value), f"expected {needle!r} in {exc.value}"
    # and a good payload carries them through (LeagueConfig round-trip)
    good = launcher.validate_league_config(
        {"registry_path": str(regp), "n_rounds": 3, "epochs": 1, "lr": 1e-3})
    from league.engine import LeagueConfig
    lc = LeagueConfig.from_dict(good)
    assert (lc.n_rounds, lc.epochs) == (3, 1) and lc.lr == pytest.approx(1e-3)


def test_gui_league_cull_fields_in_config(shared_root):
    """D103 step 4: the cull knobs exist (defaults 4 / 25% / 4) and flow
    through ``_build_config`` — the % field becomes the 0-1 fraction."""
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    app._mode_var.set("League (bot vs bot)")
    app._on_mode_change(None)
    assert app._league_cull_every.get() == 4
    assert app._league_cull_pct.get() == pytest.approx(25.0)
    assert app._league_min_roster.get() == 4
    cfg = app._build_config()
    assert cfg["cull_every"] == 4
    assert cfg["cull_frac"] == pytest.approx(0.25)   # percent -> fraction
    assert cfg["min_roster"] == 4
    app.close()


def test_validate_league_config_checks_cull_knobs(tmp_path):
    """The launcher contract for the population loop: bad cull_every /
    cull_frac / min_roster are loud config errors; the disable values
    (0 / 0.0) are LEGAL (the strict _finite_float bound would reject them)."""
    from gui import launcher
    from league.registry import BotRegistry
    regp = tmp_path / "bots.json"
    ck_a = tmp_path / "a.pt"; ck_a.write_bytes(b"fake")
    ck_b = tmp_path / "b.pt"; ck_b.write_bytes(b"fake")
    reg = BotRegistry(regp)
    reg.add("alpha", str(ck_a)); reg.add("beta", str(ck_b)); reg.save()
    bad = {"registry_path": str(regp), "cull_every": -1,
           "cull_frac": 1.5, "min_roster": 1}
    with pytest.raises(ValueError) as exc:
        launcher.validate_league_config(bad)
    for needle in ("cull_every", "cull_frac", "min_roster"):
        assert needle in str(exc.value), f"expected {needle!r} in {exc.value}"
    # the disable values are legal
    good = launcher.validate_league_config(
        {"registry_path": str(regp), "cull_every": 0, "cull_frac": 0.0,
         "min_roster": 2})
    from league.engine import LeagueConfig
    lc = LeagueConfig.from_dict(good)
    assert lc.cull_every == 0 and lc.cull_frac == 0.0 and lc.min_roster == 2


def test_gui_league_add_copies_source_checkpoint(shared_root, tmp_path, monkeypatch):
    """D103 step 3 (GUI): \"Add bot\" COPIES the source snapshot into a fresh
    league-owned file and registers the COPY (the user's \"make a copy, work
    with that, save it as new\" flow). The source stays PRISTINE, the roster
    never points at the source, and the user's live roster is untouched."""
    from gui import launcher
    from gui.desktop.app import DesktopApp
    from league.registry import BotRegistry

    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    app._mode_var.set("League (bot vs bot)")
    app._on_mode_change(None)

    # hermetic roster (NEVER the user's live league/bots.json)
    regp = tmp_path / "bots.json"
    app._league_roster_path = lambda: regp
    app._league_refresh_roster()
    assert BotRegistry(regp).all() == []

    # the source snapshot (absolute path: the dropdown map falls back to raw)
    src = tmp_path / "src.pt"
    src.write_bytes(b"pristine-weights")

    # copy destination inside the project scratch (clean relpath); clean up
    made: list = []
    def fake_fresh(kind: str = "league", out_dir: str = "scratch"):
        import uuid
        p = Path(launcher.ROOT) / "scratch" / f"league_test_{uuid.uuid4().hex[:4]}.pt"
        made.append(p)
        return p
    monkeypatch.setattr("league.learner.fresh_bot_name", fake_fresh)

    app._league_add_name.set("testbot")
    app._league_add_ckpt.set(str(src))
    app._league_add_bot()

    try:
        bot = BotRegistry(regp).get("testbot")
        copy = Path(bot.ckpt_path)
        if not copy.is_absolute():
            copy = Path(launcher.ROOT) / copy
        assert copy.resolve() != src.resolve(), "roster must point at the COPY"
        assert copy.is_file()
        assert copy.read_bytes() == b"pristine-weights"   # a real copy
        assert src.read_bytes() == b"pristine-weights"    # source untouched
        assert app._league_add_name.get() == ""           # input cleared
        rows = app._league_tree.get_children()
        assert len(rows) == 1
        assert app._league_tree.item(rows[0])["values"][3] == bot.ckpt_path
    finally:
        for p in made:
            p.unlink(missing_ok=True)
    app.close()


def test_gui_league_delete_removes_entry_and_files(shared_root, tmp_path, monkeypatch):
    """D103 step 3 (GUI): \"Delete\" removes the SELECTED bot's roster entry
    AND its checkpoint file (+ meta sidecar), after a confirm dialog.
    Without a selection it refuses (message, no crash)."""
    from gui.desktop import app as app_mod
    from gui.desktop.app import DesktopApp
    from league.registry import BotRegistry

    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    app._mode_var.set("League (bot vs bot)")
    app._on_mode_change(None)

    regp = tmp_path / "bots.json"
    ck = tmp_path / "botfile.pt"
    ck.write_bytes(b"weights")
    meta = Path(str(ck) + ".meta.json")
    meta.write_text("{}", encoding="utf-8")
    reg = BotRegistry(regp)
    reg.add("delme", str(ck))
    reg.save()

    app._league_roster_path = lambda: regp
    app._league_refresh_roster()
    children = app._league_tree.get_children()
    assert len(children) == 1

    # no selection -> refused, nothing deleted
    app._league_tree.selection_remove(children[0])
    app._league_delete_bot()
    assert len(BotRegistry(regp).all()) == 1 and ck.exists()

    # select + confirm -> entry AND files gone
    app._league_tree.selection_set(children[0])
    monkeypatch.setattr(app_mod.messagebox, "askyesno", lambda *a, **k: True)
    app._league_delete_bot()

    assert BotRegistry(regp).all() == []
    assert not ck.exists()
    assert not meta.exists()
    app.close()
