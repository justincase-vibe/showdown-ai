"""D98 — native tkinter GUI: headless construction + control-plane wiring.

The tkinter GUI cannot be visually verified in this environment (no display),
so the test constructs the app headlessly (create root, build all widgets,
destroy) and verifies that the config builder produces the right shape for
each mode.

Run from the project root:

    python -m pytest gui/tests/test_desktop.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── headless construction ────────────────────────────────────────────────────


def test_app_constructs_headlessly(shared_root):
    """The app creates a root window, builds all widgets, and destroys
    without error. Proves the widget tree is valid (no missing imports,
    no bad ttk calls)."""
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    # verify key widgets exist
    assert app.root is not None
    assert app._status_label is not None
    assert app._start_btn is not None
    assert app._stop_btn is not None
    assert app._replay_tree is not None
    assert app._ckpt_tree is not None
    assert app._log_text_widget is not None
    # mode labels were set
    assert "Self-play (collect replays)" in app._mode_labels
    assert "Imitation (BC)" in app._mode_labels
    assert "Play against a checkpoint" in app._mode_labels
    # default mode is BC (the most common task)
    assert app._current_mode == "bc"
    # cleanup
    app.close()


def test_mode_switching_rebuilds_fields(shared_root):
    """Switching modes rebuilds the per-mode field frame (no stale widgets)."""
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()

    # default is bc
    assert app._current_mode == "bc"
    assert app._bc_fixture is not None

    # switch to collect
    app._mode_var.set("Self-play (collect replays)")
    app._on_mode_change(None)
    assert app._current_mode == "selfplay-collect"
    assert app._collect_n_battles is not None

    # switch to play
    app._mode_var.set("Play against a checkpoint")
    app._on_mode_change(None)
    assert app._current_mode == "play"
    assert app._play_ckpt is not None

    # switch back to bc
    app._mode_var.set("Imitation (BC)")
    app._on_mode_change(None)
    assert app._current_mode == "bc"
    assert app._bc_fixture is not None

    app.close()


# ── config builder (control-plane wiring) ────────────────────────────────────


def test_config_builder_collect(shared_root):
    """The collect-mode config has all SelfPlayConfig fields + mode."""
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()

    # switch to collect mode
    app._mode_var.set("Self-play (collect replays)")
    app._on_mode_change(None)

    # set some values
    app._collect_n_battles.set(16)
    app._collect_fmt.set("gen9uu")
    app._collect_n_servers.set(3)
    app._collect_per_server.set(2)
    app._collect_port_base.set(50000)
    app._collect_seed.set(99)
    app._collect_max_steps.set(300)

    cfg = app._build_config()
    assert cfg["mode"] == "selfplay-collect"
    assert cfg["n_battles"] == 16
    assert cfg["battle_format"] == "gen9uu"
    assert cfg["n_servers"] == 3
    assert cfg["per_server"] == 2
    assert cfg["port_base"] == 50000
    assert cfg["seed"] == 99
    assert cfg["max_steps"] == 300
    app.close()


def test_config_builder_bc(shared_root):
    """The BC-mode config has all validate_bc_config fields + mode."""
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()

    app._mode_var.set("Imitation (BC)")
    app._on_mode_change(None)

    app._bc_fixture.set("bot/data/replays/gen9ou.jsonl")
    app._bc_n.set(100)
    app._bc_epochs.set(3)
    app._bc_batch.set(16)
    app._bc_lr.set(1e-3)
    app._bc_seed.set(7)
    app._bc_device.set("cpu")

    cfg = app._build_config()
    assert cfg["mode"] == "bc"
    assert cfg["fixture"] == "bot/data/replays/gen9ou.jsonl"
    assert cfg["n_battles"] == 100
    assert cfg["epochs"] == 3
    assert cfg["batch_size"] == 16
    assert cfg["lr"] == 1e-3
    assert cfg["seed"] == 7
    assert cfg["device"] == "cpu"
    # required fields that the GUI doesn't expose (use defaults)
    assert cfg["max_decisions"] == 40
    assert cfg["heldout_frac"] == 0.5
    app.close()


def test_config_builder_play(shared_root):
    """The play-mode config has all validate_play_config fields + mode."""
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()

    app._mode_var.set("Play against a checkpoint")
    app._on_mode_change(None)

    app._play_ckpt.set("scratch/50b_bc_a3f2.pt")
    app._play_fmt.set("gen9ou")
    app._play_bot_user.set("testbot")
    app._play_you.set("tester")
    app._play_device.set("cpu")
    app._play_greedy.set(True)

    cfg = app._build_config()
    assert cfg["mode"] == "play"
    assert cfg["ckpt_path"] == "scratch/50b_bc_a3f2.pt"
    assert cfg["battle_format"] == "gen9ou"
    assert cfg["bot_user"] == "testbot"
    assert cfg["you"] == "tester"
    assert cfg["device"] == "cpu"
    assert cfg["greedy"] is True
    assert cfg["n_battles"] == 0  # play until stopped
    app.close()


def test_config_builder_eval(shared_root):
    """The eval-mode config has all validate_eval_config fields + mode."""
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()

    app._mode_var.set("Eval vs random")
    app._on_mode_change(None)

    app._eval_ckpt.set("scratch/50b_bc_a3f2.pt")
    app._eval_n.set(20)
    app._eval_fmt.set("gen9ou")
    app._eval_seed.set(123)
    app._eval_device.set("cpu")

    cfg = app._build_config()
    assert cfg["mode"] == "eval"
    assert cfg["ckpt_path"] == "scratch/50b_bc_a3f2.pt"
    assert cfg["n_battles"] == 20
    assert cfg["battle_format"] == "gen9ou"
    assert cfg["seed"] == 123
    assert cfg["device"] == "cpu"
    app.close()


def test_collect_defaults_avoid_port_8000(shared_root):
    """The collect-mode defaults use port_base=49960 (not :8000 — the
    user's personal Showdown server)."""
    from gui.desktop.app import DesktopApp
    from gui.desktop.app import COLLECT_DEFAULTS
    assert COLLECT_DEFAULTS["port_base"] == 49960
    assert COLLECT_DEFAULTS["n_servers"] == 2
    assert COLLECT_DEFAULTS["per_server"] == 2

    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    # switch to collect mode to access the collect fields
    app._mode_var.set("Self-play (collect replays)")
    app._on_mode_change(None)
    assert app._collect_port_base.get() == 49960
    assert app._collect_n_servers.get() == 2
    assert app._collect_per_server.get() == 2
    app.close()


def test_status_bar_poll_does_not_crash(shared_root):
    """The poll loop reads launcher.status and updates the status bar
    without crashing (even when no run is active)."""
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    # call _poll directly (it would normally be called via root.after)
    app._poll()
    # the status label should have been updated
    text = app._status_label.cget("text")
    assert "Status:" in text
    app.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
