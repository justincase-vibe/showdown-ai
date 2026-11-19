"""P4.3.5 — PLAY mode (mode="play") in the GUI: "you vs the checkpoint" (D92.3).

The interactive half of "fight a checkpoint": the GUI spawns ``net.human_play``
(a poke-env Player driven by the loaded PokeNet) as a DETACHED process, and the
user plays against it in the local fork's browser tab (``http://localhost:8000``).

Run from the project root (CPU-only, NO fork needed — nothing here connects):

    python -m pytest gui/tests/test_p43_4.py -v

What is pinned:

  * ``validate_play_config``: bad ckpt -> 400 (every problem listed, not just the
    first); a good config passes and normalizes (n_battles=0 allowed, greedy
    default True, bot_user default, ``you`` optional);
  * ``mode="play"`` routes to the ``net.human_play`` entry (Popen command
    captured) and writes ``mode="play"`` into config.json on disk;
  * unknown mode -> 400 and now lists all FOUR modes (selfplay/bc/eval/play);
  * the checkpoint browser (``GET /api/checkpoints/browse``) surfaces the real
    50b BC checkpoint that the user wants to play against.

The LIVE "you actually play a battle in the browser" is a human-in-the-loop
check (D92.2's live smoke already proved the bot wins a full battle vs random);
it is intentionally NOT automated here — a headless test cannot be "you".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # gui/tests/x.py -> project root
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from gui import launcher, runstate  # noqa: E402
from gui import server as gui_server  # noqa: E402

#: a checkpoint the user actually trained and wants to play (present in scratch/).
KNOWN_CKPT = "scratch/2026-11-19_bc_50b.pt"


def play_cfg(**over) -> dict:
    """A minimal valid PLAY config (CPU, unlimited battles, greedy)."""
    cfg = {
        "mode": "play",
        "ckpt_path": KNOWN_CKPT,
        "n_battles": 0,          # play until I stop it
        "device": "cpu",
        "greedy": True,
        "battle_format": "gen9ou",
        "bot_user": "checkpointbot",
        "you": "ttvJustinCase",
        "team_seed": None,
    }
    cfg.update(over)
    return cfg


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def client(runs):
    app = gui_server.create_app(runs_dir=runs)
    with TestClient(app) as c:
        yield c


# ── validate_play_config (fast, nothing is spawned) ────────────────────────

def test_play_missing_ckpt_400(client, runs):
    """A PLAY config with a non-existent checkpoint is a 400 BEFORE spawn."""
    r = client.post("/api/train/start", json={"config": play_cfg(ckpt_path="/nope/missing.pt")})
    assert r.status_code == 400, r.text
    assert "ckpt_path" in r.json()["detail"]
    # nothing was spawned
    assert list(runs.iterdir()) == []


def test_play_no_ckpt_400(client, runs):
    """A PLAY config missing the checkpoint field entirely is a 400."""
    cfg = play_cfg()
    del cfg["ckpt_path"]
    r = client.post("/api/train/start", json={"config": cfg})
    assert r.status_code == 400, r.text
    assert "ckpt_path" in r.json()["detail"]


def test_play_all_problems_listed(client, runs):
    """A PLAY config with many errors returns 400 with EVERY problem listed."""
    bad = play_cfg()
    bad.update({
        "n_battles": -1,       # must be int >= 0
        "device": "gpu",       # must be cpu/cuda
        "greedy": "yes",       # must be a bool
        "bot_user": "",        # must be non-empty
        "you": 5,              # must be a str (empty OK, non-str not)
        "ckpt_path": "",       # must be a path
        "team_seed": "x",      # must be int >= 0 or null
    })
    r = client.post("/api/train/start", json={"config": bad})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    for needle in ("n_battles", "device", "greedy", "bot_user", "you", "ckpt_path", "team_seed"):
        assert needle in detail, f"expected {needle!r} in: {detail}"
    assert list(runs.iterdir()) == []


def test_play_valid_config_passes():
    """A well-formed PLAY config normalizes (n_battles=0 allowed, greedy True,
    bot_user defaulted, ``you`` optional, team_seed passthrough)."""
    payload = launcher.validate_play_config(play_cfg())
    assert payload["mode"] == "play"
    assert payload["n_battles"] == 0
    assert payload["greedy"] is True
    assert payload["bot_user"] == "checkpointbot"
    assert payload["you"] == "ttvJustinCase"
    assert payload["device"] == "cpu"
    assert payload["team_seed"] is None


def test_play_defaults_and_optional_you():
    """Only ckpt_path is required: the rest falls back to sane defaults, and an
    empty ``you`` is a VALID value (the bot works without knowing your name)."""
    payload = launcher.validate_play_config({"mode": "play", "ckpt_path": KNOWN_CKPT})
    assert payload["n_battles"] == 0
    assert payload["greedy"] is True
    assert payload["bot_user"] == "checkpointbot"
    assert payload["you"] == ""          # optional — NOT a validation error
    assert payload["battle_format"] == "gen9ou"


def test_play_n_battles_must_not_be_bool():
    """A bool is not a valid n_battles (bool is an int subclass in Python)."""
    with pytest.raises(ValueError, match="n_battles"):
        launcher.validate_play_config(play_cfg(n_battles=True))


# ── mode="play" routing: spawns net.human_play (Popen captured, no real proc) ─

def test_play_routes_to_human_play_entry(runs, monkeypatch):
    """mode="play" validates, then spawns the ``net.human_play`` entry and
    writes ``mode="play"`` into config.json (rounds_planned = n_battles)."""
    captured = {}

    class FakeProc:
        pid = 999999  # a PID almost certainly not running -> pid_alive False

    def fake_popen(cmd, **kw):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    info = launcher.start_trainer(play_cfg(n_battles=3), runs_dir=runs)
    assert "net.human_play" in captured["cmd"], captured["cmd"]
    assert "-m" in captured["cmd"]

    rd = Path(info["run_dir"])
    on_disk = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    assert on_disk["mode"] == "play"
    assert on_disk["n_battles"] == 3

    st = runstate.read_status(rd)
    assert st["state"] == "starting"
    assert st["rounds_planned"] == 3
    assert st["pid"] == 999999


def test_play_unlimited_shows_zero_rounds_planned(runs, monkeypatch):
    """n_battles=0 (play until stopped) is a valid run; rounds_planned=0 on disk
    (the dashboard renders that as ∞)."""
    class FakeProc:
        pid = 999998

    def fake_popen(cmd, **kw):
        return FakeProc()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    info = launcher.start_trainer(play_cfg(n_battles=0), runs_dir=runs)
    rd = Path(info["run_dir"])
    on_disk = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    assert on_disk["n_battles"] == 0
    assert runstate.read_status(rd)["rounds_planned"] == 0


# ── mode branching + checkpoint browser ──────────────────────────────────────

def test_unknown_mode_lists_all_four(client, runs):
    """An unknown mode is a 400 and now names all FOUR supported modes."""
    r = client.post("/api/train/start", json={"config": {"mode": "reinforcement"}})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "unknown mode" in detail
    for m in ("selfplay", "bc", "eval", "play"):
        assert m in detail, f"expected {m!r} in: {detail}"
    assert list(runs.iterdir()) == []


def test_browse_surfaces_known_checkpoint(client):
    """The checkpoint browser lists the user's real 50b BC checkpoint by path."""
    r = client.get("/api/checkpoints/browse")
    assert r.status_code == 200, r.text
    paths = [c["path"] for c in r.json()["checkpoints"]]
    # the known ckpt is under scratch/ (relative to the project root)
    assert any(p.replace("\\", "/").endswith("2026-11-19_bc_50b.pt") for p in paths), \
        f"known checkpoint not in browse: {paths}"


def test_play_cuda_guard(client, runs, monkeypatch):
    """device=cuda without CUDA available is a 400 before spawn (like the other
    modes) — the bot must not try to load a torch model onto a missing device."""
    monkeypatch.setattr(launcher, "cuda_available", lambda: False)
    r = client.post("/api/train/start", json={"config": play_cfg(device="cuda")})
    assert r.status_code == 400, r.text
    assert "cuda" in r.json()["detail"]
    assert list(runs.iterdir()) == []


# ── regression: a non-ASCII log char must never kill a GUI-spawned bot ─────

def test_harden_stdio_is_callable_and_idempotent():
    """harden_stdio() must be safe to call (and call twice) and must not raise,
    even where the current stdout is a normal reconfigurable stream."""
    runstate.harden_stdio()
    runstate.harden_stdio()  # idempotent — a second call must not error


def test_play_bot_survives_non_ascii_log(tmp_path):
    """REGRESSION (caught by the D92.3 live smoke, fixed 2026-11-19).

    A GUI-launched subprocess inherits the console codepage — cp1252 on this box.
    The play bot's start log contained ``\u221e`` (``n_battles or '\u221e'``), so the
    first ``print`` raised ``UnicodeEncodeError`` and the bot died the instant the
    GUI spawned it. The fix is two-fold: the log line uses ASCII ``unlimited``,
    AND every entry point calls ``runstate.harden_stdio()`` so ANY future
    non-ASCII log character can never take a live battle down.

    This pins the hardening by spawning a REAL child exactly like the launcher
    does (stdout -> a binary file) with ``PYTHONIOENCODING=cp1252`` FORCED, so
    the test is meaningful regardless of the host locale: without
    ``harden_stdio()`` the child crashes (rc != 0); with it, it survives.
    """
    import os
    import subprocess
    import sys

    log = tmp_path / "child.log"
    code = (
        "from gui import runstate; "
        "runstate.harden_stdio(); "
        "print('ok n=\\u221e device=cpu', flush=True)"
    )
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    with open(log, "ab", buffering=0) as f:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(ROOT), env=env)
    out = log.read_text(encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"cp1252 child crashed on a non-ASCII log char (rc={proc.returncode}): {out}"
    assert "ok n=" in out, out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
