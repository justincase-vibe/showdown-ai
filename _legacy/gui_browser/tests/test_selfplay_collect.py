"""D98 — selfplay-collect (capture) mode in the GUI control plane.

The self-play → replay-capture path (feature 1 → 2) was the **biggest gap**
identified in D97: the capture engine (`control/selfplay_main.py`) existed
and was tested, but `launcher.start_trainer` had no mode that spawned it,
so the replay library only filled via a hand-run engine.

D98 closes the gap:
  * `launcher.validate_selfplay_config` — loud validation (every problem listed)
  * `launcher.start_trainer` routes `selfplay-collect` → `control.selfplay_main`
  * `gui/server.py::api_start` handles `selfplay-collect` (CPU-only, no device check)
  * `launcher.browse_checkpoints` — shared checkpoint browser (extracted from
    `gui/server.py::api_browse` so the native GUI reuses the SAME D97-fixed
    meta-reading logic)

Run from the project root (CPU-only, nothing is spawned):

    python -m pytest gui/tests/test_selfplay_collect.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[2]  # gui/tests/x.py -> project root
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from gui import launcher, runstate  # noqa: E402
from gui import server as gui_server  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────────

def sp_cfg(**over) -> dict:
    """A minimal valid selfplay-collect config (CPU, 8 battles, 2x2 pool)."""
    cfg = {
        "mode": "selfplay-collect",
        "n_battles": 8,
        "battle_format": "gen9ou",
        "n_servers": 2,
        "per_server": 2,
        "port_base": 49960,
        "seed": 42,
        "max_steps": 200,
        "max_consecutive_errors": 3,
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


# ── validate_selfplay_config (fast, nothing is spawned) ──────────────────────

def test_sp_collect_valid_config_passes():
    """A well-formed selfplay-collect config passes validate_selfplay_config."""
    cfg = sp_cfg()
    payload = launcher.validate_selfplay_config(cfg)
    assert payload["mode"] == "selfplay-collect"
    assert payload["n_battles"] == 8
    assert payload["battle_format"] == "gen9ou"
    assert payload["n_servers"] == 2
    assert payload["per_server"] == 2
    assert payload["port_base"] == 49960
    # SelfPlayConfig fields should all be present (dataclass round-trip)
    assert "seed" in payload
    assert "max_steps" in payload
    assert "max_consecutive_errors" in payload
    # the pool fields
    assert "save_replays" in payload
    assert "boot_timeout" in payload
    assert "n_workers" in payload


def test_sp_collect_defaults_applied():
    """Missing fields get SelfPlayConfig defaults (the dataclass handles it)."""
    cfg = {"mode": "selfplay-collect"}  # minimal: only the mode
    payload = launcher.validate_selfplay_config(cfg)
    assert payload["n_battles"] == 8       # SelfPlayConfig default
    assert payload["battle_format"] == "gen9ou"
    assert payload["n_servers"] == 2
    assert payload["per_server"] == 2


def test_sp_collect_bad_config_all_problems(client, runs):
    """A config with multiple errors returns 400 with EVERY problem listed."""
    bad = sp_cfg()
    bad.update({
        "n_battles": 0,       # must be >= 1
        "battle_format": "",  # must be non-empty
        "n_servers": -1,      # must be >= 1
        "per_server": 0,      # must be >= 1
        "port_base": 0,       # must be >= 1
        "seed": -5,           # must be >= 1
        "max_steps": 0,       # must be >= 1
        "max_consecutive_errors": 0,  # must be >= 1
    })
    r = client.post("/api/train/start", json={"config": bad})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    for needle in ("n_battles", "battle_format", "n_servers", "per_server",
                    "port_base", "seed", "max_steps", "max_consecutive_errors"):
        assert needle in detail, f"expected {needle!r} in: {detail}"
    # nothing was spawned
    assert list(runs.iterdir()) == []


def test_sp_collect_unknown_field_ignored():
    """Unknown fields are silently dropped (SelfPlayConfig.from_dict contract)."""
    cfg = sp_cfg(unknown_field="should be ignored")
    payload = launcher.validate_selfplay_config(cfg)
    assert "unknown_field" not in payload


# ── start_trainer routing (Popen faked — nothing is actually launched) ────────

def test_start_trainer_sp_collect_routes_to_control_selfplay_main(client, runs, monkeypatch):
    """start_trainer routes selfplay-collect → control.selfplay_main with a
    config.json carrying the SelfPlayConfig dict. Popen is faked so this is
    fast and nothing is actually launched."""
    class _FakeProc:
        pid = 9999
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: _FakeProc())

    cfg = sp_cfg()
    r = client.post("/api/train/start", json={"config": cfg})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["started"] is True and body["run_dir"]

    rd = Path(body["run_dir"])
    assert (rd / "config.json").is_file()
    assert (rd / "trainer.log").is_file()

    # the config on disk must carry the SelfPlayConfig fields
    on_disk = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    assert on_disk["mode"] == "selfplay-collect"
    assert on_disk["n_battles"] == 8
    assert on_disk["battle_format"] == "gen9ou"
    assert on_disk["n_servers"] == 2
    assert on_disk["per_server"] == 2
    assert on_disk["port_base"] == 49960
    assert on_disk["seed"] == 42
    assert on_disk["max_steps"] == 200
    # SelfPlayConfig round-trip fields (from the dataclass)
    assert "save_replays" in on_disk
    assert "boot_timeout" in on_disk


def test_start_trainer_sp_collect_uses_correct_entry(monkeypatch, runs):
    """Verify the Popen command targets control.selfplay_main (not gui.trainer_main)."""
    captured_cmd: list = []

    class _FakeProc:
        pid = 8888
        def __init__(self):
            pass

    def _fake_popen(cmd, *args, **kwargs):
        captured_cmd.extend(cmd)
        return _FakeProc()

    monkeypatch.setattr(launcher.subprocess, "Popen", _fake_popen)

    cfg = sp_cfg()
    launcher.start_trainer(cfg, runs_dir=runs)

    # the command should be: python -m control.selfplay_main --config ... --run-dir ...
    assert "control.selfplay_main" in captured_cmd, \
        f"expected control.selfplay_main in cmd: {captured_cmd}"


def test_sp_collect_no_device_check():
    """selfplay-collect is always CPU (pool-based) — no device field needed."""
    cfg = sp_cfg()
    # no 'device' key at all — that's fine, the validator doesn't require it
    payload = launcher.validate_selfplay_config(cfg)
    assert "device" not in payload  # SelfPlayConfig has no device field


# ── browse_checkpoints (shared, extracted from api_browse) ───────────────────

def test_browse_checkpoints_returns_list():
    """browse_checkpoints returns a list (possibly empty) of checkpoint dicts."""
    results = launcher.browse_checkpoints()
    assert isinstance(results, list)
    for cp in results:
        assert "path" in cp
        assert "name" in cp
        assert "size_mb" in cp


def test_browse_checkpoints_meta_fields():
    """Each checkpoint dict carries the D97 meta fields (type, top1, verdict)."""
    results = launcher.browse_checkpoints()
    for cp in results:
        assert "type" in cp
        assert "n_battles" in cp
        assert "verdict" in cp
        assert "top1" in cp


def test_api_browse_uses_shared_function(client):
    """The HTTP endpoint and the shared function return the same data."""
    r = client.get("/api/checkpoints/browse")
    assert r.status_code == 200, r.text
    http_results = r.json()["checkpoints"]
    shared_results = launcher.browse_checkpoints()
    # same length and same first entry (if any)
    assert len(http_results) == len(shared_results)
    if shared_results:
        assert http_results[0]["name"] == shared_results[0]["name"]
        assert http_results[0]["path"] == shared_results[0]["path"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
