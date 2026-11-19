"""P4.3.4 — BC (imitation) mode in the GUI (D87–D90).

Run from the project root:

    python -m pytest gui/tests/test_p43_3.py -v

What is pinned (CPU-only; the fork server must be up for the live test):

  * BC-mode config validation: every problem listed, missing fixture → 400;
  * unknown mode → 400;
  * BC start/stop LIVE round-trip: a tiny 2-battle CPU run through the real
    detached subprocess reaches a final state with the BC metrics row.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # gui/tests/x.py -> project root
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from gui import launcher, runstate  # noqa: E402
from gui import server as gui_server  # noqa: E402


def wait_for(predicate, timeout: float = 300.0, interval: float = 2.0,
             what: str = "condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what}")


def bc_cfg(**over) -> dict:
    """A minimal valid BC config (CPU, 2 battles, 1 epoch)."""
    cfg = {
        "mode": "bc",
        "n_battles": 2,
        "fixture": "bot/data/replays/gen9ou.jsonl",
        "max_decisions": 40,
        "heldout_frac": 0.5,
        "epochs": 1,
        "batch_size": 8,
        "lr": 3e-3,
        "device": "cpu",
        "seed": 42,
        "ckpt_name": "test_bc.pt",
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


# ── BC-mode validation (fast, nothing is spawned) ────────────────────────

def test_bc_validation_all_problems(client, runs):
    """A BC config with multiple errors returns 400 with EVERY problem listed."""
    bad = bc_cfg()
    bad.update({
        "n_battles": 0,       # must be >= 1
        "epochs": -1,         # must be >= 1
        "lr": -0.1,           # must be > 0
        "device": "gpu",      # must be cpu/cuda
        "seed": -5,           # must be >= 0
    })
    r = client.post("/api/train/start", json={"config": bad})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    for needle in ("n_battles", "epochs", "lr", "device", "seed"):
        assert needle in detail, f"expected {needle!r} in: {detail}"
    # nothing was spawned
    assert list(runs.iterdir()) == []


def test_bc_missing_fixture_400(client, runs, monkeypatch):
    """A BC config with a non-existent fixture path returns 400 BEFORE spawn."""
    bad = bc_cfg(fixture="/nonexistent/path/replays.jsonl")
    r = client.post("/api/train/start", json={"config": bad})
    assert r.status_code == 400, r.text
    assert "fixture" in r.json()["detail"]
    assert list(runs.iterdir()) == []


def test_bc_no_fixture_400(client, runs):
    """A BC config missing the fixture field entirely returns 400."""
    bad = bc_cfg()
    del bad["fixture"]
    r = client.post("/api/train/start", json={"config": bad})
    assert r.status_code == 400, r.text
    assert "fixture" in r.json()["detail"]


def test_unknown_mode_400(client, runs):
    """A config with an unknown mode returns 400."""
    r = client.post("/api/train/start", json={"config": {"mode": "reinforcement",
                                                          "battle_format": "gen9ou"}})
    assert r.status_code == 400, r.text
    assert "unknown mode" in r.json()["detail"]
    assert list(runs.iterdir()) == []


def test_bc_cuda_guard(client, runs, monkeypatch):
    """device=cuda without CUDA available returns 400 before spawn."""
    monkeypatch.setattr(launcher, "cuda_available", lambda: False)
    r = client.post("/api/train/start", json={"config": bc_cfg(device="cuda")})
    assert r.status_code == 400, r.text
    assert "cuda" in r.json()["detail"]
    assert list(runs.iterdir()) == []


def test_bc_valid_config_passes_validation():
    """A well-formed BC config passes validate_bc_config without error."""
    cfg = bc_cfg()
    payload = launcher.validate_bc_config(cfg)
    assert payload["mode"] == "bc"
    assert payload["n_battles"] == 2
    assert payload["epochs"] == 1
    assert payload["device"] == "cpu"
    assert payload["ckpt_name"] == "test_bc.pt"


# ── D96: one-click imitation from the replay library ────────────────────

def test_bc_replay_source_defaults_n_battles():
    """replay_source (no fixture) is valid. When ``n_battles`` is UNSET it
    defaults to the selection size (a one-click 'imitate the top-N' needs only
    one knob); an EXPLICIT n_battles acts as a cap on the materialized set."""
    cfg = bc_cfg(fixture=None, replay_source={"n": 7, "key": "elo"})
    del cfg["n_battles"]   # unset -> should default to the selection size
    payload = launcher.validate_bc_config(cfg)
    assert payload["n_battles"] == 7
    assert payload["replay_source"] == {"n": 7, "key": "elo"}
    assert payload["fixture"] is None
    # an explicit n_battles is respected (a cap on the selection)
    cfg2 = bc_cfg(fixture=None, n_battles=3, replay_source={"n": 7, "key": "elo"})
    assert launcher.validate_bc_config(cfg2)["n_battles"] == 3


def test_bc_replay_source_last_key():
    cfg = bc_cfg(fixture=None, n_battles=3, replay_source={"n": 3, "key": "last"})
    payload = launcher.validate_bc_config(cfg)
    assert payload["replay_source"]["key"] == "last"
    assert payload["n_battles"] == 3


def test_bc_fixture_and_replay_source_conflict(client):
    r = client.post("/api/train/start", json={"config": bc_cfg(
        replay_source={"n": 5, "key": "elo"})})
    assert r.status_code == 400, r.text
    assert "not both" in r.json()["detail"]


def test_bc_replay_source_bad_values(client):
    for bad in ({"n": 0, "key": "elo"}, {"n": 5, "key": "bogus"}, "not-a-dict"):
        r = client.post("/api/train/start", json={"config": bc_cfg(
            fixture=None, replay_source=bad)})
        assert r.status_code == 400, f"bad={bad!r}: {r.text}"


def test_bc_neither_fixture_nor_source_400(client):
    r = client.post("/api/train/start", json={"config": bc_cfg(fixture=None)})
    assert r.status_code == 400, r.text
    assert "fixture" in r.json()["detail"]


def test_api_replays_lists_library(client):
    """GET /api/replays returns 200 + the library summary + a LIGHT metadata
    list (never logs). Works whether the library is empty or populated."""
    r = client.get("/api/replays", params={"order": "last", "n": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["order"] == "last" and body["n"] == 10
    assert "total" in body["summary"]
    assert isinstance(body["battles"], list)
    for b in body["battles"]:
        assert "log" not in b, "list must not carry logs"
        assert "id" in b


def test_api_replays_rejects_bad_args(client):
    assert client.get("/api/replays", params={"order": "bogus", "n": 5}).status_code == 422
    assert client.get("/api/replays", params={"order": "last", "n": 0}).status_code == 422


def test_resolve_fixture_materializes_top_elo(tmp_path):
    """_resolve_fixture (the bc_main wiring, in isolation) materializes the
    top-elo replay from the library into <run_dir>/fixture.jsonl."""
    import gui.bc_main as bcm
    from replays import ReplayLibrary, make_record
    libroot = str(tmp_path / "lib")
    lib = ReplayLibrary(libroot)

    class _B:
        def __init__(self, tag, move):
            self.battle_tag = tag
            self.rating = 1000.0
            self._m = move
        def _build_replay_log(self):
            return f"|turn|1\n|move|p1a|{self._m}\n|win|p1a"
    lib.save(make_record(_B("gen9ou-x", "a"), "gen9ou", players=["a", "b"], elo=100.0))
    lib.save(make_record(_B("gen9ou-y", "b"), "gen9ou", players=["a", "b"], elo=500.0))

    cfg = {"fixture": None, "replay_source": {"n": 1, "key": "elo", "library_root": libroot}}
    out = Path(bcm._resolve_fixture(cfg, tmp_path / "run"))
    assert out.name == "fixture.jsonl" and out.is_file()
    recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["id"] for r in recs] == ["gen9ou-y"]   # the top-elo (500) replay


def test_resolve_fixture_raw_path_passthrough(tmp_path):
    import gui.bc_main as bcm
    fx = tmp_path / "raw.jsonl"
    fx.write_text('{"id": "x", "log": "|win|p1a"}\n', encoding="utf-8")
    assert bcm._resolve_fixture({"fixture": str(fx), "replay_source": None}, tmp_path / "r") == str(fx)


def test_resolve_fixture_none_source_raises(tmp_path):
    import gui.bc_main as bcm
    with pytest.raises(ValueError):
        bcm._resolve_fixture({"fixture": None, "replay_source": None}, tmp_path / "r")


# ── LIVE: BC run through the real detached subprocess ────────────────────

def test_bc_live_start_and_complete(runs):
    """A tiny 2-battle BC CPU run through the REAL detached subprocess:
    start → running → completed (or stopped), with the BC metrics row
    (mode='bc', best_top1, verdict fields present).

    This exercises: gui.bc_main entry, the D86 stop/heartbeat hooks,
    the runstate protocol, and the launcher's mode branching.
    """
    info = launcher.start_trainer(bc_cfg(), runs_dir=runs)
    rd = Path(info["run_dir"])
    assert (rd / "config.json").exists()
    assert (rd / "trainer.log").exists()

    # the config on disk must carry mode='bc'
    cfg_on_disk = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    assert cfg_on_disk["mode"] == "bc"

    # wait for the run to reach a final state (2 battles CPU ≈ 30–60 s + training)
    def final():
        st = launcher.status(runs)
        return st["state"] in runstate.FINAL_STATES
    wait_for(final, 600, what="BC run completion")

    st = launcher.status(runs)
    # the run should complete (or be stopped — both are valid final states)
    assert st["state"] in ("completed", "stopped"), \
        f"state={st['state']} error={st.get('error')}"
    assert st["active"] is False
    assert st["rounds_planned"] == 1  # epochs=1

    # the metrics file should have at least one row (the final BC verdict row)
    rows = runstate.read_metrics(rd)
    assert len(rows) >= 1, f"expected >= 1 metrics row, got {len(rows)}"
    # the LAST row is the BC verdict (appended at the end of bc_main)
    last = rows[-1]
    assert last.get("mode") == "bc", f"last row mode={last.get('mode')}"
    # key BC fields should be present (may be None if stopped during replay)
    for k in ("best_top1", "verdict_pass", "stopped", "wall_s"):
        assert k in last, f"BC metrics row missing key {k!r}: {last}"

    # the trainer log should have the BC start line
    log_text = (rd / "trainer.log").read_text(encoding="utf-8", errors="replace")
    assert "BC start" in log_text or "S4.3 BC" in log_text, \
        f"log does not mention BC start: {log_text[-500:]}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
