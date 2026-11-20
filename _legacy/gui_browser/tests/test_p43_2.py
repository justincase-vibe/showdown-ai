"""P4.3.2 gate — the FastAPI control plane + WebSocket stream + dashboard.

Run from the project root:

    python -m pytest gui/tests/test_p43_2.py -v

What is pinned (CPU-only; the fork server must be up for the two live tests):

  * the app is a THIN control plane: / serves the offline dashboard, /api/*
    answers from the run protocol on disk, start/stop delegate to
    gui.launcher (400 = bad config with EVERY problem listed; 400 = cuda
    requested without CUDA — checked BEFORE anything is spawned; 409 = a run
    is already active / nothing to stop);
  * the /files endpoint is traversal-guarded at BOTH levels (run_id AND path)
    and serves real bytes;
  * /ws/stream pushes a status frame on connect, a metrics frame when a
    round line lands, and never re-pushes the same lines (file-tail);
  * GUI-restart survival at the HTTP level: a brand-new app instance over
    the same runs dir sees the SAME active run and can stop it;
  * live: one 1-round run through the real detached subprocess reaches
    'completed' with metrics + checkpoint served over HTTP, and the
    2-round run's graceful stop endpoint lands 'stopped' at the round
    boundary (the P4.3.1 contract, exercised through the HTTP surface).

Two live battles total — the established 1-battle smoke pattern (p41_gate).
"""
from __future__ import annotations

import json
import math
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


def small_cfg(**over) -> dict:
    cfg = {"battle_format": "gen9ou", "battles_per_round": 1, "n_rounds": 1,
           "max_steps": 200, "seed": 42, "device": "cpu", "save_every": 1,
           "epochs": 1, "lr": 3e-4}
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


# ── no-runs baseline (fast) ─────────────────────────────────────────────────

def test_pages_and_status_no_runs(client, runs):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Training Console" in r.text  # the real dashboard, not a stub

    st = client.get("/api/status").json()
    assert st["active"] is False and st["state"] is None and st["run_id"] is None
    assert isinstance(st["cuda_available"], bool)
    assert st["metrics_tail"] == [] and st["checkpoints"] == []

    assert client.get("/api/runs").json() == {"runs": []}
    assert client.get("/api/metrics").json() == {"run_id": None, "rows": []}
    assert client.get("/api/checkpoints").json() == {"run_id": None, "checkpoints": []}


# ── start validation (fast, nothing is spawned) ─────────────────────────────

def test_start_bad_config_400_all_problems(client, runs):
    bad = small_cfg()
    bad.update({"battles_per_round": 0, "device": "gpu", "gamma": 2.0, "epochs": "x"})
    r = client.post("/api/train/start", json={"config": bad})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    for needle in ("battles_per_round", "device", "gamma", "epochs"):
        assert needle in detail, f"expected {needle!r} in: {detail}"
    # nothing was spawned
    assert list(runs.iterdir()) == []

    # NOTE: a bare {} is VALID by design (TrainConfig defaults fill it in).
    # To exercise the battle_format check, submit an empty string:
    r2 = client.post("/api/train/start", json={"config": {"battle_format": ""}})
    assert r2.status_code == 400 and "battle_format" in r2.json()["detail"]

    # malformed body (no 'config' key at all) -> 422, not a crash
    r3 = client.post("/api/train/start", json={"not_config": 1})
    assert r3.status_code == 422


def test_cuda_guard_400_before_spawn(client, runs, monkeypatch):
    monkeypatch.setattr(launcher, "cuda_available", lambda: False)
    r = client.post("/api/train/start", json={"config": small_cfg(device="cuda")})
    assert r.status_code == 400, r.text
    assert "cuda" in r.json()["detail"]
    # the guard fired BEFORE the launcher spawned anything
    assert list(runs.iterdir()) == []


def test_stop_with_nothing_active_409(client):
    r = client.post("/api/train/stop", json={})
    assert r.status_code == 409, r.text
    assert "no active" in r.json()["detail"].lower()


# ── files endpoint: guard + real bytes (fast, no battles) ───────────────────

def test_files_serves_and_traversal_guarded(client, runs):
    rd = runs / "20250101T000000Z"
    (rd / "checkpoints").mkdir(parents=True)
    (rd / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    payload = b"torch-payload-bytes-1234"
    (rd / "checkpoints" / "selfplay_round1.pt").write_bytes(payload)

    r = client.get("/files/20250101T000000Z/checkpoints/selfplay_round1.pt")
    assert r.status_code == 200 and r.content == payload

    # unknown run / unknown file -> 404
    assert client.get("/files/nope/checkpoints/x.pt").status_code == 404
    assert client.get("/files/20250101T000000Z/checkpoints/missing.pt").status_code == 404

    # traversal attempts at BOTH levels are refused
    assert gui_server._run_dir_guarded(runs, "../outside") is None
    assert gui_server._run_dir_guarded(runs, "20250101T000000Z/..") is None
    assert gui_server._run_dir_guarded(runs, "missing") is None
    assert gui_server._run_dir_guarded(runs, "20250101T000000Z") == rd.resolve()
    assert client.get("/files/..%2F..%2Fsecret.txt").status_code == 404
    assert client.get(f"/files/20250101T000000Z/..%2F..%2F..%2Fsecret.txt").status_code == 404

    # metrics/checkpoints endpoints honour an explicit run_id + the n cap
    runstate.append_metric(rd, {"round": 1, "loss": 0.5})
    runstate.append_metric(rd, {"round": 2, "loss": 0.4})
    m = client.get("/api/metrics", params={"run_id": "20250101T000000Z", "n": 1}).json()
    assert m["run_id"] == "20250101T000000Z" and [x["round"] for x in m["rows"]] == [2]
    c = client.get("/api/checkpoints", params={"run_id": "20250101T000000Z"}).json()
    assert c["checkpoints"][0]["name"] == "selfplay_round1.pt"
    assert c["checkpoints"][0]["size_bytes"] == len(payload)


# ── live: full HTTP round-trip through the REAL detached trainer ────────────

def _recv_until(ws, type_name: str, max_frames: int = 10) -> dict:
    """Receive WS frames until one of ``type_name`` arrives (status frames
    are interleaved on every stable change — skip them)."""
    for _ in range(max_frames):
        fr = ws.receive_json()
        if fr.get("type") == type_name:
            return fr
    raise AssertionError(f"no {type_name!r} frame within {max_frames} frames")


def test_live_start_http_roundtrip_and_ws(runs):
    """start (1 battle) via HTTP -> 200 -> second start 409 -> a FRESH app
    instance (GUI restart) sees the same active run -> run completes ->
    metrics + checkpoint served over HTTP -> the WS stream delivered a
    status frame on connect AND a metrics frame for round 1."""
    app = gui_server.create_app(runs_dir=runs)
    with TestClient(app) as client:
        r = client.post("/api/train/start", json={"config": small_cfg()})
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["started"] is True and info["run_id"] and info["pid"]
        run_id = info["run_id"]

        # while active: a second start is refused
        wait_for(lambda: client.get("/api/status").json()["active"], 60,
                 what="run active")
        r2 = client.post("/api/train/start", json={"config": small_cfg()})
        assert r2.status_code == 409 and "already active" in r2.json()["detail"]

        # GUI restart survival at the HTTP level: a BRAND-NEW app over the
        # same runs dir sees the same active run from disk alone
        with TestClient(gui_server.create_app(runs_dir=runs)) as fresh:
            st = fresh.get("/api/status").json()
            assert st["active"] is True and st["run_id"] == run_id
            assert st["pid"] == info["pid"]

        # the WS stream: status frame on connect, metrics frame by the time
        # the run completes
        with client.websocket_connect("/ws/stream") as ws:
            first = ws.receive_json()
            assert first["type"] == "status" and first["status"]["run_id"] == run_id

            def done():
                return client.get("/api/status").json()["state"] in runstate.FINAL_STATES
            wait_for(done, 420, what="run completion")
            st = client.get("/api/status").json()
            assert st["state"] == "completed", f"state={st['state']} err={st.get('error')}"

            mfr = _recv_until(ws, "metrics")
            assert mfr["run_id"] == run_id and len(mfr["rows"]) == 1
            row = mfr["rows"][0]
            assert row["round"] == 1 and row["battles"] == 1
            for k in ("loss", "value", "entropy"):
                assert row.get(k) is not None and math.isfinite(row[k])

        # over plain HTTP too: metrics tail + checkpoint bytes
        m = client.get("/api/metrics").json()
        assert m["run_id"] == run_id and len(m["rows"]) == 1
        c = client.get("/api/checkpoints").json()
        assert c["checkpoints"][0]["name"] == "selfplay_round1.pt"
        f = client.get(f"/files/{run_id}/checkpoints/selfplay_round1.pt")
        assert f.status_code == 200 and len(f.content) > 0

        # and now that nothing is active: stop is a 409, not a crash
        s = client.post("/api/train/stop", json={"force": True})
        assert s.status_code == 409


def test_live_stop_endpoint_graceful(runs):
    """The stop ENDPOINT (graceful) lands at the round boundary: round 1
    completes, round 2 is skipped, final state 'stopped' — the P4.3.1
    contract through the HTTP surface."""
    app = gui_server.create_app(runs_dir=runs)
    with TestClient(app) as client:
        r = client.post("/api/train/start", json={"config": small_cfg(n_rounds=2)})
        assert r.status_code == 200, r.text
        run_id = r.json()["run_id"]
        rd = runs / run_id

        # wait until the trainer is provably inside round 1 (heartbeat fired)
        def in_round1():
            hb = runstate.read_heartbeat(rd)
            return bool(hb.get("note")) and hb.get("round") == 0
        wait_for(in_round1, 90, what="round-1 heartbeat")

        s = client.post("/api/train/stop", json={})
        assert s.status_code == 200, s.text
        assert s.json()["state"] == "stopped"

        st = client.get("/api/status").json()
        assert st["state"] == "stopped" and st["active"] is False
        rows = runstate.read_metrics(rd)
        assert len(rows) == 1 and rows[0]["round"] == 1
        assert not runstate.pid_alive(st["pid"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
