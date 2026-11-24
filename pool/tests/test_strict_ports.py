"""Hermetic tests for ``strict_ports`` (fail loudly on a foreign port).

Background (2026-11-24 incident): two concurrent learnability runs shared the
hardcoded ports 49990/50006. The pool's health check answered "healthy" for a
server owned by the OTHER run, so one run silently attached its battles to
foreign servers, and the shared-fork state-file race (ENOENT rename of
seasons.json.NEW / lastbattle.txt.NEW) crashed servers mid-battle.
``strict_ports`` makes a run ABORT instead of attaching. No real servers are
spawned here: the health check is monkeypatched.
"""
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import pool.server_pool as sp
from pool import PoolConfig, ServerPool, port_serving


def _foreign(monkeypatch) -> None:
    """Pretend every probed port already serves a Showdown instance."""
    monkeypatch.setattr(
        sp, "port_serving",
        lambda port, host="localhost", timeout=2.0: True)


def test_strict_aborts_and_never_spawns_on_foreign_ports(monkeypatch):
    _foreign(monkeypatch)
    pool = ServerPool(PoolConfig(n_servers=2, strict_ports=True))

    def no_spawn(port):
        raise AssertionError("strict mode must not spawn when the port is foreign")

    monkeypatch.setattr(pool, "_spawn", no_spawn)
    with pytest.raises(SystemExit) as ei:
        pool.start_all(wait_healthy=False)
    msg = str(ei.value)
    assert "strict_ports" in msg
    assert "netstat" in msg          # the error tells the user how to find the owner
    assert pool._procs == {}         # nothing was started, nothing to leak


def test_non_strict_keeps_the_legacy_silent_reuse(monkeypatch):
    """Documents the contrast: without strict_ports the legacy path silently
    reuses a foreign server (no raise, nothing spawned, nothing killed)."""
    _foreign(monkeypatch)
    pool = ServerPool(PoolConfig(n_servers=2, strict_ports=False))

    def no_spawn(port):
        raise AssertionError("legacy path reuses, it does not spawn")

    monkeypatch.setattr(pool, "_spawn", no_spawn)
    status = pool.start_all(wait_healthy=False)
    assert status == {p: True for p in pool.cfg.ports}
    assert pool._procs == {}         # it "reused" the foreign servers


def test_strict_passes_when_ports_are_free(monkeypatch):
    monkeypatch.setattr(
        sp, "port_serving",
        lambda port, host="localhost", timeout=2.0: False)
    pool = ServerPool(PoolConfig(n_servers=1, strict_ports=True))
    spawned = []

    class FakeProc:
        def poll(self):
            return None

    monkeypatch.setattr(pool, "_spawn", lambda port: spawned.append(port) or FakeProc())
    # no SystemExit:
    pool.start_all(wait_healthy=False)
    assert spawned == pool.cfg.ports


# ── port_serving itself (real HTTP, hermetic) ─────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def test_port_serving_true_for_a_live_instance():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        assert port_serving(port, host="127.0.0.1") is True
    finally:
        srv.shutdown()


def test_port_serving_false_for_a_closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                       # nobody is listening now
    assert port_serving(port, host="127.0.0.1", timeout=0.5) is False
