"""C1 — the persistent Node server mode (``node_calc.js --server``).

The one-shot CLI (``node node_calc.js``) reads all of stdin and exits. The
server mode keeps ONE process alive across many line-delimited requests — the
root-cause fix for the league crash (per-request ``node`` spawn churn under 8
concurrent battle threads, which hard-crashed 0xC0000409). This test pins the
server contract at the node boundary (before the Python pool, C2):

  * a damage request returns the SAME result as the one-shot CLI;
  * a speed request returns the SAME result as the one-shot CLI;
  * an ARRAY (batch) request returns an array IN ORDER (first matches the single);
  * a bad request line returns ``{error}`` for THAT line and the server
    KEEPS RUNNING (a later valid line still works) — unlike one-shot exit 1;
  * closing stdin -> a clean exit 0.

Run from the project root:

    python -m pytest damage/tests/test_node_server.py -v
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from damage import calc

HERE = Path(__file__).resolve()
NODE = "node"
SCRIPT = str(HERE.parent.parent / "node_calc.js")  # damage/tests/x.py -> damage/node_calc.js

GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}}
TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}
PIKA = {"name": "Pikachu", "level": 50}


def _req_damage():
    return {"type": "damage", "gen": 9, "attacker": GARCH, "defender": TITAN,
            "move": {"name": "earthquake"}}


def _req_speed():
    return {"type": "speed", "gen": 9, "pokemon": PIKA}


class _Server:
    """A line-based RPC client for ``node_calc.js --server`` (test helper).

    One request per line out, one line in. ``readline()`` blocks until the
    server emits a full ``...\\n`` line — which the server does per request
    (verified: it flushes each response, so no request/response interleaving)."""

    def __init__(self):
        self.p = subprocess.Popen(
            [NODE, SCRIPT, "--server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)

    def request(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if line == "":
            raise RuntimeError("server closed stdout: " + (self.p.stderr.read() or ""))
        return json.loads(line)

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def test_server_matches_oneshot_and_survives_bad_line():
    srv = _Server()
    try:
        # damage: identical to the one-shot CLI
        d = srv.request(_req_damage())
        ref = calc.calc_damage(GARCH, TITAN, "earthquake")
        assert d["min"] == ref["min"] and d["max"] == ref["max"], (d, ref)

        # speed: identical to the one-shot CLI
        s = srv.request(_req_speed())
        ref_s = calc.effective_speed(PIKA)
        assert s["speed"] == ref_s, (s, ref_s)

        # batch: an array IN ORDER; the first entry matches the single damage
        b = srv.request([_req_damage(), _req_speed()])
        assert isinstance(b, list) and len(b) == 2, b
        assert b[0]["min"] == ref["min"] and b[0]["max"] == ref["max"]
        assert b[1]["speed"] == ref_s

        # a bad request line -> {error} for THAT line ...
        bad = srv.request({"type": "bogus_type"})
        assert "error" in bad, bad

        # ... and the server KEEPS RUNNING (a later valid line still works)
        again = srv.request(_req_damage())
        assert again["min"] == ref["min"] and again["max"] == ref["max"]
    finally:
        srv.close()


def test_server_clean_exit_on_eof():
    srv = _Server()
    srv.request(_req_damage())
    srv.close()
    assert srv.p.returncode == 0, f"exit={srv.p.returncode}"


def test_one_shot_unchanged_regression():
    """The NO-flag one-shot CLI is byte-for-byte the old contract (the whole
    existing damage suite relies on it): one object in -> one object out."""
    out = subprocess.run([NODE, SCRIPT], input=json.dumps(_req_damage()),
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    d = json.loads(out.stdout)
    ref = calc.calc_damage(GARCH, TITAN, "earthquake")
    assert d["min"] == ref["min"] and d["max"] == ref["max"]
