"""C2 — the thread-safe Node worker pool (``damage/worker.py``).

Isolated from ``calc.py``'s routing (this pins the POOL itself, using the
one-shot CLI as the reference). Pinned:
  * a damage / speed / batch request through the pool == the one-shot result;
  * a calc ERROR (illegal species) raises ``RuntimeError`` (NOT ``WorkerError``)
    — the exact one-shot contract;
  * if a worker DIES mid-request it is respawned + the request retried;
  * N threads hammering the pool all get correct results (thread-safe);
  * ``shutdown_pool()`` terminates every node process; lazy re-creation works.

Run from the project root:

    python -m pytest damage/tests/test_worker_pool.py -v
"""
from __future__ import annotations

import threading
import time

from damage import calc
from damage import worker

GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}}
TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}
PIKA = {"name": "Pikachu", "level": 50}


def _dmg_req():
    return {"type": "damage", "gen": 9, "attacker": GARCH, "defender": TITAN,
            "move": {"name": "earthquake"}}


def _spd_req():
    return {"type": "speed", "gen": 9, "pokemon": PIKA}


def test_pool_matches_oneshot():
    d = worker.request(_dmg_req())
    ref = calc.calc_damage(GARCH, TITAN, "earthquake")  # one-shot reference
    assert d["min"] == ref["min"] and d["max"] == ref["max"], (d, ref)

    s = worker.request(_spd_req())
    ref_s = calc.effective_speed(PIKA)
    assert s["speed"] == ref_s, (s, ref_s)

    b = worker.request([_dmg_req(), _spd_req()])
    assert isinstance(b, list) and len(b) == 2, b
    assert b[0]["min"] == ref["min"] and b[0]["max"] == ref["max"]
    assert b[1]["speed"] == ref_s


def test_pool_calc_error_is_runtimeerror():
    """An illegal species is a CALC error, not a worker death -> RuntimeError
    (the one-shot contract calc.py relies on), never WorkerError."""
    bad = {"type": "damage", "gen": 9, "attacker": {"name": "zzz_not_real"},
           "defender": TITAN, "move": {"name": "tackle"}}
    try:
        worker.request(bad)
    except worker.WorkerError:
        raise AssertionError("illegal species must be RuntimeError, not WorkerError")
    except RuntimeError as e:
        assert "zzz_not_real" in str(e) or "not legal" in str(e)
    else:
        raise AssertionError("expected RuntimeError for an illegal species")


def test_pool_respawns_on_death():
    pool = worker.get_pool()
    victim = pool._workers[0]
    victim.p.kill()
    victim.p.wait()
    # the next request must detect the death, respawn, and retry successfully
    d = worker.request(_dmg_req())
    ref = calc.calc_damage(GARCH, TITAN, "earthquake")
    assert d["min"] == ref["min"] and d["max"] == ref["max"]


def test_pool_thread_safety():
    errs = []

    def hammer(i: int) -> None:
        try:
            for _ in range(10):
                d = worker.request(_dmg_req())
                assert "min" in d and "max" in d
                s = worker.request(_spd_req())
                assert "speed" in s
        except Exception as e:  # noqa: BLE001 - captured for the assertion
            errs.append(f"thread {i}: {e!r}")

    ts = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs


def test_shutdown_kills_workers_then_recreates():
    pool = worker.get_pool()
    ws = list(pool._workers)
    worker.shutdown_pool()
    for _ in range(100):  # up to ~5s for the node procs to exit
        if all(w.p.poll() is not None for w in ws):
            break
        time.sleep(0.05)
    assert all(w.p.poll() is not None for w in ws), "workers not terminated on shutdown"

    # lazy re-creation still works after a shutdown
    d = worker.request(_dmg_req())
    assert "min" in d and "max" in d
