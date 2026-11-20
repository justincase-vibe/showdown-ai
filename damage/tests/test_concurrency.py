"""C4 — the concurrency that CRASHED the league (8 battle threads x rapid
per-request node spawns -> 0xC0000409 at battle 318) now runs through the
bounded worker pool with NO spawn churn. This is the "workable program" proof
at the calc layer. Pinned:
  * 8 threads, each doing many batch calcs (the forward-eval shape: several
    damage calcs per decision), ALL return valid results — no crash;
  * the pool stays BOUNDED (worker count never grows past its size — no
    per-request spawn / leaked node processes), and ``shutdown_pool()`` reaps
    them all.

Run from the project root:

    python -m pytest damage/tests/test_concurrency.py -v
"""
from __future__ import annotations

import threading

from damage import calc
from damage import worker

GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}}
TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}
CHARZ = {"name": "Charizard", "nature": "Modest", "evs": {"spa": 252}}
BLAST = {"name": "Blastoise", "nature": "Modest", "evs": {"spa": 252}}

MOVES = ["earthquake", "flamethrower", "hydropump", "crunch", "closecombat"]


def test_eight_thread_battle_sim():
    """8 concurrent 'battles', each a handful of forward-eval batches (my moves
    x enemy mons). Every calc must come back valid — the old per-request spawn
    churn is gone, so there is nothing to hard-crash."""
    n_decisions = 6
    errs: list = []

    def battle(t: int) -> None:
        try:
            for _ in range(n_decisions):
                items = (
                    [{"attacker": GARCH, "defender": TITAN, "move": m} for m in MOVES[:4]]
                    + [{"attacker": CHARZ, "defender": BLAST, "move": m} for m in MOVES[:2]]
                )
                res = calc.calc_damage_many(items)
                assert len(res) == len(items)
                for r in res:
                    assert "min" in r and "max" in r, r
        except Exception as e:  # noqa: BLE001 - captured for the assertion
            errs.append(f"battle {t}: {e!r}")

    ts = [threading.Thread(target=battle, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs


def test_pool_stays_bounded_no_leak():
    """After heavy use the pool must NOT have grown past its size (no per-request
    spawn / process leak), and shutdown_pool() terminates every node process."""
    import time

    pool = worker.get_pool()
    size = pool._size
    for _ in range(200):
        calc.calc_damage(GARCH, TITAN, "earthquake")
    # bounded: exactly `size` workers alive, never one-per-calc
    assert len(worker.get_pool()._workers) == size, (
        f"pool grew to {len(worker.get_pool()._workers)} (expected {size}) — leak?")

    ws = list(worker.get_pool()._workers)
    worker.shutdown_pool()
    for _ in range(100):  # up to ~5s for the node procs to exit
        if all(w.p.poll() is not None for w in ws):
            break
        time.sleep(0.05)
    assert all(w.p.poll() is not None for w in ws), "node procs not reaped on shutdown"
