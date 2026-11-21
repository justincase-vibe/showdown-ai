"""A3 self-check (LIVE, CPU): does the HeuristicAgent actually beat the random
net floor in REAL battles? This is the thing the hermetic tests can't prove —
it exercises the full stack (harness + heuristic + driver + belief + calc +
pool) end to end.

Usage:
    CUDA_VISIBLE_DEVICES=-1 python scratch/_a3_selfcheck.py [n_battles] [fmt]
    # default: 3 battles, gen9ou   (raise n for a firmer >70% gate)

CPU-only. Starts one pool server, plays N battles of heuristic (p1) vs random
(p2), prints per-battle outcomes + the win-rate, then shuts the server down.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    fmt = sys.argv[2] if len(sys.argv) > 2 else "gen9ou"

    from pool import ResourcePool, PoolConfig
    from net.vocab_db import build_vocabs
    from fitness.ladder import random_reference
    from fitness.heuristic import heuristic_reference
    from fitness.harness import evaluate

    vocabs = build_vocabs()
    pool = ResourcePool(PoolConfig(n_servers=1, per_server=1, port_base=8907))
    try:
        pool.start(wait_healthy=True)
        print(f"[self-check] pool up; evaluating heuristic vs random "
              f"({n} battles, {fmt}) ...")
        result = evaluate(
            bot=heuristic_reference(gen=9),
            refs=[random_reference(vocabs, device="cpu")],
            n_battles=n, fmt=fmt, pool=pool, max_steps=80,
            vocabs=vocabs,   # REQUIRED for the real obs path (build_obs); see net.env
            log=lambda s: print("  ", s))
        pr = result.per_ref["random"]
        print(f"\n[self-check] heuristic vs random  ({fmt})")
        print(f"  wins={pr.wins}  losses={pr.losses}  draws={pr.draws}  "
              f"n={pr.n}  steps={pr.steps}")
        print(f"  win_rate = {pr.win_rate:.3f}   (gate: beats random, i.e. >> 0.5)")
        print(f"  fitness_score = {result.fitness_score}")
        if pr.n >= 3:
            print(f"  A3 gate (beats random > 0.5) : "
                  f"{'PASS' if pr.win_rate > 0.5 else 'REVIEW'}")
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
