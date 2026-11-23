"""D6 (2026-11-23 ruling) — the final draw-handling regression gate.

Two pairs, 50 battles each, asserts draws < 5% on BOTH:
  * strategy vs strategy   (the hysteresis self-play: the original D1 bug)
  * strategy vs heuristic  (cross-bot: the new rules vs the old)

CPU-only. Runs a small pool (2 servers x 2), sequential pairs.
Exit 0 = both pairs pass; exit 1 = at least one pair fails.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=-1 python -m tools.d6_gate
    CUDA_VISIBLE_DEVICES=-1 python -m tools.d6_gate --n 10   # quick smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

REPO_ROOT = None  # set in main


def main(argv=None) -> int:
    global REPO_ROOT
    import os
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=50, help="battles per pair")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--n-servers", type=int, default=2)
    ap.add_argument("--per-server", type=int, default=2)
    ap.add_argument("--port-base", type=int, default=49980)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--draw-threshold", type=float, default=0.05,
                    help="max allowed draw fraction (default 0.05 = 5%%)")
    args = ap.parse_args(argv)

    from net.vocab_db import build_vocabs
    from pool import PoolConfig, ResourcePool
    from fitness.heuristic import heuristic_reference
    from fitness.strategy_agent import strategy_reference
    from fitness.harness import evaluate
    from tools.learning_check import ci95

    print(f"[D6] n={args.n}/pair fmt={args.format} seed={args.seed} "
          f"draw_threshold={args.draw_threshold:.0%}")
    print(f"[D6] pool: {args.n_servers} servers x {args.per_server} "
          f"(ports {args.port_base}-{args.port_base + args.n_servers - 1})")

    vocabs = build_vocabs()
    strategy = strategy_reference(fmt=args.format)
    heuristic = heuristic_reference(gen=9)

    pool = ResourcePool(
        PoolConfig(n_servers=args.n_servers, port_base=args.port_base,
                   per_server=args.per_server, boot_timeout=120),
        log=lambda m: print(m, flush=True))
    print("[D6] starting server pool ...", flush=True)
    status = pool.start(wait_healthy=True)
    if not all(status.values()):
        print(f"[D6] WARNING: pool not fully healthy: {status}")

    all_pass = True
    results = []
    try:
        for label, bot, ref in [
            ("strategy_vs_strategy", strategy, strategy_reference(fmt=args.format)),
            ("strategy_vs_heuristic", strategy, heuristic),
        ]:
            print(f"\n[D6] === {label} ({args.n} battles) ===", flush=True)
            t0 = time.time()
            res = evaluate(
                bot=bot, refs=[ref],
                n_battles=args.n, fmt=args.format, vocabs=vocabs, device="cpu",
                seed=args.seed, max_steps=args.max_steps, greedy=True, pool=pool,
                log=lambda m: print(m, flush=True))
            pr = res.per_ref[list(res.per_ref.keys())[0]]
            lo, hi = ci95(pr.wins, pr.n)
            draw_frac = pr.draws / pr.n if pr.n else 0.0
            passed = draw_frac < args.draw_threshold
            all_pass = all_pass and passed

            rec = {
                "label": label, "n": args.n,
                "wins": pr.wins, "losses": pr.losses, "draws": pr.draws,
                "draw_frac": round(draw_frac, 4),
                "win_rate": round(pr.win_rate, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "avg_steps": round(pr.steps / pr.n, 1) if pr.n else 0,
                "elapsed_s": round(time.time() - t0, 1),
                "passed": passed,
            }
            results.append(rec)
            verdict = "PASS" if passed else "FAIL"
            print(f"[D6] {verdict} {label}: W/L/D={pr.wins}/{pr.losses}/{pr.draws} "
                  f"draw_frac={draw_frac:.1%} (threshold {args.draw_threshold:.0%}) "
                  f"avg_steps={rec['avg_steps']} elapsed={rec['elapsed_s']}s",
                  flush=True)
    finally:
        pool.shutdown()

    print(f"\n[D6] {'=' * 50}")
    print(f"[D6] SUMMARY: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    for r in results:
        print(f"  {r['label']}: {r['wins']}W/{r['losses']}L/{r['draws']}D "
              f"draw={r['draw_frac']:.1%} {'PASS' if r['passed'] else 'FAIL'}")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[D6] ts={ts}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
