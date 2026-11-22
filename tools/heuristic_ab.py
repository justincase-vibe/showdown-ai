"""Step 6.6 — the heuristic A/B: NEW (rules 1–4) vs OLD vs random.

The 6.1 design note: "A/B old vs new heuristic vs random, ~10–20 battles
each, recorded to ``runs/learning_check.csv`` under label ``step6-heuristic``.
The NEW heuristic becomes the standing Step 8 yardstick; the old after-step3
numbers stay as history."

The three bots (all deterministic, CPU-first, the ONE decision pipeline —
``league.driver.policy_action``):
  * ``strategy``  — the NEW heuristic: ``fitness.strategy_agent.StrategyAgent``
                    (the ctx brain, ``choose_action`` rules 1–4);
  * ``heuristic`` — the OLD heuristic: ``fitness.heuristic.HeuristicAgent``
                    (power * effectiveness * STAB; hazard-aware switch only);
  * ``random``    — the floor (random-init net, the ladder contract).

Matrix (n battles each): strategy vs heuristic (the headline, REPORT),
strategy vs random (PASS gate), heuristic vs random (sanity: the floor held).
Rows append to the SAME CSV as ``tools.learning_check`` (label
``step6-heuristic``, mode ``deterministic``) so the history stays one file.

CPU-first: no GPU is ever touched (the user's GPU is the LLM's).

Usage (from the repo root):
    python -m tools.heuristic_ab --n 20 --label step6-heuristic
Smoke (6 battles):
    python -m tools.heuristic_ab --n 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.learning_check import CSV_HEADER, DEFAULT_CSV, ci95  # noqa: E402


def _gen_from_format(fmt: str) -> int:
    import re
    m = re.match(r"gen(\d+)", fmt or "")
    return int(m.group(1)) if m else 9


def matrix() -> List[Tuple[str, str]]:
    """The A/B pairs, (bot, ref) — each played ``n`` battles. Pure (testable):
    ``strategy`` = the NEW heuristic (rules 1–4), ``heuristic`` = the OLD,
    ``random`` = the floor."""
    return [
        ("strategy", "heuristic"),   # the headline comparison (REPORT)
        ("strategy", "random"),      # PASS gate: the new one beats the floor
        ("heuristic", "random"),     # sanity: the OLD still beats the floor
    ]


def run_ab(
    n_battles: int,
    fmt: str = "gen9ou",
    label: str = "step6-heuristic",
    seed: int = 20261122,
    n_servers: int = 2,
    per_server: int = 2,
    port_base: int = 49970,
    max_steps: int = 200,
    csv_path: str = DEFAULT_CSV,
) -> Dict[str, Any]:
    """Play the A/B matrix and record (printed PASS/FAIL + CSV append)."""
    from net.vocab_db import build_vocabs
    from pool import PoolConfig, ResourcePool
    from fitness.ladder import random_reference
    from fitness.heuristic import heuristic_reference
    from fitness.strategy_agent import strategy_reference
    from fitness.harness import evaluate

    t0 = time.time()
    print(f"[heuristic_ab] fmt={fmt} n={n_battles}/pair label={label!r} "
          f"seed={seed}")

    vocabs = build_vocabs()
    refs = {
        "strategy": strategy_reference(fmt=fmt),
        "heuristic": heuristic_reference(gen=_gen_from_format(fmt)),
        "random": random_reference(vocabs, device="cpu"),
    }

    pool = ResourcePool(
        PoolConfig(n_servers=n_servers, port_base=port_base,
                   per_server=per_server, boot_timeout=120),
        log=lambda m: print(m, flush=True))
    print("[heuristic_ab] starting server pool ...", flush=True)
    status = pool.start(wait_healthy=True)
    if not all(status.values()):
        print(f"[heuristic_ab] WARNING: pool not fully healthy: {status}")
    try:
        records: List[Dict[str, Any]] = []
        for bot_name, ref_name in matrix():
            res = evaluate(
                bot=refs[bot_name], refs=[refs[ref_name]],
                n_battles=n_battles, fmt=fmt, vocabs=vocabs, device="cpu",
                seed=seed, max_steps=max_steps, greedy=True, pool=pool,
                log=lambda m: print(m, flush=True))
            pr = res.per_ref[ref_name]
            lo, hi = ci95(pr.wins, pr.n)
            if ref_name == "random":
                verdict = "PASS" if pr.win_rate > 0.50 else "FAIL"
            else:
                verdict = "REPORT"
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "label": label, "ckpt": bot_name, "format": fmt,
                "n": n_battles, "mode": "deterministic", "ref": ref_name,
                "wins": pr.wins, "losses": pr.losses, "draws": pr.draws,
                "win_rate": round(pr.win_rate, 4),
                "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                "avg_steps": round(pr.steps / pr.n, 1) if pr.n else 0,
                "fitness_score": round(res.fitness_score or 0.0, 4),
                "elapsed_s": round(time.time() - t0, 1),
            }
            records.append(rec)
            print(f"[heuristic_ab] {verdict:6s} {bot_name:9s} vs {ref_name:9s} "
                  f"W/L/D={pr.wins}/{pr.losses}/{pr.draws} "
                  f"wr={pr.win_rate:.3f} 95%CI=[{lo:.3f},{hi:.3f}] "
                  f"avg_steps={rec['avg_steps']}", flush=True)
        return {"records": records, "csv": csv_path}
    finally:
        pool.shutdown()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20,
                    help="battles per pair (the 6.1 note: ~10-20)")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--label", default="step6-heuristic")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--n-servers", type=int, default=2)
    ap.add_argument("--per-server", type=int, default=2)
    ap.add_argument("--port-base", type=int, default=49970)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args(argv)

    out = run_ab(
        n_battles=args.n, fmt=args.format, label=args.label, seed=args.seed,
        n_servers=args.n_servers, per_server=args.per_server,
        port_base=args.port_base, max_steps=args.max_steps, csv_path=args.csv)
    from tools.learning_check import append_csv
    append_csv(out["records"], args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
