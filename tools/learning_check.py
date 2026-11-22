"""Step 8 (Claude review 2026-11-22) — the "IS THE NET LEARNING?" check.

v1 scope = protocol section 2 (fixed-opponent evaluation), which is exactly
what the user asked for the BEFORE-number: win-rate of ONE checkpoint against
the fixed ladder (random floor + heuristic), reported for BOTH sampled play
and argmax play. Every run appends one row per (mode, reference) to a CSV so
before/after numbers are comparable across the Step 2/3/4 changes:

    runs/learning_check.csv

Later (once training runs exist) sections 1 (training health), 3 (scenario
suite), 4 (value calibration) and 5 (feature ablation) get added here — same
PASS/FAIL-per-line + CSV-append contract.

CPU-first: no GPU is ever touched (device=cpu hardcoded; the user's GPU is
the LLM's). Sequential battles on a small server pool.

Usage (from the repo root):
    python -m tools.learning_check --ckpt scratch/league_random_1001.pt \
        --n 50 --label before-step2

Smoke test of the whole pipeline (2 battles vs random, sampled):
    python -m tools.learning_check --ckpt <ckpt> --n 2 --label smoke
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_CSV = os.path.join(REPO_ROOT, "runs", "learning_check.csv")

CSV_HEADER = [
    "ts", "label", "ckpt", "format", "n", "mode", "ref",
    "wins", "losses", "draws", "win_rate", "ci95_lo", "ci95_hi",
    "avg_steps", "fitness_score", "elapsed_s",
]


def ci95(wins: int, n: int) -> "tuple[float, float]":
    """Normal-approx 95% CI for a binomial proportion (n is small-ish;
    fine for a progress yardstick, not for publication)."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    half = 1.96 * math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - half), min(1.0, p + half))


def _gen_from_format(fmt: str) -> int:
    """The @smogon/calc engine key for a format id — the ONE source of truth
    is ``legality.formats.engine_key`` (D32: champions = 0, NOT 9; mapping to
    9 would silently mis-calc ~25 moves). Regex fallback for ids outside the
    registry; unrecognised -> 9 (the primary format). B1: the fallback is
    LOUD (a warning) via ``engine_key_or_warn`` — no silent mis-keying."""
    from legality.formats import engine_key_or_warn
    return int(engine_key_or_warn(fmt, default=9))


def run_check(
    ckpt: str,
    n_battles: int,
    fmt: str = "gen9ou",
    label: str = "",
    seed: int = 20261122,
    n_servers: int = 2,
    per_server: int = 2,
    port_base: int = 49960,
    max_steps: int = 200,
    csv_path: str = DEFAULT_CSV,
) -> Dict[str, Any]:
    """Play the checkpoint vs the fixed ladder (sampled + argmax) and record.

    Returns a dict of the per-(mode, ref) records (also printed as PASS/FAIL
    lines and appended to the CSV).
    """
    from net.vocab_db import build_vocabs
    from pool import PoolConfig, ResourcePool
    from fitness.ladder import random_reference
    from fitness.heuristic import heuristic_reference
    from fitness.harness import evaluate

    t0 = time.time()
    print(f"[learning_check] ckpt={ckpt} fmt={fmt} n={n_battles}/ref "
          f"label={label!r} seed={seed}")

    vocabs = build_vocabs()
    refs = [
        random_reference(vocabs, device="cpu"),
        heuristic_reference(gen=_gen_from_format(fmt)),
    ]

    pool = ResourcePool(
        PoolConfig(n_servers=n_servers, port_base=port_base,
                   per_server=per_server, boot_timeout=120),
        log=lambda m: print(m, flush=True))
    print("[learning_check] starting server pool ...", flush=True)
    status = pool.start(wait_healthy=True)
    if not all(status.values()):
        print(f"[learning_check] WARNING: pool not fully healthy: {status}")
    try:
        records: List[Dict[str, Any]] = []
        for mode in ("sampled", "argmax"):
            greedy = mode == "argmax"
            res = evaluate(
                bot=ckpt, refs=refs, n_battles=n_battles, fmt=fmt,
                vocabs=vocabs, device="cpu", seed=seed,
                max_steps=max_steps, greedy=greedy, pool=pool,
                log=lambda m: print(m, flush=True))
            for ref in refs:
                pr = res.per_ref[ref.name]
                lo, hi = ci95(pr.wins, pr.n)
                # PASS/FAIL yardsticks (single-run sanity, per the ladder's own
                # contract: "a competent bot should beat random well above 50%"):
                #   vs random   -> PASS if win_rate > 0.50
                #   vs heuristic-> REPORT (trending judgment needs >=3 runs;
                #                    the CSV history is what PASS/FAIL uses)
                if ref.name == "random":
                    verdict = "PASS" if pr.win_rate > 0.50 else "FAIL"
                else:
                    verdict = "REPORT"
                rec = {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "label": label, "ckpt": ckpt, "format": fmt,
                    "n": n_battles, "mode": mode, "ref": ref.name,
                    "wins": pr.wins, "losses": pr.losses, "draws": pr.draws,
                    "win_rate": round(pr.win_rate, 4),
                    "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                    "avg_steps": round(pr.steps / pr.n, 1) if pr.n else 0,
                    "fitness_score": round(res.fitness_score or 0.0, 4),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                records.append(rec)
                print(f"[learning_check] {verdict:6s} {mode:7s} vs {ref.name:9s} "
                      f"W/L/D={pr.wins}/{pr.losses}/{pr.draws} "
                      f"wr={pr.win_rate:.3f} 95%CI=[{lo:.3f},{hi:.3f}] "
                      f"avg_steps={rec['avg_steps']}", flush=True)
        return {"records": records, "csv": csv_path}
    finally:
        pool.shutdown()


def append_csv(records: List[Dict[str, Any]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new:
            w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER})
    print(f"[learning_check] appended {len(records)} row(s) -> {csv_path}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt path")
    ap.add_argument("--n", type=int, default=50, help="battles per reference")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--label", default="", help="free text, recorded in the CSV")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--n-servers", type=int, default=2)
    ap.add_argument("--per-server", type=int, default=2)
    ap.add_argument("--port-base", type=int, default=49960)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args(argv)

    out = run_check(
        ckpt=args.ckpt, n_battles=args.n, fmt=args.format, label=args.label,
        seed=args.seed, n_servers=args.n_servers, per_server=args.per_server,
        port_base=args.port_base, max_steps=args.max_steps, csv_path=args.csv)
    append_csv(out["records"], args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
