"""C4 final gate evaluation — checkpoint vs the yardstick references, with CIs.

The M1 gate decision (Claude handout, approved):

    FINAL gate: at least 400 battles per opponent (strategy AND
    strategy_posterior), sampled AND argmax play, 95% CI reported.
    Decision rule by LOWER 95% CI bound of the win rate vs
    strategy_posterior (both modes pooled):
        >= 0.40 -> PASS (propose M2b)
        >= 0.25 -> CONTINUE (keep training; report which checks lag)
        <  0.25 -> STOP (run feature ablation + entropy/action-mix diagnostics)

This tool plays a CHECKPOINT (never a live learner — the net is read-only)
against named M1-pool references in both play modes, using the SAME parallel
battle loop as the M1 trainer (``gui.m1_gate_main.run_eval_battles`` — one
implementation, no duplication). Draws are NEVER counted as wins (eval
contract). Exit 0 = completed (regardless of verdict), 1 = failure.

Usage (GPU, the user's run):
    python -m tools.gate_eval --ckpt runs/<run>/checkpoints/m1gate_round80.pt \
        --opps strategy,strategy_posterior --n-per-pair 200 \
        --modes greedy,sampled --device cuda --seed 20261124 \
        --servers 6 --per-server 2 --port-base 49980

Results: printed table + ``runs/gate_eval_<UTC-ts>/result.json``.
A gate-eval dir has NO status.json, so the launcher's run discovery ignores it.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

#: M1 gate decision-rule thresholds (lower 95% CI bound vs strategy_posterior)
GATE_PASS = 0.40
GATE_CONTINUE = 0.25

#: the gate's primary yardstick (the decision rule is defined against this one)
PRIMARY_OPP = "strategy_posterior"


def wilson_lower(wins: int, n: int,
                 z: float = 1.959963984540054) -> float:
    """Lower bound of the Wilson 95% score interval for ``wins/n``.

    Wilson (not the naive normal approx) because it stays in [0,1] and is
    well-behaved at extreme proportions. Pure function -> unit-testable.
    ``n == 0`` -> nan (no data).
    """
    if n <= 0:
        return float("nan")
    p = wins / n
    z2_over_n = z * z / n
    centre = (p + z2_over_n / 2.0) / (1.0 + z2_over_n)
    half = (z * math.sqrt(p * (1.0 - p) / n + z2_over_n / 4.0)) \
        / (1.0 + z2_over_n)
    return max(0.0, centre - half)


def gate_verdict(lower_ci: float) -> str:
    """The M1 decision rule applied to a lower-CI bound. Pure function."""
    if lower_ci is None or (isinstance(lower_ci, float)
                            and math.isnan(lower_ci)):
        return "NO_DATA"
    if lower_ci >= GATE_PASS:
        return "PASS"
    if lower_ci >= GATE_CONTINUE:
        return "CONTINUE"
    return "STOP"


def pool_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool per-(mode) eval rows for ONE opponent into a single W/L/D row.

    The gate decision rule is defined on the pooled win rate (both play
    modes), so draws are summed (never wins). Pure function.
    """
    w = sum(int(r["eval_w"]) for r in rows)
    l = sum(int(r["eval_l"]) for r in rows)
    d = sum(int(r["eval_d"]) for r in rows)
    crashed = sum(int(r.get("eval_crashed", 0)) for r in rows)
    played = w + l + d
    return {
        "wins": w, "losses": l, "draws": d, "crashed": crashed,
        "n": played,
        "win_rate": (w / played) if played else float("nan"),
        "lower_ci_95": wilson_lower(w, played),
    }


def _parse_list(value: str, what: str) -> List[str]:
    out = [s.strip() for s in value.split(",") if s.strip()]
    if not out:
        raise ValueError(f"{what} must be a non-empty comma-separated list")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="tools.gate_eval")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint file (.pt) to evaluate")
    ap.add_argument("--opps", default="strategy,strategy_posterior",
                    help="comma-separated M1 pool names (default: "
                         "strategy,strategy_posterior)")
    ap.add_argument("--n-per-pair", type=int, default=200,
                    help="battles per (opponent x mode) pair (default 200)")
    ap.add_argument("--modes", default="greedy,sampled",
                    help="comma-separated: greedy and/or sampled (default both)")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--servers", type=int, default=6)
    ap.add_argument("--per-server", type=int, default=2)
    ap.add_argument("--port-base", type=int, default=49980)
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out", default=None,
                    help="result dir (default runs/gate_eval_<UTC-ts>)")
    args = ap.parse_args(argv)

    opps = _parse_list(args.opps, "--opps")
    modes = _parse_list(args.modes, "--modes")
    unknown = [m for m in modes if m not in ("greedy", "sampled")]
    if unknown:
        print(f"FATAL: unknown mode(s) {unknown} (allowed: greedy, sampled)")
        return 1
    if args.n_per_pair <= 0:
        print("FATAL: --n-per-pair must be > 0")
        return 1

    t0 = time.time()
    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1]
        / "runs" / f"gate_eval_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_absolute():
        # resolve against the PROJECT ROOT, not the CWD (the launcher's
        # same-root-resolution contract — the user may run from anywhere)
        ckpt = Path(__file__).resolve().parents[1] / ckpt
    if not ckpt.is_file():
        print(f"FATAL: checkpoint not found: {ckpt}")
        return 1

    # ── GPU guard (loud, before anything heavy) ─────────────────────────
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print("FATAL: device='cuda' requested but "
                  "torch.cuda.is_available() is False")
            return 1
        log(f"CUDA available: {torch.cuda.get_device_name(0)}")

    # ── build net + pool (same sources as the M1 trainer) ───────────────
    log(f"loading checkpoint {ckpt}")
    from net.vocab_db import build_vocabs
    from net.model import PokeNet
    from fitness.m1_pool import build_m1_pool
    from gui.m1_gate_main import run_eval_battles

    vocabs = build_vocabs()
    net = PokeNet.load_file(str(ckpt), vocabs, device=args.device)
    full_pool = build_m1_pool(fmt=args.format, vocabs=vocabs, device=args.device)
    missing = [o for o in opps if o not in full_pool]
    if missing:
        print(f"FATAL: opponent(s) {missing} not in the M1 pool "
              f"(have {full_pool.names()})")
        return 1
    log(f"pool: {full_pool.names()}  format={args.format}  "
        f"pairs={[(o, m) for o in opps for m in modes]}")

    # ── run the pairs (sequential across pairs; parallel WITHIN a pair) ──
    results: Dict[str, Any] = {
        "gate": "m1_c4_final",
        "ckpt": str(ckpt),
        "format": args.format,
        "device": args.device,
        "seed": args.seed,
        "n_per_pair": args.n_per_pair,
        "modes": modes,
        "opps": opps,
        "pairs": {},
    }
    exit_ok = True
    pair_idx = 0
    for opp in opps:
        opp_ref = full_pool.get(opp)
        results["pairs"][opp] = {}
        for mode in modes:
            # disjoint seed space per pair (never re-samples a sibling pair)
            seed_base = 2_000_000 + args.seed * 10 + pair_idx * 100_000
            pair_idx += 1
            log(f"--- pair {opp} x {mode} ({args.n_per_pair} battles) ---")
            try:
                row = run_eval_battles(
                    net, opp_ref, args.format, vocabs, args.device,
                    args.n_per_pair, args.max_steps,
                    args.servers, args.per_server, args.port_base,
                    seed_base=seed_base,
                    deterministic=(mode == "greedy"),
                    label=f"gate-{opp}-{mode}")
            except Exception as exc:  # noqa: BLE001 — record + keep going
                print(f"FATAL in pair {opp} x {mode}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                results["pairs"][opp][mode] = {"error": str(exc)}
                exit_ok = False
                continue
            results["pairs"][opp][mode] = row
            log(f"    {opp} x {mode}: {row['eval_w']}W/{row['eval_l']}L/"
                f"{row['eval_d']}D crashed={row['eval_crashed']} "
                f"wr={row['eval_wr']:.3f}")

    # ── verdict (rule defined against the PRIMARY opponent, pooled) ─────
    verdict = {"rule": "lower_ci_95 vs strategy_posterior (both modes "
                       "pooled): >=0.40 PASS, >=0.25 CONTINUE, else STOP",
               "verdict": "NO_DATA", "primary_pooled": None}
    if PRIMARY_OPP in results["pairs"]:
        primary_rows = [r for r in results["pairs"][PRIMARY_OPP].values()
                        if isinstance(r, dict) and "eval_w" in r]
        if primary_rows:
            pooled = pool_rows(primary_rows)
            verdict["primary_pooled"] = pooled
            verdict["verdict"] = gate_verdict(pooled["lower_ci_95"])
    results["verdict"] = verdict
    results["wall_s"] = round(time.time() - t0, 1)
    results["completed_ok"] = exit_ok

    (out_dir / "result.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # ── report ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"{'opponent':<22} {'mode':<9} {'W/L/D':<12} {'wr':>6} {'ci95 lo':>9}")
    print("-" * 72)
    for opp, by_mode in results["pairs"].items():
        for mode, row in by_mode.items():
            if not isinstance(row, dict) or "eval_w" not in row:
                print(f"{opp:<22} {mode:<9} ERROR: {row.get('error') if isinstance(row, dict) else row}")
                continue
            n = row["eval_w"] + row["eval_l"] + row["eval_d"]
            lo = wilson_lower(row["eval_w"], n)
            print(f"{opp:<22} {mode:<9} "
                  f"{row['eval_w']}/{row['eval_l']}/{row['eval_d']:<6} "
                  f"{row['eval_wr']:>6.3f} {lo:>9.3f}")
        if PRIMARY_OPP == opp and verdict["primary_pooled"]:
            p = verdict["primary_pooled"]
            print(f"{opp + ' (pooled)':<22} {'':<9} "
                  f"{p['wins']}/{p['losses']}/{p['draws']}".ljust(12) + " "
                  f"{p['win_rate']:>6.3f} {p['lower_ci_95']:>9.3f}")
    print("-" * 72)
    print(f"VERDICT: {verdict['verdict']}   ({verdict['rule']})")
    print(f"wall={results['wall_s']}s  result={out_dir / 'result.json'}")
    print("=" * 72)
    return 0 if exit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
