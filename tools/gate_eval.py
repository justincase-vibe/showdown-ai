"""C4 final gate evaluation — checkpoints/controls vs the yardstick, with CIs.

RULE (Claude review of master@1696bcb — supersedes the original handout):

  * The decision yardstick is plain ``strategy`` (NOT strategy_posterior).
  * The 95% lower CI bound is reported PER PLAY MODE (greedy and sampled
    separately — never pooled for the verdict), and the verdict uses the
    WORST (minimum) of the per-mode bounds:

        verdict bound >= 0.40 -> PASS
                    >= 0.25 -> CONTINUE
                    <  0.25 -> STOP

  * CONTROLS (run before trusting ANY verdict): a fresh random-init net
    and the old heuristic fight ``strategy`` with the SAME seed space (and
    thus the SAME teams) as the checkpoints. If the random-init net can
    pass the gate, the gate discriminates nothing and must be re-barred
    before interpreting the checkpoint numbers.
  * WINNER'S CURSE: run BOTH the best-eval checkpoint (r70) and the final
    one (r80) through the gate; never quote a single noisy eval peak.
  * IN-DISTRIBUTION ONLY: training and eval draw the same 25-team pool
    (StrongTeamBuilder per battle seed). A PASS says nothing about
    unseen teams yet — out-of-distribution eval is a later step.

This tool plays each SUBJECT (checkpoint net or control reference) as p1
against named yardsticks in both play modes, using the SAME parallel
battle loop as the M1 trainer (``gui.m1_gate_main.run_eval_battles`` — one
implementation, no duplication). Draws are NEVER counted as wins (eval
contract). Exit 0 = completed (regardless of verdict), 1 = infra failure.

Usage (GPU, the user's run — controls first, then both checkpoints, all
sharing the same battle seeds so every subject faces identical teams):

    python -m tools.gate_eval \
        --subject random_init,heuristic \
        --ckpt runs/20261123T195442Z/checkpoints/m1gate_round70.pt,runs/20261123T195442Z/checkpoints/m1gate_round80.pt \
        --opps strategy --n-per-pair 200 --modes greedy,sampled \
        --device cuda --seed 20261124 --servers 6 --per-server 2 --port-base 49980

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

#: M1 gate decision-rule thresholds (lower 95% CI bound vs ``strategy``).
GATE_PASS = 0.40
GATE_CONTINUE = 0.25

#: the gate's primary yardstick (decision rule is defined against THIS one;
#: Claude review 2026-11-24: plain strategy, per-mode bounds — the earlier
#: strategy_posterior-pooled rule is superseded).
PRIMARY_OPP = "strategy"

#: control subjects -> M1-pool reference names (same sources as the trainer)
SUBJECT_TO_POOL = {
    "random_init": "random",     # fresh random-init net (the floor)
    "heuristic": "heuristic",    # the OLD heuristic bot
}


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


def min_mode_bound(bounds: Dict[str, float]) -> float:
    """The verdict bound: the WORST (minimum) per-mode lower bound.

    A net that passes greedy but fails sampled (or vice versa) is NOT a
    pass — the policy must be robust in both play modes. Pure function.
    Empty / all-nan -> nan.
    """
    real = [b for b in bounds.values()
            if b is not None and not (isinstance(b, float)
                                     and math.isnan(b))]
    if not real:
        return float("nan")
    return min(real)


def pool_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool per-(mode) eval rows for ONE opponent into a single W/L/D row.

    Reporting-only (the verdict is per-mode, not pooled). Draws are summed
    (never wins). Pure function.
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


def _subject_verdict(pairs: Dict[str, Any]) -> Dict[str, Any]:
    """Per-mode bounds vs the primary yardstick + verdict = worst bound."""
    verdict = {
        "rule": ("per-mode lower CI95 vs strategy; verdict = min of modes; "
                 f">={GATE_PASS} PASS, >={GATE_CONTINUE} CONTINUE, else STOP"),
        "primary": PRIMARY_OPP,
        "per_mode_lower_ci": {},
        "verdict_bound": float("nan"),
        "verdict": "NO_DATA",
    }
    if PRIMARY_OPP in pairs:
        for mode, row in pairs[PRIMARY_OPP].items():
            if isinstance(row, dict) and "eval_w" in row:
                n = row["eval_w"] + row["eval_l"] + row["eval_d"]
                verdict["per_mode_lower_ci"][mode] = wilson_lower(
                    row["eval_w"], n)
        verdict["verdict_bound"] = min_mode_bound(
            verdict["per_mode_lower_ci"])
        verdict["verdict"] = gate_verdict(verdict["verdict_bound"])
    return verdict


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="tools.gate_eval")
    ap.add_argument("--ckpt", default=None,
                    help="comma-separated checkpoint files (.pt) to evaluate")
    ap.add_argument("--subject", default=None,
                    help="comma-separated CONTROLS: "
                         + ",".join(SUBJECT_TO_POOL)
                         + " (random-init floor / old heuristic)")
    ap.add_argument("--opps", default=PRIMARY_OPP,
                    help="comma-separated yardsticks (default: strategy)")
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
    ckpts = _parse_list(args.ckpt, "--ckpt") if args.ckpt else []
    subjects = _parse_list(args.subject, "--subject") if args.subject else []
    if not ckpts and not subjects:
        print("FATAL: give --ckpt and/or --subject")
        return 1
    bad_subjects = [s for s in subjects if s not in SUBJECT_TO_POOL]
    if bad_subjects:
        print(f"FATAL: unknown subject(s) {bad_subjects} (allowed: "
              f"{list(SUBJECT_TO_POOL)})")
        return 1

    root = Path(__file__).resolve().parents[1]
    t0 = time.time()
    out_dir = Path(args.out) if args.out else (
        root / "runs" /
        f"gate_eval_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # ── resolve checkpoints (project-root relative, same as the launcher) ─
    ckpt_paths: List[Path] = []
    for c in ckpts:
        p = Path(c).expanduser()
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            print(f"FATAL: checkpoint not found: {p}")
            return 1
        ckpt_paths.append(p)

    # ── GPU guard (loud, before anything heavy) ─────────────────────────
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print("FATAL: device='cuda' requested but "
                  "torch.cuda.is_available() is False")
            return 1
        log(f"CUDA available: {torch.cuda.get_device_name(0)}")

    # ── build pool (same sources as the M1 trainer) ─────────────────────
    from net.vocab_db import build_vocabs
    from fitness.m1_pool import build_m1_pool
    from gui.m1_gate_main import run_eval_battles

    vocabs = build_vocabs()
    full_pool = build_m1_pool(fmt=args.format, vocabs=vocabs,
                              device=args.device)
    missing = [o for o in opps if o not in full_pool]
    # check the MAPPED pool name (the subject alias itself is not a pool key)
    missing += [SUBJECT_TO_POOL[s] for s in subjects
                if SUBJECT_TO_POOL[s] not in full_pool]
    if missing:
        print(f"FATAL: name(s) {missing} not in the M1 pool "
              f"(have {full_pool.names()})")
        return 1

    results: Dict[str, Any] = {
        "gate": "m1_c4_final_controls",
        "rule": ("decision yardstick = strategy; per-mode lower CI95; "
                 "verdict = min of modes; PASS >=0.40 / CONTINUE >=0.25 / "
                 "STOP below"),
        "in_distribution_only": True,
        "in_distribution_note": ("train and eval draw the SAME 25-team pool "
                                 "(StrongTeamBuilder per battle seed); a "
                                 "PASS says nothing about unseen teams yet"),
        "format": args.format,
        "device": args.device,
        "seed": args.seed,
        "n_per_pair": args.n_per_pair,
        "modes": modes,
        "opps": opps,
        "subjects": {},
    }

    # Controls FIRST (they calibrate how much the gate discriminates).
    subject_entries = [("control:" + s, None, SUBJECT_TO_POOL[s])
                       for s in subjects]
    subject_entries += [(str(p), p, None) for p in ckpt_paths]

    exit_ok = True
    for label, ckpt_path, pool_name in subject_entries:
        log(f"=== subject {label} ===")
        if ckpt_path is not None:
            log(f"loading checkpoint {ckpt_path}")
            from net.model import PokeNet
            net = PokeNet.load_file(str(ckpt_path), vocabs,
                                    device=args.device)
            my_ref = None
        else:
            net = None
            my_ref = full_pool.get(pool_name)
        results["subjects"][label] = {
            "kind": "control" if ckpt_path is None else "checkpoint",
            "source": (pool_name if ckpt_path is None
                       else str(ckpt_path)),
            "pairs": {},
        }
        for opp in opps:
            opp_ref = full_pool.get(opp)
            results["subjects"][label]["pairs"][opp] = {}
            for pair_idx, mode in enumerate(modes):
                # SHARED seed space across all subjects with the same
                # (opp, mode): identical battle seeds -> identical teams on
                # both sides -> controls and checkpoints face the SAME
                # matchups (the Claude-review "same seeds" requirement).
                seed_base = 2_000_000 + args.seed * 10 + pair_idx * 100_000
                log(f"--- {opp} x {mode} ({args.n_per_pair} battles) ---")
                try:
                    row = run_eval_battles(
                        net, opp_ref, args.format, vocabs, args.device,
                        args.n_per_pair, args.max_steps,
                        args.servers, args.per_server, args.port_base,
                        seed_base=seed_base,
                        deterministic=(mode == "greedy"),
                        label=f"gate-{label}-{opp}-{mode}",
                        my_ref=my_ref)
                except Exception as exc:  # noqa: BLE001 — record + keep going
                    print(f"FATAL in {label} / {opp} x {mode}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    results["subjects"][label]["pairs"][opp][mode] = {
                        "error": str(exc)}
                    exit_ok = False
                    continue
                results["subjects"][label]["pairs"][opp][mode] = row
                log(f"    {opp} x {mode}: {row['eval_w']}W/"
                    f"{row['eval_l']}L/{row['eval_d']}D "
                    f"crashed={row['eval_crashed']} wr={row['eval_wr']:.3f}")
        results["subjects"][label]["verdict"] = _subject_verdict(
            results["subjects"][label]["pairs"])

    results["wall_s"] = round(time.time() - t0, 1)
    results["completed_ok"] = exit_ok

    (out_dir / "result.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # ── report ──────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"{'subject':<24} {'opp':<12} {'mode':<9} {'W/L/D':<12} "
          f"{'wr':>6} {'ci95 lo':>9}")
    print("-" * 78)
    for label, sub in results["subjects"].items():
        for opp, by_mode in sub["pairs"].items():
            for mode, row in by_mode.items():
                if not isinstance(row, dict) or "eval_w" not in row:
                    err = row.get("error") if isinstance(row, dict) else row
                    print(f"{label:<24} {opp:<12} {mode:<9} ERROR: {err}")
                    continue
                n = row["eval_w"] + row["eval_l"] + row["eval_d"]
                lo = wilson_lower(row["eval_w"], n)
                print(f"{label:<24} {opp:<12} {mode:<9} "
                      f"{row['eval_w']}/{row['eval_l']}/{row['eval_d']:<6} "
                      f"{row['eval_wr']:>6.3f} {lo:>9.3f}")
        v = sub["verdict"]
        if v["per_mode_lower_ci"]:
            bounds = " ".join(f"{m}={b:.3f}"
                              for m, b in v["per_mode_lower_ci"].items())
            print(f"  -> {label}: {bounds}  bound={v['verdict_bound']:.3f} "
                  f"VERDICT={v['verdict']}")
        print("-" * 78)
    print("NOTE: IN-DISTRIBUTION ONLY (same 25-team pool as training).")
    print("NOTE: if a CONTROL passes the gate, the gate bar is broken —")
    print("      re-bar it before trusting any checkpoint verdict.")
    print(f"wall={results['wall_s']}s  result={out_dir / 'result.json'}")
    print("=" * 78)
    return 0 if exit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
