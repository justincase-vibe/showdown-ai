"""C4 final gate evaluation — controls C1-C5 + checkpoints, with CIs.

CANONICAL SPEC: ``GATE_CONTROLS_C1_C5.md`` (Claude review + user decisions).
This tool implements it:

CONTROLS (run with the SAME seeds/teams, >= 200 battles each):
  C1  random-init net vs strategy AND strategy_posterior (both modes)
  C2  the `random` pool bot vs strategy AND strategy_posterior
      (in THIS codebase C2 is mechanically identical to C1 — the pool's
      `random` member IS a fresh `make_random_net`; the validity check reads
      the C2 row if present, else the C1 row)
  C3  the old `heuristic` bot vs strategy AND strategy_posterior
  C4  mirror: strategy vs strategy, strategy_posterior vs strategy
      (expect ~50%; p1/p2 side-bias check)
  C5  strategy-bot silent-fallback counters (ctx-None acts[0], choose_action
      exception acts[0], POV except, passes) — per-pair deltas, printed.

VERDICT (per the spec):
  * Wilson lower 95% bound per (opponent x mode) AND pooled per opponent.
  * verdict bound = the WORST (min) of the POOLED bounds of `strategy` and
    `strategy_posterior` (both, if both ran).
  * PASS >= 0.40, CONTINUE >= 0.25, STOP below — subject to the VALIDITY
    rules (printed with the verdict):
      YARDSTICK_INVALID if C2 beats the strategy yardstick > 25%, OR the C1
      random-init net's UPPER 95% bound vs strategy >= 0.40.
      A PASS is CAPPED at CONTINUE unless the net's pooled WR vs strategy
      exceeds C1's by >= 10 points with NON-OVERLAPPING 95% intervals.
  * WARNING when greedy and sampled WR differ by > 10 points (per opponent).
  * IN-DISTRIBUTION ONLY: train and eval share the same 25-team pool
    (StrongTeamBuilder per battle seed) — "not a generalization claim".
  * Both m1gate_round70 and m1gate_round80 are gated (pre-registered; no
    cherry-picking the earlier noisy 0.72 eval — winner's curse).

SUBJECTS play as p1 vs the yardstick on p2 (same placement as
`fitness.harness`), through the SAME parallel battle loop as the M1 trainer
(`gui.m1_gate_main.run_eval_battles` — one implementation, no duplication).
Draws are NEVER counted as wins (eval contract). Exit 0 = completed
(regardless of verdict), 1 = infra failure.

CLI:
  --subject  comma list of control subjects; each entry is a name
             (uses --opps) or `name:oppA,oppB` for per-subject yardsticks
             (C4 mirror). Names: random_init | random | heuristic |
             strategy | strategy_posterior.
  --ckpt     comma list of checkpoint files (each faces --opps).

Usage (two-stage recommended — controls decide validity first):

  # Run 1: controls C1, C3, C4a, C4b  (C2 is identical to C1 here)
  python -m tools.gate_eval \
      --subject random_init,heuristic,strategy:strategy,strategy_posterior:strategy \
      --opps strategy,strategy_posterior --n-per-pair 200 --modes greedy,sampled \
      --device cuda --seed 20261125 --servers 6 --per-server 2 --port-base 49980

  # Run 2 (only if the controls do NOT say YARDSTICK_INVALID): both ckpts
  python -m tools.gate_eval \
      --subject random_init \
      --ckpt runs/20261123T195442Z/checkpoints/m1gate_round70.pt,runs/20261123T195442Z/checkpoints/m1gate_round80.pt \
      --opps strategy,strategy_posterior --n-per-pair 200 --modes greedy,sampled \
      --device cuda --seed 20261125 --servers 6 --per-server 2 --port-base 49980

  (Including `--subject random_init` in Run 2 fills the validity check from
  the SAME run; otherwise pass the C1 numbers via a prior result.json.)

Results: printed table + ``runs/gate_eval_<UTC-ts>/result.json``.
A gate-eval dir has NO status.json, so the launcher's run discovery ignores it.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: M1 gate decision-rule thresholds (lower 95% CI bound, pooled per opponent).
GATE_PASS = 0.40
GATE_CONTINUE = 0.25

#: the verdict is the worst pooled bound over these yardsticks (if they ran).
PRIMARY_OPP = "strategy"
SECONDARY_OPP = "strategy_posterior"
VERDICT_OPPS = (PRIMARY_OPP, SECONDARY_OPP)

#: spec thresholds
MODE_DIFF_WARN = 0.10        # warn when greedy vs sampled WR differ by > 10 pts
C2_INVALID_WR = 0.25         # C2 (random bot) WR vs strategy above this -> invalid
C1_INVALID_UPPER = 0.40      # C1 (random-init) UPPER 95% bound vs strategy >= this -> invalid
NET_MUST_EXCEED_C1 = 0.10    # a PASS needs net_pooled_wr >= C1_wr + this

#: control subjects -> M1-pool reference names (same sources as the trainer).
#: ANY pool member can be a p1 subject, so every C1-C4 pairing is expressible.
SUBJECT_TO_POOL = {
    "random_init": "random",     # fresh random-init net (C1 — the floor)
    "random": "random",          # C2 — same pool member (identical here)
    "heuristic": "heuristic",    # C3 — the OLD heuristic bot
    "strategy": "strategy",      # C4a — the NEW rule bot (mirror)
    "strategy_posterior": "strategy_posterior",  # C4b — mirror
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


def wilson_upper(wins: int, n: int,
                 z: float = 1.959963984540054) -> float:
    """Upper bound of the Wilson 95% score interval (symmetric form)."""
    if n <= 0:
        return float("nan")
    p = wins / n
    z2_over_n = z * z / n
    centre = (p + z2_over_n / 2.0) / (1.0 + z2_over_n)
    half = (z * math.sqrt(p * (1.0 - p) / n + z2_over_n / 4.0)) \
        / (1.0 + z2_over_n)
    return min(1.0, centre + half)


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
    """The minimum of a set of (non-nan) bounds; nan if none. Pure function.

    (Kept for backward compatibility / tests; the current verdict uses the
    min of POOLED per-opponent bounds instead — see ``_entry_stats``.)
    """
    real = [b for b in bounds.values()
            if b is not None and not (isinstance(b, float)
                                     and math.isnan(b))]
    if not real:
        return float("nan")
    return min(real)


def pool_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool per-(mode) eval rows for ONE opponent into a single W/L/D row.

    Draws are summed (never wins). Pure function."""
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
        "upper_ci_95": wilson_upper(w, played),
    }


def _parse_list(value: str, what: str) -> List[str]:
    out = [s.strip() for s in value.split(",") if s.strip()]
    if not out:
        raise ValueError(f"{what} must be a non-empty comma-separated list")
    return out


def _parse_subjects(spec: Optional[str]
                    ) -> List[Tuple[str, Optional[List[str]]]]:
    """`name` (uses --opps) or `name:oppA,oppB` (per-subject yardsticks)."""
    if not spec:
        return []
    out: List[Tuple[str, Optional[List[str]]]] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            name, opps = entry.split(":", 1)
            out.append((name.strip(), _parse_list(opps, "--subject opps")))
        else:
            out.append((entry, None))
    if not out:
        raise ValueError("--subject must name at least one subject")
    return out


def _c5_stats() -> Dict[str, int]:
    """C5: the strategy bot's silent-fallback counters (0.0 if unavailable)."""
    try:
        from fitness.strategy_agent import fallback_stats
        return fallback_stats()
    except Exception:  # noqa: BLE001 — counters are advisory
        return {}


def _entry_stats(pairs: Dict[str, Any]) -> Dict[str, Any]:
    """Per-entry CI stats: per-mode bounds, pooled per opponent, verdict bound.

    Spec: print the Wilson lower 95% bound per (opponent x mode) AND pooled
    per opponent; the verdict is the WORST of the pooled bounds of `strategy`
    and `strategy_posterior`. Pure over the pair rows.
    """
    per_mode: Dict[str, Dict[str, float]] = {}
    pooled: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    for opp, by_mode in pairs.items():
        ok_rows = {m: r for m, r in by_mode.items()
                   if isinstance(r, dict) and "eval_w" in r}
        per_mode[opp] = {
            m: wilson_lower(r["eval_w"],
                            r["eval_w"] + r["eval_l"] + r["eval_d"])
            for m, r in ok_rows.items()}
        if ok_rows:
            pooled[opp] = pool_rows(list(ok_rows.values()))
            if {"greedy", "sampled"} <= set(ok_rows):
                wrs = {m: ok_rows[m]["eval_wr"] for m in ("greedy", "sampled")}
                if abs(wrs["greedy"] - wrs["sampled"]) > MODE_DIFF_WARN:
                    warnings.append(
                        f"{opp}: greedy {wrs['greedy']:.3f} vs sampled "
                        f"{wrs['sampled']:.3f} differ by "
                        f"{abs(wrs['greedy'] - wrs['sampled']):.1%} "
                        f"(> {MODE_DIFF_WARN:.0%})")
    bounds = [pooled[o]["lower_ci_95"] for o in VERDICT_OPPS if o in pooled]
    verdict_bound = min(bounds) if bounds else float("nan")
    return {
        "per_mode_lower_ci": per_mode,
        "pooled": pooled,
        "verdict_bound": verdict_bound,
        "base_verdict": gate_verdict(verdict_bound),
        "mode_diff_warnings": warnings,
    }


def build_validity(entries: Dict[str, Any]) -> Dict[str, Any]:
    """The spec's VALIDITY RULES, computed from the control entries. Pure.

    YARDSTICK_INVALID if C2 (random bot) beats the strategy yardstick > 25%,
    OR the C1 random-init net's UPPER 95% bound vs strategy >= 0.40.
    Otherwise a PASS is capped at CONTINUE unless the net's pooled WR vs
    strategy exceeds C1's by >= 10 points with non-overlapping 95% intervals.
    """
    v: Dict[str, Any] = {
        "rules": [
            "YARDSTICK_INVALID if C2 (random bot) beats strategy > 25%, OR "
            "C1 (random-init) UPPER 95% bound vs strategy >= 0.40",
            "PASS is CAPPED at CONTINUE unless net pooled WR vs strategy "
            "exceeds C1's by >= 10 pts with non-overlapping 95% intervals",
        ],
        "c1": None, "c2": None, "yardstick_invalid": False, "reasons": [],
    }

    def _pooled(label: str) -> Optional[Dict]:
        e = entries.get(label)
        if not e:
            return None
        pooled = e.get("stats", {}).get("pooled", {}).get(PRIMARY_OPP)
        if not pooled:
            return None
        return {"entry": label, **pooled}

    # EXACT labels ("control:random" must not match "control:random_init")
    c2 = _pooled("control:random")
    c1 = _pooled("control:random_init")
    if c1 is None and c2 is not None:
        c1 = dict(c2, note="C1 row absent; using the C2 row (identical net "
                           "in this codebase)")
    v["c1"] = c1
    v["c2"] = c2

    if c2 is not None and c2["win_rate"] > C2_INVALID_WR:
        v["reasons"].append(
            f"C2 random bot beats strategy {c2['win_rate']:.1%} "
            f"(> {C2_INVALID_WR:.0%})")
    if c1 is not None and c1["upper_ci_95"] >= C1_INVALID_UPPER:
        v["reasons"].append(
            f"C1 random-init net UPPER 95% bound {c1['upper_ci_95']:.3f} "
            f"vs strategy (>= {C1_INVALID_UPPER}) — a random net could pass")
    v["yardstick_invalid"] = bool(v["reasons"])
    return v


def apply_validity(base_verdict: str, net_pooled_strategy: Optional[Dict],
                   validity: Dict[str, Any]) -> str:
    """Apply the validity rules to a checkpoint's base verdict. Pure."""
    if validity.get("yardstick_invalid"):
        return "YARDSTICK_INVALID: " + "; ".join(validity["reasons"])
    if base_verdict == "PASS" and validity.get("c1") is not None \
            and net_pooled_strategy is not None:
        c1 = validity["c1"]
        margin_ok = (net_pooled_strategy["win_rate"] - c1["win_rate"]
                     >= NET_MUST_EXCEED_C1)
        overlap_ok = c1["upper_ci_95"] < net_pooled_strategy["lower_ci_95"]
        if not (margin_ok and overlap_ok):
            return ("CONTINUE (capped: PASS requires net pooled WR vs "
                    "strategy >= C1 + 10 pts with non-overlapping 95% CIs; "
                    f"margin_ok={margin_ok} overlap_ok={overlap_ok})")
    return base_verdict


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="tools.gate_eval")
    ap.add_argument("--ckpt", default=None,
                    help="comma-separated checkpoint files (.pt) to evaluate")
    ap.add_argument("--subject", default=None,
                    help="comma-separated CONTROLS: "
                         + ",".join(SUBJECT_TO_POOL)
                         + "; each may be `name` or `name:oppA,oppB` "
                         "(per-subject yardsticks, e.g. the C4 mirror)")
    ap.add_argument("--opps", default=",".join(VERDICT_OPPS),
                    help="comma-separated yardsticks (default: "
                         + ",".join(VERDICT_OPPS) + ")")
    ap.add_argument("--n-per-pair", type=int, default=200,
                    help="battles per (opponent x mode) pair (default 200)")
    ap.add_argument("--modes", default="greedy,sampled",
                    help="comma-separated: greedy and/or sampled (default both)")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--servers", type=int, default=6)
    ap.add_argument("--per-server", type=int, default=2)
    ap.add_argument("--port-base", type=int, default=49980)
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out", default=None,
                    help="result dir (default runs/gate_eval_<UTC-ts>)")
    args = ap.parse_args(argv)

    try:
        opps = _parse_list(args.opps, "--opps")
        subjects = _parse_subjects(args.subject)
    except ValueError as exc:
        print(f"FATAL: {exc}")
        return 1
    modes = _parse_list(args.modes, "--modes")
    unknown = [m for m in modes if m not in ("greedy", "sampled")]
    if unknown:
        print(f"FATAL: unknown mode(s) {unknown} (allowed: greedy, sampled)")
        return 1
    if args.n_per_pair <= 0:
        print("FATAL: --n-per-pair must be > 0")
        return 1
    ckpts = _parse_list(args.ckpt, "--ckpt") if args.ckpt else []
    if not ckpts and not subjects:
        print("FATAL: give --ckpt and/or --subject")
        return 1
    bad = [s for s, _ in subjects if s not in SUBJECT_TO_POOL]
    if bad:
        print(f"FATAL: unknown subject(s) {bad} (allowed: {list(SUBJECT_TO_POOL)})")
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
    need = set(opps)
    for s, so in subjects:
        need |= set(so) if so else set(opps)
    need |= set(SUBJECT_TO_POOL[s] for s, _ in subjects)
    missing = [o for o in sorted(need) if o not in full_pool]
    if missing:
        print(f"FATAL: name(s) {missing} not in the M1 pool "
              f"(have {full_pool.names()})")
        return 1

    results: Dict[str, Any] = {
        "gate": "m1_c4_final_c1c5",
        "spec": "GATE_CONTROLS_C1_C5.md",
        "in_distribution_only": True,
        "in_distribution_note": ("in-distribution only (same 25-team pool as "
                                 "training); not a generalization claim"),
        "format": args.format,
        "device": args.device,
        "seed": args.seed,
        "n_per_pair": args.n_per_pair,
        "modes": modes,
        "opps_default": opps,
        "entries": {},
    }

    # Controls FIRST (they decide YARDSTICK_INVALID before ckpt time is spent).
    entries: List[Tuple[str, Optional[Path], Optional[str],
                        Optional[List[str]]]] = []
    for name, so in subjects:
        entries.append((f"control:{name}", None, SUBJECT_TO_POOL[name], so))
    for p in ckpt_paths:
        entries.append((f"ckpt:{p.stem}", p, None, None))

    exit_ok = True
    for label, ckpt_path, pool_name, opps_override in entries:
        log(f"=== {label} ===")
        if ckpt_path is not None:
            log(f"loading checkpoint {ckpt_path}")
            from net.model import PokeNet
            net = PokeNet.load_file(str(ckpt_path), vocabs,
                                    device=args.device)
            my_ref = None
        else:
            net = None
            my_ref = full_pool.get(pool_name)
        entry_opps = opps_override if opps_override else opps
        results["entries"][label] = {
            "kind": "checkpoint" if ckpt_path is not None else "control",
            "source": (pool_name if ckpt_path is None else str(ckpt_path)),
            "opps": entry_opps,
            "pairs": {},
        }
        for opp in entry_opps:
            opp_ref = full_pool.get(opp)
            results["entries"][label]["pairs"][opp] = {}
            for pair_idx, mode in enumerate(modes):
                # SHARED seed space across all entries with the same
                # (opp, mode): identical battle seeds -> identical teams on
                # both sides -> controls and checkpoints face the SAME
                # matchups (the spec's "same seeds/teams").
                seed_base = 2_000_000 + args.seed * 10 + pair_idx * 100_000
                log(f"--- {opp} x {mode} ({args.n_per_pair} battles) ---")
                c5_before = _c5_stats()
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
                    results["entries"][label]["pairs"][opp][mode] = {
                        "error": str(exc)}
                    exit_ok = False
                    continue
                # C5: per-pair delta of the strategy bot's silent fallbacks
                after = _c5_stats()
                if after:
                    row["c5_fallback_delta"] = {
                        k: int(after.get(k, 0) - c5_before.get(k, 0))
                        for k in after}
                results["entries"][label]["pairs"][opp][mode] = row
                log(f"    {opp} x {mode}: {row['eval_w']}W/{row['eval_l']}L/"
                    f"{row['eval_d']}D crashed={row['eval_crashed']} "
                    f"wr={row['eval_wr']:.3f}")

    # ── verdicts (spec: pooled bounds, validity rules) ──────────────────
    for label, e in results["entries"].items():
        e["stats"] = _entry_stats(e["pairs"])
    validity = build_validity(results["entries"])
    results["validity"] = validity
    for label, e in results["entries"].items():
        if e["kind"] == "checkpoint":
            net_p = e["stats"]["pooled"].get(PRIMARY_OPP)
            e["final_verdict"] = apply_validity(
                e["stats"]["base_verdict"], net_p, validity)
        else:
            e["final_verdict"] = "CONTROL (validity input)"

    results["wall_s"] = round(time.time() - t0, 1)
    results["completed_ok"] = exit_ok

    (out_dir / "result.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # ── report ──────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"{'entry':<26} {'opp':<20} {'mode':<9} {'W/L/D':<12} "
          f"{'wr':>6} {'ci95 lo':>9}")
    print("-" * 78)
    for label, e in results["entries"].items():
        for opp, by_mode in e["pairs"].items():
            for mode, row in by_mode.items():
                if not isinstance(row, dict) or "eval_w" not in row:
                    err = row.get("error") if isinstance(row, dict) else row
                    print(f"{label:<26} {opp:<20} {mode:<9} ERROR: {err}")
                    continue
                n = row["eval_w"] + row["eval_l"] + row["eval_d"]
                lo = wilson_lower(row["eval_w"], n)
                c5 = row.get("c5_fallback_delta")
                c5s = (f"  c5={c5}" if c5 else "")
                print(f"{label:<26} {opp:<20} {mode:<9} "
                      f"{row['eval_w']}/{row['eval_l']}/{row['eval_d']:<6} "
                      f"{row['eval_wr']:>6.3f} {lo:>9.3f}{c5s}")
        st = e["stats"]
        for opp, p in st["pooled"].items():
            print(f"  {label:<26} {opp + ' (pooled)':<20} {'':<9} "
                  f"{p['wins']}/{p['losses']}/{p['draws']}".ljust(12) + " "
                  f"{p['win_rate']:>6.3f} {p['lower_ci_95']:>9.3f} "
                  f"(hi {p['upper_ci_95']:.3f})")
        for w in st["mode_diff_warnings"]:
            print(f"  WARNING {label}: {w}")
        print(f"  -> {label}: bound={st['verdict_bound']:.3f} "
              f"base={st['base_verdict']} FINAL={e['final_verdict']}")
        print("-" * 78)
    v = results["validity"]
    print("VALIDITY RULES:")
    for r in v["rules"]:
        print(f"  - {r}")
    if v["c1"]:
        print(f"  C1 random-init vs strategy: {v['c1']['win_rate']:.3f} "
              f"(upper {v['c1']['upper_ci_95']:.3f}, n={v['c1']['n']})")
    if v["c2"]:
        print(f"  C2 random bot vs strategy: {v['c2']['win_rate']:.3f} "
              f"(n={v['c2']['n']})")
    print(f"  YARDSTICK_INVALID: {v['yardstick_invalid']} "
          f"{('; ' + '; '.join(v['reasons'])) if v['reasons'] else ''}")
    print("NOTE: in-distribution only (same 25-team pool as training); "
          "not a generalization claim.")
    print(f"wall={results['wall_s']}s  result={out_dir / 'result.json'}")
    print("=" * 78)
    return 0 if exit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
