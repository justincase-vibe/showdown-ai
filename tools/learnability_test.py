"""(a) Learnability gate — "is the fixed PPO pipeline actually learning?"

Spec (2026-11-24 user DECISION, GATE_CONTROLS_C1_C5.md §0 item 2):
    random-only opponent, 50 battles/round; win rate vs random must exceed
    85% within about 15 rounds, else it is a bug.

Why the test is meaningful: at round 0 the main net is a FRESH random init
and the opponent is ALSO a random-init net (the SAME bot as the gate's
C1/C2 floor — `fitness.ladder.random_reference`), so the starting win rate
is ~50%. Getting to >85% within ~15 rounds of real PPO updates is a genuine
learning requirement: a broken loop (dead gradients, wrong advantage sign,
collapsed policy) will not get there.

No code written twice — every heavy piece is REUSED:
  * training battles : tools.train_step4_vs_pool.play_vs_checkpoint
    (the shared main-learner battle loop, main net on p1)
  * opponent         : fitness.ladder.random_reference (gate C1/C2 floor)
  * PPO update       : net.ppo.PPOTrainer — the FIX PACK under test
    (advantage normalization + minibatch 256 + approx-KL early stop 0.03)
  * eval battles     : gui.m1_gate_main.run_eval_battles (the shared
    parallel loop; imported read-only, the file is untouched)
  * stop thresholds  : DEG_ENTROPY_FLOOR / DEG_KL_HARD imported from
    gui.m1_gate_main so they cannot drift from the M1 guard

Stop rules (M1 gate DECISION): entropy collapse, switch share < 3% for 3
consecutive rounds, or KL blow-up -> STOP and report (do not train blindly).

CPU by design: the (a) decision says "CPU-light only", and the imported
train_step4_vs_pool module pins CUDA_VISIBLE_DEVICES=-1 at import time.

Run (CPU smoke — validates the whole pipeline end-to-end, small):
    python -m tools.learnability_test --smoke
Run (the actual test — 15 rounds x 50 battles vs random + evals):
    python -m tools.learnability_test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np
import torch

from net.vocab_db import build_vocabs
from net.agent import PokeAgent, make_random_net
from net.ppo import (PPOTrainer, RolloutBuffer, chosen_logprobs, fill_gae,
                     steps_to_batch)
from net.trainer import TrainConfig
from fitness.ladder import random_reference
from pool import PoolConfig, ResourcePool

# shared building blocks (imported read-only — no code written twice)
from tools.train_step4_vs_pool import play_vs_checkpoint, play_with_watchdog
from gui.m1_gate_main import (DEG_ENTROPY_FLOOR, DEG_KL_HARD,
                              run_eval_battles)

# ── the (a) gate spec ─────────────────────────────────────────────────────────
PASS_WR = 0.85                 # win rate vs random required within ~15 rounds
SWITCH_SHARE_FLOOR = 0.03      # < 3% switch actions = a policy that never switches
SWITCH_SHARE_STREAK = 3        # ...for 3 consecutive rounds -> stop (M1 decision)

FMT = "gen9ou"
TRAIN_PORT_BASE = 49990        # handoff: 49990+ = smokes (49980-49987 = training pool)
EVAL_PORT_BASE = TRAIN_PORT_BASE + 16   # DISJOINT from the live training pool
EVAL_SEED_BASE = 2_000_000     # disjoint from gate (1_000_000+) and training spaces
EVAL_MODE_OFFSET = 1_000      # per-mode seed offset: sampled/greedy stay disjoint
                              # (eval_battles <= 100 << 1000, so no overlap)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure decision logic (hermetic — unit-tested without any server)
# ═══════════════════════════════════════════════════════════════════════════════
def check_stop_rules(ppo_metrics: Dict[str, Any], switch_share: float,
                     prior_low_streak: int) -> Optional[str]:
    """M1 stop rules (2026-11-24 DECISION). Returns a stop reason or None.

    ``ppo_metrics`` : the round's ``PPOTrainer.update()`` row (may lack keys
                      when the update was skipped — handled).
    ``switch_share``: fraction of this round's main-agent actions that were
                      switches (nan when the round produced no actions).
    ``prior_low_streak``: consecutive PRIOR rounds already below the floor.
    """
    ent = ppo_metrics.get("entropy")
    if ent is not None and math.isfinite(float(ent)) and float(ent) < DEG_ENTROPY_FLOOR:
        return f"entropy collapse (entropy={float(ent):.3f} < {DEG_ENTROPY_FLOOR})"
    kl = ppo_metrics.get("kl_approx")
    if kl is not None and math.isfinite(float(kl)) and float(kl) >= DEG_KL_HARD:
        return f"KL blow-up (kl_approx={float(kl):.3f} >= {DEG_KL_HARD})"
    share_low = (isinstance(switch_share, float)
                 and math.isfinite(switch_share)
                 and switch_share < SWITCH_SHARE_FLOOR)
    if share_low and prior_low_streak >= SWITCH_SHARE_STREAK - 1:
        return (f"switch share < {SWITCH_SHARE_FLOOR:.0%} for "
                f"{SWITCH_SHARE_STREAK} consecutive rounds "
                f"(this round: {switch_share:.3f})")
    return None


def decide_verdict(evals: List[Dict[str, Any]], max_rounds: int,
                   pass_wr: float = PASS_WR, stopped: bool = False,
                   stop_reason: str = "") -> Dict[str, Any]:
    """PASS iff the SAMPLED-mode eval win rate vs random >= pass_wr at some
    round <= max_rounds.

    Sampled is the decision mode: it is the net's natural play mode (the
    gate reports both modes because the yardstick is strong; here the
    criterion is one clean pipeline-sanity number). Greedy is reported
    alongside for context.
    """
    best = None
    for e in evals:
        if e.get("round", -1) > max_rounds:
            continue
        row = e.get("sampled") or {}
        wr = row.get("eval_wr")
        if wr is None or not math.isfinite(wr):
            continue
        if best is None or wr > best["wr"]:
            best = {"round": e["round"], "wr": float(wr),
                    "w": row.get("eval_w"), "l": row.get("eval_l"),
                    "d": row.get("eval_d")}
    if best is not None and best["wr"] >= pass_wr:
        verdict = "PASS"
    elif stopped:
        verdict = "STOPPED_DEGENERATE"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "pass_wr": pass_wr,
        "max_rounds": max_rounds,
        "best_sampled": best,
        "stopped": stopped,
        "stop_reason": stop_reason,
        "note": ("vs the random-init floor (gate C1/C2 bot), sampled play; "
                 "in-distribution only (gen9ou pool, ~50/50 at round 0)"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  The run
# ═══════════════════════════════════════════════════════════════════════════════
def _run_eval(rnd: int, main_net, opp_ref, vocabs, eval_battles: int,
              max_steps: int, n_servers: int, per_server: int) -> Dict[str, Any]:
    """Both play modes vs the random floor, on the DISJOINT eval port base."""
    row: Dict[str, Any] = {"round": rnd}
    for mode in ("sampled", "greedy"):
        det = (mode == "greedy")
        r = run_eval_battles(
            main_net, opp_ref, FMT, vocabs, "cpu",
            n_battles=eval_battles, max_steps=max_steps,
            n_servers=n_servers, per_server=per_server,
            port_base=EVAL_PORT_BASE,
            seed_base=EVAL_SEED_BASE + rnd * 100_000
            + (EVAL_MODE_OFFSET if det else 0),
            deterministic=det,
            label=f"learn_eval_r{rnd}_{mode}")
        row[mode] = r
        print(f"  [eval r{rnd}] {mode}: WR={r.get('eval_wr'):.3f} "
              f"(W/L/D = {r.get('eval_w')}/{r.get('eval_l')}/{r.get('eval_d')}, "
              f"crashed={r.get('eval_crashed')})", flush=True)
    return row


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    smoke = bool(args.smoke)
    rounds = 2 if smoke else args.rounds
    battles_per_round = 10 if smoke else args.battles_per_round
    eval_every = 1 if smoke else args.eval_every
    eval_battles = 20 if smoke else args.eval_battles
    n_servers = 1 if smoke else args.n_servers
    per_server = 2
    max_steps = args.max_steps

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(args.run_dir) if args.run_dir else \
        Path("runs") / f"learnability_{ts}{'_smoke' if smoke else ''}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    print(f"[learnability] spec: WR vs random >= {PASS_WR} within "
          f"{rounds} rounds (50 battles/round spec; this run: "
          f"{battles_per_round}/round, {'SMOKE' if smoke else 'full'})", flush=True)
    print(f"[learnability] run dir: {run_dir}", flush=True)

    vocabs = build_vocabs()
    # the code under test: the fixed PPO core (fix-pack defaults:
    # minibatch 256, approx-KL early stop at 0.03, advantage normalization)
    main_net = make_random_net(vocabs, "cpu")
    main_agent = PokeAgent(main_net, seed=args.seed, device="cpu")
    opp_ref = random_reference(vocabs, device="cpu")   # the gate C1/C2 floor

    cfg = TrainConfig(
        battle_format=FMT, battles_per_round=battles_per_round,
        max_steps=max_steps, seed=args.seed, device="cpu",
        epochs=2, checkpoint_dir=str(run_dir / "checkpoints"),
        save_every=1, n_rounds=rounds, battle_timeout=args.battle_timeout,
    )
    ppo = PPOTrainer(main_net, lr=cfg.lr, clip_eps=cfg.clip_eps,
                     vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                     max_norm=cfg.max_norm, epochs=cfg.epochs, device="cpu")
    buffer = RolloutBuffer()

    config_row = {
        "spec": f"WR vs random >= {PASS_WR} within {rounds} rounds",
        "smoke": smoke, "rounds": rounds,
        "battles_per_round": battles_per_round, "eval_every": eval_every,
        "eval_battles": eval_battles, "n_servers": n_servers,
        "per_server": per_server, "max_steps": max_steps,
        "seed": args.seed, "format": FMT,
        "ppo": {"lr": cfg.lr, "clip_eps": cfg.clip_eps, "vf_coef": cfg.vf_coef,
                "ent_coef": cfg.ent_coef, "max_norm": cfg.max_norm,
                "epochs": cfg.epochs, "minibatch_size": ppo.minibatch_size,
                "target_kl": ppo.target_kl},
        "stop_rules": {"entropy_floor": DEG_ENTROPY_FLOOR, "kl_hard": DEG_KL_HARD,
                       "switch_share_floor": SWITCH_SHARE_FLOOR,
                       "switch_share_streak": SWITCH_SHARE_STREAK},
        "ports": {"train": TRAIN_PORT_BASE, "eval": EVAL_PORT_BASE},
    }
    (run_dir / "config.json").write_text(json.dumps(config_row, indent=2))

    pool = ResourcePool(
        PoolConfig(n_servers=n_servers, per_server=per_server,
                   port_base=TRAIN_PORT_BASE, boot_timeout=120),
        log=lambda m: print(f"  [pool] {m}", flush=True))
    print(f"[learnability] starting training pool "
          f"({n_servers} server(s) x {per_server}) ...", flush=True)
    if not all(pool.start(wait_healthy=True).values()):
        pool.shutdown()
        raise SystemExit("[learnability] training pool not healthy — aborting")

    metrics_path = run_dir / "metrics.jsonl"
    evals: List[Dict[str, Any]] = []
    stopped = False
    stop_reason = ""
    low_streak = 0
    rounds_completed = 0

    def _log(row: Dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf8") as fh:
            fh.write(json.dumps(row) + "\n")

    try:
        # ── round 0: the untrained baseline (~50% vs the random floor) ──
        print("[learnability] round 0 — baseline eval (untrained net)", flush=True)
        evals.append(_run_eval(0, main_net, opp_ref, vocabs, eval_battles,
                               max_steps, n_servers, per_server))
        _log({"round": 0, "kind": "baseline", "evals": evals[-1]})

        for rnd in range(1, rounds + 1):
            print(f"\n[learnability] ── round {rnd}/{rounds} "
                  f"({battles_per_round} battles vs random) ──", flush=True)
            buffer.clear()
            n_battles = n_crash = wins = losses = 0
            kinds: List[str] = []
            for i in range(battles_per_round):
                seed = args.seed + (rnd - 1) * 1000 + i
                opp_agent = opp_ref.make_agent(seed=seed + 7000)  # fresh net/battle
                with pool.battle_slot() as server_cfg:
                    r = play_with_watchdog(
                        play_vs_checkpoint, main_agent, opp_agent, FMT, seed,
                        max_steps=max_steps, server_configuration=server_cfg,
                        vocabs=vocabs, timeout_s=args.battle_timeout)
                if r.get("timed_out") or r.get("crashed"):
                    n_crash += 1
                    print(f"  [r{rnd} b{i+1}] {'TIMEOUT' if r.get('timed_out') else 'CRASH'}: "
                          f"{r.get('error', '')[:70]}", flush=True)
                    continue
                traj = r["traj"]
                if traj:
                    fill_gae(traj, gamma=cfg.gamma, lam=cfg.lam)
                    buffer.extend(traj)
                n_battles += 1
                if r.get("a1_won"):
                    wins += 1
                elif r.get("a2_won"):
                    losses += 1
                kinds.extend(r.get("action_kinds", []))
            share = (sum(1 for k in kinds if k == "switch") / len(kinds)) \
                if kinds else float("nan")
            share_low = math.isfinite(share) and share < SWITCH_SHARE_FLOOR
            prior_streak = low_streak
            low_streak = low_streak + 1 if share_low else 0

            # ── the PPO update under test (fix pack) ──
            if len(buffer) >= 2:
                batch = steps_to_batch(buffer.steps, "cpu")
                with torch.no_grad():
                    batch["old_logprob"] = chosen_logprobs(main_net, batch).clone()
                advs = np.array([s.advantage for s in buffer.steps], dtype=np.float32)
                rets = np.array([s.ret for s in buffer.steps], dtype=np.float32)
                m = ppo.update(batch, advantages=advs, returns=rets)
            else:
                m = {"skipped": True, "reason": f"too_few ({len(buffer)})"}
            n_trans = len(buffer) if "skipped" not in m else 0
            buffer.clear()

            # checkpoint EVERY round (cheap; the user's "save by type + battles" habit)
            torch.save({
                "net": main_net.state_for_save(),
                "config": cfg.to_dict(),
                "round": rnd,
                "opp": "random (C1/C2 floor)",
                "train_wins": wins, "train_losses": losses,
            }, run_dir / "checkpoints" / f"learn_r{rnd}.pt")

            row = {
                "round": rnd, "kind": "train",
                "battles": n_battles, "crashes": n_crash,
                "train_wins": wins, "train_losses": losses,
                "transitions": n_trans,
                "switch_share": share,
                "ppo": m,
            }
            _log(row)
            print(f"  [r{rnd}] battles={n_battles} (crash={n_crash}) "
                  f"W/L={wins}/{losses} trans={n_trans} "
                  f"switch_share={share:.2f} "
                  f"policy={m.get('policy', float('nan')):.4f} "
                  f"value={m.get('value', float('nan')):.4f} "
                  f"ent={m.get('entropy', float('nan')):.4f} "
                  f"kl={m.get('kl_approx', float('nan')):.5f} "
                  f"grad_steps={m.get('grad_steps', float('nan'))}", flush=True)

            reason = check_stop_rules(m, share, prior_streak)
            if reason:
                stopped = True
                stop_reason = reason
                print(f"\n[learnability] STOP (degenerate signal): {reason}", flush=True)
                _log({"round": rnd, "kind": "stop", "reason": reason})
                break

            if rnd % eval_every == 0:
                evals.append(_run_eval(rnd, main_net, opp_ref, vocabs,
                                       eval_battles, max_steps,
                                       n_servers, per_server))
            rounds_completed = rnd
    finally:
        pool.shutdown()
        print("[learnability] training pool shut down", flush=True)

    verdict = decide_verdict(evals, max_rounds=rounds, stopped=stopped,
                             stop_reason=stop_reason)
    report = {
        "spec": config_row["spec"],
        "smoke": smoke,
        "rounds_completed": rounds_completed,
        "baseline_round0": evals[0] if evals else None,
        "evals": evals,
        "verdict": verdict["verdict"],
        "pass_wr": verdict["pass_wr"],
        "best_sampled": verdict["best_sampled"],
        "stopped": stopped,
        "stop_reason": stop_reason,
        "total_seconds": round(time.time() - t0, 1),
        "note": verdict["note"],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (run_dir / "evals.json").write_text(json.dumps(evals, indent=2, default=str))

    b = verdict["best_sampled"]
    print("\n" + "=" * 68, flush=True)
    print(f"LEARNABILITY VERDICT: {verdict['verdict']}", flush=True)
    print(f"  criterion : sampled WR vs random >= {verdict['pass_wr']} "
          f"within {rounds} rounds", flush=True)
    if b:
        print(f"  best      : round {b['round']} WR={b['wr']:.3f} "
              f"(W/L/D={b['w']}/{b['l']}/{b['d']})", flush=True)
    else:
        print("  best      : no finite eval row", flush=True)
    if stopped:
        print(f"  stop      : {stop_reason}", flush=True)
    print(f"  report    : {run_dir / 'report.json'}", flush=True)
    print(f"  note      : {verdict['note']}", flush=True)
    print(f"  time      : {report['total_seconds']}s", flush=True)
    print("=" * 68, flush=True)
    return 0 if verdict["verdict"] == "PASS" else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rounds", type=int, default=15,
                   help="max training rounds (spec: ~15)")
    p.add_argument("--battles-per-round", type=int, default=50,
                   help="spec: 50 battles/round")
    p.add_argument("--eval-every", type=int, default=2,
                   help="evaluate every N rounds (round 0 baseline always)")
    p.add_argument("--eval-battles", type=int, default=100,
                   help="eval battles per play mode")
    p.add_argument("--n-servers", type=int, default=2,
                   help="training + eval servers (RAM: keep <= 2x2 each here)")
    p.add_argument("--max-steps", type=int, default=300, help="per-battle step cap")
    p.add_argument("--battle-timeout", type=int, default=240, help="per-battle watchdog (s)")
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--run-dir", type=str, default=None,
                   help="default: runs/learnability_<ts>")
    p.add_argument("--smoke", action="store_true",
                   help="2 rounds x 10 battles, eval every round x 20 battles "
                        "(pipeline validation, CPU)")
    args = p.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
