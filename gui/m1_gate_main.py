"""M1 Gate (C2 Arm A) — single-learner PPO training subprocess.

Launched by the GUI (gui.desktop) when mode="m1gate". Trains ONE main
learner (PPO) against the M1 opponent pool:
  random / heuristic / strategy / strategy_posterior (selected by config).

GPU-ready: device="cuda" uses the RTX 5090; device="cpu" for debugging.
The GPU guard fails LOUD if cuda is requested but unavailable.

Usage (normally launched by the GUI):
    python -m gui.m1_gate_main --config <run-dir>/config.json --run-dir <run-dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _finite(row: Dict[str, Any]) -> None:
    """A non-finite loss/metric means the update diverged — fail LOUD."""
    for k in ("loss", "policy", "value", "entropy", "frac_clipped", "kl_approx"):
        v = row.get(k)
        if v is None:
            continue
        if not math.isfinite(float(v)):
            raise RuntimeError(
                f"non-finite metric {k}={v!r} in round {row.get('round')!r}")


# ── Degenerate-training guard (WARN, never a hard stop) ─────────────────────
# The point is to WAKE the user to a dead net (loud log + status flag), NOT to
# kill a possibly-good run — a hard-stop false-positive would cost an overnight
# run, a warning false-positive is cheap.
#
# Signals split into IMMEDIATE (a single severe step is always meaningful) and
# SUSTAINED (must persist for ``DEG_SUSTAINED`` rounds). The split exists because
# a YOUNG net has naturally volatile KL and a low-spread value head during warmup
# (rounds 1-3); reacting to a single noisy early round is cry-wolf. A young net
# is high-entropy, so the entropy rule can't falsely fire on warmup and stays
# immediate.
DEG_ENTROPY_FLOOR = 0.30      # IMMEDIATE: entropy below this = near-deterministic
DEG_KL_HARD = 1.00            # IMMEDIATE: per-round KL at/above this = blow-up
DEG_KL_WARN = 0.50            # SUSTAINED: KL at/above this (creeping)
DEG_VALUE_STD_FLOOR = 0.02    # SUSTAINED: value std below this (no win/loss)
DEG_SUSTAINED = 2             # consecutive rounds a soft signal must hold
DEG_NO_PROGRESS_ROUNDS = 5    # straight rounds of 0 training wins
DEG_FLAT_YARDSTICK_EVALS = 3  # straight evals at exactly 0.0 wr


def _streak(rows: list, pred) -> int:
    """Count the trailing rows (from the latest backwards) that satisfy ``pred``.

    A single non-matching row breaks the streak, so this measures *persistence*,
    not just "the latest round is bad".
    """
    n = 0
    for r in reversed(rows):
        if pred(r):
            n += 1
        else:
            break
    return n


def _degenerate_check(rows: list) -> list:
    """Human-readable degenerate-training warnings, based on the recent ``rows``.

    Pure function (no side effects -> unit-testable without torch). Returns ``[]``
    when healthy. Rules:

    * **entropy collapse** — latest ``entropy`` < ``DEG_ENTROPY_FLOOR`` (immediate);
    * **KL blow-up** — latest ``kl_approx`` >= ``DEG_KL_HARD`` (immediate);
    * **KL sustained** — ``kl_approx`` >= ``DEG_KL_WARN`` for ``DEG_SUSTAINED``+
      consecutive rounds (creeping divergence);
    * **value-head collapse** — ``value_std`` < ``DEG_VALUE_STD_FLOOR`` for
      ``DEG_SUSTAINED``+ rounds (no win/loss signal);
    * **no progress** — ``DEG_NO_PROGRESS_ROUNDS`` straight rounds of 0 wins;
    * **flat yardstick** — ``DEG_FLAT_YARDSTICK_EVALS`` straight evals at 0.0.
    """
    warns: list = []
    if not rows:
        return warns
    last = rows[-1]

    def _num(v: Any) -> bool:
        return v is not None and isinstance(v, (int, float)) \
            and math.isfinite(float(v))

    # 1. entropy collapse — immediate (a young net is high-entropy, so this
    #    cannot falsely fire on warmup; low entropy is always a real signal).
    ent = last.get("entropy")
    if _num(ent) and ent < DEG_ENTROPY_FLOOR:
        warns.append(f"entropy collapse: entropy={ent:.3f} < {DEG_ENTROPY_FLOOR} "
                     f"(policy is near-deterministic)")

    # 2. KL — severe single-step blow-up is immediate; a softer elevation must
    #    persist (one big PPO step is not degeneration, sustained is).
    kl = last.get("kl_approx")
    if _num(kl) and kl >= DEG_KL_HARD:
        warns.append(f"KL blow-up: kl_approx={kl:.3f} >= {DEG_KL_HARD} "
                     f"(updates diverging)")
    elif _streak(rows, lambda r: _num(r.get("kl_approx")) and r["kl_approx"] >= DEG_KL_WARN) >= DEG_SUSTAINED:
        warns.append(f"KL sustained: kl_approx={kl:.3f} >= {DEG_KL_WARN} for "
                     f"{DEG_SUSTAINED}+ rounds (watch for divergence)")

    # 3. value-head collapse — sustained low spread (warmup is naturally low).
    vs = last.get("value_std")
    if _num(vs) and _streak(rows, lambda r: _num(r.get("value_std")) and r["value_std"] < DEG_VALUE_STD_FLOOR) >= DEG_SUSTAINED:
        warns.append(f"value-head collapse: value_std={vs:.4f} < {DEG_VALUE_STD_FLOOR} "
                     f"for {DEG_SUSTAINED}+ rounds (no win/loss signal)")

    # 4. no progress — N straight rounds of 0 training wins.
    if _streak(rows, lambda r: r.get("a1_wins", 0) == 0
               and r.get("a1_losses", 0) > 0) >= DEG_NO_PROGRESS_ROUNDS:
        warns.append(f"no progress: 0 training wins for "
                     f"{DEG_NO_PROGRESS_ROUNDS}+ straight rounds")

    # 5. flat yardstick — N straight evals at exactly 0.0 wr.
    evals = [r for r in rows if "eval_wr" in r and _num(r.get("eval_wr"))]
    if _streak(evals, lambda r: r.get("eval_wr") == 0.0) >= DEG_FLAT_YARDSTICK_EVALS:
        warns.append(f"flat yardstick: eval_wr=0.0 for "
                     f"{DEG_FLAT_YARDSTICK_EVALS}+ straight evals (not learning)")

    return warns


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="gui.m1_gate_main")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    rd = Path(args.run_dir).resolve()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    from gui import runstate
    runstate.harden_stdio()
    import torch

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    fmt = cfg.get("battle_format", "gen9ou")
    device = cfg.get("device", "cuda")
    battles = int(cfg.get("battles_per_round", 20))
    rounds = int(cfg.get("n_rounds", 10))
    epochs = int(cfg.get("epochs", 2))
    lr = float(cfg.get("lr", 3e-4))
    seed = int(cfg.get("seed", 20261123))
    max_steps = int(cfg.get("max_steps", 200))
    n_servers = int(cfg.get("n_servers", 2))
    per_server = int(cfg.get("per_server", 2))
    port_base = int(cfg.get("port_base", 49980))
    # C3 (yardstick eval): every eval_every rounds + the final round, the
    # CURRENT net (greedy) plays eval_battles vs the frozen reference
    # (default: strategy). 0 disables. Additive metrics only.
    eval_every = int(cfg.get("eval_every", 10))
    eval_battles = int(cfg.get("eval_battles", 100))
    eval_opp = cfg.get("eval_opp", "strategy")
    # S3.1: learner-side randomization (seeded 50/50 per battle in training;
    # eval can report per seat). OFF by default — defaults reproduce the
    # historical learner-is-p1 behaviour exactly.
    randomize_sides = bool(cfg.get("randomize_sides", False))

    log(f"start M1 GATE: fmt={fmt} device={device} "
        f"battles/round={battles} rounds={rounds} "
        f"pool=random:{cfg.get('opp_random', True)} "
        f"heuristic:{cfg.get('opp_heuristic', True)} "
        f"strategy:{cfg.get('opp_strategy', True)} "
        f"posterior:{cfg.get('opp_posterior', True)} "
        f"eval={eval_battles}vs{eval_opp} every {eval_every} rounds "
        f"randomize_sides={randomize_sides} "
        f"resume={cfg.get('resume_from')!r}")

    runstate.write_status(rd, state="running", started=runstate.now_iso())

    # ── GPU guard ────────────────────────────────────────────────────────
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            msg = "device='cuda' requested but torch.cuda.is_available() is False"
            runstate.write_status(rd, state="failed", error=msg, exit_code=1)
            log(f"FATAL: {msg}")
            return 1
        log(f"CUDA available: {torch.cuda.get_device_name(0)}")

    # ── build the M1 opponent pool ───────────────────────────────────────
    from net.vocab_db import build_vocabs
    from fitness.m1_pool import build_m1_pool

    vocabs = build_vocabs()
    pool_names = []
    if cfg.get("opp_random", True):
        pool_names.append("random")
    if cfg.get("opp_heuristic", True):
        pool_names.append("heuristic")
    if cfg.get("opp_strategy", True):
        pool_names.append("strategy")
    if cfg.get("opp_posterior", True):
        pool_names.append("strategy_posterior")
    if not pool_names:
        pool_names = ["random"]  # at least one opponent

    # Build the full pool, then filter to the selected opponents
    full_pool = build_m1_pool(fmt=fmt, vocabs=vocabs, device=device)
    # Create a filtered ladder with only the selected opponents
    from fitness.ladder import Ladder, Reference
    selected = [full_pool.get(n) for n in pool_names]
    pool = Ladder(selected)
    log(f"opponent pool: {pool.names()} (weights: "
        f"{[r.weight for r in selected]})")

    # ── net init (random or resume) ─────────────────────────────────────
    from net.model import PokeNet
    from net.agent import PokeAgent, make_random_net
    from net.trainer import SelfPlayTrainer, TrainConfig

    ckpt_dir = str(rd / "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    resume_from = cfg.get("resume_from")
    start_round = 0
    if resume_from:
        payload = torch.load(resume_from, map_location="cpu", weights_only=True)
        net = PokeNet.load(PokeNet.unwrap_payload(payload), vocabs, device=device)
        start_round = int(payload.get("round", 0))
        log(f"resumed from {resume_from} (round {start_round})")
    else:
        net = make_random_net(vocabs, device)

    tc = TrainConfig(
        battle_format=fmt,
        battles_per_round=battles,
        n_rounds=rounds,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
        checkpoint_dir=ckpt_dir,
        save_every=1,
    )
    trainer = SelfPlayTrainer(net, tc, agent=PokeAgent(net, seed=seed, device=device))
    trainer.round = start_round

    # ── battle loop (the M1 pool replaces the random opponent) ──────────
    def battle_hook(battle: int) -> None:
        try:
            runstate.write_heartbeat(rd, trainer.round, battle, "m1gate")
        except Exception:
            pass

    def stop_check() -> bool:
        return runstate.stop_requested(rd)

    rounds_planned = start_round + rounds
    runstate.write_status(rd, rounds_planned=rounds_planned, round=start_round)

    t0 = time.time()
    state_final = "completed"
    history: list = []   # recent metric rows -> the degenerate guard's input
    try:
        for round_no in range(rounds):
            if stop_check():
                state_final = "stopped"
                break
            runstate.write_status(rd, round=trainer.round)
            # The trainer's train_round plays against random by default.
            # For the M1 gate, we override the opponent selection.
            # The SelfPlayTrainer uses play_and_collect which plays vs random.
            # We need to use the M1 pool instead. Use the step4 approach:
            # play each battle against a randomly-selected pool member.
            row = _train_round_m1(trainer, pool, fmt, vocabs, device,
                                  battles, max_steps, n_servers, per_server,
                                  port_base, battle_hook, stop_check,
                                  randomize_sides=randomize_sides)
            row["wall_s"] = round(time.time() - t0, 1)
            _finite(row)
            # C3: the yardstick eval — every eval_every rounds + ALWAYS the
            # final round (the most important data point). full_pool is used
            # so the yardstick exists even if the user excluded `strategy`
            # from the TRAINING pool. Additive keys only — the existing
            # row/metrics contract is unchanged (6.6a discipline).
            if eval_every >= 1 and (
                    (round_no + 1) % eval_every == 0 or (round_no + 1) == rounds):
                ev = _eval_round_m1(trainer, full_pool, fmt, vocabs, device,
                                    eval_opp, eval_battles, max_steps,
                                    n_servers, per_server, port_base,
                                    battle_hook, stop_check)
                row.update(ev)
                log(f"round {row.get('round')}: EVAL vs {ev.get('eval_opp')}: "
                    f"W/L/D={ev.get('eval_w')}/{ev.get('eval_l')}/"
                    f"{ev.get('eval_d')} wr={ev.get('eval_wr')}")
            runstate.append_metric(rd, row)
            # Degenerate-training guard (WARN, never a hard stop): a dead net
            # must be LOUD in the log + status.json so the user is woken, not
            # left with a silent garbage checkpoint. Pure fn of recent rows.
            history.append(row)
            warns = _degenerate_check(history[-5:])
            if warns:
                for w in warns:
                    log(f"*** DEGENERATE (round {row.get('round')}): {w}")
                runstate.write_status(rd, degenerate_warnings=warns,
                                      degenerate_round=row.get("round"))
            else:
                runstate.write_status(rd, degenerate_warnings=[])
            log(f"round {row.get('round')}: battles={row.get('battles')} "
                f"transitions={row.get('transitions')} "
                f"loss={row.get('loss')} "
                f"ckpt={row.get('checkpoint')}")
    except Exception as exc:
        runstate.write_status(rd, state="failed", round=trainer.round,
                              error=f"{type(exc).__name__}: {exc}", exit_code=1)
        log(f"FATAL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        try:
            from damage import worker as _worker
            _worker.shutdown_pool()
        except Exception:
            pass

    runstate.write_status(rd, state=state_final, round=trainer.round, exit_code=0)
    log(f"{state_final}: rounds={trainer.round} (planned {rounds_planned}) "
        f"wall={time.time() - t0:.0f}s")
    return 0


_MOVES_LK_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _moves_lookup() -> Dict[str, Dict[str, Any]]:
    """Cached move-DB lookup for the S3.4 action-mix metric (one read per
    process; the classifier reads only category/side_condition/boosts)."""
    global _MOVES_LK_CACHE
    if _MOVES_LK_CACHE is None:
        from net.action_mix import build_moves_lookup
        _MOVES_LK_CACHE = build_moves_lookup()
    return _MOVES_LK_CACHE


def _learner_won(record: Dict[str, Any]) -> bool:
    """Learner-relative WIN from a battle record (S3.1, side-aware).

    Records are SEAT-relative (a1 = the p1 seat). When the learner sat on p2
    (``record['side'] == 'p2'``) the win is a2_won. Default (no ``side`` key,
    or "p1") = the historical a1-is-learner mapping.
    """
    if record.get("side", "p1") == "p1":
        return bool(record["a1_won"]) and not bool(record["a2_won"])
    return bool(record["a2_won"]) and not bool(record["a1_won"])


def _learner_lost(record: Dict[str, Any]) -> bool:
    """Learner-relative LOSS (see :func:`_learner_won`)."""
    if record.get("side", "p1") == "p1":
        return bool(record["a2_won"]) and not bool(record["a1_won"])
    return bool(record["a1_won"]) and not bool(record["a2_won"])


def _train_round_m1(trainer, pool, fmt, vocabs, device,
                    n_battles, max_steps, n_servers, per_server, port_base,
                    battle_hook, stop_check,
                    randomize_sides: bool = False) -> Dict[str, Any]:
    """One PPO round: play n_battles against the M1 pool, GAE, update.

    Thread-per-battle — the pool's design front (D93.2: the Phase 0 gate ran
    24 concurrent battles on 8 servers; the slot core is thread-based
    specifically so poke-env's sync step() can be called from worker threads).
    ``min(n_battles, n_servers*per_server)`` workers each pull the next battle
    index, take a slot, play ONE battle, release. The main learner's NET is
    shared READ-ONLY (eval mode, no state mutation during collection); each
    battle gets a FRESH PokeAgent (fresh per-battle RNG — the same isolation
    contract the opponents use, and it keeps the shared torch.Generator from
    being touched by two threads at once). The PPO update runs
    single-threaded AFTER all battles — on-policy, exactly as before."""
    import numpy as np
    import threading
    import torch
    from net.ppo import fill_gae, steps_to_batch, chosen_logprobs, PPOTrainer
    from net.ppo import RolloutStep, RolloutBuffer
    from net.agent import PokeAgent
    from net.env import (ShowdownEnv, StrongTeamBuilder, build_action_list,
                         build_obs, action_to_pe, random_action)
    from pool import PoolConfig, ResourcePool

    # Start the pool
    pool_cfg = PoolConfig(
        n_servers=n_servers, per_server=per_server,
        port_base=port_base, boot_timeout=120)
    srv_pool = ResourcePool(pool_cfg, log=lambda m: print(m, flush=True))
    srv_pool.start(wait_healthy=True)

    all_steps: list = []
    all_actions: list = []  # S3.4: executed learner actions (action-mix metric)
    records: list = []      # one per battle: {a1_won, a2_won, crashed, ...}
    lock = threading.Lock()
    next_idx = {"i": 0}
    stop_evt = threading.Event()

    def _worker() -> None:
        while True:
            if stop_evt.is_set() or stop_check():
                stop_evt.set()
                return
            with lock:
                if next_idx["i"] >= n_battles:
                    return
                i = next_idx["i"]
                next_idx["i"] += 1

            if battle_hook:
                battle_hook(i)

            # Pick an opponent from the pool (weighted by ladder weights) —
            # SAME (round, battle) seeding as the old sequential loop, so
            # opponent choice stays deterministic under a seed.
            import random as _rnd
            _rng = _rnd.Random(trainer.round * 1000 + i)
            refs = pool.all()
            weights = [r.weight for r in refs]
            opp_ref = _rng.choices(refs, weights=weights, k=1)[0]
            opp_agent = opp_ref.make_agent(seed=trainer.round * 1000 + i)
            # FRESH agent per battle (fresh RNG per battle; the shared net is
            # read-only here — eval mode, no parameter/state mutation).
            my_agent = PokeAgent(trainer.net, seed=trainer.round * 1000 + i,
                                 device=device)

            # S3.1: which seat the learner plays — seeded 50/50 when the flag
            # is on; default off = always p1 (the historical behaviour). The
            # draw comes from the SAME seeded stream as the opponent choice,
            # so (round, battle) determinism is preserved.
            learner_side = ("p2" if _rng.random() < 0.5 else "p1") \
                if randomize_sides else "p1"
            a1_won = a2_won = None
            crashed = False
            crash_reason = None
            traj: list = []
            actions_log: list = []
            try:
                with srv_pool.battle_slot() as server_cfg:
                    traj, a1_won, a2_won, _n = _play_one_battle(
                        my_agent, opp_agent, fmt, vocabs, device,
                        server_cfg, trainer.round * 1000 + i, max_steps,
                        action_log=actions_log, learner_side=learner_side)
            except Exception as exc:
                crashed = True
                crash_reason = f"{type(exc).__name__}: {exc}"
                print(f"  battle {i}: CRASHED ({type(exc).__name__}: {exc})",
                      flush=True)

            if not crashed:
                rec = {"side": learner_side, "a1_won": a1_won,
                       "a2_won": a2_won}
                if _learner_won(rec):
                    terminal_reward = 1.0
                elif _learner_lost(rec):
                    terminal_reward = -1.0
                else:
                    terminal_reward = -1.0  # D4: draw = -1.0
                if traj:
                    traj[-1].reward = terminal_reward
                    traj[-1].done = True
                    fill_gae(traj, gamma=trainer.cfg.gamma, lam=trainer.cfg.lam)

            with lock:
                records.append({"a1_won": a1_won, "a2_won": a2_won,
                                "opp": opp_ref.name,
                                "side": learner_side,
                                "crashed": crashed,
                                "steps": (0 if crashed else len(traj)),
                                "crash_reason": crash_reason})
                if not crashed:
                    all_steps.extend(traj)
                    all_actions.extend(actions_log)

    n_workers = max(1, min(n_battles, n_servers * per_server))
    print(f"[m1gate] {n_battles} battles x {n_workers} workers "
          f"on {n_servers} server(s) x {per_server} slot(s)", flush=True)
    try:
        threads = [threading.Thread(target=_worker, name=f"m1gate-w{k}",
                                    daemon=True) for k in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        stop_evt.set()
        srv_pool.shutdown()

    # Aggregate LEARNER-RELATIVE (S3.1: records carry the learner's seat;
    # side="p1" — the default — reproduces the historical a1-is-learner
    # mapping exactly).
    n_wins = sum(1 for r in records if _learner_won(r))
    n_losses = sum(1 for r in records if _learner_lost(r))
    n_draws = sum(1 for r in records
                  if not r["crashed"] and not _learner_won(r)
                  and not _learner_lost(r))
    n_crashed = sum(1 for r in records if r["crashed"])
    n_played = n_wins + n_losses + n_draws
    n_maxsteps = sum(1 for r in records
                     if not r["crashed"]
                     and r.get("steps") is not None
                     and r["steps"] >= max_steps)
    maxsteps_share = (n_maxsteps / n_played) if n_played else float("nan")

    # S3.4: W/L/D PER OPPONENT + realised opponent mix (crashes are in
    # "n" but in no W/L/D bucket — the draw rule above already excludes them).
    opp_wld: Dict[str, Dict[str, int]] = {}
    for r in records:
        slot = opp_wld.setdefault(r.get("opp") or "?",
                                  {"w": 0, "l": 0, "d": 0, "n": 0})
        slot["n"] += 1
        if _learner_won(r):
            slot["w"] += 1
        elif _learner_lost(r):
            slot["l"] += 1
        elif not r["crashed"]:
            slot["d"] += 1
    n_rec = len(records)
    opp_mix = {k: (v["n"] / n_rec if n_rec else 0.0) for k, v in opp_wld.items()}
    crash_reasons = [r["crash_reason"] for r in records
                     if r.get("crash_reason")]

    # PPO update
    metrics = {"battles": n_battles, "transitions": len(all_steps),
               "a1_wins": n_wins, "a1_losses": n_losses, "draws": n_draws,
               "crashed": n_crashed, "loss": float("nan"),
               "max_steps_terminations": n_maxsteps,
               "max_steps_share": maxsteps_share}
    metrics["opp_wld"] = opp_wld
    metrics["opp_mix"] = {k: round(v, 4) for k, v in opp_mix.items()}
    metrics["crash_reasons"] = crash_reasons[:5]
    # S3.4: sampled action mix (attack/switch/hazard/setup/status/other)
    if all_actions:
        from net.action_mix import action_mix
        metrics["action_mix"] = {k: round(v, 4)
                                 for k, v in action_mix(
                                     all_actions, _moves_lookup()).items()}
    else:
        metrics["action_mix"] = None
    if len(all_steps) >= 2:
        batch = steps_to_batch(all_steps, device=device)
        n = len(batch["action_index"])
        with torch.no_grad():
            batch["old_logprob"] = chosen_logprobs(trainer.net, batch).clone()
            batch["old_value"] = trainer.net(batch)["value"][:, 0, 0].clone()
            # value_std: spread of the value head's predictions across this
            # batch. Near-0 = the net assigns ~the same value to every state
            # (no win/loss discrimination) — the most insidious degenerate
            # mode, so it gets its own metric + the degenerate guard.
            value_std = (float(batch["old_value"].std().item())
                         if len(batch["old_value"]) > 1 else 0.0)
        ppo = PPOTrainer(trainer.net, lr=trainer.cfg.lr,
                         clip_eps=trainer.cfg.clip_eps,
                         vf_coef=trainer.cfg.vf_coef,
                         ent_coef=trainer.cfg.ent_coef,
                         max_norm=trainer.cfg.max_norm,
                         epochs=trainer.cfg.epochs, device=device)
        advs = np.array([s.advantage for s in all_steps], dtype=np.float32)
        rets = np.array([s.ret for s in all_steps], dtype=np.float32)
        metrics = ppo.update(batch, advantages=advs, returns=rets,
                             epochs=trainer.cfg.epochs)
        metrics["battles"] = n_battles
        metrics["transitions"] = len(all_steps)
        metrics["a1_wins"] = n_wins
        metrics["a1_losses"] = n_losses
        metrics["draws"] = n_draws
        metrics["crashed"] = n_crashed
        metrics["max_steps_terminations"] = n_maxsteps
        metrics["max_steps_share"] = maxsteps_share
        metrics["value_std"] = round(value_std, 5)
        # S3.4: value explained variance (1 - Var(ret - V) / Var(ret))
        rets_f = rets.astype(np.float64)
        vals_f = batch["old_value"].detach().cpu().numpy().astype(np.float64)
        var_ret = float(rets_f.var())
        if var_ret > 1e-12:
            explained_var = 1.0 - float((rets_f - vals_f).var()) / var_ret
        else:
            explained_var = float("nan")
        metrics["value_explained_var"] = round(explained_var, 5)
        # S3.4: log(n_legal) — mean log of the legal-action set size
        n_legal = np.clip(batch["action_mask"].detach().cpu().numpy()
                          .sum(-1), 1, None)
        metrics["log_n_legal"] = round(float(np.log(n_legal).mean()), 5)

    # Checkpoint — the standard self-play format (net.trainer.save_checkpoint):
    # {"net": state_for_save(), "round": int} — resume (main) and load_file
    # (pool snapshots / league) both read exactly this.
    ckpt_path = Path(trainer.cfg.checkpoint_dir or "checkpoints") / \
        f"m1gate_round{trainer.round + 1}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": trainer.net.state_for_save(),
                "round": trainer.round + 1}, str(ckpt_path))
    metrics["checkpoint"] = str(ckpt_path)
    metrics["round"] = trainer.round + 1
    trainer.round += 1
    return metrics


class _Greedy:
    """Deterministic (argmax) wrapper over the CURRENT net for yardstick eval.

    ``_play_one_battle`` calls ``agent.act(obs, acts)`` — which SAMPLES by
    default. The yardstick must be the GREEDY policy (deterministic, the
    gate/eval contract), so this delegates to ``deterministic=True`` without
    touching the learner's stochastic training agent."""

    def __init__(self, agent) -> None:
        self._agent = agent

    def act(self, obs, action_list):
        return self._agent.act(obs, action_list, deterministic=True)


def summarize_eval_records(records: list, max_steps: int = 200) -> Dict[str, Any]:
    """Pure aggregation of battle records into the eval row contract.

    W/L/D are reported SEPARATELY (eval contract; the D4 draw=-1.0 rule is
    training-reward only) and ``eval_wr = wins/(w+l+d)`` (nan if 0 played).
    A record where BOTH sides won (double K.O. / stall) counts as a draw.

    §2.4 observability: ``max_steps_terminations`` counts non-crashed
    records whose ``steps`` reached the loop cap (the battle ended by
    max_steps, not a real result — scored as a draw); ``max_steps_share``
    is that over played battles (nan if none played). Records without a
    ``steps`` key (older rows / test fixtures) are not counted; the default
    200 matches the standard ``max_steps``, so existing call sites are
    unchanged.
    """
    n_wins = sum(1 for r in records if r["a1_won"] and not r["a2_won"])
    n_losses = sum(1 for r in records if r["a2_won"] and not r["a1_won"])
    n_draws = sum(1 for r in records
                  if not r["crashed"]
                  and not (r["a1_won"] and not r["a2_won"])
                  and not (r["a2_won"] and not r["a1_won"]))
    n_crashed = sum(1 for r in records if r["crashed"])
    n_maxsteps = sum(
        1 for r in records
        if not r["crashed"]
        and r.get("steps") is not None and r["steps"] >= max_steps)
    played = n_wins + n_losses + n_draws
    return {
        "eval_w": n_wins,
        "eval_l": n_losses,
        "eval_d": n_draws,
        "eval_crashed": n_crashed,
        "eval_wr": (n_wins / played) if played else float("nan"),
        "max_steps_terminations": n_maxsteps,
        "max_steps_share": (n_maxsteps / played) if played else float("nan"),
    }


def run_eval_battles(net, opp_ref, fmt, vocabs, device,
                     n_battles, max_steps, n_servers, per_server, port_base,
                     seed_base: int, deterministic: bool = True,
                     battle_hook=None, stop_check=None,
                     label: str = "eval",
                     my_ref: Any = None,
                     my_side: str = "p1") -> Dict[str, Any]:
    """The ONE parallel eval-battle loop (shared by the C3 yardstick eval
    and the C4 final gate, ``tools.gate_eval`` — no second implementation).

    * plays ``n_battles`` of ``net`` (p1) vs ONE frozen reference (p2);
      ``deterministic=True`` = greedy (argmax) via :class:`_Greedy`,
      ``False`` = sample from the policy (``PokeAgent.act`` default);
      ``my_ref`` (optional) OVERRIDES the p1 subject with a pool
      ``Reference`` (random-init floor / old heuristic) — the gate
      CONTROLS (Claude review: a gate the random-init net can pass is no
      gate). When set, ``net`` may be ``None``.
      ``my_side`` (optional, default "p1") — which SEAT the subject sits
      on (S3.1): "p2" = the subject plays the p2 seat and the reference
      the p1 seat; the seat-relative result is normalised to subject-POV
      before aggregation, so the ``summarize_eval_records`` contract
      (a1_won = subject win) holds for both seats. The row gains
      ``eval_side``.
    * fresh agent per battle (same isolation contract as training
      collection; the net is read-only — no state mutation);
    * ``seed_base + i`` per battle — the CALLER picks a disjoint seed space
      so parallel evals never re-sample each other's battles;
    * returns the ``summarize_eval_records`` row (pure aggregation).
    """
    import threading
    from net.agent import PokeAgent
    from pool import PoolConfig, ResourcePool

    # Fresh agent per battle; greedy wrapper only for the deterministic mode.
    # my_ref (gate controls) takes precedence: the subject is a pool
    # Reference with the same .act contract, so _Greedy wraps it exactly
    # like a PokeAgent.
    def _make_my_agent(seed: int):
        if my_ref is not None:
            agent = my_ref.make_agent(seed=seed)
        else:
            agent = PokeAgent(net, seed=seed, device=device)
        return _Greedy(agent) if deterministic else agent

    pool_cfg = PoolConfig(
        n_servers=n_servers, per_server=per_server,
        port_base=port_base, boot_timeout=120)
    srv_pool = ResourcePool(pool_cfg, log=lambda m: print(m, flush=True))
    srv_pool.start(wait_healthy=True)

    records: list = []
    errors: list = []
    lock = threading.Lock()
    next_idx = {"i": 0}
    stop_evt = threading.Event()
    # stop_check is OPTIONAL (the gate tool passes None); the trainer always
    # supplies one — the original code called it unguarded, so keep a None
    # guard here (a None default + unguarded call = instant worker crash).
    _stop_check = stop_check if stop_check is not None else (lambda: False)

    def _worker() -> None:
        # A worker that DIES (not a crashed battle) must surface: otherwise
        # every thread could exit silently and the caller sees a quiet
        # all-zero row (the gate smoke caught exactly this: a TypeError in
        # the worker loop left records empty with no error raised).
        try:
            while True:
                if stop_evt.is_set() or _stop_check():
                    stop_evt.set()
                    return
                with lock:
                    if next_idx["i"] >= n_battles:
                        # All battles claimed: tell the other workers to
                        # stop NOW (otherwise a fast worker could grab the
                        # same last index before a slower one notices).
                        stop_evt.set()
                        return
                    i = next_idx["i"]
                    next_idx["i"] += 1

                seed = seed_base + i
                if battle_hook:
                    battle_hook(i)

                a1_won = a2_won = None
                crashed = False
                crash_reason = None
                _traj: list = []
                try:
                    my_agent = _make_my_agent(seed)
                    opp_agent = opp_ref.make_agent(seed=seed)
                    with srv_pool.battle_slot() as server_cfg:
                        _traj, a1_won, a2_won, _n = _play_one_battle(
                            my_agent, opp_agent, fmt, vocabs, device,
                            server_cfg, seed, max_steps,
                            learner_side=my_side)
                except Exception as exc:
                    crashed = True
                    crash_reason = f"{type(exc).__name__}: {exc}"
                    a1_won = a2_won = None
                    print(f"  eval battle {i}: CRASHED "
                          f"({type(exc).__name__}: {exc})", flush=True)

                # S3.1: normalise to SUBJECT-POV — when the subject sat on
                # p2 the a1/a2 seats are swapped relative to it; flip so
                # the summarize_eval_records contract (a1_won = subject
                # win) holds. my_side="p1" (default) = no flip = the
                # historical behaviour.
                if my_side == "p2":
                    a1_won, a2_won = a2_won, a1_won

                with lock:
                    records.append({"a1_won": a1_won, "a2_won": a2_won,
                                    "crashed": crashed,
                                    "steps": (0 if crashed else len(_traj)),
                                    "crash_reason": crash_reason})
        except Exception as exc:  # noqa: BLE001 — record, signal, re-raise
            with lock:
                errors.append(f"worker {threading.current_thread().name}: "
                              f"{type(exc).__name__}: {exc}")
            stop_evt.set()
            raise

    n_workers = max(1, min(n_battles, n_servers * per_server))
    print(f"[{label}] {n_battles} battles vs {getattr(opp_ref, 'name', opp_ref)!r} "
          f"mode={'greedy' if deterministic else 'sampled'} "
          f"({n_workers} workers, {n_servers} server(s) x {per_server})",
          flush=True)
    try:
        threads = [threading.Thread(target=_worker, name=f"{label}-w{k}",
                                    daemon=True) for k in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        stop_evt.set()
        srv_pool.shutdown()

    if errors:
        raise RuntimeError(
            f"{len(errors)} eval worker(s) died: {errors[:3]}")
    row = summarize_eval_records(records, max_steps)
    row["eval_battles"] = n_battles
    row["eval_side"] = my_side
    return row


def _eval_round_m1(trainer, full_pool, fmt, vocabs, device,
                   opp_name, n_battles, max_steps, n_servers, per_server,
                   port_base, battle_hook, stop_check,
                   my_side: str = "p1") -> Dict[str, Any]:
    """C3 yardstick eval: the CURRENT net (GREEDY) vs the frozen reference.

    Thin wrapper over :func:`run_eval_battles` (the shared battle loop):
    greedy mode, seeds in the DISJOINT space ``1_000_000 + round*1000 + i``
    so the eval never re-samples the training battles, additive ``eval_*``
    metric keys only (6.6a discipline), no PPO update, no checkpoint bump.
    """
    if opp_name not in full_pool:
        raise RuntimeError(
            f"eval opponent {opp_name!r} not in the M1 pool "
            f"(have {full_pool.names()})")
    opp_ref = full_pool.get(opp_name)
    row = run_eval_battles(
        trainer.net, opp_ref, fmt, vocabs, device,
        n_battles, max_steps, n_servers, per_server, port_base,
        seed_base=1_000_000 + trainer.round * 1000, deterministic=True,
        battle_hook=battle_hook, stop_check=stop_check, label="m1gate-eval",
        my_side=my_side)
    row["eval_opp"] = opp_name
    return row


def _play_one_battle(agent, opp_agent, fmt, vocabs, device,
                     server_cfg, seed, max_steps,
                     action_log: Optional[list] = None,
                     learner_side: str = "p1"):
    """Play one battle: agent vs opp_agent. Returns (traj, a1_won, a2_won, steps).

    ``learner_side`` (S3.1): which SEAT the learner (``agent``) plays.
    Default "p1" = the historical placement (agent on p1, opp on p2) — the
    loop below is textually identical to the pre-S3.1 one. "p2" = the
    learner sits on p2 and the opponent on p1: the learner consumes its
    own p2-POV observation, and a ``decides_from_battle`` opponent brain
    is told ``side="p1"``. ``a1_won``/``a2_won`` are ALWAYS seat-relative
    (a1 = the p1 seat) — a caller that needs learner-relative W/L
    normalises via the record's ``side`` (see :func:`_learner_won`).

    ``action_log`` (optional): when given, each EXECUTED learner action dict
    is appended (S3.4 action-mix metric). Default None = no logging, so
    existing call sites are unchanged.
    """
    import numpy as np
    import logging
    from net.env import (ShowdownEnv, StrongTeamBuilder, build_action_list,
                         build_obs, action_to_pe)
    from net.ppo import RolloutStep

    team = StrongTeamBuilder(fmt, seed=seed)
    env = ShowdownEnv(
        battle_format=fmt, team=team,
        server_configuration=server_cfg,
        vocabs=vocabs, log_level=logging.ERROR)
    a1, a2 = env.possible_agents
    traj: list = []
    pending = None

    try:
        obs, _info = env.reset()
        # S3.1: seat binding (after reset: both Battle objects exist and
        # stay stable for the battle). learner_side="p1" (default) =>
        # my_seat=a1 / opp_seat=a2 / my_battle=battle1 / opp_side="p2" —
        # textually the historical loop. "p2" mirrors it.
        my_is_p1 = (learner_side == "p1")
        my_seat = a1 if my_is_p1 else a2
        opp_seat = a2 if my_is_p1 else a1
        my_battle = env.battle1 if my_is_p1 else env.battle2
        opp_battle = env.battle2 if my_is_p1 else env.battle1
        opp_side = "p2" if my_is_p1 else "p1"
        for _ in range(max_steps):
            if env.battle1 is not None and env.battle1.finished:
                break
            actions = {a1: np.int64(-2), a2: np.int64(-2)}
            if (env.agent1_to_move if my_is_p1 else env.agent2_to_move):
                obs_dict = obs[my_seat]["observation"]
                acts = build_action_list(my_battle)
                d = agent.act(obs_dict, acts)
                if pending is not None:
                    traj.append(RolloutStep(
                        obs=pending[0], action_index=pending[1],
                        old_logprob=pending[2], old_value=pending[3],
                        reward=0.0, done=False))
                    pending = None
                if d.action is not None:
                    pe = action_to_pe(d.action, my_battle)
                    if pe >= 0:
                        actions[my_seat] = np.int64(pe)
                        if action_log is not None:
                            action_log.append(d.action)
                        pending = (obs_dict, d.action_index,
                                   float(d.logprob), float(d.value))
                else:
                    if pending is not None:
                        traj.append(RolloutStep(
                            obs=pending[0], action_index=pending[1],
                            old_logprob=pending[2], old_value=pending[3],
                            reward=0.0, done=False))
                        pending = None
            if (env.agent2_to_move if my_is_p1 else env.agent1_to_move):
                if getattr(opp_agent, "decides_from_battle", False):
                    act2 = opp_agent.decide(battle=opp_battle, fmt=fmt,
                                            vocabs=vocabs, side=opp_side)
                else:
                    obs2_dict = obs[opp_seat]["observation"]
                    acts2 = build_action_list(opp_battle)
                    d2 = opp_agent.act(obs2_dict, acts2)
                    act2 = d2.action if d2 else None
                if act2 is not None:
                    pe2 = action_to_pe(act2, opp_battle)
                    if pe2 >= 0:
                        actions[opp_seat] = np.int64(pe2)
            obs, rew, term, trunc, _info = env.step(actions)
            if term[a1] or term[a2] or trunc[a1] or trunc[a2]:
                break

        # Flush pending
        if pending is not None:
            traj.append(RolloutStep(
                obs=pending[0], action_index=pending[1],
                old_logprob=pending[2], old_value=pending[3],
                reward=0.0, done=False))

        a1_won = bool(env.battle1.won)
        a2_won = bool(env.battle2.won)
        steps = len(traj)
    finally:
        env.close()

    return traj, a1_won, a2_won, steps


if __name__ == "__main__":
    raise SystemExit(main())
