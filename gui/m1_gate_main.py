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
# Thresholds are tunable here. The point is to WAKE the user to a dead net
# (loud log line + status flag), NOT to kill a possibly-good run — a hard stop
# on a threshold false-positive would cost a whole overnight run.
DEG_ENTROPY_FLOOR = 0.30      # policy entropy below this = near-deterministic
DEG_KL_WARN = 0.50            # per-round KL at/above this = unstable (warn)
DEG_KL_HARD = 1.00            # ...at/above this = severe (warn harder)
DEG_VALUE_STD_FLOOR = 0.02    # value-head std below this = collapsed (no signal)
DEG_NO_PROGRESS_ROUNDS = 5    # 5 straight rounds of 0 training wins
DEG_FLAT_YARDSTICK_EVALS = 3  # 3 straight evals at exactly 0.0 wr


def _degenerate_check(rows: list) -> list:
    """Human-readable degenerate-training warnings for the LATEST round.

    Pure function of the recent ``rows`` (no side effects -> unit-testable
    without torch). Returns ``[]`` when healthy. Rules:

    * **entropy collapse** — latest ``entropy`` < ``DEG_ENTROPY_FLOOR``;
    * **KL blow-up** — latest ``kl_approx`` >= ``DEG_KL_WARN`` (severe at
      ``DEG_KL_HARD``);
    * **value-head collapse** — latest ``value_std`` < ``DEG_VALUE_STD_FLOOR``
      (the net predicts ~the same value for every state = no win/loss signal);
    * **no progress** — ``DEG_NO_PROGRESS_ROUNDS`` straight rounds of 0 wins;
    * **flat yardstick** — ``DEG_FLAT_YARDSTICK_EVALS`` straight evals at 0.0.
    """
    warns: list = []
    if not rows:
        return warns
    last = rows[-1]

    def _finite_num(v: Any) -> bool:
        return v is not None and isinstance(v, (int, float)) \
            and math.isfinite(float(v))

    ent = last.get("entropy")
    if _finite_num(ent) and ent < DEG_ENTROPY_FLOOR:
        warns.append(f"entropy collapse: entropy={ent:.3f} < {DEG_ENTROPY_FLOOR} "
                     f"(policy is near-deterministic)")

    kl = last.get("kl_approx")
    if _finite_num(kl) and kl >= DEG_KL_WARN:
        sev = "KL blow-up" if kl >= DEG_KL_HARD else "KL elevated"
        thr = DEG_KL_HARD if kl >= DEG_KL_HARD else DEG_KL_WARN
        warns.append(f"{sev}: kl_approx={kl:.3f} >= {thr} "
                     f"({'updates are diverging' if kl >= DEG_KL_HARD else 'watch for divergence'})")

    vs = last.get("value_std")
    if _finite_num(vs) and vs < DEG_VALUE_STD_FLOOR:
        warns.append(f"value-head collapse: value_std={vs:.4f} < {DEG_VALUE_STD_FLOOR} "
                     f"(net can't distinguish winning vs losing states)")

    recent = rows[-DEG_NO_PROGRESS_ROUNDS:]
    if len(recent) == DEG_NO_PROGRESS_ROUNDS and all(
            r.get("a1_wins", 0) == 0 and r.get("a1_losses", 0) > 0
            for r in recent):
        warns.append(f"no progress: 0 training wins for "
                     f"{DEG_NO_PROGRESS_ROUNDS} straight rounds")

    evals = [r for r in rows
             if "eval_wr" in r and _finite_num(r.get("eval_wr"))]
    recent_evals = evals[-DEG_FLAT_YARDSTICK_EVALS:]
    if len(recent_evals) == DEG_FLAT_YARDSTICK_EVALS and all(
            r.get("eval_wr") == 0.0 for r in recent_evals):
        warns.append(f"flat yardstick: eval_wr=0.0 for "
                     f"{DEG_FLAT_YARDSTICK_EVALS} straight evals (not learning)")

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

    log(f"start M1 GATE: fmt={fmt} device={device} "
        f"battles/round={battles} rounds={rounds} "
        f"pool=random:{cfg.get('opp_random', True)} "
        f"heuristic:{cfg.get('opp_heuristic', True)} "
        f"strategy:{cfg.get('opp_strategy', True)} "
        f"posterior:{cfg.get('opp_posterior', True)} "
        f"eval={eval_battles}vs{eval_opp} every {eval_every} rounds "
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
                                  port_base, battle_hook, stop_check)
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


def _train_round_m1(trainer, pool, fmt, vocabs, device,
                    n_battles, max_steps, n_servers, per_server, port_base,
                    battle_hook, stop_check) -> Dict[str, Any]:
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
    records: list = []      # one per battle: {a1_won, a2_won, crashed}
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

            a1_won = a2_won = None
            crashed = False
            traj: list = []
            try:
                with srv_pool.battle_slot() as server_cfg:
                    traj, a1_won, a2_won, _n = _play_one_battle(
                        my_agent, opp_agent, fmt, vocabs, device,
                        server_cfg, trainer.round * 1000 + i, max_steps)
            except Exception as exc:
                crashed = True
                print(f"  battle {i}: CRASHED ({type(exc).__name__}: {exc})",
                      flush=True)

            if not crashed:
                if a1_won and not a2_won:
                    terminal_reward = 1.0
                elif a2_won and not a1_won:
                    terminal_reward = -1.0
                else:
                    terminal_reward = -1.0  # D4: draw = -1.0
                if traj:
                    traj[-1].reward = terminal_reward
                    traj[-1].done = True
                    fill_gae(traj, gamma=trainer.cfg.gamma, lam=trainer.cfg.lam)

            with lock:
                records.append({"a1_won": a1_won, "a2_won": a2_won,
                                "crashed": crashed})
                if not crashed:
                    all_steps.extend(traj)

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

    # Aggregate exactly as the sequential loop did (same win/loss/draw rules)
    n_wins = sum(1 for r in records if r["a1_won"] and not r["a2_won"])
    n_losses = sum(1 for r in records if r["a2_won"] and not r["a1_won"])
    n_draws = sum(1 for r in records
                  if not r["crashed"]
                  and not (r["a1_won"] and not r["a2_won"])
                  and not (r["a2_won"] and not r["a1_won"]))
    n_crashed = sum(1 for r in records if r["crashed"])

    # PPO update
    metrics = {"battles": n_battles, "transitions": len(all_steps),
               "a1_wins": n_wins, "a1_losses": n_losses, "draws": n_draws,
               "crashed": n_crashed, "loss": float("nan")}
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
        metrics["value_std"] = round(value_std, 5)

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


def _eval_round_m1(trainer, full_pool, fmt, vocabs, device,
                   opp_name, n_battles, max_steps, n_servers, per_server,
                   port_base, battle_hook, stop_check) -> Dict[str, Any]:
    """C3 yardstick eval: the CURRENT net (greedy) vs the frozen reference.

    Same battle-worker pattern as :func:`_train_round_m1`, but:

    * plays ``n_battles`` GREEDY battles vs ONE fixed opponent (default
      ``strategy``) — the yardstick that tells us whether the net is
      actually learning; no PPO update, no checkpoint bump;
    * seeds are in a DISJOINT space (``1_000_000 + round*1000 + i``) so the
      eval never re-samples the training battles;
    * W/L/D are reported SEPARATELY (eval contract; the D4 draw=-1.0 rule is
      training-reward only) and ``eval_wr = wins/(w+l+d)`` (nan if 0 played);
    * additive metric keys only (``eval_*``) — the existing row contract is
      untouched (6.6a discipline).
    """
    import threading
    from net.agent import PokeAgent
    from pool import PoolConfig, ResourcePool

    if opp_name not in full_pool:
        raise RuntimeError(
            f"eval opponent {opp_name!r} not in the M1 pool "
            f"(have {full_pool.names()})")
    opp_ref = full_pool.get(opp_name)

    # Fresh greedy agent per battle (same isolation contract as training
    # collection; the net is read-only here — no state mutation).
    def _make_my_agent(seed: int) -> "_Greedy":
        return _Greedy(PokeAgent(trainer.net, seed=seed, device=device))

    pool_cfg = PoolConfig(
        n_servers=n_servers, per_server=per_server,
        port_base=port_base, boot_timeout=120)
    srv_pool = ResourcePool(pool_cfg, log=lambda m: print(m, flush=True))
    srv_pool.start(wait_healthy=True)

    records: list = []
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
                    # All battles claimed: tell the other workers to stop
                    # NOW (otherwise a fast worker could grab the same last
                    # index before the slower one notices it is done).
                    stop_evt.set()
                    return
                i = next_idx["i"]
                next_idx["i"] += 1

            seed = 1_000_000 + trainer.round * 1000 + i
            if battle_hook:
                battle_hook(i)

            my_agent = _make_my_agent(seed)
            opp_agent = opp_ref.make_agent(seed=seed)
            a1_won = a2_won = None
            crashed = False
            try:
                with srv_pool.battle_slot() as server_cfg:
                    _traj, a1_won, a2_won, _n = _play_one_battle(
                        my_agent, opp_agent, fmt, vocabs, device,
                        server_cfg, seed, max_steps)
            except Exception as exc:
                crashed = True
                print(f"  eval battle {i}: CRASHED "
                      f"({type(exc).__name__}: {exc})", flush=True)

            with lock:
                records.append({"a1_won": a1_won, "a2_won": a2_won,
                                "crashed": crashed})

    n_workers = max(1, min(n_battles, n_servers * per_server))
    print(f"[m1gate-eval] {n_battles} battles vs {opp_name!r} "
          f"({n_workers} workers, {n_servers} server(s) x {per_server})",
          flush=True)
    try:
        threads = [threading.Thread(target=_worker, name=f"m1gate-eval-w{k}",
                                    daemon=True) for k in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        stop_evt.set()
        srv_pool.shutdown()

    n_wins = sum(1 for r in records if r["a1_won"] and not r["a2_won"])
    n_losses = sum(1 for r in records if r["a2_won"] and not r["a1_won"])
    n_draws = sum(1 for r in records
                  if not r["crashed"]
                  and not (r["a1_won"] and not r["a2_won"])
                  and not (r["a2_won"] and not r["a1_won"]))
    n_crashed = sum(1 for r in records if r["crashed"])

    played = n_wins + n_losses + n_draws
    return {
        "eval_opp": opp_name,
        "eval_battles": n_battles,
        "eval_w": n_wins,
        "eval_l": n_losses,
        "eval_d": n_draws,
        "eval_crashed": n_crashed,
        "eval_wr": (n_wins / played) if played else float("nan"),
    }


def _play_one_battle(agent, opp_agent, fmt, vocabs, device,
                     server_cfg, seed, max_steps):
    """Play one battle: agent (p1) vs opp_agent (p2). Returns (traj, a1_won, a2_won, steps)."""
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
        for _ in range(max_steps):
            if env.battle1 is not None and env.battle1.finished:
                break
            actions = {a1: np.int64(-2), a2: np.int64(-2)}
            if env.agent1_to_move:
                obs_dict = obs[a1]["observation"]
                acts = build_action_list(env.battle1)
                d = agent.act(obs_dict, acts)
                if pending is not None:
                    traj.append(RolloutStep(
                        obs=pending[0], action_index=pending[1],
                        old_logprob=pending[2], old_value=pending[3],
                        reward=0.0, done=False))
                    pending = None
                if d.action is not None:
                    pe = action_to_pe(d.action, env.battle1)
                    if pe >= 0:
                        actions[a1] = np.int64(pe)
                        pending = (obs_dict, d.action_index,
                                   float(d.logprob), float(d.value))
                else:
                    if pending is not None:
                        traj.append(RolloutStep(
                            obs=pending[0], action_index=pending[1],
                            old_logprob=pending[2], old_value=pending[3],
                            reward=0.0, done=False))
                        pending = None
            if env.agent2_to_move:
                if getattr(opp_agent, "decides_from_battle", False):
                    act2 = opp_agent.decide(battle=env.battle2, fmt=fmt,
                                            vocabs=vocabs, side="p2")
                else:
                    obs2_dict = obs[a2]["observation"]
                    acts2 = build_action_list(env.battle2)
                    d2 = opp_agent.act(obs2_dict, acts2)
                    act2 = d2.action if d2 else None
                if act2 is not None:
                    pe2 = action_to_pe(act2, env.battle2)
                    if pe2 >= 0:
                        actions[a2] = np.int64(pe2)
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
