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

    log(f"start M1 GATE: fmt={fmt} device={device} "
        f"battles/round={battles} rounds={rounds} "
        f"pool=random:{cfg.get('opp_random', True)} "
        f"heuristic:{cfg.get('opp_heuristic', True)} "
        f"strategy:{cfg.get('opp_strategy', True)} "
        f"posterior:{cfg.get('opp_posterior', True)} "
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
    try:
        for _ in range(rounds):
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
            runstate.append_metric(rd, row)
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
    """One PPO round: play n_battles against the M1 pool, GAE, update."""
    import numpy as np
    from net.ppo import fill_gae, steps_to_batch, chosen_logprobs, PPOTrainer
    from net.ppo import RolloutStep, RolloutBuffer
    from net.env import (ShowdownEnv, StrongTeamBuilder, build_action_list,
                         build_obs, action_to_pe, random_action)
    from pool import PoolConfig, ResourcePool
    import random as _random

    # Start the pool
    pool_cfg = PoolConfig(
        n_servers=n_servers, per_server=per_server,
        port_base=port_base, boot_timeout=120)
    srv_pool = ResourcePool(pool_cfg, log=lambda m: print(m, flush=True))
    srv_pool.start(wait_healthy=True)

    all_steps: list = []
    n_wins = n_losses = n_draws = 0
    n_crashed = 0

    try:
        for i in range(n_battles):
            if stop_check():
                break
            if battle_hook:
                battle_hook(i)

            # Pick a random opponent from the pool
            opp_ref = pool.sample(seed=trainer.round * 1000 + i)
            opp_agent = opp_ref.make_agent(seed=trainer.round * 1000 + i)

            with srv_pool.battle_slot() as server_cfg:
                try:
                    traj, a1_won, a2_won, n_steps = _play_one_battle(
                        trainer.agent, opp_agent, fmt, vocabs, device,
                        server_cfg, trainer.round * 1000 + i, max_steps)
                except Exception as exc:
                    n_crashed += 1
                    print(f"  battle {i}: CRASHED ({type(exc).__name__}: {exc})",
                          flush=True)
                    continue

            if a1_won and not a2_won:
                n_wins += 1
                terminal_reward = 1.0
            elif a2_won and not a1_won:
                n_losses += 1
                terminal_reward = -1.0
            else:
                n_draws += 1
                terminal_reward = -1.0  # D4: draw = -1.0

            if traj:
                traj[-1].reward = terminal_reward
                traj[-1].done = True
                fill_gae(traj, gamma=trainer.cfg.gamma, lam=trainer.cfg.lam)
                all_steps.extend(traj)
    finally:
        srv_pool.shutdown()

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

    # Checkpoint
    ckpt_path = Path(trainer.cfg.checkpoint_dir or "checkpoints") / \
        f"m1gate_round{trainer.round + 1}.pt"
    PokeNet.save(trainer.net, str(ckpt_path), round=trainer.round + 1)
    metrics["checkpoint"] = str(ckpt_path)
    metrics["round"] = trainer.round + 1
    trainer.round += 1
    return metrics


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
