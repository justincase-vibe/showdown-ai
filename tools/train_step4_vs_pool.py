"""Step 4 (Fork 2) — single-learner-first experiment at scale.

The main PPO learner trains against a POOL of frozen opponent checkpoints
(the 8 league random bots) instead of pure random. This is the fair test:
the before-net also trained against the league, so the main learner's
improvement is measured against the same regime.

Design:
  * ONE main learner (fresh random init or resumed from after_step3)
  * Opponents: the 8 ``scratch/league_random_*.pt`` checkpoints (cycled)
  * PPO: on-policy, 1-2 epochs per round (same as the solo trainer)
  * Watchdog: per-battle timeout (a hung battle is killed + skipped)
  * CPU only (the GPU is the LLM's)
  * Checkpoints saved per round in ``scratch/step4/``

Run:
    cd "showdown AI-imitation" && CUDA_VISIBLE_DEVICES=-1 python scratch/step4/train_vs_pool.py
"""
from __future__ import annotations
import os, sys, time, glob

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
import torch

from net.vocab_db import build_vocabs
from net.model import PokeNet
from net.trainer import TrainConfig, SelfPlayTrainer
from net.agent import PokeAgent
from net.env import (ShowdownEnv, StrongTeamBuilder, build_action_list,
                     build_obs, action_to_pe)
from net.ppo import RolloutStep, fill_gae, steps_to_batch, chosen_logprobs, PPOTrainer
from net.ppo import RolloutBuffer
from pool import PoolConfig, ResourcePool

CKPT_DIR = os.path.join(ROOT, "scratch", "step4")
N_BATTLES_PER_ROUND = 20
N_ROUNDS = 8              # 160 battles total (same budget as after_step3)
SEED = 20261123
MAX_STEPS = 300
BATTLE_TIMEOUT = 300      # per-battle watchdog (s)

# ── find the opponent checkpoints ──────────────────────────────────────────────
def find_opponents() -> list[str]:
    """The frozen opponent pool.

    The old league_random_*.pt are pre-Step-2 (GRU) and cannot be loaded.
    The available new-layout checkpoint is the after_step3 round-8 net.
    We use it as the primary frozen opponent. For diversity, the main learner
    also faces pure random (50/50 mix) — this matches the "frozen snapshots
    + scripted bots" pool the handout describes.
    """
    after = os.path.join(ROOT, "scratch", "after_step3", "selfplay_round8.pt")
    if not os.path.exists(after):
        # Fallback: any checkpoint in after_step3/
        cks = sorted(glob.glob(os.path.join(ROOT, "scratch", "after_step3", "selfplay_round*.pt")))
        if cks:
            after = cks[-1]
        else:
            raise SystemExit(f"[step4] no after_step3 checkpoint found")
    pool = [after]
    print(f"[step4] frozen opponent pool ({len(pool)}):", flush=True)
    for f in pool:
        print(f"  {os.path.basename(f)}", flush=True)
    print(f"[step4] + random (50/50 mix for diversity)", flush=True)
    return pool


def load_opponent(path: str, vocabs) -> PokeNet:
    """Load one opponent checkpoint (pristine — never modified)."""
    return PokeNet.load_file(path, vocabs, device="cpu")


# ── the core: play one battle, main learner vs a checkpoint opponent ──────────
def play_vs_checkpoint(main_agent: PokeAgent, opp_agent: PokeAgent,
                       fmt: str, seed: int, max_steps: int = 300,
                       server_configuration=None, vocabs=None) -> dict:
    """Play ONE battle (main on p1, checkpoint opponent on p2).

    ``server_configuration`` is REQUIRED (the pool's battle_slot yields the
    correct port). Collects the MAIN agent's trajectory with sparse terminal
    reward. Returns the same shape as ``play_and_collect``.
    """
    import logging as _log
    team = StrongTeamBuilder(fmt, seed=seed)
    env = ShowdownEnv(battle_format=fmt, team=team,
                      server_configuration=server_configuration,
                      vocabs=vocabs, log_level=_log.ERROR)
    a1, a2 = env.possible_agents
    out: dict = {"traj": [], "crashed": False, "error": "", "n_illegal": 0,
                 "a1_won": None, "a2_won": None, "steps": 0, "draw": False}
    pending = None
    try:
        obs, _info = env.reset()
        for _ in range(max_steps):
            if env.battle1 is not None and env.battle1.finished:
                break
            actions = {a1: np.int64(-2), a2: np.int64(-2)}
            if env.agent1_to_move:
                if pending is not None:
                    out["traj"].append(RolloutStep(
                        obs=pending[0], action_index=pending[1],
                        old_logprob=pending[2], old_value=pending[3],
                        reward=0.0, done=False))
                    pending = None
                obs_dict = obs[a1]["observation"]
                acts = build_action_list(env.battle1)
                d = main_agent.act(obs_dict, acts)
                if d.action is None:
                    actions[a1] = np.int64(-2)
                else:
                    pe = action_to_pe(d.action, env.battle1)
                    if pe < 0:
                        out["n_illegal"] += 1
                    actions[a1] = np.int64(pe)
                    pending = (obs_dict, d.action_index, float(d.logprob), float(d.value))
            if env.agent2_to_move:
                obs2 = obs.get(a2, {}).get("observation") if isinstance(obs, dict) else None
                if obs2 is not None:
                    acts2 = build_action_list(env.battle2)
                    d2 = opp_agent.act(obs2, acts2)
                    if d2 is not None and d2.action is not None:
                        pe2 = action_to_pe(d2.action, env.battle2)
                        actions[a2] = np.int64(pe2) if pe2 >= 0 else np.int64(-2)
                if actions[a2].item() == -2 and env.battle2 is not None:
                    from net.env import random_action
                    act2 = random_action(env.battle2)
                    actions[a2] = np.int64(action_to_pe(act2, env.battle2) if act2 is not None else -2)

            obs, rew, term, trunc, _info = env.step(actions)
            out["steps"] += 1
            if pending is not None:
                r = float(rew[a1]) if a1 in rew else 0.0
                done = bool((a1 in term and term[a1]) or (a1 in trunc and trunc[a1]))
                out["traj"].append(RolloutStep(
                    obs=pending[0], action_index=pending[1],
                    old_logprob=pending[2], old_value=pending[3],
                    reward=r, done=done))
                pending = None
            if (a1 in term and term[a1]) or (a2 in term and term[a2]) or \
               (a1 in trunc and trunc[a1]) or (a2 in trunc and trunc[a2]):
                break
        out["a1_won"] = bool(env.battle1.won) if env.battle1 else False
        out["a2_won"] = bool(env.battle2.won) if env.battle2 else False
        out["draw"] = (not out["a1_won"]) and (not out["a2_won"])
    except Exception as exc:
        out["crashed"] = True
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    return out


def play_with_watchdog(fn, *args, timeout_s: int = 300, **kwargs) -> dict:
    """Run ``fn(*args, **kwargs)`` under a watchdog thread (same pattern as the trainer)."""
    if timeout_s <= 0:
        return fn(*args, **kwargs)
    import threading
    result: dict = {}
    def run():
        try:
            result.update(fn(*args, **kwargs))
        except Exception as exc:
            result.update({"crashed": True, "error": f"{type(exc).__name__}: {exc}"})
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return {"crashed": True, "timed_out": True, "traj": [], "steps": 0,
                "error": f"battle TIMEOUT after {timeout_s}s"}
    return result


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Start the pool
    pool = ResourcePool(
        PoolConfig(n_servers=1, per_server=2, port_base=8000, boot_timeout=120),
        log=lambda m: print(m, flush=True))
    print("[step4] starting server pool ...", flush=True)
    status = pool.start(wait_healthy=True)
    if not all(status.values()):
        raise SystemExit(f"[step4] pool not healthy: {status}")

    try:
        vocabs = build_vocabs()

        # Main learner (fresh random init — the "after-step3" checkpoint was
        # trained vs random; this is the next step: vs the pool).
        main_net = PokeNet(vocabs)
        main_agent = PokeAgent(main_net, seed=SEED, device="cpu")

        # Opponent pool (frozen snapshots — never modified)
        opp_paths = find_opponents()
        opponents = [load_opponent(p, vocabs) for p in opp_paths]
        n_opp = len(opponents)
        print(f"[step4] main learner ready, {n_opp} opponents loaded", flush=True)

        # PPO config (same as the solo trainer)
        cfg = TrainConfig(
            battle_format="gen9ou", battles_per_round=N_BATTLES_PER_ROUND,
            max_steps=MAX_STEPS, seed=SEED, device="cpu",
            epochs=2, checkpoint_dir=CKPT_DIR, save_every=1, n_rounds=N_ROUNDS,
        )

        # PPO internals (reuse the trainer's PPO + GAE)
        ppo = PPOTrainer(main_net, lr=cfg.lr, clip_eps=cfg.clip_eps,
                         vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                         max_norm=cfg.max_norm, epochs=cfg.epochs, device="cpu")
        buffer = RolloutBuffer()

        print(f"\n{'round':>5} {'battles':>7} {'trans':>6} {'crash':>5} {'tmo':>4} "
              f"{'win':>4} {'loss':>5} {'illeg':>5} {'policy':>8} "
              f"{'value':>8} {'entrop':>8} {'clip':>7}", flush=True)

        for rnd in range(N_ROUNDS):
            buffer.clear()
            n_battles = 0
            n_crash = 0
            n_timeout = 0
            n_wins = 0
            n_losses = 0
            n_illegal = 0
            n_trans = 0
            round_t0 = time.time()

            for i in range(N_BATTLES_PER_ROUND):
                seed = SEED + rnd * 1000 + i
                # 50/50: frozen checkpoint opponent vs random
                use_checkpoint = (i % 2 == 0)
                if use_checkpoint and n_opp > 0:
                    opp_idx = (rnd * N_BATTLES_PER_ROUND + i) % n_opp
                    opp_net = opponents[opp_idx]
                    opp_agent = PokeAgent(opp_net, seed=seed + 7000, device="cpu")
                    with pool.battle_slot() as server_cfg:
                        r = play_with_watchdog(
                            play_vs_checkpoint, main_agent, opp_agent,
                            "gen9ou", seed, max_steps=MAX_STEPS,
                            server_configuration=server_cfg, vocabs=vocabs,
                            timeout_s=BATTLE_TIMEOUT)
                else:
                    # random opponent (the old play_and_collect path)
                    from net.trainer import play_and_collect
                    with pool.battle_slot() as server_cfg:
                        r = play_with_watchdog(
                            play_and_collect, main_agent, "gen9ou", seed,
                            max_steps=MAX_STEPS,
                            server_configuration=server_cfg,
                            timeout_s=BATTLE_TIMEOUT)

                if r.get("timed_out"):
                    n_timeout += 1
                    n_crash += 1
                    print(f"    [r{rnd+1} b{i+1}] TIMEOUT ({BATTLE_TIMEOUT}s) — skipped", flush=True)
                    continue
                if r["crashed"]:
                    n_crash += 1
                    print(f"    [r{rnd+1} b{i+1}] CRASH: {r['error'][:60]}", flush=True)
                    continue

                # GAE + buffer
                traj = r["traj"]
                if traj:
                    fill_gae(traj, gamma=0.99, lam=0.95)
                    buffer.extend(traj)
                    n_trans += len(traj)
                n_battles += 1
                if r.get("a1_won"):
                    n_wins += 1
                elif r.get("a2_won"):
                    n_losses += 1
                n_illegal += r.get("n_illegal", 0)

            # PPO update
            if len(buffer) >= 2:
                batch = steps_to_batch(buffer.steps, "cpu")
                with torch.no_grad():
                    batch["old_logprob"] = chosen_logprobs(main_net, batch).clone()
                advantages = np.array([s.advantage for s in buffer.steps], dtype=np.float32)
                returns = np.array([s.ret for s in buffer.steps], dtype=np.float32)
                metrics = ppo.update(batch, advantages=advantages, returns=returns)
            else:
                metrics = {"skipped": True, "reason": f"too_few ({len(buffer)})"}
            buffer.clear()

            # Checkpoint
            ckpt_path = os.path.join(CKPT_DIR, f"step4_round{rnd+1}.pt")
            torch.save({
                "net": main_net.state_for_save(),
                "config": cfg.to_dict(),
                "round": rnd + 1,
                "opponent_pool": [os.path.basename(p) for p in opp_paths],
            }, ckpt_path)

            # Print
            elapsed = time.time() - round_t0
            wr = n_wins / max(1, n_battles)
            p = metrics.get("policy_loss", float('nan'))
            v = metrics.get("value_loss", float('nan'))
            e = metrics.get("entropy", float('nan'))
            c = metrics.get("frac_clipped", float('nan'))
            print(f"{rnd+1:>5} {n_battles:>7} {n_trans:>6} {n_crash:>5} {n_timeout:>4} "
                  f"{wr:>4.0%} {n_losses:>5} {n_illegal:>5} {p:>8.4f} "
                  f"{v:>8.4f} {e:>8.4f} {c:>7.3f}", flush=True)

        total = time.time() - t0
        print(f"\n[step4] DONE in {total:.0f}s — checkpoints in {CKPT_DIR}", flush=True)
        print(f"[step4] best checkpoint: {CKPT_DIR}/step4_round{N_ROUNDS}.pt", flush=True)

    finally:
        pool.shutdown()
        print("[step4] pool shut down", flush=True)


if __name__ == "__main__":
    main()
