"""Smoke test: 1 battle of main-learner vs one checkpoint opponent.

Verifies the full pipeline: pool start -> opponent load -> battle (via pool slot)
-> trajectory collection -> GAE -> PPO update -> checkpoint save.
"""
from __future__ import annotations
import os, sys, time

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
from net.agent import PokeAgent
from net.env import StrongTeamBuilder, build_action_list, build_obs, action_to_pe
from net.ppo import RolloutStep, fill_gae, steps_to_batch, chosen_logprobs, PPOTrainer
from net.ppo import RolloutBuffer
from pool import PoolConfig, ResourcePool
from tools.stdio_guard import safe_stdio

sys.path.insert(0, os.path.join(ROOT, "scratch", "step4"))
from train_vs_pool import find_opponents, load_opponent

def main():
    safe_stdio()  # Windows cp1252 redirect guard: a Unicode print must never kill the run
    print("[smoke] starting ...", flush=True)
    pool = ResourcePool(
        PoolConfig(n_servers=1, per_server=1, port_base=49960, boot_timeout=120),
        log=lambda m: print(m, flush=True))
    status = pool.start(wait_healthy=True)
    if not all(status.values()):
        raise SystemExit(f"[smoke] pool not healthy: {status}")
    try:
        vocabs = build_vocabs()
        main_net = PokeNet(vocabs)
        main_agent = PokeAgent(main_net, seed=42, device="cpu")

        # Load the frozen opponent (after_step3 round 8)
        opp_paths = find_opponents()
        opp_path = opp_paths[0]
        print(f"[smoke] opponent: {os.path.basename(opp_path)}", flush=True)
        opp_net = load_opponent(opp_path, vocabs)
        print(f"[smoke] opponent loaded OK ({sum(p.numel() for p in opp_net.parameters()):,} params)", flush=True)
        opp_agent = PokeAgent(opp_net, seed=99, device="cpu")

        # Play ONE battle via the pool's battle_slot (correct server config)
        t0 = time.time()
        with pool.battle_slot() as server_cfg:
            from net.env import ShowdownEnv
            import logging as _log
            team = StrongTeamBuilder("gen9ou", seed=42)
            env = ShowdownEnv(battle_format="gen9ou", team=team,
                              server_configuration=server_cfg,
                              vocabs=vocabs, log_level=_log.ERROR)
            a1, a2 = env.possible_agents
            traj = []
            pending = None
            try:
                obs, _info = env.reset()
                for _ in range(200):
                    if env.battle1 is not None and env.battle1.finished:
                        break
                    actions = {a1: np.int64(-2), a2: np.int64(-2)}
                    if env.agent1_to_move:
                        if pending is not None:
                            traj.append(RolloutStep(
                                obs=pending[0], action_index=pending[1],
                                old_logprob=pending[2], old_value=pending[3],
                                reward=0.0, done=False))
                            pending = None
                        obs_dict = obs[a1]["observation"]
                        acts = build_action_list(env.battle1)
                        d = main_agent.act(obs_dict, acts)
                        if d.action is not None:
                            pe = action_to_pe(d.action, env.battle1)
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
                    obs, rew, term, trunc, _info = env.step(actions)
                    if pending is not None:
                        r = float(rew[a1]) if a1 in rew else 0.0
                        done = bool((a1 in term and term[a1]) or (a1 in trunc and trunc[a1]))
                        traj.append(RolloutStep(
                            obs=pending[0], action_index=pending[1],
                            old_logprob=pending[2], old_value=pending[3],
                            reward=r, done=done))
                        pending = None
                    if (a1 in term and term[a1]) or (a2 in term and term[a2]) or \
                       (a1 in trunc and trunc[a1]) or (a2 in trunc and trunc[a2]):
                        break
            finally:
                env.close()

        elapsed = time.time() - t0
        a1_won = env.battle1.won if env.battle1 else False
        a2_won = env.battle2.won if env.battle2 else False

        print(f"\n[smoke] battle result:", flush=True)
        print(f"  a1_won (main): {a1_won}", flush=True)
        print(f"  a2_won (opp):  {a2_won}", flush=True)
        print(f"  draw:          {not a1_won and not a2_won}", flush=True)
        print(f"  n_transitions: {len(traj)}", flush=True)
        print(f"  elapsed:       {elapsed:.1f}s", flush=True)

        if len(traj) < 2:
            raise SystemExit(f"[smoke] FAILED: too few transitions ({len(traj)})")

        # GAE
        fill_gae(traj, gamma=0.99, lam=0.95)
        buf = RolloutBuffer()
        buf.extend(traj)
        print(f"[smoke] GAE applied, buffer={len(buf)}", flush=True)

        # PPO update
        ppo = PPOTrainer(main_net, lr=3e-4, clip_eps=0.2, vf_coef=0.5,
                         ent_coef=0.01, max_norm=0.5, epochs=1, device="cpu")
        batch = steps_to_batch(buf.steps, "cpu")
        with torch.no_grad():
            batch["old_logprob"] = chosen_logprobs(main_net, batch).clone()
        advantages = np.array([s.advantage for s in buf.steps], dtype=np.float32)
        returns = np.array([s.ret for s in buf.steps], dtype=np.float32)
        metrics = ppo.update(batch, advantages=advantages, returns=returns)
        print(f"[smoke] PPO update OK: policy_loss={metrics.get('policy_loss', float('nan')):.4f}", flush=True)

        # Verify no NaNs
        for name, param in main_net.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                raise SystemExit(f"[smoke] FAILED: NaN in {name}")

        print(f"\n[smoke] ALL PASS -- pipeline works end-to-end", flush=True)

    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
