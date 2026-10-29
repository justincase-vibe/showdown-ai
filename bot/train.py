"""Self-play training harness for the Pokemon Showdown RL bot.

Runs N concurrent battles across a *mix of formats* (e.g. gen7ou + gen7randombattle),
collects transitions into a PPO buffer, updates the shared policy on the chosen device,
and tracks rolling win-rate per format. The opponent is pluggable:

  * ``random`` -- a random-legal-move baseline (sanity check / curriculum floor).
  * ``frozen`` -- a periodically-updated copy of the trainee (default self-play; a loss
                  on one side is a win on the other, so the target is always ~matched).
  * ``live``   -- the trainee itself (both sides always current).

Everything here is device-agnostic. Run the CPU smoke test first (no GPU needed):

    python bot/train.py --smoke

Real GPU training (start only when the GPU is free):

    python bot/train.py --device cuda --num-envs 6 --buffer-size 4096 --max-battles 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make sibling modules importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import ShowdownEnv, TraineeEnv, UNIFIED_ACTION_SIZE  # noqa: E402
from obs import OBS_DIM  # noqa: E402
from policy import MaskedPPO  # noqa: E402
from teams import RandomTeamBuilder  # noqa: E402


# --------------------------------------------------------------------------- #
# Env pool construction
# --------------------------------------------------------------------------- #
def apportion(total: int, weights: List[float]) -> List[int]:
    """Distribute ``total`` items across buckets by (proportional) weight."""
    s = sum(weights) or 1.0
    raw = [total * w / s for w in weights]
    counts = [int(r) for r in raw]
    rem = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(rem):
        counts[order[i % len(order)]] += 1
    return counts


def build_envs(
    formats: List[str],
    weights: List[float],
    num_envs: int,
    seed: int,
    save_replays: Optional[str],
) -> List[TraineeEnv]:
    counts = apportion(num_envs, weights)
    envs: List[TraineeEnv] = []
    idx = 0
    for fmt, c in zip(formats, counts):
        for _ in range(c):
            team = None if "randombattle" in fmt else RandomTeamBuilder(
                battle_format=fmt, seed=seed + idx
            )
            env = ShowdownEnv(
                battle_format=fmt,
                team=team,
                strict=False,
                save_replays=save_replays,
                log_level=logging.WARNING,
            )
            envs.append(TraineeEnv(env, env_id=idx, fmt=fmt))
            idx += 1
    return envs


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
def run(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("train")

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != len(formats):
        weights = [1.0 / len(formats)] * len(formats)

    log.info(
        "formats=%s weights=%s num_envs=%d opponent=%s buffer=%d max_battles=%d device=%s",
        formats, weights, args.num_envs, args.opponent, args.buffer_size, args.max_battles, args.device,
    )

    envs = build_envs(formats, weights, args.num_envs, args.seed, args.replays)
    fmt_of = {e.id: e.fmt for e in envs}

    policy = MaskedPPO(
        obs_dim=OBS_DIM,
        n_actions=UNIFIED_ACTION_SIZE,
        hidden=tuple(int(h) for h in args.hidden.split(",")),
        lr=args.lr,
        epochs=args.ppo_epochs,
        minibatch=args.minibatch,
        device=args.device if args.device != "auto" else None,
        seed=args.seed,
    )
    log.info("policy device=%s | params=%d", policy.device,
             sum(p.numel() for p in policy.actor.parameters()))

    opponent: object = "random"
    if args.opponent == "frozen":
        opponent = MaskedPPO(
            obs_dim=OBS_DIM, n_actions=UNIFIED_ACTION_SIZE,
            hidden=tuple(int(h) for h in args.hidden.split(",")),
            device=policy.device, seed=args.seed + 1,
        )
        policy.sync_opponent(opponent)  # start matched (random-init both)
    elif args.opponent == "live":
        opponent = policy

    os.makedirs(args.ckpt_dir, exist_ok=True)
    stats_path = os.path.join(args.ckpt_dir, "stats.jsonl")

    wins: Dict[str, int] = {}
    losses: Dict[str, int] = {}
    battles_done = 0
    transitions = 0
    start = time.time()

    # reset all envs
    cur_obs: Dict[int, np.ndarray] = {}
    cur_mask: Dict[int, np.ndarray] = {}
    for e in envs:
        cur_obs[e.id] = e.reset()
        cur_mask[e.id] = e.last_mask

    def buffer_size() -> int:
        # count transitions currently queued in the policy buffer (approx via episodes)
        n = 0
        for ep in policy._episodes.values():
            n += len(ep)
        if policy._buffer is not None:
            n += len(policy._buffer["act"])
        return n

    def log_stats(tag: str, extra: dict = None):
        rec = {
            "t": int(time.time() - start),
            "battles": battles_done,
            "wins": dict(wins),
            "losses": dict(losses),
            "winrate": {k: wins[k] / (wins[k] + losses[k]) if (wins[k] + losses[k]) else None
                        for k in wins},
            "tag": tag,
        }
        if extra:
            rec.update(extra)
        line = json.dumps(rec)
        with open(stats_path, "a") as fh:
            fh.write(line + "\n")
        log.info("%s", line)

    try:
        while battles_done < args.max_battles:
            # --- collect a batch of transitions across envs ---
            for e in envs:
                if e.done:
                    # re-arm finished envs as they come up
                    cur_obs[e.id] = e.reset()
                    cur_mask[e.id] = e.last_mask
                obs = cur_obs[e.id]
                mask = cur_mask[e.id]
                if int(mask.sum()) == 0:
                    # Showdown 'wait' request: nothing to decide this turn.
                    # Advance the battle (TraineeEnv sends the default -2 order)
                    # and record the step WITHOUT a policy decision.
                    new_obs, r, done = e.step(0, opponent)
                    policy.store_pass(e.id, obs, r, policy.value(obs), done)
                else:
                    a, lp, v = policy.act(obs, mask)
                    assert 0 <= a < UNIFIED_ACTION_SIZE and mask[a] == 1, f"illegal action {a}"
                    new_obs, r, done = e.step(a, opponent)
                    policy.store(e.id, obs, a, lp, r, done, mask, v)
                transitions += 1
                if done:
                    battles_done += 1
                    key = fmt_of[e.id]
                    if r > 0:
                        wins[key] = wins.get(key, 0) + 1
                    elif r < 0:
                        losses[key] = losses.get(key, 0) + 1
                    else:
                        pass  # draw / double-kO
                    # re-arm immediately
                    cur_obs[e.id] = e.reset()
                    cur_mask[e.id] = e.last_mask
                    if battles_done % args.log_every == 0:
                        log_stats("battles")
                    if buffer_size() >= args.buffer_size:
                        break

            # --- update policy when buffer is full ---
            if buffer_size() >= args.buffer_size:
                t0 = time.time()
                upstats = policy.update()
                if args.opponent == "frozen" and policy.n_updates % args.sync_every == 0:
                    policy.sync_opponent(opponent)
                ck = os.path.join(args.ckpt_dir, "ppo.pt")
                policy.save(ck)
                log_stats("update", {
                    "update": policy.n_updates,
                    "dt": round(time.time() - t0, 3),
                    "upstats": {k: round(v, 5) for k, v in (upstats or {}).items()},
                })
    except KeyboardInterrupt:
        log.info("interrupted -- saving checkpoint")
        policy.save(os.path.join(args.ckpt_dir, "ppo.pt"))
        log_stats("interrupt")
    finally:
        for e in envs:
            e.close()
        log.info("done. battles=%d elapsed=%.1fs", battles_done, time.time() - start)


# --------------------------------------------------------------------------- #
# Smoke test: prove the whole pipeline works on CPU, no heavy training
# --------------------------------------------------------------------------- #
def smoke(args) -> None:
    """Prove the full pipeline works on CPU with no heavy training: env + teams +
    masked policy + one PPO update, across an OU format and a random-battle format."""
    logging.basicConfig(level=logging.WARNING)
    print("=== SMOKE TEST (CPU) ===")
    print(f"OBS_DIM={OBS_DIM}  ACTIONS={UNIFIED_ACTION_SIZE}")

    policy = MaskedPPO(obs_dim=OBS_DIM, n_actions=UNIFIED_ACTION_SIZE,
                       hidden=(64, 64), device="cpu", seed=0)

    # --- OU format: teams come from our pool ---
    env = ShowdownEnv(
        battle_format="gen7ou",
        team=RandomTeamBuilder(battle_format="gen7ou", seed=0),
        strict=False,
        save_replays=args.replays,
        log_level=logging.WARNING,
    )
    te = TraineeEnv(env, 0, "gen7ou")
    obs = te.reset()
    assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape} != ({OBS_DIM},)"
    assert te.last_mask.shape == (UNIFIED_ACTION_SIZE,), "mask shape wrong"
    assert np.any(te.last_mask), "action mask is all-zero!"
    print(f"[gen7ou reset] obs={obs.shape} | legal_actions={int(te.last_mask.sum())} "
          f"| finite={bool(np.all(np.isfinite(obs)))}")

    cur_obs, cur_mask = obs, te.last_mask
    steps = 0
    total_reward = 0.0
    done = False
    wait_steps = 0
    while not done and steps < 200:
        if int(cur_mask.sum()) == 0:
            # Showdown 'wait' request: nothing to decide; just advance the battle.
            new_obs, r, done = te.step(0, "random")
            policy.store_pass(0, cur_obs, r, policy.value(cur_obs), done)
            wait_steps += 1
        else:
            a, lp, v = policy.act(cur_obs, cur_mask)
            assert 0 <= a < UNIFIED_ACTION_SIZE and cur_mask[a] == 1, f"illegal action {a}"
            new_obs, r, done = te.step(a, "random")
            policy.store(0, cur_obs, a, lp, r, done, cur_mask, v)
        total_reward += r
        cur_obs, cur_mask = new_obs, te.last_mask
        steps += 1
    print(f"[gen7ou battle] steps={steps} wait_steps={wait_steps} re_picks={te.re_picks} "
          f"terminal_reward={total_reward:+.0f} (+1 win / -1 loss) | won={env.battle1.won} lost={env.battle1.lost}")

    # --- random-battle format: teams are server-generated (team=None) ---
    env2 = ShowdownEnv(battle_format="gen7randombattle", team=None, strict=False,
                       save_replays=args.replays, log_level=logging.WARNING)
    te2 = TraineeEnv(env2, 1, "gen7randombattle")
    o2 = te2.reset()
    print(f"[gen7randombattle reset] ok | obs={o2.shape} legal={int(te2.last_mask.sum())}")

    # --- one PPO update on the collected OU transitions ---
    up = policy.update()
    print(f"[ppo] update ran | {up}")

    te.close(); te2.close()
    print("=== SMOKE TEST PASSED ===")


def main() -> None:
    p = argparse.ArgumentParser(description="Pokemon Showdown self-play RL training")
    p.add_argument("--formats", default="gen7ou,gen7randombattle")
    p.add_argument("--weights", default="0.5,0.5")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--opponent", choices=["random", "frozen", "live"], default="frozen")
    p.add_argument("--buffer-size", type=int, default=2048)
    p.add_argument("--max-battles", type=int, default=200)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", default="128,128")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sync-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-dir", default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    p.add_argument("--replays", default=os.path.join(os.path.dirname(__file__), "replays"))
    p.add_argument("--smoke", action="store_true", help="run a quick CPU pipeline check")
    args = p.parse_args()

    if args.smoke:
        smoke(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
