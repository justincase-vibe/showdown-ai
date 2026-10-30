"""Self-play training harness for the Pokemon Showdown RL bot.

Runs N concurrent battles across a *mix of formats* (e.g. gen7ou + gen7randombattle),
collects transitions into a PPO buffer, updates the shared policy on the chosen device,
and tracks rolling win-rate per format. The opponent is pluggable:

  * ``random`` -- a random-legal-move baseline (sanity check / curriculum floor). Only
                  the trainee (P1) learns; P2 is a noise floor.
  * ``frozen`` -- a periodically-updated copy of the trainee (P1 learns; P2 is a lagging
                  target, not stored).
  * ``live``   -- the trainee itself plays BOTH sides (default). A loss on one side is a
                  win on the other; BOTH sides' transitions update the SAME shared policy
                  (zero-sum reward). This is the "both sides train the model" mode.

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
import torch

# Make sibling modules importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import ShowdownEnv, TraineeEnv, UNIFIED_ACTION_SIZE  # noqa: E402
from action_space import get_action_space  # noqa: E402
from obs import OBS_DIM  # noqa: E402
from policy import MaskedPPO  # noqa: E402
from teams import RandomTeamBuilder, StrongTeamBuilder  # noqa: E402
from poke_env.ps_client import ServerConfiguration  # noqa: E402


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


def _server_config(port: int) -> ServerConfiguration:
    """Build a poke_env ServerConfiguration pointing at a local server on ``port``."""
    return ServerConfiguration(
        websocket_url=f"ws://localhost:{port}/showdown/websocket",
        authentication_url="https://play.pokemonshowdown.com/action.php?",
    )


def build_envs(
    formats: List[str],
    weights: List[float],
    num_envs: int,
    seed: int,
    save_replays: Optional[str],
    team_mode: str = "strong",
    server_ports: Optional[List[int]] = None,
) -> List[TraineeEnv]:
    counts = apportion(num_envs, weights)
    envs: List[TraineeEnv] = []
    idx = 0
    if server_ports is None:
        server_ports = [8000]
    for fmt, c in zip(formats, counts):
        for _ in range(c):
            if "randombattle" in fmt:
                team = None
            elif team_mode == "random":
                team = RandomTeamBuilder(battle_format=fmt, seed=seed + idx)
            else:
                team = StrongTeamBuilder(battle_format=fmt, seed=seed + idx)
            port = server_ports[idx % len(server_ports)]
            env = ShowdownEnv(
                battle_format=fmt,
                team=team,
                server_configuration=_server_config(port),
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
def _store_opponent_side(e: "TraineeEnv", policy: "MaskedPPO", opponent, done: bool) -> None:
    """Both-sides self-play: store the opponent's (P2) transition when the opponent IS
    the shared training policy (``--opponent live``). P2's (obs, action, logprob, value)
    come from the state P2 decided from (captured in TraineeEnv.step); the reward is
    P2's zero-sum reward (``rew[a2]`` = -P1's at a terminal win/loss). Stored under a
    distinct env id so P1 and P2 are separate trajectories for per-episode GAE. No-op
    when the opponent is not the shared policy (random / frozen) or P2 had nothing to
    decide (a Showdown 'wait' request).
    """
    if opponent is not policy:
        return
    lo = e.last_opp
    if not lo or not lo.get("made"):
        return
    o2, a2, m2 = lo.get("obs"), lo.get("a"), lo.get("mask")
    if o2 is None or a2 is None or m2 is None or int(np.sum(m2)) == 0:
        return
    policy.store(
        e.id + 1_000_000, o2, a2, lo.get("lp", 0.0), e.last_opp_rew, done, m2, lo.get("v", 0.0)
    )


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

    # Parse server ports (e.g. "8000,8001,...,8009")
    if getattr(args, "server_ports", None):
        server_ports = [int(p.strip()) for p in args.server_ports.split(",") if p.strip()]
    else:
        server_ports = [8000]
    log.info("server_ports=%s (%d server(s))", server_ports, len(server_ports))

    envs = build_envs(formats, weights, args.num_envs, args.seed, args.replays,
                      team_mode=args.team_mode, server_ports=server_ports)
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
        # start matched: both begin from the same weights (random-init, or the
        # imitation warm-start loaded just below). A trainee vs a *random* opponent
        # would get an easy early win-rate spike that misleads the value baseline.
        policy.sync_opponent(opponent)
    elif args.opponent == "live":
        opponent = policy

    # ---- optional warm-start: load a pre-trained actor (e.g. the imitation BC) ----
    if getattr(args, "init_actor", None) and os.path.isfile(args.init_actor):
        try:
            sd = torch.load(args.init_actor, map_location=policy.device, weights_only=True)
            policy.actor.load_state_dict(sd)
            log.info("warm-started policy actor from %s", args.init_actor)
            if args.opponent == "frozen" and opponent is not policy:
                opponent.actor.load_state_dict(sd)  # keep the frozen target matched
        except Exception as e:
            log.warning("--init-actor load failed (%s); continuing with random init", e)
    elif getattr(args, "init_actor", None):
        log.warning("--init-actor path not found: %s", args.init_actor)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    stats_path = os.path.join(args.ckpt_dir, "stats.jsonl")

    wins: Dict[str, int] = {}
    losses: Dict[str, int] = {}
    battles_done = 0
    transitions = 0
    start = time.time()

    # --- quality tracking (switch ratio, forfeits) ---
    as_ = get_action_space()
    battle_moves = 0
    battle_switches = 0
    battle_forfeits = 0
    battle_turns = 0
    recent_battles: List[dict] = []  # last N battles for quality stats

    # reset all envs
    cur_obs: Dict[int, np.ndarray] = {}
    cur_mask: Dict[int, np.ndarray] = {}
    for e in envs:
        cur_obs[e.id] = e.reset()
        cur_mask[e.id] = e.last_mask

    def buffer_size() -> int:
        # Count ONLY what update() can actually drain: the *merged* buffer built
        # from FINISHED episodes (policy._buffer). In-progress live episodes
        # (policy._episodes) are NOT trainable yet -- update() only consumes the
        # merged buffer. Counting live episodes made the update gate fire
        # prematurely, which (a) broke out of the collect loop after the FIRST env
        # (starving the rest) and (b) spun update() on an empty merged buffer
        # (empty upstats) while battles never got to finish.
        if policy._buffer is not None:
            return len(policy._buffer["act"])
        return 0

    def log_stats(tag: str, extra: dict = None):
        # Quality metrics from recent battles
        n_recent = min(20, len(recent_battles))
        if n_recent > 0:
            recent = recent_battles[-n_recent:]
            avg_sw = round(np.mean([b["sw%"] for b in recent]))
            avg_turns = round(np.mean([b["turns"] for b in recent]), 1)
            n_forfeit = sum(1 for b in recent if b["forfeit"])
        else:
            avg_sw = 0
            avg_turns = 0
            n_forfeit = 0
        rec = {
            "t": int(time.time() - start),
            "battles": battles_done,
            "wins": dict(wins),
            "losses": dict(losses),
            "winrate": {
                k: (wins.get(k, 0) / (wins.get(k, 0) + losses.get(k, 0))
                    if (wins.get(k, 0) + losses.get(k, 0)) else None)
                for k in (set(wins) | set(losses))
            },
            "quality": {"avg_sw%": avg_sw, "avg_turns": avg_turns, "forfeits": n_forfeit},
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
                    # Track action type for quality stats
                    if a >= as_.SWITCH_BASE and a < as_.SWITCH_BASE + 6:
                        battle_switches += 1
                    else:
                        battle_moves += 1
                    new_obs, r, done = e.step(a, opponent)
                    policy.store(e.id, obs, a, lp, r, done, mask, v)
                # Both-sides self-play: also store the opponent's transition when the
                # opponent is the shared policy (--opponent live); no-op otherwise.
                _store_opponent_side(e, policy, opponent, done)
                transitions += 1
                battle_turns += 1
                if done:
                    battles_done += 1
                    key = fmt_of[e.id]
                    if r > 0:
                        wins[key] = wins.get(key, 0) + 1
                    elif r < 0:
                        losses[key] = losses.get(key, 0) + 1
                    else:
                        pass  # draw / double-kO
                    
                    # --- quality check: detect forfeit (one side >3 mons alive, <15 turns) ---
                    b1 = e.env.battle1
                    b2 = e.env.battle2
                    b1_alive = sum(1 for p in (b1.team or {}).values() if not p.fainted)
                    b2_alive = sum(1 for p in (b2.team or {}).values() if not p.fainted)
                    if battle_turns < 15 and ((b1_alive >= 5 and b2_alive == 0) or (b2_alive >= 5 and b1_alive == 0)):
                        battle_forfeits += 1
                    
                    # Log per-battle quality
                    total_actions = battle_moves + battle_switches
                    sw_pct = (battle_switches / total_actions * 100) if total_actions > 0 else 0
                    recent_battles.append({
                        "b": battles_done, "fmt": key, "turns": battle_turns,
                        "mv": battle_moves, "sw": battle_switches, "sw%": round(sw_pct),
                        "forfeit": battle_forfeits > 0 and (b1_alive >= 5 or b2_alive >= 5),
                        "result": "W" if r > 0 else ("L" if r < 0 else "D")
                    })
                    # Reset per-battle counters
                    battle_moves = 0
                    battle_switches = 0
                    battle_turns = 0
                    
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
    """Prove the full pipeline works with no heavy training: env + teams +
    masked policy + one PPO update, across an OU format and a random-battle format.

    Device-aware: default is CPU (safe -- never contends with a GPU in use). Pass
    ``--smoke --device cuda`` for a quick GPU preflight (a few CUDA act() + one CUDA
    PPO update + GPU memory report) before launching the long run.
    """
    logging.basicConfig(level=logging.WARNING)
    device = args.device if args.device in ("cpu", "cuda") else "cpu"
    print(f"=== SMOKE TEST ({device.upper()}) ===")
    print(f"OBS_DIM={OBS_DIM}  ACTIONS={UNIFIED_ACTION_SIZE}")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False -- "
                             "check the driver / that the GPU is not fully held by another process.")
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"| free/total = {torch.cuda.mem_get_info(0)[0]/1e9:.1f}/{torch.cuda.mem_get_info(0)[1]/1e9:.1f} GB")

    policy = MaskedPPO(obs_dim=OBS_DIM, n_actions=UNIFIED_ACTION_SIZE,
                       hidden=(64, 64), device=device, seed=0)

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
    p.add_argument("--team-mode", choices=["strong", "random"], default="strong",
                   help="strong=coherent curated teams (default); random=weak bag baseline")
    p.add_argument("--opponent", choices=["random", "frozen", "live"], default="live",
                   help="random/frozen = P1-only learning; live = SHARED policy plays "
                        "both sides and both transitions train the model (default)")
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
    p.add_argument("--init-actor", default=None,
                   help="path to a pre-trained actor state_dict to warm-start from (optional)")
    p.add_argument("--server-ports", default=None,
                   help="comma-separated list of Showdown server ports (e.g. 8000,8001,8002)")
    p.add_argument("--smoke", action="store_true", help="run a quick CPU pipeline check")
    args = p.parse_args()

    if args.smoke:
        smoke(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
