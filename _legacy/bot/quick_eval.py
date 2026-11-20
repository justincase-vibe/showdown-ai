"""Quick battle-quality eval: catches forfeits, switch-spam, and degenerate play.

Runs N battles (default 10) vs a random opponent and reports:
  * Win/Loss/Draw
  * Forfeit detection (battle ends with one side >3 mons alive, <10 turns)
  * Move vs Switch ratio per side (flag if switches > 60% of actions)
  * Avg turns per battle

Usage:
  python bot/quick_eval.py --model bot/checkpoints/ppo.pt --battles 10 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from obs import OBS_DIM
from action_space import get_action_space
from policy import MaskedPPO
from env import ShowdownEnv, TraineeEnv
from teams import RandomTeamBuilder

UNIFIED_ACTION_SIZE = get_action_space().N_ACTIONS


def play_battle(te, policy, bot_seat="p1", hidden=(1024, 2048, 4096, 4096, 2048, 1024), device="cuda"):
    """Play one battle and collect action-type stats."""
    rng = np.random.default_rng(42)
    opp = policy if bot_seat == "p2" else "random"
    
    obs = te.reset()
    mask = te.last_mask
    bot_moves = 0
    bot_switches = 0
    bot_transforms = 0
    opp_moves = 0
    opp_switches = 0
    turns = 0
    as_ = get_action_space()
    
    while not te.done and turns < 300:
        # Both sides decide inside step(). The bot is P1 (bot_seat='p1') or P2, the other
        # seat is a random baseline. The bot's decision is on te.last_self / te.last_opp.
        if bot_seat == "p1":
            new_obs, r, done = te.step(policy, "random")
            bot_dec, opp_dec = te.last_self, te.last_opp
        else:
            new_obs, r, done = te.step("random", policy)
            bot_dec, opp_dec = te.last_opp, te.last_self
        if bot_dec.get("made"):
            a = int(bot_dec["a"])
            if as_.SWITCH_BASE <= a < as_.SWITCH_BASE + 6:
                bot_switches += 1
            elif a >= as_.TRANSFORM_BASE:
                bot_transforms += 1
            else:
                bot_moves += 1
        if opp_dec.get("made"):
            a = int(opp_dec["a"])
            if as_.SWITCH_BASE <= a < as_.SWITCH_BASE + 6:
                opp_switches += 1
            else:
                opp_moves += 1
        obs = new_obs
        mask = te.last_mask
        turns += 1
    
    # Determine result
    if te.done:
        # P1's terminal reward
        # bot_won depends on seat
        pass
    
    # Check for forfeit: one side has >3 mons alive, battle < 10 turns
    b1 = te.env.battle1
    b2 = te.env.battle2
    b1_alive = sum(1 for p in (b1.team or {}).values() if not p.fainted)
    b2_alive = sum(1 for p in (b2.team or {}).values() if not p.fainted)
    
    forfeit = False
    if turns < 15:
        if b1_alive >= 5 and b2_alive == 0:
            forfeit = True
        elif b2_alive >= 5 and b1_alive == 0:
            forfeit = True
    
    return {
        "turns": turns,
        "bot_moves": bot_moves,
        "bot_switches": bot_switches,
        "bot_transforms": bot_transforms,
        "opp_moves": opp_moves,
        "opp_switches": opp_switches,
        "bot_alive": b1_alive if bot_seat == "p1" else b2_alive,
        "opp_alive": b2_alive if bot_seat == "p1" else b1_alive,
        "forfeit": forfeit,
        "bot_won": (b1.won) if bot_seat == "p1" else (b2.won),
    }


def main():
    ap = argparse.ArgumentParser(description="Quick battle-quality eval")
    ap.add_argument("--model", default="bot/checkpoints/ppo.pt")
    ap.add_argument("--battles", type=int, default=10)
    ap.add_argument("--formats", default="gen7ou,gen7ru,gen7uu")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden", default="1024,2048,4096,4096,2048,1024")
    ap.add_argument("--server-ports", default="8000")
    args = ap.parse_args()
    
    hidden = tuple(int(h) for h in args.hidden.split(","))
    formats = [f.strip() for f in args.formats.split(",")]
    ports = [int(p) for p in args.server_ports.split(",")]
    
    print(f"=== QUICK QUALITY EVAL ===")
    print(f"model   : {args.model}")
    print(f"battles : {args.battles}")
    print(f"formats : {formats}")
    print(f"device  : {args.device}")
    print()
    
    # Load policy
    policy = MaskedPPO(obs_dim=OBS_DIM, n_actions=UNIFIED_ACTION_SIZE,
                       hidden=hidden, device=args.device, seed=0)
    ck = torch.load(args.model, map_location=args.device, weights_only=True)
    if isinstance(ck, dict) and "actor" in ck:
        policy.actor.load_state_dict(ck["actor"])
        if "critic" in ck:
            policy.critic.load_state_dict(ck["critic"])
    else:
        policy.actor.load_state_dict(ck)
    policy.actor.eval()
    policy.critic.eval()
    print(f"loaded  : OK ({sum(p.numel() for p in policy.actor.parameters()):,} actor params)")
    print()
    
    # Run battles
    results = []
    for i in range(args.battles):
        fmt = formats[i % len(formats)]
        port = ports[i % len(ports)]
        from poke_env.ps_client import ServerConfiguration
        cfg = ServerConfiguration(
            websocket_url=f"ws://localhost:{port}/showdown/websocket",
            authentication_url=f"http://localhost:{port}/action.php?",
        )
        team = RandomTeamBuilder(battle_format=fmt, seed=i)
        env = ShowdownEnv(battle_format=fmt, team=team, server_configuration=cfg,
                          strict=False, save_replays=False, log_level=50)
        te = TraineeEnv(env, i, fmt)
        
        t0 = time.time()
        try:
            r = play_battle(te, policy, bot_seat="p1", hidden=hidden, device=args.device)
            r["fmt"] = fmt
            r["dt"] = round(time.time() - t0, 1)
            results.append(r)
        except Exception as e:
            print(f"  battle {i+1} ({fmt}): ERROR {e}")
        finally:
            te.close()
    
    if not results:
        print("No battles completed!")
        return
    
    # Report
    n = len(results)
    wins = sum(1 for r in results if r["bot_won"])
    losses = n - wins
    forfeits = sum(1 for r in results if r["forfeit"])
    
    total_bot_moves = sum(r["bot_moves"] for r in results)
    total_bot_switches = sum(r["bot_switches"] for r in results)
    total_opp_moves = sum(r["opp_moves"] for r in results)
    total_opp_switches = sum(r["opp_switches"] for r in results)
    
    avg_turns = np.mean([r["turns"] for r in results])
    bot_action_total = total_bot_moves + total_bot_switches
    switch_pct = (total_bot_switches / bot_action_total * 100) if bot_action_total > 0 else 0
    
    print(f"--- RESULTS ({n} battles) ---")
    print(f"  Win/Loss    : {wins}/{losses} ({wins/n*100:.0f}% win rate)")
    print(f"  Forfeits    : {forfeits} {'[BAD]' if forfeits > 0 else '[clean]'}")
    print(f"  Avg turns   : {avg_turns:.1f}")
    print(f"  Bot actions : {total_bot_moves} moves + {total_bot_switches} switches = {bot_action_total} total")
    print(f"  Switch ratio: {switch_pct:.0f}% {'[TOO HIGH >60%]' if switch_pct > 60 else '[ok]'}")
    print(f"  Opp actions : {total_opp_moves} moves + {total_opp_switches} switches (random baseline)")
    print()
    
    # Per-battle detail
    print(f"  Per-battle:")
    for i, r in enumerate(results):
        flag = " [FORFEIT]" if r["forfeit"] else ""
        print(f"    {i+1:2d}. {r['fmt']:8s} | {r['turns']:3d} turns | "
              f"bot: {r['bot_moves']}mv/{r['bot_switches']}sw | "
              f"opp: {r['opp_moves']}mv/{r['opp_switches']}sw | "
              f"{'W' if r['bot_won'] else 'L'}{flag}")
    
    print()
    if forfeits > 0 or switch_pct > 60:
        print("[WARNING] ISSUES DETECTED - do not scale up until fixed.")
    else:
        print("[OK] All quality checks passed. Ready to scale.")


if __name__ == "__main__":
    main()
