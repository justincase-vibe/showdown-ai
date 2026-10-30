"""Eval a self-play checkpoint (PLAN.md \u00a72.5 "Eval" + \u00a78 progress metric).

Given a checkpoint (``bot/checkpoints/ppo.pt``) it answers the questions that decide whether
the self-play run was worth it, on a FRESH session (clean critic, GPU-light):

  1. WIN-RATE vs a RANDOM opponent, from BOTH seats.
     Self-play is ~50/50 *by design* (same policy both sides), so that number is useless.
     The real "is it good" test is vs a random opponent. We test the bot as P1 AND as P2:
       * bot-P1 vs random high  +  bot-P2 vs random high  -> the skill is REAL (seats-independent).
       * bot-P1 vs random high  +  bot-P2 vs random ~50%  -> the edge is a P1-SEAT BIAS ARTIFACT
         (the bot only wins when it sits in the seat its observation/warm-start prefers).
     This is exactly the ambiguity the ~94% self-play win-rate raised, resolved by measurement.

  2. CRITIC CALIBRATION -- does the value net's "who's winning" read match the real outcome?
     Records the bot's opening value read vs the realized terminal reward, then reports
       sign_hit (want->1.0), corr (want->1.0), rmse/bias (want->0), confidence (should rise).
     This is the "guessed win-rate ~= real win-rate" metric, off-training.

GPU-safe: uses the GPU only for small per-step value/action inferences. Battle generation is
CPU (the local server).

Usage (from the project root):
  python bot/eval_selfplay.py --model bot/checkpoints/ppo.pt --battles 20                 # both seats, 10 each
  python bot/eval_selfplay.py --model bot/checkpoints/ppo.pt --seat p1 --battles 30
  python bot/eval_selfplay.py --model bot/checkpoints/ppo.pt --formats gen7ou,gen9ou --battles 12
  python bot/eval_selfplay.py --model bot/data/imitation_actor.pt --battles 8   # warm-start A/B (random critic)
  python bot/eval_selfplay.py --device cpu --battles 6                          # zero GPU contention

NOTE: ``--model`` accepts a full checkpoint (``ppo.pt``: actor+critic) or a bare actor state_dict
(``imitation_actor.pt``) -- the latter has a random-init critic, so its calibration reads ~50/50
(expected baseline, not a bug).
"""
from __future__ import annotations

import argparse
import logging
import math
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
from teams import StrongTeamBuilder

UNIFIED_ACTION_SIZE = get_action_space().N_ACTIONS


def _load_policy(path: str, device: str, hidden):
    pol = MaskedPPO(obs_dim=OBS_DIM, n_actions=UNIFIED_ACTION_SIZE, hidden=hidden, device=device, seed=0)
    ck = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ck, dict) and "actor" in ck and "critic" in ck:
        pol.actor.load_state_dict(ck["actor"])
        pol.critic.load_state_dict(ck["critic"])
        kind = "full checkpoint (actor+critic)"
    else:
        pol.actor.load_state_dict(ck)
        kind = "bare actor (critic = random init)"
    pol.actor.eval()
    pol.critic.eval()
    return pol, kind


def _new_calib():
    return {"n": 0, "s_err": 0.0, "s_abs": 0.0, "s_sq": 0.0,
            "s_v0": 0.0, "s_R": 0.0, "s_v0R": 0.0, "s_v0sq": 0.0, "s_Rsq": 0.0,
            "s_abs_v0": 0.0, "sign_hit": 0, "sign_n": 0}


def _record(calib, val0, R):
    err = val0 - R
    c = calib
    c["n"] += 1; c["s_err"] += err; c["s_abs"] += abs(err); c["s_sq"] += err * err
    c["s_v0"] += val0; c["s_R"] += R; c["s_v0R"] += val0 * R
    c["s_v0sq"] += val0 * val0; c["s_Rsq"] += R * R; c["s_abs_v0"] += abs(val0)
    if abs(val0) > 1e-9:
        c["sign_n"] += 1
        if (val0 > 0) == (R > 0):
            c["sign_hit"] += 1


def _play_battle(te, pol, bot_seat: str, rng):
    """Play one battle; bot sits in ``bot_seat`` ('p1' or 'p2'), opponent is random.

    Returns (bot_won: bool, val0: float|None) where val0 is the bot's opening value read
    of the game from ITS OWN perspective (positive = bot thinks it's winning).
    """
    opp = pol if bot_seat == "p2" else "random"
    obs, mask = te.last_obs, te.last_mask
    val0 = None
    terminal_r = 0.0  # P1-perspective terminal reward
    while not te.done:
        legal = np.flatnonzero(mask) if mask is not None else np.array([])
        if len(legal) == 0:  # Showdown 'wait' request
            new_obs, r, done = te.step(0, opp)
        elif bot_seat == "p1":
            a, _, v = pol.act(obs, mask)
            if val0 is None:
                val0 = v
            new_obs, r, done = te.step(a, opp)
        else:  # bot is P2; P1 plays random
            a1 = int(rng.choice(legal))
            new_obs, r, done = te.step(a1, opp)
            if val0 is None and te.last_opp is not None and te.last_opp.get("made"):
                val0 = te.last_opp["v"]  # bot's (P2) value read
        obs, mask = new_obs, te.last_mask
        terminal_r = r
    # terminal_r is P1's reward; map to bot's perspective
    if bot_seat == "p1":
        bot_won = terminal_r > 0
        bot_R = float(terminal_r)
    else:
        bot_won = terminal_r < 0
        bot_R = -float(terminal_r)
    return bool(bot_won), val0, bot_R


def _run_seat(fmts, per_fmt, pol, bot_seat, seed, replays):
    rng = np.random.default_rng(seed)
    calib = _new_calib()
    wins = losses = draws = 0
    by_fmt = {}
    for fi, fmt in enumerate(fmts):
        env = ShowdownEnv(battle_format=fmt, team=StrongTeamBuilder(battle_format=fmt),
                          strict=False, save_replays=replays, log_level=logging.WARNING)
        te = TraineeEnv(env, fi, fmt)
        te.reset()
        w = l = d = 0
        for _ in range(per_fmt):
            won, val0, bot_R = _play_battle(te, pol, bot_seat, rng)
            if won:
                w += 1; wins += 1
            elif bot_R < 0:
                l += 1; losses += 1
            else:
                d += 1; draws += 1
            if val0 is not None:
                _record(calib, float(val0), bot_R)
            te.reset()
        by_fmt[fmt] = (w, l, d)
        te.close()
    return wins, losses, draws, by_fmt, calib


def _print_calib(label, calib):
    n = calib["n"]
    print(f"\n{label} -- critic calibration (n={n}):")
    if n == 0:
        print("  (no completed battles)")
        return
    mn_v = calib["s_v0"] / n
    mn_R = calib["s_R"] / n
    cov = calib["s_v0R"] / n - mn_v * mn_R
    var_v = calib["s_v0sq"] / n - mn_v * mn_v
    var_R = calib["s_Rsq"] / n - mn_R * mn_R
    rmse = math.sqrt(calib["s_sq"] / n)
    mae = calib["s_abs"] / n
    bias = calib["s_err"] / n
    corr = (cov / math.sqrt(var_v * var_R)) if (var_v > 1e-12 and var_R > 1e-12) else float("nan")
    sign_hit = calib["sign_hit"] / calib["sign_n"] if calib["sign_n"] else float("nan")
    conf = calib["s_abs_v0"] / n
    def f(x):
        return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
    print(f"  sign_hit   : {f(sign_hit)}   (share of games whose WIN/LOSS the critic read right; want -> 1.0)")
    print(f"  corr       : {f(corr)}   (Pearson read-vs-outcome; want -> 1.0)")
    print(f"  rmse / mae : {rmse:.3f} / {mae:.3f}   (want -> 0)")
    print(f"  bias       : {bias:+.3f}   (want -> 0)")
    print(f"  confidence : {conf:.3f}   (mean |read|; should RISE with skill while sign_hit stays high)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "ppo.pt"))
    ap.add_argument("--formats", default="gen6ou,gen7ou,gen9ou", help="eval formats (comma list)")
    ap.add_argument("--battles", type=int, default=20, help="total battles per seat")
    ap.add_argument("--seat", choices=["p1", "p2", "both"], default="both",
                    help="which seat the bot sits in (default both)")
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--replays", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "replays"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    seats = ["p1", "p2"] if args.seat == "both" else [args.seat]

    print("=== SELF-PLAY EVAL ===")
    print(f"model     : {args.model}")
    print(f"formats   : {fmts}")
    print(f"seats     : {seats}  ({args.battles} battles each)")
    print(f"device    : {args.device}")
    pol, kind = _load_policy(args.model, args.device, hidden)
    print(f"loaded    : {kind}")

    start = time.time()
    results = {}
    for seat in seats:
        per_fmt = max(1, math.ceil(args.battles / len(fmts)))
        print(f"\n--- bot as {seat.upper()} vs random ---")
        w, l, d, by_fmt, calib = _run_seat(fmts, per_fmt, pol, seat, args.seed + (0 if seat == "p1" else 7), args.replays)
        decided = w + l
        wr = w / decided if decided else float("nan")
        results[seat] = (w, l, d, wr, calib)
        detail = "  ".join(f"{fmt} {a}-{b}" + (f"-{c}d" if c else "") for fmt, (a, b, c) in by_fmt.items())
        print(f"  win-rate vs random: {w}/{decided} = {wr:.1%}  ({detail})")
        _print_calib(f"{seat.upper()}-seat", calib)

    elapsed = time.time() - start
    print(f"\ntotal elapsed: {elapsed:.1f}s")

    # --- verdict: is the edge real or a seat-bias artifact? ---
    print("\n=== READ ME ===")
    if "p1" in results and "p2" in results:
        wr1 = results["p1"][3]
        wr2 = results["p2"][3]
        if not (isinstance(wr1, float) and math.isnan(wr1)) and not (isinstance(wr2, float) and math.isnan(wr2)):
            if wr1 >= 0.60 and wr2 >= 0.60:
                print(f"  BOTH SEATS STRONG ({wr1:.0%}/{wr2:.0%} vs random) -> the skill is REAL and seat-independent.")
            elif wr1 >= 0.60 and wr2 < 0.55:
                print(f"  P1 strong ({wr1:.0%}) but P2 weak ({wr2:.0%}) -> the edge is a P1-SEAT BIAS ARTIFACT.")
                print("  The bot mostly wins by sitting in its preferred seat, not by genuine strength.")
            elif wr2 >= 0.60 and wr1 < 0.55:
                print(f"  P2 strong ({wr2:.0%}) but P1 weak ({wr1:.0%}) -> unusual seat bias on P1.")
            else:
                print(f"  NEITHER seat clearly beats random (P1 {wr1:.0%} / P2 {wr2:.0%}) -> keep self-playing; not strong yet.")
    for seat, (w, l, d, wr, calib) in results.items():
        if kind.startswith("bare actor"):
            print(f"  NOTE ({seat}): bare actor has a random-init critic -> calibration ~50/50 is expected, not a defect.")
    print("=== END ===")


if __name__ == "__main__":
    main()
