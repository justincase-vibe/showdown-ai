"""Eval a self-play checkpoint (PLAN.md \u00a72.5 "Eval" + \u00a78 progress metric).

This is the "position benchmark" made runnable: given a checkpoint (``bot/checkpoints/ppo.pt``)
it answers the two questions that actually decide whether the 1k-battle run was worth it:

  1. CRITIC CALIBRATION  -- does the value net's "who's winning" read match the real outcome?
     For each finished battle we record the critic's opening read (value at the first decision
     state) vs the realized terminal reward (+1 win / -1 loss), then report:
       * rmse / mae / bias   (want -> 0)
       * pearson corr        (want -> 1)
       * sign_hit            (want -> 1)  <- the single most interpretable number
       * confidence          (mean |read|) <- should RISE with skill AND stay correct
     This is the SAME metric the self-play loop logs (``calib_*``), but measured on a FRESH
     eval session (clean critic, no training), so it is the honest "did it learn to read games"
     number. A bot that can't tell a winning position from a losing one has not really learned.

  2. WIN-RATE vs RANDOM  -- a hard floor. The self-play bot (both sides, zero-sum) hovers ~50/50
     against *itself* by design, so that number is useless. Against a *random* opponent it must
     be clearly above 50%. This is the "clearly beat random and dumb rules" goal from PLAN.md.

GPU-safe: uses the GPU only for small per-step value/action inferences (a few seconds), so it is
fine to run even while the training run is occupying the card. Battle generation is CPU (server).

Usage (from the project root):
  python bot/eval_selfplay.py --model bot/checkpoints/ppo.pt --battles 20
  python bot/eval_selfplay.py --model bot/checkpoints/ppo.pt --formats gen7ou,gen9ou --battles 12
  python bot/eval_selfplay.py --model bot/data/imitation_actor.pt --battles 8   # warm-start A/B
  python bot/eval_selfplay.py --device cpu --battles 6                          # zero GPU contention

NOTE: ``--model`` accepts either a full checkpoint (``ppo.pt``: actor+critic) or a bare actor
state_dict (``imitation_actor.pt``) -- the latter has no trained critic, so its calibration will
read ~50/50 (that's the expected baseline, not a bug).
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


class _RandomOpp:
    """P2 = uniform random over the legal mask (a real 'random' baseline opponent)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, obs, mask):
        idx = np.flatnonzero(mask)
        a = int(self.rng.choice(idx))
        return a, 0.0, 0.0


def _load_policy(path: str, device: str, hidden) -> MaskedPPO:
    pol = MaskedPPO(obs_dim=OBS_DIM, n_actions=UNIFIED_ACTION_SIZE, hidden=hidden, device=device, seed=0)
    ck = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ck, dict) and "actor" in ck and "critic" in ck:
        pol.actor.load_state_dict(ck["actor"])
        pol.critic.load_state_dict(ck["critic"])
        kind = "full checkpoint (actor+critic)"
    else:
        pol.actor.load_state_dict(ck)  # bare actor (e.g. imitation warm-start)
        kind = "bare actor (critic = random init)"
    pol.actor.eval()
    pol.critic.eval()
    return pol, kind


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "ppo.pt"))
    ap.add_argument("--formats", default="gen6ou,gen7ou,gen9ou", help="eval formats (comma list)")
    ap.add_argument("--battles", type=int, default=20, help="total battles across all formats")
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--replays", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "replays"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    per_fmt = max(1, math.ceil(args.battles / len(formats)))

    print(f"=== SELF-PLAY EVAL ===")
    print(f"model     : {args.model}")
    print(f"formats   : {formats}  (~{per_fmt} battles each, ~{args.battles} total)")
    print(f"device    : {args.device}  (small inferences only -- GPU-safe during training)")
    pol, kind = _load_policy(args.model, args.device, hidden)
    print(f"loaded    : {kind}")

    rng = np.random.default_rng(args.seed)
    calib = {"n": 0, "s_err": 0.0, "s_abs": 0.0, "s_sq": 0.0,
             "s_v0": 0.0, "s_R": 0.0, "s_v0R": 0.0, "s_v0sq": 0.0, "s_Rsq": 0.0,
             "s_abs_v0": 0.0, "sign_hit": 0, "sign_n": 0}
    wins = losses = draws = 0
    by_fmt: dict = {}
    start = time.time()

    done_total = 0
    for fi, fmt in enumerate(formats):
        if done_total >= args.battles:
            break
        env = ShowdownEnv(battle_format=fmt, team=StrongTeamBuilder(battle_format=fmt),
                          strict=False, save_replays=args.replays, log_level=logging.WARNING)
        te = TraineeEnv(env, fi, fmt)
        opp = _RandomOpp(np.random.default_rng(args.seed + 1000 + fi))
        obs = te.reset()
        mask = te.last_mask
        val0 = None
        w = l = d = 0
        while not te.done and done_total < args.battles:
            if int(mask.sum()) == 0:
                new_obs, r, done = te.step(0, opp)
            else:
                a, _, v = pol.act(obs, mask)
                if val0 is None:
                    val0 = v
                new_obs, r, done = te.step(a, opp)
            obs, mask = new_obs, te.last_mask
            if done:
                done_total += 1
                if r > 0:
                    w += 1; wins += 1
                elif r < 0:
                    l += 1; losses += 1
                else:
                    d += 1; draws += 1
                # critic calibration: opening read vs realized outcome
                if val0 is not None:
                    R = float(r)
                    err = val0 - R
                    c = calib
                    c["n"] += 1; c["s_err"] += err; c["s_abs"] += abs(err); c["s_sq"] += err * err
                    c["s_v0"] += val0; c["s_R"] += R; c["s_v0R"] += val0 * R
                    c["s_v0sq"] += val0 * val0; c["s_Rsq"] += R * R; c["s_abs_v0"] += abs(val0)
                    if abs(val0) > 1e-9:
                        c["sign_n"] += 1
                        if (val0 > 0) == (R > 0):
                            c["sign_hit"] += 1
                te.reset()
                obs, mask = te.last_obs, te.last_mask
                val0 = None
        by_fmt[fmt] = (w, l, d)
        te.close()

    elapsed = time.time() - start
    decided = wins + losses
    wr = wins / decided if decided else float("nan")

    print("\n=== RESULTS ===")
    print(f"battles   : {done_total}  ({' + '.join(f'{fmt} {w}-{l}' + (f'-{d}d' if d else '') for fmt, (w, l, d) in by_fmt.items())})")
    print(f"win-rate vs RANDOM: {wins}/{decided} = {wr:.1%}" + (f"  ({draws} draws)" if draws else ""))
    print(f"elapsed   : {elapsed:.1f}s")

    n = calib["n"]
    if n > 0:
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
        print(f"\ncritic calibration (n={n}):")
        print(f"  sign_hit    : {sign_hit:.2f}   (share of games whose WIN/LOSS direction the critic read right; want -> 1.0)")
        print(f"  corr        : {corr:.2f}   (Pearson read-vs-outcome; want -> 1.0)")
        print(f"  rmse / mae  : {rmse:.3f} / {mae:.3f}   (want -> 0)")
        print(f"  bias        : {bias:+.3f}   (want -> 0)")
        print(f"  confidence  : {conf:.3f}   (mean |read|; should RISE with skill while sign_hit stays high)")
    else:
        print("\ncritic calibration: (no completed battles)")

    # --- verdict helpers ---
    print("\n=== READ ME ===")
    if n > 0 and not math.isnan(sign_hit):
        if wr >= 0.60 and sign_hit >= 0.60:
            print("  PASS-ish: beats random AND reads game direction. Consider scaling up (more battles / 2k+).")
        elif wr >= 0.55:
            print("  MARGINAL: above random but critic is still weak -- more self-play likely helps.")
        else:
            print("  WEAK: not clearly above random yet -- keep iterating (this is expected at only 1k battles).")
    if kind.startswith("bare actor"):
        print("  NOTE: bare actor has a random-init critic -> calibration ~50/50 is expected, not a defect.")
    print("=== END ===")


if __name__ == "__main__":
    main()
