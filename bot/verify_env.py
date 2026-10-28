"""Torch-free verification of the battle pipeline (env + teams + obs + reward).

Plays N random-vs-random battles across the requested formats and checks:
  * observation vector has the exact contract shape and is finite,
  * the action mask is non-empty and always contains the chosen action,
  * battles run to completion and produce a +/-1 terminal reward,
  * OU teams from our pool are accepted by the server (randbat uses server teams).

This is the "can we reliably start matches?" check for the NEW env. No GPU, no torch.

    python bot/verify_env.py --battles 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import ShowdownEnv, TraineeEnv, UNIFIED_ACTION_SIZE  # noqa: E402
from obs import OBS_DIM  # noqa: E402
from teams import RandomTeamBuilder  # noqa: E402


def random_action(mask: np.ndarray) -> int:
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return 0
    return int(np.random.choice(legal))


def play_one(fmt: str, replays: str) -> dict:
    team = None if "randombattle" in fmt else RandomTeamBuilder(
        battle_format=fmt, seed=np.random.randint(10**6)
    )
    env = ShowdownEnv(
        battle_format=fmt,
        team=team,
        strict=False,
        save_replays=replays,
        log_level=logging.ERROR,
    )
    te = TraineeEnv(env, 0, fmt)
    obs = te.reset()
    assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape}"
    assert np.all(np.isfinite(obs)), "obs has non-finite values"

    cur_obs, cur_mask = obs, te.last_mask
    steps = 0
    total = 0.0
    done = False
    max_steps = 300
    while not done and steps < max_steps:
        a = random_action(cur_mask)
        assert cur_mask[a] == 1, f"illegal action {a} in mask"
        cur_obs, r, done = te.step(a, "random")
        assert cur_obs.shape == (OBS_DIM,) and np.all(np.isfinite(cur_obs))
        total += r
        steps += 1
    res = {
        "fmt": fmt,
        "steps": steps,
        "terminal_reward": total,
        "won": bool(env.battle1.won),
        "lost": bool(env.battle1.lost),
        "finished": bool(env.battle1.finished),
        "ok": done,
    }
    te.close()
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--formats", default="gen7ou,gen7randombattle,gen6ou")
    p.add_argument("--battles", type=int, default=2)
    p.add_argument("--replays", default=os.path.join(os.path.dirname(__file__), "replays"))
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    np.random.seed(0)
    print(f"OBS_DIM={OBS_DIM}  ACTIONS={UNIFIED_ACTION_SIZE}")
    all_ok = True
    for fmt in fmts:
        for i in range(args.battles):
            t0 = time.time()
            r = play_one(fmt, args.replays)
            r["dt"] = round(time.time() - t0, 1)
            ok = r.pop("ok") and r["finished"] and r["terminal_reward"] in (1.0, -1.0)
            all_ok = all_ok and ok
            print(f"[{r['fmt']}] battle#{i} steps={r['steps']} reward={r['terminal_reward']:+.0f} "
                  f"won={r['won']} lost={r['lost']} finished={r['finished']} {r['dt']}s {'OK' if ok else 'FAIL'}")
    print("=== ALL PASSED ===" if all_ok else "=== SOME FAILED ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
