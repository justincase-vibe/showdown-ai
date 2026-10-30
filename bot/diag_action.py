"""Step-level diagnostic on a LIVE observation.

Shows, for each policy, the legal actions, the raw-logit softmax probability mass split by
band (move / switch / transform), and the top-5 actions -- on the SAME live obs. Deterministic
(raw logits + softmax over the legal mask), independent of ``act()``'s sampling. This answers:
  * Is the observation sane (not ~all-zero)?
  * Is the self-play policy switch-biased, and did self-play push it MORE toward switches than
    the imitation baseline? (the 50-switches-vs-16-moves replay suggests yes)
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from obs import OBS_DIM                      # noqa: E402
from action_space import get_action_space    # noqa: E402
from policy import MaskedPPO                 # noqa: E402
from env import ShowdownEnv, TraineeEnv      # noqa: E402
from teams import StrongTeamBuilder          # noqa: E402
import logging                               # noqa: E402
logging.basicConfig(level=logging.ERROR)

AS = get_action_space()


def band(a: int) -> str:
    if a < AS.SWITCH_BASE:
        return "move"
    if a < AS.TRANSFORM_BASE:
        return "switch"
    return "transform"


def name(a: int) -> str:
    if a < AS.SWITCH_BASE:
        return f"MOVE {AS.id_to_move[a]}"
    if a < AS.TRANSFORM_BASE:
        return f"SWITCH slot {a - AS.SWITCH_BASE}"
    return f"TRANSFORM {['mega', 'dynamax', 'tera'][a - AS.TRANSFORM_BASE]}"


def load(path: str) -> MaskedPPO:
    pol = MaskedPPO(obs_dim=OBS_DIM, n_actions=AS.N_ACTIONS, hidden=(128, 128), device="cpu")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ck, dict) and "actor" in ck:
        pol.actor.load_state_dict(ck["actor"])
    else:
        pol.actor.load_state_dict(ck)
    pol.actor.eval()
    return pol


def analyze(pol: MaskedPPO, obs: np.ndarray, mask: np.ndarray, label: str) -> None:
    o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = pol.actor(o)[0].numpy()
    m = mask.astype(bool)
    legal = np.flatnonzero(m)
    lz = logits[legal]
    e = np.exp(lz - lz.max())
    p = e / e.sum()
    mass = {}
    for b in ("move", "switch", "transform"):
        idx = [i for i, a in enumerate(legal) if band(int(a)) == b]
        mass[b] = float(p[idx].sum()) if idx else 0.0
    nz = float(np.mean(np.abs(obs) > 1e-9))
    print(f"\n=== {label} ===")
    print(f"  obs: nonzero={nz:.1%}  mean={np.mean(obs):+.3f}  max|.|={np.max(np.abs(obs)):.3f}  legal={len(legal)}")
    print(f"  prob mass  move={mass['move']:.3f}  switch={mass['switch']:.3f}  transform={mass['transform']:.3f}")
    for k in range(min(5, len(legal))):
        i = int(np.argsort(-p)[k]); a = int(legal[i])
        print(f"    {p[i]:6.2%}  {name(a)}")


def main() -> None:
    fmt = "gen6ou"
    env = ShowdownEnv(battle_format=fmt, team=StrongTeamBuilder(battle_format=fmt),
                      strict=False, save_replays=False, log_level=logging.ERROR)
    te = TraineeEnv(env, 0, fmt)
    te.reset()
    ppo = load(os.path.join(HERE, "checkpoints", "ppo.pt"))
    imp = load(os.path.join(HERE, "data", "imitation_actor.pt"))
    for step in range(4):
        if te.done:
            break
        obs, mask = te.last_obs, te.last_mask
        analyze(ppo, obs, mask, f"ppo.pt (self-play, run-1)      step {step}")
        analyze(imp, obs, mask, f"imitation_actor.pt (baseline)  step {step}")
        a, _, _ = ppo.act(obs, mask)
        te.step(a, "random")
    env.close()


if __name__ == "__main__":
    main()
