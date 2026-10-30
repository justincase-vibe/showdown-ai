"""Is the switch/transform-vs-move logit collapse from INITIALIZATION or from TRAINING?

Same 3000 train states, two actors: fresh random-init vs the trained imitation actor.
  * FRESH also shows transform/switch >> move  -> init/architecture bias (fix the init/forward).
  * FRESH is flat, TRAINED is collapsed          -> the imitation loss/mask drove it (fix training).
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, "bot")
from obs import OBS_DIM                    # noqa: E402
from action_space import get_action_space  # noqa: E402
from policy import MaskedPPO               # noqa: E402

AS = get_action_space()
d = np.load("bot/data/imitation_dataset.npz")
S = d["states"][:3000]
o = torch.as_tensor(S, dtype=torch.float32)


def banding(pol: MaskedPPO, tag: str) -> None:
    with torch.no_grad():
        L = pol.actor(o).numpy()
    mb = L.mean(axis=0)
    move = mb[:AS.SWITCH_BASE].mean()
    switch = mb[AS.SWITCH_BASE:AS.TRANSFORM_BASE].mean()
    trans = mb[AS.TRANSFORM_BASE:].mean()
    spread = mb.max() - mb.min()
    print(f"{tag}: move={move:+.3f}  switch={switch:+.3f}  transform={trans:+.3f}   "
          f"(global logit spread={spread:.2f})")


banding(MaskedPPO(obs_dim=OBS_DIM, n_actions=AS.N_ACTIONS, hidden=(128, 128), device="cpu"), "FRESH init   ")
trained = MaskedPPO(obs_dim=OBS_DIM, n_actions=AS.N_ACTIONS, hidden=(128, 128), device="cpu")
trained.actor.load_state_dict(torch.load("bot/data/imitation_actor.pt", map_location="cpu", weights_only=True))
trained.actor.eval()
banding(trained, "TRAINED actor")
