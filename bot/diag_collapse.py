"""Collapse check: is the actor's output degenerate (a few fixed actions / logit outliers)?"""
from __future__ import annotations
import sys
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, "bot")
from obs import OBS_DIM                    # noqa: E402
from action_space import get_action_space  # noqa: E402
from policy import MaskedPPO               # noqa: E402

AS = get_action_space()
pol = MaskedPPO(obs_dim=OBS_DIM, n_actions=AS.N_ACTIONS, hidden=(128, 128), device="cpu")
pol.actor.load_state_dict(torch.load("bot/data/imitation_actor.pt", map_location="cpu", weights_only=True))
pol.actor.eval()

d = np.load("bot/data/imitation_dataset.npz")
S = d["states"][:3000]
o = torch.as_tensor(S, dtype=torch.float32)
with torch.no_grad():
    L = pol.actor(o).numpy()
print(f"logit stats: mean={L.mean():.3f} std={L.std():.3f} min={L.min():.3f} max={L.max():.3f}")

# per-action mean logit across samples: are some actions systematically hot?
mean_by_action = L.mean(axis=0)
top = np.argsort(-mean_by_action)[:12]
print("\nTop-12 actions by MEAN logit across 3000 train states:")
for a in top:
    a = int(a)
    band = "move" if a < AS.SWITCH_BASE else ("switch" if a < AS.TRANSFORM_BASE else "transform")
    nm = AS.id_to_move[a] if a < AS.SWITCH_BASE else (f"slot{a - AS.SWITCH_BASE}" if a < AS.TRANSFORM_BASE else ["mega", "dyn", "tera"][a - AS.TRANSFORM_BASE])
    print(f"  action {a:4d} ({band:9s} {nm:12s}): mean logit={mean_by_action[a]:+.3f}")

pred = L.argmax(axis=1)
c = Counter(pred.tolist())
print(f"\nunique actions chosen by argmax: {len(c)} / 963")
print("Top-12 argmax picks:")
for a, cnt in c.most_common(12):
    band = "move" if a < AS.SWITCH_BASE else ("switch" if a < AS.TRANSFORM_BASE else "transform")
    nm = AS.id_to_move[a] if a < AS.SWITCH_BASE else (f"slot{a - AS.SWITCH_BASE}" if a < AS.TRANSFORM_BASE else ["mega", "dyn", "tera"][a - AS.TRANSFORM_BASE])
    print(f"  action {a:4d} ({band:9s} {nm:12s}): {cnt} ({cnt / len(pred):.1%})")
