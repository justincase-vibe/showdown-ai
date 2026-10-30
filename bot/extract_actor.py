"""Extract the bare actor state_dict from a full self-play checkpoint so it can be
warm-started via ``train.py --init-actor``.

``train.py``'s ``--init-actor`` does ``torch.load(path); policy.actor.load_state_dict(sd)`` --
it expects a **bare actor state_dict**, NOT the full ``{actor, critic, opt}`` dict that
``policy.save()`` writes. Pointing it at a full checkpoint would raise, be caught, and
silently continue from *random init* (discarding the whole run). This helper pulls out the
actor and, crucially, VERIFIES the extraction by loading it into a fresh ``MaskedPPO.actor``
-- the exact code path ``--init-actor`` uses -- so a structural mismatch fails HERE, not
9 hours into an unattended run.

Note: this carries the **actor only**. The critic + optimizer reset to init (the established
warm-start pattern; run 1 and the morning CPU session both did actor-only). The actor is what
plays; the critic re-warms from reward over the first ~100-200 battles. See PLAN.md §8b.

Usage:  python bot/extract_actor.py [SRC] [DST]
"""
from __future__ import annotations
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from policy import MaskedPPO          # noqa: E402
from obs import OBS_DIM               # noqa: E402
from action_space import get_action_space  # noqa: E402

ROOT = os.path.dirname(HERE)
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bot", "checkpoints", "ppo_run1_1k.pt")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "bot", "data", "actor_run1_1k.pt")

ck = torch.load(SRC, map_location="cpu", weights_only=True)
assert isinstance(ck, dict) and "actor" in ck, f"unexpected checkpoint structure: {type(ck)}"
actor_sd = ck["actor"]
torch.save(actor_sd, DST)
n_tensors = len(actor_sd)
n_params = sum(int(p.numel()) for p in actor_sd.values() if hasattr(p, "numel"))
print(f"saved bare actor -> {DST}  ({n_tensors} tensors, {n_params:,} params)")

# VERIFY: load into a fresh policy actor -- the exact --init-actor path (policy.py line ~166).
pol = MaskedPPO(obs_dim=OBS_DIM, n_actions=get_action_space().N_ACTIONS, hidden=(128, 128), device="cpu")
reloaded = torch.load(DST, map_location="cpu", weights_only=True)
pol.actor.load_state_dict(reloaded)
print("VERIFY OK: actor loads into a fresh MaskedPPO actor (the --init-actor path). Safe to launch.")
