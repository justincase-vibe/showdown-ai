"""Decisive test: does the imitation actor behave on TRAINING observations?

Compare the actor's argmax band (move vs switch) + accuracy on the npz training states against
the label distribution (61% move / 38% switch). Also compare the training-obs sparsity to the
live-obs sparsity (~13.5% nonzero measured in diag_action.py).

  * PRED ~61% move + high accuracy + train obs denser than live  -> LIVE OBS is the bug (mismatch).
  * PRED ~100% switch even on train obs                            -> the actor never learned (train bug).
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, "bot")
from obs import OBS_DIM                       # noqa: E402
from action_space import get_action_space     # noqa: E402
from policy import MaskedPPO                  # noqa: E402

AS = get_action_space()
pol = MaskedPPO(obs_dim=OBS_DIM, n_actions=AS.N_ACTIONS, hidden=(128, 128), device="cpu")
ck = torch.load("bot/data/imitation_actor.pt", map_location="cpu", weights_only=True)
pol.actor.load_state_dict(ck)
pol.actor.eval()

d = np.load("bot/data/imitation_dataset.npz")
states = d["states"]
acts = d["actions"].astype(np.int64)
n = len(states)
idx = np.random.default_rng(0).choice(n, min(5000, n), replace=False)
S = states[idx]
A = acts[idx]
print(f"train obs (n={len(S)}): nonzero={np.mean(np.abs(S) > 1e-9):.1%}  mean={S.mean():+.3f}  max|.|={np.max(np.abs(S)):.3f}")
print(f"(live obs from diag_action.py: nonzero~13.5-15.7%  mean~+0.08  max|.|=1.0)")

o = torch.as_tensor(S, dtype=torch.float32)
with torch.no_grad():
    logits = pol.actor(o).numpy()
pred = logits.argmax(axis=1)
lab_move = float((A < AS.SWITCH_BASE).mean())
lab_switch = float(((A >= AS.SWITCH_BASE) & (A < AS.TRANSFORM_BASE)).mean())
pred_move = float((pred < AS.SWITCH_BASE).mean())
pred_switch = float(((pred >= AS.SWITCH_BASE) & (pred < AS.TRANSFORM_BASE)).mean())
acc = float((pred == A).mean())
# per-slot: is the top move-slot logit above the top switch-slot logit, on average?
top_move = logits[:, :AS.SWITCH_BASE].max(axis=1)
top_switch = logits[:, AS.SWITCH_BASE:AS.TRANSFORM_BASE].max(axis=1)
move_wins = float((top_move > top_switch).mean())
print(f"\nLABEL      : move={lab_move:.1%}  switch={lab_switch:.1%}")
print(f"PRED argmax: move={pred_move:.1%}  switch={pred_switch:.1%}")
print(f"argmax accuracy (pred==label): {acc:.1%}")
print(f"top move-slot logit > top switch-slot logit: {move_wins:.1%} of samples")
