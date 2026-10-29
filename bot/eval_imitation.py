"""Evaluate the imitation (behavioral-cloning) actor.

Raw top-1 on the 963-way space is misleading because most actions are *illegal* for a
given state (a Garchomp can't use Doom Desire; you can't switch to a fainted slot).
This script reports the metrics that actually matter:

  * **masked accuracy** — fraction of states whose top-1 prediction lies in the legal set
    (species-legal moves for that gen + the 6 switch actions).
  * **illegal-argmax rate** — how often the raw top-1 is a species-illegal move
    (the "Doom Desire on Garchomp" failure we are trying to suppress).
  * **conditional accuracy** — accuracy restricted to states where the top-1 IS legal.
  * **worked example** — for a known species (e.g. Garchomp) the top-8 predicted actions
    with their probabilities, tagged legal/illegal, to eyeball the prior.

Usage:
  python bot/eval_imitation.py [--n 50000] [--species Garchomp]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from obs import OBS_DIM
from action_space import get_action_space
from imitate import DATASET_PATH, build_actor, UNIFIED_ACTION_SIZE

SWITCH_IDS = set(range(get_action_space().SWITCH_BASE,
                       get_action_space().SWITCH_BASE + get_action_space().N_ACTIONS
                       - get_action_space().SWITCH_BASE - 3))  # 6 switch actions


def legal_action_ids(as_, species: str, gen: int) -> set:
    """Legal action ids = species-legal move ids + all 6 switch slots."""
    ids = set(SWITCH_IDS)
    for m in as_.legal_moves(species, gen):
        mid = as_.move_id(m)
        if mid is not None:
            ids.add(mid)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000, help="how many samples to evaluate")
    ap.add_argument("--species", type=str, default="Garchomp", help="species for the worked example")
    ap.add_argument("--model", type=str, default=os.path.join(os.path.dirname(DATASET_PATH), "imitation_actor.pt"))
    args = ap.parse_args()

    as_ = get_action_space()

    print("Loading dataset + model...")
    data = np.load(DATASET_PATH)
    states, actions, species, gens = data["states"], data["actions"], data["species"], data["gens"]
    n = min(args.n, len(states))
    states, actions, species, gens = states[:n], actions[:n], species[:n], gens[:n]
    print(f"  {n} samples, {states.shape[1]} dims")

    actor = build_actor(OBS_DIM, UNIFIED_ACTION_SIZE)
    sd = torch.load(args.model, map_location="cpu")
    actor.load_state_dict(sd)
    actor.eval()
    print(f"  model: {args.model}")

    # ---- build per-sample legal masks (vectorized over the eval slice) ----
    print("Building legal masks...")
    masks = np.zeros((n, UNIFIED_ACTION_SIZE), dtype=bool)
    cache = {}
    for i in range(n):
        sp = str(species[i])
        if sp == "":
            continue
        key = (sp, int(gens[i]))
        if key not in cache:
            cache[key] = legal_action_ids(as_, sp, int(gens[i]))
        for a in cache[key]:
            masks[i, a] = True
    n_with_mask = int((masks.sum(axis=1) > 0).sum())
    print(f"  {n_with_mask}/{n} samples have a legal mask ({100*n_with_mask/n:.1f}%)")

    # ---- forward pass in chunks ----
    print("Running forward pass...")
    with torch.no_grad():
        s_t = torch.from_numpy(states)
        logits = torch.cat([actor(s_t[i:i + 4096]) for i in range(0, n, 4096)], dim=0)
    probs = torch.softmax(logits, dim=-1).numpy()
    top1 = probs.argmax(axis=1)

    # ---- metrics ----
    m = masks[:n]
    has = m.sum(axis=1) > 0
    top1_legal = np.where(has, (m[np.arange(n), top1]), False)
    masked_acc = top1_legal.mean()
    illegal_rate = (1 - top1_legal[has]).mean() if has.any() else float("nan")
    cond_acc = (top1[has] == actions[has]).mean() if has.any() else float("nan")
    raw_top1 = (top1 == actions).mean()

    print("\n" + "=" * 60)
    print("IMITATION EVALUATION")
    print("=" * 60)
    print(f"  raw top-1 (963-way, unmasked)   : {100*raw_top1:5.1f}%")
    print(f"  MASKED top-1 (top-1 is legal)   : {100*masked_acc:5.1f}%   <- main metric")
    print(f"  illegal-argmax rate (masked set): {100*illegal_rate:5.1f}%   (lower = better prior)")
    print(f"  conditional top-1 (given legal) : {100*cond_acc:5.1f}%")

    # ---- worked example: top-8 predictions for a known species ----
    sp_idx = np.where((species == args.species) & has)[0]
    if len(sp_idx):
        i = int(sp_idx[len(sp_idx) // 2])
        legal = set(cache[(args.species, int(gens[i]))])
        order = np.argsort(-probs[i])[:8]
        print(f"\n  Worked example — {args.species} (gen{gens[i]}), sample {i}:")
        print(f"  replay action = {actions[i]} ({as_.id_to_move[actions[i]] if actions[i] < as_.SWITCH_BASE else 'switch/transform'})")
        for a in order:
            a = int(a)
            name = as_.id_to_move[a] if a < as_.SWITCH_BASE else ("switch" if a < as_.TRANSFORM_BASE else "transform")
            tag = "LEGAL " if a in legal else "ILLEGAL"
            print(f"    p={probs[i, a]:.3f}  {name:22s} {tag}")

    # ---- per-format masked accuracy ----
    print("\n  (Per-format breakdown available if you add a --format filter.)")


if __name__ == "__main__":
    main()
