"""Evaluate the imitation (behavioral-cloning) actor — CORRECT masked metric.

Why the raw top-1 is misleading (and the trap we hit)
------------------------------------------------------
The actor was trained with a **hard legal-mask**: in the loss, every logit outside the
legal set is pinned to a large constant, so *only the legal logits* ever receive
gradient. The model's raw logits on **illegal** moves are therefore **unconstrained**
(they sit near their init values). If you argmax over the *full* 963-space, that
unconstrained noise routinely wins → a huge "illegal-argmax" number that looks like a
catastrophic failure but is a **metric artifact**: at *live* time the mask is ALWAYS
applied first (``policy.act`` does ``logits.masked_fill(~m, -inf)``; ``env.get_action_mask``
builds the legal set), so the model can never actually emit an illegal move.

The metric that matters
-----------------------
**Masked top-1 accuracy**: apply the per-sample legal mask to the logits (illegal → -inf),
argmax, and ask "did the model pick the expert's action among the legal options?" This
mirrors exactly what the model is trained under and what it does at live time. That is the
number to watch. The raw (unmasked) numbers are reported only as diagnostics.

The legal mask here **mirrors training** (``imitate.train``):
    legal = legal_moves(species, gen)  ∪  dataset_moves[species]  ∪  {replay_action}
i.e. the species-legal prior, augmented with the moves the data actually used for that
species (the learnset-gap fix, e.g. U-turn for Garchomp), plus the expert's own action
(always allowed = safety). We deliberately do NOT blanket-add the 6 switch ids: on a
move-decision sample switches were hard-masked out of training, so their logits are
untrained noise — including them in the mask lets that noise steal the argmax and *understate*
the true imitation quality. (A "switches-allowed" variant is printed for comparison.)

Usage:
  python bot/eval_imitation.py [--n 200000] [--species Garchomp]
  python bot/eval_imitation.py --model bot/data/imitation_actor_labelsmooth_only.pt   # A/B
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from obs import OBS_DIM
from action_space import get_action_space
from imitate import DATASET_PATH, build_actor, UNIFIED_ACTION_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000, help="how many samples to evaluate")
    ap.add_argument("--species", type=str, default="Garchomp", help="species for the worked example")
    ap.add_argument("--model", type=str, default=os.path.join(os.path.dirname(DATASET_PATH), "imitation_actor.pt"))
    args = ap.parse_args()

    as_ = get_action_space()
    N = as_.N_ACTIONS

    print("Loading dataset + model...")
    data = np.load(DATASET_PATH)
    states, actions = data["states"], data["actions"]
    species, gens = data["species"], data["gens"]
    n = min(args.n, len(states))
    states, actions, species, gens = states[:n], actions[:n], species[:n], gens[:n]
    print(f"  {n} samples, {states.shape[1]} dims")

    actor = build_actor(OBS_DIM, N)
    actor.load_state_dict(torch.load(args.model, map_location="cpu"))
    actor.eval()
    print(f"  model: {args.model}")

    # ---- dataset_moves: the learnset-gap augmentation (moves the data actually used) ----
    dataset_moves = defaultdict(set)
    for sp, a in zip(species, actions):
        if sp is not None and str(sp) != "":
            dataset_moves[str(sp)].add(int(a))

    # ---- per-sample legal masks (mirror imitate.train) ----
    #   legal = legal_moves(species, gen) ∪ dataset_moves[species] ∪ {replay_action}
    # Two masks are built:
    #   masks      : training-equivalent (move-legal set + the expert's own action). MAIN.
    #   masks_sw   : masks ∪ all 6 switch ids (the *old* eval's mask) — printed for contrast,
    #                showing how untrained switch logits inflate the "wrong" rate.
    print("Building legal masks...")
    masks = np.zeros((n, N), dtype=bool)
    masks_sw = np.zeros((n, N), dtype=bool)
    has_sp = np.zeros(n, dtype=bool)
    switch_ids = list(range(as_.SWITCH_BASE, as_.SWITCH_BASE + 6))
    cache = {}
    for i in range(n):
        sp = str(species[i])
        a = int(actions[i])
        if sp == "":
            # unknown species: training fell back to plain CE (full space). For the metric
            # we restrict to {replay_action} so argmax is well-defined; excluded from the
            # species-known headline below.
            masks[i, a] = True
            masks_sw[i, a] = True
            for s in switch_ids:
                masks_sw[i, s] = True
            continue
        key = (sp, int(gens[i]))
        if key not in cache:
            ids = {as_.move_id(m) for m in as_.legal_moves(sp, int(gens[i])) if as_.move_id(m) is not None}
            ids |= {x for x in dataset_moves.get(sp, set()) if x < as_.N_MOVES}
            cache[key] = ids
        ids = set(cache[key])
        ids.add(a)  # safety: the expert's action is always legal (mirrors training)
        for x in ids:
            masks[i, x] = True
        for s in switch_ids:
            masks_sw[i, s] = True
        has_sp[i] = True
    n_with = int(has_sp.sum())
    print(f"  {n_with}/{n} samples have a known species ({100*n_with/n:.1f}%)")

    # ---- forward pass (CPU chunks) ----
    print("Running forward pass...")
    with torch.no_grad():
        s_t = torch.from_numpy(states)
        logits = torch.cat([actor(s_t[i:i + 4096]) for i in range(0, n, 4096)], dim=0)
    lg = logits.numpy().astype(np.float64)

    def masked_top1_accuracy(m):
        ml = np.where(m, lg, -np.inf)          # illegal -> -inf; replay action always finite
        top1 = ml.argmax(axis=1)
        return top1, (top1 == actions).mean()

    # MAIN metric: training-equivalent legal mask
    mtop, masked_acc = masked_top1_accuracy(masks)
    masked_acc_sp = (mtop[has_sp] == actions[has_sp]).mean() if has_sp.any() else float("nan")

    # comparison variant: the old eval's mask (all 6 switches always allowed)
    _, masked_acc_sw = masked_top1_accuracy(masks_sw)

    # diagnostics: raw (unmasked) behavior
    raw_top1 = lg.argmax(axis=1)
    raw_acc = (raw_top1 == actions).mean()
    # true "Doom-Desire on Garchomp" rate: raw argmax outside (species-legal ∪ dataset)
    # (built WITHOUT force-adding the replay action, so a learnset-gap move still counts)
    raw_illegal = np.zeros(n, dtype=bool)
    for i in range(n):
        sp = str(species[i])
        if sp == "":
            continue
        ids = set(cache.get((sp, int(gens[i])), set()))
        if raw_top1[i] not in ids:
            raw_illegal[i] = True
    sp_known_idx = np.where(has_sp)[0]
    raw_illegal_rate = raw_illegal[sp_known_idx].mean() if len(sp_known_idx) else float("nan")

    # band breakdown of the MAIN masked metric
    move_band = actions < as_.SWITCH_BASE
    sw_band = (actions >= as_.SWITCH_BASE) & (actions < as_.TRANSFORM_BASE)
    tf_band = actions >= as_.TRANSFORM_BASE
    def acc_on(idx):
        idx = np.where(idx)[0]
        return (mtop[idx] == actions[idx]).mean() if len(idx) else float("nan")

    print("\n" + "=" * 64)
    print("IMITATION EVALUATION  (correct masked metric)")
    print("=" * 64)
    print(f"  MODEL masked top-1 acc (main)  : {100*masked_acc:5.1f}%   [all samples, training-legal mask]")
    print(f"    └ species-known subset       : {100*masked_acc_sp:5.1f}%   <- headline imitation quality")
    print(f"    └ move-decisions             : {100*acc_on(move_band):5.1f}%")
    print(f"    └ switch-decisions           : {100*acc_on(sw_band):5.1f}%   ({int(sw_band.sum())} samples)")
    print(f"    └ transform-decisions        : {100*acc_on(tf_band):5.1f}%   ({int(tf_band.sum())} samples)")
    print("-" * 64)
    print(f"  (contrast) masked top-1, +all-6-switches allowed : {100*masked_acc_sw:5.1f}%   [old-eval mask]")
    print("-" * 64)
    print("  DIAGNOSTICS (raw, unmasked — NOT the live behavior):")
    print(f"    raw top-1 (963-way)          : {100*raw_acc:5.1f}%")
    print(f"    raw illegal-argmax rate      : {100*raw_illegal_rate:5.1f}%   (artifact; masked away at live time)")

    # ---- worked example: masked top-8 for a known species ----
    sp_idx = np.where((species == args.species) & has_sp)[0]
    if len(sp_idx):
        i = int(sp_idx[len(sp_idx) // 2])
        legal = set(cache[(args.species, int(gens[i]))]) | {int(actions[i])}
        ml = np.where(masks[i], lg[i], -np.inf)
        order = np.argsort(-ml)[:8]
        print(f"\n  Worked example — {args.species} (gen{gens[i]}), sample {i}:")
        print(f"  replay action = {actions[i]} ({as_.id_to_move[actions[i]] if actions[i] < as_.SWITCH_BASE else 'switch/transform'})")
        print(f"  legal set size (mask) = {len(legal)}")
        for a in order:
            a = int(a)
            name = as_.id_to_move[a] if a < as_.SWITCH_BASE else ("switch" if a < as_.TRANSFORM_BASE else "transform")
            p = float(np.exp(rg) if (rg := ml[a]) > -np.inf else 0.0)
            print(f"    p={p:.3f}  {name:22s} {'(replay)' if a == int(actions[i]) else ''}")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
