"""S4.2 — corpus BattleTrace -> model batches (BC sample construction).

Bridges the S4.0 corpus (``context_seq`` + ``decisions``) to the S4.2 net's
batch format. One SAMPLE = one (p1-POV) decision:

  * sequence  = ``context_seq[0 : t]`` (the turn-1..t turn-start snapshots; the
    decision is made at the start of turn ``t``);
  * actions   = the decision's LEGAL action list (D60) — see
    ``decision_actions``; the net's policy head emits logits over EXACTLY this
    list and the target is the human's action index inside it;
  * target    = that index (always a legal position — the human's action is in
    the list by construction, so the BC loss is well-defined).

POV: the encoder is p1-POV ("own" = p1), so S4.2 samples are p1-POV decisions.
p2-POV needs the p2 bench populated as "own" (open decision — the D68
opponent-roster question); the net itself is POV-agnostic (it reads "own" as
self), so p2-POV samples drop in once the corpus supplies them.

Action-list construction (recorded design choice, S4.2 + D70):
  * move decision:  [the mon's revealed moves (used so far) + the human's
    move] sorted, then [alive benched own mons] sorted (switch options).
    The human's move is ALWAYS in the list (that's what makes the target
    legal).
  * switch decision: [alive benched own mons] sorted (you must switch; moves
    are not legal here — the structural D60 forced-decision case).
  * D70 ORDER SHUFFLE: the final list is then PERMUTED with a deterministic
    per-decision seed (battle, turn, kind, action) and the target is re-based
    to the human's action in the shuffled order. Rationale: the sorted order
    is a POSITIONAL ARTIFACT — index 0 is always "alphabetically first",
    and the human target sits there ~50% of the time (measured), so a fixed
    "always slot 0" net clears the chance gate without learning (the fake
    S4.3 PASS). In the real game a player's menu order is meaningless, so
    the net must select by action IDENTITY (its embeddings), not position.
    The shuffle is uniform over positions, which also makes the constant-
    index baseline honest (~1/|list| = chance).

Capped at the registry's MAX_ACTIONS (12) — the list is built to fit.
"""
from __future__ import annotations

import random as _random
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from corpus.trajectories import BattleTrace, Decision
from net.encoder import encode_context, encode_sequence
from net.registry import LAYOUT, MAX_ACTIONS
from net.vocab import Vocabs


@dataclass
class Sample:
    """One BC sample: an encoded sequence + the target action index."""
    battle_id: str
    turn: int
    kind: str
    target: int
    actions: List[Dict]          # the D60 legal-action list (dicts)
    seq: List[Dict]             # encoded contexts (len = turn), each registry-shaped


def _alive_bench(ctx: Dict) -> List[str]:
    """Own benched mons that are ALIVE (not active, not fainted) — the switch
    options at this state. Sorted species ids (deterministic).

    Liveness is the ``fainted`` flag, NOT ``hp_frac``: the own side only tracks
    the ACTIVE mon's facts, so every benched own mon carries ``hp_frac=0.0`` in
    the context (benched, untouched). A benched mon is switchable iff it has
    not fainted — requiring ``hp_frac>0`` (the old code) excluded every bench
    mon and degenerated every decision to a single legal action."""
    out = []
    for blk in (ctx.get("own", {}) or {}).values():
        if blk.get("active") or blk.get("fainted"):
            continue
        name = blk.get("name", "")
        if name:
            out.append(name)
    return sorted(out)


def _revealed_moves(ctx: Dict) -> List[str]:
    """The acting mon's moves revealed so far (used in battle) — the own block
    of the ACTIVE mon carries ``moves`` (the belief's moves_used)."""
    active = (ctx.get("active", {}) or {}).get("p1")
    blk = (ctx.get("own", {}) or {}).get(active) if active else None
    return list((blk or {}).get("moves", []) or [])


def _stable_seed(*parts: object) -> int:
    """A process-stable 32-bit seed. (Builtin ``hash()`` is salted per
    process and would make the shuffle order non-reproducible across runs.)"""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0xFFFFFFFF


def _shuffle_actions(acts: List[Dict], target: int,
                     seed: Optional[int]) -> Tuple[List[Dict], int]:
    """D70: permute the action list (deterministic per-decision seed) and re-base
    the target to the human's action in the shuffled order. Removes the
    alphabetical positional prior (index 0 = 'alphabetically first') that let
    a fixed-action net clear the chance gate without learning."""
    if seed is None or len(acts) < 2:
        return acts, target
    idx = list(range(len(acts)))
    _random.Random(seed).shuffle(idx)
    return [acts[i] for i in idx], idx.index(target)


def decision_actions(ctx: Dict, dec: Decision,
                     order_seed: Optional[int] = None) -> Tuple[List[Dict], int]:
    """The decision's D60 action list + the human's target index in it.

    ``ctx`` is the FINAL (turn-t) context snapshot — the state at which the
    decision is made. Returns ``(actions, target)`` with ``target`` a legal
    position (the human's action is in the list by construction).

    ``order_seed`` (D70): when given, the final list is permuted with that
    seed and the target re-based — see :func:`_shuffle_actions`. ``build_samples``
    always passes one; direct calls without a seed keep the sorted order.
    """
    if dec.kind == "switch":
        bench = _alive_bench(ctx)[:MAX_ACTIONS]
        acts = [{"kind": "switch", "species": s} for s in bench]
        from net.encoder import _to_id
        want = _to_id(dec.action)
        for i, s in enumerate(bench):
            if _to_id(s) == want:
                return _shuffle_actions(acts, i, order_seed)
        # the switched-in mon is NOT in the alive bench (should not happen — a
        # post-faint switch-in comes from a live bench); keep it legal anyway
        acts.append({"kind": "switch", "species": dec.action})
        return _shuffle_actions(acts, len(acts) - 1, order_seed)

    # move decision
    from net.encoder import _to_id
    want = _to_id(dec.action)
    moves = []
    for m in list(_revealed_moves(ctx)) + [dec.action]:
        mid = _to_id(m)
        if mid and mid not in moves:
            moves.append(mid)
    moves = sorted(moves)[:MAX_ACTIONS]
    acts = [{"kind": "move", "move": m} for m in moves]
    target = moves.index(want)
    # pad with switch options (they are legal at a move decision: the player
    # could switch instead)
    for s in _alive_bench(ctx):
        if len(acts) >= MAX_ACTIONS:
            break
        acts.append({"kind": "switch", "species": s})
    return _shuffle_actions(acts, target, order_seed)


def build_samples(trace: BattleTrace, vocabs: Vocabs,
                  max_decisions: int = 40) -> List[Sample]:
    """All p1-POV, legal, NON-DEGENERATE decisions of one battle -> BC samples.

    Only replayed battles (``skipped is None``) and only decisions the corpus
    already verified legal (``dec.legal``) — illegal human actions (the ~0.4%)
    are not learnable and are excluded, matching the S4.0 gate's legality bar.

    D70(B) — degenerate decisions are DROPPED: a decision with exactly ONE
    option of the target's kind (one revealed move / one benched mon) puts
    the target at a constant index and carries no selection signal (it only
    trains the action-type prior, which the switch-rate gate already enforces).
    This is the honest bar given the corpus limit: public Showdown replays
    carry NO ``|request|`` (verified 2026-11-18 against the PS replay API and
    1,000 metamon-raw-replays gen9ou rows), so the full 4-move set is
    unrecoverable and the action list = revealed moves + the human's move.
    """
    if trace.skipped is not None:
        return []
    seq_ctx = trace.context_seq
    samples: List[Sample] = []
    for dec in trace.decisions:
        if dec.pov != "p1" or not dec.legal:
            continue
        if dec.turn < 1 or dec.turn > len(seq_ctx):
            continue  # no turn-start snapshot for this decision
        ctx_final = seq_ctx[dec.turn - 1]
        # D70: deterministic per-decision order shuffle (identity-based, not
        # positional, action selection).
        order_seed = _stable_seed(trace.battle_id, dec.turn, dec.kind, dec.action)
        actions, target = decision_actions(ctx_final, dec, order_seed=order_seed)
        if target >= len(actions) or target >= MAX_ACTIONS:
            continue
        # D70(B): keep only decisions with a real choice of the target's kind
        # (>= 2 options), so the target index VARIES and the net has to select.
        n_kind = sum(1 for a in actions if a.get("kind") == dec.kind)
        if n_kind < 2:
            continue
        seq = [encode_context(ctx, vocabs, actions) for ctx in seq_ctx[:dec.turn]]
        samples.append(Sample(
            battle_id=trace.battle_id, turn=dec.turn, kind=dec.kind,
            target=target, actions=actions, seq=seq,
        ))
        if len(samples) >= max_decisions:
            break
    return samples


def samples_to_batch(samples: List[Sample]) -> Dict[str, object]:
    """Pad a list of samples to one model batch.

    Returns ``{"batch": {registry fields as (B,T,*) + battle_mask (B,T)},
    "targets": np.ndarray (B,), "ts": np.ndarray (B,)}`` where ``ts[i]`` is the
    0-indexed timestep to supervise (each sample's LAST real timestep).
    """
    if not samples:
        raise ValueError("samples_to_batch: empty sample list")
    B = len(samples)
    T = max(len(s.seq) for s in samples)
    dtype_map = {"float32": np.float32, "int64": np.int64}
    batch: Dict[str, np.ndarray] = {}
    for fld in LAYOUT:
        dt = dtype_map[fld.dtype]
        arr = np.zeros((B, T) + tuple(fld.shape), dtype=dt)
        for i, s in enumerate(samples):
            n = len(s.seq)
            # LEFT-aligned: real timesteps at [0, n), padding at the tail.
            # The GRU reads left->right, so the state at any real timestep
            # depends only on real data (tail padding can't leak backwards).
            if n:
                arr[i, :n] = np.stack([s.seq[j][fld.name] for j in range(n)])
        batch[fld.name] = arr
    # battle_mask: 1.0 = real timestep, 0.0 = tail padding (per sample).
    mask = np.zeros((B, T), dtype=np.float32)
    for i, s in enumerate(samples):
        mask[i, :len(s.seq)] = 1.0
    batch["battle_mask"] = mask
    # ts = each sample's LAST real timestep (left-aligned: index n-1).
    ts = np.array([len(s.seq) - 1 for s in samples], dtype=np.int64)
    targets = np.array([s.target for s in samples], dtype=np.int64)
    return {"batch": batch, "targets": targets, "ts": ts}


def batch_to_torch(batch: Dict[str, np.ndarray], device: str = "cpu") -> Dict[str, "object"]:
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device)
            for k, v in batch.items()}
