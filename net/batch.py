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

Action-list construction (recorded design choice, S4.2 + D70 + D73 + D83):
  * move decision:  [the mon's WHOLE-battle move set (D83 — the AI always
    knows its own full moveset) ∪ revealed-so-far ∪ the human's move] padded
    to a full 4-move moveset with DECOYS (real Smogon set moves ranked by
    frequency, else the legal pool — D73) so the human's move is no longer the
    unique "new" action, then [alive benched own mons] sorted (switch options).
    The human's move is ALWAYS in the list (that's what makes the target legal).
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

Observation (recorded design choice, D85, user-approved): the own_moves
channel of every encoded context is widened to the mon's COMPLETE moveset —
whole-battle revealed moves + Smogon fillers up to 4 (the same D83 recipe as
the action list), NO "revealed" marker. With the old "revealed-so-far +
<pad>" row, the COUNT of revealed moves was itself a structural shortcut for
first-use vs repeat decisions (exactly what the D81/D82 leak gate measures);
the full set makes first-use and repeat states structurally identical — the
shortcut is removed at the source. The opponent side is deliberately
UNCHANGED: there, revealed-so-far IS the real public information.
"""
from __future__ import annotations

import random as _random
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from corpus.trajectories import BattleTrace, Decision
from net.encoder import encode_context, encode_sequence
from net.registry import LAYOUT, MAX_ACTIONS, MAX_OWN_MOVES
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
    # D81 (first-use/repeat leak check): True iff, at a MOVE decision, the acting
    # mon had NOT yet used this exact move earlier in the battle (it is a "new"
    # move for that mon). False = a repeat (the move is already in the mon's
    # revealed set). For switch decisions this is irrelevant (kept False). The
    # verdict's shortcut gate splits move-accuracy on this to catch a net that
    # just "grabs the new move" instead of learning state -> action.
    first_use: bool = False


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


REAL_MOVES = 4  # a singles moveset — the faithful action-list size (D73)


@lru_cache(maxsize=8192)
def _decoy_pool(species_id: str, fmt: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(ranked_real_set_moves, legal_pool) for a species in a format.

    Primary = the species' real Smogon set moves, ranked by how many sets use
    them (most-used first, then id) — a faithful proxy for the mon's actual
    4-move set. Fallback = the full legal pool (sorted by id). Deterministic
    DB reads; both empty on any error (graceful: no decoys).
    """
    try:
        import legality.lookup as lk
    except Exception:
        return (), ()
    ranked: List[str] = []
    try:
        freq: Dict[str, int] = {}
        for s in lk.get_common_sets(species_id, fmt):
            for m in s.get("moves", []):
                freq[m] = freq.get(m, 0) + 1
        ranked = sorted(freq, key=lambda m: (-freq[m], m))
    except Exception:
        ranked = []
    pool: Tuple[str, ...] = ()
    try:
        pool = tuple(sorted(lk.legal_moves(species_id, fmt)))
    except Exception:
        pool = ()
    return tuple(ranked), pool


def move_decoys(species_id: str, fmt: str, existing: List[str],
                k_total: int = REAL_MOVES) -> List[str]:
    """Up to ``k_total - len(existing)`` decoy move ids for a faithful full
    moveset action list (D73). Prefers real set moves, falls back to the legal
    pool; excludes ``existing`` (revealed moves + the human's move).
    Deterministic; empty when the species has no pool (graceful no-op)."""
    existing = set(existing)
    need = max(0, k_total - len(existing))
    if need == 0 or not species_id:
        return []
    ranked, pool = _decoy_pool(species_id, fmt)
    ranked_set = set(ranked)
    out: List[str] = []
    for m in list(ranked) + [p for p in pool if p not in ranked_set]:
        if m in existing or m in out:
            continue
        out.append(m)
        if len(out) >= need:
            break
    return out


def decision_actions(ctx: Dict, dec: Decision,
                     order_seed: Optional[int] = None,
                     fmt: str = "gen9ou",
                     full_moves: Optional[List[str]] = None) -> Tuple[List[Dict], int]:
    """The decision's D60 action list + the human's target index in it.

    ``ctx`` is the FINAL (turn-t) context snapshot — the state at which the
    decision is made. Returns ``(actions, target)`` with ``target`` a legal
    position (the human's action is in the list by construction).

    ``order_seed`` (D70): when given, the final list is permuted with that
    seed and the target re-based — see :func:`_shuffle_actions`. ``build_samples``
    always passes one; direct calls without a seed keep the sorted order.

    ``fmt`` (D73): the format id used to pad a move decision's action list to
    a full 4-move moveset with species-appropriate decoys — see
    :func:`move_decoys`.
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
    base = []
    # D83: prefer the mon's WHOLE-battle move set (the AI always knows its own
    # full moveset — it built the team); union with revealed-so-far so nothing
    # is lost; fall back to revealed-so-far alone when full_moves is not given
    # (direct calls / p2-POV). Decoys (D73) pad any short set to a full 4-move
    # moveset.
    revealed = list(_revealed_moves(ctx))
    source = (list(full_moves) + revealed) if full_moves is not None else revealed
    for m in source + [dec.action]:
        mid = _to_id(m)
        if mid and mid not in base:
            base.append(mid)
    # D73: pad to a full moveset with species-appropriate decoys so the human's
    # move is no longer the unique "new" action (corpus artifact: revealed ∪
    # target made the target the only unrevealed move -> a non-transferring
    # "grab the new one" prior).
    base = base + move_decoys(_to_id(dec.species) or "", fmt, base)
    moves = sorted(base)[:MAX_ACTIONS]
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
    from net.encoder import _to_id
    # D81: the mon's revealed-set-so-far (per species), in turn order, used to
    # label each move decision FIRST-USE vs REPEAT. Updated for every p1 legal
    # move decision (even ones later dropped), because the mon really did use
    # that move, so it is revealed for every later decision.
    seen_by_species: Dict[str, set] = {}
    # D83: the mon's WHOLE-battle move set (per species) — every move it used
    # across the battle, in one pass (no damage calcs). The AI always knows its
    # own full moveset (it built the team), so this is the faithful action-list
    # base (supersedes the "revealed so far" prefix); decoys (D73) pad a short
    # set to a full 4-move moveset.
    full_moves_by_species: Dict[str, set] = {}
    for dec in trace.decisions:
        if dec.pov != "p1" or not dec.legal or dec.kind != "move":
            continue
        sp = _to_id(dec.species) or ""
        mv = _to_id(dec.action) or ""
        if sp and mv:
            full_moves_by_species.setdefault(sp, set()).add(mv)

    # D85: the OBSERVATION's own_moves channel gets the same complete moveset
    # as the action list (whole-battle revealed + Smogon fillers to 4), no
    # "revealed" marker — see the module docstring. Computed per species,
    # deterministically (function of species + format only), and injected into
    # a shallow ctx copy at encode time (the encoder's pure contract is
    # untouched; the shared seq_ctx snapshots are never mutated).
    obs_cache: Dict[str, List[str]] = {}

    def _obs_moves(sp: str) -> List[str]:
        if sp not in obs_cache:
            base = sorted(full_moves_by_species.get(sp, ()))
            obs_cache[sp] = (base + move_decoys(sp, trace.formatid, base))[:MAX_OWN_MOVES]
        return obs_cache[sp]

    def _ctx_full_own(ctx: Dict) -> Dict:
        """``ctx`` with each own block's ``moves`` set to that mon's complete
        moveset (opponent side and everything else untouched)."""
        own = ctx.get("own") or {}
        if not own:
            return ctx
        new_own: Dict[str, Dict] = {}
        changed = False
        for slot, blk in own.items():
            sp = _to_id(blk.get("name") or blk.get("species") or "")
            if sp:
                full = _obs_moves(sp)
                if full and list(blk.get("moves") or []) != full:
                    blk = dict(blk)
                    blk["moves"] = list(full)
                    changed = True
            new_own[slot] = blk
        if not changed:
            return ctx
        return {**ctx, "own": new_own}

    for dec in trace.decisions:
        if dec.pov != "p1" or not dec.legal:
            continue
        if dec.turn < 1 or dec.turn > len(seq_ctx):
            continue  # no turn-start snapshot for this decision
        # D81: first-use vs repeat (the friend's leak-check signal). A move
        # decision is a REPEAT if the acting mon already used that exact move
        # earlier in the battle; otherwise FIRST-USE.
        first_use = True
        if dec.kind == "move":
            sp = _to_id(dec.species) or ""
            mv = _to_id(dec.action) or ""
            seen = seen_by_species.setdefault(sp, set())
            first_use = (mv not in seen)
            if mv:
                seen.add(mv)
        ctx_final = seq_ctx[dec.turn - 1]
        # D70: deterministic per-decision order shuffle (identity-based, not
        # positional, action selection).
        order_seed = _stable_seed(trace.battle_id, dec.turn, dec.kind, dec.action)
        _sp = _to_id(dec.species) or ""
        _full = sorted(full_moves_by_species.get(_sp, ()))
        actions, target = decision_actions(ctx_final, dec, order_seed=order_seed,
                                           fmt=trace.formatid, full_moves=_full)
        if target >= len(actions) or target >= MAX_ACTIONS:
            continue
        # D70(B): keep only decisions with a real choice of the target's kind
        # (>= 2 options), so the target index VARIES and the net has to select.
        n_kind = sum(1 for a in actions if a.get("kind") == dec.kind)
        if n_kind < 2:
            continue
        seq = [encode_context(_ctx_full_own(ctx), vocabs, actions)
               for ctx in seq_ctx[:dec.turn]]
        samples.append(Sample(
            battle_id=trace.battle_id, turn=dec.turn, kind=dec.kind,
            target=target, actions=actions, seq=seq, first_use=first_use,
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
