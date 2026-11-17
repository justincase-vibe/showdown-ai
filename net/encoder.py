"""S4.1 — observation encoder: ``get_context()`` (a dict) -> fixed tensors.

The encoder is the bridge between the belief stack's JSON-safe ``get_context()``
snapshot and the fixed float/int tensor layout declared by
:mod:`net.registry` (the single source of truth). It is PURE and DETERMINISTIC:
same context + same vocabs + same actions -> bitwise-identical tensors (the
S4.1 gate). It has no learned parameters and no network dependency (numpy only
— S4.2 converts the output to torch).

Contract (what it guarantees):
  * **fixed shapes** — every output key/shape/dtype comes from the registry;
    nothing is ad-hoc (no hand-maintained width, the D61 ``OBS_DIM`` bug class);
  * **deterministic** — pure function of its inputs (no randomness, no hashing
    of strings, no dict-iteration-order dependence: slots are read in the
    canonical a..f order);
  * **no leakage (rule 4)** — opponent UNREVEALED info (ability, item) reaches
    the output ONLY through the posterior channels (``opp_ability_post_*`` and
    ``opp_top_moves`` and the per-slot ``confidence``); a raw opponent ability /
    item id is never emitted;
  * **permutation-friendly** — a bench swap is a pure row permutation of
    ``own_slot`` / ``own_*`` (the S4.2 set-encoder is then permutation-
    invariant); the encoder itself just preserves the canonical slot order.

NOT done here (deliberately, and flagged for the user):
  * the 7 curated volatiles (Disguise / Taunt / Encore / Trap / Curse /
    Destiny Bond / Gastro Acid) and the G1–G6 additions (tailwind/weather/
    terrain turns-left, item+ability ``revealed?`` flags, own per-move PP,
    static dex join) are NOT yet in ``Facts`` / ``get_context()`` — adding them
    is a ``belief/`` change (331-baseline risk) that is the NEXT recorded
    decision, not part of this S4.1 checkpoint. The registry is shaped so they
    slot in as new fields without disturbing the existing layout.
  * the per-slot ``fainted`` flag is a RESERVED column, left ``0.0``. The
    present ``get_context()`` cannot distinguish "benched, untouched" from
    "fainted" (both read hp_frac 0), and deriving it from hp_frac emitted a
    FALSE fainted signal on untouched benched mons — a "wrong state" the net
    must never see. It therefore reads 0.0 until the dedicated ``belief/``
    flag lands (next decision); the column is reserved so that lands with NO
    shape change, and the raw evidence (hp_frac / active / moves / last_hit) is
    already in the block for the net to use. See ``_slot_vector``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import re

import numpy as np

from .registry import (
    BOOSTS,
    GLOBAL_INDEX,
    LAYOUT,
    MAX_ACTIONS,
    MAX_OWN_MOVES,
    MAX_POST,
    MAX_SLOTS,
    MAX_TOP_MOVES,
    SLOT_INDEX,
    STATUS_FLAGS,
)
from .vocab import Vocabs

#: D60 action kinds -> fixed int indices (0 = <pad>). Small fixed set, so we
#: hard-code it rather than route it through a Vocab.
ACTION_KIND_IDX: Dict[str, int] = {"move": 1, "switch": 2, "tera": 3, "transform": 4}

STATUS_COL: Dict[str, int] = {s: SLOT_INDEX[f"st_{s}"] for s in STATUS_FLAGS}
BOOST_COL: Dict[str, int] = {b: SLOT_INDEX[f"boost_{b}"] for b in BOOSTS}


def _to_id(name) -> str:
    """Showdown toID — mirrors ``legality.lookup.to_id`` /
    ``belief.log_events.to_id`` (lowercase alphanumerics only). Idempotent on an
    id, so it is ALWAYS safe to apply: it only rewrites a display name
    ("Landorus-Therian") to its id ("landorustherian"). The S4.2 vocab is built
    from the legality DB's ids, so every identity lookup here must be normalised
    the same way — otherwise a display-name own-team spec would silently OOV -> 0
    (bug #1). Kept local so net/ stays DB-free / numpy-only; the same 2-line
    function already lives in the two other modules (codebase convention, not a
    new source of truth)."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _idx(vocab, ident) -> int:
    """A vocab lookup that accepts an id OR a display name (normalised via
    ``_to_id``). Empty / None -> 0 (<pad>/OOV)."""
    return vocab.index(_to_id(ident))


def _slot_ids(side: str) -> List[str]:
    """Canonical slot order for a side: a, b, c, d, e, f (deterministic)."""
    return [f"{side}{c}" for c in "abcdef"]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _slot_vector(block: Dict, active: bool, is_own: bool) -> np.ndarray:
    """One uniform per-slot float block (own and opponent share the shape)."""
    v = np.zeros(_slot_dim(), dtype=np.float32)
    hp = float(block.get("hp_frac", 0.0) or 0.0)
    v[SLOT_INDEX["present"]] = 1.0
    v[SLOT_INDEX["active"]] = 1.0 if active else 0.0
    # fainted is a RESERVED per-slot flag, left 0.0 (see module docstring).
    # get_context() cannot yet distinguish "benched, untouched" from "fainted"
    # (both read hp_frac 0), so deriving it from hp_frac emitted a FALSE
    # fainted=1 on untouched benched mons — exactly the "wrong state" the net
    # must never see. The dedicated belief/Facts flag (open decision) will fill
    # this column without a shape change; the raw evidence (hp_frac, active,
    # moves, last_hit, ...) is already in the block for the net to use.
    v[SLOT_INDEX["fainted"]] = 0.0
    v[SLOT_INDEX["hp_frac"]] = _clamp01(hp)
    for s in block.get("statuses", []) or []:
        col = STATUS_COL.get(s)
        if col is not None:
            v[col] = 1.0
    for b, amt in (block.get("boosts", {}) or {}).items():
        col = BOOST_COL.get(b)
        if col is not None:
            # boosts range -6..+6 -> map to -1..+1 (NOT clamped to 0..1: a -3
            # boost must read -0.5, a +3 boost +0.5 — clamping to [0,1] would
            # destroy the sign, which is the whole signal).
            v[col] = max(-1.0, min(1.0, float(amt) / 6.0))
    v[SLOT_INDEX["tera_revealed"]] = 1.0 if block.get("tera") else 0.0
    v[SLOT_INDEX["last_hit_norm"]] = min(1.0, float(block.get("last_hit", 0) or 0) / 1000.0)
    v[SLOT_INDEX["hit_this_turn"]] = 1.0 if block.get("hit_this_turn") else 0.0
    v[SLOT_INDEX["last_move_failed"]] = 1.0 if block.get("last_move_failed") else 0.0
    v[SLOT_INDEX["stockpile"]] = min(3, int(block.get("stockpile", 0) or 0)) / 3.0
    v[SLOT_INDEX["used_protect"]] = 1.0 if block.get("used_protect") else 0.0
    v[SLOT_INDEX["confidence"]] = 1.0 if is_own else _clamp01(float(block.get("confidence", 0.0) or 0.0))
    return v


def _slot_dim() -> int:
    from .registry import SLOT_DIM
    return SLOT_DIM


def _side_blocks(ctx: Dict, side: str) -> Dict[str, Dict]:
    """The per-slot blocks for a side, keyed by slot id (missing -> {})."""
    key = "own" if side == "p1" else "opponent"
    return ctx.get(key, {}) or {}


def _id_row(ids: List[str], n: int, vocab) -> np.ndarray:
    out = np.zeros(n, dtype=np.int64)
    for i, ident in enumerate(ids[:n]):
        out[i] = vocab.index(_to_id(ident))
    return out


def _move_row(moves: List[str], vocab) -> np.ndarray:
    return _id_row(moves, MAX_OWN_MOVES, vocab.move)


def encode_context(ctx: Dict, vocabs: Vocabs,
                   actions: Optional[List[Dict]] = None) -> Dict[str, np.ndarray]:
    """Encode one ``get_context()`` snapshot into the registry's fixed tensors.

    Parameters
    ----------
    ctx      : the ``belief.state.BeliefState.get_context()`` dict (JSON-safe).
    vocabs   : the shared :class:`net.vocab.Vocabs` (S4.2 builds it from the
               legality DB; the SAME object feeds the embedding tables).
    actions  : optional D60 legal-action list — a list of dicts, each with
               ``kind`` in {move, switch, tera, transform} and optionally
               ``move`` (a move id) / ``species`` (a switch-in species id).
               The caller (env / S4.2) supplies the CURRENT legal list; the
               encoder only maps it to int channels. ``None``/empty -> a fully
               masked action block.

    Returns a ``dict[str, np.ndarray]`` whose keys/shapes/dtypes are EXACTLY
    the registry's ``LAYOUT``.
    """
    out: Dict[str, np.ndarray] = {}

    # ── global floats ────────────────────────────────────────────────
    g = np.zeros(len(GLOBAL_INDEX), dtype=np.float32)
    field = ctx.get("field", {}) or {}
    g[GLOBAL_INDEX["turn"]] = min(1.0, int(ctx.get("turn", 0) or 0) / 20.0)
    fainted = ctx.get("fainted", {}) or {}
    g[GLOBAL_INDEX["fainted_p1"]] = min(6, int(fainted.get("p1", 0) or 0)) / 6.0
    g[GLOBAL_INDEX["fainted_p2"]] = min(6, int(fainted.get("p2", 0) or 0)) / 6.0
    g[GLOBAL_INDEX["trickroom"]] = 1.0 if field.get("trickroom") else 0.0
    g[GLOBAL_INDEX["wonderroom"]] = 1.0 if field.get("wonderroom") else 0.0
    tw = field.get("tailwind_side", "") or ""
    g[GLOBAL_INDEX["tailwind_p1"]] = 1.0 if tw == "p1" else 0.0
    g[GLOBAL_INDEX["tailwind_p2"]] = 1.0 if tw == "p2" else 0.0
    for name, side in (("reflect", "p1"), ("reflect", "p2"),
                       ("lightscreen", "p1"), ("lightscreen", "p2"),
                       ("auroraveil", "p1"), ("auroraveil", "p2"),
                       ("stealthrock", "p1"), ("stealthrock", "p2"),
                       ("stickyweb", "p1"), ("stickyweb", "p2"),
                       ("safeguard", "p1"), ("safeguard", "p2")):
        d = field.get(name, {}) or {}
        g[GLOBAL_INDEX[f"{name}_{side}"]] = 1.0 if d.get(side) else 0.0
    for name, cap in (("spikes", 3.0), ("toxicspikes", 2.0)):
        d = field.get(name, {}) or {}
        for side in ("p1", "p2"):
            g[GLOBAL_INDEX[f"{name}_{side}"]] = _clamp01(float(d.get(side, 0) or 0) / cap)
    out["global"] = g

    # ── global id channels (embeddable) ──────────────────────────────
    out["global_weather"] = np.array([_idx(vocabs.weather, field.get("weather", ""))], dtype=np.int64)
    out["global_terrain"] = np.array([_idx(vocabs.terrain, field.get("terrain", ""))], dtype=np.int64)

    # ── per-slot uniform blocks (own p1 / opponent p2) ───────────────
    active = ctx.get("active", {}) or {}
    own_active = active.get("p1")
    opp_active = active.get("p2")

    own_blocks = _side_blocks(ctx, "p1")
    opp_blocks = _side_blocks(ctx, "p2")

    own_slot = np.zeros((MAX_SLOTS, _slot_dim()), dtype=np.float32)
    opp_slot = np.zeros((MAX_SLOTS, _slot_dim()), dtype=np.float32)
    own_species = np.zeros(MAX_SLOTS, dtype=np.int64)
    own_ability = np.zeros(MAX_SLOTS, dtype=np.int64)
    own_item = np.zeros(MAX_SLOTS, dtype=np.int64)
    own_moves = np.zeros((MAX_SLOTS, MAX_OWN_MOVES), dtype=np.int64)
    opp_species = np.zeros(MAX_SLOTS, dtype=np.int64)
    opp_moves = np.zeros((MAX_SLOTS, MAX_OWN_MOVES), dtype=np.int64)

    for i, slot in enumerate(_slot_ids("p1")):
        blk = own_blocks.get(slot)
        if not blk:
            continue
        own_slot[i] = _slot_vector(blk, active=(slot == own_active), is_own=True)
        own_species[i] = _idx(vocabs.species, blk.get("name") or blk.get("species") or "")
        own_ability[i] = _idx(vocabs.ability, blk.get("ability", ""))
        own_item[i] = _idx(vocabs.item, blk.get("item", ""))
        own_moves[i] = _move_row(blk.get("moves", []) or [], vocabs)

    for i, slot in enumerate(_slot_ids("p2")):
        blk = opp_blocks.get(slot)
        if not blk:
            continue
        opp_slot[i] = _slot_vector(blk, active=(slot == opp_active), is_own=False)
        opp_species[i] = _idx(vocabs.species, blk.get("species", ""))
        opp_moves[i] = _move_row(blk.get("moves", []) or [], vocabs)

    out["own_slot"] = own_slot
    out["opp_slot"] = opp_slot
    out["own_species"] = own_species
    out["own_ability"] = own_ability
    out["own_item"] = own_item
    out["own_moves"] = own_moves
    out["opp_species"] = opp_species
    out["opp_moves"] = opp_moves

    # ── opponent UNREVEALED info: ONLY via the posterior channels ────
    # (rule 4 / no-leakage). Tied to the ACTIVE opponent mon (the one we act
    # against); falls back to the first slot that carries a posterior.
    post_ids = np.zeros(MAX_POST, dtype=np.int64)
    post_w = np.zeros(MAX_POST, dtype=np.float32)
    top_moves = np.zeros(MAX_TOP_MOVES, dtype=np.int64)
    active_opp_blk = opp_blocks.get(opp_active) if opp_active else None
    if active_opp_blk is None:
        for slot in _slot_ids("p2"):
            if opp_blocks.get(slot):
                active_opp_blk = opp_blocks[slot]
                break
    if active_opp_blk is not None:
        posterior = active_opp_blk.get("posterior", []) or []
        for k, cand in enumerate(posterior[:MAX_POST]):
            post_ids[k] = _idx(vocabs.ability, cand.get("ability", ""))
            post_w[k] = _clamp01(float(cand.get("weight", 0.0) or 0.0))
        for k, m in enumerate((active_opp_blk.get("top_moves", []) or [])[:MAX_TOP_MOVES]):
            top_moves[k] = _idx(vocabs.move, m)
    out["opp_ability_post_ids"] = post_ids
    out["opp_ability_post_w"] = post_w
    out["opp_top_moves"] = top_moves

    # ── D60: the current legal-action list as indexed inputs ─────────
    actions = actions or []
    a_kind = np.zeros(MAX_ACTIONS, dtype=np.int64)
    a_move = np.zeros(MAX_ACTIONS, dtype=np.int64)
    a_species = np.zeros(MAX_ACTIONS, dtype=np.int64)
    a_mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
    for i, act in enumerate(actions[:MAX_ACTIONS]):
        kind = act.get("kind", "")
        a_kind[i] = ACTION_KIND_IDX.get(kind, 0)
        if act.get("move"):
            a_move[i] = _idx(vocabs.move, act["move"])
        if act.get("species"):
            a_species[i] = _idx(vocabs.species, act["species"])
        a_mask[i] = 1.0
    out["action_kind"] = a_kind
    out["action_move"] = a_move
    out["action_species"] = a_species
    out["action_mask"] = a_mask

    return out


def encode_sequence(context_seq: List[Dict], vocabs: Vocabs,
                    actions_seq: Optional[List[Optional[List[Dict]]]] = None
                    ) -> Dict[str, np.ndarray]:
    """Encode a battle's turn-start context sequence (the GRU input, S4.2).

    ``context_seq[i]`` is the turn-i+1 ``get_context()`` snapshot (from the
    S4.0 corpus). Produces, per field, a 3-D tensor ``(T, *field_shape)``
    where ``T = len(context_seq)``, plus a ``battle_mask`` of length ``T``
    (all ones here — real padding/masking is an S4.2 batching concern). This
    keeps S4.1's per-turn contract testable in isolation of the GRU.
    """
    t = len(context_seq)
    per_turn = [encode_context(ctx, vocabs,
                               (actions_seq[i] if actions_seq else None))
                for i, ctx in enumerate(context_seq)]
    # Derive keys/shapes/dtypes from the registry (single source of truth), NOT
    # from per_turn[0] — so an EMPTY sequence still emits every field with shape
    # (0, ...) and the layout can never drift between the encoder and the net.
    out: Dict[str, np.ndarray] = {}
    for fld in LAYOUT:
        stack = np.zeros((t,) + fld.shape, dtype=fld.dtype)
        for i, enc in enumerate(per_turn):
            stack[i] = enc[fld.name]
        out[fld.name] = stack
    out["battle_mask"] = np.ones(t, dtype=np.float32)
    return out
