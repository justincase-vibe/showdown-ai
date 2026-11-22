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
    the output ONLY through the posterior channels (``opp_post_per_slot_*`` and
    ``opp_topmoves_per_slot`` — per-slot since Step 3.5 — and the per-slot
    ``confidence``); a raw opponent ability / item id is never emitted;
  * **permutation-friendly** — a bench swap is a pure row permutation of
    ``own_slot`` / ``own_*`` (the S4.2 set-encoder is then permutation-
    invariant); the encoder itself just preserves the canonical slot order.

Done (D61 surface, user-approved 2026-11-17):
  * the 7 curated volatiles (Disguise / Taunt / Encore / Trap / Curse /
    Destiny Bond / Gastro Acid) — ``belief`` Facts.volatiles -> the
    ``vol_*`` slot columns (see ``_slot_vector``); Stockpile + Protect are
    their own columns (layer count / per-turn flag).
  * the per-slot ``fainted`` flag — ``belief`` ``fainted_slots`` -> the
    ``fainted`` column: 1.0 = the mon on this slot is GONE, 0.0 = alive or
    benched-untouched (both read hp_frac 0; this column is what tells them
    apart). See ``_slot_vector``.

Still NOT done (deliberately, flagged for the user):
  * the G1–G6 additions (tailwind/weather/terrain turns-left, item+ability
    ``revealed?`` flags, own per-move PP, static dex join) are NOT yet in
    ``Facts`` / ``get_context()`` — a ``belief/`` change that is a later
    recorded decision, not part of this checkpoint.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import re

import numpy as np

from .registry import (
    BOOSTS,
    DTYPES,
    GLOBAL_INDEX,
    LAYOUT,
    MAX_ACTIONS,
    MAX_OWN_MOVES,
    MAX_POST,
    MAX_SLOTS,
    MAX_TOP_MOVES,
    SHAPES,
    SLOT_INDEX,
    STATUS_FLAGS,
)
from .vocab import Vocabs

#: D60 action kinds -> fixed int indices (0 = <pad>). Small fixed set, so we
#: hard-code it rather than route it through a Vocab.
ACTION_KIND_IDX: Dict[str, int] = {"move": 1, "switch": 2, "tera": 3, "transform": 4}

STATUS_COL: Dict[str, int] = {s: SLOT_INDEX[f"st_{s}"] for s in STATUS_FLAGS}
BOOST_COL: Dict[str, int] = {b: SLOT_INDEX[f"boost_{b}"] for b in BOOSTS}

#: The SIDE-KEYED field states the observation reads as ``{p1, p2}`` — the single
#: source of truth for "which field states are per-side." BOTH the encoder's
#: global reads AND the M2b p2-POV flip (``world_model._flip_field``) derive
#: from these, so the flip can never silently miss a field the net observes
#: (a hand-maintained parallel list inside the flip is the drift bug this kills).
SIDE_KEYED_FIELDS: Tuple[str, ...] = (
    "reflect", "lightscreen", "auroraveil", "stealthrock", "stickyweb", "safeguard",
)
#: Hazard layers carry a per-side COUNT (capped), not a 0/1 flag — same per-side
#: shape, different normalization. Named separately so the caps stay with the names.
HAZARD_FIELDS: Tuple[Tuple[str, float], ...] = (("spikes", 3.0), ("toxicspikes", 2.0))
#: Every per-side field name (screens/guards + hazards) — the exact set the flip swaps.
ALL_SIDE_KEYED_FIELDS: Tuple[str, ...] = (
    SIDE_KEYED_FIELDS + tuple(n for n, _ in HAZARD_FIELDS)
)


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
    # Per-slot fainted flag (belief `fainted_slots`): 1.0 = the mon currently
    # on this slot is GONE, 0.0 = it is alive OR this is a benched mon the
    # battle has not touched yet. Both cases read hp_frac 0 — this column is
    # what tells them apart (a benched mon is switchable, a fainted one is not).
    v[SLOT_INDEX["fainted"]] = 1.0 if block.get("fainted") else 0.0
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
    # curated per-mon volatiles (belief Facts.volatiles, toID'd names)
    volset = set(block.get("volatiles", []) or [])
    for vid in ("disguise", "taunt", "encore", "trapped",
                "curse", "destinybond", "gastroacid"):
        v[SLOT_INDEX[f"vol_{vid}"]] = 1.0 if vid in volset else 0.0
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
                   actions: Optional[List[Dict]] = None,
                   static_feats: Optional[Dict[str, np.ndarray]] = None
                   ) -> Dict[str, np.ndarray]:
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
    static_feats : optional PRECOMPUTED Step-3 static fields (see
               :mod:`net.static_features`) — the encoder stays pure/DB-free:
               it only PLACES the arrays it is given. ``None`` -> the five
               static fields are zero-filled (the net reads "unknown", the
               same degradation as the id channels).

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
    # Single source of truth for the per-side field set (see SIDE_KEYED_FIELDS /
    # HAZARD_FIELDS above) — the M2b POV flip (world_model._flip_field) uses the
    # SAME tuples, so the two cannot drift apart.
    for name in SIDE_KEYED_FIELDS:
        for side in ("p1", "p2"):
            d = field.get(name, {}) or {}
            g[GLOBAL_INDEX[f"{name}_{side}"]] = 1.0 if d.get(side) else 0.0
    for name, cap in HAZARD_FIELDS:
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
    # (rule 4 / no-leakage). Step 3.5: PER-SLOT — every opponent mon's ability
    # posterior + marginal top-moves are visible, not just the active one. This
    # is what lets the net see a bench threat ("a Gengar is in the back") and
    # plan around it. A raw opponent ability/id is STILL never emitted — only
    # the belief posterior (rule 4 held). Slot 0 == the old active-only value;
    # slots 1-5 are the (new) bench. Empty/bench slots with no posterior stay
    # zero rows (the net reads "unknown", like the id channels).
    post_ids = np.zeros((MAX_SLOTS, MAX_POST), dtype=np.int64)
    post_w = np.zeros((MAX_SLOTS, MAX_POST), dtype=np.float32)
    top_moves = np.zeros((MAX_SLOTS, MAX_TOP_MOVES), dtype=np.int64)
    for i, slot in enumerate(_slot_ids("p2")):
        blk = opp_blocks.get(slot)
        if not blk:
            continue
        posterior = blk.get("posterior", []) or []
        for k, cand in enumerate(posterior[:MAX_POST]):
            post_ids[i, k] = _idx(vocabs.ability, cand.get("ability", ""))
            post_w[i, k] = _clamp01(float(cand.get("weight", 0.0) or 0.0))
        for k, m in enumerate((blk.get("top_moves", []) or [])[:MAX_TOP_MOVES]):
            top_moves[i, k] = _idx(vocabs.move, m)
    out["opp_post_per_slot_ids"] = post_ids
    out["opp_post_per_slot_w"] = post_w
    out["opp_topmoves_per_slot"] = top_moves

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

    # ── Step 3 precomputed fields (zero-filled when absent) ────────────────
    # The encoder never touches the legality DB or the damage calc itself — it
    # only PLACES the arrays ``net.static_features`` / ``net.action_features``
    # produced, so the pure/numpy/DB-free contract holds and tests can inject
    # anything. ``static_feats`` is the union of both (built by env.build_obs).
    from .static_features import STATIC_FEATURE_NAMES
    from .action_features import ACTION_DAMAGE_NAMES
    for name in (*STATIC_FEATURE_NAMES, *ACTION_DAMAGE_NAMES):
        arr = (static_feats or {}).get(name)
        if arr is None:
            arr = np.zeros(SHAPES[name], dtype=DTYPES[name])
        out[name] = arr

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
