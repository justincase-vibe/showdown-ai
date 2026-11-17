"""S4.1 — dim registry: the SINGLE SOURCE OF TRUTH for the observation layout.

PLAN.md §3c S4.1 / D61: "dim registry = the single source of truth for the
tensor layout (checksum-pinned drift-catcher test)".

This module declares, as data, exactly what the observation is: a flat, ordered
list of named :class:`Field`s, each with a FIXED ``shape`` and ``dtype``. The
encoder (:mod:`net.encoder`) fills exactly these fields; the S4.2 network reads
exactly these fields; a saved checkpoint's shapes must match these fields.

Drift-catcher: :func:`checksum` is a sha256 over the canonical layout. Adding,
removing, renaming, reshaping, or re-ordering ANY field changes the checksum,
and the pin test in ``net/tests`` FAILS — so the encoder, the net, and any
checkpoint can never silently disagree about the tensor layout. (This is the
direct answer to the old ``OBS_DIM`` over-count bug D61 calls out: the width is
now DERIVED from the registry, never hand-maintained.)

D61 structure encoded here:
  * each side = a SET of <=6 UNIFORM slot blocks (``own_slot`` / ``opp_slot``),
    with ``active`` / ``fainted`` as PER-SLOT FLAGS (column of the block), not
    block boundaries — singles = 1 active slot, doubles = 2, SAME shape;
  * own and opponent share ONE uniform block shape (differing only in
    known-vs-posterior content);
  * identity = int id channels (species/ability/item/move) for S4.2's learned
    embeddings — NOT one-hot, NOT the old lossy multi-probe hash;
  * the moveset-shape prior is CUT (redundant with the belief posterior);
  * opponent UNREVEALED info reaches the net ONLY through the posterior
    channels (``opp_ability_post_*`` / ``opp_top_moves`` / ``confidence``) —
    rule 4 (no leakage);
  * D60 coupling: the current legal-action list is emitted as indexed inputs
    (``action_*``) so the net sees what each action position IS.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ── fixed small vocabularies (numeric one-hots, NOT id-embedded) ──────────
#: the seven standard status conditions (Showdown ids).
STATUS_FLAGS: Tuple[str, ...] = ("brn", "par", "tox", "slp", "frz", "flinch", "conf")
#: the seven stat boost axes, in a FIXED order (the calc/boost key order).
BOOSTS: Tuple[str, ...] = ("atk", "def", "spa", "spd", "spe", "acc", "ev")

# ── fixed widths (doubles-robust: 6v6 singles => up to 6 slots per side) ──
MAX_SLOTS = 6          # a side has <=6 slots (D61: a SET of <=6 uniform blocks)
MAX_OWN_MOVES = 4      # a mon carries <=4 moves
MAX_POST = 5           # the belief posterior top-K (get_context exposes top-5)
MAX_TOP_MOVES = 4      # the opponent's marginal top-moves (get_context exposes 4)
MAX_ACTIONS = 12       # D60: the current legal-action list, padded to this width


@dataclass(frozen=True)
class Field:
    """One named region of the observation tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: str  # "float32" | "int64"


def _slot_features() -> Tuple[str, ...]:
    """The per-slot float feature order — ONE uniform shape for own AND
    opponent (D61). ``active``/``fainted`` are columns here (per-slot flags),
    never block boundaries."""
    feats: List[str] = ["present", "active", "fainted", "hp_frac"]
    feats += [f"st_{s}" for s in STATUS_FLAGS]          # status one-hot
    feats += [f"boost_{b}" for b in BOOSTS]             # boosts, each in [-1, 1]
    feats += [
        "tera_revealed",     # 1.0 iff a Tera type has been revealed on this mon
        "last_hit_norm",     # last move-hit damage / 1000, capped at 1.0
        "hit_this_turn",     # was hit this turn by a damaging move (D53 per-source)
        "last_move_failed",  # per-turn (fork moveLastTurnResult)
        "stockpile",         # 0..3 layers / 3 (the T5 volatile)
        "used_protect",      # per-turn Protect-family (doubles-critical)
        "confidence",        # own = 1.0; opponent = belief confidence in [0, 1]
    ]
    return tuple(feats)


SLOT_FEATURES: Tuple[str, ...] = _slot_features()
SLOT_DIM: int = len(SLOT_FEATURES)
SLOT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(SLOT_FEATURES)}


def _global_features() -> Tuple[str, ...]:
    """The global (battle-level) float features — fixed order."""
    return (
        "turn",                 # turn / 20, capped at 1.0
        "fainted_p1", "fainted_p2",   # faint count / 6
        "trickroom", "wonderroom",
        "tailwind_p1", "tailwind_p2",
        "reflect_p1", "reflect_p2",
        "lightscreen_p1", "lightscreen_p2",
        "auroraveil_p1", "auroraveil_p2",
        "stealthrock_p1", "stealthrock_p2",
        "spikes_p1", "spikes_p2",              # layers 0..3 / 3
        "toxicspikes_p1", "toxicspikes_p2",    # layers 0..2 / 2
        "stickyweb_p1", "stickyweb_p2",
        "safeguard_p1", "safeguard_p2",
    )


GLOBAL_FEATURES: Tuple[str, ...] = _global_features()
GLOBAL_DIM: int = len(GLOBAL_FEATURES)
GLOBAL_INDEX: Dict[str, int] = {name: i for i, name in enumerate(GLOBAL_FEATURES)}


def _layout() -> List[Field]:
    f: List[Field] = []
    # battle-level
    f.append(Field("global", (GLOBAL_DIM,), "float32"))
    f.append(Field("global_weather", (1,), "int64"))   # id index (embeddable)
    f.append(Field("global_terrain", (1,), "int64"))
    # per-slot uniform blocks (own + opponent share the SAME shape)
    f.append(Field("own_slot", (MAX_SLOTS, SLOT_DIM), "float32"))
    f.append(Field("opp_slot", (MAX_SLOTS, SLOT_DIM), "float32"))
    # identity id channels (int indices into the S4.2 embedding tables)
    f.append(Field("own_species", (MAX_SLOTS,), "int64"))
    f.append(Field("own_ability", (MAX_SLOTS,), "int64"))
    f.append(Field("own_item", (MAX_SLOTS,), "int64"))
    f.append(Field("own_moves", (MAX_SLOTS, MAX_OWN_MOVES), "int64"))
    f.append(Field("opp_species", (MAX_SLOTS,), "int64"))
    f.append(Field("opp_moves", (MAX_SLOTS, MAX_OWN_MOVES), "int64"))
    # opponent UNREVEALED info — ONLY via the posterior channels (rule 4)
    f.append(Field("opp_ability_post_ids", (MAX_POST,), "int64"))
    f.append(Field("opp_ability_post_w", (MAX_POST,), "float32"))
    f.append(Field("opp_top_moves", (MAX_TOP_MOVES,), "int64"))
    # D60: the current legal-action list as indexed inputs
    f.append(Field("action_kind", (MAX_ACTIONS,), "int64"))
    f.append(Field("action_move", (MAX_ACTIONS,), "int64"))
    f.append(Field("action_species", (MAX_ACTIONS,), "int64"))
    f.append(Field("action_mask", (MAX_ACTIONS,), "float32"))
    return f


#: the ordered layout — THE single source of truth.
LAYOUT: List[Field] = _layout()

#: name -> Field / shape / dtype (fast lookups).
FIELDS: Dict[str, Field] = {fld.name: fld for fld in LAYOUT}
SHAPES: Dict[str, Tuple[int, ...]] = {fld.name: fld.shape for fld in LAYOUT}
DTYPES: Dict[str, str] = {fld.name: fld.dtype for fld in LAYOUT}


def _canonical() -> str:
    """The canonical string the checksum is taken over (order-sensitive)."""
    return "\n".join(f"{fld.name}|{fld.shape}|{fld.dtype}" for fld in LAYOUT)


def checksum() -> str:
    """sha256 over the canonical layout — the drift-catcher pin.

    Any layout change (add/remove/rename/reshape/reorder) changes this, so the
    pin test in ``net/tests`` fails and forces the encoder + net + checkpoints
    to be reconciled deliberately."""
    return hashlib.sha256(_canonical().encode("utf-8")).hexdigest()


def total_elements() -> int:
    """Total number of scalar elements across all fields (sanity / logging)."""
    n = 0
    for fld in LAYOUT:
        prod = 1
        for s in fld.shape:
            prod *= s
        n += prod
    return n


def summary() -> str:
    """A human-readable layout dump (ASCII-safe for the cp1252 console)."""
    lines = [f"-- observation layout ({len(LAYOUT)} fields, {total_elements()} elements) "
             f"checksum={checksum()[:12]}"]
    for fld in LAYOUT:
        shape = "x".join(str(s) for s in fld.shape)
        lines.append(f"  {fld.name:<22} {shape:<12} {fld.dtype}")
    return "\n".join(lines)
