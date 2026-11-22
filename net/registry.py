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
    channels (``opp_post_per_slot_*`` / ``opp_topmoves_per_slot`` / per-slot
    ``confidence``) — rule 4 (no leakage); since Step 3.5 the posterior is
    PER-SLOT (every bench mon's ability/move threat is visible, not just the
    active one — the "Gengar is in the back" signal the net needs to plan
    around);
  * D60 coupling: the current legal-action list is emitted as indexed inputs
    (``action_*``) so the net sees what each action position IS;
  * Step 3 (review 2026-11-22): STATIC mon/move data (types, base stats, move
    power/type/category/priority/accuracy/effect flags) as ADDITIONAL float
    blocks next to the learned id embeddings — the type chart and move facts
    the net used to have to discover from a +1/-1 reward 50 turns away.
    Precomputed by ``net.static_features`` (legality DB) and injected into the
    pure encoder, so the encoder stays numpy/DB-free.
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

# ── Step 3 static features (review 2026-11-22, items a/c) ────────────────
#: the 18 standard types in a FIXED order (one-hot basis for `*_types`).
TYPES: Tuple[str, ...] = (
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting",
    "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost",
    "Dragon", "Dark", "Steel", "Fairy",
)
N_TYPES: int = len(TYPES)
#: the six base-stat axes in a FIXED order (normalised by /255).
STAT_AXES: Tuple[str, ...] = ("hp", "atk", "def", "spa", "spd", "spe")
N_STATS: int = len(STAT_AXES)

#: per-action static move feature width (Step 3.1): 5 scaled values + 6 flags.
#:   [0] power/200 (0 for status/special)  [1] type_idx/N_TYPES (0 = none)
#:   [2] category: physical=0.0, special=0.5, status=1.0
#:   [3] priority: (p+6)/12  ([4] accuracy/100 (None = always hits -> 1.0)
#:   [5..10] flags: is_hazard, is_screen, has_boosts, side_status,
#:           side_condition, is_tailwind
K_MOVE_STATIC: int = 11

#: per-ACTION DAMAGE feature width (Step 3.3): the calc-backed belief block —
#: "will this action KO / will I be KO'd / who is faster" per legal action.
#:   [0] min_pct  [1] max_pct  [2] exp_pct  [3] can_ko  [4] guar_ko
#:   [5] exp_ko   [6] outspeed (1.0 faster / 0.5 tie / 0.0 slower)  [7] immune
#:   [8] dmg_in   (switch: expected dmg the incoming mon takes from the opp top move)
#: Move/tera rows and switch rows share these 9 dims (see OPEN_DESIGN_CHOICES.md
#: "3.3 DESIGN"); the net learns the per-kind meaning from action_kind.
K_ACTION_DAMAGE: int = 9

#: per-OWN-MON MATCHUP feature width (Step 3.4): how THIS mon fares against the
#: whole enemy team — the board-level damage landscape for win-condition /
#: switch / threat planning. All dims bounded in [0,1] (pcts/100, /6 fractions,
#: flags). See OPEN_DESIGN_CHOICES.md "3.4 DESIGN" for the locked schema.
#:   [0] best_hit  [1] worst_hit  [2] ko_guar_frac  [3] ko_can_frac
#:   [4] taken_ko_frac  [5] survives_strongest  [6] net_ko_margin  [7] best_vs_worst
K_OWN_MATCHUP: int = 8


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
        # the seven CURATED per-mon volatiles (D61; belief Facts.volatiles)
        "vol_disguise",      # Imposter one-hit shield still unbroken
        "vol_taunt",         # move lock (4 turns)
        "vol_encore",        # forced move repeat
        "vol_trapped",       # Arena Trap / Shadow Tag
        "vol_curse",         # the Curse-move Atk+ Def+ Spe- volatile
        "vol_destinybond",   # faint-on-hit retaliation arm
        "vol_gastroacid",    # move lock (5 turns)
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
    # Step 3.1/3.2 static features (precomputed by net.static_features, the
    # encoder zero-fills when absent — keeps the encoder pure/DB-free).
    # per-action static move data (id embeddings stay; these are ADDITIONAL)
    f.append(Field("action_move_static", (MAX_ACTIONS, K_MOVE_STATIC), "float32"))
    # per-ACTION damage/KO/speed block (Step 3.3) — precomputed by
    # net.action_features (the ONE calc pipeline); the encoder zero-fills when
    # absent. The belief facts the net needs to plan, per legal action.
    f.append(Field("action_damage", (MAX_ACTIONS, K_ACTION_DAMAGE), "float32"))
    # per-OWN-MON MATCHUP summary (Step 3.4): how each own mon fares against the
    # whole enemy team (best/worst hit, KO counts both ways, net margin) — the
    # board-level coverage landscape for win-condition / switch planning. Lives
    # PER-MON (folded into the OWN-side mon encoding, the 3.5 lesson), NOT the
    # global context. Precomputed by net.matchup_features (the ONE calc batch);
    # the encoder zero-fills when absent.
    f.append(Field("own_matchup", (MAX_SLOTS, K_OWN_MATCHUP), "float32"))
    # per-slot static mon data (dual types = 0.5/0.5; base stats /255)
    f.append(Field("own_types", (MAX_SLOTS, N_TYPES), "float32"))
    f.append(Field("opp_types", (MAX_SLOTS, N_TYPES), "float32"))
    f.append(Field("own_stats", (MAX_SLOTS, N_STATS), "float32"))
    f.append(Field("opp_stats", (MAX_SLOTS, N_STATS), "float32"))
    # opponent UNREVEALED info — ONLY via the posterior channels (rule 4).
    # Step 3.5: PER-SLOT (MAX_SLOTS x K) so every bench mon's ability/move
    # threat is visible, not just the active one. Slot 0 == the old active-only
    # value (a strict superset — no signal lost, bench added).
    f.append(Field("opp_post_per_slot_ids", (MAX_SLOTS, MAX_POST), "int64"))
    f.append(Field("opp_post_per_slot_w", (MAX_SLOTS, MAX_POST), "float32"))
    f.append(Field("opp_topmoves_per_slot", (MAX_SLOTS, MAX_TOP_MOVES), "int64"))
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
