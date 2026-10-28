"""Battle -> fixed-length feature vector (v2, richer).

Design goals
------------
* **One shared policy across Gen 6-9.** Identity is encoded *neutrally* (name hashing +
  base-stat profile + type + ability), so a single net can play every generation and learn
  the "overlapping skills" (type advantage, sweep setup, hazards) rather than a gen-specific
  species table.
* **Learned opponent model, NOT a hard-coded prior.** We only encode what is *actually
  revealed/observable*: species identity, base-stat profile, ability, item, the semantics of
  the moves it has *shown*, and battle-state cues (recharge / protect / tera / dancing /
  volatile effects). We deliberately do **not** feed "this species usually runs X" as a known
  feature -- a static prior is exploited by unorthodox play; a learned model adapts to the tail
  (PLAN.md sections 3-4).
* **Robust to hidden info.** Every field is read through a safe accessor; missing/unknown -> 0.
* **Process-stable hashing.** Identity hashes use ``hashlib`` (deterministic across the 4 worker
  processes). Python's built-in ``hash()`` is salted per-process and would break the "same species
  -> same vector" invariant across workers, so it is never used here.

The layout is defined once below; ``OBS_DIM`` is the single source of truth (env.py / policy.py /
train.py import it). Unknowns zero-fill their block, and a defensive pad/truncate guarantees the
exact contract.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np

from poke_env.battle import AbstractBattle, Pokemon, PokemonType, Status, Weather
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.move_category import MoveCategory

try:  # volatile-effect cues (one-hot over a curated subset)
    from poke_env.battle.effect import Effect
except Exception:  # pragma: no cover - Effect should always exist; degrade to no cue effects
    Effect = None

# ---------------------------------------------------------------------------
# Buckets (order is the one-hot order)
# ---------------------------------------------------------------------------
TYPES: List[PokemonType] = [
    PokemonType.BUG,
    PokemonType.DARK,
    PokemonType.DRAGON,
    PokemonType.ELECTRIC,
    PokemonType.FAIRY,
    PokemonType.FIGHTING,
    PokemonType.FIRE,
    PokemonType.FLYING,
    PokemonType.GHOST,
    PokemonType.GRASS,
    PokemonType.GROUND,
    PokemonType.ICE,
    PokemonType.NORMAL,
    PokemonType.POISON,
    PokemonType.PSYCHIC,
    PokemonType.ROCK,
    PokemonType.STEEL,
    PokemonType.WATER,
]
N_TYPES = len(TYPES)

# Real (non-faint) statuses. Fainted is encoded separately.
STATUSES: List[Status] = [
    Status.BRN,
    Status.FRZ,
    Status.PAR,
    Status.PSN,
    Status.SLP,
    Status.TOX,
]
N_STATUS = len(STATUSES)

WEATHERS: List[Weather] = [
    Weather.DESOLATELAND,
    Weather.DELTASTREAM,
    Weather.HAIL,
    Weather.PRIMORDIALSEA,
    Weather.RAINDANCE,
    Weather.SANDSTORM,
    Weather.SNOW,
    Weather.SUNNYDAY,
]
N_WEATHER = len(WEATHERS) + 1  # + "no weather"

# Side conditions we track per side: entry hazards + protective screens/wind.
HAZARDS: List[SideCondition] = [
    SideCondition.STEALTH_ROCK,
    SideCondition.SPIKES,
    SideCondition.TOXIC_SPIKES,
    SideCondition.STICKY_WEB,
    SideCondition.REFLECT,
    SideCondition.LIGHT_SCREEN,
    SideCondition.TAILWIND,
    SideCondition.AURORA_VEIL,
]
N_HAZARD = len(HAZARDS)

# Move categories (real enum members -- a plain Enum, so compare to members, not strings).
CATEGORIES: List[MoveCategory] = [MoveCategory.PHYSICAL, MoveCategory.SPECIAL, MoveCategory.STATUS]
N_CAT = len(CATEGORIES)

# Base-stat profile (a species-level, gen-stable signal: tank vs. sweeper vs. wall).
BASE_STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
N_BASE = len(BASE_STATS)

# Boost stats (dynamic, per-battle).
STATS = ["atk", "def", "spa", "spd", "spe"]
N_STATS = len(STATS)

# Curated volatile-effect cues (only one-hotted when the member actually exists on the
# Effect enum, so this list may safely contain gen-specific effects).
CUE_EFFECTS: List[str] = [
    "CONFUSION",      # confused
    "ENCORE",         # move-locked to one move (read: locked Choice-like behavior)
    "DISABLE",        # best move locked
    "DESTINY_BOND",   # "takes you with me" pressure
    "CURSE",          # cursed -> drain
    "DISGUISE",       # shroud intact
    "GASTRO_ACID",    # opponent ability suppressed
    "TAUNT",          # status moves locked
    "TRICK",          # item swapped (reveals the held item!)
    "PARTIALLY_TRAPPED",  # phazing/trap (Firespin, etc.)
    "SNAP_TRAP",      # trapped
    "SILK_TRAP",      # trapped
]
N_CUE_EFFECTS = len(CUE_EFFECTS)

# Per-move semantics: power, accuracy, priority, drain, heal, recoil, inflicts-status,
# inflicts-volatile. These describe what a move *does* -- the "richer revealed-move
# semantics" upgrade -- with no external table.
N_MOVE_SEM = 8

N_MOVES = 4
N_TEAM_SLOTS = 6

# --- hashed identity sizes (neutral identity, not a moveset prior) ---
N_SPECIES_HASH = 16
N_ABILITY_HASH = 8
N_ITEM_HASH = 8
N_HASH_PROBES = 4  # multi-probe hashing: fewer collisions than a single bucket

# ---------------------------------------------------------------------------
# Block sizes (single source of truth for OBS_DIM)
# ---------------------------------------------------------------------------
MOVE_BLOCK = N_TYPES + N_CAT + N_MOVE_SEM + 1  # type, category, semantics, pp-fraction

# Active-mon cues: recharge, protect, tera, dancing, + curated cue effects.
_ACTIVE_CUES = 4 + N_CUE_EFFECTS  # = 16

# One active-mon block, shared by own (pp known) and opponent (pp hidden -> 0).
_ACTIVE = (
    1              # hp fraction
    + N_STATUS     # status
    + 1            # fainted
    + 1            # revealed
    + N_STATS      # dynamic boosts
    + N_SPECIES_HASH
    + N_BASE       # base-stat profile
    + N_TYPES      # type(s)
    + N_ABILITY_HASH
    + N_ITEM_HASH  # opponent hidden item -> consistent "unknown" hash
    + N_MOVES * MOVE_BLOCK
    + _ACTIVE_CUES
)

_OWN_SLOT = 1 + 1 + N_STATUS + N_SPECIES_HASH  # fainted, hp, status, identity
_OPP_SLOT = 1 + 1 + N_SPECIES_HASH + N_TYPES   # fainted, revealed, identity, type
_OWN_TEAM = N_TEAM_SLOTS * _OWN_SLOT
_OPP_TEAM = N_TEAM_SLOTS * _OPP_SLOT

_FIELD = (
    N_WEATHER
    + N_HAZARD          # our side
    + N_HAZARD          # opp side
    + 1                 # turn (normalized)
    + 1                 # trapped
    + 1                 # force switch
    + 1                 # can mega
    + 1                 # can z
    + 1                 # can dynamax
    + 1                 # can tera
    + 1                 # opponent has a revealed priority move (speed proxy)
    + 1                 # relative base speed (opp - own), a gen-stable speed proxy
)
_GLOBAL = 2  # own fainted fraction / opp fainted fraction

OBS_DIM = _ACTIVE * 2 + _OWN_TEAM + _OPP_TEAM + _FIELD + _GLOBAL


# ---------------------------------------------------------------------------
# Safe accessors + hashing
# ---------------------------------------------------------------------------
def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def _bool_attr(mon: Optional[Pokemon], name: str) -> bool:
    """Read a boolean-ish property if it exists for this gen, else False (gen-dependent
    properties like ``terastallized`` / ``dancing`` are absent in older gens)."""
    try:
        return bool(getattr(mon, name, None))
    except Exception:
        return False


def _hash_vec(name, dim: int, probes: int = N_HASH_PROBES) -> List[float]:
    """Process-stable multi-probe hash of a string into ``dim`` buckets.

    Sets ``probes`` buckets to ``1/probes`` (total magnitude 1.0), which dramatically reduces
    collisions versus a single bucket while staying a fixed, dense vector. Uses ``hashlib``
    (deterministic across worker processes) -- **not** Python's salted ``hash()``.
    """
    out = [0.0] * dim
    if name is None:
        return out
    s = str(name).strip().lower()
    if not s:
        return out
    for p in range(probes):
        h = int(hashlib.md5(f"{p}:{s}".encode("utf-8")).hexdigest(), 16)
        out[h % dim] += 1.0 / probes
    return out


def _hp_fraction(mon: Optional[Pokemon]) -> float:
    v = _safe(lambda: mon.current_hp_fraction, 0.0) if mon is not None else 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _status_onehot(mon: Optional[Pokemon]) -> List[float]:
    out = [0.0] * N_STATUS
    if mon is None:
        return out
    st = _safe(lambda: mon.status)
    if st is None:
        return out
    for i, s in enumerate(STATUSES):
        if st == s:
            out[i] = 1.0
            break
    return out


def _type_onehot(mon: Optional[Pokemon]) -> List[float]:
    out = [0.0] * N_TYPES
    if mon is None:
        return out
    for t in (_safe(lambda: list(mon.types), []) or []):
        if t in TYPES:
            out[TYPES.index(t)] = 1.0
    return out


def _boost_vec(mon: Optional[Pokemon]) -> List[float]:
    out = [0.0] * N_STATS
    if mon is None:
        return out
    boosts = _safe(lambda: mon.boosts, {}) or {}
    for i, stat in enumerate(STATS):
        try:
            out[i] = max(-1.0, min(1.0, float(boosts.get(stat, 0)) / 6.0))
        except Exception:
            out[i] = 0.0
    return out


def _base_stat_vec(mon: Optional[Pokemon]) -> List[float]:
    out = [0.0] * N_BASE
    if mon is None:
        return out
    bs = _safe(lambda: mon.base_stats, {}) or {}
    for i, k in enumerate(BASE_STATS):
        try:
            out[i] = min(float(bs.get(k, 0)), 255.0) / 255.0
        except Exception:
            out[i] = 0.0
    return out


def _cues_vec(mon: Optional[Pokemon]) -> List[float]:
    out = [0.0] * _ACTIVE_CUES
    if mon is None:
        return out
    out[0] = 1.0 if _safe(lambda: mon.must_recharge, False) else 0.0
    out[1] = 1.0 if _safe(lambda: mon.protect_counter, 0) else 0.0
    out[2] = 1.0 if _bool_attr(mon, "terastallized") else 0.0
    out[3] = 1.0 if _bool_attr(mon, "dancing") else 0.0
    effs = _safe(lambda: mon.effects, {}) or {}
    for i, name in enumerate(CUE_EFFECTS):
        e = None
        if Effect is not None:
            try:
                e = getattr(Effect, name, None)
            except Exception:
                e = None
        if e is not None and e in effs:
            out[4 + i] = 1.0
    return out


def _moves(mon: Optional[Pokemon]):
    if mon is None:
        return []
    mvs = _safe(lambda: list(mon.moves.values()), []) or []
    return mvs[:N_MOVES]


def _move_vec(mon: Optional[Pokemon], idx: int, include_pp: bool) -> List[float]:
    n = N_TYPES + N_CAT + N_MOVE_SEM + (1 if include_pp else 0)
    mvs = _moves(mon)
    if idx >= len(mvs) or mvs[idx] is None:
        return [0.0] * n
    m = mvs[idx]
    out: List[float] = []

    # type
    t = _safe(lambda: m.type)
    tvec = [0.0] * N_TYPES
    if t in TYPES:
        tvec[TYPES.index(t)] = 1.0
    out += tvec

    # category (compare to real enum members -- the old string compare silently failed)
    c = _safe(lambda: m.category)
    cvec = [0.0] * N_CAT
    if c in CATEGORIES:
        cvec[CATEGORIES.index(c)] = 1.0
    out += cvec

    # semantics: what this move does (power/acc/priority/drain/heal/recoil/status/volatile)
    def _f(getter):
        v = _safe(getter, 0.0)
        try:
            return float(v)
        except Exception:
            return 0.0

    out += [
        min(max(_f(lambda: m.base_power), 0.0), 120.0) / 120.0,
        min(max(_f(lambda: m.accuracy), 0.0), 1.0),
        max(-1.0, min(1.0, _f(lambda: m.priority) / 5.0)),
        min(max(_f(lambda: m.drain), 0.0), 1.0),
        min(max(_f(lambda: m.heal), 0.0), 1.0),
        min(max(_f(lambda: m.recoil), 0.0), 1.0),
        1.0 if _safe(lambda: m.status) is not None else 0.0,
        1.0 if _safe(lambda: m.volatile_status) is not None else 0.0,
    ]

    # pp fraction (own mon only; opponent hidden -> 0)
    if include_pp:
        cur = _safe(lambda: m.current_pp)
        mx = _safe(lambda: m.max_pp)
        p = 0.0
        if cur is not None and mx:
            try:
                p = max(0.0, min(1.0, float(cur) / float(mx)))
            except Exception:
                p = 0.0
        out.append(p)
    return out


def _moves_vec(mon: Optional[Pokemon], include_pp: bool) -> List[float]:
    return [x for i in range(N_MOVES) for x in _move_vec(mon, i, include_pp)]


def _weather_onehot(weather: Optional[dict]) -> List[float]:
    out = [0.0] * N_WEATHER
    keys = list(weather.keys()) if weather else []
    if not keys:
        out[-1] = 1.0  # no weather
        return out
    for w in keys:
        if w in WEATHERS:
            out[WEATHERS.index(w)] = 1.0
    return out


def _hazard_vec(conds: Optional[dict]) -> List[float]:
    out = [0.0] * N_HAZARD
    keys = list(conds.keys()) if conds else []
    for c in keys:
        if c in HAZARDS:
            out[HAZARDS.index(c)] = 1.0
    return out


def _team_mons(team: Optional[dict]):
    mons = list((team or {}).values())
    out = [None] * N_TEAM_SLOTS
    for i in range(min(N_TEAM_SLOTS, len(mons))):
        out[i] = mons[i]
    return out


# ---------------------------------------------------------------------------
# Active-mon block
# ---------------------------------------------------------------------------
def _active_block(mon: Optional[Pokemon], include_pp: bool) -> List[float]:
    if mon is None:
        return [0.0] * _ACTIVE
    f: List[float] = []
    f.append(_hp_fraction(mon))
    f += _status_onehot(mon)
    f.append(1.0 if mon.fainted else 0.0)
    f.append(1.0 if _safe(lambda: mon.revealed, False) else 0.0)
    f += _boost_vec(mon)
    f += _hash_vec(_safe(lambda: mon.name), N_SPECIES_HASH)
    f += _base_stat_vec(mon)
    f += _type_onehot(mon)
    f += _hash_vec(_safe(lambda: mon.ability), N_ABILITY_HASH)
    f += _hash_vec(_safe(lambda: mon.item), N_ITEM_HASH)
    f += _moves_vec(mon, include_pp)
    f += _cues_vec(mon)
    return f


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def embed_battle(battle: AbstractBattle) -> np.ndarray:
    """Embed a battle state into a fixed-length float32 vector of size OBS_DIM."""
    f: List[float] = []

    A = battle.active_pokemon
    O = _safe(lambda: battle.opponent_active_pokemon)

    # --- Actives (own knows pp; opponent pp hidden -> 0) ---
    f += _active_block(A, include_pp=True)
    f += _active_block(O, include_pp=False)

    # --- Own team (bench) ---
    for mon in _team_mons(battle.team):
        if mon is not None:
            f.append(1.0 if mon.fainted else 0.0)
            f.append(_hp_fraction(mon))
            f += _status_onehot(mon)
            f += _hash_vec(_safe(lambda: mon.name), N_SPECIES_HASH)
        else:
            f += [0.0] * _OWN_SLOT

    # --- Opponent team (bench): what is revealed/fainted + identity/type ---
    for mon in _team_mons(_safe(lambda: battle.opponent_team)):
        if mon is not None:
            f.append(1.0 if mon.fainted else 0.0)
            f.append(1.0 if _safe(lambda: mon.revealed, False) else 0.0)
            f += _hash_vec(_safe(lambda: mon.name), N_SPECIES_HASH)
            f += _type_onehot(mon)
        else:
            f += [0.0] * _OPP_SLOT

    # --- Field ---
    f += _weather_onehot(_safe(lambda: battle.weather, {}))
    f += _hazard_vec(_safe(lambda: battle.side_conditions, {}))
    f += _hazard_vec(_safe(lambda: battle.opponent_side_conditions, {}))
    try:
        f.append(min(int(battle.turn), 30) / 30.0)
    except Exception:
        f.append(0.0)
    f.append(1.0 if _safe(lambda: battle.trapped, False) else 0.0)
    f.append(1.0 if _safe(lambda: battle.force_switch, False) else 0.0)
    f.append(1.0 if _safe(lambda: battle.can_mega_evolve, False) else 0.0)
    f.append(1.0 if _safe(lambda: battle.can_z_move, False) else 0.0)
    f.append(1.0 if _safe(lambda: battle.can_dynamax, False) else 0.0)
    f.append(1.0 if _safe(lambda: battle.can_tera, False) else 0.0)

    # Opponent has a revealed priority move -> acts first regardless of speed (speed proxy).
    opp_pri = 0.0
    for mv in _moves(O):
        try:
            if float(mv.priority) > 0:
                opp_pri = 1.0
                break
        except Exception:
            pass
    f.append(opp_pri)

    # Relative base speed (opp - own) -- a gen-stable "which is the faster mon" proxy, since
    # base speed is known for both as soon as species are revealed.
    rel_speed = 0.0
    try:
        own_bs = _safe(lambda: A.base_stats, {}) or {} if A is not None else {}
        opp_bs = _safe(lambda: O.base_stats, {}) or {} if O is not None else {}
        own_s = float(own_bs.get("spe", 0) or 0)
        opp_s = float(opp_bs.get("spe", 0) or 0)
        if own_s or opp_s:
            rel_speed = max(-1.0, min(1.0, (opp_s - own_s) / 120.0))
    except Exception:
        rel_speed = 0.0
    f.append(rel_speed)

    # --- Global ---
    own_fainted = sum(1 for m in (battle.team or {}).values() if m.fainted)
    opp_fainted = sum(1 for m in (_safe(lambda: battle.opponent_team) or {}).values() if m.fainted)
    f.append(own_fainted / N_TEAM_SLOTS)
    f.append(opp_fainted / N_TEAM_SLOTS)

    out = np.asarray(f, dtype=np.float32)
    # Defensive: guarantee the exact contract even if a field was absent for a gen.
    if out.shape[0] < OBS_DIM:
        out = np.concatenate([out, np.zeros(OBS_DIM - out.shape[0], dtype=np.float32)])
    elif out.shape[0] > OBS_DIM:
        out = out[:OBS_DIM]
    return out
