"""Step 3.1/3.2 (review 2026-11-22) — STATIC mon/move features.

The net used to see only HP/status/boosts/hazards + learned id embeddings.
The type chart, base stats, move power/type/category/priority/accuracy and
effect flags had to be DISCOVERED from a +1/-1 reward ~50 turns away — not
realistic, so the net fell back to "the move id that wins on average"
(i.e. the strongest move). These facts are public (a species' types/base
stats are species properties, not hidden info — no rule-4 leak) and the
legality DB already carries them, so we emit them as float blocks next to
the id embeddings (the embeddings stay; these are ADDITIONAL features).

Convention (same as the rest of net/):
  * the OUTPUT is exactly five registry fields (``action_move_static``,
    ``own_types``, ``opp_types``, ``own_stats``, ``opp_stats``) as numpy
    arrays — the encoder (:func:`net.encoder.encode_context`) places them;
  * PURE w.r.t. its inputs: same ctx + actions + gen -> bitwise-identical
    arrays (no randomness, no dict-order dependence, canonical slot order);
  * DEGRADES, never raises: an unknown species/move yields a ZERO row
    (the net reads "unknown" — exactly like the existing id channels);
  * lazy ``legality`` import so ``import net.static_features`` stays cheap
    and torch-free (the same discipline as fitness/ and pool/).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .registry import (
    K_MOVE_STATIC,
    MAX_ACTIONS,
    MAX_SLOTS,
    N_STATS,
    N_TYPES,
    STAT_AXES,
    TYPES,
)

__all__ = ["STATIC_FEATURE_NAMES", "build_static_features",
           "move_static_vector", "types_onehot", "stats_vector"]

#: the exact registry fields this module fills (the encoder's zero-fill set).
STATIC_FEATURE_NAMES: tuple = (
    "action_move_static", "own_types", "opp_types", "own_stats", "opp_stats",
)

_TYPE_IDX: Dict[str, int] = {t: i for i, t in enumerate(TYPES)}

#: category -> the [2] slot value (physical=0.0, special=0.5, status=1.0).
_CATEGORY_VAL: Dict[str, float] = {"Physical": 0.0, "Special": 0.5, "Status": 1.0}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _species_data(name: str, gen: int) -> Dict:
    """``get_species(name, gen)`` with a gen-9 fallback for the static fields.

    The legality DB only stores **gen-9** species types/stats/abilities — for
    gens 1-8 ``get_species`` returns ``types=[]`` / ``stats=None`` (verified).
    Base types and base stats are effectively GEN-INVARIANT for a species, so
    the gen-9 row is the best available static proxy for an old-gen mon; without
    this the static mon features would silently zero out for every non-gen9
    format (the user wants all formats to work). Move data is per-gen and is
    handled separately by :func:`move_static_vector`, so this fallback never
    hides a real per-gen move difference.
    """
    try:
        from legality.lookup import get_species
        s = get_species(name, gen) or {}
    except Exception:  # noqa: BLE001
        return {}
    if not (s.get("types") or s.get("stats")) and gen != 9:
        try:
            s9 = get_species(name, 9) or {}
        except Exception:  # noqa: BLE001
            s9 = {}
        if s9.get("types") or s9.get("stats"):
            s = s9
    return s


def move_static_vector(move_name: str, gen: int) -> np.ndarray:
    """One K_MOVE_STATIC row for a move id. ZERO row when unknown (graceful).

    Layout (registry.K_MOVE_STATIC):
      [0] (power or 0)/200, capped 1.0     [1] type_idx/N_TYPES (0 = none)
      [2] category (phys 0.0 / spec 0.5 / status 1.0)
      [3] (priority+6)/12                   [4] accuracy/100 (None = 1.0:
                                              always hits, e.g. Foresight)
      [5] is_hazard  [6] is_screen  [7] has_boosts  [8] side_status
      [9] side_condition  [10] is_tailwind
    """
    v = np.zeros(K_MOVE_STATIC, dtype=np.float32)
    if not move_name:
        return v
    try:
        from legality.lookup import get_move
        m = get_move(move_name, gen) or {}
    except Exception:  # noqa: BLE001 — an unknown move degrades to zero, never raises
        return v
    if not m:
        return v
    power = m.get("power")
    if isinstance(power, (int, float)) and power > 0:
        v[0] = _clamp01(float(power) / 200.0)
    t = m.get("type")
    if t in _TYPE_IDX:
        v[1] = _TYPE_IDX[t] / N_TYPES
    cat = m.get("category")
    if cat in _CATEGORY_VAL:
        v[2] = _CATEGORY_VAL[cat]
    pri = m.get("priority")
    if isinstance(pri, (int, float)):
        v[3] = _clamp01((float(pri) + 6.0) / 12.0)
    acc = m.get("accuracy")
    v[4] = 1.0 if acc is None else _clamp01(float(acc) / 100.0)
    # flags: use truthiness (not `is not None`) so an empty container/"" from
    # the DB cannot falsely flag a move (null is the normal "absent" marker).
    v[5] = 1.0 if m.get("is_hazard") else 0.0
    v[6] = 1.0 if m.get("is_screen") else 0.0
    v[7] = 1.0 if m.get("boosts") else 0.0
    v[8] = 1.0 if m.get("side_status") else 0.0
    v[9] = 1.0 if m.get("side_condition") else 0.0
    v[10] = 1.0 if m.get("is_tailwind") else 0.0
    return v


def types_onehot(species_name: str, gen: int) -> np.ndarray:
    """One N_TYPES row for a species. Dual types = 0.5/0.5; unknown = zero.

    Base stats are a SEPARATE row (:func:`stats_vector`) — the two share the
    species lookup but different callers may want only one of them.
    """
    v = np.zeros(N_TYPES, dtype=np.float32)
    if not species_name:
        return v
    s = _species_data(species_name, gen)
    types = s.get("types") or ()
    if len(types) == 1:
        t = types[0]
        if t in _TYPE_IDX:
            v[_TYPE_IDX[t]] = 1.0
    elif len(types) >= 2:
        for t in types[:2]:
            if t in _TYPE_IDX:
                v[_TYPE_IDX[t]] = 0.5
    return v


def stats_vector(species_name: str, gen: int) -> np.ndarray:
    """One N_STATS row for a species (base stats /255, capped 1.0). Unknown = 0."""
    v = np.zeros(N_STATS, dtype=np.float32)
    if not species_name:
        return v
    s = _species_data(species_name, gen)
    stats = s.get("stats") or {}
    for i, axis in enumerate(STAT_AXES):
        val = stats.get(axis)
        if isinstance(val, (int, float)):
            v[i] = _clamp01(float(val) / 255.0)
    return v


def _side_slots(ctx: Dict, side: str) -> Dict[str, Dict]:
    key = "own" if side == "p1" else "opponent"
    return ctx.get(key, {}) or {}


def build_static_features(ctx: Dict, actions: Optional[List[Dict]],
                          gen: int) -> Dict[str, np.ndarray]:
    """All five Step-3 static fields for ONE ``get_context()`` snapshot.

    ``actions``: the D60 legal-action list (move actions carry ``move``).
    The output keys are EXACTLY :data:`STATIC_FEATURE_NAMES` with the
    registry shapes — the encoder (or a batch stacker) can assign them 1:1.
    """
    out: Dict[str, np.ndarray] = {
        "action_move_static": np.zeros((MAX_ACTIONS, K_MOVE_STATIC), np.float32),
        "own_types": np.zeros((MAX_SLOTS, N_TYPES), np.float32),
        "opp_types": np.zeros((MAX_SLOTS, N_TYPES), np.float32),
        "own_stats": np.zeros((MAX_SLOTS, N_STATS), np.float32),
        "opp_stats": np.zeros((MAX_SLOTS, N_STATS), np.float32),
    }

    # per-action static move data: fill the row for ANY action that carries a
    # ``move`` (i.e. ``kind`` in move AND tera — a tera action re-uses the same
    # move, whose type/power matter most precisely because tera changes type
    # effectiveness). Switch/transform actions carry no ``move`` and stay zero
    # (the switch target's species features already live on its own slot row).
    for i, act in enumerate((actions or [])[:MAX_ACTIONS]):
        if act.get("move"):
            out["action_move_static"][i] = move_static_vector(act["move"], gen)

    own_blocks = _side_slots(ctx, "p1")
    opp_blocks = _side_slots(ctx, "p2")
    for i, slot in enumerate(f"p1{c}" for c in "abcdef"):
        blk = own_blocks.get(slot)
        name = (blk or {}).get("name") or (blk or {}).get("species") or ""
        if name:
            out["own_types"][i] = types_onehot(name, gen)
            out["own_stats"][i] = stats_vector(name, gen)
    for i, slot in enumerate(f"p2{c}" for c in "abcdef"):
        blk = opp_blocks.get(slot)
        name = (blk or {}).get("species") or (blk or {}).get("name") or ""
        if name:
            out["opp_types"][i] = types_onehot(name, gen)
            out["opp_stats"][i] = stats_vector(name, gen)
    return out
