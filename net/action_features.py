"""Step 3.3 (review 2026-11-22) — the per-ACTION damage/KO/speed block.

The net used to see HP/status/hazards + static move facts but NO damage number,
NO "will this KO", and NO speed order — it had to guess "who wins this turn"
from a +1/-1 reward ~50 turns away. This module emits, for EACH legal action,
the calc-backed facts it actually needs to plan (the belief state the net
reasons from):

  * a move/tera action  -> will it KO the opponent's active (min/max/exp pct,
    guaranteed/can/expected KO, immune) and does it outspeed them;
  * a switch action     -> entry-hazard damage on the incoming mon, and how
    much the incoming mon would take from the opponent's likely (top) move.

Convention (same as the rest of net/):
  * the OUTPUT is exactly ONE registry field (``action_damage``) as a numpy
    array — the encoder (:func:`net.encoder.encode_context`) places it;
  * PURE w.r.t. its inputs: same ctx + actions + fmt + gen -> a
    bitwise-identical array (deterministic; the calc is a pure function and the
    ability/move picks are the deterministic top-1);
  * DEGRADES, never raises: an unknown species or a calc failure yields a ZERO
    row (the net reads "no damage signal") — exactly like the id channels;
  * lazy calc import so ``import net.action_features`` stays cheap and
    torch-free (the same discipline as fitness/ and pool/).

Reuse, do NOT re-derive: move rows come from ``fitness.ko.move_ko`` (which
delegates to the ONE ``damage.calc``) with the opponent's TOP-1 posterior
ability as the hidden-info override; outspeed from ``damage.calc.compare_speed``.
See ``OPEN_DESIGN_CHOICES.md`` ("3.3 DESIGN") for the locked 9-dim schema and
the top-1 / placement decisions. ``calc_fn`` / ``speed_fn`` are injectable so
tests pin the interpretation with fakes (only a couple of checks touch Node).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .registry import K_ACTION_DAMAGE, MAX_ACTIONS

__all__ = ["ACTION_DAMAGE_NAMES", "build_action_damage", "action_damage_row"]

#: the exact registry field this module fills (the encoder's zero-fill set).
ACTION_DAMAGE_NAMES: tuple = ("action_damage",)

_MOVE_KINDS = ("move", "tera")


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _top_ability(opp_block: Optional[Dict]) -> Optional[str]:
    """The opponent's TOP-1 posterior ability (deterministic), or None.

    The belief's per-slot posterior is a list of ``{ability, weight, ...}``; take
    the highest-weight candidate (ties -> the first, stable). This is the
    hidden-info override for the damage calc — the dominant pick carries the
    signal (3.3 DESIGN decision: top-1, not full posterior-weighting, for speed).
    """
    best, best_w = None, -1.0
    for cand in ((opp_block or {}).get("posterior") or []):
        w = float(cand.get("weight", 0.0) or 0.0)
        if w > best_w:
            best_w, best = w, cand
    return (best.get("ability") or None) if best is not None else None


def _active_outspeed(own: Dict, opp: Dict, opp_ability: Optional[str],
                     fmt: str, gen: int, *,
                     speed_fn: Optional[Callable] = None) -> float:
    """1.0 if my active is faster, 0.5 on a tie, 0.0 if slower; 0.5 on failure.

    Uses EFFECTIVE speed (boosts/ability/item/weather), so "who acts first" is a
    fact the net is GIVEN, not something it must guess. ``speed_fn(a, d)`` (test
    injection) returns ``'a'``/``'b'``/``'tie'``; when absent, the shared
    ``damage.calc.compare_speed`` is used (a->my active, b->opponent).
    """
    try:
        from fitness.ko import block_to_calc_mon
        a = block_to_calc_mon(own)
        d = block_to_calc_mon(opp, ability=opp_ability)
        if not a or not d:
            return 0.5
        if speed_fn is not None:
            faster = speed_fn(a, d)
        else:
            from damage import calc as _calc
            faster = _calc.compare_speed(a, d, fmt=fmt, gen=gen)["faster"]
        return 1.0 if faster == "a" else (0.0 if faster == "b" else 0.5)
    except Exception:  # noqa: BLE001 — outspeed must never kill the feature
        return 0.5


def action_damage_row(assessment: Optional[Dict], outspeed: float) -> np.ndarray:
    """One K_ACTION_DAMAGE row from a ``fitness.ko`` assessment (+ outspeed).

    ``assessment`` is the :func:`fitness.ko.ko_assessment` dict (min/max/exp
    pct, the three KO verdicts, immune). A ``None``/empty assessment degrades to
    an all-zero row (no damage signal) — the net reads "unknown", like the id
    channels. Pure — no I/O (the caller does the calc). ``dmg_in`` (dim 8) is
    0 here (move/tera); 3.3b fills it for switch rows.
    """
    v = np.zeros(K_ACTION_DAMAGE, dtype=np.float32)
    a = assessment or {}
    v[0] = _clamp01(float(a.get("min_pct", 0.0) or 0.0) / 100.0)
    v[1] = _clamp01(float(a.get("max_pct", 0.0) or 0.0) / 100.0)
    v[2] = _clamp01(float(a.get("expected_pct", 0.0) or 0.0) / 100.0)
    v[3] = 1.0 if a.get("can_ko") else 0.0
    v[4] = 1.0 if a.get("guaranteed_ko") else 0.0
    v[5] = 1.0 if a.get("expected_ko") else 0.0
    v[6] = _clamp01(outspeed)
    v[7] = 1.0 if a.get("immune") else 0.0
    return v


def build_action_damage(ctx: Dict, actions: Optional[List[Dict]], fmt: str,
                        gen: int = 9, *,
                        calc_fn: Optional[Callable] = None,
                        speed_fn: Optional[Callable] = None) -> Dict[str, np.ndarray]:
    """The ONE ``action_damage`` field for a ``get_context()`` snapshot.

    3.3a: MOVE/TERA rows are filled (will-this-KO + outspeed via the ONE calc);
    SWITCH rows are zero (3.3b adds entry-hazard % + incoming damage). Any action
    that cannot be assessed degrades to a zero row — never raises.

    :param ctx: the ``belief.state.BeliefState.get_context()`` dict.
    :param actions: the D60 legal-action list (move/tera carry ``move``,
                    switch carry ``species``).
    :param fmt / gen: the format id (wins) / engine key for the shared calc.
    :param calc_fn: injectable (tests) — the damage calc (``fitness.ko``'s
                    ``calc_fn`` contract); defaults to the shared ``damage.calc``.
    :param speed_fn: injectable (tests) — ``fn(a, d) -> 'a'|'b'|'tie'``.
    """
    out = np.zeros((MAX_ACTIONS, K_ACTION_DAMAGE), np.float32)
    acts = list(actions or [])[:MAX_ACTIONS]
    if not acts:
        return {"action_damage": out}
    own_slot = (ctx.get("active") or {}).get("p1")
    opp_slot = (ctx.get("active") or {}).get("p2")
    if own_slot is None or opp_slot is None:
        return {"action_damage": out}

    from fitness import ko as _ko  # lazy: keeps import cheap / torch-free
    own = (ctx.get("own", {}) or {}).get(own_slot, {}) or {}
    opp = (ctx.get("opponent", {}) or {}).get(opp_slot, {}) or {}
    opp_ability = _top_ability(opp)
    outspeed = _active_outspeed(own, opp, opp_ability, fmt, gen, speed_fn=speed_fn)

    for i, act in enumerate(acts):
        if act.get("kind") in _MOVE_KINDS and act.get("move"):
            try:
                assessment = _ko.move_ko(
                    ctx, own_slot, opp_slot, act["move"],
                    fmt=fmt, gen=gen, opp_ability=opp_ability, calc_fn=calc_fn)
            except Exception:  # noqa: BLE001 — a calc failure degrades to zero
                assessment = {}
            out[i] = action_damage_row(assessment, outspeed)
        # switch/transform rows stay zero in 3.3a (3.3b fills the switch rows).
    return {"action_damage": out}
