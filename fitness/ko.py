"""A3.3 (the crux) — the damage-calc KO assessment, PURE + interface-agnostic.

This is the "will this KO / will I be KO'd" core the heuristic ladder rung needs.
It is a **library function**, deliberately independent of the a/b/c interface
decision (HOW the heuristic obtains the structured belief context): it takes the
``get_context()`` snapshot (or pre-built calc dicts) + a move and returns a KO
assessment. Wire it in whichever way is chosen later — the logic is the same.

One-workflow: reuses the SINGLE damage model (``damage.calc.calc_damage``),
imported lazily so ``import fitness.ko`` stays torch-free / cheap (the same
discipline as ``fitness.heuristic``). ``calc_fn`` is injectable so tests pin the
interpretation with a fake and only a couple of real-calc checks touch Node.

Semantics (confirmed against the real calc, ``scratch/_ko_probe.py``): the calc
reports damage as a fraction of the DEFENDER'S MAX HP (``min_pct`` / ``max_pct``
/ ``expected_pct``). A defender sitting at ``hp_frac`` of max HP has ``hp_frac``
of its max HP remaining, so a move KOs it when ``damage_pct >= hp_frac * 100``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["block_to_calc_mon", "ko_assessment", "move_ko", "opp_move_ko"]

#: tiny tolerance for the KO comparison. The calc rounds pcts to 2dp
# (node_calc.js pctOf), and hp_frac*100 can carry float wobble (0.319*100 ==
# 31.900000000000002); _EPS absorbs that without crossing the 0.01 pct quantum.
_EPS = 1e-3


def _hp_frac(block: Optional[Dict], default: float = 1.0) -> float:
    """A block's ``hp_frac`` as a clamped 0.0-1.0 float.

    Missing / ``None`` -> ``default`` (1.0 = full HP). A genuine ``0.0`` is
    PRESERVED (not masked to 1.0) so ``move_ko`` / ``opp_move_ko`` thresholds
    always agree with :func:`ko_assessment`'s own clamping for the same input.
    """
    v = (block or {}).get("hp_frac", default)
    if v is None:
        v = default
    return max(0.0, min(1.0, float(v)))


def _degraded(hp: float, *, unknown_defender: bool = False,
              unknown_attacker: bool = False) -> Dict:
    """A safe 'no KO signal' assessment — SAME shape/meaning as the normal path
    (threshold = ``hp*100``, all verdicts False) plus the flag(s) explaining WHY
    the KO could not be assessed: a party's species is unknown, so the shared
    damage calc cannot run (it requires a species name on both mons). Never
    raises — a heuristic must degrade, not crash, on a partial belief state.
    """
    d: Dict[str, Any] = {
        "immune": False, "min_pct": 0.0, "max_pct": 0.0, "expected_pct": 0.0,
        "defender_hp_frac": hp, "ko_threshold_pct": hp * 100.0,
        "can_ko": False, "guaranteed_ko": False, "expected_ko": False,
    }
    if unknown_defender:
        d["unknown_defender"] = True
    if unknown_attacker:
        d["unknown_attacker"] = True
    return d


def block_to_calc_mon(block: Optional[Dict], *,
                      ability: Optional[str] = None,
                      item: Optional[str] = None,
                      tera_type: Optional[str] = None,
                      evs: Optional[Dict] = None,
                      nature: Optional[str] = None) -> Dict:
    """Map a ``get_context()`` mon block (own or opponent) to an @smogon/calc dict.

    Uses only the KNOWN fields — ``name``/``species``, ``ability``, ``item``,
    the first ``status`` condition, ``boosts`` and ``hp`` (the block's ``hp_frac``).
    EVs/IVs/nature are optional: when provided (from the posterior's top-1
    candidate) the calc uses them for more accurate damage; when absent it
    falls back to the calc's defaults. ``ability`` / ``item`` / ``tera_type`` /
    ``evs`` / ``nature`` are overridable so a caller can substitute the
    opponent's posterior's top pick (hidden info).
    Returns ``{}`` when no species is known (the caller must skip the calc).
    """
    if not block:
        return {}
    m: Dict[str, Any] = {}
    name = block.get("name") or block.get("species") or ""
    if not name:
        return {}
    m["name"] = name
    ability = ability or block.get("ability") or ""
    if ability:
        m["ability"] = ability
    item = item or block.get("item") or ""
    if item:
        m["item"] = item
    statuses = block.get("statuses") or []
    if statuses:
        m["status"] = statuses[0]  # the first condition (par/brn/tox/psn/slp/fro)
    boosts = block.get("boosts") or {}
    if boosts:
        m["boosts"] = dict(boosts)
    hp = block.get("hp_frac")
    if hp is not None:
        m["hp"] = max(0.0, min(1.0, float(hp)))
    if tera_type:
        m["teraType"] = tera_type
    # EVs + nature (from the posterior's top-1 candidate; the @smogon/calc
    # format expects a dict of 0-252 per stat and a nature string).
    if evs:
        m["evs"] = {k: int(v) for k, v in evs.items() if k in ("hp", "atk", "def", "spa", "spd", "spe")}
    if nature and nature != "Hardy":
        m["nature"] = nature
    return m


def ko_assessment(dmg: Dict, *, defender_hp_frac: float = 1.0) -> Dict:
    """Interpret ONE ``calc_damage`` result as a KO signal vs the defender's
    CURRENT HP.

    :param dmg: the ``damage.calc.calc_damage`` result (needs ``min_pct`` /
                ``max_pct`` / ``expected_pct`` / ``immune``; absent keys degrade
                to 0 / False).
    :param defender_hp_frac: the defender's current HP as a 0.0-1.0 fraction of
                max HP (default 1.0 = full HP).
    :return: a JSON-safe dict with the raw pcts, the KO threshold, and the three
             KO verdicts (``can_ko`` = max damage can KO, ``guaranteed_ko`` =
             min damage KOs, ``expected_ko`` = average damage KOs). An immune
             move can never KO.
    """
    min_pct = float(dmg.get("min_pct", 0.0) or 0.0)
    max_pct = float(dmg.get("max_pct", 0.0) or 0.0)
    exp_pct = float(dmg.get("expected_pct", 0.0) or 0.0)
    immune = bool(dmg.get("immune", False))
    hp = max(0.0, min(1.0, float(defender_hp_frac)))
    threshold = hp * 100.0  # remaining HP, as % of max HP
    return {
        "immune": immune,
        "min_pct": min_pct,
        "max_pct": max_pct,
        "expected_pct": exp_pct,
        "defender_hp_frac": hp,
        "ko_threshold_pct": threshold,
        "can_ko": (not immune) and max_pct >= threshold - _EPS,
        "guaranteed_ko": (not immune) and min_pct >= threshold - _EPS,
        "expected_ko": (not immune) and exp_pct >= threshold - _EPS,
    }


def _default_calc():
    """The ONE damage model (lazy import keeps ``fitness.ko`` cheap at import)."""
    from damage import calc as _calc
    return _calc.calc_damage


def move_ko(ctx: Dict, own_slot: str, opp_slot: str, move: Any, *,
            fmt: str, gen: int = 9, calc_fn: Optional[Any] = None,
            own_ability: Optional[str] = None, own_item: Optional[str] = None,
            opp_ability: Optional[str] = None, opp_item: Optional[str] = None,
            own_tera: Optional[str] = None, opp_tera: Optional[str] = None,
            own_evs: Optional[Dict] = None, own_nature: Optional[str] = None,
            opp_evs: Optional[Dict] = None, opp_nature: Optional[str] = None) -> Dict:
    """"Will MY move KO the opponent?" — build the calc dicts from a
    ``get_context()`` snapshot, run the shared damage calc, interpret the result.

    :param ctx: the ``belief.state.BeliefState.get_context()`` dict.
    :param own_slot / opp_slot: the active mon's slot keys in ``ctx['own']`` /
                ``ctx['opponent']`` (``ctx['active']['p1']`` / ``['p2']``).
    :param move: a move-name string (or ``{'name':..., ...}``).
    :param fmt / gen: the format id (wins) / engine key for the shared calc.
    :param calc_fn: injectable (tests) — defaults to ``damage.calc.calc_damage``.
    :param own_ability/own_item/opp_ability/opp_item/own_tera/opp_tera: optional
                overrides (e.g. the opponent's posterior top pick for hidden info).
    :return: the :func:`ko_assessment` dict. If the opponent's species is unknown
             (empty calc dict) it degrades to a safe "no KO signal" assessment.
    """
    own = (ctx.get("own", {}) or {}).get(own_slot, {}) or {}
    opp = (ctx.get("opponent", {}) or {}).get(opp_slot, {}) or {}
    attacker = block_to_calc_mon(own, ability=own_ability, item=own_item,
                                 tera_type=own_tera, evs=own_evs, nature=own_nature)
    defender = block_to_calc_mon(opp, ability=opp_ability, item=opp_item,
                                 tera_type=opp_tera, evs=opp_evs, nature=opp_nature)
    hp = _hp_frac(opp)
    if not attacker or not defender:
        # a party's species is unknown -> the shared calc cannot run (it requires
        # a name on BOTH mons). Degrade gracefully, never raise (latent-bug fix:
        # previously only the defender was checked, so an unknown attacker
        # (my mon in a degenerate ctx) would make the calc raise).
        return _degraded(hp, unknown_defender=not defender,
                         unknown_attacker=not attacker)
    calc_fn = calc_fn or _default_calc()
    dmg = calc_fn(attacker, defender, move, fmt=fmt, gen=gen)
    return ko_assessment(dmg, defender_hp_frac=hp)


def opp_move_ko(ctx: Dict, own_slot: str, opp_slot: str, move: Any, *,
                fmt: str, gen: int = 9, calc_fn: Optional[Any] = None,
                own_ability: Optional[str] = None, own_item: Optional[str] = None,
                opp_ability: Optional[str] = None, opp_item: Optional[str] = None,
                own_tera: Optional[str] = None, opp_tera: Optional[str] = None,
                own_evs: Optional[Dict] = None, own_nature: Optional[str] = None,
                opp_evs: Optional[Dict] = None, opp_nature: Optional[str] = None) -> Dict:
    """"Will the OPPONENT's move KO me?" — the reverse direction (attacker =
    opponent, defender = my active mon). Same machinery as :func:`move_ko`; the
    slots are swapped so ``own_*`` overrides apply to MY mon and ``opp_*`` to
    theirs. Lets the heuristic weigh "will I survive this turn?" when choosing.

    6.6a: the ``*_evs`` / ``*_nature`` overrides (the opponent's posterior
    top-1 build) are now accepted too, so the DEFENSIVE calc is as informed as
    the OFFENSIVE one (:func:`move_ko`) — the strategy_posterior variant passes
    the same top-1 posterior both ways.
    """
    own = (ctx.get("own", {}) or {}).get(own_slot, {}) or {}
    opp = (ctx.get("opponent", {}) or {}).get(opp_slot, {}) or {}
    defender = block_to_calc_mon(own, ability=own_ability, item=own_item,
                                 tera_type=own_tera, evs=own_evs, nature=own_nature)
    attacker = block_to_calc_mon(opp, ability=opp_ability, item=opp_item,
                                 tera_type=opp_tera, evs=opp_evs, nature=opp_nature)
    hp = _hp_frac(own)
    if not attacker or not defender:
        # the REAL case here: the ATTACKER is the opponent, whose species is
        # inferred and may be unknown. Degrade gracefully instead of letting the
        # calc raise on an empty attacker dict (the latent bug this fixes).
        return _degraded(hp, unknown_defender=not defender,
                         unknown_attacker=not attacker)
    calc_fn = calc_fn or _default_calc()
    dmg = calc_fn(attacker, defender, move, fmt=fmt, gen=gen)
    return ko_assessment(dmg, defender_hp_frac=hp)
