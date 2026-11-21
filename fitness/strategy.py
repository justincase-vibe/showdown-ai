"""A3.3b — the KO-aware move-ranking "brain" (PURE + interface-agnostic).

This is the decision logic an agent uses to prefer moves that KO the opponent,
once it has the structured belief context (``get_context()``). It is a **pure
function over the context snapshot + the candidate moves**, so it is
INDEPENDENT of the a/b/c interface decision (HOW that context is delivered to
the agent): whichever way the context is passed in, the ranking logic is the
SAME. That keeps this step safe under all three options — it commits to none.

One-workflow: reuses the SINGLE KO core (``fitness.ko``) -> the ONE damage
model (``damage.calc``). No second move table, no second damage pipeline.
Torch-free at import (``fitness.ko`` is torch-free; the damage calc is imported
lazily inside the KO core only when a real calc is actually run).

Ranking (best-first):
  1. KO ladder — ``guaranteed_ko`` > ``can_ko`` > ``expected_ko`` > none.
  2. within a tier — more expected damage (higher ``expected_pct``) wins, so a
     bigger kill / bigger chip is preferred (a sane tie-break; it also means a
     status move, which deals 0, sorts below every damaging move).
The raw damage pct comes straight from the shared calc, so this is at least as
accurate as the obs-only heuristic's ``power * effectiveness * STAB`` and
strictly better when the opponent is below full HP (where "will it KO" matters).

Degraded, never crashes: an unknown opponent (the KO core degrades) simply puts
every move in tier 0 and the sort is stable (original order) — a partial belief
state must not kill a decision.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import ko as _ko

__all__ = ["ko_ladder_score", "rank_moves", "threat_assessment", "choose_action"]

#: per-gen cached species -> display-types (lazy; avoids a net.legality import
#: at module load so ``import fitness.strategy`` stays torch-free / cheap).
_TYPES_BY_GEN: Dict[int, Dict[str, tuple]] = {}


def _species_types(species: str, gen: int) -> tuple:
    """A species' display types, or () if unknown. Cached per gen. Defensive:
    a lookup failure just yields no types (the hazard penalty degrades to 0)."""
    if not species:
        return ()
    table = _TYPES_BY_GEN.get(gen)
    if table is None:
        table = _TYPES_BY_GEN[gen] = {}
    if species in table:
        return table[species]
    types = ()
    try:
        from legality.lookup import get_species
        types = tuple(get_species(species, gen).get("types") or ())
    except Exception:  # noqa: BLE001 — an unknown species must not kill a decision
        types = ()
    table[species] = types
    return types


def _pick_switch(switches: List[Dict], ctx: Dict, gen: int = 9) -> Dict:
    """The best legal switch-in, hazard-aware from the CONTEXT (opponent-side
    Stealth Rock / Toxic Spikes penalise type-weak switch-ins). Degrades to the
    first legal switch when there's only one, no hazards, or no types known —
    never raises."""
    if len(switches) == 1:
        return switches[0]
    field = ctx.get("field", {}) or {}
    stealthrock = bool((field.get("stealthrock") or {}).get("p2"))
    toxicspikes = bool((field.get("toxicspikes") or {}).get("p2"))
    if not (stealthrock or toxicspikes):
        return switches[0]

    def score(act: Dict) -> float:
        types = set(_species_types(str(act.get("species") or ""), gen))
        if not types:
            return 0.0
        penalty = 0.0
        if stealthrock and (types & {"Rock", "Steel", "Flying", "Bug"}):
            penalty += 2.0
        if toxicspikes and (types & {"Poison", "Flying"}):
            penalty += 1.0
        return -penalty

    return max(switches, key=score)


def _move_key(move: Any) -> str:
    """A stable identity for a move spec (str or ``{'name': ...}``) — used to
    memoise the per-move KO assessment so each distinct move is calc'd once."""
    if isinstance(move, str):
        return move
    if isinstance(move, dict):
        return str(move.get("name") or move)
    return repr(move)


def ko_ladder_score(assessment: Optional[Dict]) -> tuple:
    """A sortable ``(tier, magnitude)`` for one :func:`fitness.ko` assessment.

    ``tier`` is the KO ladder (3 guaranteed > 2 can > 1 expected > 0 none);
    ``magnitude`` is the expected damage pct (a within-tier tie-break: more
    damage is better). Pure — no I/O. A ``None`` / empty assessment degrades to
    the bottom tier (no KO signal, no damage).
    """
    a = assessment or {}
    if a.get("guaranteed_ko"):
        tier = 3
    elif a.get("can_ko"):
        tier = 2
    elif a.get("expected_ko"):
        tier = 1
    else:
        tier = 0
    mag = float(a.get("expected_pct", 0.0) or 0.0)
    return (tier, mag)


def rank_moves(
    ctx: Dict,
    own_slot: str,
    opp_slot: str,
    moves: List[Any],
    *,
    fmt: str,
    gen: int = 9,
    calc_fn: Optional[Any] = None,
    own_ability: Optional[str] = None,
    own_item: Optional[str] = None,
    opp_ability: Optional[str] = None,
    opp_item: Optional[str] = None,
    own_tera: Optional[str] = None,
    opp_tera: Optional[str] = None,
) -> Dict:
    """Rank ``moves`` best-first by KO likelihood, then raw damage.

    :param ctx: the ``belief.state.BeliefState.get_context()`` snapshot.
    :param own_slot / opp_slot: the active mon's slot keys in ``ctx['own']`` /
                ``ctx['opponent']`` (``ctx['active']['p1']`` / ``['p2']``).
    :param moves: candidate move specs (a name string or ``{'name': ...}``).
    :param fmt / gen: the format id (wins) / engine key for the shared calc.
    :param calc_fn: injectable (tests) — defaults to the shared damage calc.
    :param own_* / opp_* / *_tera: optional hidden-info overrides forwarded to
                the KO core (e.g. the opponent's posterior top pick).
    :return: ``{'order': [move, ...] best-first,
                'best': move | None,
                'assessments': {move_name: ko_assessment}}``.
    Empty ``moves`` -> ``{'order': [], 'best': None, 'assessments': {}}``.
    Never raises on a partial belief state (degrades to a stable order).
    """
    moves = list(moves or [])
    if not moves:
        return {"order": [], "best": None, "assessments": {}}

    assessments: Dict[str, Dict] = {}

    def assess(move: Any) -> Dict:
        key = _move_key(move)
        if key not in assessments:
            assessments[key] = _ko.move_ko(
                ctx, own_slot, opp_slot, move, fmt=fmt, gen=gen, calc_fn=calc_fn,
                own_ability=own_ability, own_item=own_item,
                opp_ability=opp_ability, opp_item=opp_item,
                own_tera=own_tera, opp_tera=opp_tera)
        return assessments[key]

    # stable sort: equal (tier, magnitude) keeps the caller's order (determinism)
    order = sorted(moves, key=lambda m: ko_ladder_score(assess(m)), reverse=True)
    return {"order": order, "best": order[0] if order else None,
            "assessments": assessments}


def _threat_label(assessment: Optional[Dict]) -> str:
    """Map one :func:`fitness.ko` assessment to a defensive threat label.

    ``guaranteed_ko`` (the opponent's move ALWAYS KOs me) > ``can_ko`` (can, on a
    bad roll) > ``expected_ko`` (average KOs) > ``safe`` (best move still can't)
    > ``unknown`` (a party's species is unknown, so it couldn't be assessed).
    """
    a = assessment or {}
    if a.get("guaranteed_ko"):
        return "guaranteed_ko"
    if a.get("can_ko"):
        return "can_ko"
    if a.get("expected_ko"):
        return "expected_ko"
    if a.get("unknown_defender") or a.get("unknown_attacker"):
        return "unknown"
    return "safe"


def threat_assessment(
    ctx: Dict,
    own_slot: str,
    opp_slot: str,
    opp_moves: List[Any],
    *,
    fmt: str,
    gen: int = 9,
    calc_fn: Optional[Any] = None,
    own_ability: Optional[str] = None,
    own_item: Optional[str] = None,
    opp_ability: Optional[str] = None,
    opp_item: Optional[str] = None,
    own_tera: Optional[str] = None,
    opp_tera: Optional[str] = None,
) -> Dict:
    """'Will I be KO'd by the opponent's move?' — the DEFENSIVE half of the crux.

    Ranks the opponent's candidate moves (from ``ctx['opponent'][opp_slot]``'s
    ``top_moves`` / ``moves``) by how badly they'd KO MY active mon, and returns
    the worst (highest) threat. This is what lets an agent switch / protect
    PROACTIVELY (before its HP is low) instead of only reacting to a fixed
    HP threshold. Pure + interface-agnostic (takes the ``get_context()``
    snapshot); reuses the SINGLE KO core (``opp_move_ko``) -> the ONE damage
    model. Never raises on a partial belief state.

    :return: ``{'threat': 'guaranteed_ko'|'can_ko'|'expected_ko'|'safe'|'unknown',
                'worst': move | None,
                'assessments': {move_name: ko_assessment}}``.
    """
    opp_moves = list(opp_moves or [])
    if not opp_moves:
        return {"threat": "unknown", "worst": None, "assessments": {}}

    assessments: Dict[str, Dict] = {}

    def assess(move: Any) -> Dict:
        key = _move_key(move)
        if key not in assessments:
            assessments[key] = _ko.opp_move_ko(
                ctx, own_slot, opp_slot, move, fmt=fmt, gen=gen, calc_fn=calc_fn,
                own_ability=own_ability, own_item=own_item,
                opp_ability=opp_ability, opp_item=opp_item,
                own_tera=own_tera, opp_tera=opp_tera)
        return assessments[key]

    worst_move, worst_score = None, None
    for m in opp_moves:
        a = assess(m)
        s = ko_ladder_score(a)
        if worst_score is None or s > worst_score:
            worst_move, worst_score = m, s
    return {"threat": _threat_label(assessments[_move_key(worst_move)]),
            "worst": worst_move, "assessments": assessments}


def choose_action(
    ctx: Dict,
    acts: List[Dict],
    *,
    fmt: str,
    gen: int = 9,
    calc_fn: Optional[Any] = None,
    switch_threshold: float = 0.30,
    **overrides: Any,
) -> Optional[Dict]:
    """The COMPLETE KO-aware decision, ENTIRELY from the ``get_context()``
    snapshot + the legal actions — the piece that ties the brains together.

    Decision order (deterministic):
      1. no actions -> ``None`` (pass/wait).
      2. no moves legal (force-switch) -> the best switch-in (hazard-aware).
      3. threatened — the opponent's likely move would ``guaranteed_ko`` / 
         ``can_ko`` MY active — OR my active HP <= ``switch_threshold``, and a
         switch is legal -> switch (hazard-aware target).
      4. otherwise -> the best damaging move (``rank_moves``: KO ladder, then
         damage; a status move sorts last).

    Fork-NEUTRAL: a pure function over ``(ctx, acts)`` (no obs, no driver). The
    heuristic's ``act_with_context`` (whichever a/b/c interface is chosen) is a
    THIN wrapper over this. Reuses :func:`rank_moves` + :func:`threat_assessment`
    -> the SINGLE KO core -> the ONE damage model. Never raises on a partial
    belief state (degrades to a safe, legal action).

    :param ctx: the ``belief.state.BeliefState.get_context()`` snapshot
                (POV-correct: ``active.p1`` = my active, ``active.p2`` = theirs).
    :param acts: the legal action dicts (``kind`` in ``move``/``tera``/``switch``;
                move actions carry ``move``, switch actions carry ``species``).
    :return: the chosen action dict, or ``None`` to pass/wait.
    """
    acts = list(acts or [])
    if not acts:
        return None

    moves = [(a, a.get("move")) for a in acts
             if a.get("kind") in ("move", "tera") and a.get("move")]
    switches = [a for a in acts if a.get("kind") == "switch"]

    # 1) force-switch: only switches are legal.
    if not moves:
        return _pick_switch(switches, ctx, gen) if switches else acts[0]

    own_slot = (ctx.get("active") or {}).get("p1")
    opp_slot = (ctx.get("active") or {}).get("p2")

    # can't resolve the actives (degenerate ctx) -> a safe, legal move (A3.1)
    if not own_slot or not opp_slot:
        return moves[0][0]

    own_hp = _ko._hp_frac((ctx.get("own") or {}).get(own_slot))
    opp_block = (ctx.get("opponent") or {}).get(opp_slot, {}) or {}
    opp_moves = opp_block.get("top_moves") or opp_block.get("moves") or []

    # 2) defensive: will the opponent's likely move KO me this turn?
    threat = threat_assessment(
        ctx, own_slot, opp_slot, opp_moves, fmt=fmt, gen=gen,
        calc_fn=calc_fn, **overrides)["threat"]

    # 3) switch when threatened or low-HP, if a switch is legal.
    if switches and (threat in ("guaranteed_ko", "can_ko")
                     or own_hp <= switch_threshold):
        return _pick_switch(switches, ctx, gen)

    # 4) offensive: the best damaging move (KO ladder, then damage).
    names = [name for _a, name in moves]
    order = rank_moves(
        ctx, own_slot, opp_slot, names, fmt=fmt, gen=gen,
        calc_fn=calc_fn, **overrides)["order"]
    best_name = order[0] if order else names[0]
    for a, name in moves:
        if name == best_name:
            return a
    return moves[0][0]
