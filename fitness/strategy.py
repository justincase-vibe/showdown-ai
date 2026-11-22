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

# ══════════════════ Step 6.2 — rule 1: hazard setup ══════════════════

#: Canonical hazard-move names (lowercased, spaces/dashes stripped). Covers
#: the DB ``is_hazard`` flag AND the toxicspikes data gap (legality.HAZARDS
#: omits it). Hazard identities are stable across gens, so a name set is the
#: hermetic primary source; an injectable ``move_db`` is the generic fallback.
_HAZARD_NAMES = frozenset({
    "stealthrock", "spikes", "toxicspikes",
    "stickyweb", "stickybarb", "stickybarbedsteel",
})

#: hazard id -> (ctx['field'] key, kind) that shows it ALREADY UP on OUR side
#: (p1). Sticky Web is untracked in the belief field -> absent here, so it is
#: always treated as "clean" (setting it is free value).
_HAZARD_FIELD = {
    "stealthrock": ("stealthrock", "bool"),
    "spikes": ("spikes", "layers"),
    "toxicspikes": ("toxicspikes", "layers"),
    "stickyweb": ("stickyweb", "bool"),  # belief DOES track it (per-side bool)
}

#: Step 6.3 — rule 2: status moves. move name (normalized) -> (status code
#: as the belief records it — raw Showdown codes brn/par/slp/tox, or
#: "leechseed" when it's a volatile), opponent TYPES immune to the status.
#: Deliberately EXCLUDED: Leech Seed — the belief does NOT track it (not a
#: status, not in the 7 curated volatiles), so an "already seeded" check is
#: impossible and the rule would re-pick it every turn (the sticky-web bug
#: class). Adding it back requires the belief to track it first.
_STATUS_MOVES = {
    "willowisp":    ("brn", frozenset({"Fire", "Water"})),
    "thunderwave":  ("par", frozenset({"Electric"})),
    "glare":        ("par", frozenset({"Electric"})),
    "toxic":        ("tox", frozenset({"Poison", "Steel"})),
    "toxin":        ("tox", frozenset({"Poison", "Steel"})),
    "sleeppowder":  ("slp", frozenset({"Psychic"})),
    "spore":        ("slp", frozenset({"Psychic"})),
    "hypnosis":     ("slp", frozenset({"Psychic"})),
}


#: field-setup id -> opponent TYPES immune / not-affected (type-based only;
#: ability immunities like Levitate are not in the ctx — degrade: attempt
#: anyway). Empty set = universally applicable (tailwind / terrain / screens
#: "work" on everyone; they just help different types more).
_HAZARD_IMMUNE_TYPES = {
    "stealthrock": frozenset({"Flying"}),
    "spikes": frozenset({"Flying"}),
    "toxicspikes": frozenset({"Poison", "Ground", "Steel"}),
    "stickyweb": frozenset(),
    "stickybarb": frozenset(),
    "stickybarbedsteel": frozenset(),
    "tailwind": frozenset(),
    "reflect": frozenset(),
    "lightscreen": frozenset(),
    "auroraveil": frozenset(),
    "grassytterrain": frozenset(),
    "psychicterrain": frozenset(),
    "mistyterrain": frozenset(),
    "trickroom": frozenset(),
    "wonderroom": frozenset(),
}

#: Declarative field-setup table — the ONE place that knows which move sets
#: which field state and HOW to read "already up" from ctx['field'] (the
#: belief tracks ALL of these: stickyweb/safeguard per-side bool, spikes/TS
#: per-side layers, tailwind_side, terrain, screens per-side bool).
#: ``kind``: "side" = per-side key (clean iff p1 is not set); "shared" = one
#: global key (terrain) — set by EITHER side, so the opponent's terrain also
#: blocks ours (re-setting it is a wasted turn).
#: ORDER = priority when several setup moves are legal (hazards carry the
#: highest match-long value; screens are free damage reduction; terrain is the
#: most situation-dependent).
_FIELD_SETUP_RULES = (
    ("stealthrock",   ("stealthrock", "stealth rock"),                       "side",   "stealthrock"),
    ("spikes",        ("spikes",),                                           "side",   "spikes"),
    ("toxicspikes",   ("toxicspikes", "toxic spikes"),                       "side",   "toxicspikes"),
    ("stickyweb",     ("stickyweb", "sticky web", "sticky barb", "sticky barbed steel"), "side", "stickyweb"),
    ("tailwind",      ("tailwind",),                                         "side",   "tailwind_side"),
    ("reflect",       ("reflect",),                                          "side",   "reflect"),
    ("lightscreen",   ("lightscreen", "light screen"),                       "side",   "lightscreen"),
    ("auroraveil",    ("auroraveil", "aurora veil"),                         "side",   "auroraveil"),
    ("grassytterrain", ("grassytterrain", "grassy terrain", "grassy field"), "shared", "terrain"),
    ("psychicterrain", ("psychicterrain", "psychic terrain"),                "shared", "terrain"),
    ("mistyterrain",   ("mistyterrain", "misty terrain"),                    "shared", "terrain"),
    # pseudo-weather / field-wide rooms (SHARED booleans in the belief): last
    # in priority — the most team-committing setups (a teacher should only
    # show them when nothing better is legal). NOTE: the net sees the TR flag
    # as an obs channel, but `damage.calc.compare_speed` is NOT TR-aware (see
    # OPEN_DESIGN_CHOICES) — the outspeed feature is stale under TR.
    ("trickroom",      ("trickroom", "trick room"),                          "shared", "trickroom"),
    ("wonderroom",     ("wonderroom", "wonder room"),                        "shared", "wonderroom"),
)


def _norm_move(name: Any) -> str:
    """A move name normalized for hazard identity (lower, no spaces/dashes)."""
    return str(name or "").lower().replace(" ", "").replace("-", "")


def _field_setup_id(name: Any, move_db: Optional[Any] = None) -> Optional[str]:
    """Resolve a move name to a canonical field-setup id (hazard/screen/
    tailwind/terrain), or None if the move sets no tracked field state.

    Primary: the declarative table (hermetic, covers the DB flag gaps —
    e.g. toxicspikes is missing from legality.HAZARDS). Fallback: an
    injectable ``move_db`` — its ``is_hazard`` / ``is_screen`` /
    ``is_tailwind`` flags catch setups the table doesn't know. Never raises.
    """
    n = _norm_move(name)
    for fid, names, _kind, _key in _FIELD_SETUP_RULES:
        if n in names:
            return fid
    if move_db is not None:
        try:
            d = move_db(n, 9) or {}
            if d.get("is_hazard") or d.get("is_screen") or d.get("is_tailwind"):
                return n
        except Exception:  # noqa: BLE001 — a lookup failure is "not a setup"
            pass
    return None


def _opp_species(ctx: Dict) -> List[str]:
    """All KNOWN opponent species in the ctx (active + bench), deduped."""
    out: List[str] = []
    for block in (ctx.get("opponent") or {}).values():
        block = block or {}
        sp = block.get("species") or block.get("name")
        if sp and sp not in out:
            out.append(sp)
    return out


def _field_setup_clean(fid: str, ctx: Dict) -> bool:
    """Is the field state NOT already established?

    ``side`` setups (hazards/screens/tailwind): clean iff OUR side (p1) has
    not set them (the opponent's rocks don't block ours). ``shared`` setups
    (terrain): clean iff the global key is empty — the OPPONENT'S terrain also
    blocks ours (re-setting it would be a wasted turn).
    Unknown ids (a move_db fallback) degrade to clean: attempt anyway.
    """
    rule = next((r for r in _FIELD_SETUP_RULES if r[0] == fid), None)
    if rule is None:
        return True
    _fid, _names, kind, key = rule
    f = ctx.get("field") or {}
    if kind == "side":
        raw = f.get(key)
        if key == "tailwind_side":
            # belief stores the SIDE as a string ("p1"/"p2"/None), not a dict
            return raw != "p1"
        val = (raw or {}).get("p1")
    else:  # shared (terrain)
        val = f.get(key)
    if isinstance(val, bool):
        return not val
    if isinstance(val, int):
        return val <= 0
    return not val  # terrain string / missing -> not established


def _hazard_broadly_effective(fid: str, ctx: Dict, gen: int) -> bool:
    """≥ half of the KNOWN opponent species are not type-immune to the hazard.

    No known species (or all lookups fail) -> True: hazards are free value
    with no info, so a teacher still sets them (6.1 design note)."""
    immune_to = _HAZARD_IMMUNE_TYPES.get(fid, frozenset())
    if not immune_to:
        return True
    known: List[bool] = []
    for sp in _opp_species(ctx):
        types = set(_species_types(sp, gen))
        if types:
            known.append(bool(types & immune_to))
    if not known:
        return True
    return (sum(known) / len(known)) <= 0.5


def _boosts_for(name: Any, gen: int, move_db: Optional[Any]) -> Optional[Dict]:
    """A move's ``boosts`` dict (per-gen from the DB / injected stub), or None.

    The DB ``boosts`` describe the change the move applies (self-boosts are
    positive; opponent-lowering moves like Growl are NEGATIVE and therefore
    never qualify as setup — the caller checks for positive entries). Unknown
    move or lookup failure -> None (degrade, never raise).
    """
    n = _norm_move(name)
    try:
        if move_db is not None:
            d = move_db(n, gen) or {}
        else:
            from legality.lookup import get_move  # lazy: keep module import cheap
            d = get_move(n, gen) or {}
    except Exception:  # noqa: BLE001 — an unknown move is "no setup"
        return None
    b = d.get("boosts")
    return b if isinstance(b, dict) else None


def _pick_setup(moves: List, threat: str, gen: int = 9,
                move_db: Optional[Any] = None) -> Optional[Dict]:
    """Rule 3 (Step 6.4): the strongest legal SELF-BOOST action, or None.

    Chosen ONLY when ``threat == 'safe'`` (strict — not even can_ko, and NOT
    'unknown': a wasted setup turn is worse than chip, and a teacher should
    not gamble on an unassessed opponent — 6.1 design note). Among candidates,
    the strongest single positive boost wins (Swords Dance +2 over Work Up +1);
    ties keep legal order. Moves with only negative boosts (Growl) never
    qualify. Per-gen data (the DB override makes e.g. gen-1 Light Screen a
    boost, which this rule then treats as setup — correct for that gen).
    Pure + degrading; never raises.
    """
    if threat != "safe":
        return None
    best_act, best_str = None, 0
    for act, name in moves:
        b = _boosts_for(name, gen, move_db)
        if not b:
            continue
        strength = max((v for v in b.values()
                        if isinstance(v, (int, float)) and v > 0),
                       default=0)
        if strength > best_str:  # strict -> ties keep the earlier legal move
            best_act, best_str = act, strength
    return best_act


def _status_already_on(block: Optional[Dict], code: str) -> bool:
    """Does the mon already have this status? Raw Showdown codes (brn/par/
    slp/tox) from the belief; "leechseed" would check the volatiles list
    (currently no map entry needs it — see _STATUS_MOVES)."""
    b = block or {}
    if code == "leechseed":
        return "leechseed" in (b.get("volatiles") or [])
    return code in (b.get("statuses") or [])


def _pick_status(moves: List, ctx: Dict, opp_slot: Optional[str], threat: str,
                 gen: int = 9) -> Optional[Dict]:
    """Rule 2 (Step 6.3): the best legal STATUS action, or None.

    Chosen when: a status move is legal, the OPPONENT ACTIVE does not already
    have that status, the opponent's types are not immune (unknown types ->
    attempt anyway: a teacher should show the behavior), and I am NOT
    guaranteed-KO'd / can-KO'd this turn (paralysis/sleep is not worth the bad
    roll; burn is not either — conservative per the 6.1 design note).
    Priority sits BELOW field setup (hazards) and (when 6.4 lands) setup
    boosts, ABOVE plain attacks. Pure + degrading; never raises.
    """
    if threat in ("guaranteed_ko", "can_ko"):
        return None
    opp_block = (ctx.get("opponent") or {}).get(opp_slot) or {}
    opp_types = set(_species_types(
        opp_block.get("species") or opp_block.get("name"), gen))
    for act, name in moves:
        spec = _STATUS_MOVES.get(_norm_move(name))
        if spec is None:
            continue
        code, immune_to = spec
        if _status_already_on(opp_block, code):
            continue
        if immune_to and opp_types and (opp_types & immune_to):
            continue
        return act
    return None


def _pick_hazard(moves: List, ctx: Dict, threat: str,
                 gen: int = 9, move_db: Optional[Any] = None) -> Optional[Dict]:
    """Rule 1 (Step 6.2, completed): the best legal FIELD-SETUP action
    (hazard / tailwind / screen / terrain), or None. Name kept for
    compatibility; the rule covers the whole field-setup family.

    ``moves`` is the ``[(action_dict, move_name), ...]`` list from
    :func:`choose_action`. Chosen when: a setup move is legal, its field state
    is not already established (per-side setups: not on OUR side; terrain:
    no terrain at all), it is broadly effective against the known opponent
    (hazards only; screens/tailwind/terrain affect everyone), and I am NOT
    guaranteed-KO'd this turn (a guaranteed threat -> attack/switch instead;
    a can_ko risk for free field value is acceptable — 6.1 design note).
    Priority among several legal setups: hazards > tailwind > screens >
    terrain > rooms (Trick/Wonder Room last — the most team-committing
    setups, so a teacher only shows them when nothing better is legal).
    Pure + degrading: any lookup failure just skips that candidate;
    never raises.
    """
    if threat == "guaranteed_ko":
        return None
    for fid, _names, _kind, _key in _FIELD_SETUP_RULES:  # priority order
        for act, name in moves:
            if _field_setup_id(name, move_db) != fid:
                continue
            if not _field_setup_clean(fid, ctx):
                continue
            if not _hazard_broadly_effective(fid, ctx, gen):
                continue
            return act
    return None

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


def _move_type(name: Any, gen: int = 9,
               move_db: Optional[Any] = None) -> Optional[str]:
    """A move's attacking type (display name, e.g. "Fire"), or None if unknown.

    Uses the injected ``move_db`` (tests) or ``legality.lookup.get_move`` (real);
    mirrors :func:`_boosts_for` (same name-normalisation + degrade contract).
    A None type means the move is SKIPPED in resist scoring (never a wrong value).
    """
    n = _norm_move(name)
    try:
        if move_db is not None:
            d = move_db(n, gen) or {}
        else:
            from legality.lookup import get_move  # lazy: keep import cheap
            d = get_move(n, gen) or {}
    except Exception:  # noqa: BLE001 — an unknown move is "no type"
        return None
    t = d.get("type")
    return t if isinstance(t, str) and t else None


def _opp_move_type_weights(ctx: Dict, gen: int = 9,
                           move_db: Optional[Any] = None) -> List[tuple]:
    """``(attacking_type, weight)`` for the opponent ACTIVE's ``top_moves`` that
    have a KNOWN type — the rule-4 (6.5) input to resist scoring.

    Weights = the belief's per-move posterior weights when exposed
    (``top_move_weights``, same order as ``top_moves``), else UNIFORM (the 6.5
    degrade — the belief does not expose them yet). Moves with an unknown type
    are dropped (degrade, never a wrong factor). Empty list -> the resist term
    is 0 and :func:`_pick_switch` falls back to hazard-only scoring.
    """
    opp_slot = (ctx.get("active") or {}).get("p2")
    opp_block = (ctx.get("opponent") or {}).get(opp_slot) or {}
    top_moves = opp_block.get("top_moves") or []
    if not top_moves:
        return []
    weights = opp_block.get("top_move_weights")
    if not isinstance(weights, (list, tuple)) or len(weights) != len(top_moves):
        weights = [1.0 / len(top_moves)] * len(top_moves)
    out: List[tuple] = []
    for name, w in zip(top_moves, weights):
        t = _move_type(name, gen, move_db)
        if t is not None:
            try:
                out.append((t, float(w)))
            except (TypeError, ValueError):  # a non-numeric weight degrades out
                out.append((t, 0.0))
    return out


def _pick_switch(switches: List[Dict], ctx: Dict, gen: int = 9,
                 move_db: Optional[Any] = None,
                 type_chart: Optional[Dict] = None) -> Dict:
    """The best legal switch-in, RESIST-AWARE + HAZARD-AWARE (rule 4, 6.5).

    Score (lower = better; we take ``max`` of the negated penalty):
      - **resist term** (6.5): Σ weight × max(1.0, effectiveness) over the
        opponent's likely ``top_moves`` — a switch-in that is weak to what the
        opponent will probably do is penalized more; resists/immunes cost the
        same as neutral (max(1.0, ·) clamps them to 1.0). Weights are the
        belief's per-move posterior weights when exposed, else uniform.
      - **hazard term** (existing): opponent-side Stealth Rock / Toxic Spikes
        penalise type-weak switch-ins (a hazard ≈ a move the opponent always uses).

    :param move_db: injectable move->type source (tests) — default the real DB.
    :param type_chart: injectable ``{defending: {attacking: factor}}`` (tests);
        default None -> the fork-verified DB chart (D29: fork = ground truth).
        Kept injectable (like ``net.action_features``) so tests are hermetic and
        never assert a memorised chart value.

    Degrades to the FIRST legal switch when there is only one, or when neither
    resist info (no known opp move types) nor hazards apply — never raises.
    """
    if len(switches) == 1:
        return switches[0]
    field = ctx.get("field", {}) or {}
    stealthrock = bool((field.get("stealthrock") or {}).get("p2"))
    toxicspikes = bool((field.get("toxicspikes") or {}).get("p2"))

    # rule 4 (6.5): opponent's likely moves as (attacking_type, weight); empty
    # -> the resist term is 0 and scoring is hazard-only (the 6.5 degrade).
    move_type_weights = _opp_move_type_weights(ctx, gen, move_db)
    if type_chart is None:
        type_chart = {}
        if move_type_weights:
            try:
                from legality.lookup import get_type_chart  # lazy import
                type_chart = get_type_chart(gen) or {}
            except Exception:  # noqa: BLE001 — no chart -> neutral effectiveness
                type_chart = {}
    chart = type_chart

    def score(act: Dict) -> float:
        types = set(_species_types(str(act.get("species") or ""), gen))
        penalty = 0.0
        # resist term (only when the switch's types are known — otherwise the
        # design degrade: hazard-only, so an unknown-type mon isn't penalised).
        if types and move_type_weights:
            for atk_type, w in move_type_weights:
                eff = 1.0
                for dt in types:
                    eff *= float(chart.get(dt, {}).get(atk_type, 1.0))
                penalty += w * max(1.0, eff)
        # existing hazard penalties (unchanged semantics)
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
    move_db: Optional[Any] = None,
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
      3.5 (Step 6.2) hazard setup — a hazard move is legal, its field is not
         already up on our side, the known opponent is not majority-immune,
         and I am not guaranteed-KO'd -> set the hazard (free match-long value;
         the concept the net must learn to value).
      3.6 (Step 6.4) setup — a self-boost move (Swords Dance / Calm Mind / ...)
         is legal and threat is EXACTLY 'safe' -> the strongest boost
         (setup-sweeper value; a teacher only gambles setup on a known-safe
         turn).
      3.7 (Step 6.3) status — a status move (Wisp/Thunder Wave/Toxic/sleep
         moves/Glare) is legal, the opponent active lacks that status and is
         not type-immune, and I am not guaranteed/can-KO'd -> apply the status
         (match-long damage/pace value; the concept the net must learn to value).
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
        return _pick_switch(switches, ctx, gen, move_db=move_db) if switches else acts[0]

    own_slot = (ctx.get("active") or {}).get("p1")
    opp_slot = (ctx.get("active") or {}).get("p2")

    # can't resolve the actives (degenerate ctx) -> a safe, legal move (A3.1).
    # ``is None`` (NOT truthiness): real get_context() slots are "p1a"-style
    # strings, but a falsy-yet-valid key (e.g. an int slot of 0) must NOT be
    # mistaken for "missing" — that would silently bail to the first move.
    if own_slot is None or opp_slot is None:
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
        return _pick_switch(switches, ctx, gen, move_db=move_db)

    # 3.5) hazard setup (Step 6.2, rule 1): free match-long value — set rocks/
    # spikes/toxic spikes when legal, our field is clean, the known opponent
    # is not majority-immune, and I am not guaranteed-KO'd (with a switch
    # legal, the threat rule above would have fired first).
    hazard = _pick_hazard(moves, ctx, threat, gen, move_db=move_db)
    if hazard is not None:
        return hazard

    # 3.6) setup boosts (Step 6.4, rule 3): the strongest self-boost when
    # legal AND threat is exactly 'safe' (a setup turn must be near-safe;
    # unknown threat -> don't gamble) — SWORDSDANCE & co, the setup-sweeper
    # concept the net must learn to recognize and value.
    setup = _pick_setup(moves, threat, gen, move_db=move_db)
    if setup is not None:
        return setup

    # 3.7) status (Step 6.3, rule 2): burn/paralyze/sleep/toxify when legal,
    # the opponent active lacks it, isn't type-immune, and I am not in
    # guaranteed/can-KO range.
    status = _pick_status(moves, ctx, opp_slot, threat, gen)
    if status is not None:
        return status

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
