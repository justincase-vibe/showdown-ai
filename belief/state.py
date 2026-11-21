"""Step 3.1 — belief state + prior (updates land in 3.2-3.4).

Per opponent slot: a posterior distribution over FULL-SET candidates with
weights. Full sets (not per-slot marginals) so that within-set correlations
are preserved by construction — a Wish set already contains Protect (user
requirement, 2026-11-15).

Candidate schema (plain dict, JSON-safe):
    moves:   [up to 4 move ids, slot order]
    item:    item id, or '' = none/unknown (fallback candidates)
    ability: ability id
    nature:  display name (DB convention, e.g. 'Impish'; 'Hardy' = neutral)
    evs:     {'hp': int, ...} (missing stat = 0)
    tera:    tera type id, or None = UNKNOWN (None is NOT "no Tera")
    weight:  float > 0; weights per slot sum to exactly 1 (invariant, tested)

Prior (D43, 2026-11-15):
  1. Sets for the format itself (Step 1 ``get_common_sets``; probs sum to 1).
  2. The BSS format (``gen9bss3v3``) carries NO sets of its own in the DB —
     sets live in the tier tables. So: merge the same-gen tier pools
     (gen9ou + gen9uu + gen9ru) and weight every set equally (1/N). A coarse
     prior — the Bayesian updates fix it as evidence arrives.
  3. Fallback for species with no sets anywhere in the gen: uniform over
     BOUNDED legal move combos (first 8 legal moves, all C(k,4); k<4 → all of
     them) crossed with the species' abilities; nature neutral, EVs 0, no
     item, tera unknown. Coarse by design — HARD evidence (3.3) narrows it
     within a few turns.
"""
from __future__ import annotations

import copy
import itertools
import random
import re
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, List, Optional, Union

from legality import db, lookup, formats

from . import events as E          # typed events (3.1) — update() dispatches on these
from . import evidence as ev      # shared evidence engine (3.2) + SOFT calc bridge (3.4)


@dataclass
class Facts:
    """Per-mon observation registry — the "make unknown gamestates known"
    sheet (user, 2026-11-15): every turn adds information, and this is where it
    is held. ONE instance per slot (opponent slots: ``SlotBelief.facts``;
    own slots: ``BeliefState.own_facts``), reset when a NEW mon takes the slot.

    * ``hp`` / ``maxhp`` — from the log (switch-in line reveals both; every
      damage/heal line updates ``hp``). ``maxhp`` is ground truth in this fork.
    * ``statuses`` / ``boosts`` — live condition state (calc inputs for the
      damage window and the speed-order evidence).
    * ``moves_used`` — the revealed-move history (drives the Choice-lock signal;
      migrated from ``SlotBelief._used_moves`` in 3.4, 3.3 behaviour unchanged).
    * ``tera`` — the observed Tera type (HARD evidence already pinned it to the
      candidates in 3.3; kept here as the single observation record).
    * ``last_hit`` — the damage this mon last took from a MOVE hit (D48: move-
      source only). The MAGNITUDE the reflector would reflect; retained across
      turns as memory. But ``_move_with_state`` only hands it to the calc as
      ``lastHitDamage`` when the mon was hit THIS turn (``attacked_by_this_turn``
      non-empty) — the fork's reflector volatile is per-turn (``duration: 1``) and
      a STALE hit means the reflector fails (0), not 2× the old value.
    """

    hp: Optional[int] = None
    maxhp: Optional[int] = None
    statuses: set = _dc_field(default_factory=set)
    boosts: Dict[str, int] = _dc_field(default_factory=dict)
    moves_used: set = _dc_field(default_factory=set)
    tera: Optional[str] = None
    last_hit: Optional[int] = None
    # The CATEGORY ("Physical"/"Special") of the move that set ``last_hit`` —
    # the fork reflects only a MATCHING category (Counter/Metal Burst reflect a
    # physical hit; Mirror Coat/Comeuppance reflect a special one — moves.js
    # ``onDamagingHit``: ``getCategory(move) === "Physical"`` / ``"Special"``).
    # Set alongside last_hit in _apply_damage; persisted as memory like last_hit.
    last_hit_category: Optional[str] = None
    # T5 state (consumed by _move_with_state for state-BP moves)
    # avalanche/revenge gate (fork moves.ts:912-915 / 15086-15088:
    # `attackedBy.some(p => p.source === target && p.damage > 0 && p.thisTurn)`) —
    # the (side, slot) of the opponent mon(s) that hit THIS mon this turn with
    # damage > 0. PER-SOURCE: Avalanche/Revenge on target T doubles only if T
    # (not merely any mon) hit the user this turn. The observation exposes the
    # model-facing bool `hit_this_turn = len(set) > 0` (get_context), so the
    # net's dim is unchanged. Per-turn; cleared by turn_reset() / reset().
    attacked_by_this_turn: set = _dc_field(default_factory=set)
    last_move_failed: bool = False       # per-turn (fork: moveLastTurnResult)
    # Stockpile LAYERS — a fork VOLATILE (no duration): they persist across
    # turns, cleared only by switch-out (clearVolatiles) or a new mon.
    stockpile_layers: int = 0
    # Used a Protect-family move (Protect/Detect/Spiky Shield/King's Shield/
    # Guard Split) — PER-TURN (the fork clears the protect volatile at end of
    # turn, so it only blocks the turn it's used). Doubles-critical: a protected
    # opponent can't be hit, so attacking it wastes the move.
    used_protect: bool = False
    # The CURATED per-mon volatiles (D61) currently active on this mon —
    # toID set (taunt / encore / disguise / trapped / curse / destinybond /
    # gastroacid). Multi-turn by nature (Taunt 4 turns, Gastro Acid 5, …) so
    # they PERSIST across turns (turn_reset does not touch them) and are
    # cleared only by |-end| or a new mon (reset). Stockpile + Protect are
    # tracked by their own fields (layer count / per-turn flag).
    volatiles: set = _dc_field(default_factory=set)

    def reset(self) -> None:
        """Full wipe (a NEW mon takes the slot)."""
        self.hp = None
        self.maxhp = None
        self.statuses = set()
        self.boosts = {}
        self.moves_used = set()
        self.tera = None
        self.last_hit = None
        self.last_hit_category = None
        self.stockpile_layers = 0
        self.used_protect = False
        self.volatiles = set()
        self.turn_reset()

    def turn_reset(self) -> None:
        """Per-turn state wipe (called on E.Turn). Persists across turns:
        last_hit (reflector memory), stockpile_layers (fork: the stockpile
        volatile has no duration — layers survive turns, cleared only by a
        switch-out / new mon), hp, maxhp, statuses, boosts, moves_used, tera."""
        self.attacked_by_this_turn = set()
        self.last_move_failed = False
        self.used_protect = False

#: EV stat tokens as they appear in the DB spread strings ("252 HP / 216 Def")
_EV_TOKEN = {"hp": "hp", "atk": "atk", "def": "def", "spa": "spa", "spd": "spd", "spe": "spe"}

#: The Protect-family moves (doubles-critical: block attacks on the user for the
#: rest of the turn). We detect by MOVE NAME (already in MoveUsed) — the fork's
#: volatile differs per move (protect/detect -> the "protect" volatile, spikyshield
#: -> "spikyshield", guardsplit -> the protect flag; moves.js:14010/3579/17556/7991),
#: so the move name is the one stable signal. "used_protect" is set for the USER.
#: Minor edges (both rare-in-BSS, per-turn, low impact): a FAILED protect-family
#: use still sets it (these stalling moves almost always land), and Guard Split
#: marks the user (the ally it shields is not separately marked).
_PROTECT_FAMILY = frozenset({"protect", "detect", "kingsshield", "spikyshield", "guardsplit"})

#: max legal moves drawn for fallback combos (bounds candidate-space size)
_FALLBACK_MOVE_POOL = 8
_FALLBACK_SLOTS = 4

#: The HIT category each reflector requires (fork ``onDamagingHit`` checks
#: ``getCategory(move) === <this>`` — moves.js:3025-3060 / 2739-2760 / 11721).
#: A mismatch means the reflector FAILS (0), so ``lastHitDamage`` must NOT be fed
#: (the calc's ``lastHit==null -> 0`` path then fires). Keys are move ids (lower).
_REFLECTOR_REQUIRES = {
    "counter": "Physical", "metalburst": "Physical",
    "mirrorcoat": "Special", "comeuppance": "Special",
}


def _move_category(move_name: Optional[str], gen: int) -> Optional[str]:
    """The category (``Physical``/``Special``/``Status``) of a move, from the DB
    (source of truth for raw move data), PER-GEN (``gen`` = the @smogon/calc
    engine key). ``None`` for an unknown/absent move — the reflector category
    gate then can't confirm a match and stays closed. Only called on move-source
    hits (rare), so the per-call lookup is fine. Per-gen because category shifts
    across gens (Dragon Claw is Special in gen 1-3, Physical gen 4+)."""
    if not move_name:
        return None
    try:
        return lookup.get_move(move_name, gen).get("category")
    except KeyError:
        return None


def _move_with_state(move_name: str, facts: Optional[Facts],
                     defender: Optional[tuple] = None) -> Union[str, Dict]:
    """Build the move spec for the calc, attaching tracked state the attacker has.

    The calc reads what it needs (reflectors use ``lastHitDamage``; state-BP
    moves use ``hitThisTurn`` etc.).  Absent fields = no-state baseline (the
    calc's documented contract).  Returns a bare string when no state is set
    (zero overhead for the 99 % normal-move case).  P2 will add the remaining
    T5 fields to the ``state`` dict — same mechanism, no new special cases.

    ``defender`` — the ``(side, slot)`` of the mon THIS move is hitting. The
    Avalanche/Revenge gate is PER-SOURCE (fork ``attackedBy.some(p => p.source
    === target && p.damage > 0 && p.thisTurn)``): it doubles only if THAT target
    hit the user this turn, not merely any mon. So ``hitThisTurn`` is computed
    as ``defender in facts.attacked_by_this_turn``. A caller that does not know
    the target (``defender=None``) falls back to the was-hit-at-all bool — the
    over-report direction (never misses a legitimate double); the real evidence
    path always passes the target, so accuracy is preserved where it matters.

    Reflector gate (counter/mirrorcoat/metalburst/comeuppance) — the fork
    recreates the reflector volatile FRESH each turn (``beforeTurnCallback``,
    ``duration: 1``) and its ``onTry`` FAILS (``slot === null``) unless the user
    was hit by a damaging move THIS turn. ``last_hit`` (the magnitude) is
    intentionally retained across turns as reflector memory, but it must NOT be
    reflected STALE: a hit from a previous turn means the reflector fails (0),
    not 2× that stale hit. So ``lastHitDamage`` is attached only when the user
    was hit this turn (``attacked_by_this_turn`` non-empty — the same per-turn,
    move-source, damage>0 signal the Avalanche gate uses). At decision time in
    singles (no hit yet this turn) the gate is closed → the reflector reports
    base power (0); once the hit lands in the log the gate opens → 2×.

    Category gate (D55) — the fork reflects only a MATCHING category: Counter /
    Metal Burst reflect a PHYSICAL hit, Mirror Coat / Comeuppance a SPECIAL one
    (``onDamagingHit`` ``getCategory(move) === "Physical"`` / ``"Special"``).
    So ``lastHitDamage`` is attached only if the recorded hit category
    (``last_hit_category``) matches the reflector's requirement in
    ``_REFLECTOR_REQUIRES``. A mismatch (e.g. a special hit + Counter) means the
    reflector FAILS (0) — the calc's ``lastHit==null -> 0`` path fires — instead
    of over-reporting 2× the mismatched hit. ``lastHitDamage`` is a REFLECTOR-ONLY
    field (the calc reads it only for the four reflectors), so it is attached ONLY
    to a reflector move — non-reflector moves never carry it.
    """
    if facts is None:
        return move_name
    state: Dict[str, Any] = {}
    if facts.last_hit is not None and facts.attacked_by_this_turn:
        required = _REFLECTOR_REQUIRES.get(move_name.lower())
        if required is not None and facts.last_hit_category == required:
            state["lastHitDamage"] = facts.last_hit
    if facts.attacked_by_this_turn:
        if defender is None or defender in facts.attacked_by_this_turn:
            state["hitThisTurn"] = True
    if facts.last_move_failed:
        state["lastMoveFailed"] = True
    if facts.stockpile_layers > 0:
        state["stockpileLayers"] = facts.stockpile_layers
    if not state:
        return move_name
    return {"name": move_name, **state}


def _fmt_gen(fmt: str) -> int:
    m = re.match(r"gen(\d+)", fmt)
    if not m:
        raise ValueError(f"cannot read gen from format id {fmt!r}")
    return int(m.group(1))


def _parse_evs(spread: Optional[str]) -> Dict[str, int]:
    """'252 HP / 216 Def / 40 Spe' -> {'hp': 252, 'def': 216, 'spe': 40}."""
    if not spread:
        return {}
    out: Dict[str, int] = {}
    for part in spread.split("/"):
        part = part.strip()
        if not part:
            continue
        bits = part.split(" ")
        if len(bits) != 2:
            raise ValueError(f"bad EV spread token {part!r} in {spread!r}")
        val, stat = bits
        key = _EV_TOKEN.get(stat.lower())
        if key is None:
            raise ValueError(f"unknown EV stat {stat!r} in {spread!r}")
        out[key] = int(val)
    return out


def _candidate_from_set(row: Dict, weight: float) -> Dict:
    return {
        "moves": list(row["moves"]),
        "item": row["item_id"] or "",
        "ability": row["ability_id"] or "",
        "nature": row["nature"] or "Hardy",
        "evs": _parse_evs(row.get("evs")),
        "tera": row.get("tera_type") or None,
        "weight": float(weight),
    }


def _tier_pool(species: str, fmt: str) -> Optional[List[Dict]]:
    """Same-gen tier sets (equal weight) when the format itself has none."""
    gen = _fmt_gen(fmt)
    conn = db.connect()
    try:
        tiers = [
            r["format_id"]
            for r in conn.execute("SELECT DISTINCT format_id FROM strategy ORDER BY format_id")
        ]
    finally:
        conn.close()
    pool: List[Dict] = []
    for f in tiers:
        try:
            if _fmt_gen(f) != gen:
                continue
        except ValueError:
            continue
        pool.extend(lookup.get_common_sets(species, f))
    return pool or None


def _same_gen_legal_moves(species: str, fmt: str) -> List[str]:
    """Legal moves for the species from ANY same-gen format in the DB.

    Covers formats with no format-specific legality rows of their own (e.g.
    gen9championsou before its legality ingest): a gen-9 species that has a
    learnset somewhere in gen 9 gets a usable (approximate) candidate space.
    """
    fid = lookup.to_id(species)
    gen = _fmt_gen(fmt)
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT m.move_id FROM format_form_move m "
            "JOIN format f ON f.id = m.format_id "
            "WHERE m.form_id = ? AND f.gen = ? ORDER BY m.move_id",
            (fid, gen)).fetchall()
        return [r["move_id"] for r in rows]
    finally:
        conn.close()


def _fallback_candidates(species: str, fmt: str) -> List[Dict]:
    legal = lookup.legal_moves(species, fmt) or _same_gen_legal_moves(species, fmt)
    if not legal:
        return []
    sp = lookup.get_species(species, _fmt_gen(fmt))
    abilities = [a["id"] for a in sp.get("abilities") or []] or [""]
    pool = legal[:_FALLBACK_MOVE_POOL]
    combos = [list(c) for c in itertools.combinations(pool, min(_FALLBACK_SLOTS, len(pool)))]
    n = len(combos) * len(abilities)
    w = 1.0 / n
    out: List[Dict] = []
    for moves in combos:
        for ability in abilities:
            out.append(
                {
                    "moves": moves,
                    "item": "",
                    "ability": ability,
                    "nature": "Hardy",
                    "evs": {},
                    "tera": None,
                    "weight": w,
                }
            )
    return out


def build_prior(species: str, fmt: str) -> List[Dict]:
    """Candidate list for one opponent slot (weights sum to 1, all > 0).

    Raises ``ValueError`` when the species has no usable prior at all in the
    format (no sets AND no legal moves) — a loud failure, never an empty
    belief (PLAN.md Step 3 architecture: the posterior can never be empty).
    """
    rows = lookup.get_common_sets(species, fmt)
    if rows:
        return [_candidate_from_set(r, r["probability"]) for r in rows]

    tier = _tier_pool(species, fmt)
    if tier:
        w = 1.0 / len(tier)
        return [_candidate_from_set(r, w) for r in tier]

    fb = _fallback_candidates(species, fmt)
    if fb:
        return fb

    raise ValueError(f"no prior for {species!r} in {fmt!r}: no sets and no legal moves")


class SlotBelief:
    """Posterior over candidates for ONE opponent slot."""

    def __init__(self, slot: str, species: str, candidates: List[Dict]):
        if not slot.startswith("p2"):
            raise ValueError(f"belief slots are opponent-side (p2*): got {slot!r}")
        if not candidates:
            raise ValueError(f"empty candidate list for {species!r} in slot {slot!r}")
        if any(c["weight"] <= 0 for c in candidates):
            raise ValueError(f"non-positive weight in {slot!r} candidates")
        self.slot = slot
        self.species = species
        self.candidates = list(candidates)
        # per-mon observation registry (3.4 fact sheet; reset when a NEW mon
        # takes the slot). moves_used drives the Choice-lock signal (3.3).
        self.facts = Facts()
        self._choice_locked = False
        self.normalize()

    @property
    def total(self) -> float:
        return sum(c["weight"] for c in self.candidates)

    def normalize(self) -> None:
        t = self.total
        if t <= 0:
            raise ValueError(f"{self.slot!r}: cannot normalize — total weight {t}")
        for c in self.candidates:
            c["weight"] /= t

    @property
    def posterior(self) -> List[float]:
        return [c["weight"] for c in self.candidates]

    def mass(self, predicate) -> float:
        """Total posterior mass over candidates satisfying ``predicate``."""
        return sum(c["weight"] for c in self.candidates if predicate(c))

    def sample(self, n: int, seed: Optional[int] = None) -> List[Dict]:
        """Draw ``n`` full-set candidates from this slot's posterior — a multinomial
        draw (with replacement) over the candidates, using a SEEDABLE RNG so the bot
        can reproduce a decision. Full sets (not per-move marginals) is the point:
        within-set move CORRELATIONS are preserved by construction — a Wish set you
        draw already contains its Protect (user invariant, 3.3/3.5).

        Reproducible: the same ``seed`` + the same posterior -> the same samples
        (the RNG is a fresh ``random.Random(seed)``, so samples never leak state
        across calls or slots). Returns COPIES, so mutating a sample cannot touch
        the belief's live candidates.

        This is the primitive for CORRELATED / JOINT reasoning: a question like
        "does this mon run Wish AND Protect" must be answered from the full sets
        (here), NOT by multiplying ``move_probabilities()`` marginals (see there).
        """
        if n < 0:
            raise ValueError(f"sample count must be >= 0, got {n}")
        rng = random.Random(seed)
        idxs = rng.choices(range(len(self.candidates)),
                           weights=[c["weight"] for c in self.candidates], k=n)
        return [copy.deepcopy(self.candidates[i]) for i in idxs]

    def move_probabilities(self) -> Dict[str, float]:
        """Posterior mass over the sets CONTAINING each move — a per-move MARGINAL,
        order-blind (the 3.3 moveset-is-a-set invariant). ``value(m)`` = the posterior
        mass of the full sets that carry move ``m``.

        Two things this is NOT (both are traps that break the Wish⇒Protect reasoning):
          * Not a per-turn USAGE probability. It says the move is ON the mon, not that
            the mon will use it this turn (a mon carries 4 moves, uses 1). Predicting
            the turn move is the POLICY's job (belief + move order + Choice lock + PP).
          * Not a joint. These are marginals — DO NOT multiply two of them to get
            P(both): that assumes independence and is exactly wrong when the moves are
            correlated (Wish+Protect live in the SAME sets). P(Wish∧Protect)=0.5 but
            P(Wish)·P(Protect)=0.25 in the 3.5 audit case. For any joint / correlated
            question use ``sample()`` and read the full sets (``test_marginals_are_not
            independent`` pins this).

        PP is unobservable in the log, so it is deliberately ignored. Every value is in
        [0, 1]; they sum to the posterior-expected number of moves (4 for a 4-move set),
        not 1 — a mon carries several moves, and the mass of a move is the mass of the
        sets that hold it.
        """
        out: Dict[str, float] = {}
        for c in self.candidates:
            w = c["weight"]
            for m in set(c.get("moves", [])):
                out[m] = out.get(m, 0.0) + w
        return out

    def update(self, event) -> bool:
        """Apply one observed event to this slot's posterior (3.3 HARD evidence).

        Returns True if the event was applied as set-belief evidence, False if it
        is not (damage/status/boost/field lines belong to 3.4/3.6). The posterior
        stays normalized and non-empty after every call (the evidence.py EPS floor
        guarantees it — even an off-DB move never collapses the belief).

        Choice-lock: the SAME move observed a second time on this mon means the
        item is locked to a Choice item (a non-Choice item allows switching moves
        freely) — applied once, on the second sighting.
        """
        if isinstance(event, E.MoveUsed):
            move = event.move
            ev.apply_evidence(self.candidates, ev.revealed_move(move))
            if move in self.facts.moves_used and not self._choice_locked:
                ev.apply_evidence(self.candidates, ev.choice_lock())
                self._choice_locked = True
            self.facts.moves_used.add(move)
            return True
        if isinstance(event, E.AbilityRevealed):
            ev.apply_evidence(self.candidates, ev.revealed_ability(event.ability))
            return True
        if isinstance(event, E.ItemRevealed):
            ev.apply_evidence(self.candidates, ev.revealed_item(event.item))
            return True
        if isinstance(event, E.TeraUsed):
            ev.apply_evidence(self.candidates, ev.tera_used(event.tera_type))
            return True
        return False


class BeliefState:
    """The bot's belief about the opponent side (p2 slots) in one battle.

    Also carries the parts of the game state that are OBSERVABLE but not hidden
    (3.4 fact sheet, PLAN.md §3a 3.4-5): the own side (p1*) — exactly known to
    the bot, so it enters the calc as exact mon specs, never as a distribution
    (3.4-2) — and the shared field (weather / terrain / Tailwind).
    """

    def __init__(self, fmt: str, own_team: Optional[Dict[str, Dict]] = None):
        self.fmt = fmt
        self._slots: Dict[str, SlotBelief] = {}
        # own side (p1*): slot -> exact mon spec {'name', 'nature', 'evs',
        # 'item', 'ability', 'tera'?}. The bot knows its own team.
        self.own_team: Dict[str, Dict] = dict(own_team or {})
        self.own_facts: Dict[str, Facts] = {}
        # D68 (member mapping): the log's slot letters are FIELD positions (in
        # singles every switch-in lands on p1a), NOT team members. So the
        # belief tracks WHO occupies each field slot (species id) and maps
        # per-member active/fainted state onto the own_team MEMBERS (preview
        # order p1a..p1f) in get_context(). own_fainted_species is
        # order-preserving + deduped (a fainted mon is gone for the battle).
        self.own_active_species: Dict[str, str] = {}
        self.own_fainted_species: List[str] = []
        # shared field — the calc's field inputs (3.4):
        self.field = {
            "weather": None, "terrain": None,
            "tailwind_side": None, "tailwind_turns": 0,
            # pseudo-weather (field-wide): True = active
            "trickroom": False, "wonderroom": False,
            "gravity": False, "magicroom": False,
            # side conditions: side -> active (True/False)
            "reflect": {"p1": False, "p2": False},
            "lightscreen": {"p1": False, "p2": False},
            "auroraveil": {"p1": False, "p2": False},
            # entry hazards: side -> layers (0 = none)
            "spikes": {"p1": 0, "p2": 0},
            "toxicspikes": {"p1": 0, "p2": 0},
            "stealthrock": {"p1": False, "p2": False},
            "stickyweb": {"p1": False, "p2": False},
            "safeguard": {"p1": False, "p2": False},
        }
        # battle-level state
        self.turn: int = 0
        self.fainted: Dict[str, int] = {"p1": 0, "p2": 0}
        # PER-SLOT fainted flag (S4.1: the registry's reserved `fainted` column
        # is now real): slot id -> True iff the mon that was on that slot has
        # fainted and the slot has NOT since been re-occupied. Set on Faint,
        # cleared on the next SwitchIn into the same slot (a new mon is a new
        # life). This is what lets the net tell "benched, untouched" (fainted
        # = 0) from "gone" (fainted = 1) — both otherwise read hp_frac 0.
        self.fainted_slots: Dict[str, bool] = {}
        self.active: Dict[str, Optional[str]] = {"p1": None, "p2": None}
        # opponent team composition (from the |poke| preview): the FULL team in
        # preview order — the bench the net plans coverage against, not just the
        # active mon. In singles the field slots only ever hold the active (p2a),
        # so the 5 bench mons are known ONLY from this preview list. 
        # opp_fainted_species tracks which team mons are GONE (fainted), so the
        # alive bench excludes them (the opponent analog of own_fainted_species,
        # D68 — "benched, untouched" vs "gone").
        self.opp_team: List[str] = []
        self.opp_fainted_species: List[str] = []
        # per-turn move-order buffer (speed-order evidence, 3.4-4): entries are
        # ('m', side, slot, move) and order-breakers ('sw',) / ('break',).
        # Flushed on Turn / BattleEnd — the moves of ONE turn only.
        self._turn_buffer: List[tuple] = []

    def add_opponent(self, slot: str, species: str) -> SlotBelief:
        """Register an opponent mon (on switch-in) and give it its prior.

        If the slot already holds a DIFFERENT mon (a faint + re-switch into the
        same active slot, common in 3v3), the belief is RE-INITIALIZED to the new
        mon's prior — a new mon is a new set (and a fresh move history). Same
        species = the same mon still on field -> keep the existing posterior.
        """
        existing = self._slots.get(slot)
        if existing is not None and existing.species == species:
            return existing
        sb = SlotBelief(slot, species, build_prior(species, self.fmt))
        self._slots[slot] = sb
        return sb

    # ═══════════════════════════════════════════════════════════════════════════
    #  get_context() — the single API the AI (Step 4/5) consumes.
    #  JSON-serializable, complete battle state, one call.
    # ═══════════════════════════════════════════════════════════════════════════

    def get_context(self) -> Dict:
        """Full battle-state snapshot for the observation encoder (Step 4) and
        the policy network (Step 5). JSON-safe (all primitives / lists / dicts).

        Structure:
          * ``turn`` — the current turn number
          * ``fainted`` — {"p1": int, "p2": int}
          * ``active``  — {"p1": slot or None, "p2": slot or None}
          * ``field``   — weather, terrain, Trick Room, screens, hazards, Tailwind
          * ``own``     — per-slot: HP fraction, status, boosts, moves, state
          * ``opponent``— per-slot: species, HP fraction, status, boosts, moves,
                         posterior (top-K), top moves, confidence
        """
        def _mon_block(facts: Optional[Facts], team_spec: Optional[Dict]) -> Dict:
            if facts is None:
                return {"hp_frac": 0.0, "statuses": [], "boosts": {},
                        "moves": [], "last_hit": 0, "hit_this_turn": False,
                        "last_move_failed": False, "stockpile": 0,
                        "used_protect": False, "volatiles": [], "maxhp": None}
            hp_frac = self._hp_fraction(facts) or 0.0
            return {
                "hp_frac": round(hp_frac, 4),
                "statuses": sorted(facts.statuses),
                "boosts": dict(facts.boosts),
                "moves": sorted(facts.moves_used),
                "tera": facts.tera or "",
                "last_hit": facts.last_hit or 0,
                "hit_this_turn": len(facts.attacked_by_this_turn) > 0,
                "last_move_failed": facts.last_move_failed,
                "stockpile": facts.stockpile_layers,
                "used_protect": facts.used_protect,
                # curated per-mon volatiles (D61), sorted for determinism
                "volatiles": sorted(facts.volatiles),
                # M2b data-flow: the mon's max HP. PUBLIC (the switch-in line
                # reveals it; the belief tracks it as ground truth), so no leak.
                # It is CONTEXT, not OBSERVATION — the encoder reads specific
                # fields and ignores this one, so the registry checksum is
                # unchanged and existing checkpoints stay loadable. Needed by
                # the MCTS step to turn a predicted-damage delta into an
                # HP-fraction update (hp_frac - damage / maxhp). None when the
                # mon's max HP is not yet known (bench placeholders, pre-first-
                # switch opponent mon) — step must tolerate that.
                "maxhp": facts.maxhp,
            }

        ctx: Dict = {
            "turn": self.turn,
            "fainted": dict(self.fainted),
            "active": dict(self.active),
            "field": {
                "weather": self.field["weather"] or "",
                "terrain": self.field["terrain"] or "",
                "trickroom": self.field["trickroom"],
                "wonderroom": self.field["wonderroom"],
                "tailwind_side": self.field["tailwind_side"] or "",
                "reflect": {k: v for k, v in self.field["reflect"].items() if v},
                "lightscreen": {k: v for k, v in self.field["lightscreen"].items() if v},
                "auroraveil": {k: v for k, v in self.field["auroraveil"].items() if v},
                "spikes": dict(self.field["spikes"]),
                "toxicspikes": dict(self.field["toxicspikes"]),
                "stealthrock": {k: v for k, v in self.field["stealthrock"].items() if v},
                "stickyweb": {k: v for k, v in self.field["stickyweb"].items() if v},
                "safeguard": {k: v for k, v in self.field["safeguard"].items() if v},
            },
            "own": {},
            "opponent": {},
        }
        # own side (exactly known) — MEMBER-keyed (D68): the own_team slots
        # p1a..p1f are the six team MEMBERS in preview order. The log's slot
        # letters are FIELD positions (in singles every switch-in lands on
        # p1a), so a member's active/hp/status come from the field slot its
        # SPECIES occupies (Species Clause => species -> member is
        # unambiguous). When NO own-side switch-in has been observed (unit-
        # test fact injection / a log without a preview) the legacy slot-keyed
        # semantics apply, so direct fact reads keep working.
        own_event_slot = self.active.get("p1")
        member_mapping = bool(self.own_active_species)
        active_sp = (self.own_active_species.get(own_event_slot)
                     if (member_mapping and own_event_slot) else None)
        active_member = None
        if active_sp is not None:
            for slot, spec in self.own_team.items():
                if lookup.to_id(spec.get("name", "")) == active_sp:
                    active_member = slot
                    break
        for slot, spec in self.own_team.items():
            if member_mapping:
                is_active = (slot == active_member)
                facts = self.own_facts.get(own_event_slot) if is_active else None
                sp = lookup.to_id(spec.get("name", ""))
                fainted = sp in self.own_fainted_species
            else:
                is_active = (own_event_slot is not None and slot == own_event_slot)
                facts = self.own_facts.get(slot)
                fainted = bool(self.fainted_slots.get(slot, False))
            blk = ctx["own"][slot] = _mon_block(facts, spec)
            blk["name"] = spec.get("name", "")
            blk["ability"] = spec.get("ability", "")
            blk["item"] = spec.get("item", "")
            # D74: own-side moves. ``moves_used`` = the revealed "used so far"
            # set (the old ``moves`` semantics, kept for BC consumers / tests);
            # ``moves`` = the FULL known moveset when the spec carries one
            # (self-play: the agent built the team — fully known from turn 1),
            # else it stays the used-so-far set (BC corpus / specs without
            # moves). The encoder fills the ``own_moves`` channel from ``moves``.
            blk["moves_used"] = list(blk.get("moves", []) or [])
            spec_moves = spec.get("moves") or []
            if spec_moves:
                blk["moves"] = [str(m) for m in spec_moves][:4]
            # fainted flag: "benched, untouched" (0) vs "gone" (1) — both
            # otherwise read hp_frac 0 (the net must not confuse them)
            ctx["own"][slot]["fainted"] = fainted
            ctx["own"][slot]["active"] = is_active
        # the encoder's own-side active flag compares block slots to
        # ctx["active"]["p1"] — under member mapping that must be the MEMBER
        # slot (the event slot p1a does not address a member), else the active
        # flag lands on the wrong row. (p2 stays field-slot keyed.)
        if (member_mapping and active_member is not None
                and ctx["active"].get("p1") is not None):
            ctx["active"]["p1"] = active_member
        # opponent side (belief)
        for slot, sb in sorted(self._slots.items()):
            block = _mon_block(sb.facts, None)
            block["species"] = sb.species
            block["fainted"] = bool(self.fainted_slots.get(slot, False))
            # posterior summary (top-K by weight)
            sorted_cands = sorted(enumerate(sb.candidates), key=lambda x: -x[1]["weight"])
            top_k = 5
            block["posterior"] = [
                {"ability": c["ability"], "moves": c["moves"], "weight": round(w, 4)}
                for _, c in sorted_cands[:top_k]
                for w in [c["weight"]]
            ]
            # top moves (marginal over the posterior)
            move_counts: Dict[str, float] = {}
            for c in sb.candidates:
                for m in c["moves"]:
                    move_counts[m] = move_counts.get(m, 0) + c["weight"]
            block["top_moves"] = [
                m for m, _ in sorted(move_counts.items(), key=lambda x: -x[1])[:4]
            ]
            # confidence (1 - normalized entropy), clamped to [0,1] so a
            # floating-point entropy a hair above 1.0 can't render as -0.0 / >1.0
            import math
            p = sb.posterior
            ent = -sum(pi * math.log2(pi) for pi in p if pi > 0) / max(1, math.log2(len(p)))
            block["confidence"] = round(max(0.0, min(1.0, 1.0 - ent)), 4)
            ctx["opponent"][slot] = block
        # opponent BENCH (the team composition minus the active) — so the net
        # sees the FULL team for coverage planning, not just the active mon.
        # In singles the field slots only ever hold the active (p2a), so the
        # bench mons are known ONLY from the |poke| preview (self.opp_team).
        # The active is already placed at its field slot (p2a) above from its
        # own belief (species + HP + posterior); we place the ALIVE bench at
        # p2b..p2f in preview order, excluding the active and any fainted mon.
        # This REUSES the existing 6 opp_species slots — no new dimension, so
        # the registry checksum is unchanged and checkpoints stay loadable.
        active_p2 = self.active.get("p2")
        active_sp = (self._slots[active_p2].species
                     if (active_p2 is not None and active_p2 in self._slots)
                     else None)
        bench = [sp for sp in self.opp_team
                 if sp != active_sp and sp not in self.opp_fainted_species]
        for i, sp in enumerate(bench[:5]):
            slot = f"p2{'bcdef'[i]}"
            if slot in ctx["opponent"]:
                continue  # never clobber a real field-slot belief (doubles, etc.)
            ctx["opponent"][slot] = {
                "hp_frac": 0.0, "statuses": [], "boosts": {}, "moves": [],
                "tera": "", "last_hit": 0, "hit_this_turn": False,
                "last_move_failed": False, "stockpile": 0, "used_protect": False,
                "volatiles": [], "species": sp, "fainted": False,
                "posterior": [], "top_moves": [], "confidence": 0.0,
                # bench placeholders have no observed max HP yet — keep the key
                # present (maxhp None) so every mon block shares one schema and
                # consumers (the MCTS step) never KeyError on a bench slot.
                "maxhp": None,
            }
        return ctx

    # ══════════════════════════ event dispatch (3.3 + 3.4) ══════════════════════

    def update(self, event) -> bool:
        """Apply one event: fact/field bookkeeping (both sides, 3.4) AND
        set-belief evidence (3.3 HARD + 3.4 SOFT).

        Returns True only when a slot's SET POSTERIOR changed (the 3.3 contract
        — the 3.3 tests pin it, incl. Weather/Turn/Damage-without-source
        reading False). Fact and field bookkeeping happens regardless and is
        observable on ``own_facts`` / ``slot.facts`` / ``field``.
        """
        if isinstance(event, E.TeamPreview):
            # The opponent's (p2) team composition — the bench the net plans
            # coverage against, not just the active mon. Order-preserving +
            # deduped (a legal team has no duplicate species; a malformed repeat
            # is a no-op). This establishes the PRIOR team knowledge; it does not
            # change any slot's SET POSTERIOR, so it returns False (no 3.3 update).
            # The own side (p1) is fully known from the own_team spec, so p1
            # previews are inert here.
            if event.side == "p2" and event.species and event.species not in self.opp_team:
                self.opp_team.append(event.species)
            return False
        if isinstance(event, E.SwitchIn):
            self._turn_buffer.append(("sw",))
            if event.side == "p2":
                self.add_opponent(event.slot, event.species)
            self.active[event.side] = event.slot
            # a new mon on the slot: the slot is no longer "fainted" (the
            # fainted flag describes the CURRENT occupant's state)
            self.fainted_slots.pop(event.slot, None)
            self._set_switch_facts(event)
            return False
        if isinstance(event, E.SwitchOut):
            self._turn_buffer.append(("sw",))  # a switch breaks move-order semantics
            return False
        if isinstance(event, E.Faint):
            self._set_hp(event.side, event.slot, 0)
            self.fainted[event.side] = self.fainted.get(event.side, 0) + 1
            self.fainted_slots[event.slot] = True  # the occupant is gone
            if event.side != "p2":
                # D68: the fainted mon is a GONE member of the own team (the
                # net must distinguish "benched, untouched" from "gone" —
                # both otherwise read hp_frac 0)
                sp = self.own_active_species.get(event.slot)
                if sp is not None and sp not in self.own_fainted_species:
                    self.own_fainted_species.append(sp)
            else:
                # opponent (p2): the fainted mon is a GONE member of the team (the
                # net must not count it as switchable bench). Its species is the
                # occupant of the fainted field slot (p2a in singles); a fainted
                # mon is removed from the alive bench in get_context().
                sb = self._slots.get(event.slot)
                if sb is not None and sb.species not in self.opp_fainted_species:
                    self.opp_fainted_species.append(sb.species)
            if self.active.get(event.side) == event.slot:
                self.active[event.side] = None
            self._turn_buffer.append(("break",))
            return False
        if isinstance(event, E.MoveUsed):
            self._turn_buffer.append(("m", event.side, event.slot, event.move))
            # Stockpile (fork moves.ts:17956-17966 + conditions.ts volatile): each
            # successful use adds one layer, cap 3 (the 4th use fails — onTry /
            # onRestart return false — and adds nothing). The layers feed the
            # calc's Spit Up (BP 100*layers) via _move_with_state. A FAILED use
            # leaves one phantom layer (blocked Stockpile — rare-in-BSS edge,
            # accepted: rollback can't tell a capped failure from a real one).
            if event.move == "stockpile":
                facts = self._facts_for(event.side, event.slot)
                if facts is not None and facts.stockpile_layers < 3:
                    facts.stockpile_layers += 1
            # Protect-family (doubles-critical): mark the user as protected for
            # the turn (cleared by turn_reset). The observation is the signal —
            # "don't waste an attack on a protected target." Own-side facts are
            # created on-demand (same pattern as _set_hp / the MoveFailed branch).
            if event.move in _PROTECT_FAMILY:
                facts = self._facts_for(event.side, event.slot)
                if facts is None and event.side != "p2":
                    facts = self.own_facts.setdefault(event.slot, Facts())
                if facts is not None:
                    facts.used_protect = True
            # Reveal the move on the USER's own-side facts (belief moves_used).
            # The opponent side does this in SlotBelief.update; the own side must
            # mirror it so get_context's own "moves" block carries the
            # revealed-move history (the BC action list offers the acting mon's
            # known moves; the observation carries the used set). Own facts are
            # created on-demand; a NEW mon's moves_used is wiped by
            # _set_switch_facts, so per-field-slot history stays per-occupant.
            if event.side != "p2":
                facts = self._facts_for(event.side, event.slot)
                if facts is None:
                    facts = self.own_facts.setdefault(event.slot, Facts())
                facts.moves_used.add(event.move)
            return self._route_set_evidence(event)
        if isinstance(event, E.MoveFailed):
            # stompingtantrum/temperflare gate (fork moves.ts:18050/19186:
            # `moveLastTurnResult === false`): the user's last move failed ->
            # next turn's BP x2. Set on the mon whose move failed; cleared by
            # turn_reset() on E.Turn. Facts only (a failure is not set evidence).
            facts = self._facts_for(event.side, event.slot)
            if facts is None and event.side != "p2":
                facts = self.own_facts.setdefault(event.slot, Facts())
            if facts is not None:
                facts.last_move_failed = True
            return False
        if isinstance(event, E.VolatileChanged):
            # Curated per-mon volatile (D61): Taunt / Encore / Disguise / Trapped /
            # Curse / Destiny Bond / Gastro Acid. Multi-turn state on the slot's
            # mon -> Facts (persists across turns; a new mon gets fresh Facts).
            # Facts only — a volatile on the mon is not SET evidence (the adapter
            # already filtered to the curated set; unknown slots are a no-op).
            facts = self._facts_for(event.side, event.slot)
            if facts is None and event.side != "p2":
                facts = self.own_facts.setdefault(event.slot, Facts())
            if facts is not None:
                if event.active:
                    facts.volatiles.add(event.volatile)
                else:
                    facts.volatiles.discard(event.volatile)
            return False
        if isinstance(event, (E.AbilityRevealed, E.ItemRevealed)):
            return self._route_set_evidence(event)
        if isinstance(event, E.TeraUsed):
            self._record_tera(event)
            return self._route_set_evidence(event)
        if isinstance(event, E.DamageObserved):
            return self._apply_damage(event)
        if isinstance(event, E.HealObserved):
            # The heal line is `hp/maxhp`: refresh maxhp FIRST (a heal can only
            # reveal the SAME mon's true maxhp — never a smaller one — so take
            # the max, never shrink a known maxhp), THEN _set_hp clamps hp into
            # [0, maxhp]. Without this, a stale maxhp + a heal could leave
            # hp > maxhp in the belief (the blind-play root cause).
            h_side, h_slot = event.target_side, event.target_slot
            h_facts = self._facts_for(h_side, h_slot)
            if h_facts is None and h_side != "p2":
                h_facts = self.own_facts.setdefault(h_slot, Facts())
            if h_facts is not None and event.maxhp:
                h_facts.maxhp = max(h_facts.maxhp or 0, event.maxhp)
            self._set_hp(h_side, h_slot, event.hp_after)
            return False
        if isinstance(event, (E.StatusApplied, E.StatusCured)):
            self._apply_status(event)
            return False
        if isinstance(event, E.BoostChanged):
            self._apply_boost(event)
            return False
        if isinstance(event, E.WeatherChanged):
            # the clear-signal ('none') / empty means no weather -> None in the field
            w = event.weather
            self.field["weather"] = None if (not w or w == "none") else w
            return False
        if isinstance(event, E.TerrainChanged):
            t = event.terrain
            self.field["terrain"] = None if (not t or t == "none") else t
            return False
        if isinstance(event, E.TailwindChanged):
            if event.active:
                self.field["tailwind_side"] = event.side
                self.field["tailwind_turns"] = 4  # fork duration (6 w/ Persistent; fuzz will flag)
            elif event.side == self.field.get("tailwind_side"):
                self.field["tailwind_side"] = None
                self.field["tailwind_turns"] = 0
            return False
        if isinstance(event, E.FieldEffectChanged):
            return self._apply_field_effect(event)
        if isinstance(event, (E.Turn, E.BattleEnd)):
            applied = self._flush_speed_buffer()  # the moves of the turn just ended
            if self.field.get("tailwind_side") and self.field.get("tailwind_turns", 0) > 0:
                self.field["tailwind_turns"] -= 1
                if self.field["tailwind_turns"] <= 0:
                    self.field["tailwind_side"] = None
            if isinstance(event, E.Turn):
                self.turn = event.n
                # per-turn state wipe for all tracked mons (opponent + own)
                for sb in self._slots.values():
                    sb.facts.turn_reset()
                for f in self.own_facts.values():
                    f.turn_reset()
            return applied
        return False

    def _route_set_evidence(self, event) -> bool:
        """3.3 routing: reveal-style events apply to the p2 slot they name."""
        slot = getattr(event, "slot", None)
        if not slot or not slot.startswith("p2"):
            return False
        try:
            return self.slot(slot).update(event)
        except KeyError:
            return False

    # ═══════════════════════════ 3.4 fact bookkeeping ═══════════════════════════

    def _facts_for(self, side: str, slot: str) -> Optional[Facts]:
        if side == "p2":
            sb = self._slots.get(slot)
            return sb.facts if sb is not None else None
        return self.own_facts.get(slot)

    def _set_hp(self, side: str, slot: str, hp: int) -> None:
        facts = self._facts_for(side, slot)
        if facts is None and side != "p2":
            facts = self.own_facts.setdefault(slot, Facts())
        if facts is not None:
            facts.hp = self._clamp_hp(hp, facts.maxhp)

    def _set_switch_facts(self, ev: E.SwitchIn) -> None:
        """Switch-in line is `hp/maxhp`: set (or refresh) the slot's HP facts.

        Same mon re-entering (switch-out then in): HP and status PERSIST
        (game rule), boosts reset (switch-out clears them). A NEW mon: the sheet
        resets — a new mon is a new set, a new HP line, a new move history.
        """
        if ev.side == "p2":
            sb = self._slots.get(ev.slot)
            if sb is None:
                return
            facts, known = sb.facts, sb.species
        else:
            facts = self.own_facts.setdefault(ev.slot, Facts())
            # identity = the last occupant's SPECIES on this FIELD slot (D68 —
            # the log's slot letters are field positions, not team members,
            # and the own_team spec is the MEMBER roster: it must NOT be
            # rewritten by field-slot occupants).
            known = self.own_active_species.get(ev.slot)
        same_mon = known is not None and lookup.to_id(ev.species) == known and facts.hp is not None
        if same_mon:
            if ev.maxhp:
                facts.maxhp = ev.maxhp
            if ev.hp:
                facts.hp = self._clamp_hp(ev.hp, facts.maxhp)
            facts.boosts = {}
            # switch-out cleared the mon's volatiles (fork clearVolatiles):
            # Stockpile layers are gone even for a re-entering mon
            facts.stockpile_layers = 0
            return
        facts.reset()
        facts.maxhp = ev.maxhp or None
        facts.hp = self._clamp_hp(ev.hp or ev.maxhp or None, facts.maxhp)
        # 3.6 — the switch-in line reveals the mon's max HP: apply it as EV
        # evidence on the p2 slot (one-sided, IV-safe, no false rule-outs).
        if ev.side == "p2" and facts.maxhp:
            self._apply_maxhp_evidence(ev.slot, facts.maxhp)
        if ev.side != "p2":
            # record the new field-slot occupant (drives the same-mon check on
            # the NEXT switch-in and the Faint handler's fainted-member set)
            self.own_active_species[ev.slot] = lookup.to_id(ev.species)

    def _apply_status(self, ev) -> None:
        facts = self._facts_for(ev.side, ev.slot)
        if facts is None and ev.side != "p2":
            facts = self.own_facts.setdefault(ev.slot, Facts())
        if facts is None:
            return
        if isinstance(ev, E.StatusApplied):
            facts.statuses.add(ev.status)
        else:
            facts.statuses.discard(ev.status)

    def _apply_boost(self, ev: E.BoostChanged) -> None:
        facts = self._facts_for(ev.side, ev.slot)
        if facts is None and ev.side != "p2":
            facts = self.own_facts.setdefault(ev.slot, Facts())
        if facts is None:
            return
        cur = max(-6, min(6, facts.boosts.get(ev.stat, 0) + ev.delta))
        if cur == 0:
            facts.boosts.pop(ev.stat, None)  # keep the sheet clean (calc: absent = 0)
        else:
            facts.boosts[ev.stat] = cur

    def _record_tera(self, ev: E.TeraUsed) -> None:
        facts = self._facts_for(ev.side, ev.slot)
        if facts is not None:
            facts.tera = ev.tera_type

    def _apply_field_effect(self, ev: E.FieldEffectChanged) -> bool:
        """Track a field/side effect (Trick Room, Reflect, Spikes, …).
        Facts-only (no set evidence — a screen is not set evidence)."""
        name = ev.name
        side = ev.side or None
        # field-wide (pseudo-weather)
        if side is None:
            if name in self.field:
                self.field[name] = ev.active
            return False
        # per-side: boolean conditions
        if name in ("reflect", "lightscreen", "auroraveil", "stealthrock", "stickyweb", "safeguard"):
            if name in self.field:
                self.field[name][side] = ev.active
            return False
        # per-side: layered (spikes 1-3, toxic spikes 1-2). The fork emits ONE
        # |-sidestart line PER LAYER with no count in the line (moves.js:17536-17543:
        # onSideStart sets layers=1 + emit; onRestart does layers++ + the SAME
        # emit; caps at 3 Spikes / 2 Toxic Spikes), so the layer count is the
        # number of start lines. Count them here (capped at the fork's cap) and
        # reset that side to 0 on the -sideend line. (Was a 0/1 bool — it
        # collapsed 2-3 Spikes to "present", under-reporting hazard damage.)
        if name in ("spikes", "toxicspikes"):
            if name in self.field:
                cap = 3 if name == "spikes" else 2
                if ev.active:
                    self.field[name][side] = min(cap, (self.field[name][side] or 0) + 1)
                else:
                    self.field[name][side] = 0
            return False
        return False

    # ══════════════════ 3.4 SOFT evidence: damage window (3.4-1..3) ══════════════════

    def _apply_damage(self, ev: E.DamageObserved) -> bool:
        side, slot = ev.target_side, ev.target_slot
        facts = self._facts_for(side, slot)
        # capture the observation BEFORE overwriting facts.hp
        maxhp = ev.maxhp or (facts.maxhp if facts is not None else 0) or 0
        prev = facts.hp if facts is not None and facts.hp is not None else maxhp
        delta = maxhp if ev.ko else max(0, prev - ev.hp_after)  # D40: a KO dealt >= maxhp
        self._set_hp(side, slot, ev.hp_after)
        if facts is not None:
            if ev.maxhp:
                facts.maxhp = ev.maxhp
            # the reflector input (3.6): the fork records ONLY real move hits
            # (`gotAttacked` -> `attackedBy` / the counter volatile, set from the
            # move-hit loop at battle-actions.ts:1051); residual (burn/recoil) and
            # field (hazard) damage is NEVER reflected, so it must not clobber the
            # last move hit (D48 bug-hunt: a burn tick after a hit was overwriting
            # it, and a later Counter/Mirror Coat calc would have reflected the tick).
            if ev.source_kind == "move":
                facts.last_hit = delta
                # the hit's category (D55 reflector gate): the fork reflects only
                # a matching category (Counter/Metal Burst physical, Mirror Coat/
                # Comeuppance special), so record what kind of move set last_hit.
                facts.last_hit_category = _move_category(ev.move, formats.engine_key(self.fmt))
                # avalanche gate (fork moves.ts:912-915: `p.damage > 0`): only a
                # hit that actually dealt damage sets the flag — a resisted or
                # immune (0-damage) hit must not, mirroring the fork's some()
                # over attackedBy. Cleared by turn_reset() on E.Turn. Record
                # WHICH mon dealt it (side, slot) so the per-source Avalanche/
                # Revenge gate can check "did THIS target hit me"; a 0-damage or
                # unattributable/self source records nothing (the fork's some()
                # never qualifies those).
                if delta > 0 and ev.source_slot and ev.source_slot != slot:
                    facts.attacked_by_this_turn.add((ev.source_side, ev.source_slot))
        # evidence only for attributable MOVE-source damage with a DISTINCT
        # attacker; residual/unattributable lines update the fact only (3.4-6)
        if (ev.source_kind != "move" or not ev.move or not ev.source_slot
                or ev.source_slot == slot):
            return False
        return self._damage_evidence(ev, facts, delta, maxhp, prev)

    def _damage_evidence(self, dmg: E.DamageObserved, target_facts: Optional[Facts],
                         delta: int, maxhp: int, prev: int) -> bool:
        """Update the belief (p2) slot ONLY (3.4-2): the other mon is exactly
        known (own team). Both-sides-belief (multi) is skipped for 3.4."""
        target_p2 = dmg.target_side == "p2"
        source_p2 = dmg.source_side == "p2"
        if target_p2 == source_p2:
            return False
        # Screen-aware field (B5): Reflect halves PHYSICAL damage behind the
        # defender, Light Screen halves SPECIAL — each lands on the side of the
        # mon it protects, so name the attacking / defending side (the slot's
        # side id, which is also the key the belief field uses). Wonder Room
        # (shared) rides along in the same dict.
        field = ev.calc_field(self.field, attacker_side=dmg.source_side,
                              defender_side=dmg.target_side)
        if target_p2:
            # candidate = DEFENDER (p2); attacker = exact own mon
            sb = self._slots.get(dmg.target_slot)
            spec = self.own_team.get(dmg.source_slot)
            if sb is None or spec is None:
                return False
            a_facts = self.own_facts.get(dmg.source_slot)
            a_mon = ev.mon_options(spec, status=ev.status_of(a_facts.statuses if a_facts else None),
                                  boosts=(a_facts.boosts if a_facts else {}),
                                  hp_fraction=self._hp_fraction(a_facts))
            d_status = ev.status_of(target_facts.statuses) if target_facts else None
            d_boosts = target_facts.boosts if target_facts else {}
            d_frac = max(0.0, min(1.0, prev / maxhp)) if maxhp else None
            # The attacker (p1) is the one who took the prior hit (reflector source).
            # defender = the mon being hit (dmg.target_*); the per-source avalanche
            # gate checks whether THAT mon hit the attacker this turn.
            move_spec = _move_with_state(dmg.move, a_facts,
                                        defender=(dmg.target_side, dmg.target_slot))
            # P4.2: batch ALL candidates' windows into ONE Node spawn (same
            # numbers as the per-candidate defender_window calls — shared cache).
            windows = ev.defender_windows_many(
                sb.candidates, sb.species, a_mon, d_status, d_boosts,
                d_frac, move_spec, field, self.fmt)
            wins_by_id = {id(c): w for c, w in zip(sb.candidates, windows)}

            def lik(c: Dict) -> float:
                return ev.damage_likelihood(wins_by_id[id(c)], delta,
                                            ko=dmg.ko, maxhp=maxhp)
        else:
            # candidate = ATTACKER (p2); defender = exact own mon
            sb = self._slots.get(dmg.source_slot)
            spec = self.own_team.get(dmg.target_slot)
            if sb is None or spec is None:
                return False
            d_facts = self.own_facts.get(dmg.target_slot)
            d_mon = ev.mon_options(spec, status=ev.status_of(d_facts.statuses if d_facts else None),
                                  boosts=(d_facts.boosts if d_facts else {}),
                                  hp_fraction=self._hp_fraction(d_facts))
            a_status = ev.status_of(sb.facts.statuses)
            a_frac = self._hp_fraction(sb.facts)
            # The attacker (p2) is the one who took the prior hit (reflector source).
            # defender = the mon being hit (dmg.target_*); the per-source avalanche
            # gate checks whether THAT mon hit the attacker this turn.
            move_spec = _move_with_state(dmg.move, sb.facts,
                                        defender=(dmg.target_side, dmg.target_slot))
            # P4.2: batch ALL candidates' windows into ONE Node spawn (see the
            # defender direction above).
            windows = ev.attacker_windows_many(
                sb.candidates, sb.species, a_status, sb.facts.boosts, a_frac,
                d_mon, move_spec, field, self.fmt)
            wins_by_id = {id(c): w for c, w in zip(sb.candidates, windows)}

            def lik(c: Dict) -> float:
                return ev.damage_likelihood(wins_by_id[id(c)], delta,
                                            ko=dmg.ko, maxhp=maxhp)
        ev.apply_evidence(sb.candidates, lik)
        return True

    @staticmethod
    def _hp_fraction(facts: Optional[Facts]) -> Optional[float]:
        if facts is not None and facts.hp is not None and facts.maxhp:
            return max(0.0, min(1.0, facts.hp / facts.maxhp))
        return None

    @staticmethod
    def _clamp_hp(hp: Optional[int], maxhp: Optional[int]) -> Optional[int]:
        """Clamp an observed HP into ``[0, maxhp]``. A mon's current HP can never
        exceed its max HP, so ``hp > maxhp`` is always a data inconsistency (a
        stale ``maxhp`` or a malformed line), never a real state — clamp it down.
        ``None`` stays ``None`` (HP not yet known); a ``None`` maxhp clamps only
        the lower bound. This enforces the belief invariant ``0 <= hp <= maxhp``
        so the HP fraction handed to the calc and the observation is always in
        [0, 1] (no blind-play, no false EV/max-HP deduction)."""
        if hp is None:
            return None
        if maxhp:
            return max(0, min(hp, maxhp))
        return max(0, hp)

    # ══════════════════ 3.6 SOFT evidence: max-HP consistency ══════════════════

    def _apply_maxhp_evidence(self, slot: str, observed_maxhp: int) -> bool:
        """3.6 — the switch-in line reveals the mon's max HP; use it as EV evidence.

        Rules out candidates whose HP-EV spread could not produce that much HP
        (their IV-31 maxHP is lower) and confirms one that can (one-sided,
        IV-safe — never a false rule-out). Applied on the p2 slot the switch-in
        names. Idempotent: re-applying the same observed max HP to an already
        narrowed posterior leaves consistent candidates at the same relative
        weight (a candidate that floors at EPS stays floored; the confirmed one
        stays at ~1), so a same-mon re-entry is a no-op, not a double-count.
        """
        sb = self._slots.get(slot)
        if sb is None or not observed_maxhp:
            return False
        field = ev.calc_field(self.field)
        # P4.2: batch ALL candidates' max-HP probes into ONE Node spawn (same
        # fixed probe attacker / Tackle as maxhp_for — shared cache keys).
        windows = ev.maxhp_windows_many(sb.candidates, sb.species, field, self.fmt)
        maxhp_by_id = {id(c): w["maxhp"] for c, w in zip(sb.candidates, windows)}

        def lik(c: Dict) -> float:
            return ev.maxhp_likelihood(maxhp_by_id[id(c)], observed_maxhp)
        ev.apply_evidence(sb.candidates, lik)
        return True

    # ══════════════════ 3.4 SOFT evidence: speed order (3.4-4) ══════════════════

    def _flush_speed_buffer(self) -> bool:
        """On a Turn/BattleEnd, the moves buffered for the turn just ended are
        in ACTION order (fork: the `|move|` announce lines fire in the order the
        actions resolve). For each equal-priority pair with no switch/faint
        between them: first mover's effective speed >= the second's; a tie is
        CONSISTENT (the sim breaks ties randomly) — 3.4-4."""
        buf, self._turn_buffer = self._turn_buffer, []
        moves = [(i, e) for i, e in enumerate(buf) if e[0] == "m"]
        applied = False
        for i in range(len(moves)):
            for j in range(i + 1, len(moves)):
                ii, a = moves[i]
                jj, b = moves[j]
                if any(e[0] in ("sw", "break") for e in buf[ii + 1:jj]):
                    continue  # order between them is no longer speed semantics
                applied = self._pair_speed(a, b) or applied
        return applied

    def _pair_speed(self, a: tuple, b: tuple) -> bool:
        _, side_a, slot_a, move_a = a
        _, side_b, slot_b, move_b = b
        try:
            gen = formats.engine_key(self.fmt)
            pa = lookup.get_move(move_a, gen)["priority"]
            pb = lookup.get_move(move_b, gen)["priority"]
        except KeyError:
            return False  # unknown move — no priority, no evidence
        if pa != pb:
            return False  # priority decides the order — no speed evidence
        a_belief, b_belief = side_a == "p2", side_b == "p2"
        if a_belief == b_belief:
            return False  # both own (no belief) or both belief (multi) — 3.4-2
        field = ev.calc_field(self.field)
        # Trick Room inverts action order: normally the FIRST mover is the
        # faster one; under Trick Room the first mover is the SLOWER one (B5),
        # so the "first mover is faster" comparison flips for the whole turn.
        first_ge = not bool(self.field.get("trickroom"))  # first mover >= second

        def known_speed(side: str, slot: str) -> Optional[int]:
            spec = self.own_team.get(slot)
            if spec is None:
                return None
            facts = self.own_facts.get(slot)
            mon = ev.mon_options(spec, status=ev.status_of(facts.statuses if facts else None),
                                boosts=(facts.boosts if facts else {}))
            tw = self.field.get("tailwind_side") == side
            return ev.speed_of(mon, fmt=self.fmt, tailwind=tw, field=field)

        # a is the FIRST action of the turn, b the SECOND (buffer order).
        if a_belief:
            # candidate moved FIRST: faster (normal) / slower (Trick Room)
            other = known_speed(side_b, slot_b)
            if other is None:
                return False
            sb = self._slots.get(slot_a)
            if sb is None:
                return False
            a_status = ev.status_of(sb.facts.statuses)
            a_boosts = dict(sb.facts.boosts)
            tw = self.field.get("tailwind_side") == side_a

            def lik(c: Dict) -> float:
                mon = ev.mon_options(c, status=a_status, boosts=a_boosts, name=sb.species)
                s = ev.speed_of(mon, fmt=self.fmt, tailwind=tw, field=field)
                ok = (s >= other) if first_ge else (s <= other)
                return 1.0 if ok else 0.0
        else:
            # candidate moved SECOND: slower (normal) / faster (Trick Room)
            other = known_speed(side_a, slot_a)
            if other is None:
                return False
            sb = self._slots.get(slot_b)
            if sb is None:
                return False
            b_status = ev.status_of(sb.facts.statuses)
            b_boosts = dict(sb.facts.boosts)
            tw = self.field.get("tailwind_side") == side_b

            def lik(c: Dict) -> float:
                mon = ev.mon_options(c, status=b_status, boosts=b_boosts, name=sb.species)
                s = ev.speed_of(mon, fmt=self.fmt, tailwind=tw, field=field)
                ok = (other >= s) if first_ge else (other <= s)
                return 1.0 if ok else 0.0
        ev.apply_evidence(sb.candidates, lik)
        return True

    # ══════════════════════════════ accessors ══════════════════════════════

    def slot(self, slot: str) -> SlotBelief:
        try:
            return self._slots[slot]
        except KeyError:
            raise KeyError(f"no belief for slot {slot!r} (add_opponent first)") from None

    @property
    def slots(self) -> List[str]:
        return list(self._slots)
