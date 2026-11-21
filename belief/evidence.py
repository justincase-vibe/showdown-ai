"""Step 3.2 — shared evidence engine.

ONE module (user requirement: nothing written twice) that turns a single piece
of observed evidence into a per-candidate LIKELIHOOD and updates the posterior.
It is used by BOTH the set posterior (3.3–3.5) and the HP/EV window inference
(3.6), so the two never drift apart.

Design (PLAN.md §3, 3.2 + 3.3):
  * An evidence = a likelihood function ``candidate -> [0, 1]``.
  * HARD evidence (revealed move / ability / item, Tera, Choice-lock) is exact
    0/1 — a candidate either satisfies it or it doesn't.
  * SOFT evidence (damage window, speed order — implemented in 3.4 over the Step-2
    calc) is (0, 1]; the mechanism is the same, only the returned range differs.
  * "unknown is not disconfirmed": a candidate whose dimension is UNKNOWN
    (``item == ""`` / ``tera is None``) is CONSISTENT with a reveal of that
    dimension (likelihood 1) — absence of information is not evidence of a
    specific value. A candidate that pins a SPECIFIC value must match it.
  * EPS floor: no likelihood may be exactly 0 — ``apply_evidence`` floors every
    likelihood at ``EPS`` before multiplying, so the posterior can NEVER empty
    out (off-DB sets survive; the AI always has a normalizable belief to act on).
    After renormalization a HARD non-match reads ~EPS (≈0) while a match reads
    ~1 — so HARD evidence is effectively decisive without ever collapsing the
    distribution to all-zeros.

The HARD section is pure math over the candidate dicts (no DB, no damage calc,
no fork) — trivially unit-testable. The SOFT section (3.4) computes damage
windows and effective speeds through the Step-2 calc (@smogon/calc = source of
truth for the damage model; the fork is the battle environment) and dedupes
identical calls, so both the set posterior (3.3–3.5) and the HP/EV inference
(3.6) consume ONE verified number.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

#: EPS floor — no evidence may fully rule out a candidate. Small enough that a
#: HARD non-match is ≈0 after renormalization; large enough to avoid float
#: underflow (we renormalize after every update, so no accumulation).
EPS = 1e-9

#: Choice items — using the SAME move twice means the item is locked to one of
#: these (a non-Choice item allows switching moves freely).
CHOICE_ITEMS = frozenset({"choiceband", "choicescarf", "choicespecs"})

#: A likelihood: candidate dict -> float in [0, 1].
Likelihood = Callable[[Dict], float]


def _floor(lik: float, eps: float) -> float:
    if lik < 0.0:
        raise ValueError(f"likelihood must be >= 0, got {lik}")
    return max(lik, eps)


def apply_evidence(candidates: List[Dict], likelihood: Likelihood,
                   eps: float = EPS) -> List[Dict]:
    """Update ``candidates`` weights IN PLACE with one piece of evidence.

    ``weight_i <- weight_i * max(likelihood(c_i), eps)``, then renormalize so
    the weights sum to exactly 1. Returns the same (mutated) list.

    The EPS floor guarantees the posterior never empties: even if the evidence
    is inconsistent with EVERY candidate (an off-DB set), all weights floor at
    ``eps`` and renormalize to a uniform, still-normalizable belief.
    """
    if not candidates:
        raise ValueError("apply_evidence: empty candidate list")
    for c in candidates:
        c["weight"] *= _floor(likelihood(c), eps)
    total = sum(c["weight"] for c in candidates)
    if total <= 0.0:  # defensive — impossible with a positive prior + eps floor
        n = len(candidates)
        for c in candidates:
            c["weight"] = 1.0 / n
        return candidates
    for c in candidates:
        c["weight"] /= total
    return candidates


# ── HARD evidence (exact 0/1; the eps floor is applied by apply_evidence) ──

def revealed_move(move: str) -> Likelihood:
    """The set contains the revealed move. Membership test — ORDER-BLIND (a
    moveset is a set: wish+protect+foulplay = foulplay+protect+wish)."""
    def lik(c: Dict) -> float:
        return 1.0 if move in c.get("moves", []) else 0.0
    return lik


def revealed_ability(ability: str) -> Likelihood:
    """The set's ability matches. (Ability is always a SPECIFIC value in every
    candidate — the fallback picks one of the species' abilities — so there is
    no "unknown" case to be lenient about.)"""
    def lik(c: Dict) -> float:
        return 1.0 if c.get("ability", "") == ability else 0.0
    return lik


def revealed_item(item: str) -> Likelihood:
    """The set's item is consistent with the reveal. A candidate with
    ``item == ""`` (unknown — the fallback models no item) is CONSISTENT
    (likelihood 1); a candidate pinning a SPECIFIC item must match it."""
    def lik(c: Dict) -> float:
        citem = c.get("item", "")
        if citem == "":
            return 1.0
        return 1.0 if citem == item else 0.0
    return lik


def tera_used(tera_type: str) -> Likelihood:
    """The set's Tera type is consistent with the reveal. ``tera is None`` means
    UNKNOWN (NOT "no Tera") -> consistent (likelihood 1); a candidate pinning a
    SPECIFIC Tera type must match it."""
    def lik(c: Dict) -> float:
        ctera = c.get("tera")
        if ctera is None:
            return 1.0
        return 1.0 if ctera == tera_type else 0.0
    return lik


def choice_lock() -> Likelihood:
    """The SAME move was used twice -> the item is locked to a Choice item.
    A candidate with an unknown item (``""``) is CONSISTENT (it could be a
    Choice item); a candidate pinning a SPECIFIC non-Choice item is ruled out;
    a candidate pinning a Choice item is confirmed."""
    def lik(c: Dict) -> float:
        citem = c.get("item", "")
        if citem == "":
            return 1.0
        return 1.0 if citem in CHOICE_ITEMS else 0.0
    return lik


#: name -> factory, so ``update()`` (3.3) can dispatch by evidence type.
HARD_EVIDENCE = {
    "revealed_move": revealed_move,
    "revealed_ability": revealed_ability,
    "revealed_item": revealed_item,
    "tera_used": tera_used,
    "choice_lock": choice_lock,
}


# ══════════════════ SOFT evidence (3.4): damage window + speed order ══════════════════
#
# The 3.4 SOFT evidence (PLAN.md §3a, awaiting user review — the bold
# recommendations are implemented): a HARD damage window (observed delta in the
# Step-2 calc's [min,max] OR crit [min,max] -> 1, else 0->EPS) and the
# speed-order signal (equal-priority moves: first mover's effective speed >=
# the second's; ties consistent). Both are computed with the Step-2 calc
# (@smogon/calc = source of truth for the damage model), so the likelihoods
# inherit its verification. The pure functions (``damage_likelihood``,
# ``mon_options``) stay testable without Node; only ``window_for`` /
# ``speed_of`` spawn a process — and IDENTICAL calls are deduped in ``_CACHE``
# (the window depends on the mon spec + move + field, NOT on the set, so a
# 693-candidate slot costs at most one spawn per unique spec).
from typing import Any  # noqa: E402  (SOFT section keeps the HARD section pure)

import json as _json

#: belief field id -> @smogon/calc field value.
# Accepts BOTH the canonical ids the belief tests use ("rain"/"sun"/"psychic")
# AND the raw Showdown log ids the adapter emits from real ladder logs
# ("raindance"/"sunnyday"/"psyshic"): the adapter does toID on the log token, so
# a real battle carries the move-id form (|weather|raindance), NOT the short form.
# Without the real ids, calc_field silently dropped rain/sun/psychic-terrain in
# real battles -> wrong damage/speed likelihoods (regression-tested).
# Proper follow-up (open decision): canonicalise at the source so the belief
# state, calc, and encoder all share ONE id set; for now this defensive mapping
# matches the calc bridge's normItem/normAbility (accept id OR name) pattern.
_WEATHER = {"rain": "Rain", "raindance": "Rain",
            "sun": "Sun", "sunnyday": "Sun", "strongwinds": "Sun",
            "sand": "Sand", "sandstorm": "Sand",
            "hail": "Hail", "snow": "Hail"}
_TERRAIN = {"electric": "Electric", "grassy": "Grass", "misty": "Misty",
            "psychic": "Psychic", "psyshic": "Psychic"}

#: the calc takes ONE status per mon; order by damage/speed interaction
#: (paralysis first: it is the only status with both).
# Codes are the Showdown ids the adapter stores verbatim from |-status| lines,
# which are ALSO @smogon/calc's status codes (verified against the library:
# it checks hasStatus('slp') for Wake-Up Slap x2 / Dream Eater and
# hasStatus('psn') for Facade x2). The old tuple listed "skl" — a code NEITHER
# Showdown nor the calc uses (sleep is "slp"), so a sleeping mon's status was
# silently dropped from every calc window; "psn" (regular poison) was likewise
# never passed through. (A mon has at most ONE status, so order is fallback,
# not priority; "frz" is passed harmlessly — the calc never checks it.)
_STATUS_ORDER = ("par", "brn", "slp", "frz", "tox", "psn")


def status_of(statuses) -> Optional[str]:
    """A fact sheet's status set -> the single status the calc takes (None if
    the sheet carries no damage-relevant status)."""
    for s in _STATUS_ORDER:
        if s in (statuses or ()):
            return s
    return None


def calc_field(field: Optional[Dict], attacker_side: Optional[str] = None,
               defender_side: Optional[str] = None) -> Dict:
    """``BeliefState.field`` -> the calc's ``field`` input ({} when nothing set).

    Weather / terrain are shared by both mons. The SCREENS are per-side, so the
    caller names the Showdown side of the attacking / defending mon (``p1``/
    ``p2``) and each screen lands on THAT mon's side (the library halves
    physical damage behind ``defenderSide.isReflect`` and special damage behind
    ``defenderSide.isLightScreen`` — verified: 129-153 -> 64-76). Wonder Room
    is shared (it swaps Atk/Def and SpA/SpD for both mons).

    Without the side arguments the screens are dropped — the pre-fix behaviour,
    which left every screened window 2x too wide (weak, never-wrong evidence).
    """
    field = field or {}
    out: Dict[str, Any] = {}
    w = _WEATHER.get(field.get("weather") or "")
    if w:
        out["weather"] = w
    t = _TERRAIN.get(field.get("terrain") or "")
    if t:
        out["terrain"] = t
    if field.get("wonderroom"):
        out["isWonderRoom"] = True
    for side, key in ((attacker_side, "attackerSide"), (defender_side, "defenderSide")):
        if not side:
            continue
        s: Dict[str, bool] = {}
        if (field.get("reflect") or {}).get(side):
            s["isReflect"] = True
        if (field.get("lightscreen") or {}).get(side):
            s["isLightScreen"] = True
        if s:
            out[key] = s
    return out


def mon_options(spec: Dict, status: Optional[str] = None, boosts: Optional[Dict] = None,
                hp_fraction: Optional[float] = None, name: Optional[str] = None) -> Dict:
    """A belief candidate (or an exact own-team spec) -> @smogon/calc mon options.

    ``status`` / ``boosts`` / ``hp_fraction`` are the LIVE facts of the mon in
    the slot (3.4 fact sheet); the candidate carries only the hidden set dims.
    Unknown dims stay absent (``item == ""`` / ``tera is None`` -> full-HP /
    no-item / no-Tera baseline, same as the prior).

    ``name``: the mon's species. An exact own-team spec already carries it
    (``spec['name']``); a belief candidate does NOT — the species is a separate
    observation on ``SlotBelief.species``, so the candidate callers pass it here.
    """
    mon: Dict[str, Any] = {"name": name if name is not None else spec["name"]}
    if spec.get("nature"):
        mon["nature"] = spec["nature"]
    if spec.get("evs"):
        mon["evs"] = dict(spec["evs"])
    if spec.get("item"):
        mon["item"] = spec["item"]
    if spec.get("ability"):
        mon["ability"] = spec["ability"]
    if spec.get("tera"):
        mon["teraType"] = spec["tera"]
    if status:
        mon["status"] = status
    b = {k: v for k, v in (boosts or {}).items() if v}
    if b:
        mon["boosts"] = b
    if hp_fraction is not None:
        # A mon's current HP can NEVER exceed its max HP, so a fraction > 1.0 is
        # always a data inconsistency in the belief (a stale maxhp or a
        # malformed line), never a real state. Clamp into [0, 1] at this
        # belief->calc boundary: the raw calc throws on hp > 1 (node_calc.js),
        # and that throw is what degrades a live observation to a blind
        # zero-obs. Clamping (a HP fraction > 1 is invalid -> full-HP is the
        # closest valid reading) keeps the calc reachable and the obs sane.
        mon["hp"] = max(0.0, min(1.0, hp_fraction))
    return mon


#: calc-result dedupe (PLAN.md §3a 3.4-7): pure function of the inputs, so a
#: module-level cache is safe across battles. Capped — the 3.8 bench will say
#: if a bigger cache or a batch endpoint is worth it.
_CACHE: Dict[str, Any] = {}


def _cached(key: str, compute) -> Any:
    hit = _CACHE.get(key)
    if hit is None:
        hit = compute()
        if len(_CACHE) > 8192:
            _CACHE.clear()
        _CACHE[key] = hit
    return hit


def damage_likelihood(window: Dict, observed_delta: int, ko: bool = False,
                      maxhp: int = 0) -> float:
    """HARD-window likelihood (3.4-1): the observed hit is consistent iff it
    lies inside the candidate's calc window OR crit window.

    * non-KO hit: ``min <= delta <= max`` (normal) or ``crit_min <= delta <=
      crit_max`` (the sim can crit, base rate 1/24 in gens 7-9). A candidate
      whose ``min > delta`` would have KO'd this hit — ruled out.
    * KO line (D40, the log clips at 0): consistent iff ``max >= maxhp`` or
      ``crit_max >= maxhp``; an always-KO candidate (``min > maxhp``) is
      consistent with a KO but inconsistent with any surviving hit.
    * zero observed delta (a miss): consistent with everything — the move may
      have missed; no discrimination, no harm.
    """
    if ko:
        return 1.0 if (window["max"] >= maxhp or window["crit_max"] >= maxhp) else 0.0
    if observed_delta <= 0:
        return 1.0
    if window["min"] > observed_delta:
        return 0.0
    if window["min"] <= observed_delta <= window["max"]:
        return 1.0
    if window["crit_min"] and window["crit_min"] <= observed_delta <= window["crit_max"]:
        return 1.0
    return 0.0


def window_for(attacker: Dict, defender: Dict, move, field: Optional[Dict] = None,
               fmt: Optional[str] = None, gen: int = 9) -> Dict:
    """The Step-2 calc's damage window for one (attacker, defender, move, field)
    tuple — ``{'min','max','crit_min','crit_max'}`` in HP (immune -> all 0).
    Deduped: candidates sharing a spec pay one Node spawn each (3.4-7)."""
    key = _json.dumps([fmt, gen, attacker, defender, move, field], sort_keys=True)
    return _cached(key, lambda: _calc_window(attacker, defender, move, field, fmt, gen))


def _window_from_result(r: Dict) -> Dict:
    """One raw ``calc_damage`` result dict -> belief window dict. Shared by the
    single and batch paths — ONE place for the immune/crit/maxhp mapping."""
    if r.get("immune"):
        return {"min": 0, "max": 0, "crit_min": 0, "crit_max": 0,
                "maxhp": int(r.get("maxhp", 0) or 0)}
    c = r.get("crit") or {}
    return {
        "min": int(r.get("min", 0) or 0),
        "max": int(r.get("max", 0) or 0),
        "crit_min": int(c.get("min", 0) or 0),
        "crit_max": int(c.get("max", 0) or 0),
        "maxhp": int(r.get("maxhp", 0) or 0),  # the defender's max HP (3.6: EV inference)
    }


def _calc_window(attacker, defender, move, field, fmt, gen) -> Dict:
    from damage import calc  # lazy: keeps the pure section of this module import-light
    r = calc.calc_damage(attacker=attacker, defender=defender, move=move,
                         field=field, fmt=fmt, gen=gen)
    return _window_from_result(r)


def windows_many(items: List[Dict]) -> List[Dict]:
    """Batched ``window_for`` (P4.2): N (attacker, defender, move) tuples in ONE
    Node spawn instead of N separate spawns.

    ``items`` has the ``damage.calc.calc_damage_many`` item contract: ``attacker``
    / ``defender`` / ``move`` (required) + optional ``field`` / ``fmt`` / ``gen``.
    Returns the windows (the SAME shape as ``window_for``) IN INPUT ORDER.

    Cache semantics are identical to ``window_for``: hits come off ``_CACHE``
    under the SAME keys (the single and batch paths share ONE cache), the misses
    go out as ONE ``calc_damage_many`` spawn, and a per-item calc error raises
    ``RuntimeError`` exactly like the single path (a real error — never a silent
    zero). Empty ``items`` returns ``[]`` without spawning.

    P4.2 motivation (PLAN.md D78, measured on one fresh gen9ou self-play battle):
    252 single Node spawns = 56s = 72% of wall time. Batching the per-candidate
    windows of one evidence event collapses that event to a single spawn.
    """
    if not items:
        return []
    # SAME key expression and element ORDER as window_for
    # ([fmt, gen, attacker, defender, move, field]) so both paths populate and
    # read one shared _CACHE — a batch result is a hit for the later identical
    # single call and vice versa.
    keys = [_json.dumps([it.get("fmt"), it.get("gen", 9), it["attacker"],
                         it["defender"], it["move"], it.get("field")],
                        sort_keys=True) for it in items]
    results: List[Optional[Dict]] = [None] * len(items)
    misses: List[int] = []
    for i, key in enumerate(keys):
        hit = _CACHE.get(key)
        if hit is not None:
            results[i] = hit
        else:
            misses.append(i)
    if misses:
        from damage import calc  # lazy: one real spawn for the whole batch
        # Dedupe identical specs WITHIN the batch (the 3.8 bench's fallback slots
        # are 210 candidates sharing ~3 specs — one Node-side compute per unique
        # spec, distributed back to every candidate that carries it).
        seen: set = set()
        uniq_keys: List[str] = []
        uniq_items: List[Dict] = []
        for i in misses:
            k = keys[i]
            if k not in seen:
                seen.add(k)
                uniq_keys.append(k)
                uniq_items.append(items[i])
        raw = calc.calc_damage_many(uniq_items)
        if len(raw) != len(uniq_items):  # transport contract violation (defensive)
            raise RuntimeError(
                f"calc_damage_many returned {len(raw)} results for {len(uniq_items)} items")
        if len(_CACHE) > 8192:  # same cap as _cached
            _CACHE.clear()
        for u, r in enumerate(raw):
            if "error" in r:
                raise RuntimeError(r["error"])  # same loud failure as the single path
            _CACHE[uniq_keys[u]] = _window_from_result(r)
        for i in misses:
            results[i] = _CACHE[keys[i]]
    return results  # type: ignore[return-value]


def defender_window(candidate: Dict, species: str, attacker_mon: Dict,
                    defender_status: Optional[str], defender_boosts: Optional[Dict],
                    defender_hp_fraction: Optional[float], move, field: Optional[Dict],
                    fmt: Optional[str] = None, gen: int = 9) -> Dict:
    """Window when the CANDIDATE is the defender (the attacker is exactly known).

    ``species`` names the candidate's mon (a belief candidate has no 'name';
    the species is the observed fact on the slot)."""
    mon = mon_options(candidate, status=defender_status, boosts=defender_boosts,
                      hp_fraction=defender_hp_fraction, name=species)
    return window_for(attacker_mon, mon, move, field, fmt, gen)


def attacker_window(candidate: Dict, species: str, attacker_status: Optional[str],
                    attacker_boosts: Optional[Dict], attacker_hp_fraction: Optional[float],
                    defender_mon: Dict, move, field: Optional[Dict] = None,
                    fmt: Optional[str] = None, gen: int = 9) -> Dict:
    """Window when the CANDIDATE is the attacker (the defender is exactly known).

    ``species`` names the candidate's mon (a belief candidate has no 'name';
    the species is the observed fact on the slot)."""
    mon = mon_options(candidate, status=attacker_status, boosts=attacker_boosts,
                      hp_fraction=attacker_hp_fraction, name=species)
    return window_for(mon, defender_mon, move, field, fmt, gen)


def defender_windows_many(candidates: List[Dict], species: str, attacker_mon: Dict,
                          defender_status: Optional[str], defender_boosts: Optional[Dict],
                          defender_hp_fraction: Optional[float], move,
                          field: Optional[Dict], fmt: Optional[str] = None,
                          gen: int = 9) -> List[Dict]:
    """Batched ``defender_window`` (P4.2): one window per candidate, ALL in ONE
    Node spawn (results in candidate order; identical numbers to the per-candidate
    calls — same mon construction, same shared _CACHE keys)."""
    items = []
    for c in candidates:
        mon = mon_options(c, status=defender_status, boosts=defender_boosts,
                          hp_fraction=defender_hp_fraction, name=species)
        items.append({"attacker": attacker_mon, "defender": mon, "move": move,
                      "field": field, "fmt": fmt, "gen": gen})
    return windows_many(items)


def attacker_windows_many(candidates: List[Dict], species: str,
                          attacker_status: Optional[str],
                          attacker_boosts: Optional[Dict],
                          attacker_hp_fraction: Optional[float],
                          defender_mon: Dict, move, field: Optional[Dict] = None,
                          fmt: Optional[str] = None, gen: int = 9) -> List[Dict]:
    """Batched ``attacker_window`` (P4.2): one window per candidate, ONE spawn."""
    items = []
    for c in candidates:
        mon = mon_options(c, status=attacker_status, boosts=attacker_boosts,
                          hp_fraction=attacker_hp_fraction, name=species)
        items.append({"attacker": mon, "defender": defender_mon, "move": move,
                      "field": field, "fmt": fmt, "gen": gen})
    return windows_many(items)


def speed_of(mon: Dict, fmt: Optional[str] = None, tailwind: bool = False,
             field: Optional[Dict] = None, gen: int = 9) -> int:
    """Effective action speed (Step-2 calc, full @smogon/calc mirror). Deduped."""
    key = _json.dumps(["spd", fmt, gen, mon, tailwind, field], sort_keys=True)

    def _compute() -> int:
        from damage import calc  # lazy
        return calc.effective_speed(mon, fmt=fmt, gen=gen, tailwind=tailwind, field=field)

    return _cached(key, _compute)


# ══════════════════ 3.6 SOFT evidence: max-HP consistency ══════════════════
#
# The switch-in line (`hp/maxhp`) reveals the mon's max HP (verified in 3.1: the
# adapter learns it from the opponent's switch-in, incl. the maxhp==100 dedupe and
# slot-reuse cases). max HP is a FUNCTION of the candidate's HP-EV spread, so the
# observed max HP is EV evidence: it rules out any candidate whose spread could not
# produce that much HP, and confirms one that can. This is the clean, IV-safe form
# of the "residual damage (burn/poison)" signal from 3.4 decision #6 — a burn deals
# max(1, floor(maxhp/16)) and a poison max(1, floor(maxhp/8)), so BOTH are just
# another observation of the same max HP; the switch-in line is the direct one.
#
# The calc's `maxhp` field is the DEFENDER's maxHP (node_calc.js `defender.maxHP()`),
# independent of the attacker — verified immune-safe (a Normal-immune defender still
# returns its maxhp). So we read it with a fixed generic probe attacker (any valid
# (attacker, defender, move) tuple; the attacker never affects the value).

#: fixed probe attacker for reading the defender's max HP — generic (a Normal-type
#: mon / Tackle) so it carries no damage-side meaning of its own.
_MAXHP_PROBE_ATTACKER: Dict[str, Any] = {"name": "pikachu", "ability": "static", "evs": {}}


def maxhp_for(candidate: Dict, species: str, field: Optional[Dict] = None,
              fmt: Optional[str] = None, gen: int = 9) -> int:
    """A candidate's max HP (IV 31 — the calc's default) as a function of its EV
    spread, read straight off the Step-2 calc's ``maxhp`` field (the DEFENDER's
    maxHP, so the probe attacker is irrelevant to the value).

    Deduped in ``_CACHE``: candidates sharing an EV spread cost ONE Node spawn
    each (same cost profile as the 3.4 damage window — the 3.8 bench decides if it
    matters; do NOT pre-optimize). Because the calc uses IV 31 (the maximum IV),
    this is the mon's maxHP *maximum* for that EV spread — the basis of the
    one-sided consistency rule below (a real mon can only have <= this HP for the
    same spread, so we never rule out a spread that could actually be right).
    """
    mon = mon_options(candidate, name=species)
    return int(window_for(_MAXHP_PROBE_ATTACKER, mon, "tackle", field, fmt, gen)["maxhp"])


def maxhp_windows_many(candidates: List[Dict], species: str,
                       field: Optional[Dict] = None, fmt: Optional[str] = None,
                       gen: int = 9) -> List[Dict]:
    """Batched ``maxhp_for`` (P4.2): every candidate's max-HP probe window in ONE
    spawn (same fixed probe attacker + Tackle, same shared _CACHE keys)."""
    items = []
    for c in candidates:
        mon = mon_options(c, name=species)
        items.append({"attacker": _MAXHP_PROBE_ATTACKER, "defender": mon,
                      "move": "tackle", "field": field, "fmt": fmt, "gen": gen})
    return windows_many(items)


def maxhp_likelihood(candidate_maxhp: int, observed_maxhp: int) -> float:
    """One-sided max-HP consistency (IV-safe — no false rule-outs):

    * the calc's ``maxhp`` (IV 31) is the mon's maxHP *maximum* for that EV spread,
      so ``observed > candidate`` is IMPOSSIBLE -> ruled out (0 -> EPS floor);
    * ``observed <= candidate`` is CONSISTENT (the real mon may run lower IVs, so
      its true max HP is at most the IV-31 value) -> 1.0.

    Deliberately one-sided, NOT a tight two-sided band: a higher-HP-EV candidate
    stays consistent even when it is not the true spread, so a candidate that
    *could* be right is never ruled out. ``observed_maxhp <= 0`` (no max HP known
    yet) is consistent with everything — no signal, no harm.
    """
    if observed_maxhp <= 0:
        return 1.0
    return 1.0 if observed_maxhp <= candidate_maxhp else 0.0
