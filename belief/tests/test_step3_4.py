"""Step 3.4 gate — SOFT evidence (damage window + speed order) via the Step-2
calc, plus the per-slot fact sheet.

Run from the project root:

    python -m pytest belief/tests -v

Sub-step 3.4 (PLAN.md §3): make the belief consume the Step-2 damage calc as a
SOFT signal. The observed hit is consistent with a candidate iff it falls inside
that candidate's calc window (normal OR crit range); a KO line (D40) is consistent
iff the candidate COULD have KO'd. The speed-order signal: for equal-priority
moves the first mover is faster (a tie is consistent); different priority means
priority decided the order, so the order carries no speed evidence. Both are
computed by the ONE shared window module (evidence.py) over the Step-2 calc, so
the set posterior (3.3–3.5) and the HP/EV inference (3.6) consume the same number.

The damage windows below are verified against the Step-2 calc (@smogon/calc =
source of truth for the damage model; the fork is the battle environment + the
validation oracle) — the pinned observed deltas are chosen to sit comfortably
inside one candidate's window and outside the other's, so a calc drift that moves
the windows is caught here too.

  * defender direction  — the candidate is the victim; the attacker is exactly
                          known (own team). SpD EVs hidden: 0-SpD [88,105] vs
                          252-SpD [70,84] (Blastoise max HP 299).
  * attacker direction  — the candidate is the attacker; the victim is exactly
                          known. Atk EVs hidden: 0-atk [135,160] vs 252-atk
                          [180,213].
  * KO-clip (D40)       — a KO line discriminates a KO-capable config from one
                          that cannot KO even on a crit; an always-KO config is
                          inconsistent with any surviving hit.
  * immunity            — an immune candidate is ruled out by a real hit.
  * speed order         — first mover faster / second mover faster / priority
                          differs → no evidence.
  * fact sheet          — hp/maxhp (switch-in), damage, statuses, boosts, Tera,
                          move history, last-hit; own-side facts; field + Tailwind.
  * live integration    — one real seeded fork battle fed through the full
                          log→event→belief pipeline stays sane and fires evidence.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from belief import events as E
from belief import evidence as ev
from belief.log_events import LogAdapter
from belief.state import BeliefState, SlotBelief, _move_category, _move_with_state
from legality import formats

ROOT = Path(__file__).resolve().parents[2]  # belief/tests/x.py -> project root

# ── helpers ──────────────────────────────────────────────────────────────────

def cand(**kw):
    """A valid belief candidate (the schema build_prior produces, minus name —
    the species lives on SlotBelief.species, not in the candidate)."""
    base = {"moves": ["tackle", "a", "b", "c"], "item": "", "ability": "x",
            "nature": "Hardy", "evs": {}, "tera": None, "weight": 1.0}
    base.update(kw)
    return base


def install(bs: BeliefState, slot: str, species: str, cands) -> SlotBelief:
    """White-box: put a controlled candidate set into a belief slot (3.3 tests
    manipulate .candidates the same way)."""
    sb = SlotBelief(slot, species, cands)
    bs._slots[slot] = sb
    return sb


def weights(bs, slot):
    return {i: c["weight"] for i, c in enumerate(bs._slots[slot].candidates)}


def approx_one(w):  # a HARD match after renormalization
    return w > 0.999


def approx_eps(w):  # a HARD non-match floors at EPS -> ~0 after renormalization
    return 0 < w < 1e-6


# ══════════════════════════ damage window: pure rule ══════════════════════════

def test_damage_likelihood_rule():
    """The KO-clip rule (D40) + the normal/crit/miss window logic, in isolation
    (no Node) — this is the contract both directions and the 3.6 EV inference use."""
    maxhp = 301
    ko_capable = {"min": 286, "max": 339, "crit_min": 430, "crit_max": 508}  # max 339 >= 301
    never_ko   = {"min": 166, "max": 196, "crit_min": 250, "crit_max": 294}  # crit 294 < 301
    always_ko  = {"min": 320, "max": 400, "crit_min": 480, "crit_max": 560}  # min 320 > 301

    # KO line: consistent iff the candidate COULD KO (max>=maxhp OR crit_max>=maxhp)
    assert ev.damage_likelihood(ko_capable, 0, ko=True, maxhp=maxhp) == 1.0
    assert ev.damage_likelihood(never_ko, 0, ko=True, maxhp=maxhp) == 0.0
    assert ev.damage_likelihood(always_ko, 0, ko=True, maxhp=maxhp) == 1.0

    # an always-KO candidate (min > maxhp) is INCONSISTENT with a surviving hit
    assert ev.damage_likelihood(always_ko, 250, ko=False, maxhp=maxhp) == 0.0
    # a normal surviving hit inside the window is consistent
    assert ev.damage_likelihood(never_ko, 180, ko=False, maxhp=maxhp) == 1.0
    # a hit BELOW the window's min would have KO'd -> ruled out
    assert ev.damage_likelihood(ko_capable, 250, ko=False, maxhp=maxhp) == 0.0
    # a hit inside the CRIT range is consistent (the sim can crit)
    assert ev.damage_likelihood(ko_capable, 470, ko=False, maxhp=maxhp) == 1.0
    # a miss (observed_delta <= 0) is consistent with everything
    assert ev.damage_likelihood(never_ko, 0, ko=False, maxhp=maxhp) == 1.0


# ══════════════════════ damage window: defender direction ═════════════════════

def test_damage_defender_direction():
    """Own Charizard (Modest 252 SpA + Life Orb) hits the opponent's Blastoise with
    Flamethrower. The Blastoise's SpD is hidden: 0-SpD deals [88,105], 252-SpD
    deals [70,84] (disjoint). The observed delta selects exactly one candidate."""
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}

    def fresh():
        bs = BeliefState("gen9ou", own_team=own)
        return bs, install(bs, "p2a", "blastoise", [cand(evs={"spd": 0}), cand(evs={"spd": 252})])

    def hit(delta):
        bs, sb = fresh()
        applied = bs.update(E.DamageObserved(
            target_side="p2", target_slot="p2a", hp_after=299 - delta, maxhp=299,
            ko=False, source_kind="move", move="flamethrower",
            source_side="p1", source_slot="p1a"))
        return applied, sb

    # delta 95 is in 0-SpD [88,105], outside 252-SpD [70,84]
    applied, sb = hit(95)
    assert applied is True
    assert approx_one(sb.candidates[0]["weight"]) and approx_eps(sb.candidates[1]["weight"])

    # delta 75 is in 252-SpD [70,84], outside 0-SpD [88,105]
    applied, sb = hit(75)
    assert applied is True
    assert approx_eps(sb.candidates[0]["weight"]) and approx_one(sb.candidates[1]["weight"])

    # the posterior is a valid distribution after a SOFT update
    assert sum(sb.posterior) == pytest.approx(1.0, abs=1e-9)


# ══════════════════════ damage window: attacker direction ═════════════════════

def test_damage_attacker_direction():
    """The opponent's Garchomp hits our Blastoise with Earthquake. The Garchomp's
    Atk is hidden: Hardy/0-atk deals [135,160], Adamant/252-atk deals [180,213]."""
    own = {"p1a": {"name": "blastoise", "nature": "Hardy"}}

    def fresh():
        bs = BeliefState("gen9ou", own_team=own)
        return bs, install(bs, "p2a", "garchomp",
                           [cand(nature="Hardy", evs={"atk": 0}),
                            cand(nature="Adamant", evs={"atk": 252})])

    def hit(delta):
        bs, sb = fresh()
        applied = bs.update(E.DamageObserved(
            target_side="p1", target_slot="p1a", hp_after=299 - delta, maxhp=299,
            ko=False, source_kind="move", move="earthquake",
            source_side="p2", source_slot="p2a"))
        return applied, sb

    # delta 150 is in 0-atk [135,160], outside 252-atk [180,213]
    applied, sb = hit(150)
    assert applied is True
    assert approx_one(sb.candidates[0]["weight"]) and approx_eps(sb.candidates[1]["weight"])

    # delta 190 is in 252-atk [180,213], outside 0-atk [135,160]
    applied, sb = hit(190)
    assert applied is True
    assert approx_eps(sb.candidates[0]["weight"]) and approx_one(sb.candidates[1]["weight"])


# ══════════════════════════════ damage window: KO-clip ════════════════════════

def test_ko_clip_integration():
    """D40: a KO line (`0 fnt`) means the true damage was >= max HP. A KO-capable
    config is consistent with a KO; a config that cannot KO even on a crit is not.
    Reverse: a surviving hit below a config's min means that config would have KO'd.

    Garchomp Earthquake vs our Golduck (max HP 301): the strong config (Adamant
    252 Atk + Life Orb) deals [286,339] — a KO; the weak config (Hardy, 0 Atk)
    deals [166,196] and its crit tops out at 294 < 301, so it can NEVER KO."""
    own = {"p1a": {"name": "golduck", "nature": "Hardy"}}

    def fresh():
        bs = BeliefState("gen9ou", own_team=own)
        return bs, install(bs, "p2a", "garchomp",
                           [cand(nature="Adamant", evs={"atk": 252}, item="lifeorb"),
                            cand(nature="Hardy", evs={}, item="")])

    # KO line -> the KO-capable config wins
    bs, sb = fresh()
    applied = bs.update(E.DamageObserved(
        target_side="p1", target_slot="p1a", hp_after=0, maxhp=301, ko=True,
        source_kind="move", move="earthquake", source_side="p2", source_slot="p2a"))
    assert applied is True
    assert approx_one(sb.candidates[0]["weight"]) and approx_eps(sb.candidates[1]["weight"])

    # surviving hit delta 180 (in [166,196]) -> the weak config wins; the strong
    # config's min (286) > 180, so it would have KO'd and is ruled out
    bs, sb = fresh()
    applied = bs.update(E.DamageObserved(
        target_side="p1", target_slot="p1a", hp_after=301 - 180, maxhp=301, ko=False,
        source_kind="move", move="earthquake", source_side="p2", source_slot="p2a"))
    assert applied is True
    assert approx_eps(sb.candidates[0]["weight"]) and approx_one(sb.candidates[1]["weight"])


# ══════════════════════════════ damage window: immunity ═══════════════════════

def test_immune_candidate_ruled_out():
    """A Ground hit on a Levitate Duskull does 0 (immune) — so a real observed
    delta rules the immune candidate out and confirms the one that hits
    (Insomnia, [68,81]). This is the fork-exact type-immunity gate from Step 2,
    consumed as belief evidence here."""
    own = {"p1a": {"name": "charizard", "nature": "Hardy"}}
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "duskull",
                 [cand(ability="levitate"), cand(ability="insomnia")])
    applied = bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=181 - 75, maxhp=181, ko=False,
        source_kind="move", move="earthquake", source_side="p1", source_slot="p1a"))
    assert applied is True
    assert approx_eps(sb.candidates[0]["weight"])  # Levitate (immune) ruled out
    assert approx_one(sb.candidates[1]["weight"])   # Insomnia confirmed


# ════════════════════════════════ speed order ═════════════════════════════════

def test_speed_first_mover_faster():
    """Our Garchomp (speed 240) moves first, then the opponent's Blastoise. Equal
    priority (both 0) -> the first mover is faster -> Blastoise is slower, so the
    0-Spe candidate (192) is confirmed and the 252-Spe one (255) is ruled out."""
    bs = BeliefState("gen9ou", own_team={"p1a": {"name": "garchomp", "evs": {"spe": 0}}})
    sb = install(bs, "p2a", "blastoise", [cand(evs={"spe": 0}), cand(evs={"spe": 252})])
    bs.update(E.MoveUsed(side="p1", slot="p1a", move="earthquake"))
    bs.update(E.MoveUsed(side="p2", slot="p2a", move="surf"))
    assert bs.update(E.Turn(n=1)) is True
    assert approx_one(sb.candidates[0]["weight"]) and approx_eps(sb.candidates[1]["weight"])


def test_speed_second_mover_faster():
    """Reversed: the opponent's Blastoise moves first -> it is faster -> the
    252-Spe candidate (255) is confirmed and the 0-Spe one (192) is ruled out."""
    bs = BeliefState("gen9ou", own_team={"p1a": {"name": "garchomp", "evs": {"spe": 0}}})
    sb = install(bs, "p2a", "blastoise", [cand(evs={"spe": 0}), cand(evs={"spe": 252})])
    bs.update(E.MoveUsed(side="p2", slot="p2a", move="surf"))
    bs.update(E.MoveUsed(side="p1", slot="p1a", move="earthquake"))
    assert bs.update(E.Turn(n=1)) is True
    assert approx_eps(sb.candidates[0]["weight"]) and approx_one(sb.candidates[1]["weight"])


def test_speed_priority_differs_no_evidence():
    """Different priority (Aqua Jet +1 vs Surf 0) means priority decided the order —
    the order carries NO speed evidence, so the posterior must be unchanged."""
    bs = BeliefState("gen9ou", own_team={"p1a": {"name": "garchomp", "evs": {"spe": 0}}})
    sb = install(bs, "p2a", "blastoise", [cand(evs={"spe": 0}), cand(evs={"spe": 252})])
    before = list(sb.posterior)
    bs.update(E.MoveUsed(side="p1", slot="p1a", move="aquajet"))
    bs.update(E.MoveUsed(side="p2", slot="p2a", move="surf"))
    assert bs.update(E.Turn(n=1)) is False
    assert sb.posterior == before


def test_speed_switch_between_moves_no_evidence():
    """A switch between the two moves breaks the speed-order semantics (the second
    mover is a different mon), so no speed evidence is applied."""
    bs = BeliefState("gen9ou", own_team={"p1a": {"name": "garchomp", "evs": {"spe": 0}}})
    sb = install(bs, "p2a", "blastoise", [cand(evs={"spe": 0}), cand(evs={"spe": 252})])
    before = list(sb.posterior)
    bs.update(E.MoveUsed(side="p2", slot="p2a", move="surf"))
    bs.update(E.SwitchOut(side="p2", slot="p2a"))   # order-breaker between the two moves
    bs.update(E.MoveUsed(side="p1", slot="p1a", move="earthquake"))
    assert bs.update(E.Turn(n=1)) is False
    assert sb.posterior == before


# ═══════════════════════════════ fact sheet ══════════════════════════════════

def test_facts_bookkeeping():
    """The per-slot fact sheet (PLAN.md §3a 3.4-5): every turn adds information and
    this is where it is held. Own-side facts are tracked too; the field is shared."""
    bs = BeliefState("gen9ou", own_team={"p1a": {"name": "garchomp", "evs": {"spe": 0}}})
    sb = install(bs, "p2a", "blastoise", [cand(), cand()])

    # switch-in reveals hp/maxhp
    bs.update(E.SwitchIn(side="p2", slot="p2a", species="blastoise", hp=183, maxhp=183))
    assert sb.facts.hp == 183 and sb.facts.maxhp == 183

    # damage updates hp and records the last hit (the 3.6 reflector input)
    bs.update(E.DamageObserved(target_side="p2", target_slot="p2a", hp_after=120, maxhp=183,
                               ko=False, source_kind="move", move="flamethrower",
                               source_side="p1", source_slot="p1a"))
    assert sb.facts.hp == 120 and sb.facts.last_hit == 63  # 183 - 120

    # statuses add / cure
    bs.update(E.StatusApplied(side="p2", slot="p2a", status="brn"))
    assert sb.facts.statuses == {"brn"}
    bs.update(E.StatusCured(side="p2", slot="p2a", status="brn"))
    assert sb.facts.statuses == set()

    # boosts accumulate and clear back to 0 (kept out of the sheet)
    bs.update(E.BoostChanged(side="p2", slot="p2a", stat="atk", delta=1))
    bs.update(E.BoostChanged(side="p2", slot="p2a", stat="atk", delta=1))
    assert sb.facts.boosts == {"atk": 2}
    bs.update(E.BoostChanged(side="p2", slot="p2a", stat="atk", delta=-2))
    assert sb.facts.boosts == {}

    # boosts clamp at +/-6
    for _ in range(8):
        bs.update(E.BoostChanged(side="p2", slot="p2a", stat="def", delta=1))
    assert sb.facts.boosts == {"def": 6}

    # Tera is recorded
    bs.update(E.TeraUsed(side="p2", slot="p2a", tera_type="fire"))
    assert sb.facts.tera == "fire"

    # a NEW mon taking the slot gets a fresh sheet; the SAME mon keeps its facts
    install(bs, "p2a", "gengar", [cand(), cand()])
    bs._slots["p2a"].facts.hp = 50
    bs.add_opponent("p2a", "gengar")  # same species -> keep
    assert bs._slots["p2a"].facts.hp == 50

    # own-side facts are tracked exactly
    bs.update(E.DamageObserved(target_side="p1", target_slot="p1a", hp_after=90, maxhp=183,
                               ko=False, source_kind="move", move="earthquake",
                               source_side="p2", source_slot="p2a"))
    assert bs.own_facts["p1a"].hp == 90


def test_field_and_tailwind_expiry():
    """Weather / terrain / Tailwind land in the shared field; Tailwind expires on a
    turn count (the exact off-by-one is irrelevant — it only ever makes a mon faster)."""
    bs = BeliefState("gen9ou", own_team={})
    bs.update(E.WeatherChanged(weather="rain"))
    bs.update(E.TerrainChanged(terrain="electric"))
    bs.update(E.TailwindChanged(side="p2", active=True))
    assert bs.field["weather"] == "rain" and bs.field["terrain"] == "electric"
    assert bs.field["tailwind_side"] == "p2" and bs.field["tailwind_turns"] == 4

    for _ in range(4):
        bs.update(E.Turn(n=1))
    assert bs.field["tailwind_side"] is None and bs.field["tailwind_turns"] == 0

    # clearing the weather resets the field slot
    bs.update(E.WeatherChanged(weather="none"))
    assert bs.field["weather"] is None


def test_calc_field_accepts_real_log_ids():
    """The adapter does ``to_id`` on the raw log token (log_events.py), so a
    REAL ladder battle stores the SHOWDOWN id in ``bs.field`` — ``raindance``
    from ``|weather| rain dance``, ``sunnyday``, ``psyshic`` (``|terrain|
    psychic terrain`` minus ``terrain``) — NOT the short canonical ids these
    tests used to feed ("rain"/"sun"/"psychic"). ``calc_field`` must map BOTH
    forms to the calc's field value, else rain/sun/psychic-terrain were
    silently dropped from every real-battle damage/speed likelihood (the
    false-truth the canonical-id tests masked)."""
    # real log ids (what the adapter emits from actual logs)
    assert ev.calc_field({"weather": "raindance"}) == {"weather": "Rain"}
    assert ev.calc_field({"weather": "sunnyday"}) == {"weather": "Sun"}
    assert ev.calc_field({"terrain": "psyshic"}) == {"terrain": "Psychic"}
    # canonical ids still work (the pre-fix test vocabulary)
    assert ev.calc_field({"weather": "rain"}) == {"weather": "Rain"}
    assert ev.calc_field({"weather": "sun"}) == {"weather": "Sun"}
    assert ev.calc_field({"terrain": "psychic"}) == {"terrain": "Psychic"}
    # the other weather/terrain ids stay mapped
    assert ev.calc_field({"weather": "sandstorm"}) == {"weather": "Sand"}
    assert ev.calc_field({"weather": "snow"}) == {"weather": "Hail"}
    assert ev.calc_field({"terrain": "electric"}) == {"terrain": "Electric"}
    # nothing set -> empty field
    assert ev.calc_field({"weather": None}) == {}
    assert ev.calc_field({}) == {}
    assert ev.calc_field(None) == {}


def test_status_of_passes_showdown_status_codes_through():
    """The adapter stores the RAW Showdown status code from ``|-status|`` lines
    (verified fork protocol: par/slp/frz/tox/psn/brn), and those codes ARE
    @smogon/calc's codes — the library checks ``hasStatus('slp')`` (Wake-Up
    Slap x2, Dream Eater's hit gate) and ``hasStatus('psn')`` (Facade x2).
    The old ``_STATUS_ORDER`` listed "skl" (a code neither Showdown nor the
    calc uses; sleep is "slp"), so a sleeping mon's status was silently DROPPED
    from every calc window — Wake-Up Slap / Dream Eater windows computed at
    half power, Facade-from-poison halved. Same false-truth class as the
    weather-id bug: belief vocabulary and calc vocabulary must share one set.
    """
    for code in ("par", "brn", "slp", "frz", "tox", "psn"):
        assert ev.status_of({code}) == code, code
    assert ev.status_of(set()) is None
    assert ev.status_of(None) is None


def test_spikes_and_toxicspikes_track_layer_count():
    """The fork emits ONE |-sidestart line PER Spikes / Toxic-Spikes layer (no
    count in the line; moves.js:17536-17543: onSideStart sets layers=1 + emit,
    onRestart does layers++ + the SAME emit). The belief must therefore COUNT the
    start lines: Spikes 0->1->2->3 (cap 3), Toxic Spikes 0->1->2 (cap 2), and a
    -sideend line resets that side to 0. The old code collapsed both to a 0/1
    bool, so 2-3 Spikes read as "present" (hazard damage under-reported)."""
    bs = BeliefState("gen9ou", own_team={})
    for _ in range(3):
        bs.update(E.FieldEffectChanged(name="spikes", side="p2", active=True))
    assert bs.field["spikes"]["p2"] == 3
    # a 4th start can't exceed the fork's cap of 3
    bs.update(E.FieldEffectChanged(name="spikes", side="p2", active=True))
    assert bs.field["spikes"]["p2"] == 3
    # Toxic Spikes cap at 2
    for _ in range(3):
        bs.update(E.FieldEffectChanged(name="toxicspikes", side="p1", active=True))
    assert bs.field["toxicspikes"]["p1"] == 2
    # the observation carries the true layer count
    assert bs.get_context()["field"]["spikes"]["p2"] == 3
    # a -sideend line (hazard cleared) resets that side to 0
    bs.update(E.FieldEffectChanged(name="spikes", side="p2", active=False))
    assert bs.field["spikes"]["p2"] == 0
    assert bs.get_context()["field"]["spikes"]["p2"] == 0


def test_hazard_observation_includes_toxicspikes_and_safeguard():
    """Toxic Spikes + Safeguard are tracked in ``self.field`` but were NOT
    surfaced in ``get_context()`` (only spikes/stealthrock/stickyweb were) —
    hazard state is now complete for the Step-4 observation (user: "if the
    hazards are not correctly observed, please add it")."""
    bs = BeliefState("gen9ou", own_team={})
    bs.update(E.FieldEffectChanged(name="toxicspikes", side="p2", active=True))
    bs.update(E.FieldEffectChanged(name="toxicspikes", side="p2", active=True))  # 2 layers
    bs.update(E.FieldEffectChanged(name="safeguard", side="p1", active=True))
    field = bs.get_context()["field"]
    assert field["toxicspikes"]["p2"] == 2
    assert field["safeguard"]["p1"] is True
    # the other side reads as 0 (toxicspikes) / is filtered out (safeguard keeps
    # only active sides)
    assert field["toxicspikes"]["p1"] == 0
    assert "p2" not in field["safeguard"]


def test_used_protect_set_and_turn_reset():
    """Protect-family moves (Protect/Detect/Spiky Shield/King's Shield/Guard
    Split) set the user's used_protect for the turn (doubles-critical: don't
    waste an attack on a protected target); a non-protect move does not; cleared
    on E.Turn; surfaced on both own and opponent mon blocks."""
    bs = BeliefState("gen9ou", own_team={"p1a": {"name": "garchomp"}})
    install(bs, "p2a", "gengar", [cand(), cand()])

    # opponent uses Spiky Shield -> protected this turn (opponent block)
    bs.update(E.MoveUsed(side="p2", slot="p2a", move="spikyshield"))
    assert bs._slots["p2a"].facts.used_protect is True
    assert bs.get_context()["opponent"]["p2a"]["used_protect"] is True

    # own side uses Protect -> own block
    bs.update(E.MoveUsed(side="p1", slot="p1a", move="protect"))
    assert bs.own_facts["p1a"].used_protect is True
    assert bs.get_context()["own"]["p1a"]["used_protect"] is True

    # a non-protect move does NOT set a fresh mon's flag
    install(bs, "p2b", "blastoise", [cand()])
    bs.update(E.MoveUsed(side="p2", slot="p2b", move="hydropump"))
    assert bs._slots["p2b"].facts.used_protect is False

    # cleared on the next turn (both sides)
    bs.update(E.Turn(n=2))
    assert bs._slots["p2a"].facts.used_protect is False
    assert bs.own_facts["p1a"].used_protect is False
    assert bs.get_context()["opponent"]["p2a"]["used_protect"] is False


def test_confidence_clamped_to_unit_interval():
    """confidence = 1 - normalized entropy must stay in [0,1]; a floating-point
    entropy a hair above 1.0 used to round to -0.0 on the observation."""
    bs = BeliefState("gen9ou", own_team={})
    install(bs, "p2a", "gengar", [cand(), cand()])
    bs.update(E.SwitchIn(side="p2", slot="p2a", species="gengar", hp=170, maxhp=170))
    conf = bs.get_context()["opponent"]["p2a"]["confidence"]
    assert 0.0 <= conf <= 1.0
    assert conf == abs(conf)  # never -0.0


def test_residual_damage_updates_fact_but_no_set_evidence():
    """Residual / unattributable damage (no move source) still updates the HP fact
    and records the last hit, but produces NO set-belief evidence (3.4-6: deferred
    to 3.6)."""
    bs = BeliefState("gen9ou", own_team={})
    sb = install(bs, "p2a", "blastoise", [cand(), cand()])
    bs.update(E.SwitchIn(side="p2", slot="p2a", species="blastoise", hp=183, maxhp=183))
    before = list(sb.posterior)
    applied = bs.update(E.DamageObserved(target_side="p2", target_slot="p2a", hp_after=170,
                                         maxhp=183, ko=False, source_kind="residual"))
    assert applied is False                 # no set evidence
    assert sb.posterior == before           # ...belief untouched
    assert sb.facts.hp == 170               # ...but the fact is recorded


def test_residual_burn_tick_does_not_disturb_posterior():
    """End-to-end regression (D48): a burn tick on the defender must NOT be
    consumed as MOVE damage. Before the fix the adapter pinned it to the last
    ``|move|`` line; because the tick's delta is far below that move's window, it
    floored the CORRECT candidate to ~0 (collapsing the posterior to uniform).
    Driven through the REAL adapter so the log→event→belief path is what is tested.
    """
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "blastoise", [cand(evs={"spd": 0}), cand(evs={"spd": 252})])
    a = LogAdapter()
    a.feed_lines([
        "|move|p1a: Charizard|Flamethrower|p2a: Blastoise",
        "|-damage|p2a: Blastoise|204/299",   # delta 95 -> the SpD-0 window
    ])
    for e in a.events:
        bs.update(e)
    assert approx_one(sb.candidates[0]["weight"])     # narrowed to the SpD-0 set
    # the burn tick must NOT be consumed as move damage (regression, D48):
    # the adapter classifies it residual, so the belief records the HP fact but
    # applies NO set evidence (the true SpD-0 set stays the leader).
    a.feed_lines(["|-damage|p2a: Blastoise|190/299 brn|[from] brn"])
    burn = a.events[-1]
    assert burn.source_kind == "residual"              # adapter: not a move hit
    assert bs.update(burn) is False                    # belief: no set evidence
    assert approx_one(sb.candidates[0]["weight"])      # ...true set still the leader
    assert sb.facts.hp == 190                          # ...but the HP fact is updated


def test_residual_tick_does_not_clobber_last_hit():
    """D48 bug-hunt regression: ``facts.last_hit`` (the reflector input for
    Counter/Mirror Coat/Metal Burst) tracks the last MOVE hit only. The fork sets
    its counter volatile / ``attackedBy`` from the move-hit loop alone
    (``gotAttacked``, battle-actions.ts:1051) — a burn tick is never reflected,
    so it must not overwrite the last real hit. Before the fix the tick's delta
    clobbered it and a later reflector calc would have reflected the tick.
    """
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "blastoise", [cand(evs={"spd": 0})])
    a = LogAdapter()
    a.feed_lines([
        "|move|p1a: Charizard|Flamethrower|p2a: Blastoise",
        "|-damage|p2a: Blastoise|204/299",   # move hit: last_hit = 95
    ])
    for e in a.events:
        bs.update(e)
    assert sb.facts.last_hit == 95
    # a burn tick (delta 14) is residual: HP fact updates, last_hit does NOT
    # (feed only the NEW event — a.events is cumulative; re-feeding the earlier
    # move-hit event would re-apply it with delta 0, a test artifact, not a log)
    a.feed_lines(["|-damage|p2a: Blastoise|190/299 brn|[from] brn"])
    burn = a.events[-1]
    assert burn.source_kind == "residual"          # adapter: not a move hit
    assert bs.update(burn) is False                # belief: facts only, no evidence
    assert sb.facts.hp == 190
    assert sb.facts.last_hit == 95                 # the tick must NOT clobber it


def test_reflector_preserves_narrowed_posterior():
    """P0 regression (2026-11-16): a reflector move (Counter/Mirror Coat/Metal
    Burst/Come Up Pance) must NOT collapse an already-narrowed posterior.

    Before the fix, ``_damage_evidence`` passed the move as a bare string to the
    calc; ``_normalise_move`` dropped ``lastHitDamage``; the calc returned
    ``immune/[0,0]`` for reflectors without prior-hit data; ``damage_likelihood``
    returned 0.0 for every candidate; ``apply_evidence`` floored all to EPS →
    uniform. The prior evidence was destroyed.

    After the fix, ``_move_with_state`` attaches the attacker's ``facts.last_hit``
    to the move spec; the calc returns a valid window (2× prior hit); both
    candidates get likelihood 1.0 (reflectors are stat-independent); the prior
    narrowing is preserved.
    """
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}

    def fresh():
        bs = BeliefState("gen9ou", own_team=own)
        return bs, install(bs, "p2a", "blastoise",
                           [cand(evs={"spd": 0}), cand(evs={"spd": 252})])

    # Step 1: a discriminating move hit narrows the posterior
    bs, sb = fresh()
    bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=204, maxhp=299,
        ko=False, source_kind="move", move="flamethrower",
        source_side="p1", source_slot="p1a"))
    assert approx_one(sb.candidates[0]["weight"])  # 0-SpD wins

    # Step 2: the opponent uses a reflector — must NOT wipe the narrowing.
    # Mirror Coat (not Counter) because Step 1's hit was SPECIAL (Flamethrower):
    # the fork reflects only a MATCHING category (D55), so Mirror Coat reflects
    # 2 × last_hit here; stat-independent, so both candidates get 1.0 → the prior
    # evidence is preserved, not destroyed.
    bs.update(E.DamageObserved(
        target_side="p1", target_slot="p1a", hp_after=109, maxhp=299,
        ko=False, source_kind="move", move="mirrorcoat",
        source_side="p2", source_slot="p2a"))
    assert approx_one(sb.candidates[0]["weight"]), \
        f"reflector collapsed the posterior: {[c['weight'] for c in sb.candidates]}"
    assert approx_eps(sb.candidates[1]["weight"])


def test_reflector_stale_hit_not_reflected():
    """D54 (2026-11-16) reflector stale-hit gate. The fork recreates the
    reflector volatile FRESH each turn (``beforeTurnCallback`` + ``duration: 1``)
    and its ``onTry`` FAILS (``slot === null``) if the user was not hit by a
    damaging move THIS turn — so a hit from a PREVIOUS turn means the reflector
    fails (0), it does NOT reflect 2× that stale hit.

    ``facts.last_hit`` (the magnitude) is intentionally retained across turns as
    memory; the fix gates it: ``_move_with_state`` attaches ``lastHitDamage`` only
    when ``attacked_by_this_turn`` is non-empty (this-turn, move-source, damage>0).
    White-box (no Node): the positive case (hit this turn -> gate open) and the
    stale case (hit last turn -> gate closed) are pinned directly on the spec.
    """
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "blastoise", [cand()])

    # Turn N: p2a is hit by p1a (Aqua Jet = PHYSICAL, matching Counter) ->
    # last_hit set, last_hit_category="Physical", attacked_by_this_turn set
    bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=204, maxhp=299,
        ko=False, source_kind="move", move="aquajet",
        source_side="p1", source_slot="p1a"))
    assert sb.facts.last_hit == 95
    assert sb.facts.last_hit_category == "Physical"   # D55: category recorded
    assert ("p1", "p1a") in sb.facts.attacked_by_this_turn
    # SAME turn + category match: the gate is OPEN -> Counter reflects this hit
    spec_this = _move_with_state("counter", sb.facts, defender=("p1", "p1a"))
    assert isinstance(spec_this, dict) and spec_this["lastHitDamage"] == 95

    # Turn boundary: the per-turn signal clears, but last_hit (memory) persists
    bs.update(E.Turn())
    assert sb.facts.last_hit == 95                      # memory retained
    assert sb.facts.attacked_by_this_turn == set()      # this-turn signal cleared

    # Turn N+1: p2a Counters WITHOUT being hit this turn -> gate CLOSED -> no
    # lastHitDamage (fork's onTry fails -> Counter deals 0, NOT 2*95).
    spec_stale = _move_with_state("counter", sb.facts, defender=("p1", "p1a"))
    if isinstance(spec_stale, dict):
        assert "lastHitDamage" not in spec_stale, f"stale over-report: {spec_stale}"
    else:
        assert spec_stale == "counter"


def test_reflector_gate_category_is_per_gen():
    """A1 (2026-11-16): the reflector gate records the hit's category PER-GEN
    (D55), sourced via the format's engine key — not the gen-9 base row.
    Dragon Claw is Special in gen 1-3 (type-based physical/special split),
    Physical gen 4+, so the SAME move name classifies differently by generation.
    This pins the format → engine-key → per-gen-category wiring the D55 gate
    consumes (belief/state.py ``_move_category(ev.move, formats.engine_key(fmt))``).
    """
    # via the real wiring path (format -> engine key -> per-gen category)
    assert _move_category("dragonclaw", formats.engine_key("gen1ou")) == "Special"
    assert _move_category("dragonclaw", formats.engine_key("gen9bss3v3")) == "Physical"
    # gen 3 directly: the type-based split still makes it special
    assert _move_category("dragonclaw", 3) == "Special"
    # unknown move -> None (gate stays closed), any gen
    assert _move_category("nonexistentmove123", 9) is None


def test_reflector_category_gate():
    """D55 (2026-11-16) reflector CATEGORY gate. The fork reflects only a
    MATCHING category (moves.js ``onDamagingHit``: Counter/Metal Burst record a
    PHYSICAL hit, Mirror Coat/Comeuppance a SPECIAL one); a mismatch means the
    reflector FAILS (0). So ``lastHitDamage`` is attached only when the recorded
    hit category (``facts.last_hit_category``) matches the reflector's
    requirement (``_REFLECTOR_REQUIRES``). White-box (no Node): a SPECIAL hit
    opens Mirror Coat but NOT Counter; a PHYSICAL hit opens Counter but NOT
    Mirror Coat. Non-reflector moves are never category-gated.
    """
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}

    def hit_and_check(hit_move, hit_category):
        bs = BeliefState("gen9ou", own_team=own)
        sb = install(bs, "p2a", "blastoise", [cand()])
        bs.update(E.DamageObserved(
            target_side="p2", target_slot="p2a", hp_after=204, maxhp=299,
            ko=False, source_kind="move", move=hit_move,
            source_side="p1", source_slot="p1a"))
        assert sb.facts.last_hit_category == hit_category
        has = lambda mv: (isinstance(_move_with_state(mv, sb.facts,
                            defender=("p1", "p1a")), dict)
                          and "lastHitDamage"
                          in _move_with_state(mv, sb.facts, defender=("p1", "p1a")))
        return has

    # SPECIAL hit (Flamethrower): Mirror Coat reflects, Counter does NOT.
    has = hit_and_check("flamethrower", "Special")
    assert has("mirrorcoat"), "special hit should open Mirror Coat"
    assert not has("counter"), "special hit must NOT open Counter"
    assert not has("metalburst"), "special hit must NOT open Metal Burst"
    assert has("comeuppance"), "special hit should open Come Up Pance"

    # PHYSICAL hit (Aqua Jet): Counter reflects, Mirror Coat does NOT.
    has = hit_and_check("aquajet", "Physical")
    assert has("counter"), "physical hit should open Counter"
    assert has("metalburst"), "physical hit should open Metal Burst"
    assert not has("mirrorcoat"), "physical hit must NOT open Mirror Coat"
    assert not has("comeuppance"), "physical hit must NOT open Come Up Pance"

    # A non-reflector move is never category-gated (its state is unrelated).
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "blastoise", [cand()])
    bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=204, maxhp=299,
        ko=False, source_kind="move", move="flamethrower",
        source_side="p1", source_slot="p1a"))
    spec = _move_with_state("tackle", sb.facts, defender=("p1", "p1a"))
    # tackle is a normal move: no lastHitDamage is ever attached to it
    assert not (isinstance(spec, dict) and "lastHitDamage" in spec)


# ═══════════════════════════════ live integration ═════════════════════════════

def test_hit_this_turn_set_and_turn_reset():
    """T5 wiring (2026-11-16): the avalanche gate records the (side, slot) of
    the mon that hit this one and is cleared on E.Turn. The fork's avalanche
    bpCallback (moves.ts:912-915) doubles only if ``p.damage > 0`` from the
    target this turn, so a 0-damage follow-up must not clear an already-set
    entry (the fork's ``some()`` over attackedBy keeps the 95-damage entry)."""
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "blastoise", [cand()])
    # a real hit (delta 95 > 0) on the target -> the attacker is recorded
    bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=204, maxhp=299,
        ko=False, source_kind="move", move="flamethrower",
        source_side="p1", source_slot="p1a"))
    assert ("p1", "p1a") in sb.facts.attacked_by_this_turn
    # the state pipe must hand it to the calc (avalanche reads hitThisTurn),
    # computed PER-SOURCE: p1a is the mon that hit me, so Avalanche on p1a doubles
    spec = _move_with_state("avalanche", sb.facts, defender=("p1", "p1a"))
    assert spec["name"] == "avalanche" and spec["hitThisTurn"] is True
    # a 0-damage follow-up hit (resist/immune) must NOT clear the entry
    bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=204, maxhp=299,
        ko=False, source_kind="move", move="thunderbolt",
        source_side="p1", source_slot="p1a"))
    assert ("p1", "p1a") in sb.facts.attacked_by_this_turn
    # next turn: cleared
    bs.update(E.Turn(n=2))
    assert sb.facts.attacked_by_this_turn == set()


def test_zero_damage_hit_does_not_set_hit_this_turn():
    """Fork avalanche gate (moves.ts:914 ``p.damage > 0``): a resisted or
    immune hit (delta 0) must NOT record the attacker — the fork never
    doubles off a 0-damage entry."""
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "blastoise", [cand()])
    bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=299, maxhp=299,
        ko=False, source_kind="move", move="flamethrower",
        source_side="p1", source_slot="p1a"))
    assert sb.facts.attacked_by_this_turn == set()
    spec = _move_with_state("avalanche", sb.facts, defender=("p1", "p1a"))
    assert "hitThisTurn" not in spec


def test_hit_this_turn_is_per_source():
    """The avalanche/revenge gate is PER-SOURCE (fork moves.ts:912-915 /
    15086-15088: ``attackedBy.some(p => p.source === target && p.damage > 0 &&
    p.thisTurn)``): Avalanche doubles only if THE TARGET hit the user this turn
    — not merely any mon. So if p1a hit me, Avalanche on p1a doubles, but
    Avalanche on a DIFFERENT mon (p1b, which did not hit me) does not."""
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"},
           "p1b": {"name": "garchomp", "nature": "Jolly", "evs": {"atk": 252},
                   "item": "lifeorb", "ability": "sandrushing"}}
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "blastoise", [cand()])
    # only p1a hit me this turn (p1b did not)
    bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=204, maxhp=299,
        ko=False, source_kind="move", move="flamethrower",
        source_side="p1", source_slot="p1a"))
    assert sb.facts.attacked_by_this_turn == {("p1", "p1a")}
    # the observation still exposes the model-facing bool (was hit at all)
    assert bs.get_context()["opponent"]["p2a"]["hit_this_turn"] is True
    # Avalanche on p1a (the mon that hit me) -> doubles
    on_p1a = _move_with_state("avalanche", sb.facts, defender=("p1", "p1a"))
    assert on_p1a["hitThisTurn"] is True
    # Avalanche on p1b (a DIFFERENT mon) -> does NOT double (the per-source gate)
    on_p1b = _move_with_state("avalanche", sb.facts, defender=("p1", "p1b"))
    assert "hitThisTurn" not in on_p1b
    # next turn: the per-source set clears too
    bs.update(E.Turn(n=2))
    assert sb.facts.attacked_by_this_turn == set()


def test_last_move_failed_set_and_turn_reset():
    """T5 wiring (2026-11-16): ``last_move_failed`` (the stompingtantrum /
    temperflare gate, fork moves.ts:18050/19186 ``moveLastTurnResult === false``)
    is set on the mon whose move failed — both sides — and cleared on E.Turn."""
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "blastoise", [cand()])
    # opponent side: set on the p2 slot's facts
    bs.update(E.MoveFailed(side="p2", slot="p2a", move="struggle"))
    assert sb.facts.last_move_failed is True
    # the state pipe must hand it to the calc (stompingtantrum reads it)
    spec = _move_with_state("stompingtantrum", sb.facts)
    assert spec["lastMoveFailed"] is True
    # own side: own_facts created on demand (same pattern as the other fact handlers)
    bs.update(E.MoveFailed(side="p1", slot="p1a", move="struggle"))
    assert bs.own_facts["p1a"].last_move_failed is True
    # next turn: both cleared
    bs.update(E.Turn(n=2))
    assert sb.facts.last_move_failed is False
    assert bs.own_facts["p1a"].last_move_failed is False


def test_stockpile_layers_track_cap_and_switch_clear():
    """T5 wiring (2026-11-16): Stockpile adds one layer per use (fork cap 3 —
    moves.ts:17956-17966: the 4th use fails, adding nothing). The LAYERS
    PERSIST ACROSS TURNS (fork: the stockpile volatile has no duration — it is
    cleared only by switch-out / a new mon), a re-entering mon starts at 0,
    and the layers feed the calc's Spit Up (BP 100*layers) via the pipe."""
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "seadra", [cand()])
    for _ in range(3):
        bs.update(E.MoveUsed(side="p2", slot="p2a", move="stockpile"))
    assert sb.facts.stockpile_layers == 3
    # cap: the 4th use fails in the fork — no 4th layer
    bs.update(E.MoveUsed(side="p2", slot="p2a", move="stockpile"))
    assert sb.facts.stockpile_layers == 3
    # PERSISTS across turns (fork: a volatile, no duration) — a turn wipe would
    # have broken a multi-turn Stockpile → Spit Up sequence
    bs.update(E.Turn(n=2))
    assert sb.facts.stockpile_layers == 3
    # the state pipe hands the layers to the calc (Spit Up reads stockpileLayers)
    spec = _move_with_state("spitup", sb.facts)
    assert spec["stockpileLayers"] == 3
    # switch-out cleared the volatile: a re-entering mon starts at 0
    bs.update(E.SwitchIn(side="p2", slot="p2a", species="seadra", hp=100, maxhp=150))
    assert sb.facts.stockpile_layers == 0


def test_last_move_failed_from_real_log_lines():
    """End-to-end regression (the 2026-11-16 bug: the flag was unreachable from
    real logs — the fork never emits `movefail`). The fork's real failure
    lines are `|-fail|<user>` (after the |move| announce) and
    `|cant|<user>|<reason>` (aborted attempt) — both must land on
    facts.last_move_failed via the adapter and clear on E.Turn."""
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}
    bs = BeliefState("gen9ou", own_team=own)
    sb = install(bs, "p2a", "blastoise", [cand()])
    a = LogAdapter()
    a.feed_lines([
        "|move|p2a: Blastoise|Stomping Tantrum|p1a: Charizard",
        "|-fail|p2a: Blastoise",
    ])
    failed = a.events[-1]
    assert failed.type == "move_failed" and failed.move == "stompingtantrum"
    assert bs.update(failed) is False            # facts only, no set evidence
    assert sb.facts.last_move_failed is True
    spec = _move_with_state("stompingtantrum", sb.facts)
    assert spec["lastMoveFailed"] is True        # ...and the calc sees it
    bs.update(E.Turn(n=2))
    assert sb.facts.last_move_failed is False


def _run_smoke_battle():
    """Run the seeded fork battle keeper and return its raw log lines (shared with
    the 3.1 smoke test's pattern)."""
    script = ROOT / "scratch" / "step3_1_battle_log.js"
    proc = subprocess.run(["node", str(script)], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise AssertionError(f"smoke keeper failed (rc={proc.returncode})\n{proc.stderr}")
    lines, in_block = [], False
    for raw in proc.stdout.splitlines():
        if raw.strip() == "=== BATTLE LOG START ===":
            in_block = True
            continue
        if raw.strip() == "=== BATTLE LOG END ===":
            break
        if in_block and raw.startswith("|"):
            lines.append(raw)
    return lines


def test_live_log_belief_pipeline():
    """The full pipeline on a real seeded BSS battle: log -> events -> BeliefState.
    The belief must stay sane (non-empty, normalized) at every point, the damage
    events must actually fire evidence, and no log line may be left unmapped.
    The BSS teams (scratch/step3_1_battle_log.js) are our own side exactly, so the
    damage windows are computed against real own-team specs."""
    lines = _run_smoke_battle()
    assert len(lines) > 5, "smoke battle log looks empty"

    # our side (p1) from the keeper: Jolly 252/252/252, no items
    own = {
        "p1a": {"name": "garchomp", "nature": "Jolly", "evs": {"atk": 252, "spa": 252, "spe": 252}},
        "p1b": {"name": "blastoise", "nature": "Jolly", "evs": {"atk": 252, "spa": 252, "spe": 252}},
        "p1c": {"name": "tyranitar", "nature": "Jolly", "evs": {"atk": 252, "spa": 252, "spe": 252}},
    }
    bs = BeliefState("gen9bss3v3", own_team=own)
    adapter = LogAdapter()

    damage_events = 0
    damage_evidence = 0
    unknown = []
    for line in lines:
        e = adapter.feed_line(line)
        if e is None:
            continue
        applied = bs.update(e)
        if e.type == "damage_observed":
            damage_events += 1
            if applied:
                damage_evidence += 1
        if e.type == "unknown":
            unknown.append(e.raw)

    # the belief never empties and is always a valid distribution
    assert bs.slots, "expected at least one opponent slot"
    for slot in bs.slots:
        sb = bs.slot(slot)
        assert sb.candidates, f"{slot}: empty belief"
        assert sum(sb.posterior) == pytest.approx(1.0, abs=1e-9), f"{slot}: posterior not normalized"
        assert all(w > 0 for w in sb.posterior), f"{slot}: a weight went non-positive"

    # damage happened AND produced evidence (the SOFT path fired on real data)
    assert damage_events >= 1, "expected at least one damage observation"
    assert damage_evidence >= 1, "expected at least one damage event to produce evidence"

    # no log line was left unmapped (observable bug-catcher)
    assert unknown == [], f"unmapped log lines: {unknown}"

    # facts were populated for the active opponent slot
    active = bs.slots[-1]
    f = bs.slot(active).facts
    assert f.maxhp is not None and f.maxhp > 0, "active slot max HP should be known"
    json.dumps(bs.slot(active).candidates, default=list)  # candidates stay JSON-safe


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
