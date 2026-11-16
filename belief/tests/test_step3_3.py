"""Step 3.3 gate — ``update()``: HARD evidence applied to the posterior.

Run from the project root:

    python -m pytest belief/tests -v

The plan's four spec tests (PLAN.md §3, sub-step 3.3):
  * revealed in every sample   -> after revealing a move, ~all mass is on sets with it
  * illegal = 0               -> sets not satisfying the evidence read ~EPS (≈0, but >0)
  * sums to 1                 -> the posterior is a valid distribution after every update
  * Choice shift              -> the same move twice shifts mass toward Choice items
plus routing / re-initialization integration (own-side & field events ignored; a new
mon taking a slot gets a fresh prior).
"""
from __future__ import annotations

import pytest

from belief import events as E
from belief import evidence as ev
from belief.state import BeliefState, SlotBelief


# ═══════════════════════════════ the 4 spec tests ════════════════════════════

# A controlled pair: exactly ONE candidate has the revealed move, one does not.
# (The precision spec tests need a known legal/illegal split; a real prior like
#  garchomp's may put the same move in SEVERAL sets, which legitimately shares mass.)
def _split_cands():
    return [
        {"moves": ["earthquake", "a", "b", "c"], "item": "", "ability": "x",
         "nature": "Hardy", "evs": {}, "tera": None, "weight": 1.0},
        {"moves": ["flamethrower", "d", "e", "f"], "item": "", "ability": "x",
         "nature": "Hardy", "evs": {}, "tera": None, "weight": 1.0},
    ]


def test_spec_revealed_in_every_sample():
    # After revealing move X, ~all surviving mass is on candidates containing X.
    sb = SlotBelief("p2a", "garchomp", _split_cands())
    assert sb.update(E.MoveUsed(side="p2", slot="p2a", move="earthquake")) is True
    assert sb.mass(lambda c: "earthquake" in c["moves"]) > 0.999, \
        "the revealed move should hold ~all posterior mass"


def test_spec_illegal_is_zero():
    # The candidate NOT satisfying the evidence reads ~EPS (≈0) — but stays > 0
    # (the never-empty guard: ruled out, not deleted); the legal one holds ~all mass.
    sb = SlotBelief("p2a", "garchomp", _split_cands())
    sb.update(E.MoveUsed(side="p2", slot="p2a", move="earthquake"))
    legal = [c for c in sb.candidates if "earthquake" in c["moves"]][0]
    illegal = [c for c in sb.candidates if "earthquake" not in c["moves"]][0]
    assert legal["weight"] > 0.999, f"legal candidate should hold ~all mass, got {legal}"
    assert 0 < illegal["weight"] < 1e-6, f"illegal should be ~EPS but >0, got {illegal}"
    assert sum(sb.posterior) == pytest.approx(1.0, abs=1e-9)


def test_spec_sums_to_one():
    # The posterior is a valid probability distribution after every update.
    bs = BeliefState("gen9ou")
    sb = bs.add_opponent("p2a", "garchomp")
    c0 = sb.candidates[0]
    bs.update(E.MoveUsed(side="p2", slot="p2a", move=c0["moves"][0]))
    assert sum(sb.posterior) == pytest.approx(1.0, abs=1e-9)
    bs.update(E.AbilityRevealed(side="p2", slot="p2a", ability=c0["ability"]))
    assert sum(sb.posterior) == pytest.approx(1.0, abs=1e-9)
    bs.update(E.TeraUsed(side="p2", slot="p2a", tera_type="dragon"))
    assert sum(sb.posterior) == pytest.approx(1.0, abs=1e-9)


def test_spec_choice_shift():
    # Using the SAME move twice shifts mass toward Choice-item candidates and
    # rules out a pinned non-Choice item (a non-Choice item allows free switching).
    def c(item):
        return {"moves": ["thunderbolt", "wish", "protect", "spark"], "item": item,
                "ability": "plus", "nature": "Hardy", "evs": {}, "tera": None, "weight": 1.0}
    sb = SlotBelief("p2a", "ampharos", [c("choiceband"), c("leftovers"), c("")])
    before = sb.mass(lambda x: x["item"] in ev.CHOICE_ITEMS)
    sb.update(E.MoveUsed(side="p2", slot="p2a", move="thunderbolt"))
    sb.update(E.MoveUsed(side="p2", slot="p2a", move="thunderbolt"))
    after = sb.mass(lambda x: x["item"] in ev.CHOICE_ITEMS)
    assert after > before, f"Choice mass should shift up: {before} -> {after}"
    leftovers = [x for x in sb.candidates if x["item"] == "leftovers"][0]
    assert leftovers["weight"] < 1e-6, "pinned non-Choice item must be ruled out"
    assert sum(sb.posterior) == pytest.approx(1.0, abs=1e-9)


# ══════════════════════════════ integration ══════════════════════════════════

def test_ability_reveal_narrows_to_matching():
    bs = BeliefState("gen9ou")
    sb = bs.add_opponent("p2a", "garchomp")
    ab = sb.candidates[0]["ability"]
    assert bs.update(E.AbilityRevealed(side="p2", slot="p2a", ability=ab)) is True
    assert sb.mass(lambda c: c["ability"] == ab) > 0.999


def test_item_reveal_keeps_unknown_and_matching():
    def c(item, ab="plus"):
        return {"moves": ["a", "b", "c", "d"], "item": item, "ability": ab,
                "nature": "Hardy", "evs": {}, "tera": None, "weight": 1.0}
    sb = SlotBelief("p2a", "ampharos", [c("choiceband"), c("leftovers"), c("")])
    sb.update(E.ItemRevealed(side="p2", slot="p2a", item="choiceband"))
    assert sb.mass(lambda x: x["item"] in ("choiceband", "")) > 0.9
    assert [x for x in sb.candidates if x["item"] == "leftovers"][0]["weight"] < 1e-6


def test_update_ignores_own_side_and_non_set_evidence():
    bs = BeliefState("gen9ou")
    bs.add_opponent("p2a", "garchomp")
    assert bs.update(E.MoveUsed(side="p1", slot="p1a", move="earthquake")) is False
    assert bs.update(E.WeatherChanged(weather="rain")) is False
    assert bs.update(E.Turn(n=1)) is False
    assert bs.update(E.DamageObserved(target_side="p2", target_slot="p2a",
                                      hp_after=100, maxhp=183)) is False  # 3.4, not 3.3
    # the belief is untouched by the ignored events
    assert sum(bs.slot("p2a").posterior) == pytest.approx(1.0, abs=1e-9)


def test_new_mon_in_slot_gets_fresh_prior():
    bs = BeliefState("gen9ou")
    bs.add_opponent("p2a", "garchomp")
    m = bs.slot("p2a").candidates[0]["moves"][0]
    bs.update(E.MoveUsed(side="p2", slot="p2a", move=m))  # narrows garchomp
    # a DIFFERENT mon takes p2a (faint + re-switch) -> fresh prior, not garchomp's
    bs.add_opponent("p2a", "gengar")
    sb = bs.slot("p2a")
    assert sb.species == "gengar"
    assert sum(sb.posterior) == pytest.approx(1.0, abs=1e-9)
    assert all(w > 0 for w in sb.posterior)
    # garchomp's revealed move does NOT dominate a fresh gengar prior
    assert sb.mass(lambda c: m in c["moves"]) < 0.9


def test_same_species_reuse_keeps_posterior():
    bs = BeliefState("gen9ou")
    bs.add_opponent("p2a", "garchomp")
    m = bs.slot("p2a").candidates[0]["moves"][0]
    bs.update(E.MoveUsed(side="p2", slot="p2a", move=m))
    before = list(bs.slot("p2a").posterior)
    # re-adding the SAME mon (still on field) must NOT reset the posterior
    sb2 = bs.add_opponent("p2a", "garchomp")
    assert sb2 is bs.slot("p2a")
    assert sb2.posterior == before
