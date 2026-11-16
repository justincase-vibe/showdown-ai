"""Step 3.2 gate — shared evidence engine (belief/evidence.py).

Run from the project root:

    python -m pytest belief/tests -v

Locks the 3.2 deliverable (PLAN.md §3, sub-step 3.2): a per-evidence likelihood
+ the EPS floor + the never-empty guard, with a unit test per HARD evidence
function. The SOFT evidence functions (damage window, speed order) are built in
3.4 over the Step-2 calc; their mechanism (floor + apply) is the same one tested
here.
"""
from __future__ import annotations

import pytest

from belief import evidence as ev
from belief.state import SlotBelief


def cand(moves, item="", ability="abilityx", tera=None, weight=1.0):
    """A full candidate dict (the shape build_prior produces)."""
    return {"moves": list(moves), "item": item, "ability": ability,
            "nature": "Hardy", "evs": {}, "tera": tera, "weight": weight}


def _weights(cands):
    return [c["weight"] for c in cands]


# ════════════════════════ EPS floor / never-empty guard ═══════════════════════

def test_posterior_never_empties_off_db_set_survives():
    # Reveal a move NO candidate has -> every likelihood is 0 -> all floor at
    # EPS -> renormalize to a UNIFORM, still-normalizable belief. This is the
    # "off-DB sets survive" guarantee: the belief never collapses to all-zeros.
    cands = [cand(["a", "b", "c", "d"]), cand(["e", "f", "g", "h"]),
             cand(["i", "j", "k", "l"])]
    ev.apply_evidence(cands, ev.revealed_move("zzz_not_in_any_set"))
    w = _weights(cands)
    assert sum(w) == pytest.approx(1.0, abs=1e-9)
    assert all(x > 0 for x in w), "posterior must never be empty"
    assert w == pytest.approx([1 / 3] * 3, abs=1e-6), "all ruled out -> uniform"


def test_hard_evidence_is_effectively_decisive_but_not_exact_zero():
    # One candidate matches, three don't -> the match reads ~1, the non-matches
    # read ~EPS (≈0) — decisive in practice, yet none is exactly 0.
    cands = [cand(["earthquake", "a", "b", "c"]),
             cand(["flamethrower", "d", "e", "f"]),
             cand(["thunderbolt", "g", "h", "i"]),
             cand(["icebeam", "j", "k", "l"])]
    ev.apply_evidence(cands, ev.revealed_move("earthquake"))
    w = _weights(cands)
    assert sum(w) == pytest.approx(1.0, abs=1e-9)
    assert w[0] > 0.999, f"matching candidate should hold ~all mass, got {w[0]}"
    for x in w[1:]:
        assert 0 < x < 1e-6, f"non-match should be ~EPS but >0, got {x}"


def test_eps_floor_rejects_negative_likelihood():
    cands = [cand(["a"])]
    with pytest.raises(ValueError):
        ev.apply_evidence(cands, lambda c: -0.5)


def test_renormalizes_after_every_update():
    cands = [cand(["a", "b"]), cand(["c", "d"]), cand(["e", "f"])]
    ev.apply_evidence(cands, ev.revealed_move("a"))
    assert sum(_weights(cands)) == pytest.approx(1.0, abs=1e-9)
    ev.apply_evidence(cands, ev.revealed_ability("abilityx"))
    assert sum(_weights(cands)) == pytest.approx(1.0, abs=1e-9)


def test_multiple_hard_evidence_compose():
    # Only the candidate satisfying BOTH reveals keeps its mass.
    cands = [cand(["a", "b"], ability="plus"),
             cand(["a", "c"], ability="minus"),
             cand(["d", "b"], ability="plus")]
    ev.apply_evidence(cands, ev.revealed_move("a"))
    ev.apply_evidence(cands, ev.revealed_ability("minus"))
    w = _weights(cands)
    assert sum(w) == pytest.approx(1.0, abs=1e-9)
    assert w[1] > 0.999, f"only the (a,minus) candidate should survive, got {w}"


# ══════════════════════════════ per-evidence ═════════════════════════════════

def test_revealed_move_is_order_blind_membership():
    # wish+protect+foulplay = foulplay+protect+wish (user invariant).
    cands = [cand(["foulplay", "protect", "wish", "struggle"]),
             cand(["earthquake", "roost", "dragonclaw", "toxic"])]
    ev.apply_evidence(cands, ev.revealed_move("wish"))  # 'wish' is at index 2
    w = _weights(cands)
    assert w[0] > 0.999 and w[1] < 1e-6


def test_revealed_ability_specific():
    cands = [cand(["a"], ability="plus"), cand(["b"], ability="static")]
    ev.apply_evidence(cands, ev.revealed_ability("static"))
    w = _weights(cands)
    assert w[1] > 0.999 and w[0] < 1e-6


def test_revealed_item_unknown_is_consistent():
    # item=="" is UNKNOWN -> consistent with any reveal; a specific item must match.
    cands = [cand(["a"], item=""),            # unknown -> survives a reveal
             cand(["b"], item="leftovers"),   # specific, wrong -> ruled out
             cand(["c"], item="choiceband")]  # specific, right -> confirmed
    ev.apply_evidence(cands, ev.revealed_item("choiceband"))
    w = _weights(cands)
    assert sum(w) == pytest.approx(1.0, abs=1e-9)
    assert w[0] > 0.4, f"unknown-item candidate must survive, got {w}"
    assert w[2] > 0.4, f"matching specific item must survive, got {w}"
    assert w[1] < 1e-6, f"wrong specific item must be ruled out, got {w}"


def test_tera_used_unknown_is_consistent():
    # tera=None is UNKNOWN (not "no Tera") -> consistent; a specific type must match.
    cands = [cand(["a"], tera=None),        # unknown -> survives
             cand(["b"], tera="water"),     # specific, wrong -> ruled out
             cand(["c"], tera="fire")]      # specific, right -> confirmed
    ev.apply_evidence(cands, ev.tera_used("fire"))
    w = _weights(cands)
    assert w[0] > 0.4 and w[2] > 0.4
    assert w[1] < 1e-6


def test_choice_lock_confirms_choice_rules_out_others():
    cands = [cand(["a"], item="choiceband"),  # confirmed
             cand(["b"], item="leftovers"),   # ruled out
             cand(["c"], item="")]            # unknown -> consistent
    ev.apply_evidence(cands, ev.choice_lock())
    w = _weights(cands)
    assert sum(w) == pytest.approx(1.0, abs=1e-9)
    assert w[0] > 0.4 and w[2] > 0.4
    assert w[1] < 1e-6


def test_all_choice_items_recognized():
    for it in ("choiceband", "choicescarf", "choicespecs"):
        c = cand(["a"], item=it)
        assert ev.choice_lock()(c) == 1.0
    assert ev.choice_lock()(cand(["a"], item="lifeorb")) == 0.0


def test_hard_evidence_registry_dispatch():
    # update() (3.3) will dispatch by name — the registry must cover the HARD set.
    for name in ("revealed_move", "revealed_ability", "revealed_item",
                 "tera_used", "choice_lock"):
        assert name in ev.HARD_EVIDENCE
    # every factory returns a callable likelihood
    for name, factory in ev.HARD_EVIDENCE.items():
        if name == "choice_lock":
            lik = factory()
        else:
            lik = factory("placeholder")
        assert callable(lik)


def test_slot_belief_integrated_update_stays_normalized():
    # The evidence engine drives SlotBelief.candidates (what 3.3 update() uses).
    sb = SlotBelief("p2a", "ampharos", [
        cand(["thunderbolt", "wish", "protect", "spark"], item="", ability="plus"),
        cand(["flamethrower", "slash", "wingattack", "roost"], item="", ability="static"),
    ])
    ev.apply_evidence(sb.candidates, ev.revealed_move("wish"))
    assert sum(sb.posterior) == pytest.approx(1.0, abs=1e-9)
    assert sb.mass(lambda c: "wish" in c["moves"]) > 0.999
