"""A3.3b — the KO-aware move-ranking brain (``fitness.strategy``), hermetic.

Pinned with a FAKE calc (exact per-move pcts, no Node, no GPU, deterministic).
The brain is interface-AGNOSTIC (a pure function over the ``get_context()``
snapshot), so these tests hold regardless of the a/b/c wiring choice.

Run:  python -m pytest fitness/tests/test_strategy.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fitness import strategy as S  # noqa: E402


# ── a minimal get_context() snapshot (both species known -> no degradation) ──
def _ctx(own="charizard", opp="venusaur", opp_hp=0.5):
    return {
        "own": {"p1a": {"name": own, "hp_frac": 1.0, "ability": "", "item": "",
                        "statuses": [], "boosts": {}}},
        "opponent": {"p2a": {"name": opp, "hp_frac": opp_hp, "ability": "",
                             "item": "", "statuses": [], "boosts": {}}},
    }


def _dmg(min_pct, max_pct, exp_pct, immune=False):
    return {"min_pct": min_pct, "max_pct": max_pct, "expected_pct": exp_pct,
            "immune": immune}


def fake_calc(table):
    """A calc_fn that maps move-name -> (min,max,exp) pct (deterministic).
    Unknown moves degrade to 0 damage (no KO) so choose_action tests can mix
    known and unknown moves."""
    calls = {"n": 0}

    def _calc(attacker, defender, move, fmt=None, gen=None):
        name = move if isinstance(move, str) else move.get("name")
        calls["n"] += 1
        m, x, e = table.get(name, (0, 0, 0))
        return _dmg(m, x, e)

    return _calc, calls


# ── choose_action: the complete decision (ties the brains + switch together) ──
def _full_ctx(own_hp=0.8, own="charizard", opp="venusaur", opp_hp=0.5,
              opp_moves=("hammer",), stealthrock=False, toxicspikes=False):
    return {
        "active": {"p1": "p1a", "p2": "p2a"},
        "own": {"p1a": {"name": own, "hp_frac": own_hp, "ability": "",
                        "item": "", "statuses": [], "boosts": {},
                        "moves": ["ember", "tackle"]}},
        "opponent": {"p2a": {"species": opp, "hp_frac": opp_hp, "ability": "",
                             "item": "", "statuses": [], "boosts": {},
                             "top_moves": list(opp_moves)}},
        "field": {"stealthrock": {"p2": 1} if stealthrock else {},
                  "toxicspikes": {"p2": 1} if toxicspikes else {}},
    }


def _acts(move_names, switch_species=()):
    acts = [{"kind": "move", "move": m} for m in move_names]
    acts += [{"kind": "switch", "species": s} for s in switch_species]
    return acts


def test_choose_switches_when_threatened():
    # opponent's hammer (min 85) KOs my 80%-HP active (threshold 80) -> guaranteed
    # threat -> switch, even though my HP is healthy.
    calc, _ = fake_calc({"hammer": (85, 95, 90)})
    ctx = _full_ctx(own_hp=0.8, opp_moves=("hammer",))
    acts = _acts(["ember", "tackle"], ["garchomp", "torterra"])
    assert S.choose_action(ctx, acts, fmt="gen9ou", calc_fn=calc)["kind"] == "switch"


def test_choose_best_move_when_safe():
    # opponent's hammer is harmless (10-20% vs my 80%); my ember (min 50) KOs the
    # 50%-HP opponent -> no switch, play the KO move.
    calc, _ = fake_calc({"hammer": (10, 20, 15), "ember": (50, 60, 55),
                         "tackle": (5, 10, 7)})
    ctx = _full_ctx(own_hp=0.8, opp_hp=0.5, opp_moves=("hammer",))
    acts = _acts(["ember", "tackle"], ["garchomp"])
    assert S.choose_action(ctx, acts, fmt="gen9ou", calc_fn=calc) == \
        {"kind": "move", "move": "ember"}


def test_choose_switches_when_low_hp():
    # opponent's move is safe, but my HP (20%) <= 30% threshold -> switch
    calc, _ = fake_calc({"hammer": (10, 20, 15)})
    ctx = _full_ctx(own_hp=0.2, opp_moves=("hammer",))
    acts = _acts(["ember", "tackle"], ["garchomp", "torterra"])
    assert S.choose_action(ctx, acts, fmt="gen9ou", calc_fn=calc)["kind"] == "switch"


def test_choose_force_switch_when_no_moves():
    calc, _ = fake_calc({})
    ctx = _full_ctx()
    acts = [{"kind": "switch", "species": "garchomp"},
            {"kind": "switch", "species": "torterra"}]
    assert S.choose_action(ctx, acts, fmt="gen9ou", calc_fn=calc)["kind"] == "switch"


def test_choose_attacks_when_threatened_but_no_switch():
    # threatened (opponent KOs me) but NO switch legal -> fall back to attacking
    # with the best move (ember KOs the 50%-HP opponent).
    calc, _ = fake_calc({"hammer": (85, 95, 90), "ember": (50, 60, 55),
                         "tackle": (5, 10, 7)})
    ctx = _full_ctx(own_hp=0.8, opp_hp=0.5, opp_moves=("hammer",))
    acts = _acts(["ember", "tackle"])  # no switches
    assert S.choose_action(ctx, acts, fmt="gen9ou", calc_fn=calc) == \
        {"kind": "move", "move": "ember"}


def test_choose_tera_action_is_a_move():
    calc, _ = fake_calc({"ember": (50, 60, 55)})
    ctx = _full_ctx(own_hp=0.8, opp_hp=0.5, opp_moves=())
    acts = [{"kind": "tera", "move": "ember"}, {"kind": "move", "move": "tackle"}]
    # tera ember (KO) beats tackle; no opponent moves -> no defensive switch
    assert S.choose_action(ctx, acts, fmt="gen9ou", calc_fn=calc) == \
        {"kind": "tera", "move": "ember"}


def test_choose_no_acts_passes():
    assert S.choose_action(_full_ctx(), [], fmt="gen9ou",
                           calc_fn=lambda *a, **k: {"min_pct": 0, "max_pct": 0,
                                                   "expected_pct": 0}) is None


def test_choose_hazard_aware_switch_target():
    # Stealth Rock on the opponent's side: prefer the switch-in NOT weak to it.
    # Dragonite is Flying/Dragon (SR-weak); Garchomp is Ground/Dragon (SR-safe).
    calc, _ = fake_calc({"hammer": (85, 95, 90)})
    ctx = _full_ctx(own_hp=0.8, opp_moves=("hammer",), stealthrock=True)
    acts = _acts(["ember"], ["dragonite", "garchomp"])
    assert S.choose_action(ctx, acts, fmt="gen9ou", calc_fn=calc) == \
        {"kind": "switch", "species": "garchomp"}


def test_choose_degenerate_ctx_first_move():
    calc, _ = fake_calc({})
    ctx = {"active": {"p1": None, "p2": None}, "own": {}, "opponent": {}, "field": {}}
    acts = _acts(["ember", "tackle"])
    # can't resolve actives -> a safe, legal move (the first one)
    assert S.choose_action(ctx, acts, fmt="gen9ou", calc_fn=calc) == \
        {"kind": "move", "move": "ember"}


def test_choose_int_slot_zero_is_not_treated_as_missing():
    # Regression for the `is None` (not truthiness) active guard: slot keys here
    # are INTS (0/1). Slot 0 is falsy, so the old `not own_slot` guard would have
    # bailed to the FIRST move. The hardened guard engages the brain instead: the
    # 30%-HP opponent is guaranteed-KO'd by "hammer", so it must beat the weak
    # "tackle" that happens to sit first in `acts`.
    calc, _ = fake_calc({"hammer": (85, 95, 90), "tackle": (5, 10, 7)})
    ctx = {
        "active": {"p1": 0, "p2": 1},
        "own": {0: {"name": "charizard", "hp_frac": 1.0, "ability": "",
                   "item": "", "statuses": [], "boosts": {},
                   "moves": ["tackle", "hammer"]}},
        "opponent": {1: {"species": "venusaur", "hp_frac": 0.3, "ability": "",
                         "item": "", "statuses": [], "boosts": {},
                         "top_moves": []}},
        "field": {},
    }
    acts = _acts(["tackle", "hammer"])  # tackle first = the degenerate bail target
    assert S.choose_action(ctx, acts, fmt="gen9ou", calc_fn=calc) == \
        {"kind": "move", "move": "hammer"}


# ── ko_ladder_score: the pure (tier, magnitude) mapping ──────────────────────
def test_ko_ladder_score_tiers():
    assert S.ko_ladder_score({"guaranteed_ko": True, "expected_pct": 90})[0] == 3
    assert S.ko_ladder_score({"can_ko": True, "expected_pct": 60})[0] == 2
    assert S.ko_ladder_score({"expected_ko": True, "expected_pct": 40})[0] == 1
    assert S.ko_ladder_score({"can_ko": False})[0] == 0
    assert S.ko_ladder_score(None) == (0, 0.0)  # degrades to bottom


def test_ko_ladder_score_within_tier_magnitude():
    # same tier (can_ko), higher expected damage sorts higher
    low = S.ko_ladder_score({"can_ko": True, "expected_pct": 50})
    high = S.ko_ladder_score({"can_ko": True, "expected_pct": 70})
    assert low[0] == high[0] == 2 and high[1] > low[1]


# ── rank_moves: the ordering the heuristic will consume ──────────────────────
def test_rank_prefers_ko_ladder_guaranteed_can_none():
    calc, _ = fake_calc({"hammer": (95, 100, 98),      # guaranteed KO (min>=50)
                         "ember": (45, 55, 50),        # can KO (max>=50, min<50)
                         "tackle": (10, 20, 15)})      # no KO
    r = S.rank_moves(_ctx(opp_hp=0.5), "p1a", "p2a",
                     ["tackle", "ember", "hammer"], fmt="gen9ou", calc_fn=calc)
    assert r["order"][0] == "hammer" and r["best"] == "hammer"
    assert r["order"] == ["hammer", "ember", "tackle"]


def test_rank_within_tier_higher_damage_wins():
    # both "can KO" (max>=50, min<50): the bigger expected damage ranks first
    calc, _ = fake_calc({"a": (40, 60, 55), "b": (40, 52, 48)})
    r = S.rank_moves(_ctx(opp_hp=0.5), "p1a", "p2a", ["a", "b"],
                     fmt="gen9ou", calc_fn=calc)
    assert r["order"] == ["a", "b"]


def test_rank_status_move_sorts_below_damaging_move():
    # a status move deals 0 (no KO, no damage) -> below a non-KO damaging move
    calc, _ = fake_calc({"wish": (0, 0, 0), "ember": (10, 20, 15)})
    r = S.rank_moves(_ctx(opp_hp=0.5), "p1a", "p2a", ["wish", "ember"],
                     fmt="gen9ou", calc_fn=calc)
    assert r["order"][0] == "ember" and r["order"][-1] == "wish"


def test_rank_degrades_stably_when_opponent_unknown():
    # opponent species unknown -> the KO core degrades (all tier 0); the sort is
    # STABLE (original order) and NEVER raises. The fake calc must not be called
    # for an unknown defender (the KO core short-circuits before the calc).
    def boom(*a, **k):
        raise AssertionError("calc must not run when the defender is unknown")
    ctx = {"own": {"p1a": {"name": "charizard", "hp_frac": 1.0}},
           "opponent": {"p2a": {}}}  # no species
    r = S.rank_moves(ctx, "p1a", "p2a", ["ember", "tackle"],
                     fmt="gen9ou", calc_fn=boom)
    assert r["order"] == ["ember", "tackle"]  # stable, original order
    assert r["best"] == "ember"
    assert r["assessments"]["ember"]["unknown_defender"] is True


# ══ Step 6.2 — rule 1: hazard setup (hermetic; fake calc, no Node/GPU) ══

def test_hazard_rule_sets_stealthrock_over_weak_attack():
    # SR legal + our field clean + opp (venusaur) not Flying -> hazard over 15% attack
    calc, _ = fake_calc({"ember": (10, 20, 15)})
    ctx = _full_ctx(opp_moves=())
    a = S.choose_action(ctx, _acts(["Stealth Rock", "ember"]),
                        fmt="gen9ou", calc_fn=calc)
    assert a["move"] == "Stealth Rock"


def test_hazard_rule_skips_when_our_field_already_has_rocks():
    calc, _ = fake_calc({"hammer": (40, 60, 50)})
    ctx = _full_ctx(opp_moves=())
    ctx["field"]["stealthrock"] = {"p1": True}
    a = S.choose_action(ctx, _acts(["Stealth Rock", "hammer"]),
                        fmt="gen9ou", calc_fn=calc)
    assert a["move"] == "hammer"


def test_hazard_rule_skips_when_guaranteed_ko_threat():
    # opp hammer (min 85) KOs my 80%-HP active -> guaranteed_ko -> attack, not SR
    calc, _ = fake_calc({"hammer": (85, 95, 90), "ember": (10, 20, 15)})
    ctx = _full_ctx(own_hp=0.8, opp_moves=("hammer",))
    a = S.choose_action(ctx, _acts(["Stealth Rock", "ember"]),
                        fmt="gen9ou", calc_fn=calc)
    assert a["move"] == "ember"


def test_hazard_rule_skips_spikes_vs_flying_opponent():
    # pidgeot is Flying -> immune to Spikes (immune fraction 1.0 > 0.5) -> attack
    calc, _ = fake_calc({"ember": (10, 20, 15)})
    ctx = _full_ctx(opp="pidgeot", opp_moves=())
    a = S.choose_action(ctx, _acts(["Spikes", "ember"]),
                        fmt="gen9ou", calc_fn=calc)
    assert a["move"] == "ember"


def test_hazard_rule_sets_when_half_bench_vulnerable():
    # active pidgeot (Flying/immune) + bench venusaur (vulnerable) ->
    # immune fraction 0.5 <= 0.5 -> hazards still set (design: >= half vulnerable)
    calc, _ = fake_calc({"ember": (10, 20, 15)})
    ctx = _full_ctx(opp="pidgeot", opp_moves=())
    ctx["opponent"]["p2b"] = {"species": "venusaur", "hp_frac": 0.0,
                               "statuses": [], "boosts": {}, "moves": []}
    a = S.choose_action(ctx, _acts(["Spikes", "ember"]),
                        fmt="gen9ou", calc_fn=calc)
    assert a["move"] == "Spikes"


def test_hazard_rule_toxic_spikes_name_set_covers_db_gap():
    # toxicspikes is NOT in legality.HAZARDS — the canonical name set must
    # still classify it. Steel (magnezone) immune -> attack; Water -> set.
    calc, _ = fake_calc({"ember": (10, 20, 15)})
    ctx_steel = _full_ctx(opp="magnezone", opp_moves=())
    a = S.choose_action(ctx_steel, _acts(["Toxic Spikes", "ember"]),
                        fmt="gen9ou", calc_fn=calc)
    assert a["move"] == "ember"
    ctx_water = _full_ctx(opp="blastoise", opp_moves=())
    b = S.choose_action(ctx_water, _acts(["Toxic Spikes", "ember"]),
                        fmt="gen9ou", calc_fn=calc)
    assert b["move"] == "Toxic Spikes"


def test_hazard_rule_ignores_opp_side_hazards():
    # the OPPONENT's rocks (p2) must not block MY setup — only p1 does.
    calc, _ = fake_calc({"ember": (10, 20, 15)})
    ctx = _full_ctx(opp_moves=())
    ctx["field"]["stealthrock"] = {"p2": True}
    a = S.choose_action(ctx, _acts(["Stealth Rock", "ember"]),
                        fmt="gen9ou", calc_fn=calc)
    assert a["move"] == "Stealth Rock"


def test_hazard_rule_sticky_web_untracked_field_still_set():
    # sticky web is not in the belief field -> always "clean" -> free value
    calc, _ = fake_calc({"ember": (10, 20, 15)})
    ctx = _full_ctx(opp_moves=())
    a = S.choose_action(ctx, _acts(["Sticky Web", "ember"]),
                        fmt="gen9ou", calc_fn=calc)
    assert a["move"] == "Sticky Web"


def test_rank_empty_moves():
    r = S.rank_moves(_ctx(), "p1a", "p2a", [], fmt="gen9ou",
                     calc_fn=lambda *a, **k: _dmg(0, 0, 0))
    assert r == {"order": [], "best": None, "assessments": {}}


def test_rank_memoises_one_calc_per_distinct_move():
    # each distinct move is calc'd exactly once, even if it appears twice
    calc, calls = fake_calc({"ember": (10, 20, 15), "tackle": (5, 10, 7)})
    S.rank_moves(_ctx(opp_hp=0.5), "p1a", "p2a",
                 ["ember", "tackle", "ember"], fmt="gen9ou", calc_fn=calc)
    assert calls["n"] == 2  # ember once + tackle once (not 3)


def test_rank_accepts_dict_move_specs():
    calc, _ = fake_calc({"hammer": (95, 100, 98), "tackle": (10, 20, 15)})
    r = S.rank_moves(_ctx(opp_hp=0.5), "p1a", "p2a",
                     [{"name": "tackle"}, {"name": "hammer"}],
                     fmt="gen9ou", calc_fn=calc)
    assert r["order"][0] == {"name": "hammer"} and r["best"] == {"name": "hammer"}


# ── threat_assessment: the DEFENSIVE half (will the opponent KO me?) ──────────
def _own_ctx(own_hp=0.5, own="charizard", opp="venusaur"):
    return {
        "own": {"p1a": {"name": own, "hp_frac": own_hp, "ability": "",
                        "item": "", "statuses": [], "boosts": {}}},
        "opponent": {"p2a": {"name": opp, "hp_frac": 1.0, "ability": "",
                             "item": "", "statuses": [], "boosts": {}}},
    }


def test_threat_detects_guaranteed_ko():
    # my mon at 50% HP (threshold 50): "hammer" min 60 >= 50 -> guaranteed KO
    calc, _ = fake_calc({"hammer": (60, 70, 65), "tackle": (10, 20, 15)})
    t = S.threat_assessment(_own_ctx(own_hp=0.5), "p1a", "p2a",
                            ["tackle", "hammer"], fmt="gen9ou", calc_fn=calc)
    assert t["threat"] == "guaranteed_ko" and t["worst"] == "hammer"


def test_threat_picks_the_hardest_ko_as_worst():
    # my mon at 50% (threshold 50):
    #   c: min 50 >= 50 -> guaranteed_ko  (hardest)
    #   a: max 55 >= 50, min 45 < 50 -> can_ko
    #   b: max 45 < 50 -> no KO
    calc, _ = fake_calc({"a": (45, 55, 50), "b": (40, 45, 42), "c": (50, 60, 55)})
    t = S.threat_assessment(_own_ctx(own_hp=0.5), "p1a", "p2a",
                            ["a", "b", "c"], fmt="gen9ou", calc_fn=calc)
    assert t["threat"] == "guaranteed_ko" and t["worst"] == "c"


def test_threat_can_ko_tier():
    # my mon at 45% (threshold 45): "ember" max 50 >= 45 but min 40 < 45 -> can_ko
    calc, _ = fake_calc({"ember": (40, 50, 45), "tackle": (5, 10, 7)})
    t = S.threat_assessment(_own_ctx(own_hp=0.45), "p1a", "p2a",
                            ["tackle", "ember"], fmt="gen9ou", calc_fn=calc)
    assert t["threat"] == "can_ko" and t["worst"] == "ember"


def test_threat_safe_when_no_move_kos():
    # my mon full HP (threshold 100): every move deals < 100 -> safe;
    # the worst is still the higher-damage one
    calc, _ = fake_calc({"a": (10, 20, 15), "b": (30, 40, 35)})
    t = S.threat_assessment(_own_ctx(own_hp=1.0), "p1a", "p2a",
                            ["a", "b"], fmt="gen9ou", calc_fn=calc)
    assert t["threat"] == "safe" and t["worst"] == "b"


def test_threat_unknown_when_opponent_species_unknown():
    # opponent species unknown -> opp_move_ko degrades (unknown_attacker) -> "unknown"
    def boom(*a, **k):
        raise AssertionError("calc must not run when the attacker is unknown")
    ctx = {"own": {"p1a": {"name": "charizard", "hp_frac": 0.5}},
           "opponent": {"p2a": {}}}  # no species
    t = S.threat_assessment(ctx, "p1a", "p2a", ["ember", "hammer"],
                            fmt="gen9ou", calc_fn=boom)
    assert t["threat"] == "unknown" and t["worst"] is not None
    assert t["assessments"]["ember"]["unknown_attacker"] is True


def test_threat_empty_moves():
    t = S.threat_assessment(_own_ctx(), "p1a", "p2a", [], fmt="gen9ou",
                            calc_fn=lambda *a, **k: {"min_pct": 0, "max_pct": 0,
                                                    "expected_pct": 0, "immune": False})
    assert t == {"threat": "unknown", "worst": None, "assessments": {}}


# ── REAL integration: real BeliefState -> real get_context() -> the brain ──────
# Closes the hermetic-only gap: every other brain test above uses a HAND-CRAFTED
# ctx dict. This proves the brain consumes a REAL get_context() snapshot (field
# names, slot keys, top_moves) WITHOUT silently degrading, using the REAL shared
# damage calc (CPU + Node, no GPU). If a field the brain reads were missing from
# get_context(), this would fail (hermetic tests would not catch it).
def test_real_get_context_feeds_the_brain_end_to_end():
    from belief import events as E
    from belief.state import BeliefState

    own = {"p1a": {"name": "garchomp", "ability": "roughskin", "item": "lifeorb",
                   "evs": {"atk": 252},
                   "moves": ["earthquake", "dragonclaw", "stoneedge", "toxic"]}}
    bs = BeliefState("gen9ou", own_team=own)
    bs.add_opponent("p2a", "blastoise")  # real prior -> real posterior/top_moves
    bs.update(E.SwitchIn(side="p1", slot="p1a", species="garchomp", hp=183, maxhp=183))
    bs.update(E.SwitchIn(side="p2", slot="p2a", species="blastoise", hp=183, maxhp=183))
    # knock the opponent down to 1 HP -> any real damaging move must KO it
    bs.update(E.DamageObserved(target_side="p2", target_slot="p2a", hp_after=1,
                               maxhp=183, ko=False, source_kind="move",
                               move="dragonclaw", source_side="p1", source_slot="p1a"))
    bs.update(E.MoveUsed(side="p2", slot="p2a", move="hydropump"))  # reveal a move

    ctx = bs.get_context()
    own_slot, opp_slot = ctx["active"]["p1"], ctx["active"]["p2"]

    # 1) structure: the exact fields the brain reads are present (the pinned gap).
    assert ctx["own"][own_slot]["name"] == "garchomp"
    assert ctx["opponent"][opp_slot]["species"] == "blastoise"
    assert 0.0 < ctx["opponent"][opp_slot]["hp_frac"] <= 1.0
    assert ctx["opponent"][opp_slot]["top_moves"], "opponent moves must be known"

    # 2) the brain runs on the REAL ctx with the REAL calc (no crash / no degrade).
    moves = ctx["own"][own_slot]["moves"]
    acts = [{"kind": "move", "move": m} for m in moves]
    r = S.rank_moves(ctx, own_slot, opp_slot, moves, fmt="gen9ou")
    assert r["order"], "brain must not silently degrade to an empty order"
    # 3) at 1 HP the top move is a genuine KO (calc + interpretation agree).
    assert (r["assessments"][r["best"]]["guaranteed_ko"]
            or r["assessments"][r["best"]]["can_ko"])
    # 4) choose_action returns a legal DAMAGING move (never the status 'toxic').
    chosen = S.choose_action(ctx, acts, fmt="gen9ou")
    assert chosen is not None and chosen.get("move") in moves
    assert chosen.get("move") != "toxic"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
