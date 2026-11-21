"""A3.3 crux — the damage-calc KO assessment (``fitness.ko``), tested hermetically.

The interpretation (``ko_assessment``) is pinned with a FAKE calc result (exact
pcts, no Node, no GPU, deterministic). A few REAL-calc checks (``-k real``) run
the shared ``damage.calc`` (CPU + Node, no GPU) to confirm the interpretation
agrees with the actual damage model on clear-cut scenarios (guaranteed KO,
can't-KO, immune).

Run:  python -m pytest fitness/tests/test_ko.py -v          # hermetic (default)
      python -m pytest fitness/tests/test_ko.py -k real      # + the real-calc checks
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fitness.ko import block_to_calc_mon, ko_assessment, move_ko, opp_move_ko


# ── a minimal get_context() snapshot (just the active own/opp blocks) ──────────
def _ctx(own_hp=1.0, opp_hp=1.0, own="charizard", opp="venusaur",
         own_ability="", opp_ability="", own_status=(), opp_status=()):
    def _blk(sp, hp, ability=(), item="", statuses=()):
        return {"name": sp, "hp_frac": hp, "ability": ability, "item": item,
                "statuses": list(statuses), "boosts": {}}
    return {
        "own": {"p1a": _blk(own, own_hp, own_ability, statuses=list(own_status))},
        "opponent": {"p2a": _blk(opp, opp_hp, opp_ability, statuses=list(opp_status))},
    }


# ── block_to_calc_mon: context block -> @smogon/calc dict ──────────────────────
def test_block_to_calc_mon_maps_known_fields():
    m = block_to_calc_mon(
        {"name": "garchomp", "hp_frac": 0.42, "ability": "roughskin",
         "item": "lifeorb", "statuses": ["brn"], "boosts": {"atk": 1}})
    assert m == {"name": "garchomp", "hp": 0.42, "ability": "roughskin",
                 "item": "lifeorb", "status": "brn", "boosts": {"atk": 1}}


def test_block_to_calc_mon_uses_species_for_opp_and_overrides():
    # opponent blocks carry ``species`` (not ``name``); hidden-info overrides win.
    m = block_to_calc_mon(
        {"species": "garchomp", "hp_frac": 1.0, "ability": "", "item": ""},
        ability="sandveil", item="choicescarf", tera_type="dragon")
    assert m["name"] == "garchomp"
    assert m["ability"] == "sandveil" and m["item"] == "choicescarf"
    assert m["teraType"] == "dragon"


def test_block_to_calc_mon_empty_when_no_species():
    assert block_to_calc_mon({}) == {}
    assert block_to_calc_mon({"hp_frac": 0.5}) == {}  # no species -> no calc dict


def test_block_to_calc_mon_clamps_hp():
    assert block_to_calc_mon({"name": "x", "hp_frac": 1.7})["hp"] == 1.0
    assert block_to_calc_mon({"name": "x", "hp_frac": -0.2})["hp"] == 0.0


# ── ko_assessment: the PURE interpretation (fake dmg, exact pcts) ─────────────
def _dmg(min_pct, max_pct, exp_pct, immune=False):
    return {"min_pct": min_pct, "max_pct": max_pct, "expected_pct": exp_pct,
            "immune": immune}


def test_ko_guaranteed_when_min_exceeds_remaining_hp():
    a = ko_assessment(_dmg(60, 70, 65), defender_hp_frac=0.5)  # threshold 50%
    assert a["guaranteed_ko"] and a["can_ko"] and a["expected_ko"]


def test_ko_can_but_not_guaranteed():
    # min 40 < threshold 45 <= max 50 -> can KO (max) but not guaranteed (min).
    a = ko_assessment(_dmg(40, 50, 45), defender_hp_frac=0.45)
    assert a["can_ko"] and not a["guaranteed_ko"] and a["expected_ko"]


def test_ko_cannot_when_max_below_remaining_hp():
    a = ko_assessment(_dmg(20, 30, 25), defender_hp_frac=1.0)  # threshold 100%
    assert not a["can_ko"] and not a["guaranteed_ko"] and not a["expected_ko"]


def test_ko_immune_never_kos():
    a = ko_assessment(_dmg(0, 0, 0, immune=True), defender_hp_frac=0.1)
    assert a["immune"] and not a["can_ko"] and not a["guaranteed_ko"]


def test_ko_threshold_scales_with_defender_hp():
    # same move (50-60% of max HP): KOs a 40%-HP mon, not a full-HP one.
    d = _dmg(50, 60, 55)
    assert ko_assessment(d, defender_hp_frac=0.40)["guaranteed_ko"]
    assert not ko_assessment(d, defender_hp_frac=1.0)["can_ko"]


def test_ko_assessment_is_json_safe():
    import json
    json.dumps(ko_assessment(_dmg(1, 2, 3), defender_hp_frac=0.5))  # must not raise


# ── move_ko / opp_move_ko: the integration (fake calc_fn -> deterministic) ────
def test_move_ko_builds_dicts_and_interprets():
    seen = {}

    def fake_calc(attacker, defender, move, fmt=None, gen=None):
        seen["attacker"], seen["defender"], seen["move"] = attacker, defender, move
        return _dmg(50, 60, 55)

    a = move_ko(_ctx(own_hp=1.0, opp_hp=0.4), "p1a", "p2a", "ember",
                fmt="gen9ou", calc_fn=fake_calc)
    assert seen["attacker"]["name"] == "charizard"
    assert seen["defender"]["name"] == "venusaur"
    assert seen["defender"]["hp"] == 0.4
    assert a["guaranteed_ko"] and a["can_ko"] and a["expected_ko"]


def test_opp_move_ko_is_the_reverse_direction():
    def fake_calc(attacker, defender, move, fmt=None, gen=None):
        # the OPPOSITE side must be the attacker here (opp mon attacks my mon)
        assert attacker["name"] == "venusaur" and defender["name"] == "charizard"
        return _dmg(80, 90, 85)

    a = opp_move_ko(_ctx(own_hp=0.3, opp_hp=1.0), "p1a", "p2a", "ember",
                    fmt="gen9ou", calc_fn=fake_calc)
    assert a["defender_hp_frac"] == 0.3          # MY mon is the defender
    assert a["guaranteed_ko"] and a["can_ko"]    # 80% >= 30% threshold


def test_move_ko_degrades_when_defender_unknown():
    ctx = {"own": {"p1a": {"name": "charizard", "hp_frac": 1.0}},
           "opponent": {"p2a": {}}}  # no species
    a = move_ko(ctx, "p1a", "p2a", "ember", fmt="gen9ou",
                calc_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no calc")))
    assert a["unknown_defender"] and not a["can_ko"]
    # degraded shape matches the normal path's meaning: threshold = hp*100 (not 0)
    assert a["defender_hp_frac"] == 1.0 and a["ko_threshold_pct"] == 100.0


def test_hp_frac_zero_is_preserved_not_masked_to_full():
    # A genuine 0.0 hp_frac (degenerate/stale) must NOT be masked to 1.0: it has
    # to agree with ko_assessment's own clamping (threshold 0 -> any non-immune
    # move "KOs"), so move_ko / opp_move_ko / ko_assessment stay consistent.
    def fake_calc(attacker, defender, move, fmt=None, gen=None):
        return _dmg(1, 2, 1.5)
    a = move_ko(_ctx(opp_hp=0.0), "p1a", "p2a", "ember", fmt="gen9ou",
                calc_fn=fake_calc)
    assert a["defender_hp_frac"] == 0.0 and a["ko_threshold_pct"] == 0.0
    assert a["guaranteed_ko"] and a["can_ko"]
    # and it is exactly the pure interpretation of the same dmg at hp_frac 0.0
    pure = ko_assessment(_dmg(1, 2, 1.5), defender_hp_frac=0.0)
    assert (a["guaranteed_ko"], a["can_ko"], a["expected_ko"]) == \
        (pure["guaranteed_ko"], pure["can_ko"], pure["expected_ko"])


def test_missing_hp_frac_defaults_to_full_hp():
    # no hp_frac key at all -> 1.0 (full HP), same as ko_assessment's default.
    def fake_calc(attacker, defender, move, fmt=None, gen=None):
        return _dmg(50, 60, 55)
    ctx = {"own": {"p1a": {"name": "charizard"}},
           "opponent": {"p2a": {"name": "venusaur"}}}  # no hp_frac anywhere
    a = move_ko(ctx, "p1a", "p2a", "ember", fmt="gen9ou", calc_fn=fake_calc)
    assert a["defender_hp_frac"] == 1.0 and a["ko_threshold_pct"] == 100.0
    assert not a["can_ko"]  # 50-60% cannot KO a full-HP mon


# ── REAL-calc checks (CPU + Node, no GPU): the interpretation matches the model ─
def test_real_calc_guaranteed_ko_at_low_hp():
    from damage import calc
    # ember (40 BP, 2x vs Grass) vs a 10%-HP Venusaur: ~32-38% >= 10% -> KO.
    a = move_ko(_ctx(own_hp=1.0, opp_hp=0.10), "p1a", "p2a", "ember", fmt="gen9ou")
    assert a["immune"] is False
    assert a["guaranteed_ko"] and a["can_ko"]


def test_real_calc_cannot_ko_full_hp_weak_move():
    # ember (40 BP) vs a full-HP Venusaur: max_pct well under 100 -> no KO.
    # (no calc_fn injected -> move_ko uses the REAL shared damage calc: CPU + Node.)
    a = move_ko(_ctx(own_hp=1.0, opp_hp=1.0), "p1a", "p2a", "ember", fmt="gen9ou")
    assert not a["can_ko"] and not a["guaranteed_ko"]


def test_real_calc_immune_move():
    # Normal 'tackle' vs Ghost Gengar -> immune, never KOs (even at 10% HP).
    a = move_ko(_ctx(own_hp=1.0, opp_hp=0.1, opp="gengar"),
                "p1a", "p2a", "tackle", fmt="gen9ou")
    assert a["immune"] and not a["can_ko"] and not a["guaranteed_ko"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
