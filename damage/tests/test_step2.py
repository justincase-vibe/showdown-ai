"""Step 2 acceptance tests (build plan: "Damage Calculator Wrapper").

Run from the project root:

    python -m pytest damage/tests -v

What this verifies (the Step 2 gate):
  * 15-20 hand-checked matchups match the @smogon/calc reference within rounding
    (values below were independently recomputed from the standard damage formula
    ``floor(floor(floor((2L/5+2)*BP*atk/def)/50)+2)`` with the library's own base
    stats, natures, STAB/type/burn, then cross-checked against @smogon/calc);
  * STAB, type effectiveness (super-effective / neutral / resisted), burn;
  * boosts (offence +2, offence -2, defence +2) move damage in the right direction;
  * weather (Rain boosts Water, Sun boosts Fire) and screens (Reflect / Light
    Screen halve physical / special);
  * known items (Life Orb x1.3, Choice Band x1.5 Atk) and ability (Adaptability
    doubles STAB) when present;
  * critical hits are a SEPARATE value and ignore a negative Attack drop;
  * immunities return 0;
  * the speed-comparison helper agrees with known orderings (Choice Scarf,
    paralysis, boosts, Tailwind).

Every expected range is hard-coded from the verified reference above — the test
asserts the wrapper reproduces it, so a regression in either the formula path or
the subprocess marshalling fails loudly.
"""
from __future__ import annotations

import pytest

from damage import calc


# ── standard test mons (IVs default to 31 in the wrapper) ─────────────
GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}}       # Dragon/Ground
TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}         # Rock/Dark
CHARZ = {"name": "Charizard", "nature": "Modest", "evs": {"spa": 252}}       # Fire/Flying
BLAST = {"name": "Blastoise", "nature": "Modest", "evs": {"spa": 252}}       # Water
TOGE = {"name": "Togekiss", "nature": "Modest", "evs": {"spa": 252}}         # Fairy/Flying
LUC = {"name": "Lucario", "nature": "Jolly", "evs": {"atk": 252}}            # Fighting/Steel

TITAN_341 = 341  # Tyranitar Bold 252 Def, 0 HP EV -> max HP (library-reported)


def dmg(attacker, defender, move, field=None):
    return calc.calc_damage(attacker, defender, move, field=field)


def rng(result):
    return (result["min"], result["max"])


# ── API surface ───────────────────────────────────────────────────────
def test_result_shape():
    r = dmg(GARCH, TITAN, "earthquake")
    for key in ("min", "max", "expected", "min_pct", "max_pct", "expected_pct",
                "crit", "immune", "maxhp", "desc"):
        assert key in r, f"missing key {key}"
    assert isinstance(r["min"], int) and isinstance(r["max"], int)
    assert r["crit"]["min"] <= r["crit"]["max"]
    assert r["min"] <= r["expected"] <= r["max"]
    assert 0.0 <= r["min_pct"] <= r["max_pct"]


def test_pct_is_fraction_of_maxhp():
    r = dmg(GARCH, TITAN, "earthquake")
    assert r["maxhp"] == TITAN_341
    assert r["max_pct"] == pytest.approx(r["max"] / TITAN_341 * 100, abs=0.01)


# ── hand-checked matchups (STAB / type / burn / neutral) ──────────────
def test_stab_super_effective_earthquake():
    # Garchomp (Dragon/Ground) EQ = 1.5x STAB x 2x (Ground vs Rock) = 3x
    assert rng(dmg(GARCH, TITAN, "earthquake")) == (242, 288)


def test_stab_neutral_outrage():
    # Outrage (Dragon) vs Rock/Dark = 1x, x 1.5 STAB
    assert rng(dmg(GARCH, TITAN, "outrage")) == (145, 172)


def test_resisted_flamethrower():
    # Fire vs Rock/Dark = 0.5x, x 1.5 STAB = 0.75x  (special -> uses SpD)
    assert rng(dmg(CHARZ, TITAN, "flamethrower")) == (72, 84)


def test_burn_halves_physical():
    assert rng(dmg({**GARCH, "status": "brn"}, TITAN, "earthquake")) == (121, 144)


def test_resisted_dragon_pulse_no_stab():
    # Dragon Pulse vs Fighting/Steel = 0.5x, no STAB (Togekiss is Fairy/Flying)
    assert rng(dmg(TOGE, LUC, "dragonpulse")) == (64, 76)


# ── boosts move damage in the right direction ─────────────────────────
def test_offence_boost_plus2_doubles():
    assert rng(dmg({**GARCH, "boosts": {"atk": 2}}, TITAN, "earthquake")) == (486, 572)


def test_offence_boost_minus2_halves():
    assert rng(dmg({**GARCH, "boosts": {"atk": -2}}, TITAN, "earthquake")) == (122, 146)


def test_defence_boost_plus2_halves():
    assert rng(dmg(GARCH, {**TITAN, "boosts": {"def": 2}}, "earthquake")) == (122, 146)


def test_boost_direction_ordering():
    base = dmg(GARCH, TITAN, "earthquake")["max"]
    up = dmg({**GARCH, "boosts": {"atk": 2}}, TITAN, "earthquake")["max"]
    down = dmg({**GARCH, "boosts": {"atk": -2}}, TITAN, "earthquake")["max"]
    assert up > base > down


# ── weather ───────────────────────────────────────────────────────────
def test_rain_boosts_water():
    dry = dmg(BLAST, TITAN, "surf")
    rain = dmg(BLAST, TITAN, "surf", field={"weather": "Rain"})
    assert rng(dry) == (242, 288)
    assert rng(rain) == (366, 432)
    assert rain["max"] == pytest.approx(dry["max"] * 1.5, abs=1)


def test_sun_boosts_fire():
    dry = dmg(CHARZ, TITAN, "flamethrower")
    sun = dmg(CHARZ, TITAN, "flamethrower", field={"weather": "Sun"})
    assert rng(sun) == (107, 126)
    assert sun["max"] == pytest.approx(dry["max"] * 1.5, abs=1)


# ── screens halve the matching category ──────────────────────────────
def test_reflect_halves_physical():
    assert rng(dmg(GARCH, TITAN, "earthquake",
                   field={"defenderSide": {"isReflect": True}})) == (121, 144)


def test_light_screen_halves_special():
    assert rng(dmg(CHARZ, TITAN, "flamethrower",
                   field={"defenderSide": {"isLightScreen": True}})) == (36, 42)


def test_reflect_does_not_affect_special():
    # Light Screen should NOT reduce a physical move; Reflect should NOT reduce special.
    phys = dmg(GARCH, TITAN, "earthquake", field={"defenderSide": {"isLightScreen": True}})
    assert phys["max"] == 288  # unchanged
    spec = dmg(CHARZ, TITAN, "flamethrower", field={"defenderSide": {"isReflect": True}})
    assert spec["max"] == 84  # unchanged


# ── items & ability (when known) ─────────────────────────────────────
def test_life_orb_boosts_1_3():
    assert rng(dmg({**GARCH, "item": "Life Orb"}, TITAN, "earthquake")) == (315, 374)


def test_choice_band_boosts_atk_1_5():
    assert rng(dmg({**GARCH, "item": "Choice Band"}, TITAN, "earthquake")) == (362, 428)


def test_adaptability_doubles_stab():
    # 1.5 STAB -> 2.0 STAB, i.e. base x (2/1.5)
    assert rng(dmg({**GARCH, "ability": "Adaptability"}, TITAN, "earthquake")) == (324, 384)


# ── critical hits: separate value, ignore negative Attack drop ────────
def test_crit_is_separate_and_higher():
    r = dmg(GARCH, TITAN, "earthquake")
    assert r["crit"]["min"] > r["min"]
    assert r["crit"]["max"] > r["max"]
    assert r["crit"]["min"] <= r["crit"]["expected"] <= r["crit"]["max"]


def test_crit_ignores_negative_attack_boost():
    r = dmg({**GARCH, "boosts": {"atk": -2}}, TITAN, "earthquake")
    # non-crit is halved by the -2 drop...
    assert (r["min"], r["max"]) == (122, 146)
    # ...but the crit ignores the drop and is ~1.5x the unboosted base (242-288).
    assert (r["crit"]["min"], r["crit"]["max"]) == (366, 432)


# ── immunities return 0 ───────────────────────────────────────────────
@pytest.mark.parametrize("attacker,defender,move", [
    ({"name": "Pikachu"}, {"name": "Garchomp"}, "thunderbolt"),   # Electric vs Ground
    ({"name": "Machamp"}, {"name": "Gengar"}, "closecombat"),      # Fighting vs Ghost
    (GARCH, {"name": "Togekiss"}, "outrage"),                       # Dragon vs Fairy
])
def test_immunity_returns_zero(attacker, defender, move):
    r = dmg(attacker, defender, move)
    assert r["immune"] is True
    assert (r["min"], r["max"]) == (0, 0)
    assert r["expected"] == 0
    assert r["min_pct"] == 0 and r["max_pct"] == 0


# ── speed-comparison helper ──────────────────────────────────────────
def test_speed_base_values():
    assert calc.effective_speed({"name": "Garchomp"}) == 240
    assert calc.effective_speed({"name": "Garchomp", "nature": "Jolly"}) == 264
    assert calc.effective_speed({"name": "Tyranitar"}) == 158


def test_speed_boosts():
    g = {"name": "Garchomp"}
    assert calc.effective_speed({**g, "boosts": {"spe": 1}}) == 360
    assert calc.effective_speed({**g, "boosts": {"spe": 2}}) == 480
    assert calc.effective_speed({**g, "boosts": {"spe": -1}}) == 160
    assert calc.effective_speed({**g, "boosts": {"spe": -2}}) == 120


def test_speed_choice_scarf():
    assert calc.effective_speed({"name": "Garchomp", "item": "Choice Scarf"}) == 360
    assert calc.effective_speed({"name": "Tyranitar", "item": "Choice Scarf"}) == 237


def test_speed_paralysis():
    # gen 9: paralysis x0.5 (matches @smogon/calc and this project's sim; see D30)
    assert calc.effective_speed({"name": "Garchomp", "status": "par"}) == 120
    assert calc.effective_speed({"name": "Garchomp", "item": "Choice Scarf", "status": "par"}) == 180


def test_speed_tailwind():
    assert calc.effective_speed({"name": "Garchomp"}, tailwind=True) == 480


def test_speed_ordering_healthy():
    assert calc.compare_speed({"name": "Garchomp"}, {"name": "Tyranitar"})["faster"] == "a"


def test_speed_paralysis_flips_order():
    # Paralyzed Garchomp (120) is slower than healthy Tyranitar (158).
    assert calc.compare_speed({"name": "Garchomp", "status": "par"},
                              {"name": "Tyranitar"})["faster"] == "b"


def test_speed_scarf_close_call():
    # Scarfed Tyranitar (237) still loses to healthy Garchomp (240), but beats a
    # paralyzed Garchomp (120).
    assert calc.compare_speed({"name": "Garchomp"},
                              {"name": "Tyranitar", "item": "Choice Scarf"})["faster"] == "a"
    assert calc.compare_speed({"name": "Garchomp", "status": "par"},
                              {"name": "Tyranitar", "item": "Choice Scarf"})["faster"] == "b"


def test_speed_boost_can_flip_order():
    garch = {"name": "Garchomp"}
    assert calc.compare_speed({**garch, "boosts": {"spe": 2}}, {"name": "Tyranitar"})["faster"] == "a"
    # A -2 Speed drop (120) puts Garchomp under Tyranitar (158).
    assert calc.compare_speed({**garch, "boosts": {"spe": -2}}, {"name": "Tyranitar"})["faster"] == "b"


def test_speed_tailwind_flips_order():
    # Tailwind Garchomp (480) vs Tyranitar (158): Garchomp faster.
    assert calc.compare_speed({"name": "Garchomp"}, {"name": "Tyranitar"},
                              a_tailwind=True)["faster"] == "a"
