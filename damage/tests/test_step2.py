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


def test_calc_damage_many_batch_matches_individual():
    """Batch mode (forward-eval path): N damage calcs in ONE Node spawn return
    the SAME results, IN ORDER, as N individual calc_damage calls — and one bad
    item yields {'error': ...} for just that item (the rest still return)."""
    items = [
        {"attacker": GARCH, "defender": TITAN, "move": "earthquake"},
        {"attacker": CHARZ, "defender": BLAST, "move": "fireblast",
         "field": {"weather": "Sun"}},
        {"attacker": LUC, "defender": TITAN, "move": "closecombat"},
    ]
    batch = calc.calc_damage_many(items)
    assert len(batch) == 3
    for it, b in zip(items, batch):
        single = calc.calc_damage(it["attacker"], it["defender"], it["move"],
                                  field=it.get("field"))
        assert b == single, f"batch != single for {it['move']}"

    # a bad item (mon with no name) is isolated: {error} for it, the rest fine
    mixed = [
        {"attacker": GARCH, "defender": TITAN, "move": "earthquake"},
        {"attacker": {"no_name": True}, "defender": TITAN, "move": "earthquake"},
        {"attacker": LUC, "defender": TITAN, "move": "closecombat"},
    ]
    out = calc.calc_damage_many(mixed)
    assert len(out) == 3
    assert "min" in out[0] and "max" in out[0]      # good
    assert "error" in out[1]                          # the bad one is isolated
    assert "min" in out[2] and "max" in out[2]        # good


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


# ── item / ability IDS must behave like their display names ───────────
# @smogon/calc resolves item/ability by DISPLAY NAME and silently ignores ids,
# but the AI / legality DB and the sim fork all carry ids. The engine therefore
# must normalise id -> name (see node_calc.js normItem/normAbility). These tests
# pin that contract so a silent drop of an item/ability effect can't regress.
def test_item_id_matches_display_name():
    assert rng(dmg({**GARCH, "item": "lifeorb"}, TITAN, "earthquake")) == rng(
        dmg({**GARCH, "item": "Life Orb"}, TITAN, "earthquake")) == (315, 374)
    assert rng(dmg({**GARCH, "item": "choiceband"}, TITAN, "earthquake")) == rng(
        dmg({**GARCH, "item": "Choice Band"}, TITAN, "earthquake")) == (362, 428)


def test_ability_id_matches_display_name():
    assert rng(dmg({**GARCH, "ability": "adaptability"}, TITAN, "earthquake")) == rng(
        dmg({**GARCH, "ability": "Adaptability"}, TITAN, "earthquake")) == (324, 384)


def test_item_id_not_silently_dropped():
    # Life Orb (x1.3) must change the number vs no item — guards against the
    # library silently ignoring an id (the historical bug: 37-44 with 'lifeorb').
    no_item = dmg(GARCH, TITAN, "earthquake")
    with_id = dmg({**GARCH, "item": "lifeorb"}, TITAN, "earthquake")
    assert with_id["max"] > no_item["max"]


# ── 10 extended hand-checked matchups (2026-11-09, "double-check the calc") ───
# Each range is the engine's output, independently cross-checked against the live
# fork sim (scratch_crosscheck_calc_vs_sim.js). Broadens coverage: STAB,
# super-effective, resisted, dual-type, immunity, item, +2 boost, priority move,
# and ability. These pin concrete numbers so any engine regression fails loudly.
def test_ext_close_combat_super_eff_stab():
    a = {"name": "Lucario", "nature": "Jolly", "evs": {"atk": 252}}
    d = {"name": "Chansey", "nature": "Bold", "evs": {"def": 252}}
    assert rng(dmg(a, d, "close combat")) == (692, 816)


def test_ext_scald_resisted():
    a = {"name": "Toxapex", "nature": "Calm", "evs": {"spa": 252}}
    d = {"name": "Garchomp", "nature": "Bold", "evs": {"def": 252}}
    assert rng(dmg(a, d, "scald")) == (85, 102)


def test_ext_brave_bird_super_eff_stab():
    a = {"name": "Corviknight", "nature": "Adamant", "evs": {"atk": 252}}
    d = {"name": "Mewtwo", "nature": "Calm", "evs": {"spd": 252}}
    assert rng(dmg(a, d, "brave bird")) == (180, 213)


def test_ext_surging_strikes_dual_type():
    a = {"name": "Urshifu", "nature": "Adamant", "evs": {"atk": 252}}
    d = {"name": "Toxapex", "nature": "Calm", "evs": {"spd": 252}}
    assert rng(dmg(a, d, "surging strikes")) == (48, 57)


def test_ext_earth_power_flying_immunity():
    a = {"name": "Landorus", "nature": "Modest", "evs": {"spa": 252}}
    d = {"name": "Hydreigon", "nature": "Calm", "evs": {"spd": 252}}
    r = dmg(a, d, "earth power")
    assert r["immune"] is True and r["min"] == r["max"] == 0


def test_ext_life_orb_ground():
    a = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}, "item": "lifeorb"}
    d = {"name": "Suicune", "nature": "Calm", "evs": {"spd": 252}}
    # Engine-verified (gen9 @smogon/calc): 208-246, i.e. no-LO range 160-189 x1.3
    assert rng(dmg(a, d, "earthquake")) == (208, 246)


def test_ext_psychic():
    a = {"name": "Alakazam", "nature": "Timid", "evs": {"spa": 252}}
    d = {"name": "Garchomp", "nature": "Calm", "evs": {"spd": 252}}
    assert rng(dmg(a, d, "psychic")) == (121, 144)


def test_ext_outrage_plus2_atk():
    a = {"name": "Dragonite", "nature": "Adamant", "evs": {"atk": 252}, "boosts": {"atk": 2}}
    d = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}
    assert rng(dmg(a, d, "outrage")) == (297, 351)


def test_ext_flame_charge():
    a = {"name": "Blaziken", "nature": "Modest", "evs": {"spa": 252}}
    d = {"name": "Lucario", "nature": "Bold", "evs": {"def": 252}}
    assert rng(dmg(a, d, "flame charge")) == (102, 122)


def test_ext_aqua_jet_adaptability_priority():
    a = {"name": "Inteleon", "nature": "Jolly", "evs": {"atk": 252}, "ability": "adaptability"}
    d = {"name": "Chansey", "nature": "Bold", "evs": {"def": 252}}
    r = dmg(a, d, "aqua jet")
    assert (r["min"], r["max"]) == (130, 154)
    assert (r["crit"]["min"], r["crit"]["max"]) == (194, 230)


# ── Guts / Aurora Veil / burn interactions (2026-11-09, engine-verified,  ──
# ── plus Guts + veil cross-checked against the LIVE fork sim — 13/13 over 4 ──
# ── seeds in scratch_crosscheck_calc_vs_sim.js) ──────────────────────────
GUTS_MACH = {"name": "Machamp", "nature": "Adamant", "evs": {"atk": 252}}       # Guts user
CHANSEY = {"name": "Chansey", "nature": "Bold", "evs": {"def": 252}}


def test_guts_burned_physical_gets_1_5_not_halved():
    # Fork ground truth: guts.onModifyAtk -> any status => Atk x1.5;
    # battle-actions.js skips the burn halving for Guts users. So a BURNED Guts
    # user hits at 1.5x, not 0.5x. (Library id-vs-name gotcha: 'guts' id must
    # normalise to 'Guts' or the effect silently drops to 0.5x.)
    assert rng(dmg({**GUTS_MACH, "ability": "guts", "status": "brn"},
                   CHANSEY, "closecombat")) == (1278, 1506)
    # cross-check against the non-Guts burned baseline (below): exactly x1.5


def test_guts_burned_id_and_name_agree():
    # id and display name must both work (same contract as the item/ability fix)
    assert rng(dmg({**GUTS_MACH, "ability": "guts", "status": "brn"},
                   CHANSEY, "closecombat")) == \
        rng(dmg({**GUTS_MACH, "ability": "Guts", "status": "brn"},
                CHANSEY, "closecombat")) == (1278, 1506)


def test_guts_healthy_is_noop():
    # Guts needs a status condition; healthy Guts == healthy no-ability (852-1004)
    assert rng(dmg({**GUTS_MACH, "ability": "guts"}, CHANSEY, "closecombat")) == (852, 1004)


def test_burn_halves_non_guts_physical():
    # Non-Guts burned physical = 0.5x  (426-502 == 852-1004 x0.5)
    assert rng(dmg({**GUTS_MACH, "ability": "limber", "status": "brn"},
                   CHANSEY, "closecombat")) == (426, 502)


def test_guts_burned_crit_keeps_boost_ignores_burn():
    # Crit ignores the burn (as always) but still carries the Guts x1.5 Atk:
    # healthy (852-1004) crit = 1278-1506; burned Guts crit = that x1.5 = 1920-2258.
    r = dmg({**GUTS_MACH, "ability": "guts", "status": "brn"}, CHANSEY, "closecombat")
    assert (r["crit"]["min"], r["crit"]["max"]) == (1920, 2258)


def test_guts_vs_burned_non_guts_ratio():
    # Burned non-Guts = x0.5 (burn); burned Guts = x1.5 (Guts, no burn). So a
    # burned Guts user hits a burned non-Guts user for 3x (1.5 / 0.5), not 1.5x.
    non_guts = dmg({**GUTS_MACH, "ability": "limber", "status": "brn"}, CHANSEY, "closecombat")
    guts = dmg({**GUTS_MACH, "ability": "guts", "status": "brn"}, CHANSEY, "closecombat")
    assert guts["max"] == pytest.approx(non_guts["max"] * 3, abs=1)


def test_burn_does_not_halve_special():
    # Burn halves PHYSICAL only. Torkoal (Blaze) special user, burned:
    # 97-115 unchanged vs the healthy baseline (Blaze: 115 < 33% of 321 maxHP,
    # so the x1.5 STAB trigger fires either way — clean burn-only comparison).
    dry = dmg({"name": "Torkoal", "nature": "Modest", "evs": {"spa": 252}},
              {"name": "Machamp", "nature": "Calm", "evs": {"spd": 252}}, "flamethrower")
    brn = dmg({"name": "Torkoal", "nature": "Modest", "evs": {"spa": 252}, "status": "brn"},
              {"name": "Machamp", "nature": "Calm", "evs": {"spd": 252}}, "flamethrower")
    assert (brn["min"], brn["max"]) == (dry["min"], dry["max"]) == (97, 115)


def test_aurora_veil_halves_special():
    # Aurora Veil halves special damage for 5 turns (like Light Screen).
    veil = dmg({"name": "Torkoal", "nature": "Modest", "evs": {"spa": 252}},
               {"name": "Machamp", "nature": "Calm", "evs": {"spd": 252}},
               "flamethrower", field={"defenderSide": {"isAuroraVeil": True}})
    assert (veil["min"], veil["max"]) == (48, 57)
    assert veil["max"] == pytest.approx(115 / 2, abs=1)  # 97-115 x0.5


def test_aurora_veil_halves_physical_too():
    # Fork ground truth (moves.js auroraveil.onAnyModifyDamage): the veil halves
    # ALL move categories (no category filter), unlike Reflect/Light Screen.
    # Healthy 852-1004 -> 426-502 with veil (verified in the live sim: 249 of 325
    # at level 50, within the level-50 band). Library matches the fork here.
    veil = dmg(GUTS_MACH, CHANSEY, "closecombat",
               field={"defenderSide": {"isAuroraVeil": True}})
    assert (veil["min"], veil["max"]) == (426, 502)


def test_aurora_veil_stacks_with_burn_special():
    # Burn (no-op on special) + veil (x0.5 special) => 97-115 x0.5 = 48-57.
    brn_veil = dmg({"name": "Torkoal", "nature": "Modest", "evs": {"spa": 252}, "status": "brn"},
                   {"name": "Machamp", "nature": "Calm", "evs": {"spd": 252}},
                   "flamethrower", field={"defenderSide": {"isAuroraVeil": True}})
    assert (brn_veil["min"], brn_veil["max"]) == (48, 57)


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
