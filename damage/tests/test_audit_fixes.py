"""Step 2 audit (PLAN.md D42) — regression tests for the three bugs fixed in the
post-T6 code audit.

  * B1 — the special-move immunity gate is now TERA-AWARE (gen 9 + champions):
    the fork's getImmunity uses target.getTypes(), which returns the
    terastallized type when terastallized and not Stellar. Old code checked
    only the BASE types, so e.g. Super Fang wrongly "hit" a Tera-Ghost
    Sandshrew in BSS.
  * B2 — the paralysis speed factor is now per-gen and FORK-EXACT: gens 1-4
    x0.25, gen 5 x0.5 (fork, the battle environment, wins over the library's
    x0.25 — documented divergence), gen 6 x0.25, gens 7-9/champions x0.5.
  * B3 — the speed endpoint is now a full mirror of @smogon/calc
    getFinalSpeed (weather/terrain speed abilities, Iron Ball/EV items,
    Slow Start, Quick Feet, Quark Drive/Protosynthesis, Unburden,
    Quick Powder, gen 1-2 999 cap, pokeRound rounding). ``abilityOn`` /
    ``boostedStat`` mon flags and a top-level ``field`` are the new inputs.

Every value below was cross-verified against the library's own getFinalSpeed
and the fork's Tera-immunity semantics in ``scratch/verify_speed_tera.js``
(40/40 PASS).
"""
from __future__ import annotations

import pytest

from damage import calc

# ── B1: Tera-aware special-move immunity ──────────────────────────────────
def _dmg(att: str, defn: dict, move: str, gen: int) -> dict:
    return calc.calc_damage({"name": att, "level": 50}, defn, move, gen=gen)


def test_gen9_superfang_tera_ghost_sandshrew_immune():
    r = _dmg("Marowak", {"name": "Sandshrew", "level": 50, "teraType": "Ghost"}, "superfang", 9)
    assert r["immune"] is True
    assert (r["min"], r["max"]) == (0, 0)


def test_gen9_superfang_sandshrew_control_hits():
    r = _dmg("Marowak", {"name": "Sandshrew", "level": 50}, "superfang", 9)
    assert r["immune"] is False
    assert (r["min"], r["max"]) == (62, 62)  # half of 125 max HP


def test_gen9_superfang_stellar_ignores_tera():
    # Stellar keeps the base types -> Super Fang (Normal) hits.
    r = _dmg("Marowak", {"name": "Sandshrew", "level": 50, "teraType": "Stellar"}, "superfang", 9)
    assert r["immune"] is False
    assert (r["min"], r["max"]) == (62, 62)


def test_gen1_tera_type_ignored():
    # No Tera in gen 1: teraType must have no effect.
    r = _dmg("Marowak", {"name": "Sandshrew", "level": 50, "teraType": "Ghost"}, "superfang", 1)
    assert r["immune"] is False
    assert r["max"] == 78


def test_tera_type_lowercase_id_still_stabs():
    # B4 — the DB (strategy.tera_type) and the adapter (to_id) pass lowercase
    # Tera ids ('fire'), but the library compares teraType with === against
    # CAPITALIZED names ('Fire'): lowercase silently disabled Tera STAB, Tera
    # Shell and Stellar handling in every belief window (Tera-fire Charizard's
    # Flamethrower computed at 31-37 instead of 42-50 — no STAB). The bridge
    # now normalizes at the boundary (the normItem/normAbility pattern), so
    # id and name both work and agree.
    lower = calc.calc_damage({"name": "Charizard", "level": 50, "teraType": "fire"},
                             {"name": "Garchomp", "level": 50}, "flamethrower", gen=9)
    upper = calc.calc_damage({"name": "Charizard", "level": 50, "teraType": "Fire"},
                             {"name": "Garchomp", "level": 50}, "flamethrower", gen=9)
    plain = calc.calc_damage({"name": "Charizard", "level": 50},
                             {"name": "Garchomp", "level": 50}, "flamethrower", gen=9)
    assert (lower["min"], lower["max"]) == (42, 50)  # STAB applied
    assert (upper["min"], upper["max"]) == (42, 50)  # name form still works
    assert (plain["min"], plain["max"]) == (31, 37)  # no tera -> no STAB


def test_field_screens_and_wonderroom_bridge():
    # B5 — the calc's Field supports per-side screens and Wonder Room, but the
    # bridge dropped isWonderRoom entirely (buildField never forwarded it to
    # the Field constructor) and the belief's calc_field never emitted the
    # side conditions — screened windows were 2x too wide (weak evidence) and
    # Wonder Room battles computed with UNSWAPPED stats (wrong evidence).
    att = {"name": "garchomp", "level": 100, "evs": {"atk": 252}}
    defn = {"name": "blastoise", "level": 100, "evs": {"def": 252}}
    plain = calc.calc_damage(att, defn, "earthquake", gen=9)
    assert (plain["min"], plain["max"]) == (129, 153)
    # Reflect halves physical damage on the defending side
    refl = calc.calc_damage(att, defn, "earthquake", gen=9,
                            field={"defenderSide": {"isReflect": True}})
    assert (refl["min"], refl["max"]) == (64, 76)
    # Light Screen halves special damage
    att2 = {"name": "blastoise", "level": 100, "evs": {"spa": 252}}
    defn2 = {"name": "garchomp", "level": 100, "evs": {"spd": 0}}
    plain2 = calc.calc_damage(att2, defn2, "surf", gen=9)
    ls = calc.calc_damage(att2, defn2, "surf", gen=9,
                          field={"defenderSide": {"isLightScreen": True}})
    assert (plain2["min"], plain2["max"]) == (127, 150)
    assert (ls["min"], ls["max"]) == (63, 75)
    # Wonder Room swaps Atk<->Def for BOTH mons: Garchomp Atk 130 > Def 95 and
    # Blastoise Def 100 > Atk 83 -> the window moves (before the fix it was
    # silently identical to the no-WR window).
    wr = calc.calc_damage(att, defn, "earthquake", gen=9,
                          field={"isWonderRoom": True})
    assert (wr["min"], wr["max"]) == (157, 186)


def test_champions_superfang_tera_ghost_gengar_immune():
    # Champions = engine 0: Tera-aware gate applies here too (BSS target).
    # (Gengar/Farfetchd/Beartic are in the 359-species champions whitelist;
    # Sandshrew and Marowak are NOT — see test below.)
    r = calc.calc_damage(
        {"name": "Farfetchd", "level": 50},
        {"name": "Gengar", "level": 50, "teraType": "Ghost"},
        "superfang", gen=0,
    )
    assert r["immune"] is True
    assert (r["min"], r["max"]) == (0, 0)


def test_champions_superfang_bearctic_control_hits():
    r = calc.calc_damage(
        {"name": "Farfetchd", "level": 50},
        {"name": "Beartic", "level": 50},
        "superfang", gen=0,
    )
    assert r["immune"] is False
    assert (r["min"], r["max"]) == (85, 85)


def test_gen9_fissure_tera_flying_onix_immune():
    r = calc.calc_damage(
        {"name": "Marowak", "level": 50},
        {"name": "Onix", "level": 50, "teraType": "Flying"},
        "fissure", gen=9,
    )
    assert r["immune"] is True
    assert (r["min"], r["max"]) == (0, 0)


def test_gen9_fissure_onix_control():
    r = calc.calc_damage({"name": "Marowak", "level": 50}, {"name": "Onix", "level": 50}, "fissure", gen=9)
    assert r["immune"] is False
    assert (r["min"], r["max"]) == (110, 110)


def test_gen9_fissure_tera_flying_onix_iron_ball_still_hits():
    # Iron Ball grounds a Tera-Flying mon (fork isGrounded checks it first).
    r = calc.calc_damage(
        {"name": "Marowak", "level": 50},
        {"name": "Onix", "level": 50, "teraType": "Flying", "item": "Iron Ball"},
        "fissure", gen=9,
    )
    assert r["immune"] is False
    assert (r["min"], r["max"]) == (110, 110)


def test_gen9_sheercold_tera_ice_gengar_immune():
    r = calc.calc_damage(
        {"name": "Marowak", "level": 50},
        {"name": "Gengar", "level": 50, "teraType": "Ice"},
        "sheercold", gen=9,
    )
    assert r["immune"] is True
    assert (r["min"], r["max"]) == (0, 0)


def test_gen9_sheercold_gengar_control():
    r = calc.calc_damage({"name": "Marowak", "level": 50}, {"name": "Gengar", "level": 50}, "sheercold", gen=9)
    assert r["immune"] is False
    assert (r["min"], r["max"]) == (135, 135)


def test_gen9_fissure_levitate_ignores_tera_ghost():
    # Levitate is an independent Ground check; immune either way.
    r = calc.calc_damage(
        {"name": "Marowak", "level": 50},
        {"name": "Farfetchd", "level": 50, "teraType": "Ghost", "ability": "Levitate"},
        "fissure", gen=9,
    )
    assert r["immune"] is True
    assert (r["min"], r["max"]) == (0, 0)


def test_champions_illegal_species_fails_loudly():
    # Marowak is not in the 359-species champions whitelist: clean error,
    # never a silent number.
    with pytest.raises(RuntimeError, match="not legal in the gen-0 engine"):
        calc.calc_damage(
            {"name": "Marowak", "level": 50},
            {"name": "Gengar", "level": 50},
            "superfang", gen=0,
        )


# ── B2: per-gen paralysis speed factor ────────────────────────────────────
J = {"name": "Jolteon", "level": 50}


@pytest.mark.parametrize(
    "gen,want",
    [
        (9, 75),    # x0.5
        (8, 75),    # x0.5
        (7, 75),    # x0.5
        (6, 37),    # x0.25 (gen6 mod)
        (5, 75),    # FORK x0.5 — the library says 37 (x0.25); fork = environment wins
        (4, 37),    # x0.25 (gen4 mod)
        (3, 37),    # x0.25
        (2, 45),    # x0.25 on the RBY stat model (base 180)
        (1, 45),    # x0.25 on the RBY stat model (base 180)
        (0, 75),    # champions: x0.5
    ],
)
def test_paralysis_speed_per_gen(gen, want):
    assert calc.effective_speed({**J, "status": "par"}, gen=gen) == want


def test_gen5_par_divergence_is_documented():
    # Explicit pin of the ONE fork-vs-library divergence: we follow the fork
    # (x0.5) because it is the battle environment; the library would give 37.
    assert calc.effective_speed({**J, "status": "par"}, gen=5) == 75


# ── B3: full getFinalSpeed mirror ─────────────────────────────────────────
def test_gen9_swift_swim_rain():
    r = calc.effective_speed(
        {"name": "Pelipper", "level": 50, "ability": "Swift Swim"},
        gen=9, field={"weather": "Rain"},
    )
    assert r == 170


def test_gen9_chlorophyll_sun():
    r = calc.effective_speed(
        {"name": "Sceptile", "level": 50, "ability": "Chlorophyll"},
        gen=9, field={"weather": "Sun"},
    )
    assert r == 280


def test_gen9_slush_rush_hail():
    r = calc.effective_speed(
        {"name": "Beartic", "level": 50, "ability": "Slush Rush"},
        gen=9, field={"weather": "Hail"},
    )
    assert r == 140


def test_gen9_surge_surfer_electric_terrain():
    r = calc.effective_speed(
        {"name": "Cramorant", "level": 50, "ability": "Surge Surfer"},
        gen=9, field={"terrain": "Electric"},
    )
    assert r == 210


def test_gen9_iron_ball_halves_speed():
    assert calc.effective_speed({"name": "Joltik", "level": 50, "item": "Iron Ball"}, gen=9) == 42


def test_gen9_slow_start_halves_speed_when_active():
    assert calc.effective_speed(
        {"name": "Machamp", "level": 50, "ability": "Slow Start", "abilityOn": True}, gen=9
    ) == 37


def test_gen9_quick_feet_par_exemption():
    # Quick Feet x1.5 replaces the par x0.5 -> Carnivine 66 -> 99.
    assert calc.effective_speed(
        {"name": "Carnivine", "level": 50, "ability": "Quick Feet", "status": "par"}, gen=9
    ) == 99


def test_gen9_quick_feet_par_with_scarf():
    assert calc.effective_speed(
        {"name": "Carnivine", "level": 50, "ability": "Quick Feet", "status": "par",
         "item": "Choice Scarf"},
        gen=9,
    ) == 148


def test_gen9_quark_drive_boosts_speed():
    assert calc.effective_speed(
        {"name": "Great Tusk", "level": 50, "ability": "Quark Drive", "boostedStat": "spe"}, gen=9
    ) == 160


def test_gen9_quark_drive_auto_no_speed_boost():
    # 'auto' (stat not Spe) leaves speed untouched — library-consistent.
    assert calc.effective_speed(
        {"name": "Great Tusk", "level": 50, "ability": "Quark Drive", "boostedStat": "auto"},
        gen=9, field={"terrain": "Electric"},
    ) == 107


def test_gen9_unburden_doubles_speed_when_active():
    assert calc.effective_speed(
        {"name": "Carnivine", "level": 50, "ability": "Unburden", "abilityOn": True}, gen=9
    ) == 132


def test_gen9_unburden_suppressed_by_scarf():
    # Choice Scarf suppresses Unburden -> only the x1.5 scarf applies.
    assert calc.effective_speed(
        {"name": "Carnivine", "level": 50, "ability": "Unburden", "abilityOn": True,
         "item": "Choice Scarf"},
        gen=9,
    ) == 132


def test_gen9_tailwind_doubles():
    assert calc.effective_speed({"name": "Garchomp"}, gen=9, tailwind=True) == 480


def test_gen9_par_with_scarf():
    assert calc.effective_speed(
        {"name": "Garchomp", "status": "par", "item": "Choice Scarf"}, gen=9
    ) == 180


def test_gen2_boost_table():
    # Gen 1-2 boost numerators ([66, 50, 40, ...]) — library-consistent values.
    assert calc.effective_speed({**J, "boosts": {"spe": 1}}, gen=2) == 271
    assert calc.effective_speed({**J, "boosts": {"spe": -1}}, gen=2) == 119
    assert calc.effective_speed({**J, "boosts": {"spe": 4}}, gen=2) == 543


def test_champions_tailwind():
    # Champions default battle level is 50 (library gen-0 default), not 100.
    assert calc.effective_speed({"name": "Garchomp"}, gen=0, tailwind=True) == 244


def test_champions_chlorophyll_sun():
    assert calc.effective_speed(
        {"name": "Sceptile", "level": 50, "ability": "Chlorophyll"},
        gen=0, field={"weather": "Sun"},
    ) == 280


def test_compare_speed_ordering_and_tie():
    fast = {"name": "Jolteon", "level": 50}
    slow = {"name": "Garchomp", "level": 50}
    r = calc.compare_speed(fast, slow, gen=9)
    assert r["faster"] == "a"
    t = calc.compare_speed(fast, dict(fast), gen=9)
    assert t["faster"] == "tie"
    assert t["a"] == t["b"]
