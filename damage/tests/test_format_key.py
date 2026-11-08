"""Format-keyed calc tests (PLAN.md D32 + D33) — the Step 2 format addendum.

The damage API is keyed by *format* (user: the calc applies to ALL formats,
BSS is only the primary target). Two things the format determines:

1. **Engine key** (D32): the format's ``mod`` — ``genN`` -> N, ``champions``
   -> 0 (the library's native champions mechanics set; a DIFFERENT damage
   model, not gen 9 + BP tweaks).
2. **Battle level** (D33): the format's level rules, mirroring the fork's
   team-validator. BSS 3v3 / VGC / champions battle at level 50, LC at 5,
   ladder at 100. Without ``fmt`` the raw engine path applies (library
   default level 100) — the 34 Test Step 2 tests cover that path.

Reference values (measured from @smogon/calc v0.12.0, Garchomp Adamant
252 Atk vs Tyranitar Bold 252 Def, Earthquake):
  * gen 9:  L100 = 242-288 · L50 = 128-152 · L5 = 24-30
  * champions (engine 0): 120-144 at EVERY level (level is hard-coded 50 in
    the library, pokemon.js:39 — matching the fork's Adjust Level = 50)
Speed (Jolly Garchomp): L100 = 264 · L50 = 134 (neutral L50 = 122).
"""
from __future__ import annotations

import pytest

from damage import calc
from legality import formats

GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}}
TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}


# ── engine-key mapping (pure, no node) ───────────────────────────────
def test_engine_key_mapping():
    assert formats.engine_key("gen1ou") == 1
    assert formats.engine_key("gen5lc") == 5
    assert formats.engine_key("gen9ou") == 9
    assert formats.engine_key("gen9bss3v3") == 9              # our primary target: vanilla gen 9
    assert formats.engine_key("gen9championsou") == 0         # champions mechanics set
    assert formats.engine_key("gen9championsvgc2026regmc") == 0


def test_engine_key_unknown_format_raises():
    with pytest.raises(KeyError):
        formats.engine_key("gen9notarealformat")


# ── BSS: level-50 format, vanilla gen-9 engine ───────────────────────
def test_bss_battles_at_level_50():
    # BSS 3v3 = Adjust Level Down 50: a standard team (no level given) and a
    # level-100 team both battle at 50 → the L50 reference (128-152), NOT the
    # L100 reference (242-288).
    standard = calc.calc_damage(GARCH, TITAN, "earthquake", fmt="gen9bss3v3")
    assert (standard["min"], standard["max"]) == (128, 152)

    explicit100 = calc.calc_damage(
        {**GARCH, "level": 100}, {**TITAN, "level": 100}, "earthquake",
        fmt="gen9bss3v3")
    assert (explicit100["min"], explicit100["max"]) == (128, 152)


def test_raw_gen9_path_unchanged():
    # Without fmt: raw engine semantics, library default level 100 —
    # the Step 2 gate value must keep working (242-288).
    a = calc.calc_damage(GARCH, TITAN, "earthquake", gen=9)
    b = calc.calc_damage(GARCH, TITAN, "earthquake")
    assert (a["min"], a["max"]) == (b["min"], b["max"]) == (242, 288)


# ── other level rules through the calc API ───────────────────────────
def test_vgc_battles_at_level_50():
    r = calc.calc_damage(GARCH, TITAN, "earthquake", fmt="gen9vgc2025regi")
    assert (r["min"], r["max"]) == (128, 152)


def test_lc_battles_at_level_5():
    r = calc.calc_damage(GARCH, TITAN, "earthquake", fmt="gen9lc")
    assert (r["min"], r["max"]) == (24, 30)


def test_illegal_level_for_format_rejected():
    # LC caps level at 5 — the sim's validator would reject a level-100 set,
    # so the calc must not silently invent numbers for it.
    with pytest.raises(ValueError):
        calc.calc_damage({**GARCH, "level": 100}, TITAN, "earthquake", fmt="gen9lc")
    with pytest.raises(ValueError):
        calc.calc_damage({**GARCH, "level": 150}, TITAN, "earthquake", fmt="gen9ou")
    with pytest.raises((KeyError, ValueError)):
        calc.calc_damage(GARCH, TITAN, "earthquake", fmt="notarealformat")


# ── champions: different engine AND fixed level 50 ───────────────────
def test_champions_uses_champions_engine():
    # champions (engine key 0) is a different damage model: 120-144 at EVERY
    # level (library hard-codes 50 — pokemon.js:39 — matching the fork's
    # Adjust Level = 50). Must NOT equal the gen-9 numbers.
    champ = calc.calc_damage(GARCH, TITAN, "earthquake", fmt="gen9championsou")
    assert (champ["min"], champ["max"]) == (120, 144)

    explicit100 = calc.calc_damage(
        {**GARCH, "level": 100}, {**TITAN, "level": 100}, "earthquake",
        fmt="gen9championsou")
    assert (explicit100["min"], explicit100["max"]) == (120, 144)

    gen9 = calc.calc_damage(GARCH, TITAN, "earthquake", gen=9)
    assert (gen9["min"], gen9["max"]) == (242, 288)
    assert champ["max"] != gen9["max"]  # different engine, not a BP tweak


def test_champions_move_data_differs_from_gen9():
    # Slash: BP 80 in champions (fork mods/champions/moves.ts) vs 70 in gen 9 —
    # and the champions engine computes its own numbers. Assert it runs, is
    # positive, and differs from the gen-9 result (proves the key switched).
    champ = calc.calc_damage(GARCH, TITAN, "slash", fmt="gen9championsou")
    gen9 = calc.calc_damage(GARCH, TITAN, "slash", gen=9)
    assert champ["max"] > 0
    assert (champ["min"], champ["max"]) != (gen9["min"], gen9["max"])


# ── speed: level rules apply to the speed helper too ─────────────────
def test_speed_accepts_fmt():
    # BSS battles at level 50 → L50 speed values (134 Jolly / 122 neutral),
    # not the L100 defaults (264 / 240).
    assert calc.effective_speed({"name": "Garchomp", "nature": "Jolly"}, fmt="gen9bss3v3") == 134
    assert calc.effective_speed({"name": "Garchomp"}, fmt="gen9bss3v3") == 122
    # raw path keeps the Step 2 gate values
    assert calc.effective_speed({"name": "Garchomp", "nature": "Jolly"}, gen=9) == 264
    assert calc.effective_speed({"name": "Garchomp"}, gen=9) == 240
