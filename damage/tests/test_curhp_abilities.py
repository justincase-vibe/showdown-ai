"""Step 2 — HP-gated ability effects (current-HP pass-through).

@smogon/calc gates several damage-affecting effects on a mon's CURRENT HP, which
the wrapper used to omit (so the library always assumed FULL HP — low-HP attacker
effects never fired, full-HP-only defender effects always did). The engine now
accepts each mon's HP as a fraction (``hp``, 0.0-1.0) or absolute ``curHP``; these
tests pin every HP-gated effect so the belief state can represent them (D36).

Ground truth is the official @smogon/calc engine (per the user's instruction for
this audit). Every expected range is engine-verified (scratch/probe_curhp.py). Each
effect is asserted three ways: (a) it fires in the right direction vs the full-HP
control, (b) the concrete engine numbers, and (c) id-vs-display-name agreement —
the same name contract that historically silently dropped effects (D34/D35).
"""
from __future__ import annotations

import pytest

from damage import calc


def dmg(att, defn, move, field=None):
    return calc.calc_damage(att, defn, move, field=field)


def rng(result):
    return (result["min"], result["max"])


TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}
MACH = {"name": "Machamp", "nature": "Calm", "evs": {"spd": 252}}
GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}}
TORK = {"name": "Torkoal", "nature": "Modest", "evs": {"spa": 252}}
BLAS = {"name": "Blastoise", "nature": "Modest", "evs": {"spa": 252}}
ROSE = {"name": "Roserade", "nature": "Modest", "evs": {"spa": 252}}


# ── Defeatist: attacker <= 1/2 HP => Atk AND SpA x0.5 ─────────────────
def test_defeatist_halves_physical_below_half_hp():
    full = dmg({**GARCH, "ability": "defeatist"}, TITAN, "earthquake")
    low = dmg({**GARCH, "ability": "defeatist", "hp": 0.5}, TITAN, "earthquake")
    assert rng(full) == (242, 288)
    assert rng(low) == (122, 146)
    assert low["max"] < full["max"] * 0.6  # roughly halved


def test_defeatist_halves_special_below_half_hp():
    # special move -> SpA is halved too (Defeatist hits both offensive stats)
    assert rng(dmg({**TORK, "ability": "defeatist"}, MACH, "flamethrower")) == (97, 115)
    assert rng(dmg({**TORK, "ability": "defeatist", "hp": 0.5}, MACH, "flamethrower")) == (49, 58)


def test_defeatist_inactive_above_half_hp():
    # 0.6 > 1/2 -> no effect (full-HP control)
    assert rng(dmg({**GARCH, "ability": "defeatist", "hp": 0.6}, TITAN, "earthquake")) == (242, 288)


def test_defeatist_id_matches_display_name():
    assert rng(dmg({**GARCH, "ability": "defeatist", "hp": 0.5}, TITAN, "earthquake")) == \
        rng(dmg({**GARCH, "ability": "Defeatist", "hp": 0.5}, TITAN, "earthquake")) == (122, 146)


# ── Overgrow/Blaze/Torrent/Swarm: attacker <= 1/3 HP => matching-type x1.5 ─
def test_blaze_boosts_fire_below_third_hp():
    assert rng(dmg({**TORK, "ability": "blaze"}, MACH, "flamethrower")) == (97, 115)      # full: no trigger
    assert rng(dmg({**TORK, "ability": "blaze", "hp": 0.25}, MACH, "flamethrower")) == (145, 172)
    assert rng(dmg({**TORK, "ability": "blaze", "hp": 0.5}, MACH, "flamethrower")) == (97, 115)  # > 1/3: no effect


def test_overgrow_boosts_grass_below_third_hp():
    assert rng(dmg({**ROSE, "ability": "overgrow"}, TITAN, "gigadrain")) == (264, 312)
    assert rng(dmg({**ROSE, "ability": "overgrow", "hp": 0.25}, TITAN, "gigadrain")) == (392, 464)


def test_torrent_boosts_water_below_third_hp():
    assert rng(dmg({**BLAS, "ability": "torrent"}, TITAN, "surf")) == (242, 288)
    assert rng(dmg({**BLAS, "ability": "torrent", "hp": 0.25}, TITAN, "surf")) == (362, 428)


def test_swarm_boosts_bug_below_third_hp():
    # Garchomp is NOT Bug -> no STAB, so the x1.5 here is Swarm alone (clean ratio).
    full = dmg({**GARCH, "ability": "swarm"}, TITAN, "xscissor")
    low = dmg({**GARCH, "ability": "swarm", "hp": 0.25}, TITAN, "xscissor")
    assert rng(full) == (130, 154)
    assert rng(low) == (194, 230)
    assert low["max"] == pytest.approx(full["max"] * 1.5, abs=1)


def test_lowhp_stab_id_matches_display_name():
    assert rng(dmg({**TORK, "ability": "blaze", "hp": 0.25}, MACH, "flamethrower")) == \
        rng(dmg({**TORK, "ability": "Blaze", "hp": 0.25}, MACH, "flamethrower")) == (145, 172)


# ── Multiscale / Shadow Shield / Tera Shell: defender AT FULL HP only ──
def test_multiscale_halves_at_full_hp_only():
    assert rng(dmg(GARCH, {**TITAN, "ability": "multiscale"}, "earthquake")) == (121, 144)        # full HP: x0.5
    assert rng(dmg(GARCH, {**TITAN, "ability": "multiscale", "hp": 0.9}, "earthquake")) == (242, 288)  # < full: off


def test_shadow_shield_halves_at_full_hp_only():
    assert rng(dmg(GARCH, {**TITAN, "ability": "shadowshield"}, "earthquake")) == (121, 144)
    assert rng(dmg(GARCH, {**TITAN, "ability": "shadowshield", "hp": 0.9}, "earthquake")) == (242, 288)


def test_tera_shell_caps_effectiveness_at_full_hp_only():
    # Tera Shell (gen789) sets type effectiveness to 0.5 at full HP, OVERRIDING the
    # 2x of EQ vs Tyranitar -> reads 0.25x the super-effective baseline. < full HP: off.
    assert rng(dmg(GARCH, {**TITAN, "ability": "terashell"}, "earthquake")) == (60, 72)
    assert rng(dmg(GARCH, {**TITAN, "ability": "terashell", "hp": 0.9}, "earthquake")) == (242, 288)


def test_shadow_shield_id_matches_display_name():
    assert rng(dmg(GARCH, {**TITAN, "ability": "shadowshield"}, "earthquake")) == \
        rng(dmg(GARCH, {**TITAN, "ability": "Shadow Shield"}, "earthquake")) == (121, 144)


# ── Brine: defender <= 1/2 HP => BP x1.5 ──────────────────────────────
def test_brine_boosts_when_defender_low_hp():
    assert rng(dmg(BLAS, TITAN, "brine")) == (176, 210)                      # defender full: base BP 40
    assert rng(dmg(BLAS, {**TITAN, "hp": 0.5}, "brine")) == (350, 414)      # defender <= 1/2: x1.5
    assert rng(dmg(BLAS, {**TITAN, "hp": 0.6}, "brine")) == (176, 210)      # defender > 1/2: no effect


# ── Water Spout / Eruption: BP scales with attacker HP ────────────────
def test_water_spout_scales_with_attacker_hp():
    full = dmg(BLAS, TITAN, "waterspout")
    low = dmg({**BLAS, "hp": 0.5}, TITAN, "waterspout")
    assert rng(full) == (404, 476)
    assert rng(low) == (200, 236)
    assert low["max"] == pytest.approx(full["max"] / 2, abs=3)  # BP 150 -> ~75 (floor drift)


def test_eruption_scales_with_attacker_hp():
    assert rng(dmg(TORK, TITAN, "eruption")) == (101, 119)
    assert rng(dmg({**TORK, "hp": 0.5}, TITAN, "eruption")) == (50, 59)


# ── Flail / Reversal: BP inverse-scales with attacker HP ──────────────
def test_flail_increases_as_attacker_hp_drops():
    assert rng(dmg(GARCH, TITAN, "flail")) == (8, 10)          # full HP: weakest
    assert rng(dmg({**GARCH, "hp": 0.1}, TITAN, "flail")) == (60, 71)  # low HP: much stronger


def test_reversal_increases_as_attacker_hp_drops():
    assert rng(dmg(GARCH, TITAN, "reversal")) == (68, 80)
    assert rng(dmg({**GARCH, "hp": 0.1}, TITAN, "reversal")) == (484, 572)


# ── input handling & validation ───────────────────────────────────────
def test_curhp_absolute_alias_works():
    # curHP is clamped to [1, maxHP]. A huge value == full HP (no Defeatist trigger).
    assert rng(dmg({**GARCH, "ability": "defeatist", "curHP": 10**9}, TITAN, "earthquake")) == (242, 288)
    # curHP = 1 is <= maxHP/2 -> Defeatist fires (same low-HP value as hp=0.5).
    assert rng(dmg({**GARCH, "ability": "defeatist", "curHP": 1}, TITAN, "earthquake")) == (122, 146)


def test_hp_out_of_range_raises():
    with pytest.raises(RuntimeError):
        calc.calc_damage({**GARCH, "ability": "defeatist", "hp": 1.5}, TITAN, "earthquake")
    with pytest.raises(RuntimeError):
        calc.calc_damage({**GARCH, "ability": "defeatist", "hp": -0.1}, TITAN, "earthquake")


def test_no_hp_keeps_full_hp_semantics():
    # Omitting hp entirely is unchanged: full HP, so no low-HP ability fires.
    assert rng(dmg({**GARCH, "ability": "defeatist"}, TITAN, "earthquake")) == (242, 288)
