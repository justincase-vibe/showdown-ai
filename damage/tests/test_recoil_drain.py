"""Step 2 — T6: recoil / drain (attacker-side self-damage and healing).

@smogon/calc returns only the damage DEALT (defender side); it does not model
move recoil, drain healing, or Life Orb self-damage. The AI needs those to
evaluate moves (e.g. "Brave Bird KOs but I lose 33% — worth it?"), so the
wrapper COMPLETES the response with fork-exact attacker-side blocks:

  * ``recoil``   — self-damage: ``struggleRecoil`` (25% of user max HP),
    ``mindBlownRecoil``/``chloroblastRecoil`` (50% of user max HP), or a
    fraction of the damage DEALT (Double-Edge, Brave Bird, gen 1-3 Struggle…).
  * ``heal``     — drain fraction of the damage DEALT (Drain Punch, Giga
    Drain, …); net gain is additionally capped by the user's missing HP — the
    belief state applies that cap (it knows current HP).
  * ``lifeOrb``  — fixed 10% of the user's max HP, damaging moves only.
  * ``critChance`` — the fork's BASE critical-hit rate for the move in that
    engine (1/24 gens 7-9/champions, 1/16 gens <= 6, 1/8 for critRatio-2
    moves like Slash). Battle-state modifiers (Super Luck, Scope Lens, Focus
    Energy, Lucky Chant, Merciless, …) are belief-state facts, not calc inputs.

Fork semantics pinned here (verified in dist code + live battles,
``scratch/verify_recoil_drain_fork.js`` = 14/14):

  * recoil/drain are computed on the damage ACTUALLY DEALT, and the fork
    clamps dealt damage to the target's REMAINING HP (overkill deals less).
    The block therefore brackets
    ``[f(min(normalMin, defenderRemainingHP)), f(min(critMax, defenderRemainingHP))]``.
  * TWO recoil formula variants exist in the fork: gens 5-9 and the champions
    engine use ``round`` (base ``applyRecoilDamage``); gens 1-4 inherit the
    gen4 script override which uses ``floor``. Same split for drain
    (``gen<=4`` floor, ``gen>4`` round).
  * ``struggleRecoil`` (gen 4+) = floor/round of user max HP / 4 depending on
    the same split; it is NOT damage-dependent.
  * recoil/drain only happen when the move DEALT damage: miss or immune => no
    block (the fork guards on ``move.totalDamage > 0``).
  * Life Orb = ``baseMaxhp/10`` (all engines in this fork), gated on
    non-Status category (and ``!move.ohko`` in the gen6 mod).

Gen 1-2 note: @smogon/calc uses the authentic RBY DV/stat-EXP stat model and
the fork uses the modern formula; they agree at IV 31 + EV 0 (the level 50
used here), so gen 1-2 tests use zero EVs. (With EVs the models diverge by
~1 — a known gen 1/2 limitation, documented in PLAN.md D41.) The gen 1-2 dex
base stats are the RBY-era values (e.g. Sandshrew Def 85), which BOTH use.
"""
from __future__ import annotations

import math

from damage import calc

EV = {"atk": 252, "spa": 252}
ZERO = {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}


def jround(x: float) -> int:
    """JS ``Math.round`` (half-up) — the fork's rounding, NOT Python's banker's."""
    return int(math.floor(x + 0.5))


def mon(sp: str, evs: dict | None = None, **kw) -> dict:
    d = {"name": sp, "level": 50, "nature": "Docile", "evs": dict(EV if evs is None else evs)}
    d.update(kw)
    return d


def dmg(att: dict, move, defn: dict, **kw) -> dict:
    return calc.calc_damage(att, defn, move, **kw)


# ── champions (engine 0) — KO-clamped recoil ────────────────────────────
def test_champs_bravebird_ko_clamped_recoil():
    # Damage 201-237 (crit 301-355) vs a 154-HP Blastoise: the fork clamps
    # dealt damage to 154 BEFORE computing recoil (live: dealt 154 -> 51,
    # not round(237*33/100) = 78).
    r = dmg(mon("charizard"), "brave bird", mon("blastoise"), gen=0)
    assert (r["min"], r["max"]) == (201, 237)
    assert r["maxhp"] == 154
    assert r["recoil"] == {"min": 51, "max": 51}   # round(154 * 33/100) = 51
    assert r.get("heal") is None and r.get("lifeOrb") is None


def test_champs_recoil_clamped_to_defender_cur_hp():
    # Defender passed at 20 HP (belief state): damage response is still the
    # full-HP numbers, but recoil is computed on the CLAMPED dealt damage.
    r = dmg(mon("charizard"), "brave bird", mon("blastoise", curHP=20), gen=0)
    assert (r["min"], r["max"]) == (201, 237)
    assert r["recoil"] == {"min": 7, "max": 7}     # round(20 * 33/100) = round(6.6) = 7


# ── gen 1-4 FLOOR vs gen 5+ ROUND (fork gen4 script override) ───────────
def test_gen1_struggle_recoil_half_floor():
    # recoil [1,2], gen1-4 formula = floor(dmg/2). Damage 14-17, crit 27-32.
    r = dmg(mon("pikachu", evs=ZERO), "struggle", mon("sandshrew", evs=ZERO), gen=1)
    assert (r["min"], r["max"]) == (14, 17)
    assert (r["crit"]["min"], r["crit"]["max"]) == (27, 32)
    assert r["recoil"] == {"min": 14 // 2, "max": 32 // 2}   # floor formula: (7, 16)
    # The fork's distinguishing live value (29 dealt -> floor(14.5) = 14, where
    # round would give 15) is pinned in test_gen1_live_fork_critical_is_bracketed.


def test_gen1_live_fork_critical_is_bracketed():
    # Live fork battle (gen1randombattle, seed 1): Pikachu Struggle CRIT
    # (gen1 crit = level x2), dealt 29 (in our crit range 27-32), recoil
    # floor(29/2) = 14 (in our recoil block 7-16). The belief-state contract:
    # fork outcomes must fall inside the calc's brackets.
    r = dmg(mon("pikachu", evs=ZERO), "struggle", mon("sandshrew", evs=ZERO), gen=1)
    assert 27 <= 29 <= 32
    assert r["recoil"]["min"] <= 14 <= r["recoil"]["max"]


def test_gen2_struggle_recoil_quarter_floor():
    # gen2/3 Struggle = [1,4] of damage dealt (fork data), floor formula.
    r = dmg(mon("pikachu", evs=ZERO), "struggle", mon("sandshrew", evs=ZERO), gen=2)
    assert (r["min"], r["max"]) == (14, 17)
    assert r["recoil"] == {"min": 14 // 4, "max": 32 // 4}   # floor: (3, 8)


def test_gen1_doubleedge_25pct_vs_gen3_33pct():
    # Double-Edge recoil fraction: 25% in gen 1/2, 33% from gen 3 on — and
    # BOTH use the gen 1-4 floor formula.
    r1 = dmg(mon("pikachu", evs=ZERO), "double edge", mon("sandshrew", evs=ZERO), gen=1)
    assert (r1["min"], r1["max"]) == (28, 33)
    assert r1["recoil"] == {"min": 28 * 25 // 100, "max": 62 * 25 // 100}   # floor: (7, 15)
    r3 = dmg(mon("pikachu", evs=ZERO), "double edge", mon("sandshrew", evs=ZERO), gen=3)
    assert (r3["min"], r3["max"]) == (33, 39)
    assert r3["recoil"] == {"min": 33 // 3, "max": 78 // 3}                 # floor: (11, 26)


# ── drain (heal) — floor gen<=4, round gen>4 ────────────────────────────
def test_gen4_drain_punch_heal_floor_half():
    # machamp Drain Punch vs a 325-HP chansey: both the normal (492-582) and
    # crit (986-1164) spans clamp to 325 dealt -> heal floor(325/2) = 162.
    r = dmg(mon("machamp"), "drain punch", mon("chansey"), gen=4)
    assert (r["min"], r["max"]) == (492, 582)
    assert r["heal"] == {"min": 162, "max": 162}
    assert r.get("recoil") is None


def test_gen9_giga_drain_heal_round_half():
    # gen>4 drain = ROUND (half-up): heal min round(15/2) = 8, max
    # round(27/2) = 14 (Python's round(13.5) would give 14 by luck, but
    # round(7.5) would give 8 only with half-up — pinned via jround below).
    r = dmg(mon("venusaur", hp=0.5), "giga drain", mon("charizard"), gen=9)
    assert (r["min"], r["max"]) == (15, 18)
    assert (r["crit"]["min"], r["crit"]["max"]) == (23, 27)
    assert r["heal"] == {"min": jround(15 / 2), "max": jround(27 / 2)}     # (8, 14)


# ── own-max-HP recoil + Life Orb ─────────────────────────────────────────
def test_gen9_struggle_recoil_is_maxhp_quarter():
    # charizard max HP 153 in this setup -> round(153/4) = 38, exact.
    r = dmg(mon("charizard"), "struggle", mon("blastoise"), gen=9)
    assert r["recoil"] == {"min": jround(153 / 4), "max": jround(153 / 4)} == {"min": 38, "max": 38}


def test_gen9_mind_blown_recoil_is_maxhp_half():
    # round(153/2) = 77, exact (both formula variants round here).
    r = dmg(mon("charizard"), "mind blown", mon("blastoise"), gen=9)
    assert r["recoil"] == {"min": 77, "max": 77}


def test_gen9_life_orb_is_floor_tenth_of_maxhp():
    # floor(153/10) = 15, exact; item id form (fork contract).
    r = dmg(mon("charizard", item="lifeorb"), "tackle", mon("blastoise"), gen=9)
    assert r["lifeOrb"] == {"min": 15, "max": 15}
    assert r.get("recoil") is None and r.get("heal") is None


def test_life_orb_not_applied_to_status_moves():
    # The fork guards on move.category !== "Status"; status moves deal no
    # damage (wrapper: immune) so no lifeOrb block either way.
    r = dmg(mon("charizard", item="lifeorb"), "thunder wave", mon("blastoise"), gen=9)
    assert r.get("lifeOrb") is None
    assert r["immune"] is True


# ── immunity: no recoil, no critChance ──────────────────────────────────
def test_immune_move_has_no_recoil_or_critchance():
    r = dmg(mon("charizard"), "take down", mon("gengar"), gen=9)
    assert r["immune"] is True
    assert r.get("recoil") is None and r.get("heal") is None and r.get("lifeOrb") is None
    assert r.get("critChance") is None


def test_gen1_immune_move_has_no_recoil():
    # Electric vs Ground: fork runImmunity blocks before any formula.
    r = dmg(mon("pikachu", evs=ZERO), "thunder shock", mon("sandshrew", evs=ZERO), gen=1)
    assert r["immune"] is True
    assert r.get("recoil") is None and r.get("critChance") is None


# ── critChance (base fork rate; battle-state modifiers are belief state) ─
def test_crit_chance_base_rate_by_engine():
    # 1/24 in gens 7-9 and champions (engine 0); 1/16 in gens <= 6.
    for gen in (0, 7, 8, 9):
        r = dmg(mon("charizard"), "brave bird", mon("blastoise"), gen=gen)
        assert abs(r["critChance"] - 1 / 24) < 1e-12, gen
    for gen in (5, 6):
        r = dmg(mon("charizard"), "brave bird", mon("blastoise"), gen=gen)
        assert abs(r["critChance"] - 1 / 16) < 1e-12, gen
    # gen 1: base roll = spe/4 per the fork's RBY crit rules -> 1/16 table slot
    r1 = dmg(mon("pikachu", evs=ZERO), "struggle", mon("sandshrew", evs=ZERO), gen=1)
    assert abs(r1["critChance"] - 1 / 16) < 1e-12


def test_crit_chance_slash_is_eighth():
    # critRatio 2 (Slash) -> 1/8 in every engine's table.
    r9 = dmg(mon("charizard"), "slash", mon("blastoise"), gen=9)
    assert r9["critChance"] == 1 / 8
    r0 = dmg(mon("charizard"), "slash", mon("blastoise"), gen=0)
    assert r0["critChance"] == 1 / 8


def test_crit_chance_absent_for_special_damage_moves():
    # The fork returns before the crit roll for special-damage moves.
    r = dmg(mon("chansey"), "seismic toss", mon("charizard"), gen=9)
    assert r.get("critChance") is None
