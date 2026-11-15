"""Step 2 — T5 state-dependent base-power moves (``bpCallback``).

These are normal base-power moves whose *base power* the fork re-scales from
transient battle state (``basePowerCallback``). @smogon/calc v0.12.0 has no
formula for them — they fall to ``default: basePower = move.bp`` (the static data
value). Consequences:
  * the moves whose data base power is 0 (``beatup``, ``spitup``) returned
    **0 damage** — a hard belief-state bug (the AI would think they deal nothing);
  * the others returned the correct no-state baseline but could not see the
    transient state that doubles / re-scales them in the fork.

The wrapper now COMPLETES the calc (D39): it reads the fork's per-engine-key base
power (``special_bp_moves.json`` — the fork is the source of truth for raw move
data; champions bans the pledges so they are null there), computes the effective
base power from the belief-state flags, and hands it to @smogon/calc via
``overrides.basePower``. The library then applies STAB / typing / boosts /
immunities / crits exactly as for any normal move — one unified damage pipeline.

Ground truth:
  * the base-power *rules* are fork-exact (``scratch/verify_t5_fork.js`` layer A
    calls the fork's own ``basePowerCallback`` and matches our rules, 20/20);
  * the BP -> damage conversion is @smogon/calc (the source of truth for the
    damage model); the pinned ranges below are engine-verified
    (``scratch/probe_t5_calc.js``) and re-confirmed through the Python API.

State flags (all optional; absent = the no-state baseline the fork would use):
  ``hitThisTurn`` (avalanche), ``lastMoveFailed`` (stompingtantrum/temperflare),
  ``pledgeCombo`` (fire/grass/waterpledge), ``allyFainted`` (lastrespects),
  ``stockpileLayers`` (spitup), ``beatupAllies`` (beatup).
"""
from __future__ import annotations

import pytest

from damage import calc

EV = {"atk": 252, "spa": 252}


def user(sp: str) -> dict:
    return {"name": sp, "level": 50, "nature": "Docile", "evs": dict(EV)}


def dmg(att: dict, move, defn: dict, **kw) -> dict:
    return calc.calc_damage(att, defn, move, **kw)


def rng(r: dict) -> tuple:
    return (r["min"], r["max"])


# Defenders (gen 9, level 50, Docile, 252 Atk/SpA — the probe setup).
CHANSEY = {"name": "chansey", "level": 50, "nature": "Docile", "evs": dict(EV)}
HITMONLEE = {"name": "hitmonlee", "level": 50, "nature": "Docile", "evs": dict(EV)}
# Blastoise is legal in BOTH gen 9 and the champions engine (chansey/hitmonlee
# are not champions-legal — the champions table is a 359-species BSS whitelist).
BLASTOISE = {"name": "blastoise", "level": 50, "nature": "Docile", "evs": dict(EV)}


# ── Avalanche (BP 60, x2 if the user was hit this turn) ────────────────
def test_avalanche_baseline():
    assert rng(dmg(user("blastoise"), "avalanche", CHANSEY)) == (122, 144)


def test_avalanche_doubled_when_hit_this_turn():
    r = dmg(user("blastoise"), {"name": "avalanche", "hitThisTurn": True}, CHANSEY)
    assert rng(r) == (243, 287)          # BP 120
    assert r["immune"] is False


# ── Pledges (BP 80, or 150 on a combo) ─────────────────────────────────
def test_firepledge_baseline_and_combo():
    assert rng(dmg(user("charizard"), "firepledge", CHANSEY)) == (58, 70)
    r = dmg(user("charizard"), {"name": "firepledge", "pledgeCombo": True}, CHANSEY)
    assert rng(r) == (109, 130)          # BP 150


def test_grasspledge_baseline_and_combo():
    assert rng(dmg(user("venusaur"), "grasspledge", CHANSEY)) == (55, 66)
    r = dmg(user("venusaur"), {"name": "grasspledge", "pledgeCombo": True}, CHANSEY)
    assert rng(r) == (103, 123)


def test_waterpledge_baseline_and_combo():
    assert rng(dmg(user("blastoise"), "waterpledge", CHANSEY)) == (51, 60)
    r = dmg(user("blastoise"), {"name": "waterpledge", "pledgeCombo": True}, CHANSEY)
    assert rng(r) == (93, 111)


# ── Last Respects (BP 50 + 50 per fainted ally) ────────────────────────
def test_lastrespects_zero_fainted():
    assert rng(dmg(user("houndstone"), "lastrespects", HITMONLEE)) == (60, 72)


def test_lastrespects_scales_with_fainted_allies():
    r = dmg(user("houndstone"), {"name": "lastrespects", "allyFainted": 1}, HITMONLEE)
    assert rng(r) == (118, 141)          # BP 100
    r2 = dmg(user("houndstone"), {"name": "lastrespects", "allyFainted": 2}, HITMONLEE)
    assert rng(r2) == (178, 210)         # BP 150


def test_lastrespects_immune_vs_normal():
    # Ghost move vs Normal: type immunity still applies (library handles it).
    r = dmg(user("houndstone"), "lastrespects", CHANSEY)
    assert r["immune"] is True and rng(r) == (0, 0)


# ── Spit Up (BP 100 per Stockpile layer; 0 layers => the move fails) ───
def test_spitup_fails_without_stockpile():
    r = dmg(user("ekans"), "spitup", CHANSEY)
    assert rng(r) == (0, 0) and r["immune"] is True


def test_spitup_scales_with_stockpile_layers():
    r = dmg(user("ekans"), {"name": "spitup", "stockpileLayers": 1}, CHANSEY)
    assert rng(r) == (28, 34)            # BP 100
    r3 = dmg(user("ekans"), {"name": "spitup", "stockpileLayers": 3}, CHANSEY)
    assert rng(r3) == (84, 99)           # BP 300


# ── Stomping Tantrum / Temper Flare (BP 75, x2 if the previous move failed)
def test_stompingtantrum_baseline_and_failed():
    assert rng(dmg(user("venusaur"), "stompingtantrum", CHANSEY)) == (151, 178)
    r = dmg(user("venusaur"), {"name": "stompingtantrum", "lastMoveFailed": True}, CHANSEY)
    assert rng(r) == (301, 355)          # BP 150


def test_temperflare_baseline_and_failed():
    assert rng(dmg(user("charizard"), "temperflare", CHANSEY)) == (229, 271)
    r = dmg(user("charizard"), {"name": "temperflare", "lastMoveFailed": True}, CHANSEY)
    assert rng(r) == (459, 541)          # BP 150


# ── Beat Up (multi-hit; each ally deals 5 + floor(baseAtk/10) BP, own stats)
def test_beatup_fails_with_no_participants():
    r = dmg(user("mankey"), {"name": "beatup", "beatupAllies": []}, CHANSEY)
    assert rng(r) == (0, 0) and r["immune"] is True


def test_beatup_sums_per_ally_damage():
    allies = [user("mankey"), user("charizard")]  # both base Atk -> BP 13
    r = dmg(user("mankey"), {"name": "beatup", "beatupAllies": allies}, CHANSEY)
    # mankey BP13 = 27-32 ; charizard BP13 = 28-33 ; total 55-65 (engine-verified)
    assert rng(r) == (55, 65)
    assert r["immune"] is False


# ── Champions engine (key 0): the pledges are banned there ─────────────
def test_pledges_banned_in_champions():
    # In the champions engine the pledges are isNonstandard, so they are not
    # usable — the calc must report 0 / immune for them, not crash.
    # (defender must be champions-legal; Blastoise is, and Fire hits it
    # super-effectively, so any leaked damage would show as > 0.)
    for name in ("firepledge", "grasspledge", "waterpledge"):
        r = calc.calc_damage(user("charizard"), BLASTOISE, name, gen=0)
        assert r["max"] == 0, f"{name} should not deal damage in champions"


def test_illegal_species_in_champions_raises_clean_error():
    # Chansey is not in the champions (BSS) species whitelist. The wrapper must
    # fail loudly with a clean error — never an opaque library crash, and never
    # silent numbers for a species that cannot exist in that format.
    with pytest.raises(RuntimeError, match="not legal in the gen-0 engine"):
        calc.calc_damage(user("chansey"), BLASTOISE, "tackle", gen=0)


# ── crit block sanity: a critical hit does not reduce the boosted range ──
def test_state_bp_moves_report_a_crit_range():
    r = dmg(user("venusaur"), {"name": "stompingtantrum", "lastMoveFailed": True}, CHANSEY)
    assert r["crit"]["max"] >= r["max"] > 0
    b = dmg(user("blastoise"), {"name": "avalanche", "hitThisTurn": True}, CHANSEY)
    assert b["crit"]["max"] >= b["max"] > 0
