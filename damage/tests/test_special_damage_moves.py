"""Step 2 — Special-damage (non-basePower) moves.

@smogon/calc v0.12.0 returns 0 for these moves (they are absent from its
mechanics): they deal damage from a *formula*, not a base power. The wrapper now
COMPLETES the calc with the official Smogon rules (Bulbapedia spec), fork-verified
integer-exact (``scratch/sweep_t1.js``). @smogon/calc stays the source of truth for
every other move; this only fills its documented gaps (D37).

Ground truth for the pinned values is the fork's live sim — the VALIDATION ORACLE
(the calc must match it, because the belief state predicts in-game damage). These
tests pin:
  * the HP-fraction / state formulas (``superfang``/``ruination``/``naturesmadness``/
    ``guardianofalola``/``finalgambit``/``endeavor``) at their fork-verified numbers;
  * ``psywave``'s random range;
  * Super Fang's **gen-aware Ghost immunity** — it is Normal-typed fixed damage, so
    it is immune vs Ghost in gen 2+, but gen 1 ignores Ghost immunity and it hits;
  * the reflectors (``counter``/``mirrorcoat``/``metalburst``/``comeuppance``),
    which scale off the *prior hit* — the belief state passes that as
    ``lastHitDamage``; without it the move fails (0), matching the fork's thisTurn gate.

Every expected range is fork-verified (``scratch/sweep_t1.js``) and re-confirmed
through the Python API (``scratch/probe_t1_fix.py``).
"""
from __future__ import annotations

from damage import calc

# Sweep setup (gen 9: SpA/SpD are independent, so this EV spread is valid).
EV = {"atk": 252, "spa": 252}
CHANSEY = {"name": "chansey", "level": 50, "nature": "Docile", "evs": dict(EV)}
CHANSEY_MAXHP = 325  # fork- and calc-agreed max HP for this setup


def user(sp: str) -> dict:
    """Attacker at the sweep's level/nature/EVs."""
    return {"name": sp, "level": 50, "nature": "Docile", "evs": dict(EV)}


def dmg(att: dict, move, defn: dict = CHANSEY, **kw) -> dict:
    return calc.calc_damage(att, defn, move, **kw)


def rng(result: dict) -> tuple:
    return (result["min"], result["max"])


def mv(name: str, **kw) -> dict:
    d = {"name": name}
    d.update(kw)
    return d


# ── HP-fraction / state formulas (attacker moves) ──────────────────────
def test_superfang_halves_defender_hp():
    r = dmg(user("sandshrew"), "superfang")
    assert rng(r) == (162, 162)            # floor(chansey 325 / 2)
    assert r["immune"] is False
    assert r["maxhp"] == CHANSEY_MAXHP


def test_ruination_halves_defender_hp():
    assert rng(dmg(user("tinglu"), "ruination")) == (162, 162)


def test_naturesmadness_identical_to_ruination():
    # Past move, same formula as Ruination -> same number.
    assert rng(dmg(user("chansey"), "naturesmadness")) == (162, 162)


def test_guardian_of_alola_is_three_quarters():
    assert rng(dmg(user("chansey"), "guardianofalola")) == (243, 243)  # floor(325 * 3/4)


def test_final_gambit_deals_attacker_hp():
    # Damage == attacker's current HP (diglett 85), then it faints.
    assert rng(dmg(user("diglett"), "finalgambit")) == (85, 85)


def test_endeavor_sets_defender_to_attacker_hp():
    # defender 325 - attacker(pikachu) 110 = 215.
    assert rng(dmg(user("pikachu"), "endeavor")) == (215, 215)


def test_endeavor_fails_when_attacker_not_lower_hp():
    # defender at 20% (curHP 65) < attacker full (325) -> max(0, 65-325) = 0, immune.
    low_def = {**CHANSEY, "hp": 0.2}
    r = dmg(user("chansey"), "endeavor", defn=low_def)
    assert rng(r) == (0, 0)
    assert r["immune"] is True


def test_psywave_random_range_gen5():
    # Past (gen 5): random(50,151) * level/100 -> [floor(50*50/100), floor(150*50/100)].
    r = dmg(user("pikachu"), "psywave", gen=5)
    assert rng(r) == (25, 75)


# ── Super Fang gen-aware Ghost immunity ────────────────────────────────
def test_superfang_ghost_immune_gen2plus():
    gengar = {"name": "gengar", "level": 50, "nature": "Docile", "evs": dict(EV)}
    r = dmg(user("sandshrew"), "superfang", defn=gengar)  # gen 9 (default)
    assert rng(r) == (0, 0)
    assert r["immune"] is True


def test_superfang_hits_ghost_in_gen1():
    # gen 1/2 requires SpA == SpD (library constraint); gen 1 ignores Ghost immunity.
    ev12 = {"atk": 252, "spa": 252, "spd": 252}
    att = {"name": "sandshrew", "level": 50, "nature": "Docile", "evs": dict(ev12)}
    gengar = {"name": "gengar", "level": 50, "nature": "Docile", "evs": dict(ev12)}
    r = dmg(att, "superfang", defn=gengar, gen=1)
    assert rng(r) == (83, 83)              # floor(166 / 2) — NOT immune
    assert r["immune"] is False


# ── Reflectors: scale off the prior hit (belief-state `lastHitDamage`) ─
def test_counter_reflects_twice_prior_hit():
    assert rng(dmg(user("hitmonchan"), mv("counter", lastHitDamage=16))) == (32, 32)   # 2 x 16
    assert rng(dmg(user("hitmonchan"), mv("counter", lastHitDamage=1))) == (2, 2)      # 2 x 1
    assert rng(dmg(user("hitmonchan"), mv("counter", lastHitDamage=10))) == (20, 20)   # 2 x 10


def test_mirrorcoat_reflects_twice_prior_hit():
    assert rng(dmg(user("squirtle"), mv("mirrorcoat", lastHitDamage=9))) == (18, 18)  # 2 x 9


def test_metalburst_reflects_one_and_a_half():
    assert rng(dmg(user("bastiodon"), mv("metalburst", lastHitDamage=2))) == (3, 3)    # floor(1.5*2)=3
    assert rng(dmg(user("bastiodon"), mv("metalburst", lastHitDamage=27))) == (40, 40)  # floor(1.5*27)=40
    assert rng(dmg(user("bastiodon"), mv("metalburst", lastHitDamage=1))) == (1, 1)    # clamped to >= 1


def test_comeuppance_reflects_one_and_a_half():
    assert rng(dmg(user("honchkrow"), mv("comeuppance", lastHitDamage=27))) == (40, 40)  # floor(1.5*27)=40


def test_reflector_fails_without_prior_hit():
    # No prior hit observed -> the move fails (0), matching the fork's thisTurn gate.
    for name, att in (("counter", "hitmonchan"), ("mirrorcoat", "squirtle"),
                      ("metalburst", "bastiodon"), ("comeuppance", "honchkrow")):
        r = dmg(user(att), name)
        assert rng(r) == (0, 0), name
        assert r["immune"] is True, name


# ── Fixed/state damage is not crit-affected; normal moves still use the lib ─
def test_fixed_damage_not_crit_affected():
    r = dmg(user("sandshrew"), "superfang")
    assert rng(r["crit"]) == (162, 162) == rng(r)  # crit range == normal range


def test_normal_move_still_routes_through_library():
    # A base-power move is NOT in the special set -> calc library path, non-zero.
    garchomp = {"name": "garchomp", "level": 50, "nature": "Adamant", "evs": {"atk": 252}}
    r = dmg(garchomp, "earthquake")
    assert r["max"] > 0
    assert r["min"] <= r["max"]
