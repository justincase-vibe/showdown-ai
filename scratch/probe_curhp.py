"""Scratch probe: read off engine-verified ranges for HP-gated ability effects.

Run from the project root:
    python scratch/probe_curhp.py
"""
from damage import calc


def rng(att, defn, move, field=None):
    r = calc.calc_damage(att, defn, move, field=field)
    return (r["min"], r["max"])


TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}
MACH = {"name": "Machamp", "nature": "Calm", "evs": {"spd": 252}}
GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}}
TORK = {"name": "Torkoal", "nature": "Modest", "evs": {"spa": 252}}
BLAS = {"name": "Blastoise", "nature": "Modest", "evs": {"spa": 252}}
ROSE = {"name": "Roserade", "nature": "Modest", "evs": {"spa": 252}}

print("=== Defeatist (attacker <=1/2 -> Atk & SpA x0.5) ===")
print("phys  full :", rng({**GARCH, "ability": "defeatist"}, TITAN, "earthquake"))
print("phys  hp.5 :", rng({**GARCH, "ability": "defeatist", "hp": 0.5}, TITAN, "earthquake"))
print("phys  hp.6 :", rng({**GARCH, "ability": "defeatist", "hp": 0.6}, TITAN, "earthquake"))
print("spec  full :", rng({**TORK, "ability": "defeatist"}, MACH, "flamethrower"))
print("spec  hp.5 :", rng({**TORK, "ability": "defeatist", "hp": 0.5}, MACH, "flamethrower"))
print("id=name  hp.5:", rng({**GARCH, "ability": "defeatist", "hp": 0.5}, TITAN, "earthquake"),
      "==", rng({**GARCH, "ability": "Defeatist", "hp": 0.5}, TITAN, "earthquake"))

print("\n=== Overgrow/Blaze/Torrent/Swarm (attacker <=1/3 -> matching type x1.5) ===")
print("Blaze   full:", rng({**TORK, "ability": "blaze"}, MACH, "flamethrower"))
print("Blaze   hp.25:", rng({**TORK, "ability": "blaze", "hp": 0.25}, MACH, "flamethrower"))
print("Blaze   hp.5 :", rng({**TORK, "ability": "blaze", "hp": 0.5}, MACH, "flamethrower"))
print("Overgrow full:", rng({**ROSE, "ability": "overgrow"}, TITAN, "gigadrain"))
print("Overgrow hp.25:", rng({**ROSE, "ability": "overgrow", "hp": 0.25}, TITAN, "gigadrain"))
print("Torrent  full:", rng({**BLAS, "ability": "torrent"}, TITAN, "surf"))
print("Torrent  hp.25:", rng({**BLAS, "ability": "torrent", "hp": 0.25}, TITAN, "surf"))
print("Swarm    full:", rng({**GARCH, "ability": "swarm"}, TITAN, "xscissor"))
print("Swarm    hp.25:", rng({**GARCH, "ability": "swarm", "hp": 0.25}, TITAN, "xscissor"))
print("Blaze id=name hp.25:", rng({**TORK, "ability": "blaze", "hp": 0.25}, MACH, "flamethrower"),
      "==", rng({**TORK, "ability": "Blaze", "hp": 0.25}, MACH, "flamethrower"))

print("\n=== Multiscale/Shadow Shield/Tera Shell (defender AT FULL HP only -> x0.5) ===")
for ab in ("multiscale", "shadowshield", "terashell"):
    print(f"{ab:14} full:", rng(GARCH, {**TITAN, "ability": ab}, "earthquake"))
    print(f"{ab:14} hp.9:", rng(GARCH, {**TITAN, "ability": ab, "hp": 0.9}, "earthquake"))
print("ShadowShield id=name full:", rng(GARCH, {**TITAN, "ability": "shadowshield"}, "earthquake"),
      "==", rng(GARCH, {**TITAN, "ability": "Shadow Shield"}, "earthquake"))

print("\n=== Brine (defender <=1/2 -> BP x1.5) ===")
print("Brine defn full:", rng(BLAS, TITAN, "brine"))
print("Brine defn hp.5:", rng(BLAS, {**TITAN, "hp": 0.5}, "brine"))
print("Brine defn hp.6:", rng(BLAS, {**TITAN, "hp": 0.6}, "brine"))

print("\n=== Water Spout / Eruption (BP scales with attacker HP) ===")
print("WaterSpout full :", rng(BLAS, TITAN, "waterspout"))
print("WaterSpout hp.5 :", rng({**BLAS, "hp": 0.5}, TITAN, "waterspout"))
print("Eruption  full  :", rng(TORK, TITAN, "eruption"))
print("Eruption  hp.5  :", rng({**TORK, "hp": 0.5}, TITAN, "eruption"))

print("\n=== Flail / Reversal (BP inverse-scales with attacker HP) ===")
print("Flail    full :", rng(GARCH, TITAN, "flail"))
print("Flail    hp.1 :", rng({**GARCH, "hp": 0.1}, TITAN, "flail"))
print("Reversal full :", rng(GARCH, TITAN, "reversal"))
print("Reversal hp.1 :", rng({**GARCH, "hp": 0.1}, TITAN, "reversal"))
