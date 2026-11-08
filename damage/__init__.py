"""damage — Step 2 damage-calculation module (build plan: "Damage Calculator Wrapper").

A thin, Python-friendly wrapper over the official @smogon/calc damage model (via a
Node subprocess), giving:

    from damage import calc
    calc.calc_damage(attacker, defender, move, field=None, gen=9)
        -> min/max/expected damage (HP + % of max HP) + a separate critical-hit block
    calc.compare_speed(a, b, gen=9)
        -> {'a','b','faster'}  (Choice Scarf / paralysis / boosts / Tailwind aware)

The Node CLI (``node_calc.js``) is the source of truth for the numbers; ``calc.py``
only marshals JSON and exposes the public API.
"""
from . import calc

__all__ = ["calc"]
