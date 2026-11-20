"""C3 — the public calc API is routed through the worker pool, with the one-shot
spawn as the fallback. Pinned:
  * ``calc_damage`` / ``compare_speed`` / ``calc_damage_many`` return correct
    results (now via the pool);
  * a calc ERROR (illegal species) still raises ``RuntimeError`` (the one-shot
    contract, unchanged);
  * with ``POKE_CALC_ONESHOT`` set, the SAME public API still works — the
    one-shot fallback — so a disabled/broken pool never breaks a calc.

Run from the project root:

    python -m pytest damage/tests/test_pool_routing.py -v
"""
from __future__ import annotations

import pytest

from damage import calc

GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}}
TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}


def test_public_api_via_pool():
    d = calc.calc_damage(GARCH, TITAN, "earthquake")
    assert d["min"] > 0 and d["max"] >= d["min"]
    assert "crit" in d and "maxhp" in d and "immune" in d

    s = calc.compare_speed(GARCH, TITAN)
    assert s["faster"] in ("a", "b", "tie")

    m = calc.calc_damage_many([
        {"attacker": GARCH, "defender": TITAN, "move": "earthquake"},
        {"attacker": TITAN, "defender": GARCH, "move": "crunch"},
    ])
    assert len(m) == 2 and all("min" in x and "max" in x for x in m)


def test_calc_error_still_raises():
    bad = {"name": "zzz_not_real"}
    with pytest.raises(RuntimeError):
        calc.calc_damage(bad, TITAN, "tackle")


def test_fallback_oneshot_when_pool_disabled(monkeypatch):
    """POKE_CALC_ONESHOT makes worker.request raise WorkerError, so calc must
    fall back to the one-shot spawn and STILL return the correct number."""
    monkeypatch.setenv("POKE_CALC_ONESHOT", "1")
    d = calc.calc_damage(GARCH, TITAN, "earthquake")
    ref = calc._node_oneshot({
        "type": "damage", "gen": 9, "attacker": GARCH, "defender": TITAN,
        "move": {"name": "earthquake"}, "field": None})
    assert d["min"] == ref["min"] and d["max"] == ref["max"], (d, ref)
    # and a calc error still raises through the fallback
    with pytest.raises(RuntimeError):
        calc.calc_damage({"name": "zzz_not_real"}, TITAN, "tackle")
