"""Production regression: @smogon/calc Field.gameType must be 'Doubles' for doubles formats.

Root cause (B1 finding, 2026-11-24):
    The calc library defaults ``Field.gameType`` to 'Singles' (field.js:23),
    selecting the singles screen ratio (0.5). The official doubles rule is a 2/3
    reduction (2732/4096) that activates only when ``gameType != 'Singles'``
    (gen789.js:1263, champions.js:735).

    Without the fix, doubles Reflect/Light Screen/Aurora Veil under-screen by
    a factor of 4/3: the AI's belief state would compute 50 % screen reduction
    when the real rule is 33 %.

    The fix is in ``damage/calc.py`` — ``_game_type(fmt)`` and
    ``_with_game_type(field, fmt)`` inject the correct ``gameType`` at the
    single point where the field dict is assembled for the Node worker, so
    every call path (calc_damage, calc_damage_many, batch, evidence.py)
    picks it up automatically when ``fmt`` is passed.

What this file verifies (production, not just B1):
    1. Reflect (physical, single-target) in doubles is exactly 4/3 of singles.
    2. Light Screen (special, single-target) in doubles is exactly 4/3 of singles.
    3. A NON-screen, SINGLE-TARGET calc is identical in singles and doubles
       (gameType has no side-effect on non-screen, non-spread damage).
    4. Spread moves (Earthquake) in doubles get the ×0.75 (3072/4096) multiplier
       (correct game mechanic — verifies the library applies it when gameType=Doubles).
    5. Unit tests for ``_game_type`` / ``_with_game_type`` (no Node needed).

NOTE: ``earthquake`` is a SPREAD move in doubles (target='allAdjacent') and gets
a ×0.75 damage reduction (gen789.js:1224-1227). Screen tests therefore use
``dragonclaw`` (single-target, Dragon STAB vs Rock/Dark = 2x super-effective).

Run from the project root:

    python -m pytest damage/tests/test_game_type_doubles.py -v
"""
from __future__ import annotations

import pytest

from damage import calc


# ── shared mon specs ─────────────────────────────────────────────────
# level=50 is set EXPLICITLY so both singles (gen9ou, default L100) and doubles
# (gen9vgc2025regi, default L50) use the same level. Without this, the level
# difference masks the gameType effect we are testing.
GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}, "level": 50}
TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}, "level": 50}
CHARZ = {"name": "Charizard", "nature": "Modest", "evs": {"spa": 252}, "level": 50}
LUC   = {"name": "Lucario",  "nature": "Modest", "evs": {"spa": 252}, "level": 50}

# ── valid format ids from the registry ────────────────────────────────
SINGLES_FMT = "gen9ou"            # game_type = "singles"
DOUBLES_FMT = "gen9vgc2025regi"   # game_type = "doubles"


# ── helper ───────────────────────────────────────────────────────────
def _dmg(att, dfn, move, field=None, fmt=None):
    return calc.calc_damage(att, dfn, move, field=field, fmt=fmt)


# ── 1. Reflect (single-target move): doubles is 4/3 of singles ──────
def test_reflect_doubles_is_4_3_of_singles():
    """With Reflect active, single-target physical damage in doubles is (2/3) of
    base; in singles it is (1/2) of base.  Ratio = (2/3)/(1/2) = 4/3.

    Uses dragonclaw (single-target, no spread multiplier) to isolate the
    screen ratio from the doubles spread-move penalty."""
    singles = _dmg(GARCH, TITAN, "dragonclaw",
                   field={"defenderSide": {"isReflect": True}},
                   fmt=SINGLES_FMT)
    doubles = _dmg(GARCH, TITAN, "dragonclaw",
                   field={"defenderSide": {"isReflect": True}},
                   fmt=DOUBLES_FMT)

    assert singles["max"] > 0, "singles Reflect should still deal damage"
    assert doubles["max"] > 0, "doubles Reflect should still deal damage"

    expected_doubles_max = singles["max"] * 4 / 3
    assert doubles["max"] == pytest.approx(expected_doubles_max, abs=1), (
        f"Reflect singles/doubles ratio wrong: "
        f"singles_max={singles['max']} doubles_max={doubles['max']} "
        f"expected_doubles≈{expected_doubles_max:.1f}"
    )

    expected_doubles_min = singles["min"] * 4 / 3
    assert doubles["min"] == pytest.approx(expected_doubles_min, abs=1), (
        f"Reflect singles/doubles min ratio wrong: "
        f"singles_min={singles['min']} doubles_min={doubles['min']}"
    )


# ── 2. Light Screen (single-target move): doubles is 4/3 of singles ─
def test_light_screen_doubles_is_4_3_of_singles():
    """Same rule for special moves under Light Screen (single-target move)."""
    singles = _dmg(CHARZ, TITAN, "flamethrower",
                   field={"defenderSide": {"isLightScreen": True}},
                   fmt=SINGLES_FMT)
    doubles = _dmg(CHARZ, TITAN, "flamethrower",
                   field={"defenderSide": {"isLightScreen": True}},
                   fmt=DOUBLES_FMT)

    assert singles["max"] > 0
    assert doubles["max"] > 0

    expected_doubles_max = singles["max"] * 4 / 3
    assert doubles["max"] == pytest.approx(expected_doubles_max, abs=1), (
        f"Light Screen singles/doubles ratio wrong: "
        f"singles_max={singles['max']} doubles_max={doubles['max']}"
    )


# ── 3. Non-screen, single-target: gameType has NO side effect ────────
def test_non_screen_single_target_unchanged_by_game_type():
    """A normal single-target damage calc (no screens, no spread) should be
    identical in singles and doubles — gameType only affects the screen
    multiplier and the spread-move penalty, nothing else."""
    singles = _dmg(GARCH, TITAN, "dragonclaw", fmt=SINGLES_FMT)
    doubles = _dmg(GARCH, TITAN, "dragonclaw", fmt=DOUBLES_FMT)

    assert singles["max"] > 0
    assert doubles["max"] > 0

    assert singles["max"] == doubles["max"], (
        f"Non-screen single-target calc should be identical: "
        f"singles={singles['max']} doubles={doubles['max']}"
    )
    assert singles["min"] == doubles["min"]


# ── 4. Spread move in doubles: ×0.75 (3072/4096) is applied ──────────
def test_spread_move_doubles_gets_075_multiplier():
    """Earthquake targets 'allAdjacent' in doubles and gets ×0.75 (gen789.js:1227).
    In singles it targets one mon and gets no spread penalty.
    This verifies gameType=Doubles is correctly reaching the spread logic."""
    singles = _dmg(GARCH, TITAN, "earthquake", fmt=SINGLES_FMT)
    doubles = _dmg(GARCH, TITAN, "earthquake", fmt=DOUBLES_FMT)

    assert singles["max"] > 0
    assert doubles["max"] > 0

    # In doubles: earthquake gets ×0.75 (3072/4096 ≈ 0.75)
    expected_doubles = singles["max"] * 3072 / 4096
    assert doubles["max"] == pytest.approx(expected_doubles, abs=1), (
        f"Spread move doubles/singles ratio wrong: "
        f"singles_max={singles['max']} doubles_max={doubles['max']} "
        f"expected_doubles≈{expected_doubles:.1f}"
    )


# ── 5. Unit tests: _game_type / _with_game_type (no Node needed) ─────
def test_game_type_helper_mappings():
    assert calc._game_type(SINGLES_FMT) == "Singles"
    assert calc._game_type(DOUBLES_FMT) == "Doubles"
    assert calc._game_type(None) is None
    assert calc._game_type("nonexistent_format_xyz") is None


def test_with_game_type_injects_field():
    out = calc._with_game_type({"defenderSide": {"isReflect": True}}, DOUBLES_FMT)
    assert out is not None
    assert out.get("gameType") == "Doubles"
    assert out["defenderSide"] == {"isReflect": True}

    out = calc._with_game_type({"defenderSide": {"isReflect": True}}, SINGLES_FMT)
    assert out is not None
    assert out.get("gameType") == "Singles"

    out = calc._with_game_type(None, DOUBLES_FMT)
    assert out is not None
    assert out.get("gameType") == "Doubles"

    out = calc._with_game_type(None, None)
    assert out is None
