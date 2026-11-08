"""Format -> engine-key mapping tests (PLAN.md D32; addendum to the Step 2 gate).

The damage API must be keyable by *format* (user: the calc applies to ALL
formats, BSS is only the primary target). The engine key comes from the
format's ``mod``, not its ``gen`` column:

* ``genN`` mods  -> N  (standard @smogon/calc mechanics);
* ``champions``  -> 0  (the library's native champions mechanics set — a
                     DIFFERENT damage model, verified: level-independent and
                     not gen-9 numbers; see assertions below).

Reference values (measured from @smogon/calc v0.12.0, Garchomp Adamant
252 Atk vs Tyranitar Bold 252 Def):

* gen 9 (vanilla, our BSS format):  EQ = 242-288 @L100 (the Step 2 gate value)
* champions (engine key 0):         EQ = 120-144 @L100 AND @L50 (level-independent)
* champions Slash BP is 80 vs gen-9 70 (fork ``data/mods/champions/moves.ts``),
  but because the whole engine differs, do NOT assert a BP ratio — assert the
  engine's own verified numbers instead.
"""
from __future__ import annotations

import pytest

from damage import calc
from legality import formats

GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}}
TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}


# ── the mapping itself (pure, no node) ───────────────────────────────
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


# ── behaviour through the calc API ───────────────────────────────────
def test_bss_numbers_identical_to_gen9():
    # BSS 3v3 is vanilla gen 9: fmt= must be bit-identical to the raw key,
    # and must still reproduce the Step 2 gate value (242-288).
    a = calc.calc_damage(GARCH, TITAN, "earthquake", fmt="gen9bss3v3")
    b = calc.calc_damage(GARCH, TITAN, "earthquake", gen=9)
    assert (a["min"], a["max"]) == (b["min"], b["max"]) == (242, 288)


def test_champions_uses_champions_engine():
    # champions format (key 0) must NOT produce gen-9 numbers: its damage
    # model is level-independent (120-144 at L100 and L50, verified).
    champ = calc.calc_damage(GARCH, TITAN, "earthquake", fmt="gen9championsou")
    assert (champ["min"], champ["max"]) == (120, 144)

    gen9 = calc.calc_damage(GARCH, TITAN, "earthquake", gen=9)
    assert (gen9["min"], gen9["max"]) == (242, 288)
    assert champ["max"] < gen9["max"]  # different engine, not a BP tweak


def test_champions_move_data_differs_from_gen9():
    # Slash: BP 80 in champions (fork mods/champions/moves.ts) vs 70 in gen 9.
    # The champions engine computes its own numbers — assert it runs and is
    # positive, and that it is NOT the gen-9 result (proves the key switched).
    champ = calc.calc_damage(GARCH, TITAN, "slash", fmt="gen9championsou")
    gen9 = calc.calc_damage(GARCH, TITAN, "slash", gen=9)
    assert champ["max"] > 0
    assert (champ["min"], champ["max"]) != (gen9["min"], gen9["max"])


def test_calc_rejects_unknown_format():
    with pytest.raises((KeyError, ValueError)):
        calc.calc_damage(GARCH, TITAN, "earthquake", fmt="notarealformat")


def test_speed_accepts_fmt():
    # gen-9 speed model via the format key (Jolly 264 / neutral 240 gate values)
    assert calc.effective_speed({"name": "Garchomp", "nature": "Jolly"}, fmt="gen9bss3v3") == 264
    assert calc.effective_speed({"name": "Garchomp"}, fmt="gen9bss3v3") == 240
