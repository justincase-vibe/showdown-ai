"""B1 — the ALL-formats engine-key + reference-damage regression (M1-gate plan).

The approved spec (OPEN_DESIGN_CHOICES, "SMALL ADDITIONS" B1):

  * ``engine_key`` resolves for EVERY format in ``legality.formats`` — a
    new mod or a registry edit that breaks the mapping must fail HERE, not
    silently mid-battle in the calc;
  * calc-vs-sim agreement on reference damages per format: one reference
    damage per format, PINNED. The pins were measured from @smogon/calc
    against this fork's data (the fork = ground truth, D29); the live sim
    cross-check for the primary format is D34 (11 mechanics x 24 seeds).
  * WARN on silent fallback: ``engine_key_or_warn`` (used by the A/B,
    learning-check and strategy-agent call sites) must emit a WARNING
    whenever it falls back — the D32 champions-mis-key bug class must be
    VISIBLE in logs, never hiding in the numbers.

Reference pairs (Dragonite vs Gengar Bold 252 Def — both legal in every
gen; the move is chosen per gen family so the attacker's relevant stat is
the right one):
  * PHYSICAL pair — Dragonite Adamant 252 Atk, ``earthquake``:
    gens 1, 2 and 7-9 (earthquake is SPECIAL in gens 3-6, so it pins 0
    there — exactly the trap this test family-split avoids);
  * SPECIAL pair — Dragonite Modest 252 SpA, ``psychic``:
    gens 3-6.
Pins measured 2026-11-22 (single batch calc each).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from damage import calc  # noqa: E402
from legality import formats  # noqa: E402

PHYS = {"name": "Dragonite", "nature": "Adamant", "evs": {"atk": 252}}
SPEC = {"name": "Dragonite", "nature": "Modest", "evs": {"spa": 252}}
GENG = {"name": "Gengar", "nature": "Bold", "evs": {"def": 252}}

#: (min, max) reference damage per format. Deterministic (the calc is a pure
#: function) — any drift means the engine key, level rule or move data moved.
PINS = {
    # gen 1-2 (physical pair; level 100 ladder / level-5 LC)
    "gen1ou": (243, 286), "gen1uu": (243, 286), "gen1ubers": (243, 286),
    "gen2ou": (243, 286), "gen2uu": (243, 286), "gen2ubers": (243, 286),
    "gen1lc": (23, 28),
    # gen 3-6 (SPECIAL pair — earthquake is special here and would pin 0)
    "gen3ou": (229, 270), "gen3uu": (229, 270), "gen3ru": (229, 270),
    "gen3ubers": (229, 270), "gen3lc": (20, 24),
    "gen4ou": (228, 270), "gen4uu": (228, 270), "gen4ubers": (228, 270),
    "gen4lc": (20, 24), "gen4vgc2010": (120, 142),
    "gen5ou": (228, 270), "gen5uu": (228, 270), "gen5ru": (228, 270),
    "gen5ubers": (228, 270), "gen5lc": (20, 24), "gen5vgc2013": (120, 142),
    "gen6ou": (228, 270), "gen6uu": (228, 270), "gen6ru": (228, 270),
    "gen6ubers": (228, 270), "gen6lc": (20, 24), "gen6vgc2016": (120, 142),
    # gen 7-9 (physical pair)
    "gen7ou": (242, 286), "gen7uu": (242, 286), "gen7ru": (242, 286),
    "gen7ubers": (242, 286), "gen7lc": (22, 28), "gen7vgc2019": (124, 148),
    "gen8ou": (242, 286), "gen8uu": (242, 286), "gen8ru": (242, 286),
    "gen8ubers": (242, 286), "gen8lc": (22, 28), "gen8vgc2022": (124, 148),
    "gen9ou": (242, 286), "gen9uu": (242, 286), "gen9ru": (242, 286),
    "gen9ubers": (242, 286), "gen9lc": (22, 28), "gen9vgc2025regi": (124, 148),
    # champions: the library's native mechanics set (engine key 0, level 50)
    "gen9championsou": (92, 110),
    "gen9championsvgc2026regmc": (92, 110),
    # our primary target: vanilla gen 9 at level 50 (Adjust Level Down 50)
    "gen9bss3v3": (124, 148),
}


def _ref(fmt: str):
    """The (mon, move) reference pair for a format's gen family."""
    gen = formats.by_id(fmt)["gen"]
    if gen in (3, 4, 5, 6):
        return SPEC, "psychic"
    return PHYS, "earthquake"


# ── engine key resolves for EVERY registered format (pure, no node) ───
def test_engine_key_resolves_for_every_format():
    ids = formats.format_ids()
    assert len(ids) >= 49, "the registry must still cover the target formats"
    assert len(ids) == len(set(ids)), "duplicate format ids"
    resolved = {}
    for fmt in ids:
        resolved[fmt] = formats.engine_key(fmt)  # KeyError/ValueError -> FAIL
    # the two mechanics families that D32 exists for are both present
    assert 0 in set(resolved.values()), "champions (engine 0) must be registered"
    assert 9 in set(resolved.values()), "gen-9 formats must be registered"


def test_every_format_has_a_reference_pin():
    # the pin table must cover EXACTLY the registry — a new format without a
    # pin (or a pin for a deleted format) fails here.
    assert set(PINS) == set(formats.format_ids())


# ── reference damage per format (calc via the node pool) ─────────────
@pytest.mark.parametrize("fmt", sorted(PINS))
def test_reference_damage_pinned_per_format(fmt):
    mon, move = _ref(fmt)
    r = calc.calc_damage(mon, GENG, move, fmt=fmt)
    assert ("min" in r), f"calc failed for {fmt}: {r}"
    assert (r["min"], r["max"]) == PINS[fmt], \
        f"{fmt}: expected {PINS[fmt]}, got ({r['min']}, {r['max']})"


# ── the LOUD fallback (B1: no silent engine-key fallbacks) ───────────
def test_engine_key_or_warn_registry_values_unchanged():
    # the helper must be transparent on the registry path (no behaviour drift)
    assert formats.engine_key_or_warn("gen9ou") == 9
    assert formats.engine_key_or_warn("gen9championsou") == 0
    assert formats.engine_key_or_warn("gen1lc") == 1


def test_engine_key_or_warn_fallback_values():
    # same fallback semantics the call sites relied on (regex genN, else 9)
    assert formats.engine_key_or_warn("gen8bss3v3") == 8
    assert formats.engine_key_or_warn("champions") == 9
    assert formats.engine_key_or_warn("totallybogus") == 9
    assert formats.engine_key_or_warn("totallybogus", default=3) == 3


def test_engine_key_or_warn_warns_on_fallback(caplog):
    # THE B1 point: a fallback must be VISIBLE. caplog on the exact logger.
    with caplog.at_level(logging.WARNING, logger="legality.formats"):
        got = formats.engine_key_or_warn("champions")
    assert got == 9
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warns, "fallback must emit a WARNING (no silent mis-keying)"
    assert "champions" in warns[0].getMessage()


def test_registry_path_does_not_warn(caplog):
    # and the normal path stays quiet (no warning spam in real battles)
    with caplog.at_level(logging.WARNING, logger="legality.formats"):
        assert formats.engine_key_or_warn("gen9ou") == 9
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
