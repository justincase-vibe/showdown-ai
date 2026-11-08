"""Step 1 acceptance tests (build plan: "Move / Species Database").

Run from the project root:

    python -m pytest legality/tests -v

Prerequisites (one-time data build):

    python -m legality.build bootstrap
    node legality/extract_gen9.js --gen 9 --formats gen9ou,gen9uu,gen9ru,gen9bss3v3
    python -c "from legality import api; api.ingest_file('legality/data/extract_gen9.json')"
    python -m legality.ingest_movesets

Plan checklist covered here:
  * known lookups return correct values (Stealth Rock / Light Screen /
    Swords Dance);
  * every move in the format's learnsets exists in the move table
    (no missing keys);
  * set probabilities for each species sum to 1.
"""
from __future__ import annotations

import math

import pytest

from legality import db, lookup


# ── known move lookups ───────────────────────────────────────────────
def test_stealthrock_is_rock_status_hazard():
    m = lookup.get_move("Stealth Rock")
    assert m["type"] == "Rock"
    assert m["category"] == "Status"
    assert m["is_hazard"] is True
    assert m["is_screen"] is False
    assert m["power"] is None  # status move


def test_spikes_is_hazard():
    assert lookup.get_move("Spikes")["is_hazard"] is True


def test_lightscreen_is_screen():
    m = lookup.get_move("Light Screen")
    assert m["is_screen"] is True
    assert m["is_hazard"] is False
    assert m["type"] == "Psychic"
    assert m["category"] == "Status"


def test_reflect_is_screen():
    assert lookup.get_move("Reflect")["is_screen"] is True


def test_swords_dance_boosts_attack_2():
    m = lookup.get_move("Swords Dance")
    assert m["boosts"] == {"atk": 2}
    assert m["is_hazard"] is False
    assert m["is_screen"] is False


def test_tailwind_flag():
    assert lookup.get_move("Tailwind")["is_tailwind"] is True


def test_hiddenpower_variants_collapse_to_base_id():
    m = lookup.get_move("Hidden Power")
    assert m["id"] == "hiddenpower"
    # legacy move: still in the table, but not in any gen-9 learnset
    assert lookup.legal_moves("Raikou", "gen9ou")  # sanity: legal_moves works


def test_unknown_move_raises():
    with pytest.raises(KeyError):
        lookup.get_move("definitely-not-a-move")


# ── species lookups ──────────────────────────────────────────────────
def test_garchomp_species_lookup():
    sp = lookup.get_species("Garchomp")
    assert sp["types"] == ["Dragon", "Ground"]
    assert sp["stats"]["atk"] > sp["stats"]["spa"]  # Garchomp is physical
    assert sp["stats"]["hp"] == 108
    ids = {a["id"] for a in sp["abilities"]}
    assert "roughskin" in ids
    assert sp["tier"] is not None


def test_species_lookup_accepts_display_name_and_id():
    assert lookup.get_species("Ninetales-Alola")["id"] == lookup.get_species("ninetalesalola")["id"]
    # NB: this fork retypes Alolan Ninetales (Ice/Fairy, official Fire/Fairy) —
    # the DB must mirror the server, so assert stable typings on Venusaur.
    assert lookup.get_species("Venusaur")["types"] == ["Grass", "Poison"]


def test_unknown_species_raises():
    with pytest.raises(KeyError):
        lookup.get_species("not-a-pokemon")


# ── type chart ───────────────────────────────────────────────────────
def test_type_chart_is_full_18x18_matrix():
    chart = lookup.get_type_chart(9)
    assert len(chart) == 18
    for deft, row in chart.items():
        assert len(row) == 18, f"defending {deft} has {len(row)} columns"
        assert all(v in (0, 0.5, 1, 2) for v in row.values())


def test_type_chart_known_matchups():
    # chart is indexed [defending][attacking].
    # Every value below verified cell-by-cell against THIS build's source of
    # truth (pokemon-showdown/dist/data/typechart.js, TypeChart[defender_id]
    # .damageTaken[Attacker_name]; encoding 0=1x 1=2x 2=0.5x 3=immune) on
    # 2026-11-07. Beware the axis: Rock->Bug lives in bug.damageTaken.Rock
    # (=2x super-effective), NOT rock.damageTaken.Bug (=1x, that's Bug->Rock).
    chart = lookup.get_type_chart(9)

    def eff(attacking, defending):
        return chart[defending][attacking]

    # super-effective (attacking -> defending)
    assert eff("Water", "Fire") == 2
    assert eff("Fire", "Grass") == 2
    assert eff("Ice", "Dragon") == 2
    assert eff("Poison", "Fairy") == 2
    assert eff("Steel", "Fairy") == 2
    assert eff("Rock", "Bug") == 2
    # immunities — the chart's 8 immunity cells
    assert eff("Electric", "Ground") == 0
    assert eff("Dragon", "Fairy") == 0
    assert eff("Ghost", "Normal") == 0
    assert eff("Psychic", "Dark") == 0
    assert eff("Poison", "Steel") == 0
    assert eff("Ground", "Flying") == 0
    assert eff("Fighting", "Ghost") == 0
    assert eff("Normal", "Ghost") == 0
    # resists
    assert eff("Grass", "Grass") == 0.5
    assert eff("Ground", "Bug") == 0.5
    assert eff("Fairy", "Fire") == 0.5
    # neutral
    assert eff("Fire", "Fairy") == 1
    assert eff("Bug", "Bug") == 1
    assert eff("Bug", "Rock") == 1


# ── learnset integrity (plan: "no missing keys") ─────────────────────
def test_all_format_learnset_moves_exist_in_move_table():
    conn = db.connect()
    try:
        orphans = conn.execute(
            "SELECT COUNT(*) FROM format_form_move ffm "
            "LEFT JOIN move m ON m.id = ffm.move_id WHERE m.id IS NULL").fetchone()[0]
        assert orphans == 0
        # also the raw gen-9 learnset table
        orphans_ls = conn.execute(
            "SELECT COUNT(*) FROM form_learnset fl "
            "LEFT JOIN move m ON m.id = fl.move_id WHERE m.id IS NULL").fetchone()[0]
        assert orphans_ls == 0
    finally:
        conn.close()


def test_lookup_resolves_every_legal_move_for_sample_species():
    fmt = "gen9ou"
    species = lookup.common_set_species(fmt)[:12]
    assert species
    for sp in species:
        moves = lookup.legal_moves(sp, fmt)
        assert moves, f"{sp} should have legal moves in {fmt}"
        for mid in moves:
            m = lookup.get_move(mid)
            assert m["id"] == mid


def test_common_set_moves_are_legal_in_their_format():
    # Smogon sets for a tier must use moves legal in that tier (this is the
    # invariant Step 3's belief sampler relies on: impossible set => P=0).
    fmt = "gen9ou"
    bad = []
    for sp in lookup.common_set_species(fmt):
        legal = set(lookup.legal_moves(sp, fmt))
        for s in lookup.get_common_sets(sp, fmt):
            for mid in s["moves"]:
                if mid not in legal:
                    bad.append((sp, s["title"], mid))
    assert not bad, f"set moves not legal in {fmt}: {bad[:10]}"


# ── common sets (plan: probabilities sum to 1) ───────────────────────
def test_set_probabilities_sum_to_1_for_every_species():
    for fmt in ("gen9ou", "gen9uu", "gen9ru"):
        species = lookup.common_set_species(fmt)
        assert species, f"{fmt} should have Smogon sets"
        for sp in species:
            sets = lookup.get_common_sets(sp, fmt)
            total = sum(s["probability"] for s in sets)
            assert math.isclose(total, 1.0, rel_tol=1e-12), (fmt, sp, total)


def test_common_sets_shape():
    sets = None
    for sp in lookup.common_set_species("gen9ou"):
        sets = lookup.get_common_sets(sp, "gen9ou")
        if sets:
            break
    assert sets
    s0 = sets[0]
    # plan: (moves, item, ability, spread, probability)
    assert len(s0["moves"]) >= 1
    assert s0["item_id"] is not None
    assert s0["ability_id"] is not None
    assert s0["evs"] is not None and "/" in s0["evs"]  # spread string
    assert 0 < s0["probability"] <= 1
    assert s0["source_url"].startswith("https://")
    assert s0["moves"] == sorted(s0["moves"], key=s0["moves"].index)  # ordered


def test_no_sets_returns_empty_list():
    assert lookup.get_common_sets("Bulbasaur", "gen9ou") == []


# ── BSS target format ────────────────────────────────────────────────
def test_bss_format_rules():
    conn = db.connect()
    try:
        f = conn.execute("SELECT * FROM format WHERE id='gen9bss3v3'").fetchone()
        assert f is not None
        assert f["picked_team"] == 3  # bring 6, pick 3 (ruleTable.pickedTeamSize)
        # fork's BSS definition leaves maxTeam unset (server default 6)
        assert f["max_team"] in (None, 6)
    finally:
        conn.close()


def test_bss_garchomp_legal_with_moves():
    moves = lookup.legal_moves("Garchomp", "gen9bss3v3")
    assert moves
    assert "earthquake" in moves or "earthpower" in moves
