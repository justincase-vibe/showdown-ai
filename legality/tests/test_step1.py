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


# ── per-gen move lookups (A1: move_gen overrides, engine key 0-9) ──────
# gen is the @smogon/calc ENGINE KEY (0 = champions, 1-9 = gen). The gen-9
# ``move`` row is the base; a move_gen override (extract_move_gen.js, fork-
# sourced) replaces the per-gen data where the move changed.
def test_per_gen_category_dragonclaw():
    # Dragon Claw is SPECIAL in gen 1-3 (type-based split), PHYSICAL gen 4+.
    assert lookup.get_move("Dragon Claw", 1)["category"] == "Special"
    assert lookup.get_move("Dragon Claw", 2)["category"] == "Special"
    assert lookup.get_move("Dragon Claw", 3)["category"] == "Special"
    assert lookup.get_move("Dragon Claw", 4)["category"] == "Physical"
    assert lookup.get_move("Dragon Claw", 9)["category"] == "Physical"


def test_per_gen_priority_protect():
    # Protect priority: 2 (gen 1-2), 3 (gen 3-4), 4 (gen 5+).
    assert lookup.get_move("Protect", 1)["priority"] == 2
    assert lookup.get_move("Protect", 2)["priority"] == 2
    assert lookup.get_move("Protect", 3)["priority"] == 3
    assert lookup.get_move("Protect", 4)["priority"] == 3
    assert lookup.get_move("Protect", 9)["priority"] == 4


def test_per_gen_power_champions_vs_gen9():
    # Slash is 80 BP in champions (engine key 0), 70 in gen 9 (base).
    assert lookup.get_move("Slash", 0)["power"] == 80
    assert lookup.get_move("Slash", 9)["power"] == 70


def test_per_gen_default_gen9_and_flags_preserved():
    # No-override path: gen 9 (base) values unchanged; effect flags stay on the
    # base row (they are gen-9; the calc models them per-gen via its JSON tables).
    m = lookup.get_move("Dragon Claw", 9)
    assert m["category"] == "Physical"
    assert m["power"] == 80
    assert m["is_hazard"] is False and m["is_screen"] is False
    # an override row still carries the base row's effect flags
    g1 = lookup.get_move("Dragon Claw", 1)
    assert g1["category"] == "Special" and g1["is_hazard"] is False


def test_per_gen_flags_lightscreen_is_not_a_screen_in_gen1():
    # A2 (2026-11-16): in gen 1 Light Screen was a stat-boost move (no side
    # condition), so it is NOT a "screen" that gen — the gen-9 base flag would
    # have misclassified it. Gen 4+ it sets a side condition. (fork-verified)
    g1 = lookup.get_move("Light Screen", 1)
    assert g1["is_screen"] is False and g1["side_condition"] is None
    assert lookup.get_move("Light Screen", 9)["is_screen"] is True
    assert lookup.get_move("Light Screen", 4)["is_screen"] is True


def test_per_gen_flags_boost_magnitude():
    # A2: Tail Glow boosts +3 in gen 5+, +2 in gen 1-4; Amnesia boosts BOTH SpA
    # and SpD in gen 1 (pre type-split) but only SpD in gen 9. (fork-verified)
    assert lookup.get_move("Tail Glow", 9)["boosts"] == {"spa": 3}
    assert lookup.get_move("Tail Glow", 1)["boosts"] == {"spa": 2}
    assert lookup.get_move("Amnesia", 9)["boosts"] == {"spd": 2}
    assert lookup.get_move("Amnesia", 1)["boosts"] == {"spa": 2, "spd": 2}
    # champions (engine key 0) doubles Toxic Thread's speed drop
    assert lookup.get_move("Toxic Thread", 0)["boosts"] == {"spe": -2}
    assert lookup.get_move("Toxic Thread", 9)["boosts"] == {"spe": -1}


def test_per_gen_flags_recoil():
    # A2: recoil changes per gen — Double Edge 1/4 (gen 1) vs 1/3 (gen 9);
    # Struggle 1/2 (gen 1) vs none (gen 9). (fork-verified)
    assert lookup.get_move("Double Edge", 9)["recoil"] == 33
    assert lookup.get_move("Double Edge", 1)["recoil"] == 25
    assert lookup.get_move("Struggle", 9)["recoil"] is None
    assert lookup.get_move("Struggle", 1)["recoil"] == 50


def test_per_gen_flags_gen6_plus_unchanged():
    # A2: gen 6-8 have NO effect-flag diffs vs the gen-9 base (all per-gen flag
    # changes are in gen 1-5 + champions), so the base row is already correct
    # and no override is needed for these.
    assert lookup.get_move("Light Screen", 6)["is_screen"] is True
    assert lookup.get_move("Tail Glow", 7)["boosts"] == {"spa": 3}


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
        # D33: Flat Rules = Adjust Level Down 50 → standard teams battle at 50
        # (maxLevel 100 is the CAP, not the battle level — team-validator.ts:614)
        assert f["min_level"] == 1
        assert f["max_level"] == 100
        assert f["adjust_level_down"] == 50
        assert f["adjust_level"] is None
        assert f["default_level"] == 50  # canonical battle level
    finally:
        conn.close()


def test_format_battle_levels():
    """D33: every format carries its fork-verified level rules (DB v4)."""
    conn = db.connect()
    try:
        rows = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM format")}
        # no format left without level rules (the old silent trap)
        for fid, f in rows.items():
            assert f["default_level"] is not None, fid
            assert f["min_level"] is not None and f["max_level"] is not None, fid
        # spot checks against the fork ruleTable dump (2026-11-08, TeamValidator)
        assert rows["gen9ou"]["default_level"] == 100
        assert rows["gen9championsou"]["default_level"] == 50
        assert rows["gen9championsou"]["adjust_level"] == 50
        assert rows["gen9championsvgc2026regmc"]["default_level"] == 50
        assert rows["gen9vgc2025regi"]["default_level"] == 50
        assert rows["gen9lc"]["default_level"] == 5 and rows["gen9lc"]["max_level"] == 5
        assert rows["gen1lc"]["default_level"] == 5 and rows["gen1lc"]["adjust_level"] == 5
        assert rows["gen4vgc2010"]["default_level"] == 50
        assert rows["gen4vgc2010"]["adjust_level_down"] == 50
    finally:
        conn.close()


def test_battle_level_resolves_like_the_validator():
    """battle_level mirrors team-validator.ts (lines 611–615, 1144)."""
    bl = lookup.battle_level
    # BSS (Adjust Level Down 50): default 50; 100 -> 50; below the threshold stays
    assert bl("gen9bss3v3") == 50
    assert bl("gen9bss3v3", 100) == 50
    assert bl("gen9bss3v3", 50) == 50
    assert bl("gen9bss3v3", 30) == 30
    # champions (Adjust Level = 50, exact): always 50
    assert bl("gen9championsou") == 50
    assert bl("gen9championsou", 100) == 50
    assert bl("gen9championsou", 20) == 50
    # LC gen9 (maxLevel 5): default 5; level 100 is ILLEGAL -> ValueError
    assert bl("gen9lc") == 5
    with pytest.raises(ValueError):
        bl("gen9lc", 100)
    # gen1lc (Adjust Level = 5, exact; maxLevel stays 100): 100 is legal, -> 5
    assert bl("gen1lc", 100) == 5
    # ladder: level passes through
    assert bl("gen9ou", 100) == 100
    assert bl("gen9ou", 50) == 50
    with pytest.raises(ValueError):
        bl("gen9ou", 150)
    # unknown format (api._require_format raises ValueError)
    with pytest.raises(ValueError):
        bl("gen9notarealformat")


def test_format_levels_match_fork():
    """D29 guard: DB level rules must equal the SERVER's live ruleTable dump.

    The registry bakes the fork's values; this test re-derives them from the
    server build's own TeamValidator so the baked data can't drift from the
    sim the bot actually fights on.
    """
    import json
    import os
    import subprocess

    from legality import formats
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sim = os.path.join(root, "pokemon-showdown", "dist", "sim", "index.js")
    script = (
        "const { TeamValidator } = require(process.argv[1]); "
        "const ids = JSON.parse(process.argv[2]); "
        "console.log(JSON.stringify(ids.map(id => { const rt = new TeamValidator(id).ruleTable; "
        "let bl = rt.defaultLevel; "
        "if (rt.adjustLevelDown && bl >= rt.adjustLevelDown) bl = rt.adjustLevelDown; "
        "if (rt.adjustLevel) bl = rt.adjustLevel; "
        "return { id, minLevel: rt.minLevel, maxLevel: rt.maxLevel, "
        "adjustLevel: rt.adjustLevel, adjustLevelDown: rt.adjustLevelDown, bl }; })));"
    )
    proc = subprocess.run(
        ["node", "-e", script, sim, json.dumps(formats.format_ids())],
        capture_output=True, text=True, timeout=60, cwd=root)
    assert proc.returncode == 0, proc.stderr
    fork = {r["id"]: r for r in json.loads(proc.stdout)}

    conn = db.connect()
    try:
        for fid in formats.format_ids():
            row = conn.execute("SELECT * FROM format WHERE id=?", (fid,)).fetchone()
            f = fork[fid]
            assert row["min_level"] == f["minLevel"], fid
            assert row["max_level"] == f["maxLevel"], fid
            assert row["adjust_level"] == (f["adjustLevel"] or None), fid
            assert row["adjust_level_down"] == (f["adjustLevelDown"] or None), fid
            assert row["default_level"] == f["bl"], fid
    finally:
        conn.close()


def test_bss_garchomp_legal_with_moves():
    moves = lookup.legal_moves("Garchomp", "gen9bss3v3")
    assert moves
    assert "earthquake" in moves or "earthpower" in moves
