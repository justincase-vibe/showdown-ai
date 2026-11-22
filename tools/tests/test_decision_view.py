"""B2 — decision_view: hermetic pins (no server, no torch).

The pure parts (log-state parsing + table rendering) are pinned here; the
live smoke (real log + checkpoint + calc) is run manually against a league
replay — see the tool's docstring.

The sample log uses the FORK's native protocol (verified against league
replays, runs/replays/gen9ou.jsonl).

Run:  python -m pytest tools/tests/test_decision_view.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.decision_view import (  # noqa: E402
    build_log_battle, cut_log_at_turn, parse_log_state, render_table,
)

SAMPLE_LOG = """|gen|9
|tier|OU
|start
|teamsize|p1|6
|teamsize|p2|6
|poke|p1|Garchomp, M|
|poke|p1|Sylveon, F|
|poke|p1|Gengar, M|
|poke|p1|Talonflame, M|
|poke|p1|Toxapex, F|
|poke|p1|Scizor, M|
|poke|p2|Blastoise, M|
|poke|p2|Gliscor, M|
|poke|p2|Clefable, F|
|poke|p2|Dragalge, F|
|poke|p2|Hawlucha, M|
|poke|p2|Ting-Lu, F|
|end
|turn|1
|switch|p1a: Garchomp|Garchomp, M|290/290
|switch|p2a: Blastoise|Blastoise, M|288/288
|move|p1a: Garchomp|Earthquake|p2a: Blastoise
|-damage|p2a: Blastoise|144/288
|move|p2a: Blastoise|Surf|p1a: Garchomp
|-damage|p1a: Garchomp|203/290
|turn|2
|move|p1a: Garchomp|Dragon Claw|p2a: Blastoise
|faint|p2a: Blastoise
|switch|p2a: Gliscor|Gliscor, M|244/244
|move|p1a: Garchomp|Earthquake|p2a: Gliscor
|-damage|p2a: Gliscor|97/244
"""


def test_parse_log_state_team_preview_order():
    st = parse_log_state(SAMPLE_LOG, "p1")
    assert st["team"]["p1"] == ["Garchomp", "Sylveon", "Gengar", "Talonflame",
                                "Toxapex", "Scizor"]
    assert st["team"]["p2"][:3] == ["Blastoise", "Gliscor", "Clefable"]


def test_parse_log_state_active_follows_last_switch_in():
    st = parse_log_state(SAMPLE_LOG, "p1")
    assert st["active"]["p1"] == "Garchomp"
    assert st["active"]["p2"] == "Gliscor"          # after Blastoise fainted
    assert st["fainted"]["p2"] == {"Blastoise"}


def test_parse_log_state_revealed_moves_ordered_deduped_ids():
    st = parse_log_state(SAMPLE_LOG, "p1")
    # order of first use, de-duped, Showdown ids (Earthquake used twice -> once)
    assert st["used_moves"]["p1"] == ["earthquake", "dragonclaw"]
    assert st["used_moves"]["p2"] == ["surf"]


def test_build_log_battle_action_surface():
    b = build_log_battle(SAMPLE_LOG, "gen9ou", "p1")
    from net.env import build_action_list
    acts = build_action_list(b)
    # revealed moves (2) each with a tera twin (gen9) + 5 alive bench switches
    kinds = [a["kind"] for a in acts]
    assert kinds.count("move") == 2
    assert kinds.count("tera") == 2
    assert kinds.count("switch") == 5
    # p2: the fainted mon (Blastoise) AND the active (Gliscor) are NOT
    # switch options — 6 - 1 active - 1 fainted = 4 switches
    b2 = build_log_battle(SAMPLE_LOG, "gen9ou", "p2")
    sp = [a.get("species") for a in build_action_list(b2) if a["kind"] == "switch"]
    assert "Blastoise" not in sp
    assert "Gliscor" not in sp
    assert len(sp) == 4


def test_cut_log_at_turn():
    cut = cut_log_at_turn(SAMPLE_LOG, 2)
    # includes everything up to the turn-2 marker, EXcludes turn 2's moves
    assert "|turn|2" in cut
    assert "Dragon Claw" not in cut          # turn-2 move excluded
    assert "Earthquake" in cut               # turn-1 move included
    # a turn beyond the log's end -> unchanged (modulo trailing newline)
    assert cut_log_at_turn(SAMPLE_LOG, 99).rstrip("\n") == SAMPLE_LOG.rstrip("\n")


def test_no_tera_outside_gen9():
    b = build_log_battle(SAMPLE_LOG, "gen8ou", "p1")
    assert b.can_tera is False
    from net.env import build_action_list
    assert all(a["kind"] != "tera" for a in build_action_list(b))


def test_forme_active_matches_base_preview():
    # switch details carry the FORME, the team preview the base name — the
    # active must still resolve to the previewed team member (real case from
    # league replays: preview 'Samurott', switch 'Samurott-Hisui').
    log = ("|start\n"
           "|poke|p1|Samurott, M|\n"
           "|poke|p1|Sylveon, F|\n"
           "|end\n"
           "|switch|p1a: Samurott|Samurott-Hisui, F|361/361\n"
           "|move|p1a: Samurott|Knock Off|p2a: Ting-Lu\n")
    st = parse_log_state(log)
    assert st["active"]["p1"] == "Samurott-Hisui"   # the forme, from details
    b = build_log_battle(log, "gen9ou", "p1")
    assert b.active_pokemon is not None
    assert b.active_pokemon.species == "Samurott"     # matched to the member
    from net.env import build_action_list
    acts = build_action_list(b)
    assert [a["move"] for a in acts if a["kind"] == "move"] == ["knockoff"]


# ── render_table (pure) ────────────────────────────────────────────────
def _asmt(**kw):
    base = {"immune": False, "min_pct": 20.0, "max_pct": 95.0,
            "expected_pct": 55.0, "can_ko": True, "guaranteed_ko": False,
            "expected_ko": True, "ko_bench": []}
    base.update(kw)
    return base


def test_render_table_full():
    acts = [{"kind": "move", "move": "earthquake"},
            {"kind": "tera", "move": "earthquake"},
            {"kind": "switch", "species": "sylveon"}]
    t = render_table(
        acts, prob=[0.6, 0.3, 0.1], value=0.72, entropy=0.45,
        assessments={"earthquake": _asmt(ko_bench=["Gliscor"])})
    assert "0.720" in t and "0.450" in t            # value + entropy header
    assert "60.00" in t                              # 0.6 -> 60.00%
    assert "20.0-95.0%" in t and "possible" in t     # active-enemy KO flags
    assert "Gliscor" in t                            # bench KO list
    assert "sylveon" in t                            # switch row present
    assert t.count("20.0-95.0%") == 2                # tera twin shares the row data


def test_render_table_degraded_and_immune():
    acts = [{"kind": "move", "move": "nightshade"}]
    t = render_table(acts, prob=[1.0],
                     assessments={"nightshade": _asmt(immune=True,
                                                       unknown_defender=True,
                                                       min_pct=0, max_pct=0,
                                                       expected_pct=0,
                                                       can_ko=False,
                                                       guaranteed_ko=False,
                                                       expected_ko=False)})
    assert "unknown species" in t and "IMMUNE" in t


def test_render_table_without_net():
    # no checkpoint -> policy column is '—', no value header
    t = render_table([{"kind": "switch", "species": "gengar"}])
    assert "—" in t and "state value" not in t
