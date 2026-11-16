"""Step 3.1 gate (belief state + event model + prior + log→event adapter).

Run from the project root:

    python -m pytest belief/tests -v

This locks in the Step 3.1 deliverable (PLAN.md §3, sub-step 3.1):
  * ``build_prior`` — a per-slot candidate list whose weights sum to exactly 1:
      - a format's own Smogon sets (gen9ou Garchomp = 2 sets, 0.5/0.5);
      - a merged same-gen tier prior when the format carries no sets of its
        own (BSS has no sets of its own → tiers gen9ou/uu/ru merged);
      - a bounded legal-move fallback for legal species with no sets anywhere;
      - a loud ``ValueError`` for a species with neither (no empty belief).
  * ``BeliefState`` / ``SlotBelief`` — opponent-side (p2*) slots only, posterior
    weights sum to 1, candidate dims all present and positive.
  * the log→event adapter (``belief/log_events.py``) — hand-written protocol
    lines map to the exact event sequence, the sim's doubled hp lines are
    deduped (incl. the maxhp==100 and ``0 fnt`` KO edges), noise is ignored,
    genuinely-unmapped lines surface as ``Unknown``, and the ``[from]
    ability:/item:`` context reveals are attributed to the ``[of]`` holder slot.
  * a SMOKE test that runs one real seeded fork battle and asserts the event
    stream is sane (>=1 start, >=2 p2 switch-ins, >=2 move uses, >=1 damage
    with a real denominator, zero Unknown) — the "one live battle" evidence
    the plan asks for before the evidence engine (3.2) is built.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from belief import events as E
from belief.log_events import LogAdapter, events_from_log
from belief.state import BeliefState, SlotBelief, build_prior

ROOT = Path(__file__).resolve().parents[2]  # belief/tests/x.py -> project root

# A legal BSS species that has moves but NO Smogon sets anywhere in gen 9
# (the genuine fallback case). The "no prior at all" path (test_no_prior_raises)
# is exercised by a CAP (colossoil — genuinely nonstandard: no sets AND no legal
# moves), NOT by a regular species: after the Step-1 learnset fix, formerly-
# absent legal species (abra, machamp, …) now get a legal-move fallback prior
# instead of raising.
FALLBACK_SPECIES = "ampharos"
FALLBACK_ABILITIES = ["plus", "static"]


# ═══════════════════════════ prior / belief state ═══════════════════════════

def test_prior_format_own_sets_sum_to_one():
    cands = build_prior("garchomp", "gen9ou")
    assert len(cands) == 2
    assert sum(c["weight"] for c in cands) == pytest.approx(1.0, abs=1e-9)
    assert all(c["weight"] > 0 for c in cands)


def test_prior_bss_merged_from_tiers():
    # BSS carries no sets of its own — the prior is built from the gen-9 tiers.
    cands = build_prior("garchomp", "gen9bss3v3")
    assert len(cands) >= 1
    assert sum(c["weight"] for c in cands) == pytest.approx(1.0, abs=1e-9)
    # every candidate is a full set (moves + ability), not a per-move marginal
    assert all(len(c["moves"]) > 0 and c["ability"] for c in cands)


def test_prior_fallback_for_legal_species_without_sets():
    cands = build_prior(FALLBACK_SPECIES, "gen9bss3v3")
    assert len(cands) >= 1
    assert sum(c["weight"] for c in cands) == pytest.approx(1.0, abs=1e-9)
    for c in cands:
        # fallback candidates: neutral nature, zero EVs, no item, unknown tera
        assert c["nature"] == "Hardy"
        assert c["evs"] == {}
        assert c["item"] == ""
        assert c["tera"] is None
        assert c["ability"] in FALLBACK_ABILITIES
        assert 1 <= len(c["moves"]) <= 4


def test_prior_candidate_dims_present():
    for c in build_prior("garchomp", "gen9bss3v3"):
        for key in ("moves", "item", "ability", "nature", "evs", "tera", "weight"):
            assert key in c, f"candidate missing {key}"
        assert isinstance(c["moves"], list)
        assert isinstance(c["evs"], dict)


def test_no_prior_raises_for_absent_species():
    # A CAP is genuinely nonstandard — no Smogon sets AND no legal moves in a
    # standard format — so build_prior has no prior to build. The never-empty
    # guard must fail LOUDLY here (no silent empty belief). colossoil is a
    # recognizable, stable CAP (a fusion) that can never be legal in standard
    # formats, so this keeps testing the real "no data at all" edge case.
    with pytest.raises(ValueError):
        build_prior("colossoil", "gen9bss3v3")


def test_belief_state_p2_only_guard():
    bs = BeliefState("gen9bss3v3")
    sb = bs.add_opponent("p2a", "garchomp")
    assert sb.slot == "p2a"
    with pytest.raises(ValueError):
        bs.add_opponent("p1a", "garchomp")  # own side is not a belief slot


def test_belief_state_rejects_empty_and_nonpositive():
    with pytest.raises(ValueError):
        SlotBelief("p2a", "x", [])
    with pytest.raises(ValueError):
        SlotBelief("p2a", "x", [{"moves": ["tackle"], "weight": 0.0}])


def test_slot_belief_posterior_and_mass():
    sb = BeliefState("gen9bss3v3").add_opponent("p2a", "garchomp")
    assert sum(sb.posterior) == pytest.approx(1.0, abs=1e-9)
    # mass over ALL candidates is the whole posterior; mass over an impossible
    # predicate is 0
    assert sb.mass(lambda c: True) == pytest.approx(1.0, abs=1e-9)
    assert sb.mass(lambda c: "earthquake" in c["moves"] and False) == 0.0
    # every candidate is a dict of the same shape
    assert all(set(c) >= {"moves", "item", "ability", "nature", "evs", "tera", "weight"}
               for c in sb.candidates)


def test_moveset_is_an_unordered_set_user_invariant():
    """User invariant (2026-11-16): ``wish+protect+foulplay = foulplay+protect+wish``.

    A moveset IS a set — order carries no gameplay meaning (the AI picks which
    move to use each turn). Two consequences pinned here so a future refactor
    can't quietly make order matter:

      (a) the prior must not contain two candidates that are the SAME set
          (moves + item + ability + nature + evs + tera) differing ONLY in move
          order — that would double-count one moveset and split its posterior
          mass. The fallback already builds combos with ``itertools.combinations``
          (each 4-subset once, NOT permutations); this locks that in, and checks
          the set/tier prior paths too.
      (b) the belief addresses a set's moves by MEMBERSHIP, never position: two
          candidate lists identical except for one candidate's move order are
          indistinguishable to every belief op (posterior + the membership mass
          that the 3.3 revealed-move filter will use).
    """
    # (a) no order-variant duplicates in ANY prior path
    for species, fmt in (
        ("garchomp", "gen9ou"),            # a format's own Smogon sets
        ("garchomp", "gen9bss3v3"),        # merged same-gen tier pool
        (FALLBACK_SPECIES, "gen9bss3v3"),  # bounded legal-move fallback
    ):
        seen = {}
        for c in build_prior(species, fmt):
            key = (
                frozenset(c["moves"]),
                c["item"], c["ability"], c["nature"],
                tuple(sorted(c["evs"].items())), c["tera"],
            )
            assert key not in seen, (
                f"{species}/{fmt}: order-variant duplicate candidate {key} would "
                f"double-count one moveset and split its mass")
            seen[key] = c
            assert len(set(c["moves"])) == len(c["moves"]), f"repeated move in {c}"

    # (b) order of one candidate's moves is semantically irrelevant
    base = {
        "moves": ["wish", "protect", "foulplay", "struggle"],
        "item": "leftovers", "ability": "prankster", "nature": "Bold",
        "evs": {"hp": 252, "def": 252}, "tera": None, "weight": 1.0,
    }
    reordered = dict(base)
    reordered["moves"] = ["foulplay", "protect", "wish", "struggle"]  # same set
    sb_a = SlotBelief("p2a", "slowking", [dict(base)])
    sb_b = SlotBelief("p2a", "slowking", [dict(reordered)])
    assert sb_a.posterior == sb_b.posterior
    for m in ("wish", "foulplay", "protect", "struggle", "earthquake"):
        assert sb_a.mass(lambda c, m=m: m in c["moves"]) == \
               sb_b.mass(lambda c, m=m: m in c["moves"])


# ═══════════════════════════════ event model ════════════════════════════════

def test_events_are_json_safe():
    evs = [
        E.BattleStart(fmt="gen9bss3v3"),
        E.Turn(n=1),
        E.SwitchIn(side="p2", slot="p2a", species="garchomp"),
        E.MoveUsed(side="p1", slot="p1a", move="earthquake",
                   target_side="p2", target_slot="p2a"),
        E.DamageObserved(target_side="p2", target_slot="p2a", hp_after=100,
                         maxhp=183, ko=False, source_kind="move",
                         move="earthquake", source_side="p1", source_slot="p1a"),
        E.AbilityRevealed(side="p2", slot="p2a", ability="drizzle"),
        E.Unknown(raw="|some|unknown|line"),
    ]
    for ev in evs:
        d = ev.to_dict()
        assert d["type"] == ev.type
        json.dumps(d)  # must not raise


# ═══════════════════════════════ log→event adapter ══════════════════════════

def _types(evs):
    return [e.type for e in evs]


def test_adapter_switch_in_learns_maxhp_and_drops_duplicate():
    evs = events_from_log([
        "|switch|p2a: Garchomp|Garchomp, L50, M|183/183",
        "|switch|p2a: Garchomp|Garchomp, L50, M|100/100",  # sim duplicate
    ])
    assert _types(evs) == ["switch_in"]
    si = evs[0]
    assert si.slot == "p2a" and si.side == "p2"
    assert si.species == "garchomp"
    # max HP ground truth learned from the switch-in line
    a = LogAdapter()
    a.feed_lines(["|switch|p2a: Garchomp|Garchomp, L50|183/183"])
    assert a.maxhp_by_slot["p2a"] == 183


def test_adapter_move_used_with_target():
    ev = events_from_log(["|move|p1a: Garchomp|Earthquake|p2a: Charizard"])
    assert _types(ev) == ["move_used"]
    m = ev[0]
    assert m.move == "earthquake" and m.slot == "p1a"
    assert m.target_slot == "p2a" and m.target_side == "p2"


def test_adapter_move_used_no_target():
    ev = events_from_log(["|move|p1a: Garchomp|Swords Dance"])
    m = ev[0]
    assert m.move == "swordsdance" and m.target_slot is None and m.target_side is None


def test_adapter_damage_dedupe_keeps_every_real_hit():
    # Garchomp (maxhp 183) takes Flamethrower each turn: real + %/100 duplicate.
    lines = ["|switch|p1a: Garchomp|Garchomp, L50|183/183",
             "|switch|p1a: Garchomp|Garchomp, L50|100/100"]
    for hp in (143, 107, 71):
        lines += ["|turn|1", f"|-damage|p1a: Garchomp|{hp}/183",
                  f"|-damage|p1a: Garchomp|{round(hp/183*100)}/100"]
    evs = [e for e in events_from_log(lines) if e.type == "damage_observed"]
    assert [e.hp_after for e in evs] == [143, 107, 71]
    assert all(e.maxhp == 183 for e in evs)
    assert all(not e.ko for e in evs)


def test_adapter_ko_line_clips_to_zero():
    evs = events_from_log([
        "|switch|p2a: Gengar|Gengar, L50|170/170",
        "|switch|p2a: Gengar|Gengar, L50|100/100",
        "|move|p1a: Garchomp|Earthquake|p2a: Gengar",
        "|-damage|p2a: Gengar|0 fnt",   # KO line (no denominator)
        "|-damage|p2a: Gengar|0 fnt",   # the sim's duplicate KO line
    ])
    d = [e for e in evs if e.type == "damage_observed"]
    assert len(d) == 1
    assert d[0].ko is True and d[0].hp_after == 0 and d[0].maxhp == 170


def test_adapter_maxhp100_dedupe():
    # A maxhp==100 mon: the real and duplicate lines are IDENTICAL (X/100) —
    # only the back-to-back identity dedupe can drop the duplicate.
    evs = events_from_log([
        "|switch|p2a: X|X, L50|100/100",
        "|-damage|p2a: X|50/100",
        "|-damage|p2a: X|50/100",   # duplicate
        "|-damage|p2a: X|20/100",   # next real hit (different value → kept)
        "|-damage|p2a: X|20/100",   # duplicate
    ])
    d = [e for e in evs if e.type == "damage_observed"]
    assert [e.hp_after for e in d] == [50, 20]


def test_adapter_residual_not_attributed_to_move():
    """A residual tick (burn) must NOT be attributed to the most recent move.

    Fork-verified (2026-11-16, probe_residual.js): a pure move hit logs exactly
    two payloads (``|-damage|slot|hp/max``); a residual line adds a bracketed
    context tag — burn ``… brn|[from] brn``, recoil ``|[from] Recoil``, confusion
    ``|[from] confusion``, powder ``|[silent]``. That 3rd payload is the
    move-vs-residual signal, and residual damage is NOT move evidence (3.4-6).
    Regression D48: before the fix the tick was pinned to the stale ``|move|``
    line and consumed as (wrong) move damage.
    """
    a = LogAdapter()
    a.feed_lines([
        "|move|p1a: Charizard|Flamethrower|p2a: Blastoise",
        "|-damage|p2a: Blastoise|204/299",                # move hit -> attributed
        "|-damage|p2a: Blastoise|190/299 brn|[from] brn",  # burn tick -> residual
    ])
    dmg = [e for e in a.events if e.type == "damage_observed"]
    assert len(dmg) == 2
    assert dmg[0].source_kind == "move" and dmg[0].move == "flamethrower"
    assert dmg[1].source_kind == "residual"
    assert dmg[1].move is None and dmg[1].source_slot is None


def test_adapter_move_failure_lines_to_movefailed():
    """Fork move-attempt-failure protocol (fork-verified 2026-11-16):

    * an ANNOUNCED move that fails (miss / immunity / Protect / blocked)
      logs ``|move|...`` then ``|-fail|<user>`` (battle-actions.ts:656-659 —
      NO move name in the fail payload); the move name is recovered from the
      user's own announce line;
    * an ABORTED attempt (par/slp/frz/flinch/recharge — conditions.ts
      onBeforeMove; nopp — battle-actions.ts:340, move name in the line)
      logs ``|cant|<user>|<reason>[|Move]`` with NO announce.

    Both are the fork's ``moveThisTurnResult=false`` paths (runMove lines
    319/331/342) — the next turn's Stomping Tantrum / Temper Flare doubles
    (moves.ts:18050/19186). Non-move-failure cant reasons (trp, ability
    blocks, ...) must stay silent.
    """
    # announced then failed: move name recovered from the announce line
    a = LogAdapter()
    a.feed_lines([
        "|move|p2a: Blastoise|Stomping Tantrum|p1a: Charizard",
        "|-fail|p2a: Blastoise",
    ])
    ev = a.events[-1]
    assert ev.type == "move_failed"
    assert (ev.side, ev.slot, ev.move) == ("p2", "p2a", "stompingtantrum")

    # aborted before the announce: no move name available
    a2 = LogAdapter()
    a2.feed_lines(["|cant|p2a: Blastoise|par"])
    assert a2.events[-1].type == "move_failed"
    assert a2.events[-1].move == ""

    # nopp carries the move name in the line
    a3 = LogAdapter()
    a3.feed_lines(["|cant|p2a: Blastoise|nopp|Stomping Tantrum"])
    assert a3.events[-1].type == "move_failed"
    assert a3.events[-1].move == "stompingtantrum"

    # non-move-attempt cant reasons stay silent (no event at all)
    a4 = LogAdapter()
    a4.feed_lines(["|cant|p2a: Blastoise|trp"])
    assert a4.events == []


def test_adapter_slot_reuse_updates_maxhp():
    # p2a is Garchomp (183), then a new mon (Blastoise, 150) takes the active
    # slot after a faint. The slot's max HP must follow the new mon.
    a = LogAdapter()
    a.feed_lines(["|switch|p2a: Garchomp|Garchomp, L50|183/183",
                  "|switch|p2a: Garchomp|Garchomp, L50|100/100",
                  "|faint|p2a: Garchomp",
                  "|switch|p2a: Blastoise|Blastoise, L50|150/150",
                  "|switch|p2a: Blastoise|Blastoise, L50|100/100"])
    assert a.maxhp_by_slot["p2a"] == 150


def test_adapter_noise_and_empty_lines_are_ignored():
    lines = ["", "   ", "|t:|1794783370", "|gametype|singles",
             "|clearpoke", "|poke|p1|Garchomp, L50, M|", "|teampreview|3",
             "|split|p1", "|-immune|p2a: Charizard", "|-resisted|p1a: Garchomp",
             "|-supereffective|p2a: X", "|upkeep", "|-crit|p2a: X"]
    assert events_from_log(lines) == []


def test_adapter_prepare_line_is_ignored():
    # |-prepare|ATTACKER|MOVE (SIM-PROTOCOL.md:594) is the client charge/aim
    # animation line — the real move is the |move| line we DO map. It must be
    # ignored, NOT surfaced as Unknown (the 3.7 "benign prepare" Unknowns).
    evs = events_from_log(["|-prepare|p1a: Garchomp|Focus Punch|p2a: Charizard"])
    assert evs == []
    # and a genuinely-unknown tag still surfaces (the bug-catcher stays live)
    assert _types(events_from_log(["|-totally-bogus-tag|p1a: X|42"])) == ["unknown"]


def test_adapter_unknown_is_surfaced():
    # A line with a tag we neither map nor ignore must surface as Unknown.
    evs = events_from_log(["|-totally-bogus-tag|p1a: X|42"])
    assert _types(evs) == ["unknown"]
    assert evs[0].raw.startswith("|-totally-bogus-tag")


def test_adapter_malformed_line_surfaces_not_crash():
    # A known tag with a bad payload must not abort the battle parse.
    evs = events_from_log(["|turn|not-a-number", "|turn|2"])
    assert evs[0].type == "unknown"          # the bad line is surfaced
    assert evs[1].type == "turn" and evs[1].n == 2   # ...and parsing continues


def test_adapter_item_from_context():
    evs = events_from_log(["|-enditem|p2a: Gengar|[from] item: Choice Band"])
    items = [e for e in evs if e.type == "item_revealed"]
    assert len(items) == 1
    assert items[0].slot == "p2a" and items[0].item == "choiceband"


def test_adapter_ability_from_weather_context():
    evs = events_from_log(["|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Starmie"])
    assert any(e.type == "weather_changed" for e in evs)
    ab = [e for e in evs if e.type == "ability_revealed"]
    assert len(ab) == 1
    assert ab[0].slot == "p2a" and ab[0].ability == "drizzle"


def test_adapter_status_inflict_attributes_ability_to_holder():
    # The VICTIM (p1a) takes the status; the HOLDER (p2a) owns the ability.
    evs = events_from_log(["|-status|p1a: Garchomp|par|[from] ability: Static|[of] p2a: Joltik"])
    st = [e for e in evs if e.type == "status_applied"]
    ab = [e for e in evs if e.type == "ability_revealed"]
    assert st and st[0].slot == "p1a" and st[0].status == "par"
    assert ab and ab[0].slot == "p2a" and ab[0].ability == "static"


def test_adapter_direct_ability_change():
    evs = events_from_log(["|-ability|p2a: Eiscue|Ice Face|Huge Power|[from] Skill Swap"])
    ab = [e for e in evs if e.type == "ability_revealed"]
    assert ab and ab[0].slot == "p2a" and ab[0].ability == "iceface"


def test_adapter_terastallize():
    evs = events_from_log(["|-terastallize|p2a: Garchomp|Fire"])
    t = [e for e in evs if e.type == "tera_used"]
    assert t and t[0].slot == "p2a" and t[0].tera_type == "fire"


def test_adapter_boost_signs():
    evs = events_from_log([
        "|-boost|p2a: X|atk|1",
        "|-unboost|p2a: X|def|1",
        "|-setboost|p2a: X|atk|6",
    ])
    got = {(e.stat, e.delta) for e in evs if e.type == "boost_changed"}
    assert ("atk", 1) in got
    assert ("def", -1) in got
    assert ("atk", 6) in got


def test_adapter_hitcount():
    evs = events_from_log(["|-hitcount|p1a: X|3"])
    h = [e for e in evs if e.type == "hit_count"]
    assert h and h[0].n == 3


def test_adapter_faint_and_outcome():
    evs = events_from_log(["|faint|p2a: Gengar", "|win|p1"])
    assert _types(evs) == ["faint", "battle_end"]
    assert evs[1].winner_side == "p1"


# ═══════════════════════════════ live battle smoke ══════════════════════════

def _run_smoke_battle():
    """Run the seeded fork battle keeper and return its raw log lines."""
    script = ROOT / "scratch" / "step3_1_battle_log.js"
    proc = subprocess.run(
        ["node", str(script)], cwd=str(ROOT),
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise AssertionError(f"smoke keeper failed (rc={proc.returncode})\n{proc.stderr}")
    lines = []
    in_block = False
    for raw in proc.stdout.splitlines():
        if raw.strip() == "=== BATTLE LOG START ===":
            in_block = True
            continue
        if raw.strip() == "=== BATTLE LOG END ===":
            break
        if in_block and raw.startswith("|"):
            lines.append(raw)
    return lines


def test_smoke_live_battle_event_stream():
    lines = _run_smoke_battle()
    assert len(lines) > 5, "smoke battle log looks empty"
    evs = events_from_log(lines)
    types = [e.type for e in evs]

    # the core evidence is present
    assert "battle_start" in types
    p2_switches = [e for e in evs if e.type == "switch_in" and e.side == "p2"]
    assert len(p2_switches) >= 2, "expected at least two p2 switch-ins"
    moves = [e for e in evs if e.type == "move_used"]
    assert len(moves) >= 2, "expected at least two move uses"
    dmg = [e for e in evs if e.type == "damage_observed"]
    assert dmg, "expected at least one damage observation"
    assert all(d.maxhp > 0 for d in dmg), "every damage event must carry a real max HP"

    # the sim's doubled hp lines are gone
    unknown = [e for e in evs if e.type == "unknown"]
    assert unknown == [], f"unmapped log lines surfaced as Unknown: {[e.raw for e in unknown]}"

    # sane ordering: the first Turn precedes the first MoveUsed
    first_turn = types.index("turn")
    first_move = types.index("move_used")
    assert first_turn < first_move

    # readable per-turn report (observable bug-catcher) — printed for -s runs
    print("\n=== STEP 3.1 SMOKE: EVENT REPORT ===")
    for e in evs:
        print("  ", e.to_dict())
    print(f"  ({len(evs)} events, {len(dmg)} damage, {len(unknown)} unknown)")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
