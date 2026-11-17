"""S4.0 corpus pipeline tests — OFFLINE, fixture-based, fast.

Covers the S4.0 gate (PLAN.md §3c):
  * adapter  — 0 unknown-tag errors on the real gen9ou ladder fixture (DB-
               independent; runs over EVERY line, including skipped battles);
  * replay   — both POVs, turn-start contexts, legality rate >= 95%,
               switch-rate in [0,1], determinism;
  * puller   — deterministic + resumable cache hit, no network (fake fetcher);
  * skip     — a species with no DB prior is skipped with a precise reason
               (documents the known gen9ou top-OU DB gap — P0 blocker);
  * forced   — |drag|/|replace| map to SwitchIn(forced=True) and are NOT
               player decisions (keeps the human switch-rate accurate).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from belief.events import SwitchIn, Unknown
from belief.log_events import LogAdapter
from corpus.pull import load_local, pull_battles
from corpus.report import corpus_report, format_report
from corpus.trajectories import replay_battle, replay_many

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bot" / "data" / "replays" / "gen9ou.jsonl"
HAS_FIXTURE = FIXTURE.exists()

needs_fixture = pytest.mark.skipif(
    not HAS_FIXTURE, reason="gen9ou.jsonl human-ladder fixture missing")


def _lines_of(battle: dict):
    log = battle.get("log")
    return log.split("\n") if isinstance(log, str) else list(log or [])


# ── adapter gate: 0 unknown tags (DB-independent) ─────────────────────
@needs_fixture
def test_adapter_zero_unknown_on_all_200():
    battles = load_local(str(FIXTURE), 200)
    assert len(battles) == 200
    unknown = 0
    for b in battles:
        adapter = LogAdapter()
        for line in _lines_of(b):
            if isinstance(adapter.feed_line(line), Unknown):
                unknown += 1
    assert unknown == 0  # the S4.0 "0 unknown-tag errors" gate


# ── replay: structure, POVs, contexts ──────────────────────────────────
@needs_fixture
def test_replay_both_povs_and_turn_start_contexts():
    traces = [t for t in replay_many(load_local(str(FIXTURE), 30))
              if t.skipped is None]
    assert traces, "no battles replayed in the fixture sample"
    for t in traces:
        assert t.n_decisions > 0
        assert {"p1", "p2"} <= {d.pov for d in t.decisions}
        # one turn-start snapshot per turn, none missing, none extra
        assert len(t.context_seq) == t.n_turns >= 1
        for ctx in t.context_seq:
            for key in ("turn", "fainted", "active", "field", "own", "opponent"):
                assert key in ctx


@needs_fixture
def test_replay_determinism():
    b = load_local(str(FIXTURE), 1)[0]
    assert replay_battle(b).to_dict() == replay_battle(b).to_dict()


# ── replay: S4.0 numeric gates ─────────────────────────────────────────
@needs_fixture
def test_legal_rate_ge_95():
    rep = corpus_report(replay_many(load_local(str(FIXTURE), 40)))
    assert rep["n_decisions"] > 0
    assert rep["legal_rate"] >= 0.95  # ">=95% yield a legal action"


@needs_fixture
def test_switch_rate_in_unit_interval_and_report_fields():
    rep = corpus_report(replay_many(load_local(str(FIXTURE), 40)))
    assert 0.0 <= rep["human_switch_rate"] <= 1.0  # the S4.3 target
    for key in ("n_battles_total", "n_replayed", "n_skipped", "skipped_reasons",
                "format_mix", "rating_dist", "n_decisions", "n_legal",
                "legal_rate", "n_switch_decisions", "n_move_decisions",
                "human_switch_rate", "unknown_count", "unknown_tags",
                "both_povs_rate"):
        assert key in rep
    # the console report must print on a cp1252 Windows console
    format_report(rep).encode("ascii")


# ── puller: deterministic + resumable, no network ──────────────────────
def _fake_fetcher(calls: dict):
    def fetcher(formatid: str, max_battles: int, min_rating) -> list:
        calls["n"] += 1
        return [{"id": f"b{i}", "formatid": formatid,
                 "rating": 1500.0 + i, "log": "|start"}
                for i in range(max_battles)]
    return fetcher


def test_resumable_puller_cache_hit(tmp_path):
    calls = {"n": 0}
    fetcher = _fake_fetcher(calls)
    out_dir = str(tmp_path / "replays")

    first = pull_battles("gen9ou", max_battles=200, min_rating=1400.0,
                         out_dir=out_dir, fetcher=fetcher)
    assert len(first) == 200
    assert calls["n"] == 1
    assert (tmp_path / "replays" / "gen9ou.jsonl").exists()

    second = pull_battles("gen9ou", max_battles=200, min_rating=1400.0,
                          out_dir=out_dir, fetcher=fetcher)
    assert calls["n"] == 1  # cache hit — the network was NOT re-hit
    assert [b["id"] for b in second] == [b["id"] for b in first]  # deterministic


def test_puller_refetches_short_cache(tmp_path):
    calls = {"n": 0}

    def short_fetcher(formatid: str, max_battles: int, min_rating) -> list:
        calls["n"] += 1
        return [{"id": f"b{i}", "formatid": formatid, "rating": 1500.0 + i,
                 "log": "|start"} for i in range(50)]

    a = pull_battles("gen9uu", max_battles=200, min_rating=1400.0,
                     out_dir=str(tmp_path), fetcher=short_fetcher)
    assert len(a) == 50 and calls["n"] == 1

    b = pull_battles("gen9uu", max_battles=200, min_rating=1400.0,
                     out_dir=str(tmp_path), fetcher=short_fetcher)
    # cache holds fewer than max_battles -> re-pull (still deterministic)
    assert calls["n"] == 2
    assert [r["id"] for r in b] == [r["id"] for r in a]


# ── forced entries: drag/replace are NOT player decisions ──────────────
def test_forced_entry_flags():
    drag = LogAdapter().feed_line("|drag|p2a 1| Kyurem, 120 / 183")
    assert isinstance(drag, SwitchIn) and drag.forced is True
    repl = LogAdapter().feed_line("|replace|p2a 1| Kyurem, 120 / 183")
    assert isinstance(repl, SwitchIn) and repl.forced is True
    sw = LogAdapter().feed_line("|switch|p2a 1| Kyurem, 183 / 183")
    assert isinstance(sw, SwitchIn) and sw.forced is False


def test_drag_not_a_switch_decision():
    """A |drag| mid-battle updates the belief but must NOT emit a switch
    decision (it is game-forced, not chosen) — switch-rate stays accurate."""
    battle = {
        "id": "synthetic-drag", "formatid": "gen9ou", "rating": 1500.0,
        "log": "\n".join([
            "|gen|9",
            "|start",
            "|teamsize| 6",
            "|poke| p1a 1 | Kyurem",
            "|poke| p2a 1 | Great Tusk",
            "|switch| p1a 1 | Kyurem, 252 / 252",   # leads (pre-turn-1, forced)
            "|switch| p2a 1 | Great Tusk, 183 / 183",
            "|turn| 1",
            "|drag| p2a 1 | Kyurem, 120 / 183",     # game-forced entry
            "|move| p1a 1 | Ice Beam | p2a 1",
            "|turn| 2",
            "|move| p1a 1 | Freeze-Dry | p2a 1",
            "|win| p1",
        ]),
    }
    t = replay_battle(battle)
    assert t.skipped is None
    assert t.unknown_count == 0
    assert [d for d in t.decisions if d.kind == "switch"] == []
    moves = [d for d in t.decisions if d.kind == "move"]
    assert [d.action for d in moves] == ["icebeam", "freezedry"]
    assert all(d.legal for d in moves)
    assert t.winner_side == "p1"


# ── winner resolution: |win| payloads are USERNAMES in ladder logs ─────
def test_win_by_username_resolves_side():
    """Ladder logs end with `|win| <winner-username>` (verified on the
    gen9ou fixture: all 2000 battles), not `|win| p1`. The adapter must
    resolve the username to a side via the `|player|` identity lines."""
    from belief.events import BattleEnd
    ad = LogAdapter()
    ad.feed_line("|player|p1|☆CT1 bea|avatar|1990")
    ad.feed_line("|player|p2|Mimikyu Star|avatar|2010")
    ev = ad.feed_line("|win| ☆CT1 bea")
    assert isinstance(ev, BattleEnd) and ev.winner_side == "p1"
    # case-insensitive usernames (Showdown ids are casefolded)
    ad2 = LogAdapter()
    ad2.feed_line("|player| p2 | Mimikyu Star | x | 2010")
    ev2 = ad2.feed_line("|win| mimikyu star")
    assert ev2.winner_side == "p2"
    # direct side payloads (sim logs) keep working
    ad3 = LogAdapter()
    ev3 = ad3.feed_line("|win| p2")
    assert ev3.winner_side == "p2"
    # unresolvable username -> raw kept (observable), NOT silently None
    ev4 = LogAdapter().feed_line("|win| ghost user")
    assert ev4.winner_side == "ghost user"


@needs_fixture
def test_winner_side_resolved_to_side_on_real_battles():
    """On real ladder battles, winner_side must end up as 'p1'/'p2' —
    the false-truth trap of this session: a synthetic `|win| p1` test passed
    while EVERY real battle carried a username."""
    traces = [t for t in replay_many(load_local(str(FIXTURE), 20))
              if t.skipped is None]
    assert traces
    for t in traces:
        assert t.winner_side in ("p1", "p2"), \
            f"unresolved winner_side: {t.winner_side!r} (battle {t.battle_id})"


# ── clean skip: no-prior species (the known P0 DB gap) ─────────────────
def test_no_prior_species_cleanly_skipped():
    """Landorus(-Therian) has no gen9ou prior in the current DB (fork tier
    data tags it Uber) — the walker must SKIP with a precise reason, not
    crash, so the corpus stays clean and the gap is reportable."""
    battle = {
        "id": "synthetic-no-prior", "formatid": "gen9ou", "rating": 1500.0,
        "log": "\n".join([
            "|start",
            "|poke| p1a 1 | Landorus-Therian",
            "|poke| p2a 1 | Kyurem",
            "|turn| 1",
            "|switch| p1a 1 | Landorus-Therian, 245 / 245",
            "|switch| p2a 1 | Kyurem, 252 / 252",
        ]),
    }
    t = replay_battle(battle)
    assert t.skipped is not None
    assert t.skipped.startswith("no prior")
    assert t.decisions == [] and t.context_seq == []
