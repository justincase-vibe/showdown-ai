"""Phase 2 core tests — replay library (save / index / fetch).

All FAST: no servers, no GPU, no network. One test uses a REAL fixture record
(from ``bot/data/replays/gen9ou.jsonl``) through the REAL BC walker
(``corpus.trajectories.replay_battle``) to pin the byte-compatibility contract.

Run:  python -m pytest replays/tests/test_replays.py -q
"""
import threading

import pytest

from replays import ReplayLibrary, make_record


# ── fakes ───────────────────────────────────────────────────────────────────
class FakeBattle:
    """Duck-typed poke-env Battle (the only surface ``make_record`` uses)."""

    def __init__(self, tag="gen9ou-123456", rating=1500.0,
                 log="|turn|1\n|move|p1a|tackle\n|win|p1a"):
        self.battle_tag = tag
        self.rating = rating
        self._log = log

    def _build_replay_log(self):
        return self._log


class FakeNamedBattle(FakeBattle):
    """Adds the identity surface poke-env exposes (used to map username-based
    fallback win lines — ``_build_replay_events`` writes ``|win|<username>``
    when the server dropped the terminal event)."""

    def __init__(self, role="p1", me="Alice", opp="Bob", won=None, **kw):
        super().__init__(**kw)
        self.player_role = role
        self.player_username = me
        self.opponent_username = opp
        self.won = won
        self.players = (me, opp) if role == "p1" else (opp, me)


def _rec(battle, fid="gen9ou", **kw):
    return make_record(battle, fid, players=["alice", "bob"], **kw)


# ── record capture ──────────────────────────────────────────────────────────
def test_make_record_has_fixture_fields():
    r = _rec(FakeBattle())
    for k in ("id", "formatid", "format", "rating", "players", "log"):
        assert k in r, f"missing fixture field {k}"
    assert r["id"] == "gen9ou-123456"
    assert r["rating"] == 1500.0
    assert r["players"] == ["alice", "bob"]


def test_make_record_parses_winner_side_p2():
    r = _rec(FakeBattle(log="|turn|1\n|move|p2a|tackle\n|win|p2a"))
    assert r["winner_side"] == "p2"


def test_make_record_winner_tie_is_none():
    r = _rec(FakeBattle(log="|turn|1\n|tie"))
    assert r["winner_side"] is None


def test_make_record_canonicalizes_username_win_line_p1_won():
    # poke-env fallback line: local player (p1) won -> |win|Alice -> |win|p1a
    b = FakeNamedBattle(role="p1", me="Alice", opp="Bob", won=True,
                        log="|turn|1\n|win|Alice")
    r = make_record(b, "gen9ou")
    assert r["log"].splitlines()[-1] == "|win|p1a"
    assert r["winner_side"] == "p1"
    assert r["players"] == ["Alice", "Bob"]


def test_make_record_canonicalizes_username_win_line_p2_won():
    # local player is p2 and WON -> |win|Alice must map to p2, not p1
    b = FakeNamedBattle(role="p2", me="Alice", opp="Bob", won=True,
                        log="|turn|1\n|win|Alice")
    r = make_record(b, "gen9ou")
    assert r["log"].splitlines()[-1] == "|win|p2a"
    assert r["winner_side"] == "p2"


def test_make_record_canonicalizes_username_win_line_opponent_won():
    # local player is p1 and LOST (opponent Bob won) -> |win|Bob -> |win|p2a
    b = FakeNamedBattle(role="p1", me="Alice", opp="Bob", won=False,
                        log="|turn|1\n|win|Bob")
    r = make_record(b, "gen9ou")
    assert r["log"].splitlines()[-1] == "|win|p2a"
    assert r["winner_side"] == "p2"


def test_make_record_keeps_unknown_username_line():
    # no identity at all -> do not guess, leave the log untouched
    b = FakeBattle(log="|turn|1\n|win|Someone")
    r = make_record(b, "gen9ou")
    assert r["winner_side"] is None
    assert r["log"].splitlines()[-1] == "|win|Someone"


def test_make_record_defaults_and_meta():
    r = _rec(FakeBattle(), elo=1650.5, bots=["botA", "botB"])
    assert r["elo"] == 1650.5 and r["bots"] == ["botA", "botB"]
    assert r["ts"] and r["format"] == "gen9ou"


# ── save / fetch_last ───────────────────────────────────────────────────────
def test_save_and_fetch_last_order(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    for i in range(5):
        lib.save(_rec(FakeBattle(tag=f"gen9ou-{1000 + i}")))
    last2 = lib.fetch_last(2)
    assert [r["id"] for r in last2] == ["gen9ou-1003", "gen9ou-1004"]
    assert lib.count() == 5


def test_save_dedup_by_id(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    first = lib.save(_rec(FakeBattle(tag="gen9ou-1")))
    again = lib.save(_rec(FakeBattle(tag="gen9ou-1")))
    assert first["deduped"] is False and again["deduped"] is True
    assert lib.count() == 1
    assert lib.fetch_last(10)[0]["id"] == "gen9ou-1"


def test_save_dedupes_recapture_even_with_new_timestamp(tmp_path):
    # re-capturing the SAME battle (identical id+log, fresh ts) must dedup —
    # the key excludes ts by design
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="gen9ou-1"), ts="2026-01-01T00:00:00Z"))
    again = lib.save(_rec(FakeBattle(tag="gen9ou-1"), ts="2026-02-02T00:00:00Z"))
    assert again["deduped"] is True
    assert lib.count() == 1


def test_save_keeps_same_id_different_log(tmp_path):
    # D95.1: the fork's battle tag is a per-server counter — two servers CAN
    # issue the same tag for different concurrent battles. Both must be kept.
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="battle-gen9ou-7",
                             log="|turn|1\n|move|p1a|tackle\n|win|p1a")))
    out = lib.save(_rec(FakeBattle(tag="battle-gen9ou-7",
                                   log="|turn|1\n|move|p1a|flamethrower\n|win|p2a")))
    assert out["deduped"] is False
    assert lib.count() == 2
    assert {r["log"].split("\n")[1] for r in lib.fetch_last(2)} == {
        "|move|p1a|tackle", "|move|p1a|flamethrower"}


def test_save_requires_id(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    with pytest.raises(ValueError):
        lib.save({"formatid": "gen9ou", "log": "x"})


def test_fetch_last_per_format(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="gen9ou-1"), fid="gen9ou"))
    lib.save(_rec(FakeBattle(tag="gen9uu-1"), fid="gen9uu"))
    assert lib.fetch_last(10, formatid="gen9uu")[0]["id"] == "gen9uu-1"
    assert lib.count("gen9uu") == 1 and lib.count() == 2
    assert set(lib.formats()) == {"gen9ou", "gen9uu"}


# ── fetch_top ───────────────────────────────────────────────────────────────
def test_fetch_top_by_elo(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="gen9ou-lo"), elo=1400.0))
    lib.save(_rec(FakeBattle(tag="gen9ou-mid"), elo=1550.0))
    lib.save(_rec(FakeBattle(tag="gen9ou-hi"), elo=1600.0))
    lib.save(_rec(FakeBattle(tag="gen9ou-none")))          # elo=None -> sinks
    top2 = lib.fetch_top(2, key="elo")
    assert [r["id"] for r in top2] == ["gen9ou-hi", "gen9ou-mid"]
    all4 = lib.fetch_top(4, key="elo")
    assert all4[-1]["id"] == "gen9ou-none"                 # missing key last


def test_fetch_top_ties_most_recent_first(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="gen9ou-old"), elo=1500.0))
    lib.save(_rec(FakeBattle(tag="gen9ou-new"), elo=1500.0))
    top = lib.fetch_top(2, key="elo")
    assert [r["id"] for r in top] == ["gen9ou-new", "gen9ou-old"]


def test_fetch_top_unknown_key_rejected(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    with pytest.raises(ValueError):
        lib.fetch_top(1, key="nope")


def test_fetch_top_by_rating_p4_era(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="gen9ou-a", rating=1200.0)))
    lib.save(_rec(FakeBattle(tag="gen9ou-b", rating=1800.0)))
    assert lib.fetch_top(1, key="rating")[0]["id"] == "gen9ou-b"


# ── concurrency (the P4/P5 trainer saves from N battle threads) ─────────────
def test_concurrent_saves_no_lost_no_corruption(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    n_threads, per = 8, 25

    def worker(t):
        for i in range(per):
            lib.save(_rec(FakeBattle(tag=f"gen9ou-{t}-{i}")))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert lib.count() == n_threads * per
    got = lib.fetch_last(n_threads * per)
    assert len({r["id"] for r in got}) == n_threads * per   # no dupes, no losses


# ── index durability ────────────────────────────────────────────────────────
def test_index_survives_reload(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="gen9ou-1"), elo=1500.0))
    lib2 = ReplayLibrary(str(tmp_path))          # fresh object, same root
    assert lib2.count() == 1
    assert lib2.fetch_top(1, key="elo")[0]["id"] == "gen9ou-1"


def test_cold_start_rebuilds_index_from_jsonl(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="gen9ou-1")))
    # simulate a lost index: a new library must rebuild it from the JSONL
    lib.index_path.unlink()
    lib2 = ReplayLibrary(str(tmp_path))
    assert lib2.count() == 1
    assert lib2.fetch_last(1)[0]["id"] == "gen9ou-1"


def test_corrupt_index_rebuilds(tmp_path):
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="gen9ou-1")))
    lib.index_path.write_text("{not json", encoding="utf-8")
    lib2 = ReplayLibrary(str(tmp_path))
    assert lib2.count() == 1


# ── THE compatibility contract: a stored record must replay through the BC walker
def test_stored_record_roundtrips_through_bc_walker(tmp_path):
    import os
    from corpus.trajectories import replay_battle

    fixture = os.path.join("bot", "data", "replays", "gen9ou.jsonl")
    if not os.path.exists(fixture):
        pytest.skip("fixture not present in this checkout")
    with open(fixture, "r", encoding="utf-8", errors="replace") as fh:
        rec = __import__("json").loads(fh.readline())

    lib = ReplayLibrary(str(tmp_path))
    lib.save(rec)
    out = lib.fetch_last(1)[0]
    for k in ("id", "formatid", "format", "rating", "players", "log"):
        assert out[k] == rec[k], f"field {k} changed in the round trip"

    trace = replay_battle(out)
    assert trace.skipped is None, f"BC walker skipped the stored record: {trace.skipped}"
    assert trace.n_decisions > 0, "the stored record yielded no decisions"


# ── bug-hunt pins (2026-11-19 small-bug hunt) ───────────────────────────────
def test_username_with_p1_prefix_is_not_a_side_token():
    # a real username like ``p1xel`` must NOT be read as side p1 — no identity
    # means we do not guess (winner stays None, log stays untouched)
    r = make_record(FakeBattle(log="|turn|1\n|win|p1xel"), "gen9ou")
    assert r["winner_side"] is None
    assert r["log"].splitlines()[-1] == "|win|p1xel"


def test_username_with_p1_prefix_is_mapped_when_identity_known():
    # ...but WITH identity the SAME username maps correctly to its real side
    b = FakeNamedBattle(role="p2", me="p1xel", opp="Bob", won=True,
                        log="|turn|1\n|win|p1xel")
    r = make_record(b, "gen9ou")
    assert r["log"].splitlines()[-1] == "|win|p2a"
    assert r["winner_side"] == "p2"


def test_scan_survives_partial_final_line(tmp_path):
    # a crash mid-save leaves a partial last line — cold start must skip it,
    # not crash, and the good record must still be reachable at its offset
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="gen9ou-good")))
    with open(lib.path("gen9ou"), "ab") as fh:
        fh.write(b'{"id": "gen9ou-cut", "log": "|win|p1')   # no closing brace
    lib2 = ReplayLibrary(str(tmp_path))
    assert lib2.count() == 1
    assert lib2.fetch_last(1)[0]["id"] == "gen9ou-good"


def test_save_sanitizes_formatid(tmp_path):
    # the formatid becomes a FILENAME — traversal/odd chars are neutralized
    lib = ReplayLibrary(str(tmp_path))
    out = lib.save(_rec(FakeBattle(), fid="../evil"))
    import os
    assert os.path.basename(out["path"]) == "___evil.jsonl"
    assert lib.count() == 1


def test_corrupt_index_reader_does_not_rewrite(tmp_path):
    # a READER instance hitting a corrupt index must scan in memory and NOT
    # clobber the index file (the writer process self-heals it on next save)
    lib = ReplayLibrary(str(tmp_path))
    lib.save(_rec(FakeBattle(tag="gen9ou-1")))
    corrupt = b"{not json"
    lib.index_path.write_bytes(corrupt)
    lib2 = ReplayLibrary(str(tmp_path))
    assert lib2.count() == 1
    assert lib2.index_path.read_bytes() == corrupt, "reader must not rewrite the index"
