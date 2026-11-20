"""D101 — replay viewer (build_replay_html / write_replay_file) + get_by_id.

Format contract (verified against the fork, _client_src/play.pokemonshowdown.com):
  * battle-log.ts::createReplayFile (~line 1811) — the file layout;
  * panel-battle.tsx::handleDrop (~line 470) — drag&drop acceptance
    (needs <title> + battle-log-data script; reverses \\/ -> / on load).

Run:  python -m pytest replays/tests/test_view.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FAKE_RECORD = {
    "id": "49960-battle-gen9ou-3196",
    "formatid": "gen9ou",
    "format": "gen9ou",
    "players": ["ShowdownEnv pte6q", "ShowdownEnv nnwrs"],
    "log": ">battle-gen9ou-3196\n|init|battle\n|damage|p2a: X|43/100\n|turn|2\n",
}


# ── build_replay_html ────────────────────────────────────────────────────────

def test_html_has_required_contract_parts():
    from replays.view import build_replay_html
    h = build_replay_html(FAKE_RECORD)
    assert h.startswith("<!DOCTYPE html>")
    assert '<meta charset="utf-8" />' in h
    assert "<!-- version 1 -->" in h
    # title = <tier> replay: <p1> vs. <p2> (client uses it for the tab/panel name)
    assert "<title>gen9ou replay: ShowdownEnv pte6q vs. ShowdownEnv nnwrs</title>" in h
    # the battle-log-data script tag (handleDrop's acceptance contract)
    assert '<script type="text/plain" class="battle-log-data">' in h
    # replayid input (room name)
    assert 'name="replayid" value="49960-battle-gen9ou-3196"' in h
    # browser bootstrap (replay-embed.js, default global client route)
    assert "replay-embed.js?version" in h
    assert "play.pokemonshowdown.com/js/" in h


def test_html_escapes_slashes_in_log():
    """The client reverses \\/ -> / on load (handleDrop), so the stored file
    MUST contain backslash-escaped slashes for the log data."""
    from replays.view import build_replay_html
    h = build_replay_html(FAKE_RECORD)
    m = re.search(r'class="battle-log-data">(.*)</script>', h, re.S)
    assert m is not None
    data = m.group(1)
    assert "43\\/100" in data      # escaped form present
    assert "43/100" not in data   # raw (unescaped) form absent
    # the rest of the log is intact
    assert "|init|battle" in data


def test_html_escapes_html_in_title_and_players():
    from replays.view import build_replay_html
    rec = dict(FAKE_RECORD, players=["<script>alert(1)</script>", 'Bob "B"'])
    h = build_replay_html(rec)
    # title = "{tier} replay: {p1} vs. {p2}" — the markup must be escaped, not injected
    assert "<script>alert(1)</script>" not in h
    assert "replay: &lt;script&gt;alert(1)&lt;/script&gt; vs." in h
    assert 'Bob &quot;B&quot;' in h


def test_html_missing_id_raises():
    from replays.view import build_replay_html
    with pytest.raises(ValueError):
        build_replay_html({"log": "|init|battle"})


# ── write_replay_file ────────────────────────────────────────────────────────

def test_write_replay_file_idempotent(tmp_path):
    from replays.view import build_replay_html, write_replay_file
    p1 = write_replay_file(FAKE_RECORD, out_dir=tmp_path)
    first = p1.read_text(encoding="utf-8")
    p2 = write_replay_file(FAKE_RECORD, out_dir=tmp_path)
    assert p1 == p2 == tmp_path / "49960-battle-gen9ou-3196.html"
    assert p2.read_text(encoding="utf-8") == first
    # content matches the builder
    assert first == build_replay_html(FAKE_RECORD)


def test_write_replay_file_sanitizes_filename(tmp_path):
    from replays.view import write_replay_file
    rec = dict(FAKE_RECORD, id="weird/id with spaces?")
    p = write_replay_file(rec, out_dir=tmp_path)
    assert "/" not in p.stem and p.parent == tmp_path
    assert p.exists()


# ── ReplayLibrary.get_by_id ─────────────────────────────────────────────────

def test_get_by_id_roundtrip(tmp_path):
    from replays import ReplayLibrary
    lib = ReplayLibrary(root=str(tmp_path))
    lib.save(FAKE_RECORD)
    rec = lib.get_by_id("49960-battle-gen9ou-3196")
    assert rec is not None
    assert rec["log"] == FAKE_RECORD["log"]
    assert rec["players"] == FAKE_RECORD["players"]
    assert lib.get_by_id("no-such-id") is None


def test_get_by_id_shared_id_returns_last_saved(tmp_path):
    """Two DIFFERENT battles sharing an id are both kept (D95.1); get_by_id
    returns the most recent one (index order = save order)."""
    from replays import ReplayLibrary
    lib = ReplayLibrary(root=str(tmp_path))
    lib.save(dict(FAKE_RECORD, log="|init|battle\n|turn|1\nOLD"))
    lib.save(dict(FAKE_RECORD, log="|init|battle\n|turn|1\nNEW"))
    rec = lib.get_by_id(FAKE_RECORD["id"])
    assert rec is not None
    assert rec["log"].endswith("NEW")
