"""League bot registry tests — add, remove, ELO update, standings, persistence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.registry import BotRegistry, Bot, ELO_INITIAL, ELO_K


@pytest.fixture
def reg(tmp_path):
    return BotRegistry(path=tmp_path / "bots.json")


def test_add_and_get(reg):
    reg.add("alice", "scratch/50b_bc_a3f2.pt")
    bot = reg.get("alice")
    assert bot.name == "alice"
    assert bot.ckpt_path == "scratch/50b_bc_a3f2.pt"
    assert bot.elo == ELO_INITIAL
    assert bot.wins == 0


def test_remove(reg):
    reg.add("alice", "a.pt")
    reg.remove("alice")
    with pytest.raises(KeyError):
        reg.get("alice")
    assert len(reg) == 0


def test_remove_missing_raises(reg):
    with pytest.raises(KeyError, match="not in roster"):
        reg.remove("ghost")


def test_update_result_win(reg):
    reg.add("alice", "a.pt")
    reg.add("bob", "b.pt")
    reg.update_result("alice", "bob", winner="alice")
    assert reg.get("alice").wins == 1
    assert reg.get("bob").losses == 1
    # alice's ELO went up, bob's went down
    assert reg.get("alice").elo > ELO_INITIAL
    assert reg.get("bob").elo < ELO_INITIAL
    # total ELO is conserved (zero-sum)
    assert reg.get("alice").elo + reg.get("bob").elo == 2 * ELO_INITIAL


def test_update_result_loss(reg):
    reg.add("alice", "a.pt")
    reg.add("bob", "b.pt")
    reg.update_result("alice", "bob", winner="bob")
    assert reg.get("bob").wins == 1
    assert reg.get("alice").losses == 1
    assert reg.get("bob").elo > ELO_INITIAL
    assert reg.get("alice").elo < ELO_INITIAL


def test_update_result_draw(reg):
    reg.add("alice", "a.pt")
    reg.add("bob", "b.pt")
    reg.update_result("alice", "bob", winner=None)
    assert reg.get("alice").draws == 1
    assert reg.get("bob").draws == 1
    # ELO unchanged for a draw (both start at the same rating)
    assert reg.get("alice").elo == ELO_INITIAL
    assert reg.get("bob").elo == ELO_INITIAL


def test_elo_magnitudes_standard(reg):
    """Exact standard ELO magnitudes (K=32), pinned against the formula.

    A=1500 vs B=1700: E_A = 1/(1+10^(200/400)) = 0.24025...
      A wins -> A +round(32*(1-E_A)) = +24,  B -24
      B wins -> A +round(32*(0-E_A)) = -8,   B +8   (upset reward)
      draw   -> A +round(32*(0.5-E_A)) = +8, B -8
    The OLD loss branch used ``-K*(1-E_A)`` (giving -24/+24 instead of
    -8/+8) — the direction tests above cannot catch that; this one can.
    """
    reg.add("a", "a.pt", elo=1500)
    reg.add("b", "b.pt", elo=1700)
    reg.update_result("a", "b", winner="a")
    assert reg.get("a").elo == 1524
    assert reg.get("b").elo == 1676


def test_elo_magnitudes_underdog_win(reg):
    """An underdog win must pay the upset premium (K*(1-E_underdog))."""
    reg.add("a", "a.pt", elo=1500)
    reg.add("b", "b.pt", elo=1700)
    reg.update_result("a", "b", winner="b")          # b (favorite) wins
    assert reg.get("a").elo == 1492                    # -8, not -24
    assert reg.get("b").elo == 1708

    reg.add("c", "c.pt", elo=1700)
    reg.update_result("a", "c", winner="a")          # a (underdog) wins
    # a is now 1492 vs c's 1700 -> E_a=0.2319 -> upset premium round(32*0.768)=+25
    assert reg.get("a").elo == 1492 + 25
    assert reg.get("c").elo == 1700 - 25


def test_elo_draw_unequal_ratings(reg):
    """A draw between UNEQUAL ratings shifts ELO toward the favorite's
    expectation (standard ELO: score 0.5 for both)."""
    reg.add("a", "a.pt", elo=1500)
    reg.add("b", "b.pt", elo=1700)
    reg.update_result("a", "b", winner=None)
    assert reg.get("a").elo == 1508                    # +8
    assert reg.get("b").elo == 1692                    # -8


def test_elo_convergence(reg):
    """A bot that wins repeatedly should climb in ELO; one that loses
    should drop. After 10 wins/losses, the gap should be significant."""
    reg.add("strong", "s.pt")
    reg.add("weak", "w.pt")
    for _ in range(10):
        reg.update_result("strong", "weak", winner="strong")
    assert reg.get("strong").elo > ELO_INITIAL + 50
    assert reg.get("weak").elo < ELO_INITIAL - 50


def test_standings(reg):
    reg.add("alice", "a.pt")
    reg.add("bob", "b.pt")
    reg.add("carol", "c.pt")
    reg.update_result("alice", "bob", winner="alice")
    reg.update_result("alice", "carol", winner="carol")
    s = reg.standings()
    assert len(s) == 3
    # alice won 1, lost 1 → ELO near initial
    # bob lost 1 → ELO below initial
    # carol won 1 → ELO above initial
    assert s[0]["name"] == "carol"  # highest ELO
    assert s[2]["name"] == "bob"    # lowest ELO


def test_persistence(tmp_path):
    """The roster survives a save + reload."""
    path = tmp_path / "bots.json"
    reg1 = BotRegistry(path=path)
    reg1.add("alice", "a.pt")
    reg1.add("bob", "b.pt")
    reg1.update_result("alice", "bob", winner="alice")

    reg2 = BotRegistry(path=path)
    assert len(reg2) == 2
    assert reg2.get("alice").wins == 1
    assert reg2.get("bob").losses == 1
    assert reg2.get("alice").elo == reg1.get("alice").elo


def test_len_and_contains(reg):
    reg.add("alice", "a.pt")
    assert len(reg) == 1
    assert "alice" in reg
    assert "bob" not in reg


def test_names(reg):
    reg.add("alice", "a.pt")
    reg.add("bob", "b.pt")
    assert set(reg.names()) == {"alice", "bob"}
