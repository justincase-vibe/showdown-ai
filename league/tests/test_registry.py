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
