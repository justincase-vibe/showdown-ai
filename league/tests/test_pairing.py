"""League pairing tests — round_robin, elo_weighted, best_of_n."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.pairing import round_robin, elo_weighted, best_of_n
from league.registry import Bot


def _bots(*names: str, **elo_over: int) -> list:
    """Create a list of bots with the given names (optional ELO overrides)."""
    bots = []
    for i, name in enumerate(names):
        elo = elo_over.get(name, 1500 + i * 100)
        bots.append(Bot(name=name, ckpt_path=f"{name}.pt", elo=elo))
    return bots


def test_round_robin_two_bots():
    bots = _bots("a", "b")
    pairs = round_robin(bots)
    assert pairs == [("a", "b")]


def test_round_robin_three_bots():
    bots = _bots("a", "b", "c")
    pairs = round_robin(bots)
    assert len(pairs) == 3  # 3*2/2 = 3
    assert set(pairs) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_round_robin_four_bots():
    bots = _bots("a", "b", "c", "d")
    pairs = round_robin(bots)
    assert len(pairs) == 6  # 4*3/2 = 6
    # every pair appears exactly once
    assert len(set(pairs)) == 6


def test_round_robin_deterministic_order():
    """Pairs are sorted by name (reproducible)."""
    bots = _bots("z", "a", "m")
    pairs = round_robin(bots)
    assert pairs[0][0] < pairs[0][1]  # first pair is alphabetical
    # all pairs have a < b
    for a, b in pairs:
        assert a < b


def test_elo_weighted_two_bots():
    bots = _bots("a", "b")
    pairs = elo_weighted(bots)
    assert len(pairs) == 1
    # higher ELO bot is listed first
    assert pairs[0] == ("b", "a")  # b has higher ELO (1600 vs 1500)


def test_elo_weighted_four_bots():
    bots = _bots("a", "b", "c", "d")  # ELO: 1500, 1600, 1700, 1800
    pairs = elo_weighted(bots)
    assert len(pairs) == 2  # 4//2 = 2
    # top (d=1800) vs bottom (a=1500), second (c=1700) vs third (b=1600)
    assert pairs[0] == ("d", "a")
    assert pairs[1] == ("c", "b")


def test_elo_weighted_odd_bots():
    """3 bots → 1 pair (the middle one gets a bye)."""
    bots = _bots("a", "b", "c")  # ELO: 1500, 1600, 1700
    pairs = elo_weighted(bots)
    assert len(pairs) == 1
    # top (c=1700) vs bottom (a=1500)
    assert pairs[0] == ("c", "a")


def test_elo_weighted_single_bot():
    bots = _bots("a")
    assert elo_weighted(bots) == []


def test_best_of_n():
    bots = _bots("a", "b", "c")
    pairs = best_of_n(bots, n=3)
    assert len(pairs) == 9  # 3 pairs * 3 rounds
    # each base pair appears 3 times
    from collections import Counter
    counts = Counter(pairs)
    assert counts[("a", "b")] == 3
    assert counts[("a", "c")] == 3
    assert counts[("b", "c")] == 3
