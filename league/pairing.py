"""Pairing — decide which bots fight in the league.

Two strategies (both pure functions, no I/O):

  * **round_robin** — every bot plays every other bot exactly once per round.
    The simplest, most fair strategy. Good for small leagues (2-8 bots).

  * **elo_weighted** — pair the highest-rated bot with the lowest-rated,
    the second-highest with the second-lowest, etc. (like a Swiss system).
    Good for larger leagues where you want competitive matches.

Both return a list of ``(bot_a_name, bot_b_name)`` pairs.
"""
from __future__ import annotations

from typing import List, Tuple

from .registry import Bot


def round_robin(bots: List[Bot]) -> List[Tuple[str, str]]:
    """Every bot plays every other bot exactly once.

    For N bots, this produces N*(N-1)/2 pairs.
    Order is deterministic (sorted by name) so tests are reproducible.
    """
    names = sorted(b.name for b in bots)
    pairs: List[Tuple[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))
    return pairs


def elo_weighted(bots: List[Bot]) -> List[Tuple[str, str]]:
    """Pair highest with lowest, second-highest with second-lowest, etc.

    Like a Swiss system: the top seed plays the bottom seed, the second
    seed plays the second-from-bottom, etc. For an odd number of bots,
    one bot gets a bye (the middle one).

    Returns at most len(bots) // 2 pairs.
    """
    if len(bots) < 2:
        return []
    sorted_bots = sorted(bots, key=lambda b: -b.elo)
    pairs: List[Tuple[str, str]] = []
    n = len(sorted_bots)
    for i in range(n // 2):
        a = sorted_bots[i]
        b = sorted_bots[n - 1 - i]
        if a.name != b.name:  # guard against self-pairing (odd N)
            pairs.append((a.name, b.name))
    return pairs


def best_of_n(bots: List[Bot], n: int = 1) -> List[Tuple[str, str]]:
    """Play each pairing ``n`` times (best-of-n).

    Returns ``n * len(pairs)`` pairs (each pairing repeated n times).
    """
    base = round_robin(bots)
    pairs: List[Tuple[str, str]] = []
    for _ in range(max(1, n)):
        pairs.extend(base)
    return pairs
