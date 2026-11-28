"""Two-clock league schedule — match the bot pairings to team matchups, 1:1.

The league evolves TWO populations:
  * **bots** (policies) — paired by ``league/pairing.py`` (N pairings/round,
    already implemented and pinned);
  * **teams** (lineups) — the living pool (``league/teams.py``), which needs
    its own clock: every round the pool must also produce N matchups (one per
    bot pairing), so battle ``i`` is "bot pairing ``i`` piloting team
    matchup ``i``".

Design (user-approved 2026-11-28, "balanced two-clock schedule"):
  * The matchup sequence is a **round-robin** over the current pool (circle
    method), so over time EVERY team meets EVERY other team — the mirror
    win-rate is evened out and the team ELO sees a clear signal.
  * The 1:1 index alignment **ROTATES each round** (start offset advances
    by ~one circle round per league round), so no bot pairing permanently
    pilots the same team: each team is piloted by several different bot
    pairings over a cull cycle → pilot confounding (bot strength entangled
    with one team's strength) cancels.
  * **Mirrors** (same team both sides, ``mirror_frac`` of the slots) are a
    pure bot-vs-bot signal: the teams cancel, and in ``TeamRegistry`` a
    mirror is ELO-NEUTRAL BY CONSTRUCTION — ``update_result(x, x, ...)``
    nets zero (pinned by ``test_schedule.py::test_mirror_is_elo_neutral``).
  * **ZERO extra battles**: the team clock reuses the existing N battle
    slots; it only decides WHICH teams those slots carry.
  * The pool is passed in per round, so cull/refresh changes are picked up
    automatically (the round-robin sequence is rebuilt when the pool key
    changes).

Pure + deterministic: no I/O, seeded rng, no torch. Thread-safety is not a
concern (the engine calls it once per round, single-threaded, before the
battle workers start).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

#: default mirror share of the round's slots (1 in 4 is a pure bot-vs-bot
#: signal; the other 3/4 carry the team signal)
DEFAULT_MIRROR_FRAC = 0.25

#: cache: pool key -> flattened round-robin pair sequence
_SEQ_CACHE: Dict[Tuple, List[Tuple[str, str]]] = {}


def base_species_key(species: str) -> str:
    """Conservative forme-stripped identity (same rule as league/teams.py)."""
    return species.split("-", 1)[0].strip()


def circle_rounds(tids: Sequence[str]) -> List[List[Tuple[str, str]]]:
    """Full round-robin via the circle method: ``len(tids) - 1`` rounds,
    each with ``len(tids) // 2`` distinct pairs; every pair meets exactly
    once over all rounds. Deterministic (input sorted).

    Odd pools get a BYE in one pair slot per round (the standard treatment;
    the bye team simply plays one fewer match per circle round)."""
    ids = sorted(tids)
    if len(ids) < 2:
        raise ValueError(f"circle_rounds needs >= 2 teams, got {len(ids)}")
    if len(ids) % 2:
        ids = ids + [None]      # even-out with a BYE placeholder (rotates)
    n = len(ids)
    half = n // 2
    rounds: List[List[Tuple[str, str]]] = []
    ring = list(ids)
    for _ in range(n - 1):
        pairs: List[Tuple[str, str]] = []
        for i in range(half):
            a, b = ring[i], ring[n - 1 - i]
            if a is None or b is None:
                continue         # the bye sits out THIS circle round only
            pairs.append((a, b) if a <= b else (b, a))
        rounds.append(pairs)
        # rotate: ring[0] stays fixed, the last element moves to position 1
        ring = [ring[0]] + [ring[-1]] + ring[1:-1]
    return rounds


def _seq_for(pool_key: Tuple[str, ...]) -> List[Tuple[str, str]]:
    """The pool's flattened round-robin sequence (cached by pool identity)."""
    if pool_key not in _SEQ_CACHE:
        seq: List[Tuple[str, str]] = []
        for rnd in circle_rounds(list(pool_key)):
            seq.extend(rnd)
        # a pair must be distinct (no self-pairs) — the circle method
        # guarantees it, but a 2-team pool degenerates to a 1-pair sequence
        _SEQ_CACHE[pool_key] = seq
    return _SEQ_CACHE[pool_key]


def team_matchups(pool_tids: Sequence[str], n_matchups: int,
                  round_idx: int = 0, seed: int = 0,
                  mirror_frac: float = DEFAULT_MIRROR_FRAC
                  ) -> List[Tuple[str, str]]:
    """Exactly ``n_matchups`` team matchups for league round ``round_idx``.

    * every non-mirror matchup is a DISTINCT pair from the pool's round-robin
      (no repeated pair within one round; over enough rounds every pair
      meets — the round-robin coverage);
    * ``mirror_frac`` of the slots are MIRRORS ``(t, t)`` — pure bot-vs-bot
      signal, ELO-neutral for the team registry;
    * the round-robin cursor ROTATES with ``round_idx`` (start offset =
      ``round_idx * (pool // 2)``), so the 1:1 alignment with the bot
      pairings shifts each round (pilot diversity);
    * deterministic under ``(seed, round_idx, pool, n, mirror_frac)``.

    Raises ``ValueError`` for a pool of < 2 teams (a league cannot schedule
    team signal without at least one pair)."""
    tids = sorted(set(pool_tids))
    if len(tids) < 2:
        raise ValueError(f"team_matchups needs >= 2 pool teams, got {len(tids)}")
    n = max(1, int(n_matchups))
    seq = _seq_for(tuple(tids))
    if not seq:
        raise ValueError(f"team_matchups: empty round-robin for pool {tids}")

    frac = max(0.0, min(1.0, float(mirror_frac)))
    n_mirrors = int(round(n * frac))
    n_pairs = max(0, n - n_mirrors)

    # rotation: the start offset walks the round-robin sequence by EXACTLY
    # the number of pairs this round consumes (stride = n_pairs), so the
    # walk is contiguous — no gaps, no overlaps. (An earlier version used
    # pool//2 as the stride, which left ~25% of pairs permanently uncovered
    # because mirror slots eat into the pair budget — pinned by
    # test_schedule.py::test_coverage_all_pairs_meet_over_rounds.)
    stride = max(1, n_pairs)
    start = (int(round_idx) * stride) % len(seq)

    rng = random.Random(int(seed) * 1000003 + int(round_idx))
    out: List[Tuple[str, str]] = []
    seen = set()
    i = 0
    while len(out) < n_pairs and i <= len(seq):
        pair = seq[(start + i) % len(seq)]
        i += 1
        if pair in seen:
            if i > len(seq):
                break  # whole sequence exhausted; mirrors fill the rest
            continue
        seen.add(pair)
        out.append(pair)
    while len(out) < n:
        out.append((rng.choice(tids),) * 2)   # mirror slot
    # break the index alignment (seeded): the circle method pins one team to
    # pair slot 0 every round — without a shuffle that team would be
    # PERMANENTLY piloted by the same bot pairing (exactly the pilot
    # confounding this schedule exists to cancel). The round-robin sequence
    # still guarantees coverage; the shuffle guarantees pilot diversity.
    rng.shuffle(out)
    return out


def pilot_indices(matchups: Sequence[Tuple[str, str]]) -> Dict[str, List[int]]:
    """tid -> the bot-pairing indices it will pilot in this round (a mirror
    team is listed once — one battle slot). Introspection / test helper."""
    out: Dict[str, List[int]] = {}
    for idx, (a, b) in enumerate(matchups):
        out.setdefault(a, []).append(idx)
        if b != a:
            out.setdefault(b, []).append(idx)
    return out


def clear_cache() -> None:
    """Drop the round-robin sequence cache (tests; pool churn is unbounded)."""
    _SEQ_CACHE.clear()
