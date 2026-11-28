"""Two-clock schedule tests (league/schedule.py).

Pins the balance properties the approved design relies on:
  * circle-round correctness (every pair meets exactly once, even + odd pools)
  * per-round validity: exactly N matchups, pool-only teams, no duplicate
    non-mirror pair, mirrors are (t, t)
  * round-robin COVERAGE: over enough rounds every pair of teams has met
  * PILOT DIVERSITY: no team is permanently piloted by the same bot pairing
    (the circle-method slot-0 pinning is broken by the per-round shuffle)
  * determinism under (seed, round_idx, pool, n)
  * pool churn (cull / refresh) is picked up per round
  * mirrors are ELO-NEUTRAL in TeamRegistry (x vs x nets zero)

Run:  python -m pytest league/tests/test_schedule.py -k "not live" -q
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.schedule import (
    circle_rounds,
    clear_cache,
    pilot_indices,
    team_matchups,
)
from league.teams import TeamRegistry

POOL8 = [f"t{i}" for i in range(8)]
POOL_ODD = [f"o{i}" for i in range(7)]


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield
    clear_cache()


# ═══════════════════════════════════════════════════════════════════════════
#  circle rounds
# ═══════════════════════════════════════════════════════════════════════════

def test_circle_rounds_even_full_round_robin():
    rounds = circle_rounds(POOL8)
    assert len(rounds) == 7                       # N-1 circle rounds
    assert all(len(r) == 4 for r in rounds)       # N/2 pairs per round
    seen = set()
    for r in rounds:
        for a, b in r:
            assert a != b and a in POOL8 and b in POOL8
            key = (a, b) if a < b else (b, a)
            assert key not in seen                # each pair exactly once
            seen.add(key)
    assert len(seen) == 8 * 7 // 2                # C(8,2)


def test_circle_rounds_odd_bye_treated():
    rounds = circle_rounds(POOL_ODD)
    assert len(rounds) == 7                       # N-1 rounds (N=7)
    all_pairs = set()
    for r in rounds:
        for a, b in r:
            key = (a, b) if a < b else (b, a)
            assert key not in all_pairs
            all_pairs.add(key)
    assert all_pairs == set(itertools.combinations(POOL_ODD, 2))
    # 7 teams -> 7 circle rounds (8 slots - 1); each team sits out exactly
    # once (bye), so plays 6 of 7 rounds
    appearances = {t: sum(1 for r in rounds for a, b in r if t in (a, b))
                   for t in POOL_ODD}
    assert all(v == 6 for v in appearances.values()), appearances


def test_circle_rounds_too_small():
    with pytest.raises(ValueError, match=">= 2"):
        circle_rounds(["solo"])
    with pytest.raises(ValueError, match=">= 2"):
        circle_rounds([])


# ═══════════════════════════════════════════════════════════════════════════
#  per-round validity
# ═══════════════════════════════════════════════════════════════════════════

def test_exactly_n_matchups():
    out = team_matchups(POOL8, n_matchups=4, round_idx=0, seed=1)
    assert len(out) == 4
    out = team_matchups(POOL8, n_matchups=28, round_idx=3, seed=1)  # full league
    assert len(out) == 28


def test_matchups_use_pool_teams_only():
    out = team_matchups(POOL8, n_matchups=4, round_idx=0, seed=1)
    for a, b in out:
        assert a in POOL8 and b in POOL8


def test_non_mirror_pairs_distinct_within_round():
    out = team_matchups(POOL8, n_matchups=4, round_idx=0, seed=1,
                        mirror_frac=0.25)
    non_mirror = [(a, b) for a, b in out if a != b]
    keys = {(a, b) if a < b else (b, a) for a, b in non_mirror}
    assert len(keys) == len(non_mirror)


def test_no_pair_repeats_within_one_round_even_at_high_n():
    # n = 12 > C(8,2)/2: the round still never repeats a non-mirror pair
    out = team_matchups(POOL8, n_matchups=12, round_idx=0, seed=1)
    non_mirror = [(a, b) for a, b in out if a != b]
    keys = {(a, b) if a < b else (b, a) for a, b in non_mirror}
    assert len(keys) == len(non_mirror)
    assert len(keys) <= 28


def test_mirrors_are_same_team_both_sides():
    out = team_matchups(POOL8, n_matchups=8, round_idx=0, seed=1,
                        mirror_frac=0.5)
    mirrors = [(a, b) for a, b in out if a == b]
    assert mirrors, "expected some mirrors at mirror_frac=0.5"
    assert all(a == b and a in POOL8 for a, b in mirrors)


def test_mirror_frac_zero_and_one():
    out = team_matchups(POOL8, n_matchups=4, round_idx=0, seed=1,
                        mirror_frac=0.0)
    assert all(a != b for a, b in out)
    out = team_matchups(POOL8, n_matchups=4, round_idx=0, seed=1,
                        mirror_frac=1.0)
    assert all(a == b for a, b in out)


def test_pool_too_small_raises():
    with pytest.raises(ValueError, match=">= 2"):
        team_matchups(["solo"], n_matchups=1)


# ═══════════════════════════════════════════════════════════════════════════
#  round-robin coverage (the clear-signal property)
# ═══════════════════════════════════════════════════════════════════════════

def test_coverage_all_pairs_meet_over_rounds():
    """Over a full walk of the round-robin sequence, EVERY pair of teams has
    appeared as a non-mirror matchup (the mirror win-rate is evened out)."""
    pool = POOL8
    # n=4 @ mirror_frac 0.25 -> 3 pairs + 1 mirror per round; stride=3, so
    # ceil(28/3) = 10 rounds walk the whole sequence; 12 for margin
    n_rounds = 12
    met = set()
    for r in range(n_rounds):
        for a, b in team_matchups(pool, n_matchups=4, round_idx=r, seed=7):
            if a != b:
                met.add((a, b) if a < b else (b, a))
    assert met == set(itertools.combinations(pool, 2))


def test_coverage_odd_pool():
    pool = POOL_ODD
    met = set()
    for r in range(24):
        for a, b in team_matchups(pool, n_matchups=3, round_idx=r, seed=7):
            if a != b:
                met.add((a, b) if a < b else (b, a))
    assert met == set(itertools.combinations(pool, 2))


# ═══════════════════════════════════════════════════════════════════════════
#  pilot diversity (the confounding-cancellation property)
# ═══════════════════════════════════════════════════════════════════════════

def test_no_team_permanently_piloted_by_one_pairing():
    """Over several rounds, every team must be piloted at >= 2 different
    bot-pairing indices — a fixed team->pairing binding is the exact
    confounding the two-clock design cancels (the circle method's slot-0
    pinning is what the per-round shuffle breaks)."""
    pool = POOL8
    n_rounds = 8
    idx_sets = {t: set() for t in pool}
    for r in range(n_rounds):
        out = team_matchups(pool, n_matchups=4, round_idx=r, seed=11)
        pi = pilot_indices(out)
        for t, idxs in pi.items():
            idx_sets[t].update(idxs)
    stuck = {t: v for t, v in idx_sets.items() if len(v) < 2}
    assert not stuck, f"teams stuck to one bot pairing: {stuck}"


def test_pilot_indices_helper():
    out = team_matchups(POOL8, n_matchups=4, round_idx=0, seed=1)
    pi = pilot_indices(out)
    # every non-mirror team is listed; a mirror team listed once (one slot)
    for a, b in out:
        assert a in pi
        if a != b:
            assert b in pi
    total_slots = sum(len(v) for v in pi.values())
    n_mirrors = sum(1 for a, b in out if a == b)
    assert total_slots == 2 * len(out) - n_mirrors


# ═══════════════════════════════════════════════════════════════════════════
#  determinism + pool churn
# ═══════════════════════════════════════════════════════════════════════════

def test_deterministic_under_seed_and_round():
    a = team_matchups(POOL8, n_matchups=4, round_idx=5, seed=42)
    b = team_matchups(POOL8, n_matchups=4, round_idx=5, seed=42)
    assert a == b
    # different seed -> a different rng stream (mirrors / shuffle differ)
    d = team_matchups(POOL8, n_matchups=4, round_idx=5, seed=43)
    assert d == team_matchups(POOL8, n_matchups=4, round_idx=5, seed=43)


def test_pool_churn_cull_and_refresh():
    pool = list(POOL8)
    # cull: drop two teams -> the schedule must use ONLY the current pool
    pool = [t for t in pool if t not in ("t0", "t1")]
    out_culled = team_matchups(pool, n_matchups=3, round_idx=1, seed=1)
    for a, b in out_culled:
        assert a in pool and b in pool
    # refresh: a NEW team joins -> it becomes schedulable (and legal)
    pool = pool + ["fresh_01"]
    out_fresh = team_matchups(pool, n_matchups=4, round_idx=2, seed=1)
    for a, b in out_fresh:
        assert a in pool and b in pool


def test_n_matchups_beyond_pair_count_fills_mirrors():
    """n > C(8,2) is degenerate but must terminate (mirrors fill the rest)."""
    out = team_matchups(POOL8, n_matchups=30, round_idx=0, seed=1)
    assert len(out) == 30
    non_mirror = [(a, b) for a, b in out if a != b]
    keys = {(a, b) if a < b else (b, a) for a, b in non_mirror}
    assert len(keys) <= 28 and len(keys) == len(non_mirror)


# ═══════════════════════════════════════════════════════════════════════════
#  mirror ELO-neutrality (the property that makes mirrors safe to schedule)
# ═══════════════════════════════════════════════════════════════════════════

def test_mirror_is_elo_neutral(tmp_path):
    """A mirror battle (team x piloted by bot A vs team x piloted by bot B)
    must be ELO-NEUTRAL for the team registry: x vs x nets zero in
    TeamRegistry.update_result (win and loss both land on x)."""
    reg = TeamRegistry(path=tmp_path / "teams.json")
    from league.teams import structured_build, load_sets, ELO_INITIAL
    mons = structured_build(load_sets("gen9ou"), __import__("random").Random(1))
    reg.add("x", mons)
    elo0 = reg.get("x").elo
    reg.update_result("x", "x", winner="x")
    reg.update_result("x", "x", winner=None)
    assert reg.get("x").elo == elo0 == ELO_INITIAL
    # the counters both move (W and L) — win_rate dilutes toward 0.5,
    # but the RATING, which is what ELO selection uses, is untouched
    assert reg.get("x").wins == 1 and reg.get("x").losses == 1
