"""A3.3b — the KO-aware move-ranking brain (``fitness.strategy``), hermetic.

Pinned with a FAKE calc (exact per-move pcts, no Node, no GPU, deterministic).
The brain is interface-AGNOSTIC (a pure function over the ``get_context()``
snapshot), so these tests hold regardless of the a/b/c wiring choice.

Run:  python -m pytest fitness/tests/test_strategy.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fitness import strategy as S  # noqa: E402


# ── a minimal get_context() snapshot (both species known -> no degradation) ──
def _ctx(own="charizard", opp="venusaur", opp_hp=0.5):
    return {
        "own": {"p1a": {"name": own, "hp_frac": 1.0, "ability": "", "item": "",
                        "statuses": [], "boosts": {}}},
        "opponent": {"p2a": {"name": opp, "hp_frac": opp_hp, "ability": "",
                             "item": "", "statuses": [], "boosts": {}}},
    }


def _dmg(min_pct, max_pct, exp_pct, immune=False):
    return {"min_pct": min_pct, "max_pct": max_pct, "expected_pct": exp_pct,
            "immune": immune}


def fake_calc(table):
    """A calc_fn that maps move-name -> (min,max,exp) pct (deterministic)."""
    calls = {"n": 0}

    def _calc(attacker, defender, move, fmt=None, gen=None):
        name = move if isinstance(move, str) else move.get("name")
        calls["n"] += 1
        m, x, e = table[name]
        return _dmg(m, x, e)

    return _calc, calls


# ── ko_ladder_score: the pure (tier, magnitude) mapping ──────────────────────
def test_ko_ladder_score_tiers():
    assert S.ko_ladder_score({"guaranteed_ko": True, "expected_pct": 90})[0] == 3
    assert S.ko_ladder_score({"can_ko": True, "expected_pct": 60})[0] == 2
    assert S.ko_ladder_score({"expected_ko": True, "expected_pct": 40})[0] == 1
    assert S.ko_ladder_score({"can_ko": False})[0] == 0
    assert S.ko_ladder_score(None) == (0, 0.0)  # degrades to bottom


def test_ko_ladder_score_within_tier_magnitude():
    # same tier (can_ko), higher expected damage sorts higher
    low = S.ko_ladder_score({"can_ko": True, "expected_pct": 50})
    high = S.ko_ladder_score({"can_ko": True, "expected_pct": 70})
    assert low[0] == high[0] == 2 and high[1] > low[1]


# ── rank_moves: the ordering the heuristic will consume ──────────────────────
def test_rank_prefers_ko_ladder_guaranteed_can_none():
    calc, _ = fake_calc({"hammer": (95, 100, 98),      # guaranteed KO (min>=50)
                         "ember": (45, 55, 50),        # can KO (max>=50, min<50)
                         "tackle": (10, 20, 15)})      # no KO
    r = S.rank_moves(_ctx(opp_hp=0.5), "p1a", "p2a",
                     ["tackle", "ember", "hammer"], fmt="gen9ou", calc_fn=calc)
    assert r["order"][0] == "hammer" and r["best"] == "hammer"
    assert r["order"] == ["hammer", "ember", "tackle"]


def test_rank_within_tier_higher_damage_wins():
    # both "can KO" (max>=50, min<50): the bigger expected damage ranks first
    calc, _ = fake_calc({"a": (40, 60, 55), "b": (40, 52, 48)})
    r = S.rank_moves(_ctx(opp_hp=0.5), "p1a", "p2a", ["a", "b"],
                     fmt="gen9ou", calc_fn=calc)
    assert r["order"] == ["a", "b"]


def test_rank_status_move_sorts_below_damaging_move():
    # a status move deals 0 (no KO, no damage) -> below a non-KO damaging move
    calc, _ = fake_calc({"wish": (0, 0, 0), "ember": (10, 20, 15)})
    r = S.rank_moves(_ctx(opp_hp=0.5), "p1a", "p2a", ["wish", "ember"],
                     fmt="gen9ou", calc_fn=calc)
    assert r["order"][0] == "ember" and r["order"][-1] == "wish"


def test_rank_degrades_stably_when_opponent_unknown():
    # opponent species unknown -> the KO core degrades (all tier 0); the sort is
    # STABLE (original order) and NEVER raises. The fake calc must not be called
    # for an unknown defender (the KO core short-circuits before the calc).
    def boom(*a, **k):
        raise AssertionError("calc must not run when the defender is unknown")
    ctx = {"own": {"p1a": {"name": "charizard", "hp_frac": 1.0}},
           "opponent": {"p2a": {}}}  # no species
    r = S.rank_moves(ctx, "p1a", "p2a", ["ember", "tackle"],
                     fmt="gen9ou", calc_fn=boom)
    assert r["order"] == ["ember", "tackle"]  # stable, original order
    assert r["best"] == "ember"
    assert r["assessments"]["ember"]["unknown_defender"] is True


def test_rank_empty_moves():
    r = S.rank_moves(_ctx(), "p1a", "p2a", [], fmt="gen9ou",
                     calc_fn=lambda *a, **k: _dmg(0, 0, 0))
    assert r == {"order": [], "best": None, "assessments": {}}


def test_rank_memoises_one_calc_per_distinct_move():
    # each distinct move is calc'd exactly once, even if it appears twice
    calc, calls = fake_calc({"ember": (10, 20, 15), "tackle": (5, 10, 7)})
    S.rank_moves(_ctx(opp_hp=0.5), "p1a", "p2a",
                 ["ember", "tackle", "ember"], fmt="gen9ou", calc_fn=calc)
    assert calls["n"] == 2  # ember once + tackle once (not 3)


def test_rank_accepts_dict_move_specs():
    calc, _ = fake_calc({"hammer": (95, 100, 98), "tackle": (10, 20, 15)})
    r = S.rank_moves(_ctx(opp_hp=0.5), "p1a", "p2a",
                     [{"name": "tackle"}, {"name": "hammer"}],
                     fmt="gen9ou", calc_fn=calc)
    assert r["order"][0] == {"name": "hammer"} and r["best"] == {"name": "hammer"}


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
