"""Step 3.4 — per-OWN-MON MATCHUP summary gate tests (PLAN.md §3c, review 2026-11-22).

The net needed the BOARD-LEVEL damage landscape ("how does each of my mons fare
against the whole enemy team?") for win-condition / switch / threat planning.
These tests pin the 8-dim per-own-mon summary produced by
``net.matchup_features.build_own_matchup`` with an INJECTED batch calc_fn (hermetic
— no Node), plus ONE real-calc/DB integration test (skips if Node is unavailable).

The interpretation locked (OPEN_DESIGN_CHOICES.md "3.4 DESIGN"):
  [0] best_hit       max over enemy mons of (own strongest move exp% / 100)
  [1] worst_hit      max over enemy mons of (opp top move exp% / 100)
  [2] ko_guar_frac   # enemy mons the own mon KOs guaranteed (min% >= their HP) / n
  [3] ko_can_frac    # enemy mons the own mon CAN KO (max% >= their HP) / n
  [4] taken_ko_frac  # enemy mons that CAN KO the own mon (opp max% >= own HP) / n
  [5] survives_strongest  1.0 iff worst_hit < 1.0
  [6] net_ko_margin  ko_can_frac - taken_ko_frac   (clipped [-1,1])
  [7] best_vs_worst  best_hit - worst_hit           (clipped [-1,1])
"""
from __future__ import annotations

import numpy as np
import pytest

from net.matchup_features import (
    OWN_MATCHUP_NAMES,
    build_own_matchup,
    own_attack_move,
    opp_attack_move,
    NEUTRAL_DAMAGING_MOVE,
)
from net.registry import K_OWN_MATCHUP, MAX_SLOTS


def _ctx(own=None, opp=None) -> dict:
    """A minimal get_context() dict for the matchup module (own p1a, opp p2a/p2b)."""
    return {
        "own": {"p1a": own} if own else {},
        "opponent": opp or {},
    }


def _fake_calc(table, ability_table=None):
    """A ``calc_damage_many(items)``-shaped fake.

    ``table`` maps ``(attacker_name, defender_name, move) -> (min, max, exp)``
    (pct of defender max HP). Any unlisted item -> a zero-damage dict (no KO).
    ``ability_table`` (optional) maps ``defender_name -> ability`` the fake saw
    on that defender, so a test can assert the top-1 posterior ability override
    reached the calc.
    """
    ability_table = ability_table if ability_table is not None else {}

    def fake(items):
        for it in items:
            ability_table.setdefault(
                it["defender"]["name"], it["defender"].get("ability", ""))
        out = []
        for it in items:
            key = (it["attacker"]["name"], it["defender"]["name"], it["move"])
            dmg = table.get(key)
            if dmg is None:
                out.append({"min_pct": 0.0, "max_pct": 0.0, "expected_pct": 0.0,
                            "immune": False})
            else:
                mn, mx, ex = dmg
                out.append({"min_pct": mn, "max_pct": mx, "expected_pct": ex,
                            "immune": False})
        return out
    return fake


def _ctx_full():
    """garchomp (own) vs blastoise(70%)/gengar(80%) — deterministic KO facts."""
    return {
        "own": {"p1a": {
            "name": "garchomp", "hp_frac": 1.0,
            "moves": ["earthquake", "waterpulse", "dragonclaw", "outrage"],
        }},
        "opponent": {
            "p2a": {
                "name": "blastoise", "hp_frac": 0.7, "moves": ["surf"],
                "top_moves": ["surf"],
                "posterior": [{"ability": "rapidswap", "weight": 0.9}],
            },
            "p2b": {
                "name": "gengar", "hp_frac": 0.8, "moves": ["shadowball"],
                "top_moves": ["shadowball"],
                "posterior": [{"ability": "levitate", "weight": 1.0}],
            },
        },
    }


# ── shape / field contract ─────────────────────────────────────────────
def test_output_shape_and_field():
    out = build_own_matchup(_ctx_full(), "gen9bss3v3", calc_fn=_fake_calc({}))
    assert set(out) == set(OWN_MATCHUP_NAMES) == {"own_matchup"}
    v = out["own_matchup"]
    assert v.shape == (MAX_SLOTS, K_OWN_MATCHUP)
    assert v.dtype == np.float32


# ── best_hit / worst_hit (max over enemy mons) ─────────────────────────
def test_best_hit_is_max_expected_pct():
    table = {
        # garchomp's strongest move (outrage, 120 power) vs the two enemies:
        ("garchomp", "blastoise", "outrage"): (70.0, 80.0, 80.0),
        ("garchomp", "gengar", "outrage"): (30.0, 35.0, 35.0),
    }
    out = build_own_matchup(_ctx_full(), "gen9bss3v3", calc_fn=_fake_calc(table))
    v = out["own_matchup"][0]
    assert v[0] == pytest.approx(0.80)   # best_hit = max(0.80, 0.35)


def test_worst_hit_is_max_incoming_expected_pct():
    table = {
        # the two enemies' top moves vs garchomp:
        ("blastoise", "garchomp", "surf"): (40.0, 50.0, 50.0),
        ("gengar", "garchomp", "shadowball"): (60.0, 70.0, 70.0),
    }
    out = build_own_matchup(_ctx_full(), "gen9bss3v3", calc_fn=_fake_calc(table))
    v = out["own_matchup"][0]
    assert v[1] == pytest.approx(0.70)   # worst_hit = max(0.50, 0.70)


# ── KO verdicts use the DEFENDER's current HP fraction ─────────────────
def test_guaranteed_ko_counts_min_vs_defender_hp():
    # blastoise at 70%: min 80% >= 70% -> guaranteed KO (counts).
    # gengar at 80%: min 30% < 80% -> not guaranteed (does not count).
    table = {
        ("garchomp", "blastoise", "outrage"): (80.0, 90.0, 85.0),
        ("garchomp", "gengar", "outrage"): (30.0, 35.0, 35.0),
    }
    out = build_own_matchup(_ctx_full(), "gen9bss3v3", calc_fn=_fake_calc(table))
    v = out["own_matchup"][0]
    assert v[2] == pytest.approx(1.0 / 2.0)   # 1 of 2 enemy mons guaranteed-KO'd


def test_can_ko_uses_max_pct_not_min():
    # blastoise at 70%: min 60% < 70% (not guaranteed) but max 75% >= 70% -> CAN KO.
    # gengar at 80%: max 35% < 80% -> cannot.
    table = {
        ("garchomp", "blastoise", "outrage"): (60.0, 75.0, 70.0),
        ("garchomp", "gengar", "outrage"): (30.0, 35.0, 35.0),
    }
    out = build_own_matchup(_ctx_full(), "gen9bss3v3", calc_fn=_fake_calc(table))
    v = out["own_matchup"][0]
    assert v[2] == pytest.approx(0.0)        # no guaranteed KO
    assert v[3] == pytest.approx(1.0 / 2.0)  # 1 of 2 CAN be KO'd


def test_taken_ko_uses_own_hp_fraction():
    # garchomp at 100%: surf max 50% < 100% (no), shadowball max 70% < 100% (no).
    # garchomp at 60%: surf max 50% < 60% (no), shadowball max 70% >= 60% -> taken.
    ctx = _ctx_full()
    ctx["own"]["p1a"]["hp_frac"] = 0.6
    table = {
        ("blastoise", "garchomp", "surf"): (40.0, 50.0, 50.0),
        ("gengar", "garchomp", "shadowball"): (60.0, 70.0, 70.0),
    }
    out = build_own_matchup(ctx, "gen9bss3v3", calc_fn=_fake_calc(table))
    v = out["own_matchup"][0]
    assert v[4] == pytest.approx(1.0 / 2.0)  # 1 of 2 enemy mons CAN KO garchomp


# ── survives / net margin / best-vs-worst ──────────────────────────────
def test_survives_strongest_flag():
    # worst_hit = 0.70 < 1.0 -> survives (1.0); worst_hit = 1.2 -> does not (0.0).
    t_survive = {("gengar", "garchomp", "shadowball"): (60.0, 70.0, 70.0),
                 ("blastoise", "garchomp", "surf"): (40.0, 50.0, 50.0)}
    out = build_own_matchup(_ctx_full(), "gen9bss3v3", calc_fn=_fake_calc(t_survive))
    assert out["own_matchup"][0][5] == pytest.approx(1.0)
    t_doomed = {("gengar", "garchomp", "shadowball"): (110.0, 130.0, 120.0)}
    out2 = build_own_matchup(_ctx_full(), "gen9bss3v3", calc_fn=_fake_calc(t_doomed))
    assert out2["own_matchup"][0][5] == pytest.approx(0.0)


def test_net_ko_margin_and_best_vs_worst():
    # garchomp: can-KO 1/2, taken-KO 0 -> net +0.5; best 0.80, worst 0.70 -> +0.10.
    # (worst_hit = max over enemies of their top-move exp% = max(0.50, 0.70) = 0.70)
    table = {
        ("garchomp", "blastoise", "outrage"): (80.0, 90.0, 80.0),
        ("garchomp", "gengar", "outrage"): (30.0, 35.0, 35.0),
        ("blastoise", "garchomp", "surf"): (40.0, 50.0, 50.0),
        ("gengar", "garchomp", "shadowball"): (60.0, 70.0, 70.0),
    }
    out = build_own_matchup(_ctx_full(), "gen9bss3v3", calc_fn=_fake_calc(table))
    v = out["own_matchup"][0]
    assert v[0] == pytest.approx(0.80)       # best_hit = max(0.80, 0.35)
    assert v[1] == pytest.approx(0.70)       # worst_hit = max(0.50, 0.70)
    assert v[3] == pytest.approx(0.5)        # can-KO frac
    assert v[4] == pytest.approx(0.0)        # taken-KO frac
    assert v[6] == pytest.approx(0.5)        # net margin = 0.5 - 0.0
    assert v[7] == pytest.approx(0.10)       # best - worst = 0.80 - 0.70


# ── degradation (never raises; zero rows) ──────────────────────────────
def test_unknown_species_degrades_to_zero():
    ctx = {"own": {"p1a": {"name": "garchomp", "hp_frac": 1.0,
                           "moves": ["waterpulse"]}},
           "opponent": {"p2a": {"name": "", "hp_frac": 1.0,
                                "top_moves": ["surf"]}}}
    out = build_own_matchup(ctx, "gen9bss3v3", calc_fn=_fake_calc({}))
    assert np.all(out["own_matchup"] == 0.0)


def test_calc_error_degrades_to_zero():
    def boom(items):
        raise RuntimeError("calc down")
    out = build_own_matchup(_ctx_full(), "gen9bss3v3", calc_fn=boom)
    assert np.all(out["own_matchup"] == 0.0)


def test_bad_batch_length_degrades_to_zero():
    def short(items):
        return []   # wrong length -> the module must not trust it
    out = build_own_matchup(_ctx_full(), "gen9bss3v3", calc_fn=short)
    assert np.all(out["own_matchup"] == 0.0)


# ── move/ability selection (the deterministic picks) ───────────────────
def test_opp_top1_posterior_ability_reaches_calc():
    seen = {}
    table = {("garchomp", "blastoise", "outrage"): (0.0, 0.0, 0.0)}
    out = build_own_matchup(_ctx_full(), "gen9bss3v3",
                            calc_fn=_fake_calc(table, ability_table=seen))
    # the opp's top-1 posterior ability (rapidswap) must be the override the
    # calc saw on that defender.
    assert seen.get("blastoise") == "rapidswap"


def test_opp_attack_move_falls_back_to_first_observed():
    blk = {"name": "gengar", "top_moves": [], "moves": ["shadowball", "thunderbolt"]}
    assert opp_attack_move(blk) == "shadowball"
    blk2 = {"name": "gengar", "top_moves": ["focusbolt", "shadowball"]}
    assert opp_attack_move(blk2) == "focusbolt"


def test_turn1_opp_no_revealed_move_uses_neutral_baseline():
    # opponent revealed NO move (top_moves empty, moves empty) but its species is
    # known -> the module must still produce a signal (Body Slam baseline), not
    # silently degrade to zero (the "turn-1 matchup unknown" edge).
    ctx = {"own": {"p1a": {"name": "garchomp", "hp_frac": 1.0,
                           "moves": ["earthquake"]}},
           "opponent": {"p2a": {"name": "blastoise", "hp_frac": 1.0,
                                "top_moves": [], "moves": []}}}
    table = {("garchomp", "blastoise", "earthquake"): (0.0, 0.0, 0.0),
             ("blastoise", "garchomp", NEUTRAL_DAMAGING_MOVE): (40.0, 45.0, 45.0)}
    out = build_own_matchup(ctx, "gen9bss3v3", calc_fn=_fake_calc(table))
    v = out["own_matchup"][0]
    # the opponent's neutral-baseline move (45% exp) is the worst_hit -> nonzero
    assert v[1] == pytest.approx(0.45)


# ── one real-calc / DB integration test (skips without Node) ───────────
def test_real_calc_shape_and_bounds():
    try:
        from damage.calc import calc_damage_many  # noqa: F401
    except Exception:
        pytest.skip("damage calc unavailable (no Node runtime)")
    out = build_own_matchup(_ctx_full(), "gen9bss3v3")   # real calc, no injection
    v = out["own_matchup"]
    assert v.shape == (MAX_SLOTS, K_OWN_MATCHUP)
    assert np.all(np.isfinite(v))
    # all dims must be bounded (the net relies on bounded inputs): pcts/100,
    # /6 fractions, flags -> [0,1]; net margin / best-vs-worst -> [-1,1].
    for d in range(6):
        assert (v[:, d] >= 0.0).all() and (v[:, d] <= 1.0).all(), f"dim {d}"
    assert (v[:, 6] >= -1.0).all() and (v[:, 6] <= 1.0).all()
    assert (v[:, 7] >= -1.0).all() and (v[:, 7] <= 1.0).all()
