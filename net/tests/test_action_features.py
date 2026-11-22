"""Step 3.3a tests — the per-ACTION damage/KO/speed block (move/tera rows).

Hermetic: the damage calc and speed comparison are INJECTED as fakes (the
``calc_fn`` / ``speed_fn`` contract), so no Node subprocess is spawned and the
interpretation (KO verdicts, outspeed mapping, degradation) is pinned by logic,
not by the calc. Only a couple of the broader suite's real-calc checks touch
Node.
"""
from __future__ import annotations

import numpy as np
import pytest

from net.action_features import (
    ACTION_DAMAGE_NAMES,
    action_damage_row,
    build_action_damage,
)
from net.registry import K_ACTION_DAMAGE, MAX_ACTIONS


def _ctx(opp_hp: float = 1.0):
    return {
        "active": {"p1": "p1a", "p2": "p2a"},
        "own": {"p1a": {"name": "garchomp", "hp_frac": 1.0, "ability": "roughskin"}},
        "opponent": {"p2a": {"species": "blastoise", "hp_frac": opp_hp,
                             "posterior": [{"ability": "torrent", "weight": 0.9}]}},
    }


def _actions():
    return [
        {"kind": "move", "move": "bigmove"},
        {"kind": "move", "move": "smallmove"},
        {"kind": "tera", "move": "bigmove"},
        {"kind": "switch", "species": "gengar"},
    ]


def fake_calc(attacker, defender, move, fmt=None, gen=9):
    """A deterministic stand-in for ``damage.calc.calc_damage`` (same contract)."""
    name = move if isinstance(move, str) else (move or {}).get("name", "")
    table = {
        "bigmove": (60, 70, 65, False),   # 60-70% -> KOs a 50%-hp defender
        "smallmove": (10, 30, 20, False),  # 10-30% -> no KO at 50%
        "immov": (0, 0, 0, True),           # immune
    }
    mn, mx, ex, imm = table.get(name, (0, 0, 0, False))
    return {"min_pct": mn, "max_pct": mx, "expected_pct": ex, "immune": imm}


def test_output_shape_and_field():
    out = build_action_damage(_ctx(), _actions(), fmt="gen9bss3v3",
                              calc_fn=fake_calc, speed_fn=lambda a, d: "a")
    assert set(out) == set(ACTION_DAMAGE_NAMES) == {"action_damage"}
    assert out["action_damage"].shape == (MAX_ACTIONS, K_ACTION_DAMAGE)
    assert out["action_damage"].dtype == np.float32


def test_ko_verdicts_and_damage_pcts():
    out = build_action_damage(_ctx(opp_hp=0.5), _actions(), fmt="gen9bss3v3",
                              calc_fn=fake_calc, speed_fn=lambda a, d: "a")
    v = out["action_damage"]
    big, small = v[0], v[1]
    # bigmove (60-70% vs a 50%-hp defender) -> guaranteed / can / expected KO
    assert big[3] == 1.0 and big[4] == 1.0 and big[5] == 1.0
    # smallmove (10-30% vs 50%) -> NO KO
    assert small[3] == 0.0 and small[4] == 0.0 and small[5] == 0.0
    # damage pcts land in the right dims (/100)
    assert abs(big[0] - 0.60) < 1e-6 and abs(big[1] - 0.70) < 1e-6
    assert abs(big[2] - 0.65) < 1e-6


def test_immune_flag_never_kos():
    ctx = _ctx()
    out = build_action_damage(ctx, [{"kind": "move", "move": "immov"}],
                              fmt="gen9bss3v3", calc_fn=fake_calc,
                              speed_fn=lambda a, d: "a")
    v = out["action_damage"][0]
    assert v[7] == 1.0                 # immune
    assert v[3] == 0.0 and v[4] == 0.0 and v[5] == 0.0  # immune can never KO


def test_tera_action_carries_move_damage():
    out = build_action_damage(_ctx(opp_hp=0.5), _actions(), fmt="gen9bss3v3",
                              calc_fn=fake_calc, speed_fn=lambda a, d: "a")
    v = out["action_damage"]
    # a tera action re-uses the same move id -> the SAME damage facts
    assert np.allclose(v[0], v[2])


def test_outspeed_mapping():
    base = dict(fmt="gen9bss3v3", calc_fn=fake_calc)
    move_rows = np.array([0, 1, 2])  # the move/tera rows (row 3 is a switch = zero in 3.3a)
    for fn, expect in (("a", 1.0), ("b", 0.0), ("tie", 0.5)):
        out = build_action_damage(_ctx(), _actions(),
                                  speed_fn=lambda a, d, f=fn: f, **base)
        assert (out["action_damage"][move_rows, 6] == expect).all()


def test_switch_row_zero_in_33a():
    # 3.3a boundary: switch rows are zero (3.3b fills entry-hazard + dmg_in).
    out = build_action_damage(_ctx(), _actions(), fmt="gen9bss3v3",
                              calc_fn=fake_calc, speed_fn=lambda a, d: "a")
    assert np.all(out["action_damage"][3] == 0.0)


def test_unknown_defender_degrades_no_fake_damage():
    ctx = {
        "active": {"p1": "p1a", "p2": "p2a"},
        "own": {"p1a": {"name": "garchomp", "hp_frac": 1.0}},
        "opponent": {"p2a": {"species": "", "hp_frac": 1.0}},  # unknown
    }
    out = build_action_damage(ctx, _actions(), fmt="gen9bss3v3",
                              calc_fn=fake_calc, speed_fn=lambda a, d: "tie")
    v = out["action_damage"][0]
    # no damage signal (dims 0-5,7,8 zero); outspeed neutral (0.5) is allowed
    assert np.all(v[[0, 1, 2, 3, 4, 5, 7, 8]] == 0.0)


def test_no_active_degrades_gracefully():
    ctx = {"own": {"p1a": {"name": "garchomp"}},
           "opponent": {"p2a": {"species": "blastoise"}}}
    out = build_action_damage(ctx, _actions(), fmt="gen9bss3v3",
                              calc_fn=fake_calc, speed_fn=lambda a, d: "a")
    assert np.all(out["action_damage"] == 0.0)


def test_calc_failure_degrades_row_not_raise():
    def boom(attacker, defender, move, fmt=None, gen=9):
        raise RuntimeError("calc down")
    out = build_action_damage(_ctx(), _actions(), fmt="gen9bss3v3",
                              calc_fn=boom, speed_fn=lambda a, d: "tie")
    v = out["action_damage"][0]
    # the per-action exception is caught -> zero damage row, no raise
    assert np.all(v[[0, 1, 2, 3, 4, 5, 7, 8]] == 0.0)


def test_action_damage_row_handles_none_assessment():
    v = action_damage_row(None, outspeed=0.5)
    assert v.shape == (K_ACTION_DAMAGE,)
    # no damage signal in dims 0-5,7,8; outspeed (dim 6) is the neutral 0.5
    assert np.all(v[[0, 1, 2, 3, 4, 5, 7, 8]] == 0.0) and v[6] == 0.5


def test_top1_posterior_ability_selected():
    # the dominant posterior ability (not the first listed) is the override
    ctx = _ctx()
    ctx["opponent"]["p2a"]["posterior"] = [
        {"ability": "weaknesspolicy", "weight": 0.2},
        {"ability": "torrent", "weight": 0.8},
    ]
    from net.action_features import _top_ability
    assert _top_ability(ctx["opponent"]["p2a"]) == "torrent"
