"""Step 3.3 tests — the per-ACTION damage/KO/speed block (move/tera + switch rows).

Hermetic: the damage calc and speed comparison are INJECTED as fakes (the
``calc_fn`` / ``speed_fn`` contract), so no Node subprocess is spawned and the
interpretation (KO verdicts, outspeed mapping, degradation) is pinned by logic,
not by the calc. The final test (real calc + real DB) is the one integration
seam that touches Node.
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
        "ohno": (110, 120, 115, False),    # 110-120% -> guaranteed KO at any HP
        "fiftysixty": (50, 60, 55, False),  # 50-60% -> KOs a 50%-hp mon only
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


# ── 3.3b: switch rows (entry-hazard % + incoming damage from the opp top move) ──

def _field(**kw):
    f = {"spikes": {"p1": 0, "p2": 0}, "toxicspikes": {"p1": 0, "p2": 0},
         "stealthrock": {}, "stickyweb": {}}
    f.update(kw)
    return f


def _chart(rock_eff, defender_type="Fire"):
    """A minimal ``get_type_chart``-shaped table ({defender: {attacker: factor}})
    with ONE defending type; the species chosen in each test carries that type.
    """
    return {defender_type: {"Rock": rock_eff, "Fire": 2.0}}


def test_hazard_pct_stealthrock_type_scaled():
    # Real chart (D29, DB-verified): Fire takes 2x Rock, Ghost 1x, Psychic 1x.
    # Real species + fake chart, so only the effectiveness under test varies.
    from net.action_features import hazard_pct
    f = _field(stealthrock={"p1": True, "p2": False})
    assert hazard_pct("charizard", f, gen=9, type_chart=_chart(2.0)) == pytest.approx(0.25)
    assert hazard_pct("gengar", f, gen=9, type_chart=_chart(1.0, "Ghost")) == pytest.approx(0.125)
    assert hazard_pct("mew", f, gen=9, type_chart=_chart(1.0, "Psychic")) == pytest.approx(0.125)
    # no SR on my side (only on theirs) -> no damage
    assert hazard_pct("charizard", _field(stealthrock={"p2": True}), gen=9,
                      type_chart=_chart(2.0)) == 0.0


def test_hazard_pct_spikes_layers_and_ground_immune():
    from net.action_features import hazard_pct
    f1 = _field(spikes={"p1": 1, "p2": 0})
    f3 = _field(spikes={"p1": 3, "p2": 0})
    ch = _chart(1.0)
    assert hazard_pct("mew", f1, gen=9, type_chart=ch) == pytest.approx(0.125)
    assert hazard_pct("mew", f3, gen=9, type_chart=ch) == pytest.approx(0.375)
    # Ground-type incoming mon is immune to Spikes (real DB: Groudon = Ground)
    assert hazard_pct("groudon", f1, gen=9, type_chart=_chart(1.0)) == 0.0
    # a type missing from the chart degrades to neutral 1.0 (no crash)
    ch_missing = {"Ghost": {"Rock": 2.0}}  # no 'Fire' row at all
    assert hazard_pct("charizard", f1, gen=9, type_chart=ch_missing) == pytest.approx(0.125)


def test_hazard_pct_toxicspikes_and_stickyweb_excluded():
    from net.action_features import hazard_pct
    f = _field(toxicspikes={"p1": 2, "p2": 0})
    assert hazard_pct("mew", f, gen=9, type_chart=_chart(1.0)) == pytest.approx(0.25)
    # Sticky Web is a speed trap, not damage -> excluded
    assert hazard_pct("mew", _field(stickyweb={"p1": True}), gen=9,
                      type_chart=_chart(1.0)) == 0.0
    # stacked hazards add up (12.5 + 12.5 + 12.5)
    f = _field(spikes={"p1": 1, "p2": 0}, toxicspikes={"p1": 1, "p2": 0},
               stealthrock={"p1": True})
    assert hazard_pct("mew", f, gen=9, type_chart=_chart(1.0)) == pytest.approx(0.375)


def test_hazard_pct_degrades_to_zero_on_failure():
    from net.action_features import hazard_pct
    assert hazard_pct("garchomp", None, gen=9) == 0.0
    # unknown species -> no types -> NEUTRAL 1.0 per missing type (documented
    # fallback; still 12.5% for one SR layer), NOT zero
    assert hazard_pct("", _field(stealthrock={"p1": True}), gen=9, type_chart=_chart(2.0)) == pytest.approx(0.125)
    # real DB chart (D29): Garchomp = Ground 2x x Dragon 0.5x Rock -> 6.25%
    assert hazard_pct("garchomp", _field(stealthrock={"p1": True}), gen=9,
                      type_chart=None) == pytest.approx(0.0625)


def test_switch_row_hazard_and_incoming_damage():
    ctx = _ctx()
    ctx["field"] = _field(stealthrock={"p1": True, "p2": False})
    # the incoming mon is one of MY OWN mons (ctx['own']) at full HP
    ctx["own"]["charizard"] = {"name": "charizard", "hp_frac": 1.0}
    # the opponent's top move deals 110-120% -> guaranteed KO on a full-HP mon
    ctx["opponent"]["p2a"]["top_moves"] = ["ohno"]
    out = build_action_damage(
        ctx, [{"kind": "switch", "species": "charizard"}], fmt="gen9bss3v3",
        calc_fn=fake_calc, speed_fn=lambda a, d: "a", type_chart=_chart(2.0))
    v = out["action_damage"][0]
    assert v[8] == pytest.approx(0.25)          # hazard: 12.5% x 2x Rock (Fire, D29)
    assert v[4] == 1.0 and v[3] == 1.0 and v[5] == 1.0  # guaranteed KO on full HP
    assert v[6] == 0.0                           # outspeed is a move-row dim only


def test_switch_ko_verdicts_use_incoming_hp_not_active():
    # THE 3.3b bug-guard: a 50-60% move KOs a 50%-HP mon (min 50% >= 50%
    # threshold) yet canNOT KO a full mon (min 50% < 100%). If the verdicts
    # were scored vs MY ACTIVE's HP (50%) instead of the INCOMING mon's (100%),
    # guaranteed_ko would wrongly be 1 here.
    ctx = _ctx(opp_hp=0.5)
    ctx["own"]["gengar"] = {"name": "gengar", "hp_frac": 1.0}
    ctx["opponent"]["p2a"]["top_moves"] = ["fiftysixty"]
    out = build_action_damage(
        ctx, [{"kind": "switch", "species": "gengar"}], fmt="gen9bss3v3",
        calc_fn=fake_calc, speed_fn=lambda a, d: "a", type_chart=_chart(1.0))
    v = out["action_damage"][0]
    assert v[3] == 0.0 and v[4] == 0.0 and v[5] == 0.0  # no KO at full HP


def test_switch_no_top_move_gives_hazard_only_row():
    ctx = _ctx()
    ctx["field"] = _field(spikes={"p1": 2, "p2": 0})
    ctx["own"]["gengar"] = {"name": "gengar", "hp_frac": 1.0}
    ctx["opponent"]["p2a"]["top_moves"] = []  # no posterior, no observed move
    out = build_action_damage(
        ctx, [{"kind": "switch", "species": "gengar"}], fmt="gen9bss3v3",
        calc_fn=fake_calc, speed_fn=lambda a, d: "a", type_chart=_chart(1.0))
    v = out["action_damage"][0]
    assert v[8] == pytest.approx(0.25)
    assert np.all(v[[0, 1, 2, 3, 4, 5, 7]] == 0.0)  # damage dims empty


def test_top_move_prefers_posterior_marginal_over_observed():
    ctx = _ctx()
    ctx["opponent"]["p2a"]["top_moves"] = ["marginal_top"]
    ctx["opponent"]["p2a"]["moves"] = ["used_earlier"]
    from net.action_features import _top_move
    assert _top_move(ctx["opponent"]["p2a"]) == "marginal_top"
    del ctx["opponent"]["p2a"]["top_moves"]
    assert _top_move(ctx["opponent"]["p2a"]) == "used_earlier"


def test_switch_row_scores_against_incoming_mon_not_active_hp():
    # the same 50-60% move vs a 50%-HP incoming mon -> guaranteed KO
    # (min 50% >= 50% threshold): proves the verdict follows the INCOMING
    # mon's HP, not my active's.
    ctx = _ctx(opp_hp=0.5)
    ctx["own"]["gengar"] = {"name": "gengar", "hp_frac": 0.5}
    ctx["opponent"]["p2a"]["top_moves"] = ["fiftysixty"]  # 50-60%
    out = build_action_damage(
        ctx, [{"kind": "switch", "species": "gengar"}], fmt="gen9bss3v3",
        calc_fn=fake_calc, speed_fn=lambda a, d: "a", type_chart=_chart(1.0))
    v = out["action_damage"][0]
    assert v[4] == 1.0 and v[3] == 1.0 and v[5] == 1.0  # 50-60% vs a 50% mon = guaranteed KO


def test_switch_calc_failure_degrades_to_hazard_only():
    def boom(attacker, defender, move, fmt=None, gen=9):
        raise RuntimeError("calc down")
    ctx = _ctx()
    ctx["field"] = _field(spikes={"p1": 1, "p2": 0})
    ctx["own"]["gengar"] = {"name": "gengar", "hp_frac": 1.0}
    ctx["opponent"]["p2a"]["top_moves"] = ["bigmove"]
    out = build_action_damage(
        ctx, [{"kind": "switch", "species": "gengar"}], fmt="gen9bss3v3",
        calc_fn=boom, speed_fn=lambda a, d: "a", type_chart=_chart(1.0))
    v = out["action_damage"][0]
    assert v[8] == pytest.approx(0.125)  # hazard survives the calc failure
    assert np.all(v[[0, 1, 2, 3, 4, 5, 7]] == 0.0)


def test_real_calc_switchin_hazard_and_incoming():
    # REAL calc + REAL DB (the only integration seam): hazard % from the DB
    # type chart (D29) + the opponent's top move (Surf) through the real
    # pipeline. Magcargo (Fire/Rock, 4x Water) vs Garchomp (Dragon/Ground,
    # 1x Water) — the two switch rows must separate on the damage dims.
    ctx = _ctx()
    ctx["field"] = _field(stealthrock={"p1": True, "p2": False})
    ctx["own"]["garchomp"] = {"name": "garchomp", "hp_frac": 1.0}
    ctx["own"]["magcargo"] = {"name": "magcargo", "hp_frac": 1.0}
    ctx["opponent"]["p2a"]["top_moves"] = ["surf"]
    out = build_action_damage(
        ctx, [{"kind": "switch", "species": "garchomp"},
              {"kind": "switch", "species": "magcargo"}],
        fmt="gen9bss3v3", speed_fn=lambda a, d: "a")
    garchomp_row, magcargo_row = out["action_damage"][0], out["action_damage"][1]
    # hazard (Rock, D29 chart): Garchomp Ground 2x x Dragon 0.5x -> 6.25%;
    # Magcargo Fire 2x x Rock 1x -> 25%
    assert garchomp_row[8] == pytest.approx(0.0625)
    assert magcargo_row[8] == pytest.approx(0.25)
    # Surf (Water): Magcargo Fire 2x x Rock 2x = 4x (guaranteed KO); Garchomp
    # Dragon 0.5x x Ground 2x = 1x (no KO at full HP)
    assert magcargo_row[2] > garchomp_row[2]
    assert magcargo_row[4] == 1.0
    assert garchomp_row[4] == 0.0


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
