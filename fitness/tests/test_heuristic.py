"""A3.1 — HeuristicAgent hermetic tests (no server, no GPU, no battle object).

The heuristic is pure: ``.act(obs, acts)`` over a decoded ``own_slot`` + the
legal ``acts`` + the move DB. So the tests build a tiny ``own_slot`` from the
registry's real column indices and assert the decision directly. torch is only
imported via ``net.registry`` (for the column indices) — never a net, never CUDA.

Run from the project root:  python -m pytest fitness/tests -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fitness.heuristic import Decision, HeuristicAgent, heuristic_reference
from fitness.ladder import Reference


def _obs(own_hp: float | None = None, active: bool = True) -> dict:
    """A minimal ``own_slot`` block with the REAL registry column indices."""
    from net.registry import MAX_SLOTS, SLOT_DIM, SLOT_INDEX
    slot = np.zeros((MAX_SLOTS, SLOT_DIM), dtype=np.float32)
    a = SLOT_INDEX["active"]
    p = SLOT_INDEX["present"]
    h = SLOT_INDEX["hp_frac"]
    if active:
        slot[0, a] = 1.0
        slot[0, p] = 1.0
        if own_hp is not None:
            slot[0, h] = float(own_hp)
    return {"own_slot": slot}


def _obs_types(own_species: str, opp_species: str, own_hp: float = 1.0) -> dict:
    """An obs whose active own/opp mon have KNOWN species (for type scoring)."""
    from net.registry import MAX_SLOTS, SLOT_DIM, SLOT_INDEX
    from net.vocab_db import build_vocabs
    voc = build_vocabs()
    own_slot = np.zeros((MAX_SLOTS, SLOT_DIM), dtype=np.float32)
    opp_slot = np.zeros((MAX_SLOTS, SLOT_DIM), dtype=np.float32)
    own_sp = np.zeros(MAX_SLOTS, dtype=np.int64)
    opp_sp = np.zeros(MAX_SLOTS, dtype=np.int64)
    own_slot[0, SLOT_INDEX["active"]] = 1.0
    own_slot[0, SLOT_INDEX["present"]] = 1.0
    own_slot[0, SLOT_INDEX["hp_frac"]] = float(own_hp)
    opp_slot[0, SLOT_INDEX["active"]] = 1.0
    opp_slot[0, SLOT_INDEX["present"]] = 1.0
    opp_slot[0, SLOT_INDEX["hp_frac"]] = 1.0
    own_sp[0] = voc.species.index(own_species)
    opp_sp[0] = voc.species.index(opp_species)
    return {"own_slot": own_slot, "opp_slot": opp_slot,
            "own_species": own_sp, "opp_species": opp_sp}


# ── legality + basic behaviour ──────────────────────────────────────────────
def test_pass_on_empty_acts():
    d = HeuristicAgent(gen=9).act(_obs(own_hp=1.0), [])
    assert d.action is None
    assert d.action_index == -1


def test_force_switch_picks_a_legal_switch():
    acts = [{"kind": "switch", "species": "pikachu"},
            {"kind": "switch", "species": "gengar"}]
    d = HeuristicAgent(gen=9).act(_obs(own_hp=1.0), acts)
    assert d.action in acts
    assert d.action["kind"] == "switch"


def test_switches_when_own_hp_is_low_but_plays_move_when_high():
    acts = [{"kind": "move", "move": "earthquake"},
            {"kind": "switch", "species": "gengar"}]
    low = HeuristicAgent(gen=9).act(_obs(own_hp=0.10), acts)
    assert low.action is acts[1]          # bailed out at 10% HP
    high = HeuristicAgent(gen=9).act(_obs(own_hp=0.90), acts)
    assert high.action is acts[0]         # at 90% HP it fights


def test_no_switch_available_falls_back_to_move():
    acts = [{"kind": "move", "move": "earthquake"}]  # trapped: no switch
    d = HeuristicAgent(gen=9).act(_obs(own_hp=0.05), acts)
    assert d.action is acts[0]


def test_zero_hp_active_row_reads_unknown_not_low():
    """2026-11-23 draw-bug guard: an active-flagged row at 0 HP is a BROKEN
    own-state (a live active mon can never be at 0 HP — it would be fainted
    and only switches would be legal). It must read as UNKNOWN, never as
    "critically low" (which would force a switch every turn and ping-pong a
    200-step stall against any switching opponent)."""
    o = _obs(own_hp=0.0)
    assert HeuristicAgent(gen=9)._my_active_hp(o) is None
    acts = [{"kind": "move", "move": "earthquake"},
            {"kind": "switch", "species": "gengar"}]
    d = HeuristicAgent(gen=9).act(o, acts)
    assert d.action is acts[0]          # attacks; does not switch on fake 0 HP


# ── move scoring ─────────────────────────────────────────────────────────────
def test_picks_highest_power_move():
    from legality.lookup import get_move
    moves = ["tackle", "earthquake"]                 # 40 vs 100
    power = {m: get_move(m, 9).get("power") or 0 for m in moves}
    expected = max(moves, key=lambda m: power[m])
    assert power[expected] > 0                        # sanity: it IS damaging
    acts = [{"kind": "move", "move": m} for m in moves]
    d = HeuristicAgent(gen=9).act(_obs(own_hp=1.0), acts)
    assert d.action is acts[moves.index(expected)]


def test_status_move_ranked_below_damaging_move():
    acts = [{"kind": "move", "move": "toxic"},        # status, no power
            {"kind": "move", "move": "tackle"}]       # 40 power
    d = HeuristicAgent(gen=9).act(_obs(own_hp=1.0), acts)
    assert d.action is acts[1]


def test_unknown_move_still_chosen_if_only_option():
    acts = [{"kind": "move", "move": "not-a-real-move-xyz"}]
    d = HeuristicAgent(gen=9).act(_obs(own_hp=1.0), acts)
    assert d.action is acts[0]                        # legal > "best"


# ── A3.2: type-aware move scoring (effectiveness * STAB) ──────────────────
def test_stab_boosts_the_matching_type_move():
    # charizard (Fire/Flying) vs venusaur (Grass/Poison), both 40-power:
    #   ember   (Fire)  : eff 2.0 * STAB 1.5  -> 120   (wins)
    #   aqua jet(Water) : eff 0.5 * 1.0       ->  20
    o = _obs_types("charizard", "venusaur")
    acts = [{"kind": "move", "move": "ember"},
            {"kind": "move", "move": "aqua jet"}]
    d = HeuristicAgent(gen=9).act(o, acts)
    assert d.action is acts[0]                         # ember (STAB Fire)


def test_avoids_an_immune_move_in_favour_of_a_damaging_one():
    # charizard (Fire/Flying) vs gengar (Ghost/Poison), both 40-power:
    #   tackle   (Normal) : eff 0.0 (immune to Ghost) ->  0
    #   aqua jet (Water)  : eff 1.0                   -> 40   (wins)
    o = _obs_types("charizard", "gengar")
    acts = [{"kind": "move", "move": "tackle"},
            {"kind": "move", "move": "aqua jet"}]
    d = HeuristicAgent(gen=9).act(o, acts)
    assert d.action is acts[1]                         # aqua jet (tackle is immune)


def test_type_scoring_degrades_to_power_when_types_unknown():
    # no species in obs -> effectiveness 1.0, no STAB -> pure power wins
    o = _obs(own_hp=1.0)                               # own_slot only, no species
    acts = [{"kind": "move", "move": "tackle"},
            {"kind": "move", "move": "earthquake"}]
    d = HeuristicAgent(gen=9).act(o, acts)
    assert d.action is acts[1]                         # earthquake (100) > tackle (40)


def _obs_switch_hazards(stealthrock_opp: bool = False,
                        toxic_opp: bool = False, low_hp: float = 0.05) -> dict:
    """A low-HP obs with hazards up on the OPPONENT's side (p2)."""
    from net.registry import (MAX_SLOTS, SLOT_DIM, SLOT_INDEX, GLOBAL_INDEX,
                              GLOBAL_DIM)
    own_slot = np.zeros((MAX_SLOTS, SLOT_DIM), dtype=np.float32)
    own_slot[0, SLOT_INDEX["active"]] = 1.0
    own_slot[0, SLOT_INDEX["present"]] = 1.0
    own_slot[0, SLOT_INDEX["hp_frac"]] = float(low_hp)
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    if stealthrock_opp:
        g[GLOBAL_INDEX["stealthrock_p2"]] = 1.0
    if toxic_opp:
        g[GLOBAL_INDEX["toxicspikes_p2"]] = 1.0
    return {"own_slot": own_slot, "global": g}


# ── A3.2b: hazard-aware switch target ──────────────────────────────────────
def test_switch_avoids_a_mon_weak_to_stealth_rock():
    # SR up on the opponent's side -> my switch-ins take Rock damage.
    # tyranitar (Rock) is weak; garchomp (Ground) is safe -> pick garchomp.
    o = _obs_switch_hazards(stealthrock_opp=True)
    acts = [{"kind": "switch", "species": "tyranitar"},
            {"kind": "switch", "species": "garchomp"}]
    d = HeuristicAgent(gen=9).act(o, acts)
    assert d.action is acts[1]                          # garchomp (not weak to SR)


def test_switch_avoids_a_poison_mon_under_toxic_spikes():
    o = _obs_switch_hazards(toxic_opp=True)
    acts = [{"kind": "switch", "species": "muk"},         # Poison -> takes spikes
            {"kind": "switch", "species": "garchomp"}]     # Ground -> safe
    d = HeuristicAgent(gen=9).act(o, acts)
    assert d.action is acts[1]                          # garchomp


def test_switch_is_first_when_no_hazards_are_up():
    # no hazards -> the A3.1 behaviour (first legal switch), still a switch.
    o = _obs(own_hp=0.05)                               # low HP, forces a switch
    acts = [{"kind": "switch", "species": "tyranitar"},
            {"kind": "switch", "species": "garchomp"},
            {"kind": "move", "move": "tackle"}]
    d = HeuristicAgent(gen=9).act(o, acts)
    assert d.action is acts[0]
    assert d.action["kind"] == "switch"


# ── determinism + decision shape (the strong-reference contract) ───────────
def test_deterministic():
    acts = [{"kind": "move", "move": "tackle"},
            {"kind": "move", "move": "earthquake"},
            {"kind": "switch", "species": "gengar"}]
    o = _obs(own_hp=1.0)
    a = HeuristicAgent(gen=9)
    d1 = a.act(o, acts)
    d2 = a.act(o, acts)
    assert d1.action_index == d2.action_index
    assert d1.action is d2.action


def test_decision_shape_is_valid():
    acts = [{"kind": "move", "move": "tackle"},
            {"kind": "move", "move": "earthquake"}]
    d = HeuristicAgent(gen=9).act(_obs(own_hp=1.0), acts)
    assert isinstance(d, Decision)
    assert 0 <= d.action_index < len(acts)
    assert d.action is acts[d.action_index]
    assert d.logprob == 0.0                            # one-hot -> log(1)
    assert 0.0 <= d.value <= 1.0
    assert d.prob[d.action_index] == 1.0


# ── the ladder wiring ───────────────────────────────────────────────────────
def test_heuristic_reference_builds_a_valid_agent():
    r = heuristic_reference(gen=9)
    assert isinstance(r, Reference)
    assert r.name == "heuristic"
    assert r.weight > 1.0                              # above the random floor
    agent = r.agent(seed=123)
    assert isinstance(agent, HeuristicAgent)
    assert agent.gen == 9


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
