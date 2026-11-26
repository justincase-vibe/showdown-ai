"""Step 6.6 — the StrategyAgent (the ctx brain as a playable agent), hermetic.

Pins the battle-time contract: ``decide(battle, fmt, vocabs)`` returns a LEGAL
action (or None on a pass/wait turn), degrades to the first legal action when
the belief is unavailable, and never raises. Reuses the D92 fake battle
(``net.tests.test_env_obs._FakeBattle``) — no server, no torch.

Run:  python -m pytest fitness/tests/test_strategy_agent.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fitness.strategy_agent import (  # noqa: E402
    StrategyAgent,
    _gen_from_format,
    strategy_reference,
    strategy_posterior_reference,
)
from fitness.strategy import _posterior_overrides  # noqa: E402


def test_gen_from_format():
    assert _gen_from_format("gen9ou") == 9
    assert _gen_from_format("gen8bss3v3") == 8
    # D32: champions is a DIFFERENT mechanics set (engine 0) — NOT gen 9
    # (champions Slash BP=80 vs gen-9 70). The registry is the source of
    # truth (legality.formats.engine_key); a bare "champions" is not a
    # registered format id -> the documented primary-format fallback.
    assert _gen_from_format("gen9championsou") == 0
    assert _gen_from_format("champions") == 9      # unrecognised -> primary


def test_strategy_reference_shapes_and_agent():
    ref = strategy_reference(fmt="gen9ou")
    assert ref.name == "strategy"
    assert ref.weight == 2.0                        # same rung as `heuristic`
    agent = ref.agent(seed=3)
    assert isinstance(agent, StrategyAgent)
    assert agent.gen == 9 and agent.fmt == "gen9ou"
    assert agent.decides_from_battle is True        # the driver hook marker


def test_decide_returns_a_legal_action_for_fake_battle():
    from net.tests.test_env_obs import _FakeBattle, _FakeMon
    from net.env import build_action_list
    b = _FakeBattle(active=_FakeMon("garchomp", moves=["outrage", "earthquake"]),
                    avail_moves=["outrage", "earthquake"], tera=True)
    acts = build_action_list(b)
    assert acts, "the fake battle must offer legal actions"
    a = StrategyAgent(fmt="gen9ou", gen=9)
    got = a.decide(b, fmt="gen9ou")
    assert got is not None
    assert got in acts                              # a LEGAL action


def test_decide_pass_turn_returns_none():
    from net.tests.test_env_obs import _FakeBattle
    b = _FakeBattle(active=None, avail_moves=[], switches=(), wait=True)
    assert StrategyAgent(fmt="gen9ou").decide(b, fmt="gen9ou") is None


def test_decide_degrades_to_first_legal_when_belief_is_broken():
    from net.tests.test_env_obs import _FakeBattle, _FakeMon
    from net.env import build_action_list
    b = _FakeBattle(active=_FakeMon("garchomp", moves=["outrage"]),
                    avail_moves=["outrage"], log=RuntimeError("boom"))
    acts = build_action_list(b)
    assert acts
    got = StrategyAgent(fmt="gen9ou").decide(b, fmt="gen9ou")
    assert got is not None and got in acts          # legal, never raises


# ══════════════ 6.6a — the strategy_posterior variant (A2) ══════════════

def test_strategy_posterior_reference_shape():
    """The SECOND yardstick (6.6a): same name-rung contract, but the agent is
    built with posterior=True (the informed damage calc)."""
    ref = strategy_posterior_reference(fmt="gen9ou")
    assert ref.name == "strategy_posterior"
    assert ref.weight == 2.0                        # same rung as `strategy`
    agent = ref.agent(seed=3)
    assert isinstance(agent, StrategyAgent)
    assert agent.posterior is True                  # the informed variant
    assert agent.decides_from_battle is True


def test_strategy_yardstick_unchanged():
    """The EXISTING yardstick stays known-info-only (posterior=False) — the
    user's 6.6a: keep `strategy` as-is, add `strategy_posterior` as a SECOND
    variant, do NOT redefine the yardstick."""
    plain = strategy_reference(fmt="gen9ou").agent(seed=3)
    assert plain.posterior is False                 # unchanged yardstick
    # both variants share the same rules/weight — only the calc differs
    assert strategy_reference("gen9ou").weight == \
        strategy_posterior_reference("gen9ou").weight


def test_posterior_overrides_extracts_top1_and_public():
    """Hermetic: _posterior_overrides returns the opponent's TOP-1 posterior
    build (ability/evs/nature by weight) + the public tera + the own known
    build — the EXACT set the net's action features pass (6.6a fairness)."""
    ctx = {
        "active": {"p1": "p1a", "p2": "p2a"},
        "own": {"p1a": {"tera": "Dragon", "evs": {"hp": 252}, "nature": "Adamant"}},
        "opponent": {"p2a": {
            "tera": "Water",
            "posterior": [
                {"ability": "Swift Swim", "evs": {"spa": 252},
                 "nature": "Modest", "weight": 0.7},
                {"ability": "Rain Dish", "evs": {"hp": 252},
                 "nature": "Calm", "weight": 0.3},
            ],
        }},
    }
    ov = _posterior_overrides(ctx, "p1a", "p2a")
    # top-1 by weight (Swift Swim 0.7 > Rain Dish 0.3) — NOT the lower-weight pick
    assert ov["opp_ability"] == "Swift Swim"
    assert ov["opp_evs"] == {"spa": 252}
    assert ov["opp_nature"] == "Modest"
    # public / revealed tera (both sides)
    assert ov["own_tera"] == "Dragon"
    assert ov["opp_tera"] == "Water"
    # own known build
    assert ov["own_evs"] == {"hp": 252}
    assert ov["own_nature"] == "Adamant"


def test_posterior_overrides_empty_when_no_posterior():
    """No posterior (a belief that has not inferred a build yet) -> empty dict,
    so the calc degrades to known-info-only (the plain `strategy` path)."""
    ctx = {
        "active": {"p1": "p1a", "p2": "p2a"},
        "own": {"p1a": {}},
        "opponent": {"p2a": {"posterior": [], "tera": ""}},
    }
    ov = _posterior_overrides(ctx, "p1a", "p2a")
    assert ov == {}


# ══════════════ 2026-11-25 — the p1 harness contract (act) ══════════════
# The frozen eval loop (gui.m1_gate_main._play_one_battle) drives the p1
# SUBJECT through agent.act(obs, acts) for EVERY subject — the yardstick
# probe crashed 200/400 battles with
# "AttributeError: 'StrategyAgent' object has no attribute 'act'" (C4a/C4b
# mirror controls). These tests pin the fix: act() returns a
# Decision-shaped object, decides on the EXACT ctx the obs was built from
# (net.env.build_obs stashes it per-side, thread-local), degrades to the
# first legal action when the ctx is unavailable (A3.1), and never raises.

def test_act_returns_decision_shape_and_degrades_without_ctx():
    """No ctx stashed on this thread (the A3.1 degrade) -> the first legal
    action, Decision-shaped (the harness reads .action/.action_index/
    .logprob/.value), never an exception."""
    from net.env import set_last_ctx
    set_last_ctx("p1", None)  # simulate a failed belief build (or a fresh thread)
    acts = [{"kind": "move", "move": "outrage"},
            {"kind": "move", "move": "earthquake"}]
    d = StrategyAgent(fmt="gen9ou").act(None, acts)
    for f in ("action", "action_index", "logprob", "value", "entropy", "prob"):
        assert hasattr(d, f), f"missing contract field {f}"
    assert d.action == acts[0]
    assert d.action_index == 0
    assert d.logprob == 0.0 and d.value == 0.5 and d.entropy == 0.0


def test_act_accepts_the_greedy_wrapper_signature():
    """The frozen loop wraps p1 subjects in _Greedy, which calls
    act(obs, action_list, deterministic=True) — the signature must accept
    that exact call (contract parity with PokeAgent/HeuristicAgent)."""
    from net.env import set_last_ctx
    set_last_ctx("p1", None)
    acts = [{"kind": "move", "move": "outrage"}]
    d = StrategyAgent(fmt="gen9ou").act(None, acts, deterministic=True)
    assert d.action == acts[0]


def test_act_pass_turn_returns_pass_decision():
    """An empty action list is a pass/wait turn (the harness checks
    d.action is None) — the SAME contract decide() expresses as None."""
    d = StrategyAgent(fmt="gen9ou").act(None, [])
    assert d.action is None
    assert d.action_index == -1


def test_act_matches_decide_on_the_same_ctx():
    """THE regression: act() (the p1 harness path, ctx via the build_obs
    stash) must make the SAME decision decide() makes (the p2/driver path,
    ctx built from the live battle) — one decision pipeline."""
    from net.tests.test_env_obs import _FakeBattle, _FakeMon
    from net.tests.test_s41 import make_ctx, make_vocabs
    from net.env import build_action_list, build_obs
    b = _FakeBattle(
        active=_FakeMon("garchomp", moves=["outrage", "earthquake"]),
        avail_moves=["outrage", "earthquake"],
        switches=["tornadustherian"],
        tera=True,
    )
    acts = build_action_list(b)
    assert acts, "the fake battle must offer legal actions"
    # the p2/driver path (ground truth) — a FRESH agent (D2 switch memory is
    # per-battle/per-agent; the two paths must each start clean).
    expected = StrategyAgent(fmt="gen9ou", gen=9).decide(b, fmt="gen9ou")
    assert expected is not None
    # the p1 harness path: build_obs stashes the EXACT ctx it encodes.
    obs = build_obs(b, "gen9ou", make_vocabs(make_ctx()), "p1")
    d = StrategyAgent(fmt="gen9ou", gen=9).act(obs, acts)
    assert d.action is not None
    assert d.action == expected        # the SAME decision, both paths
    assert d.action in acts            # and LEGAL


def test_act_degrades_to_first_legal_when_ctx_build_failed():
    """build_obs stashes None when the belief build fails (it degrades to a
    zero obs) — act() must mirror decide()'s A3.1 degrade (first legal)."""
    from net.env import set_last_ctx
    set_last_ctx("p1", None)
    acts = [{"kind": "move", "move": "outrage"},
            {"kind": "switch", "species": "tornadustherian"}]
    d = StrategyAgent(fmt="gen9ou").act(None, acts)
    assert d.action == acts[0] and d.action_index == 0


def test_act_and_decide_share_the_c5_counters():
    """C5: the p1 subject's decisions must be counted too (the gate prints
    the silent-fallback counters per pair) — act() bumps the same module
    counters decide() does."""
    from fitness.strategy_agent import fallback_stats
    from net.env import set_last_ctx
    before = fallback_stats()
    set_last_ctx("p1", None)
    acts = [{"kind": "move", "move": "outrage"}]
    StrategyAgent(fmt="gen9ou").act(None, acts)
    after = fallback_stats()
    assert after["decides"] == before["decides"] + 1
    assert after["ctx_none_acts0"] == before["ctx_none_acts0"] + 1
