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
