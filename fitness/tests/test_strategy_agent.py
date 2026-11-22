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
)


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
