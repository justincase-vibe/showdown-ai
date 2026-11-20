"""League driver tests — FAST + offline (no fork, no pool, no GPU).

Two layers, both hermetic:
  * ``_step_loop`` / ``play_agent_battle_on`` — the step-loop's scheduling
    (whose turn, pass vs act, termination, on_finish/close contract) via a
    SCRIPTED FAKE ENV + a fake decision seam (the ``decide=`` / ``env_factory=``
    injection points);
  * ``policy_action`` — the REAL decision path (build_obs -> PokeAgent.act)
    with a real random-init net + the lightweight fake battle the D92 tests
    use (``net.tests.test_env_obs._FakeBattle``), plus its defensive
    never-raise contract.

The LIVE checkpoint-vs-checkpoint path (real ShowdownEnv + fork) is the
league's live test, run manually / in CI with a server pool.

Run:  python -m pytest league/tests/test_driver.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.driver import (
    DecisionGuard,
    _step_loop,
    play_agent_battle_on,
    policy_action,
)


# ── fake battle / fake env ──────────────────────────────────────────────────
class _Move:
    def __init__(self, mid):
        self.id = mid


class _Mon:
    def __init__(self, species="garchomp", moves=("tackle",)):
        self.species = species
        self.base_species = species
        self.moves = {i: _Move(m) for i, m in enumerate(moves)}


class _Battle:
    def __init__(self):
        self.finished = False
        self.won = False
        self.active_pokemon = _Mon(moves=("tackle",))
        self.team = {"p1a": _Mon()}


class _FakeEnv:
    """The ShowdownEnv surface the driver loop uses, scripted.

    ``turns``: how many full steps to allow before the battle ends on its own;
    ``finish_after``: after N steps, ``battle1.finished`` becomes True and
    ``battle1.won`` = True (the top-of-loop check then breaks)."""

    def __init__(self, turns: int = 10, finish_after: int = 99):
        self.possible_agents = ("A1", "A2")
        self.battle1 = _Battle()
        self.battle2 = _Battle()
        self.agent1_to_move = True
        self.agent2_to_move = True
        self._turns = turns
        self._finish_after = finish_after
        self._steps = 0
        self.actions_seen = []
        self.closed = 0
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return {}, None

    def step(self, actions):
        self.actions_seen.append(dict(actions))
        self._steps += 1
        if self._steps >= self._finish_after:
            self.battle1.finished = True
            self.battle1.won = True
        term = {"A1": self._steps >= self._turns, "A2": self._steps >= self._turns}
        return {}, None, term, {"A1": False, "A2": False}, None

    def close(self, force: bool = True, wait: bool = True):
        self.closed += 1


ACT_TACKLE = {"kind": "move", "move": "tackle"}


# ── the step loop ───────────────────────────────────────────────────────────
def test_step_loop_act_and_pass_scheduling():
    env = _FakeEnv(turns=5)
    n = _step_loop(env, lambda b: ACT_TACKLE, lambda b: None, max_steps=50)
    assert n == 5                                   # until max turn cap (term)
    assert len(env.actions_seen) == 5
    for a in env.actions_seen:
        assert a["A1"] == 6                          # pe int for "tackle" (idx 0)
        assert a["A2"] == -2                         # pass (decide -> None)


def test_step_loop_terminates_when_battle_finished():
    env = _FakeEnv(turns=50, finish_after=2)
    n = _step_loop(env, lambda b: ACT_TACKLE, lambda b: ACT_TACKLE, max_steps=50)
    assert n == 2
    assert env.battle1.finished


def test_step_loop_respects_max_steps():
    env = _FakeEnv(turns=1000, finish_after=9999)
    n = _step_loop(env, lambda b: ACT_TACKLE, lambda b: None, max_steps=7)
    assert n == 7
    assert len(env.actions_seen) == 7


def test_step_loop_no_move_turns_only_pass():
    env = _FakeEnv(turns=2)
    env.agent1_to_move = False
    env.agent2_to_move = False
    n = _step_loop(env, lambda b: ACT_TACKLE, lambda b: ACT_TACKLE, max_steps=10)
    assert n == 2
    assert all(a["A1"] == -2 and a["A2"] == -2 for a in env.actions_seen)


# ── play_agent_battle_on (the battle-driver contract) ──────────────────────
def test_full_contract_capture_close_and_result():
    env = _FakeEnv(turns=3, finish_after=3)
    captured = {}

    def on_finish(battle):
        captured["battle"] = battle

    def decide(agent, battle, fmt, vocabs, greedy):
        return ACT_TACKLE if agent == "AG_A" else None

    a1, a2, n = play_agent_battle_on(
        "server-cfg", agent_a="AG_A", agent_b="AG_B",
        decide=decide, on_finish=on_finish, env_factory=lambda: env)
    assert (a1, a2, n) == (True, False, 3)          # battle1.won / battle2 not
    assert captured["battle"] is env.battle1        # the capture hook fired
    assert env.reset_calls == 1
    assert env.closed == 1                          # teardown happened


def test_close_called_even_when_decide_raises():
    env = _FakeEnv(turns=3)

    def decide(agent, battle, fmt, vocabs, greedy):
        raise RuntimeError("policy exploded")

    with pytest.raises(RuntimeError, match="policy exploded"):
        play_agent_battle_on("cfg", agent_a="A", agent_b="B",
                             decide=decide, env_factory=lambda: env)
    assert env.closed == 1                          # finally: always teardown


def test_play_agent_battle_on_needs_both_agents():
    with pytest.raises(TypeError):
        play_agent_battle_on("cfg", agent_a="A")    # agent_b missing


# ── policy_action (the real decision path, hermetic) ───────────────────────
def _vocabs():
    from net.tests.test_env_obs import _vocabs as v
    return v()


def _agent(vocabs):
    import torch
    from net.agent import PokeAgent, make_random_net
    net = make_random_net(vocabs, device="cpu")
    return PokeAgent(net, seed=7, device="cpu")


def test_policy_action_returns_legal_action_or_pass():
    from net.tests.test_env_obs import _FakeBattle, _FakeMon
    from net.env import build_action_list
    v = _vocabs()
    b = _FakeBattle(active=_FakeMon("garchomp", moves=["outrage", "earthquake"]),
                    avail_moves=["outrage", "earthquake"], tera=True)
    acts = build_action_list(b)
    assert acts, "the fake battle must offer legal actions"
    got = policy_action(_agent(v), b, "gen9ou", v, greedy=True)
    assert got is None or got in acts               # legal (or a pass turn)


def test_policy_action_pass_on_wait_turn():
    from net.tests.test_env_obs import _FakeBattle
    v = _vocabs()
    b = _FakeBattle(active=None, avail_moves=[], switches=(), wait=True)
    assert policy_action(_agent(v), b, "gen9ou", v, greedy=True) is None


# ── the capture seam (D103 step 2: experience collection) ─────────────────
def test_policy_action_capture_fires_with_the_exact_decision():
    """The capture hook must receive the EXACT obs + full Decision (index /
    logprob / value) of the action the loop will play — that is what the
    bot's buffer will learn from."""
    from net.tests.test_env_obs import _FakeBattle, _FakeMon
    from net.env import build_action_list
    v = _vocabs()
    b = _FakeBattle(active=_FakeMon("garchomp", moves=["outrage", "earthquake"]),
                    avail_moves=["outrage", "earthquake"], tera=True)
    acts = build_action_list(b)
    assert acts, "the fake battle must offer legal actions"
    caps = []
    got = policy_action(_agent(v), b, "gen9ou", v, greedy=True,
                        capture=lambda obs, d: caps.append((obs, d)))
    assert got is not None                     # legal actions -> a real action
    assert len(caps) == 1
    obs, d = caps[0]
    assert d.action is got                     # the SAME decision the loop plays
    assert d.action in acts                    # legal
    assert d.action_index >= 0
    assert d.logprob <= 0.0                    # a log-probability
    assert 0.0 <= d.value <= 1.0               # the value head's P(win) range
    assert isinstance(obs, dict) and len(obs) > 0


def test_policy_action_capture_not_fired_on_wait_turn():
    """A pass/wait is not a transition: no capture, nothing to learn from."""
    from net.tests.test_env_obs import _FakeBattle
    v = _vocabs()
    b = _FakeBattle(active=None, avail_moves=[], switches=(), wait=True)
    caps = []
    assert policy_action(_agent(v), b, "gen9ou", v, greedy=True,
                         capture=lambda obs, d: caps.append(1)) is None
    assert caps == []


def test_policy_action_never_raises_on_broken_battle():
    from net.tests.test_env_obs import _FakeBattle
    v = _vocabs()
    b = _FakeBattle(log=RuntimeError("boom"))       # _build_replay_log raises
    warnings = []
    got = policy_action(_agent(v), b, "gen9ou", v, greedy=True,
                        log=warnings.append)
    assert got is None                              # degraded to a pass turn
    assert any("decision error" in w for w in warnings)


# ── DecisionGuard (the D102 pass-fest guard) ────────────────────────────────
# The D102 device-mismatch bug made EVERY decision raise; the old policy_action
# swallowed each one and degraded to a pass, so a broken net silently pass-fested
# all 3200 battles. The guard still tolerates a SINGLE glitch (a live battle must
# not crash on one observation hiccup) but fails LOUD after N back-to-back errors.

def test_decision_guard_degrades_a_single_glitch_to_pass():
    g = DecisionGuard(max_consecutive=3)
    assert g.on_error(RuntimeError("glitch")) is None   # 1st -> pass this turn
    assert g.consecutive == 1


def test_decision_guard_resets_on_success():
    g = DecisionGuard(max_consecutive=2)
    g.on_error(RuntimeError("1"))                       # consecutive = 1
    g.on_success()                                       # a good decision
    assert g.consecutive == 0                            # streak reset
    g.on_error(RuntimeError("2"))                       # a NEW streak starts
    assert g.consecutive == 1


def test_decision_guard_raises_after_consecutive_threshold():
    g = DecisionGuard(max_consecutive=3)
    g.on_error(RuntimeError("a"))                       # 1 (pass)
    g.on_error(RuntimeError("b"))                       # 2 (pass)
    with pytest.raises(RuntimeError, match="consecutive decision errors"):
        g.on_error(RuntimeError("c"))                   # 3 -> fail loud


def test_policy_action_with_guard_degrades_then_raises():
    from net.tests.test_env_obs import _FakeBattle
    v = _vocabs()
    b = _FakeBattle(log=RuntimeError("boom"))           # every decision raises
    g = DecisionGuard(max_consecutive=3)
    assert policy_action(_agent(v), b, "gen9ou", v, greedy=True,
                         guard=g) is None               # 1st glitch -> pass
    assert policy_action(_agent(v), b, "gen9ou", v, greedy=True,
                         guard=g) is None               # 2nd glitch -> pass
    with pytest.raises(RuntimeError, match="consecutive decision errors"):
        policy_action(_agent(v), b, "gen9ou", v, greedy=True, guard=g)


def test_play_agent_battle_on_fails_loud_on_systematic_decision_error():
    """The end-to-end guard: a decision path that raises on EVERY turn must
    FAIL the battle (propagate) when max_decision_errors is set — not pass-fest
    to the turn cap. (The D102 device bug, which the old code pass-fested.)

    The fake env's battles have no ``_build_replay_log``, so every real
    ``policy_action`` call raises (AttributeError) — a systematic failure, the
    same shape as "100% of decisions raise". The guard must fail the battle
    loud after the threshold instead of running to the turn cap.
    """
    env = _FakeEnv(turns=50, finish_after=9999)         # would run to the cap
    env.agent1_to_move = True
    env.agent2_to_move = True
    agent = _agent(_vocabs())

    with pytest.raises(RuntimeError, match="consecutive decision errors"):
        play_agent_battle_on(
            "cfg", agent_a=agent, agent_b=agent,
            env_factory=lambda: env, max_decision_errors=3,
        )
    assert env.closed == 1                              # teardown still happened


def test_play_agent_battle_on_legacy_passfest_without_guard():
    """Contrast: with NO guard (max_decision_errors=None, the legacy behaviour)
    the SAME broken decision path degrades every turn to a pass and runs all
    the way to the turn cap — exactly what silently wasted the D102 run.
    (Pinned so the guard's presence is what changes the outcome.)"""
    env = _FakeEnv(turns=50, finish_after=9999)
    env.agent1_to_move = True
    env.agent2_to_move = True
    agent = _agent(_vocabs())
    a1, a2, n = play_agent_battle_on(
        "cfg", agent_a=agent, agent_b=agent,
        env_factory=lambda: env, max_decision_errors=None)
    assert n == 50                                      # ran to the cap, no raise
    assert not a1 and not a2                            # nobody ever acted -> nobody won
