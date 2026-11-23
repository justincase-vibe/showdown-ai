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
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.driver import (
    DecisionGuard,
    _step_loop,
    play_agent_battle_on,
    policy_action,
    NoProgressTracker,
    progress_signature,
)
from league import driver  # noqa: E402  (monkeypatch target for _cap_signature)


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


class _ObsSpyAgent:
    """Duck-typed agent recording the obs it was handed (torch-free)."""

    def __init__(self):
        self.obs = None
        self.acts: list = []

    def act(self, obs, acts, deterministic=False):
        self.obs = obs
        self.acts = list(acts or [])
        class _D:
            pass
        d = _D()
        d.action = self.acts[0] if self.acts else None
        return d


def test_policy_action_uses_caller_side_verbatim():
    """2026-11-23 POV fix: the caller's KNOWN side (the env's ground truth —
    battle1=p1, battle2=p2, the same mapping the training observation uses)
    must be used VERBATIM; the my_side species-guess is bypassed. For the
    SAME battle object (the p2 player's), the obs handed to the agent must
    equal build_obs(..., "p2") exactly. Pre-fix this path ran my_side, which
    failed on real fork logs and built the p2 agent's obs from p1's POV —
    the root cause of the switch-standoff draws."""
    import numpy as np
    from net.env import build_obs
    from net.tests.test_env_obs import _FakeBattle, _FakeMon
    v = _vocabs()
    log = ("|poke|p1a|Garchomp, M|\n"
           "|poke|p1b|Tornadus, M|\n"
           "|poke|p2a|Tyranitar, F|\n"
           "|poke|p2b|Excadrill, F|")
    b = _FakeBattle(log=log,
                    team={"p2a": _FakeMon("tyranitar"),
                          "p2b": _FakeMon("excadrill")},
                    active=_FakeMon("tyranitar", moves=["crunch"]),
                    avail_moves=["crunch"])
    spy = _ObsSpyAgent()
    got = policy_action(spy, b, "gen9ou", v, greedy=True, side="p2")
    assert got is not None
    expect = build_obs(b, "gen9ou", v, "p2")
    assert set(spy.obs.keys()) == set(expect.keys())
    for k in expect:
        np.testing.assert_array_equal(spy.obs[k], expect[k],
                                      err_msg=f"obs[{k}] differs")


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


# ── the ctx-brain hook (Step 6.6: decides_from_battle) ──────────────────────
class _CtxAgent:
    """A StrategyAgent-shaped policy (the ``decides_from_battle`` marker).

    Its ``act`` MUST NOT be called (the hook routes through ``decide``);
    a call to it is a wiring error and fails the test loudly."""

    decides_from_battle = True

    def __init__(self, action: Any = ACT_TACKLE, boom: bool = False):
        self._action = action
        self._boom = boom
        self.decide_calls = 0

    def decide(self, battle: Any, fmt: str, vocabs: Any) -> Any:
        self.decide_calls += 1
        if self._boom:
            raise RuntimeError("brain exploded")
        return self._action

    def act(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - wiring guard
        raise AssertionError("act() must not be called for a ctx brain")


def test_policy_action_routes_ctx_brain_through_decide():
    from net.tests.test_env_obs import _FakeBattle, _FakeMon
    a = _CtxAgent()
    b = _FakeBattle(active=_FakeMon("garchomp", moves=["outrage"]),
                    avail_moves=["outrage"])
    got = policy_action(a, b, "gen9ou", None, greedy=True)
    assert got == ACT_TACKLE
    assert a.decide_calls == 1


def test_policy_action_ctx_brain_pass_turn_returns_none():
    from net.tests.test_env_obs import _FakeBattle
    a = _CtxAgent(action=None)          # no legal action this moment
    assert policy_action(a, _FakeBattle(), "gen9ou", None, greedy=True) is None


def test_policy_action_ctx_brain_guard_counts_success():
    from net.tests.test_env_obs import _FakeBattle
    g = DecisionGuard(max_consecutive=3)
    policy_action(_CtxAgent(), _FakeBattle(), "gen9ou", None, greedy=True,
                  guard=g)
    assert g.consecutive == 0           # a successful decision resets/holds 0


def test_policy_action_ctx_brain_error_degrades_to_pass():
    from net.tests.test_env_obs import _FakeBattle
    warnings: list = []
    got = policy_action(_CtxAgent(boom=True), _FakeBattle(), "gen9ou", None,
                        greedy=True, log=warnings.append)
    assert got is None                  # the brain's raise degrades to a pass
    assert any("decision error" in w for w in warnings)


# ── D3 (2026-11-23) — the no-progress cap ───────────────────────────────────

class _StallBattle:
    """A battle whose log is a PROTECT-SPAM stall: turn after turn of
    |move|Protect with NO progress lines (the exact stall D3 must end)."""

    def __init__(self):
        self.finished = False
        self.won = False
        self.active_pokemon = _Mon(moves=("tackle", "protect"))
        self.team = {"p1a": _Mon()}
        self._turn = 0
        self._log = ["|gen9|", "|tier|Gen 9 OU", "|start|"]

    def _emit_turn(self):
        self._turn += 1
        self._log.append(f"|turn|{self._turn}|")
        self._log.append("|move|p1a: Garchomp|Protect|p1a: Garchomp")
        self._log.append("|move|p2a: Blastoise|Protect|p2a: Blastoise")

    def _build_replay_log(self):
        return "\n".join(self._log)


class _DamageBattle(_StallBattle):
    """The same stall but with a -damage line EVERY turn (legit play — the
    cap must never trip)."""

    def _emit_turn(self):
        super()._emit_turn()
        self._log.append("|-damage|p2a: Blastoise|80%|")


class _StallEnv:
    """The ShowdownEnv surface, scripted: every step emits one full stall
    turn on the log; term fires at ``max_turns`` steps."""

    def __init__(self, max_turns: int = 100):
        self.possible_agents = ("A1", "A2")
        self.battle1 = _StallBattle()
        self.battle2 = _StallBattle()
        self.agent1_to_move = True
        self.agent2_to_move = True
        self._max_turns = max_turns
        self._steps = 0
        self.closed = 0
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return {}, None

    def step(self, actions):
        self._steps += 1
        self.battle1._emit_turn()
        self.battle2._emit_turn()
        term = {"A1": self._steps >= self._max_turns,
                "A2": self._steps >= self._max_turns}
        return {}, None, term, {"A1": False, "A2": False}, None

    def close(self, force: bool = True, wait: bool = True):
        self.closed += 1


# ── NoProgressTracker (the pure state machine) ─────────────────────────────

def test_tracker_protect_spam_accumulates_and_trips():
    t = NoProgressTracker(limit=3)
    for _ in range(2):
        t.feed_lines(["|turn|1|", "|move|p1a|Protect|p1a",
                      "|move|p2a|Protect|p2a"])
    assert t.streak == 2 and not t.tripped
    t.feed_lines(["|turn|3|", "|move|p1a|Protect|p1a"])
    assert t.streak == 3 and t.tripped


def test_tracker_switch_spam_is_no_progress():
    # a switch ping-pong (|switch|/|setactive|) is exactly the stall the cap
    # exists to end — it must NOT count as progress
    t = NoProgressTracker(limit=2)
    for i in (1, 2):
        t.feed_lines([f"|turn|{i}|",
                      "|switch|p1a: Garchomp|183/183|",
                      "|setactive|Garchomp",
                      "|switch|p2a: Blastoise|190/190|",
                      "|setactive|Blastoise"])
    assert t.tripped


def test_tracker_fork_upkeep_phase_marker_is_NOT_progress():
    # 2026-11-23 regression (D6-draw investigation): THIS FORK emits a bare
    # ``|upkeep`` phase marker on EVERY turn unconditionally (sim/battle.ts
    # ``residual`` case; SIM-PROTOCOL.md: "Signals the upkeep phase of the
    # turn"). It is NOT game-state progress. Listing it as a progress tag
    # reset the streak every single turn and silently disabled the D3 cap in
    # every real battle (the fake-env tests never emitted the marker, so they
    # stayed green). Real item upkeep arrives as ``-damage/-heal [from] item``
    # — covered by the existing tests.
    t = NoProgressTracker(limit=3)
    for i in (1, 2, 3):
        t.feed_lines([f"|turn|{i}|",
                      "|switch|p1a: Garchomp|183/183|",
                      "|switch|p2a: Blastoise|190/190|",
                      "|upkeep"])   # the fork's per-turn phase marker
    assert t.streak == 3 and t.tripped


def test_tracker_progress_resets_the_streak():
    # a turn's progress is judged at the NEXT boundary (a turn is over only
    # once the following |turn| line arrives) — the semantics the loop uses
    t = NoProgressTracker(limit=3)
    t.feed_lines(["|turn|1|", "|move|p1a|Protect|p1a"])
    assert t.streak == 1
    t.feed_lines(["|turn|2|", "|-damage|p2a|Blastoise|80%|"])  # turn 2's damage
    t.feed_lines(["|turn|3|", "|switch|p1a|Garchomp|183/183|"])
    assert t.streak == 0                    # ...resets at the turn-3 boundary
    t.feed_lines(["|turn|4|", "|-status|p2a|par|"])             # turn 4's status
    t.feed_lines(["|turn|5|", "|move|p1a|Protect|p1a"])
    assert t.streak == 0                    # ...resets at the turn-5 boundary
    for i in (6, 7, 8):                     # build a full streak again
        t.feed_lines([f"|turn|{i}|", "|move|p1a|Protect|p1a"])
    assert t.streak == 3 and t.tripped


def test_tracker_hazard_boost_and_chip_are_progress():
    t = NoProgressTracker(limit=5)
    t.feed_lines(["|turn|1|", "|move|p1a|Stealth Rock|p1a"])       # announce
    t.feed_lines(["|turn|2|", "|-sidestart|p1|Stealth Rock|"])     # field set
    t.feed_lines(["|turn|3|", "|-boost|p1a|atk|2|"])               # boost
    t.feed_lines(["|turn|4|", "|-side damage|p2a|Blastoise|90%|"]) # chip
    t.feed_lines(["|turn|5|", "|move|p1a|Protect|p1a"])
    assert t.streak == 0                # each progress turn reset at its next boundary


def test_tracker_junk_lines_are_ignored():
    t = NoProgressTracker(limit=1)
    t.feed_lines(["", "not a line", "|", "||", None, "   "])
    assert t.streak == 0 and not t.tripped


def test_tracker_limit_floor():
    assert NoProgressTracker(limit=0).limit == 1   # max(1, ...) floor
    assert NoProgressTracker(limit=-5).limit == 1


# ── progress_signature (the belief-diff confirmation) ───────────────────────

def _ctx_base():
    return {
        "turn": 5,
        "fainted": {"p1": 0, "p2": 0},
        "active": {"p1": "p1a", "p2": "p2a"},
        "field": {"weather": "", "terrain": "", "trickroom": False,
                  "wonderroom": False, "tailwind_side": "",
                  "reflect": {}, "lightscreen": {}, "auroraveil": {},
                  "spikes": {}, "toxicspikes": {}, "stealthrock": {},
                  "stickyweb": {}, "safeguard": {}},
        "own": {"p1a": {"hp_frac": 1.0, "statuses": [], "boosts": {},
                        "moves": ["ember"], "tera": "", "last_hit": 0,
                        "hit_this_turn": False, "last_move_failed": False,
                        "stockpile": 0, "used_protect": True,
                        "volatiles": [], "maxhp": 183}},
        "opponent": {"p2a": {"species": "blastoise", "hp_frac": 0.5,
                             "statuses": [], "boosts": {}, "moves": [],
                             "tera": "", "last_hit": 0,
                             "hit_this_turn": False,
                             "last_move_failed": False, "stockpile": 0,
                             "used_protect": False, "volatiles": [],
                             "maxhp": 183,
                             "posterior": [{"ability": "torrent",
                                            "weight": 1.0}],
                             "top_moves": ["surf"],
                             "top_move_weights": [1.0]}},
    }


def test_signature_invariant_to_action_history():
    # turn counter, moves_used (Protect spam), the active slots (switch
    # ping-pong), the per-turn flags and the opponent's posterior/top_moves
    # are NOT progress — none of them may change the signature
    import copy
    base = progress_signature(_ctx_base())
    v = copy.deepcopy(_ctx_base())
    v["turn"] = 99
    v["own"]["p1a"]["moves"] = ["ember", "protect", "protect"]
    v["own"]["p1a"]["used_protect"] = False
    v["own"]["p1a"]["hit_this_turn"] = True
    v["active"] = {"p1": "p1b", "p2": "p2b"}
    v["opponent"]["p2a"]["top_moves"] = ["surf", "icebeam"]
    v["opponent"]["p2a"]["posterior"] = []
    assert progress_signature(v) == base


def test_signature_invariant_to_switch_ping_pong():
    # 2026-11-23 regression (D6-draw investigation): the belief is NOT
    # switch-invariant — in singles every switch-in lands on the active field
    # slot, so the active mon's block carries real data and the benched mon's
    # block is a 0/None placeholder, and the two FLIP on every switch (two
    # mons with different maxhp, as in the real ping-pong repro). The
    # signature must be IDENTICAL in both phases, else the cap can never
    # confirm in the exact stall it exists to end (it reset forever and the
    # battle ran to the 200-step cap — the D6 battle #17 draw).
    import copy
    a = _ctx_base()
    a["own"] = {  # phase A: mon X (maxhp 386) active in p1a
        "p1a": {"hp_frac": 1.0, "statuses": [], "boosts": {}, "moves": [],
                "tera": "", "last_hit": 0, "hit_this_turn": False,
                "last_move_failed": False, "stockpile": 0,
                "used_protect": False, "volatiles": [], "maxhp": 386},
        "p1b": {"hp_frac": 0.0, "statuses": [], "boosts": {}, "moves": [],
                "tera": "", "last_hit": 0, "hit_this_turn": False,
                "last_move_failed": False, "stockpile": 0,
                "used_protect": False, "volatiles": [], "maxhp": None},
    }
    b = copy.deepcopy(a)
    b["own"] = {  # phase B: mon Y (maxhp 321) active in p1a, X benched
        "p1a": {"hp_frac": 1.0, "statuses": [], "boosts": {}, "moves": [],
                "tera": "", "last_hit": 0, "hit_this_turn": False,
                "last_move_failed": False, "stockpile": 0,
                "used_protect": False, "volatiles": [], "maxhp": 321},
        "p1b": {"hp_frac": 0.0, "statuses": [], "boosts": {}, "moves": [],
                "tera": "", "last_hit": 0, "hit_this_turn": False,
                "last_move_failed": False, "stockpile": 0,
                "used_protect": False, "volatiles": [], "maxhp": None},
    }
    b["opponent"]["p2a"]["maxhp"] = 999   # opponent side flips too
    b2 = copy.deepcopy(b)
    b2["opponent"]["p2a"]["maxhp"] = 100
    assert progress_signature(a) == progress_signature(b)
    assert progress_signature(b) == progress_signature(b2)


def test_signature_confirms_ko_and_field():
    # the new contract (2026-11-23): the signature is the belief safety net
    # for the ruling's switch-invariant clauses — a KO (fainted count) or a
    # hazard/screen/weather/room change MUST change it; per-mon HP/status/
    # boost/volatile changes are the SCANNER's job (tag list) and do NOT
    # change the signature (they flip on every switch in the belief, which is
    # exactly why they were removed from it).
    import copy
    base = progress_signature(_ctx_base())
    v = copy.deepcopy(_ctx_base()); v["fainted"] = {"p1": 1, "p2": 0}
    assert progress_signature(v) != base          # KO -> confirmed change
    v = copy.deepcopy(_ctx_base())
    v["field"]["stealthrock"] = {"p1": 1}
    assert progress_signature(v) != base          # hazard set -> change
    v = copy.deepcopy(_ctx_base())
    v["field"]["weather"] = "RainDance"
    assert progress_signature(v) != base          # weather -> change
    # per-mon state: the scanner's territory, NOT in the signature
    v = copy.deepcopy(_ctx_base()); v["own"]["p1a"]["hp_frac"] = 0.9
    assert progress_signature(v) == base
    v = copy.deepcopy(_ctx_base())
    v["opponent"]["p2a"]["statuses"] = ["par"]
    assert progress_signature(v) == base
    v = copy.deepcopy(_ctx_base()); v["own"]["p1a"]["boosts"] = {"atk": 2}
    assert progress_signature(v) == base
    v = copy.deepcopy(_ctx_base())
    v["opponent"]["p2a"]["volatiles"] = ["curse"]
    assert progress_signature(v) == base


def test_signature_none_ctx_is_none():
    assert progress_signature(None) is None


# ── _step_loop integration (fake env + injected signature) ──────────────────

def test_step_loop_cap_ends_a_protect_stall_early(monkeypatch):
    monkeypatch.setattr(driver, "_cap_signature", lambda env, fmt: ("state",))
    env = _StallEnv()
    logs: list = []
    n = _step_loop(env, lambda b: None, lambda b: None, max_steps=200,
                   no_progress_limit=3, log=logs.append)
    assert n == 3                       # tripped on the 3rd no-progress turn
    assert not env.battle1.won and not env.battle2.won   # -> a DRAW
    # (close() is play_agent_battle_on's contract, not _step_loop's)
    assert any("no-progress cap" in w for w in logs)


def test_step_loop_cap_respects_progress(monkeypatch):
    monkeypatch.setattr(driver, "_cap_signature", lambda env, fmt: ("state",))
    env = _StallEnv()
    env.battle1 = _DamageBattle()      # a -damage line every turn
    n = _step_loop(env, lambda b: None, lambda b: None, max_steps=8,
                   no_progress_limit=2)
    assert n == 8                      # the streak never builds up


def test_step_loop_cap_confirms_a_scanner_miss(monkeypatch):
    # the belief shows a state change the scanner missed -> the streak RESETS
    # and the battle keeps running (no false-positive draw)
    calls = {"n": 0}

    def _sig(env, fmt):
        calls["n"] += 1
        return ("state", calls["n"])   # a different signature every call

    monkeypatch.setattr(driver, "_cap_signature", _sig)
    env = _StallEnv()
    n = _step_loop(env, lambda b: None, lambda b: None, max_steps=10,
                   no_progress_limit=3)
    assert n == 10                     # confirmation never agreed -> no cap


def test_step_loop_cap_needs_a_confirmed_signature(monkeypatch):
    # belief unavailable (signature None) -> the cap must NOT trip (the step
    # cap backstops the stall as a draw, exactly as today)
    monkeypatch.setattr(driver, "_cap_signature", lambda env, fmt: None)
    env = _StallEnv()
    n = _step_loop(env, lambda b: None, lambda b: None, max_steps=10,
                   no_progress_limit=3)
    assert n == 10


def test_step_loop_cap_disabled():
    env = _StallEnv()
    n = _step_loop(env, lambda b: None, lambda b: None, max_steps=10,
                   no_progress_limit=0)
    assert n == 10                     # the step cap alone governs
