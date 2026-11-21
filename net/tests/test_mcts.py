"""M1 — tests for the MCTS (PUCT) search core.

These prove the SEARCH works (the forward-looking engine for the long-term
scenarios), WITHOUT a net or a live sim: a mock :class:`WorldModel` supplies
deterministic transitions + a deliberately MISLEADING policy prior, so only a
correct LOOKAHEAD search can pick the right action. A single-step / 1-step-greedy
net (argmax the policy prior) fails these — which is exactly the gap MCTS closes.

Run:  CUDA_VISIBLE_DEVICES=-1 python -m pytest net/tests/test_mcts.py -v
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pytest

from net.mcts import MCTS


# ── mock world models (deterministic, net-agnostic) ────────────────────────

class CorridorWorld:
    """A 2-ply "corridor": the agent picks an action; the opponent has ONE
    fixed response; then a terminal reward (from the agent's view) is delivered.

    The policy prior at the root is deliberately MISLEADING (it favors the
    trap action), so a 1-step-greedy net (argmax prior) picks the wrong move
    and only a lookahead search picks the right one — the Dragonite analog:
    "the move that looks best now (A) is actually a trap; the setup move (B)
    wins once you look one turn ahead."
    """

    def __init__(self, rewards: Dict[str, float], prior: Dict[str, float]) -> None:
        self.rewards = rewards      # agent action -> terminal reward (agent view)
        self.prior = prior          # agent action -> policy prior (misleading)

    def actions(self, state: Any) -> List[Any]:
        if state == "root":
            return list(self.rewards.keys())
        if isinstance(state, str) and state.endswith("_opp"):
            return ["opp_response"]
        return []  # terminal

    def step(self, state: Any, action: Any) -> Tuple[Any, float, bool]:
        if state == "root":
            return f"{action}_opp", 0.0, False
        if isinstance(state, str) and state.endswith("_opp"):
            act = state[: -len("_opp")]
            return f"term_{act}", float(self.rewards[act]), True
        raise ValueError(f"CorridorWorld.step: unexpected terminal {state!r}")

    def value(self, state: Any) -> float:
        return 0.0  # uninformative leaf — the mock relies on REACHING terminals

    def policy(self, state: Any, actions: Sequence[Any]) -> List[float]:
        if state == "root":
            return [self.prior.get(a, 0.3) for a in actions]
        return [1.0] * len(actions)


class BranchingWorld:
    """A 2-ply tree where the OPPONENT branches (the double-switch analog):
    the right agent move is the one that wins even against the opponent's BEST
    response. A 1-step net that doesn't search the opponent's reply cannot see
    that one of the agent's moves is exploitable."""

    def __init__(self, table: Dict[str, Dict[str, float]], prior: Dict[str, float]) -> None:
        # table[agent_action][opp_response] = terminal reward (agent view)
        self.table = table
        self.prior = prior
        self.opp_pref = {a: sorted(t, key=lambda o: -t[o]) for a, t in table.items()}

    def actions(self, state: Any) -> List[Any]:
        if state == "root":
            return list(self.table.keys())
        if isinstance(state, str) and state in self.table:
            return list(self.table[state].keys())
        return []  # terminal

    def step(self, state: Any, action: Any) -> Tuple[Any, float, bool]:
        if state == "root":
            return action, 0.0, False  # agent's action becomes the opponent's state
        if isinstance(state, str) and state in self.table:
            return f"term_{state}_{action}", float(self.table[state][action]), True
        raise ValueError(f"BranchingWorld.step: unexpected terminal {state!r}")

    def value(self, state: Any) -> float:
        return 0.0

    def policy(self, state: Any, actions: Sequence[Any]) -> List[float]:
        if state == "root":
            return [self.prior.get(a, 0.4) for a in actions]
        # opponent policy: it prefers its best response (most negative for agent)
        pref = self.opp_pref.get(state, [a for a in actions])
        rank = {o: i for i, o in enumerate(reversed(pref))}
        return [10.0 ** rank.get(a, 0) for a in actions]


class ValueLeafWorld:
    """A 1-ply world where the next states are NON-terminal and the search is
    cut at ``max_depth`` — so the leaf value must come from ``wm.value`` (the
    value-head contract). Proves the value head is used as the leaf eval."""

    def __init__(self, values: Dict[str, float]) -> None:
        self.values = values

    def actions(self, state: Any) -> List[Any]:
        return ["X", "Y"] if state == "root" else []

    def step(self, state: Any, action: Any) -> Tuple[Any, float, bool]:
        assert state == "root"
        return f"state_{action}", 0.0, False  # non-terminal

    def value(self, state: Any) -> float:
        return float(self.values.get(state, 0.0))

    def policy(self, state: Any, actions: Sequence[Any]) -> List[float]:
        return [1.0] * len(actions)


# ── M1.1 — lookahead beats 1-step greedy (the Dragonite win-condition analog) ──

def test_mcts_beats_one_step_greedy_on_a_lookahead_trap():
    # A is the TRAP (looks best by prior, but loses); B is the setup move (wins
    # when you look one turn ahead); C loses. The 1-step-greedy net (argmax the
    # prior) picks A. Only a lookahead search picks B.
    wm = CorridorWorld(rewards={"A": -1.0, "B": +1.0, "C": -1.0},
                       prior={"A": 0.6, "B": 0.2, "C": 0.2})
    # the 1-step-greedy baseline (argmax the policy prior) picks the trap A:
    greedy = max(wm.prior, key=wm.prior.get)
    assert greedy == "A", "test setup: the misleading prior must favor the trap"

    got = MCTS(wm, iterations=400).best_action("root")
    assert got == "B", f"MCTS should look ahead and pick the winning move B, got {got}"


def test_mcts_root_values_rank_the_winning_move_first():
    wm = CorridorWorld(rewards={"A": -1.0, "B": +1.0, "C": -1.0},
                       prior={"A": 0.6, "B": 0.2, "C": 0.2})
    ranked = MCTS(wm, iterations=400).root_values("root")
    assert ranked[0][0] == "B"
    # the trap A and the losing C should both be ranked BELOW the winner B:
    by_action = {a: v for a, v in ranked}
    assert by_action["B"] > by_action["A"]
    assert by_action["B"] > by_action["C"]


# ── M1.2 — search the OPPONENT's response (the double-switch analog) ────────

def test_mcts_searches_opponent_response_and_avoids_the_exploitable_move():
    # "switch" wins against BOTH opponent replies; "attack" wins only if the
    # opponent plays a weak reply (their best reply KOs me). The correct move
    # requires evaluating the opponent's counter — a 1-step net misses it.
    table = {
        "switch": {"opp_weak": +1.0, "opp_best": +1.0},   # always wins
        "attack": {"opp_weak": +1.0, "opp_best": -1.0},   # exploitable
    }
    wm = BranchingWorld(table, prior={"attack": 0.7, "switch": 0.3})
    greedy = max(wm.prior, key=wm.prior.get)
    assert greedy == "attack", "test setup: the misleading prior must favor attack"

    got = MCTS(wm, iterations=800).best_action("root")
    assert got == "switch", f"MCTS should see the exploitable reply and pick switch, got {got}"


# ── M1.3 — the value head is used as the leaf eval (M3 contract) ────────────

def test_mcts_uses_value_head_as_leaf_when_search_is_cut_short():
    # The next states are non-terminal and max_depth cuts the search, so the
    # leaf value must come from wm.value (the value-head contract, in (-1,+1)).
    wm = ValueLeafWorld(values={"state_X": +0.5, "state_Y": -0.5})
    got = MCTS(wm, iterations=200, max_depth=1).best_action("root")
    assert got == "X", f"MCTS should pick the higher value-leaf state, got {got}"


# ── M1.4 — contracts ────────────────────────────────────────────────────────

def test_no_legal_action_returns_none():
    class NoActions:
        def actions(self, state):
            return []
        def step(self, state, action):
            raise AssertionError("step must not be called with no actions")
        def value(self, state):
            return 0.0
        def policy(self, state, actions):
            return []
    assert MCTS(NoActions(), iterations=50).best_action("root") is None


def test_best_action_is_deterministic():
    wm = CorridorWorld(rewards={"A": -1.0, "B": +1.0, "C": -1.0},
                       prior={"A": 0.6, "B": 0.2, "C": 0.2})
    picks = {MCTS(wm, iterations=300).best_action("root") for _ in range(5)}
    assert picks == {"B"}, f"the search must be deterministic, got {picks}"


def test_mcts_never_returns_an_illegal_action():
    wm = CorridorWorld(rewards={"A": -1.0, "B": +1.0, "C": -1.0},
                       prior={"A": 0.6, "B": 0.2, "C": 0.2})
    got = MCTS(wm, iterations=300).best_action("root")
    assert got in set(wm.actions("root")), f"{got!r} is not a legal root action"


def test_policy_length_mismatch_fails_loudly_not_silently_drops_actions():
    """A WorldModel whose policy returns the WRONG number of priors must fail
    LOUDLY (ValueError), not silently drop legal actions via the expand zip.
    This is the 'no silent failures' guard on the search core, and it protects
    the real M2b world model (its policy must match its action list)."""
    class BadPriorWorld:
        def actions(self, state):
            return ["a", "b", "c"]
        def step(self, state, action):
            return "term", 0.0, True
        def value(self, state):
            return 0.0
        def policy(self, state, actions):
            return [0.5, 0.5]  # 2 priors for 3 actions -> contract violation
    with pytest.raises(ValueError, match="priors"):
        MCTS(BadPriorWorld(), iterations=4).best_action("root")
