"""M1 — MCTS (PUCT) search core: the forward-looking engine.

WHY THIS EXISTS
---------------
The policy+value net is SINGLE-STEP: it maps a state to one action and a value,
but it never LOOKS AHEAD. The long-term goal scenarios all require lookahead:

  * "realize a win condition"  — Dragonite can DDance + 6-0 once the tank is at
    60% AND the speed-resist (Gengar) is dead. (multi-condition, multi-turn)
  * "evaluate a double switch" — my switch AND the opponent's response change
    the expected win rate. (counterfactual)
  * "spot the opponent's setup line" — a free Swords Dance enables a sweep.

MCTS (Monte-Carlo Tree Search) is the engine that adds LOOKAHEAD on top of the
existing net: it searches a tree of future states, using the net's policy head
for PRIORS (where to look) and its value head as the LEAF eval (how good a
state is), with a pluggable :class:`WorldModel` providing the actual game
transitions (the Showdown sim + damage calc — wired in M2).

DESIGN (net-agnostic + testable WITHOUT a net or a live sim)
------------------------------------------------------------
* :class:`WorldModel` is a tiny interface: ``actions(state)``,
  ``step(state, action) -> (next, reward, done)``, ``value(state)``,
  ``policy(state, actions)``. The real one wraps the sim + damage calc +
  PokeNet (M2); a mock one powers the unit tests. MCTS itself is game-agnostic.
* Reward space: the SAME (−1, +1) zero-sum convention as ``net/ppo.py``
  (win +1 / loss −1 / draw −1, 2026-11-23 ruling). The net's value head
  emits a (0,1) win-prob and
  is mapped to reward space as ``2V − 1`` at the WorldModel boundary (M2) —
  identical to ppo.py's contract, so there is NO second convention.
* Selection: **PUCT** (AlphaZero's UCB-with-prior):
  ``score = Q(s,a) + cpuct · P(s,a) · sqrt(N(s)) / (1 + N(s,a))``, maximize.
  Q = the child's value, P = the policy prior, N = visits.
* Value perspective: a node's value is stored from the LAST-MOVER's perspective
  (the player who moved to reach it), so the selector ALWAYS maximizes — correct
  for both players. The backup flips the sign one step at a time from the
  root-player-perspective terminal reward.
* Deterministic: no sampling in the core (the net's action sampling is a policy
  concern, not a search concern). ``best_action`` returns the most-visited
  child — the search's preferred action (robust, standard MCTS).

SCOPE (deliberately NOT all-at-once)
------------------------------------
This is the FOUNDATION: the search core + the WorldModel contract, verified with
a mock world model (no GPU, no live battle, no net). Wiring the REAL sim +
damage calc + PokeNet as the WorldModel is M2; surfacing the value head as an
interpretable state evaluation is M3; the derived "reasoning" observation signals
are M4. Each is a separate, tested step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence, Tuple


@dataclass
class Node:
    """One search-tree node = one game state.

    ``value`` is accumulated over the simulations that passed through it
    (PUCT backup). It is stored from the LAST-MOVER's perspective (see module
    docstring) so selection is always "maximize." ``terminal_reward`` is the
    terminal outcome from the ROOT player's perspective (win +1 / loss −1).
    """
    state: Any
    prior: float = 0.0
    parent: Optional["Node"] = None
    action: Optional[Any] = None          # the action taken to REACH this node
    depth: int = 0
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    terminal: bool = False
    terminal_reward: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class WorldModel(Protocol):
    """The game transitions + the net's heads, as callable oracles.

    A ``state`` is anything the WorldModel understands (a belief context, an
    encoded obs, a sim snapshot). The WorldModel is the ONLY thing that knows
    how to advance the game — so :class:`MCTS` is game- and net-agnostic.

    ``step``'s reward and ``value``/``policy`` are from the ROOT (deciding)
    player's perspective: ``value(state)`` = expected outcome in (−1, +1);
    ``policy`` = unnormalized priors over the actions (MCTS normalizes).
    """

    def actions(self, state: Any) -> List[Any]:
        """The legal actions at ``state`` (the D60 legal list). Empty = terminal."""
        ...

    def step(self, state: Any, action: Any) -> Tuple[Any, float, bool]:
        """Apply ``action`` -> (next_state, reward_from_root_view, done)."""
        ...

    def value(self, state: Any) -> float:
        """The value-head leaf eval for ``state``, in (−1, +1)."""
        ...

    def policy(self, state: Any, actions: Sequence[Any]) -> List[float]:
        """The policy-head PRIORS over ``actions`` (unnormalized ok)."""
        ...


def _puuct(node: Node, child: Node, cpuct: float) -> float:
    """PUCT score = Q + cpuct · P · sqrt(N) / (1 + N_child). Maximize."""
    return (child.value
            + cpuct * child.prior * math.sqrt(max(node.visits, 1)) / (1.0 + child.visits))


class MCTS:
    """PUCT Monte-Carlo Tree Search over a :class:`WorldModel`.

    ``best_action(state)`` runs ``iterations`` searches from ``state`` and
    returns the most-visited child — the action the search prefers.
    """

    def __init__(self, wm: WorldModel, iterations: int = 600,
                 cpuct: float = 1.5, max_depth: Optional[int] = None) -> None:
        self.wm = wm
        self.iterations = max(1, int(iterations))
        self.cpuct = cpuct
        self.max_depth = max_depth

    # ── expansion ────────────────────────────────────────────────────
    def _expand(self, node: Node) -> None:
        acts = self.wm.actions(node.state)
        if not acts:
            node.terminal = True
            return
        priors = self.wm.policy(node.state, acts)
        if not priors:
            priors = [1.0] * len(acts)      # null prior -> uniform over the legal set
        if len(priors) != len(acts):
            # A mismatched prior list would silently DROP legal actions via the
            # zip below (a silent-failure mode) — fail LOUDLY instead. This also
            # guards the real M2b world model: its policy must match its actions,
            # so a slip corrupts the search visibly, not by quietly losing moves.
            raise ValueError(
                f"WorldModel.policy returned {len(priors)} priors for {len(acts)} "
                f"actions (the prior list must match the action list)")
        total = float(sum(priors)) or 1.0
        for a, p in zip(acts, priors):
            nxt, reward, done = self.wm.step(node.state, a)
            node.children.append(
                Node(state=nxt, prior=p / total, parent=node, action=a,
                     depth=node.depth + 1, terminal=done, terminal_reward=reward))
        node.terminal = False

    # ── selection ────────────────────────────────────────────────────
    def _select(self, node: Node) -> Node:
        if node.terminal:
            return node
        if self.max_depth is not None and node.depth >= self.max_depth:
            return node
        if not node.children:
            self._expand(node)
        if not node.children:
            return node  # terminal (no actions) or a dead end
        best, best_score = None, float("-inf")
        for c in node.children:
            s = _puuct(node, c, self.cpuct)
            if s > best_score:
                best, best_score = c, s
        assert best is not None
        return self._select(best)

    # ── value + backup ───────────────────────────────────────────────
    def _leaf_value(self, node: Node) -> float:
        if node.terminal:
            return float(node.terminal_reward)      # already root-player view
        return float(self.wm.value(node.state))     # WorldModel gives root view

    def _backup(self, node: Node, leaf_value: float) -> None:
        """PUCT backup with the last-mover perspective flip.

        ``leaf_value`` is the terminal reward from the ROOT player's perspective.
        A node at depth ``d`` was reached by the root player's move iff ``d`` is
        odd (the root player moves first), so its value is ``+leaf_value`` when
        odd and ``−leaf_value`` when even; each step up the tree the last-mover
        flips, so the sign flips too.
        """
        v = leaf_value if (node.depth % 2 == 1) else -leaf_value
        while node is not None:
            node.visits += 1
            node.value_sum += v
            node = node.parent
            v = -v

    # ── the search ───────────────────────────────────────────────────
    def best_action(self, state: Any) -> Optional[Any]:
        root = Node(state=state, depth=0)
        self._expand(root)
        if not root.children:
            return None  # no legal action (pass/wait)
        for _ in range(self.iterations):
            leaf = self._select(root)
            self._backup(leaf, self._leaf_value(leaf))  # walk UP from the leaf
        return max(root.children, key=lambda c: c.visits).action

    def root_values(self, state: Any) -> List[Tuple[Any, float]]:
        """Run the search and return the root's children as (action, value)
        pairs sorted best-first (the search's preference ranking). Useful for
        M3 (state/action interpretation) and for debugging the search."""
        root = Node(state=state, depth=0)
        self._expand(root)
        for _ in range(self.iterations):
            leaf = self._select(root)
            self._backup(leaf, self._leaf_value(leaf))  # walk UP from the leaf
        ranked = sorted(((c.action, c.value) for c in root.children),
                        key=lambda x: -x[1])
        return ranked
