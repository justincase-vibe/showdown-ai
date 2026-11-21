"""A3.1 — the ``HeuristicAgent``: a strong, DETERMINISTIC, CPU-first policy.

It is a ``PokeAgent``-compatible drop-in — ``.act(obs, acts, deterministic)``
returns a ``Decision``-shaped object — so it flows through the SAME
``league.driver.policy_action`` path the net uses. The harness therefore plays
exactly what the heuristic does: no second decision pipeline, no code twice.

Design (CPU-first / single source of truth):
* Decodes structured state from ``obs`` using the S4.1 **registry** indices
  (``own_slot`` active/present/hp_frac) — never hand-maintained column numbers.
* Static move data from the belief state's OWN lookup
  (``legality.lookup.get_move(name, gen)``) — no second move table.
* **Lazy imports** (like ``fitness.ladder.random_reference``): importing this
  module pulls NO torch; torch/registry/legality are touched only inside
  ``.act`` (the module stays CPU-first, torch is only imported when a battle is
  actually played — the same discipline as the rest of ``fitness/``).

A3.1 behaviour (legal + sensible baseline; the >70%-beats-random gate is A3.3):
* empty ``acts``                        -> pass (``action=None``)
* no moves legal (force-switch)         -> pick a switch
* own active HP <= ``switch_threshold`` and a switch is legal -> switch
* else                                  -> the highest-base-power damaging move
                                            (status / unknown-power moves rank last)

Deterministic: identical ``(obs, acts)`` always yields the identical action —
a strong reference should be a reproducible opponent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["HeuristicAgent", "Decision"]


@dataclass
class Decision:
    """A ``net.agent.Decision``-SHAPED return (duck-typed by the driver).

    Defined here (not imported from ``net.agent``) so ``import fitness.heuristic``
    stays torch-free. The play path reads only ``.action`` (driver) and
    ``.action_index`` / ``.logprob`` / ``.value`` (``league.collect``); ``.prob`` /
    ``.entropy`` are present for shape-consistency with the net's Decision.
    """
    action: Optional[Dict]
    action_index: int
    logprob: float
    value: float
    entropy: float
    prob: np.ndarray


class HeuristicAgent:
    """A deterministic rule-based opponent for the fitness ladder.

    ``gen`` is the @smogon/calc ENGINE KEY passed to ``legality.lookup.get_move``
    (0 = champions, 1-9 = gen) so move power/accuracy are correct per format.
    Defaults to 9 (the primary format). ``seed`` is accepted to satisfy the
    ``Reference.make_agent(seed)`` contract but is unused — the policy is fully
    deterministic.
    """

    def __init__(self, gen: int = 9, switch_threshold: float = 0.30,
                 seed: Optional[int] = None) -> None:
        self.gen = int(gen)
        self.switch_threshold = float(switch_threshold)
        self.seed = seed

    # ── obs decoding (defensive: any missing/malformed field just degrades) ──
    def _my_active_hp(self, obs: Any) -> Optional[float]:
        """The active own-mon's HP fraction in [0,1], or None if undecodable."""
        slot = obs.get("own_slot") if isinstance(obs, dict) else None
        if slot is None:
            return None
        slot = np.asarray(slot)
        if slot.ndim != 2:
            return None
        try:
            from net.registry import SLOT_INDEX
            active_i, present_i, hp_i = (SLOT_INDEX["active"],
                                        SLOT_INDEX["present"],
                                        SLOT_INDEX["hp_frac"])
        except Exception:  # registry unavailable -> cannot read, degrade
            return None
        if slot.shape[1] <= max(active_i, present_i, hp_i):
            return None
        active_rows = [i for i in range(slot.shape[0]) if float(slot[i, active_i]) > 0.5]
        if not active_rows:
            return None
        a = active_rows[0]
        if float(slot[a, present_i]) < 0.5:
            return None
        return float(slot[a, hp_i])

    # ── move scoring (the belief state's own lookup; no second table) ──
    def _move_score(self, act: Dict) -> int:
        """A damaging move ranks by base power; status / unknown rank last (0)."""
        mid = act.get("move")
        if not mid:
            return 0
        try:
            from legality.lookup import get_move
            m = get_move(mid, self.gen)
        except Exception:  # unknown move -> rank last, but still legal
            return 0
        if (m.get("category") or "").lower() == "status":
            return 0
        return int(m.get("power") or 0)

    # ── the policy ─────────────────────────────────────────────────────
    def act(self, obs: Any, action_list: Optional[List[Dict]],
            deterministic: bool = False) -> Decision:
        """Choose one legal action (deterministic). ``deterministic`` is accepted
        for contract parity but the policy is always deterministic."""
        acts: List[Dict] = list(action_list or [])
        if not acts:
            return self._decision(acts, -1)

        moves = [a for a in acts if a.get("kind") in ("move", "tera")]
        switches = [a for a in acts if a.get("kind") == "switch"]

        # 1) force-switch: only switches are legal.
        if not moves:
            return self._decision(acts, acts.index(switches[0]))

        # 2) critically low + can switch -> bail before we faint for nothing.
        hp = self._my_active_hp(obs)
        if switches and hp is not None and hp <= self.switch_threshold:
            return self._decision(acts, acts.index(switches[0]))

        # 3) play the best damaging move (highest base power; ties -> first).
        best = max(moves, key=self._move_score)
        return self._decision(acts, acts.index(best))

    # ── decision packing ─────────────────────────────────────────────────
    def _decision(self, acts: List[Dict], idx: int) -> Decision:
        if idx < 0:  # a pass/wait turn
            return Decision(None, -1, 0.0, 0.5, 0.0,
                            np.zeros(0, dtype=np.float32))
        prob = np.zeros(len(acts), dtype=np.float32)
        prob[idx] = 1.0  # a deterministic one-hot (logprob 0.0, entropy 0.0)
        return Decision(acts[idx], idx, 0.0, 0.5, 0.0, prob)


def heuristic_reference(gen: int = 9) -> "Reference":  # noqa: F821 (lazy)
    """Build the mid-ladder ``heuristic`` reference (weight 2, above the random
    floor). Lazy so this import stays cheap; mirrors ``random_reference``.
    """
    from .ladder import Reference

    def make_agent(seed: Optional[int] = None) -> HeuristicAgent:
        return HeuristicAgent(gen=gen, seed=seed)

    return Reference(
        name="heuristic", make_agent=make_agent, weight=2.0,
        description="deterministic rule-based opponent (switch-when-low + "
                    "highest-power move); a strong bot beats this most of the time")
