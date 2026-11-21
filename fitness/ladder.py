"""The reference ladder — the fixed opponents a bot is scored against.

A *reference* is a named, factory-produced policy (a ``PokeAgent``-compatible
object exposing ``.act(obs, actions, deterministic)``) plus a fitness *weight*
(stronger references count more in the aggregate score). The ladder is the
"ladder of increasingly-strong opponents" a bot climbs to get an ABSOLUTE
strength number, not just relative ELO within the league.

The container is framework-agnostic: it stores ``make_agent`` callables and
only invokes them when a battle is played, so importing this module never pulls
torch (the CPU-first discipline shared by ``pool`` / ``league``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Reference", "Ladder", "random_reference"]


@dataclass
class Reference:
    """One opponent on the ladder.

    ``make_agent(seed) -> agent``: build a FRESH policy for a single battle
    (fresh RNG per battle — the league's per-battle isolation contract).
    ``weight`` scales this reference's win-rate in the aggregate fitness score
    (stronger references should weigh more, e.g. random=1, heuristic=2,
    champion=3).
    """
    name: str
    make_agent: Callable[[Optional[int]], Any]
    weight: float = 1.0
    description: str = ""

    def agent(self, seed: Optional[int] = None) -> Any:
        """A fresh agent for one battle (delegates to ``make_agent``)."""
        return self.make_agent(seed)


class Ladder:
    """An ordered set of references (the opponents a bot is scored against)."""

    def __init__(self, refs: Optional[List[Reference]] = None):
        self._refs: Dict[str, Reference] = {}
        for r in (refs or []):
            self._refs[r.name] = r

    def add(self, ref: Reference) -> "Ladder":
        self._refs[ref.name] = ref
        return self

    def get(self, name: str) -> Reference:
        if name not in self._refs:
            raise KeyError(f"unknown reference {name!r} (have {list(self._refs)})")
        return self._refs[name]

    def names(self) -> List[str]:
        return list(self._refs)

    def all(self) -> List[Reference]:
        return list(self._refs.values())

    def __len__(self) -> int:
        return len(self._refs)

    def __iter__(self):
        return iter(self._refs.values())

    def __contains__(self, name: str) -> bool:
        return name in self._refs


def random_reference(vocabs: Any, device: str = "cpu") -> Reference:
    """The FLOOR reference: a random-init net. A competent bot should beat it
    well above 50% (it plays legal moves but with no strategy)."""

    def make_agent(seed: Optional[int] = None) -> Any:
        from net.agent import make_random_net, PokeAgent

        net = make_random_net(vocabs, device=device, seed=seed)
        return PokeAgent(net, seed=seed, device=device)

    return Reference(
        name="random", make_agent=make_agent, weight=1.0,
        description="random-init net (the floor; a competent bot beats this >50%)")
