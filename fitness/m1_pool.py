"""M1 gate (Phase C, C1) — the opponent pool for the single-learner experiment.

SINGLE source of truth for "who does the main learner train against" (the
M1-gate handout, section 4):

  * ``random``             — the floor (random-init net)
  * ``heuristic``          — the OLD heuristic (power * eff * STAB, hazard-aware switch)
  * ``strategy``           — the NEW heuristic (``choose_action`` rules 1-4)  [the PASS yardstick]
  * ``strategy_posterior`` — strategy WITH the opponent's top-1 posterior in the calc
  * ``self_<i>``           — frozen snapshots of the main learner (as they accumulate)

The SAME references the C4 harness and the A/B baseline (``tools/heuristic_ab``)
use — no second opponent table (one-workflow). Both agent interfaces are
supported downstream by ``league.driver.policy_action`` (``decides_from_battle``
ctx brains AND ``PokeAgent`` nets), so the pool is interface-agnostic.

CPU-first: importing this module pulls NO torch — every reference builds its
agent lazily inside ``make_agent`` (the same discipline as
``fitness.ladder.random_reference`` and the other factories). The snapshot net
loads lazily AND is cached (one disk read; the frozen net is shared across
battles — inference-only, never trained, so sharing is safe under the M1 gate's
sequential CPU play). Each battle still gets a FRESH agent (fresh RNG per
battle — the league's per-battle isolation contract).
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from .ladder import Ladder, Reference

__all__ = ["M1_CORE_NAMES", "snapshot_reference", "build_m1_pool"]

#: the 4 standing (non-snapshot) pool members, in ladder order.
M1_CORE_NAMES = ("random", "heuristic", "strategy", "strategy_posterior")


def _engine_key(fmt: str) -> int:
    """The @smogon/calc engine key for a format id.

    ``legality.formats`` is the source of truth (D32: champions = 0, NOT 9);
    the fallback is LOUD (``engine_key_or_warn``, B1) — no silent mis-keying.
    Lazy import keeps this module cheap at import time.
    """
    from legality.formats import engine_key_or_warn
    return int(engine_key_or_warn(fmt, default=9))


def snapshot_reference(path, vocabs: Any, device: str = "cpu",
                       name: Optional[str] = None,
                       weight: float = 1.0) -> Reference:
    """A FROZEN self-snapshot checkpoint as a pool member.

    Loads the net LAZILY (first ``make_agent``) and CACHES it (one disk read;
    the frozen net is shared across battles — inference-only, never trained, so
    sharing is safe even with the M1 gate's thread-per-battle workers: the net
    is read-only during collection). Each battle still gets a FRESH
    :class:`~net.agent.PokeAgent` (fresh RNG per battle).
    """
    path = os.fspath(path)
    cache: dict = {}

    def make_agent(seed: Optional[int] = None) -> Any:
        net = cache.get("net")
        if net is None:
            from net.model import PokeNet
            net = PokeNet.load_file(path, vocabs, device=device)
            cache["net"] = net
        from net.agent import PokeAgent
        return PokeAgent(net, seed=seed, device=device)

    return Reference(
        name=name or os.path.basename(path),
        make_agent=make_agent, weight=weight,
        description=f"frozen self-snapshot ({path})")


def build_m1_pool(fmt: str = "gen9ou", vocabs: Any = None, device: str = "cpu",
                  snapshot_paths: Optional[Iterable[str]] = None) -> Ladder:
    """Assemble the M1-gate opponent pool as a :class:`~fitness.ladder.Ladder`.

    The 4 standing references (random / heuristic / strategy /
    strategy_posterior) + one ``self_<i>`` reference per ``snapshot_path``
    (in order, ``i`` starting at 1). ``vocabs`` is required by the net
    references; if omitted it is built (CPU) — pass it to avoid rebuilding per
    call. The ladder is ordered (``Ladder`` preserves insertion order).
    """
    if vocabs is None:
        from net.vocab_db import build_vocabs
        vocabs = build_vocabs()

    from .ladder import random_reference
    from .heuristic import heuristic_reference
    from .strategy_agent import strategy_reference, strategy_posterior_reference

    ladder = Ladder()
    ladder.add(random_reference(vocabs, device=device))
    ladder.add(heuristic_reference(gen=_engine_key(fmt)))
    ladder.add(strategy_reference(fmt=fmt))
    ladder.add(strategy_posterior_reference(fmt=fmt))

    for i, p in enumerate(snapshot_paths or ()):
        ladder.add(snapshot_reference(p, vocabs, device=device, name=f"self_{i + 1}"))
    return ladder
