"""Team pools for OU-format self-play + random team builder.

The pools are **server-verified**: ``bot/teamgen.py`` validates candidates against
the live server's own TeamValidator (``/vtm``) per format and caches the result in
``bot/team_pools.json``. A pool verified there is guaranteed to be accepted in real
battles. ``RandomTeamBuilder(battle_format)`` draws a fresh random 6-mon team from
the format's pool on every ``yield_team()`` call; poke-env calls it once per agent,
so each side independently draws its own team.

Random-battle formats need no team (the server generates them) -- pass ``team=None``
to the env for those.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

from poke_env.teambuilder import Teambuilder, TeambuilderPokemon

HERE = os.path.dirname(os.path.abspath(__file__))
POOLS_PATH = os.path.join(HERE, "team_pools.json")
N_TEAM = 6


def load_pools() -> Dict[str, List[Dict]]:
    if not os.path.exists(POOLS_PATH):
        raise RuntimeError(
            f"no team pools found at {POOLS_PATH} -- run `python bot/teamgen.py` first"
        )
    with open(POOLS_PATH, encoding="utf8") as fh:
        return json.load(fh)


def load_verified_pool(battle_format: str) -> List[Dict]:
    pools = load_pools()
    if battle_format not in pools:
        raise RuntimeError(
            f"no verified team pool for '{battle_format}' -- "
            f"run `python bot/teamgen.py --formats {battle_format}`"
        )
    return pools[battle_format]


def _to_tb(mon: Dict) -> TeambuilderPokemon:
    return TeambuilderPokemon(
        species=mon["species"],
        item=mon["item"],
        ability=mon["ability"],
        moves=list(mon["moves"]),
        nature=mon["nature"],
        evs=list(mon["evs"]),
        ivs=[31, 31, 31, 31, 31, 31],
    )


class RandomTeamBuilder(Teambuilder):
    """Yields a fresh random N-mon team from the format's verified pool per call."""

    def __init__(self, battle_format: str, seed: Optional[int] = None,
                 n: int = N_TEAM):
        pool = load_verified_pool(battle_format)
        if len(pool) < n:
            raise ValueError(
                f"pool for {battle_format} has {len(pool)} mons, need {n} -- "
                f"run `python bot/teamgen.py --refresh`"
            )
        self.battle_format = battle_format
        self.pool = pool
        self.n = n
        self._rng = random.Random(seed)

    def yield_team(self) -> str:
        picks = self._rng.sample(self.pool, self.n)
        return Teambuilder.join_team([_to_tb(m) for m in picks])


class FixedTeamBuilder(Teambuilder):
    """Always yields the same team (first N mons of the pool). Useful for debugging."""

    def __init__(self, battle_format: str, n: int = N_TEAM):
        pool = load_verified_pool(battle_format)
        self.battle_format = battle_format
        self.team = [m for m in pool[:n]]

    def yield_team(self) -> str:
        return Teambuilder.join_team([_to_tb(m) for m in self.team])


def random_team_string(battle_format: str, n: int = N_TEAM,
                       seed: Optional[int] = None) -> str:
    """One random team from the format's verified pool, as a packed showdown string."""
    return RandomTeamBuilder(battle_format, seed, n).yield_team()
