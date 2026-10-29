"""Team pools for multi-format self-play.

Two team-drawing modes, both **server-verified** (the local Showdown server's own
TeamValidator accepted every team at build time, so they are guaranteed to be
accepted in real battles):

* ``StrongTeamBuilder`` -- yields one **coherent** curated team (the real 6 that go
  together). This is what teaches *team synergy* and is what self-play should face.
  ``bot/smogon_teams.py`` builds these from Smogon's council sample teams (+ a few
  community teams to top up) and stores them per format in ``team_pools.json``.
* ``RandomTeamBuilder`` -- draws a fresh **random 6-mon bag** from the format's flat
  mon pool. Kept as the *weak* baseline to contrast against the strong teams later
  (the synergy/evaluation tool the user deferred).

Pool file layout (``bot/team_pools.json``). New formats are a dict::

    { "teams": [ [mon x6], ... ],     # coherent teams (StrongTeamBuilder)
      "flat":  [ mon, mon, ... ],     # union of every mon (RandomTeamBuilder)
      "meta":  { "provenance": {"sample": a, "community": b}, ... } }

Legacy gen6ou/gen7ou pools may still be a bare ``[mon, ...]`` list; ``load_verified_pool``
handles both, so nothing breaks.

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


def load_pools() -> Dict:
    if not os.path.exists(POOLS_PATH):
        raise RuntimeError(
            f"no team pools found at {POOLS_PATH} -- "
            f"run `python bot/smogon_teams.py` first"
        )
    with open(POOLS_PATH, encoding="utf8") as fh:
        return json.load(fh)


def _entry(battle_format: str) -> Dict:
    pools = load_pools()
    if battle_format not in pools:
        raise RuntimeError(
            f"no verified team pool for '{battle_format}' -- "
            f"run `python bot/smogon_teams.py --formats {battle_format}`"
        )
    return pools[battle_format]


def _dedupe_by_species(mons: List[Dict]) -> List[Dict]:
    """Drop duplicate *species* from a flat mon pool (first occurrence wins).

    The stored ``flat`` pools are the concatenation of every mon from every coherent
    team, so a popular species (Garchomp, Landorus, …) appears several times. A random
    *bag* that draws two of the same species is **illegal** (Species Clause) and the
    server rejects it at battle start -- which poke-env surfaces as a misleading
    ``asyncio.TimeoutError: Agent is not challenging`` (the challenge never completes
    because the team is bounced). Deduping by species guarantees any N-subset is a
    legal team, since each distinct species came from an already-legal team.
    """
    seen: set = set()
    out: List[Dict] = []
    for m in mons:
        key = str(m.get("species", "")).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def load_verified_pool(battle_format: str) -> List[Dict]:
    """Flat list of **unique-species** legal mons for the format.

    (back-compat with bare-list pools; also deduped so random-bag teams stay legal).
    """
    entry = _entry(battle_format)
    if isinstance(entry, dict):
        return _dedupe_by_species(entry.get("flat", []))
    return _dedupe_by_species(entry)  # legacy bare list


def load_teams(battle_format: str) -> List[List[Dict]]:
    """Coherent curated teams for the format (empty list if none available)."""
    entry = _entry(battle_format)
    if isinstance(entry, dict):
        return entry.get("teams", [])
    # legacy bare list has no coherent teams
    return []


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


class StrongTeamBuilder(Teambuilder):
    """Yields one **coherent** curated team per ``yield_team()`` call.

    poke-env calls ``yield_team()`` once per agent, so each side independently draws
    a full coherent 6-mon team (not a random bag). This is the strong baseline for
    self-play.
    """

    def __init__(self, battle_format: str, seed: Optional[int] = None):
        teams = load_teams(battle_format)
        if not teams:
            raise RuntimeError(
                f"no coherent teams for '{battle_format}' -- "
                f"run `python bot/smogon_teams.py --formats {battle_format}`"
            )
        self.battle_format = battle_format
        self.teams = teams
        self._rng = random.Random(seed)

    def yield_team(self) -> str:
        team = self._rng.choice(self.teams)
        return Teambuilder.join_team([_to_tb(m) for m in team])


class RandomTeamBuilder(Teambuilder):
    """Yields a fresh random N-mon bag from the format's flat pool per call.

    The *weak* baseline -- kept to contrast against ``StrongTeamBuilder`` for the
    synergy/evaluation tool.
    """

    def __init__(self, battle_format: str, seed: Optional[int] = None,
                 n: int = N_TEAM):
        pool = load_verified_pool(battle_format)
        if len(pool) < n:
            raise ValueError(
                f"flat pool for {battle_format} has {len(pool)} mons, need {n} -- "
                f"run `python bot/smogon_teams.py --refresh`"
            )
        self.battle_format = battle_format
        self.pool = pool
        self.n = n
        self._rng = random.Random(seed)

    def yield_team(self) -> str:
        picks = self._rng.sample(self.pool, self.n)
        return Teambuilder.join_team([_to_tb(m) for m in picks])


class FixedTeamBuilder(Teambuilder):
    """Always yields the same coherent team (first one in the pool). For debugging.

    Falls back to the first N flat mons if the format has no coherent teams.
    """

    def __init__(self, battle_format: str, n: int = N_TEAM):
        self.battle_format = battle_format
        teams = load_teams(battle_format)
        if teams:
            self.team = list(teams[0])
        else:
            self.team = [m for m in load_verified_pool(battle_format)[:n]]

    def yield_team(self) -> str:
        return Teambuilder.join_team([_to_tb(m) for m in self.team])


def random_team_string(battle_format: str, n: int = N_TEAM,
                       seed: Optional[int] = None) -> str:
    """One random bag team from the format's flat pool, as a packed showdown string."""
    return RandomTeamBuilder(battle_format, seed, n).yield_team()


def strong_team_string(battle_format: str, seed: Optional[int] = None) -> str:
    """One coherent curated team, as a packed showdown string."""
    return StrongTeamBuilder(battle_format, seed).yield_team()
