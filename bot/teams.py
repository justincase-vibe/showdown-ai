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
from typing import Dict, List, Optional, Tuple

from poke_env.teambuilder import Teambuilder, TeambuilderPokemon

HERE = os.path.dirname(os.path.abspath(__file__))
POOLS_PATH = os.path.join(HERE, "team_pools.json")
DATA_DIR = os.path.join(HERE, "data")
TEAMS_PATH = os.path.join(DATA_DIR, "teams.json")
MOVESETS_PATH = os.path.join(DATA_DIR, "movesets.json")
N_TEAM = 6
STAT_IDX = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}


def load_pools() -> Dict:
    if not os.path.exists(POOLS_PATH):
        raise RuntimeError(
            f"no team pools found at {POOLS_PATH} -- "
            f"run `python bot/smogon_teams.py` first"
        )
    with open(POOLS_PATH, encoding="utf8") as fh:
        return json.load(fh)


def available_formats() -> List[str]:
    """Every format that has a usable team pool, sorted (gen order).

    SINGLE SOURCE OF TRUTH for the GUI's format dropdowns (D102): a format is
    selectable iff the bot can actually build teams for it — the pool is the
    gate (``StrongTeamBuilder`` raises on a missing pool). Returns [] when the
    pool file does not exist yet (GUI degrades to free text)."""
    try:
        pools = load_pools()
    except RuntimeError:
        return []
    return sorted(k for k in pools if isinstance(pools.get(k), dict) and pools[k].get("teams"))


def _entry(battle_format: str) -> Dict:
    pools = load_pools()
    if battle_format not in pools:
        raise RuntimeError(
            f"no verified team pool for '{battle_format}' -- "
            f"run `python bot/smogon_teams.py --formats {battle_format}`"
        )
    return pools[battle_format]


def _base_species(species: str) -> str:
    """Normalize a species name to its base form for Species Clause dedup.

    Safest approach: strip everything after the first dash. This catches ALL form
    variants (Mega, Therian, Bond, Zygarde-10, Hoopa-Unbound, Kyurem-Black, etc.).
    It may over-dedupe regional forms (Hisuian Zigzagoon vs Zigzagoon), but that's
    a conservative direction -- we never create an illegal team, we just exclude a
    rare legal combination. Far safer than missing a form and getting rejected.
    """
    s = species.strip().lower()
    if "-" in s:
        s = s.split("-", 1)[0]
    return s


def _dedupe_by_species(mons: List[Dict]) -> List[Dict]:
    """Drop duplicate *base species* from a flat mon pool (first occurrence wins).

    The stored ``flat`` pools are the concatenation of every mon from every coherent
    team, so a popular species (Garchomp, Landorus, …) appears several times. A random
    *bag* that draws two of the same base species is **illegal** (Species Clause) and
    the server rejects it at battle start. We dedupe by *base* species (stripping
    form suffixes like -Mega, -Therian, -Bond) so "Lopunny" and "Lopunny-Mega"
    count as the same slot.
    """
    seen: set = set()
    out: List[Dict] = []
    for m in mons:
        raw = str(m.get("species", "")).strip()
        key = _base_species(raw)
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
    """Coherent curated teams for the format (empty list if none available).

    Reads ``data/teams.json`` first (clean per-format list of 6-mon teams), then falls
    back to the legacy ``team_pools.json`` so older pools keep working."""
    if os.path.exists(TEAMS_PATH):
        try:
            with open(TEAMS_PATH, encoding="utf8") as fh:
                teams = json.load(fh).get(battle_format, [])
            if teams:
                return teams
        except Exception:
            pass
    entry = _entry(battle_format)
    if isinstance(entry, dict):
        return entry.get("teams", [])
    # legacy bare list has no coherent teams
    return []


def load_movesets(battle_format: str) -> Optional[Dict[str, Dict]]:
    """Competitive sets for the format: ``{species: {set_name: set_obj}}``.

    ``None`` if ``data/movesets.json`` is missing or has no entry for the format.
    Each ``set_obj`` is the raw Smogon set: ``moves`` may contain ``[alternatives]``,
    ``item``/``ability``/``nature`` may be a string OR a list, ``evs``/``ivs`` are
    partial stat dicts, ``teratypes`` only appears for gen9."""
    if not os.path.exists(MOVESETS_PATH):
        return None
    try:
        with open(MOVESETS_PATH, encoding="utf8") as fh:
            entry = json.load(fh).get(battle_format)
    except Exception:
        return None
    return entry if isinstance(entry, dict) and entry else None


def _pick_one(rng: "random.Random", v):
    """Resolve a possibly-list Smogon field (alternatives) to a single value."""
    if isinstance(v, (list, tuple)):
        v = [x for x in v if x not in (None, "")]
        if not v:
            return None
        return rng.choice(v)
    return v


def _evs_list(rng: "random.Random", d) -> List[int]:
    """Parse a Smogon ``evs`` field into a 6-int list.

    ``evs`` may be a single partial dict (``{"hp": 252, ...}``) OR a list of
    alternative spreads (``[{...}, {...}]``) -- pick one alternative first. A missing
    / empty spread legitimately resolves to all-zero, which the build-time filter
    will catch (0-EV mons are rejected by the server), so we keep it as-is here.
    """
    if isinstance(d, (list, tuple)):
        d = _pick_one(rng, d)
    evs = [0, 0, 0, 0, 0, 0]
    if isinstance(d, dict):
        for k, val in d.items():
            i = STAT_IDX.get(str(k).lower())
            if i is not None:
                evs[i] = int(val)
    return evs


def _ivs_list(rng: "random.Random", d):
    """Parse a Smogon ``ivs`` field into a 6-int list, or ``None`` if unspecified.

    ``ivs`` is a partial dict (e.g. ``{"atk": 0}``) or a list of such dicts
    (alternatives) -- resolve a list to one alternative first. ``None`` means "keep
    the 31/31 defaults" which is what ``_to_tb`` uses.
    """
    if isinstance(d, (list, tuple)):
        d = _pick_one(rng, d)
    if not isinstance(d, dict) or not d:
        return None
    ivs = [31, 31, 31, 31, 31, 31]
    for k, val in d.items():
        i = STAT_IDX.get(str(k).lower())
        if i is not None:
            ivs[i] = int(val)
    return ivs


def _resolve_set(rng: "random.Random", species: str, set_obj: Dict) -> Dict:
    """Resolve a raw Smogon set (alternatives + partial stats) into a concrete mon-dict."""
    moves: List[str] = []
    for slot in (set_obj.get("moves") or [])[:4]:
        cands = slot if isinstance(slot, (list, tuple)) else [slot]
        # Exclude moves already chosen (a move can be an alternative in MORE than one
        # slot, e.g. Quagsire [.., [Toxic, Earthquake], [Counter, Earthquake]]). A
        # duplicate move is ILLEGAL ("multiple copies of X"); dropping a slot to keep
        # <4 distinct moves is legal, so prefer that over a duplicate.
        cands = [c for c in cands if c and c not in moves]
        if cands:
            moves.append(rng.choice(cands))
    return dict(
        species=species,
        item=_pick_one(rng, set_obj.get("item")),
        ability=_pick_one(rng, set_obj.get("ability")),
        moves=moves,
        nature=(_pick_one(rng, set_obj.get("nature")) or "Neutral"),
        evs=_evs_list(rng, set_obj.get("evs")),
        ivs=_ivs_list(rng, set_obj.get("ivs")),
        tera_type=(_pick_one(rng, set_obj.get("teratypes"))
                   if "teratypes" in set_obj else None),
    )


def _to_tb(mon: Dict) -> TeambuilderPokemon:
    """Build a TeambuilderPokemon from our mon-dict.

    Tolerant of both sources: coherent teams always carry item/ability/moves/nature/evs;
    competitive sets may omit ability and carry 0-IVs + a Tera type."""
    tb = TeambuilderPokemon(
        species=mon.get("species"),
        item=mon.get("item") or None,
        ability=mon.get("ability") or None,
        moves=list(mon.get("moves") or []),
        nature=mon.get("nature") or None,
        evs=list(mon.get("evs") or [0, 0, 0, 0, 0, 0]),
        ivs=list(mon["ivs"]) if mon.get("ivs") else [31, 31, 31, 31, 31, 31],
    )
    if mon.get("tera_type"):
        tb.tera_type = mon["tera_type"]
    return tb


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
    """Yields a fresh **random team built from competitive sets** per ``yield_team()``.

    For the format, each species has one or more named sets (``data/movesets.json``).
    A team is 6 **distinct base-species** (Species-Clause safe -- forms of the same
    base, e.g. Rotom-Wash vs Rotom-Heat, count as one), each given a randomly chosen
    set whose alternatives (moves / item / ability / nature / Tera type) are resolved
    to concrete values. This is the *high-variety* baseline that stops the policy
    overfitting to the handful of coherent lineups; it is deliberately a bag of
    individually-legal sets, not a hand-built team.

    Falls back to the legacy flat pool (``load_verified_pool``) if the format has no
    competitive-set data.
    """

    def __init__(self, battle_format: str, seed: Optional[int] = None,
                 n: int = N_TEAM):
        self.battle_format = battle_format
        self.n = n
        self._rng = random.Random(seed)
        # Group competitive sets by BASE species (Species Clause safety).
        options: Dict[str, List[Tuple[str, Dict]]] = {}
        sets_by_species = load_movesets(battle_format)
        if sets_by_species:
            for species, named in sets_by_species.items():
                if not isinstance(named, dict):
                    continue
                base = _base_species(species)
                for _name, set_obj in named.items():
                    if isinstance(set_obj, dict):
                        options.setdefault(base, []).append((species, set_obj))
        if len(options) >= n:
            self._options = options
            self._legacy_pool = None
        else:
            pool = load_verified_pool(battle_format)
            if len(pool) < n:
                raise ValueError(
                    f"no team source for {battle_format}: movesets has "
                    f"{len(options)} species, flat pool {len(pool)} mons, need {n} -- "
                    f"run `python bot/build_movesets.py`"
                )
            self._options = None
            self._legacy_pool = pool

    def yield_team(self) -> str:
        if self._legacy_pool is not None:
            picks = self._rng.sample(self._legacy_pool, self.n)
            return Teambuilder.join_team([_to_tb(m) for m in picks])
        bases = list(self._options.keys())
        chosen = self._rng.sample(bases, self.n)
        mons = []
        for base in chosen:
            species, set_obj = self._rng.choice(self._options[base])
            mons.append(_resolve_set(self._rng, species, set_obj))
        return Teambuilder.join_team([_to_tb(m) for m in mons])


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


class MixedTeamBuilder(Teambuilder):
    """Yields a **mix** of coherent teams and random competitive-set teams.

    Per ``yield_team()`` call, with probability ``p_strong`` it draws a coherent
    ``StrongTeamBuilder`` team (real lineups that teach synergy), otherwise a
    ``RandomTeamBuilder`` team (a varied bag of individually-legal sets). This is the
    default training mode: the policy sees BOTH a realistic strong opponent and a wide
    variety of legal lineups in the same run -- which is what keeps it from memorizing
    the small set of coherent teams. Degrades to whichever source exists if one is
    missing for the format.
    """

    def __init__(self, battle_format: str, seed: Optional[int] = None,
                 p_strong: float = 0.5, n: int = N_TEAM):
        self.battle_format = battle_format
        self._rng = random.Random(seed)
        self.p_strong = max(0.0, min(1.0, float(p_strong)))
        self._strong = None
        self._random = None
        try:
            self._strong = StrongTeamBuilder(battle_format, seed=seed)
        except Exception:
            self._strong = None
        try:
            self._random = RandomTeamBuilder(battle_format, seed=seed, n=n)
        except Exception:
            self._random = None
        if self._strong is None and self._random is None:
            raise RuntimeError(
                f"no team source for '{battle_format}' -- "
                f"run `python bot/build_movesets.py`"
            )

    def yield_team(self) -> str:
        want_strong = self._rng.random() < self.p_strong
        if want_strong and self._strong is not None:
            return self._strong.yield_team()
        if not want_strong and self._random is not None:
            return self._random.yield_team()
        if self._strong is not None:
            return self._strong.yield_team()
        return self._random.yield_team()


def random_team_string(battle_format: str, n: int = N_TEAM,
                       seed: Optional[int] = None) -> str:
    """One random bag team from the format's flat pool, as a packed showdown string."""
    return RandomTeamBuilder(battle_format, seed, n).yield_team()


def strong_team_string(battle_format: str, seed: Optional[int] = None) -> str:
    """One coherent curated team, as a packed showdown string."""
    return StrongTeamBuilder(battle_format, seed).yield_team()


def mixed_team_string(battle_format: str, p_strong: float = 0.5,
                      seed: Optional[int] = None) -> str:
    """One team (coherent with prob ``p_strong``, else a random competitive-set team)."""
    return MixedTeamBuilder(battle_format, seed, p_strong=p_strong).yield_team()
