"""Team registry + generator — the league's LIVING team pool (the builder's body).

Mirrors ``league/registry.py`` (BotRegistry) exactly:
  * One JSON file: ``league/teams.json``
  * ELO K=32, initial 1500 — the same convention as bots
  * Thread-safe (RLock): concurrent league battles call ``update_result -> save``
  * Framework-agnostic (no torch)

What it adds for teams:
  * ``origin``: ``"curated"`` (human-built, PROTECTED — never culled) or
    ``"generated"`` (builder output — cullable/breedable). The two populations
    are kept distinct in the registry AND in every report (user ruling
    2026-11-28: differentiate human teams from generated ones).
  * ``parent``: the ``tid`` a generated team was mutated from — the attribution
    link that makes (parent, child) pairs into controlled synergy experiments.
  * NO move tags / hand-designed features (project ruling): a mon is identified
    by species + moves + item + ability + nature + evs, exactly like the curated
    pools in ``bot/team_pools.json``. Move MEANING lives in the learned
    embeddings downstream (``net/team_brain.py``).

Generator legality is BY CONSTRUCTION: every mon's (species, moves, item,
ability, nature, evs) comes from ``bot/data/movesets.json`` — the same
Smogon-verified competitive-set data the curated pools were built from — so
the server's TeamValidator accepts generated teams the same way it accepted
the curated ones. No re-validation loop in the hot path.
"""
from __future__ import annotations

import json
import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: default registry location (relative to the project root)
DEFAULT_TEAMS = Path("league") / "teams.json"

#: ELO constants (K=32, initial 1500 — same as bots)
ELO_K = 32
ELO_INITIAL = 1500

#: origin values
ORIGIN_CURATED = "curated"
ORIGIN_GENERATED = "generated"

#: mutation op names (for attribution + tests)
OP_SWAP_MON = "swap_mon"
OP_SWAP_SET = "swap_set"
OP_SWAP_ITEM = "swap_item"

#: EV stat order (must match bot/team_pools.json / net/env.py _STAT_IDX)
_STAT_IDX = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}

HERE = Path(__file__).resolve().parent
_SETS_PATH = HERE.parent / "bot" / "data" / "movesets.json"
_POOLS_PATH = HERE.parent / "bot" / "team_pools.json"


def base_species(species: str) -> str:
    """Conservative forme-stripped identity (``Rotom-Wash`` -> ``Rotom``).

    Used ONLY to keep generated teams free of forme-duplicate ambiguity —
    stricter than the server requires, never looser (bot/teams.py does the
    same for its random bags)."""
    return species.split("-", 1)[0].strip()


@dataclass
class Team:
    """One entry in the team pool."""
    tid: str
    origin: str = ORIGIN_GENERATED
    parent: Optional[str] = None          # tid mutated from (None for curated / structured)
    mons: List[Dict] = field(default_factory=list)   # 6 mon-dicts (pools format)
    elo: int = ELO_INITIAL
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_fought: Optional[str] = None     # ISO timestamp
    created_round: Optional[int] = None   # league round that minted it (bookkeeping)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Team":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TeamRegistry:
    """JSON-backed pool of league teams with ELO tracking.

    Single source of truth for "which teams are in the league and how well
    they're doing." The engine schedules team-vs-team matchups (league/
    schedule.py), runs them piloted by bot pairs, and writes results back
    via :meth:`update_result`.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else DEFAULT_TEAMS
        self._teams: Dict[str, Team] = {}
        self._lock = threading.RLock()
        self._load()

    # ── persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for t in data.get("teams", []):
                team = Team.from_dict(t)
                self._teams[team.tid] = team
        except Exception:
            self._teams = {}

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {"teams": [t.to_dict() for t in self._teams.values()]}
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── pool management ──────────────────────────────────────────────────

    def add(self, tid: str, mons: List[Dict], origin: str = ORIGIN_GENERATED,
            parent: Optional[str] = None, elo: int = ELO_INITIAL,
            created_round: Optional[int] = None) -> Team:
        """Add a team (or update in place if the tid already exists)."""
        with self._lock:
            team = Team(tid=tid, origin=origin, parent=parent, mons=list(mons),
                        elo=elo, created_round=created_round)
            self._teams[tid] = team
            self.save()
        return team

    def remove(self, tid: str) -> None:
        """Remove a team. CURATED teams are protected (user ruling): the
        human-built baseline is never culled by the builder."""
        with self._lock:
            if tid not in self._teams:
                raise KeyError(f"team {tid!r} not in pool")
            if self._teams[tid].origin == ORIGIN_CURATED:
                raise ValueError(
                    f"team {tid!r} is curated (human-built) and is PROTECTED — "
                    "the builder may not cull it")
            del self._teams[tid]
            self.save()

    def get(self, tid: str) -> Team:
        if tid not in self._teams:
            raise KeyError(f"team {tid!r} not in pool")
        return self._teams[tid]

    def all(self) -> List[Team]:
        return list(self._teams.values())

    def tids(self) -> List[str]:
        return list(self._teams.keys())

    def generated(self) -> List[Team]:
        return [t for t in self._teams.values() if t.origin == ORIGIN_GENERATED]

    def origin_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self._teams.values():
            out[t.origin] = out.get(t.origin, 0) + 1
        return out

    # ── results (ELO) ────────────────────────────────────────────────────

    def update_result(self, a: str, b: str, winner: Optional[str]) -> None:
        """Record a team-vs-team result and update ELO (thread-safe).

        ``winner`` is a team tid (``a`` or ``b``) or ``None`` (draw). Same
        standard ELO math as BotRegistry (K=32, delta = K*(score - E_a))."""
        with self._lock:
            ta = self.get(a)
            tb = self.get(b)
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if winner == a:
                ta.wins += 1
                tb.losses += 1
                delta = self._elo_delta(ta.elo, tb.elo, 1.0)
            elif winner == b:
                tb.wins += 1
                ta.losses += 1
                delta = self._elo_delta(ta.elo, tb.elo, 0.0)
            else:
                ta.draws += 1
                tb.draws += 1
                delta = self._elo_delta(ta.elo, tb.elo, 0.5)
            ta.elo += delta
            tb.elo -= delta
            ta.last_fought = now
            tb.last_fought = now
            self.save()

    @staticmethod
    def _elo_delta(rating_a: int, rating_b: int, score: float) -> int:
        expected = 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
        return round(ELO_K * (score - expected))

    # ── standings / selection ────────────────────────────────────────────

    def standings(self, origin: Optional[str] = None) -> List[Dict[str, Any]]:
        """Pool sorted by ELO desc, then tid. ``origin`` filters the population."""
        teams = [t for t in self._teams.values() if origin is None or t.origin == origin]
        teams = sorted(teams, key=lambda t: (-t.elo, t.tid))
        return [
            {
                "tid": t.tid,
                "origin": t.origin,
                "parent": t.parent,
                "species": [m.get("species") for m in t.mons],
                "elo": t.elo,
                "wins": t.wins,
                "losses": t.losses,
                "draws": t.draws,
                "win_rate": round(t.wins / max(1, t.wins + t.losses), 3),
                "last_fought": t.last_fought,
            }
            for t in teams
        ]

    def bottom_generated(self, n: int) -> List[Team]:
        """The n weakest GENERATED teams (cull candidates). Curated are
        excluded by construction."""
        gen = sorted(self.generated(), key=lambda t: (t.elo, t.tid))
        return gen[:n]

    def top_generated(self, n: int) -> List[Team]:
        gen = sorted(self.generated(), key=lambda t: (-t.elo, t.tid))
        return gen[:n]

    def __len__(self) -> int:
        return len(self._teams)

    def __contains__(self, tid: str) -> bool:
        return tid in self._teams


# ═══════════════════════════════════════════════════════════════════════════
#  generator  (mutations of good teams + structured builds)
# ═══════════════════════════════════════════════════════════════════════════

def load_sets(battle_format: str = "gen9ou",
              path: Optional[Path] = None) -> Dict[str, List[Dict]]:
    """{species: [set, set, ...]} from the verified competitive-set data.

    Each set is ``{name, moves, item, ability, nature, evs}`` (evs a dict).
    This is the SAME data source the curated pools were built from."""
    p = Path(path) if path is not None else _SETS_PATH
    if not p.is_file():
        raise RuntimeError(f"no movesets data at {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    entry = raw.get(battle_format)
    if not isinstance(entry, dict) or not entry:
        raise RuntimeError(f"no competitive sets for '{battle_format}' in {p}")
    sets: Dict[str, List[Dict]] = {}
    for species, named in entry.items():
        if not isinstance(named, dict):
            continue
        lst = []
        for name, s in named.items():
            if not isinstance(s, dict) or not s.get("moves"):
                continue
            lst.append({
                "name": name,
                "moves": list(s.get("moves") or [])[:4],
                "item": s.get("item"),
                "ability": s.get("ability"),
                "nature": s.get("nature"),
                "evs": dict(s.get("evs") or {}),
            })
        if lst:
            sets[species] = lst
    return sets


def _evs_to_list(evs: Optional[Dict]) -> List[int]:
    out = [0, 0, 0, 0, 0, 0]
    for k, v in (evs or {}).items():
        if k in _STAT_IDX and isinstance(v, (int, float)):
            out[_STAT_IDX[k]] = int(v)
    return out


def resolve_set(species: str, s: Dict) -> Dict:
    """A set dict -> a mon-dict in team_pools.json format (the exchange format
    the drivers / team brain / replay logs all understand)."""
    mon = {
        "species": species,
        "moves": list(s.get("moves") or [])[:4],
        "item": s.get("item"),
        "ability": s.get("ability"),
        "nature": s.get("nature"),
        "evs": _evs_to_list(s.get("evs")),
        "ivs": [31] * 6,
    }
    return mon


def _team_bases(mons: List[Dict]) -> set:
    return {base_species(m.get("species", "")) for m in mons}


def mutate(mons: List[Dict], sets: Dict[str, List[Dict]], rng: random.Random,
           op: Optional[str] = None) -> Tuple[List[Dict], str]:
    """ONE single-slot mutation of a team -> (new_mons, op_used).

    Ops:
      * ``swap_mon``  — replace one slot's species+set with another species' set
        (excludes species already on the team by base identity)
      * ``swap_set``  — same species, a DIFFERENT competitive set (only if the
        species has ≥2 sets in the data)
      * ``swap_item`` — same species+set, item drawn from the species' other sets
        (falls back to the format's item pool)

    ``op=None`` picks a weighted random op (swap_mon 0.5 / swap_set 0.3 /
    swap_item 0.2) and degrades gracefully when an op is inapplicable.

    Contract: the result changes EXACTLY ONE slot (the other five mons are
    the same objects) and never a NO-OP — ``swap_set`` only picks a set that
    resolves to different fields, ``swap_item`` only picks a different item
    (re-picking the current value would silently clone the parent team; that
    was the 58%/80% no-op bug fixed 2026-11-28, pinned by
    league/tests/test_teams.py::test_mutate_exactly_one_slot_changed).
    Result is legal by construction (sets data is verified; bases deduped).
    """
    if len(mons) != 6:
        raise ValueError(f"mutate: expected 6 mons, got {len(mons)}")
    bases = _team_bases(mons)
    candidates = [OP_SWAP_MON, OP_SWAP_SET, OP_SWAP_ITEM]
    weights = [0.5, 0.3, 0.2]
    order = [op] if op is not None else list(rng.choices(candidates, weights=weights, k=3))
    # deterministic shuffle when op is None (rng-driven, reproducible)
    if op is None:
        order = candidates[:]
        rng.shuffle(order)
    last_err: Optional[Exception] = None
    for want in order:
        try:
            if want == OP_SWAP_MON:
                return _mut_swap_mon(mons, sets, bases, rng)
            if want == OP_SWAP_SET:
                return _mut_swap_set(mons, sets, rng)
            if want == OP_SWAP_ITEM:
                return _mut_swap_item(mons, sets, rng)
        except _Inapplicable as e:
            last_err = e
            continue
    raise RuntimeError(f"mutate: no applicable op (last: {last_err})")


class _Inapplicable(Exception):
    """The requested op has no valid target in this team (try another op)."""


def _mon_fields(m: Dict) -> Tuple:
    """Comparable identity of a mon's set-derived fields (moves/item/ability/
    nature/evs). Used to detect NO-OP mutations: re-picking the exact set (or
    item) a slot already carries changes nothing and would silently clone the
    parent team. Extra keys (``ivs``, ``tera_type``) are ignored — a no-op is
    judged on the fields the generator actually controls."""
    evs = m.get("evs")
    evs = tuple(evs) if isinstance(evs, (list, tuple)) else tuple(evs or ())
    return (
        tuple(m.get("moves") or ()),
        m.get("item"),
        m.get("ability"),
        m.get("nature"),
        evs,
    )


def _mut_swap_mon(mons, sets, bases, rng):
    species_pool = [sp for sp in sets if base_species(sp) not in bases]
    if not species_pool:
        raise _Inapplicable("no new species available")
    i = rng.randrange(6)
    sp = rng.choice(species_pool)
    new = mons[:6]
    new[i] = resolve_set(sp, rng.choice(sets[sp]))
    return new, OP_SWAP_MON


def _mut_swap_set(mons, sets, rng):
    # A slot is eligible only if the species has at least one set that resolves
    # to DIFFERENT fields than the mon already carries — re-picking the same
    # set is a no-op (observed at 80% on generated teams before this fix) and
    # would clone the parent instead of breeding it.
    def _different_sets(i: int):
        sp = mons[i].get("species")
        if sp not in sets:
            return []
        cur = _mon_fields(mons[i])
        return [s for s in sets[sp] if _mon_fields(resolve_set(sp, s)) != cur]

    eligible = [i for i in range(6) if _different_sets(i)]
    if not eligible:
        raise _Inapplicable("no slot offers a different set for its species")
    i = rng.choice(eligible)
    sp = mons[i]["species"]
    new = mons[:6]
    new[i] = resolve_set(sp, rng.choice(_different_sets(i)))
    return new, OP_SWAP_SET


def _mut_swap_item(mons, sets, rng):
    # item pool: the species' own sets first, then the whole format's items —
    # EXCLUDING the item the slot already carries (a re-pick is a no-op clone;
    # observed at 58% on generated teams before this fix).
    all_items: List[str] = []
    for lst in sets.values():
        for s in lst:
            if s.get("item"):
                all_items.append(s["item"])
    if not all_items:
        raise _Inapplicable("no items in set data")
    for i in list(rng.sample(range(6), 6)):
        sp = mons[i]["species"]
        own = [s["item"] for s in sets.get(sp, []) if s.get("item")]
        base_pool = own or all_items
        pool = [it for it in base_pool if it != mons[i].get("item")]
        if not pool:
            continue  # this slot can't change item; try another slot
        new = mons[:6]
        m = dict(new[i])
        m["item"] = rng.choice(pool)
        new[i] = m
        return new, OP_SWAP_ITEM
    raise _Inapplicable("no slot offers a different item")


def structured_build(sets: Dict[str, List[Dict]], rng: random.Random) -> List[Dict]:
    """A FRESH coherent-enough team: 6 DISTINCT species (by base identity),
    each with a random competitive set. Broader species coverage than pure
    mutation; still legal by construction. (Not a random BAG of random mons —
    every mon is a verified competitive set.)"""
    species = list(sets.keys())
    if len(species) < 6:
        raise RuntimeError("structured_build: fewer than 6 species with sets")
    chosen: List[str] = []
    seen = set()
    while len(chosen) < 6:
        sp = rng.choice(species)
        b = base_species(sp)
        if b in seen:
            continue
        seen.add(b)
        chosen.append(sp)
    return [resolve_set(sp, rng.choice(sets[sp])) for sp in chosen]


# ═══════════════════════════════════════════════════════════════════════════
#  seeding the curated baseline
# ═══════════════════════════════════════════════════════════════════════════

def seed_curated(reg: TeamRegistry, battle_format: str = "gen9ou",
                 pools_path: Optional[Path] = None, prefix: str = "cur") -> int:
    """Load the human-built pool (bot/team_pools.json) into the registry as
    ``origin=curated`` teams (ids ``<prefix>_00..``). Idempotent: existing
    tids are left untouched. Returns the number of curated teams in the pool."""
    p = Path(pools_path) if pools_path is not None else _POOLS_PATH
    if not p.is_file():
        raise RuntimeError(f"no team pools at {p}")
    pools = json.loads(p.read_text(encoding="utf-8"))
    entry = pools.get(battle_format)
    teams = (entry or {}).get("teams") if isinstance(entry, dict) else entry
    if not teams:
        raise RuntimeError(f"no curated teams for '{battle_format}' in {p}")
    n = 0
    for i, mons in enumerate(teams):
        if not isinstance(mons, list) or len(mons) != 6:
            continue
        tid = f"{prefix}_{i:02d}"
        if tid in reg:
            n += 1
            continue
        reg.add(tid, mons, origin=ORIGIN_CURATED, parent=None)
        n += 1
    return n


def new_generated_tid(reg: TeamRegistry, parent: Optional[str] = None,
                      rnd: Optional[int] = None, prefix: str = "gen") -> str:
    """A collision-free tid for a generated team (parent+round in the name for
    readability, random suffix for uniqueness)."""
    import os
    tag = (parent or "fresh").replace("gen_", "").replace("cur_", "c")
    base = f"{prefix}_{tag}_r{rnd if rnd is not None else 0}"
    cand = f"{base}_{os.urandom(2).hex()}"
    while cand in reg:
        cand = f"{base}_{os.urandom(2).hex()}"
    return cand
