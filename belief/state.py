"""Step 3.1 — belief state + prior (updates land in 3.2-3.4).

Per opponent slot: a posterior distribution over FULL-SET candidates with
weights. Full sets (not per-slot marginals) so that within-set correlations
are preserved by construction — a Wish set already contains Protect (user
requirement, 2026-11-15).

Candidate schema (plain dict, JSON-safe):
    moves:   [up to 4 move ids, slot order]
    item:    item id, or '' = none/unknown (fallback candidates)
    ability: ability id
    nature:  display name (DB convention, e.g. 'Impish'; 'Hardy' = neutral)
    evs:     {'hp': int, ...} (missing stat = 0)
    tera:    tera type id, or None = UNKNOWN (None is NOT "no Tera")
    weight:  float > 0; weights per slot sum to exactly 1 (invariant, tested)

Prior (D43, 2026-11-15):
  1. Sets for the format itself (Step 1 ``get_common_sets``; probs sum to 1).
  2. The BSS format (``gen9bss3v3``) carries NO sets of its own in the DB —
     sets live in the tier tables. So: merge the same-gen tier pools
     (gen9ou + gen9uu + gen9ru) and weight every set equally (1/N). A coarse
     prior — the Bayesian updates fix it as evidence arrives.
  3. Fallback for species with no sets anywhere in the gen: uniform over
     BOUNDED legal move combos (first 8 legal moves, all C(k,4); k<4 → all of
     them) crossed with the species' abilities; nature neutral, EVs 0, no
     item, tera unknown. Coarse by design — HARD evidence (3.3) narrows it
     within a few turns.
"""
from __future__ import annotations

import itertools
import re
from typing import Dict, List, Optional

from legality import db, lookup

#: EV stat tokens as they appear in the DB spread strings ("252 HP / 216 Def")
_EV_TOKEN = {"hp": "hp", "atk": "atk", "def": "def", "spa": "spa", "spd": "spd", "spe": "spe"}

#: max legal moves drawn for fallback combos (bounds candidate-space size)
_FALLBACK_MOVE_POOL = 8
_FALLBACK_SLOTS = 4


def _fmt_gen(fmt: str) -> int:
    m = re.match(r"gen(\d+)", fmt)
    if not m:
        raise ValueError(f"cannot read gen from format id {fmt!r}")
    return int(m.group(1))


def _parse_evs(spread: Optional[str]) -> Dict[str, int]:
    """'252 HP / 216 Def / 40 Spe' -> {'hp': 252, 'def': 216, 'spe': 40}."""
    if not spread:
        return {}
    out: Dict[str, int] = {}
    for part in spread.split("/"):
        part = part.strip()
        if not part:
            continue
        bits = part.split(" ")
        if len(bits) != 2:
            raise ValueError(f"bad EV spread token {part!r} in {spread!r}")
        val, stat = bits
        key = _EV_TOKEN.get(stat.lower())
        if key is None:
            raise ValueError(f"unknown EV stat {stat!r} in {spread!r}")
        out[key] = int(val)
    return out


def _candidate_from_set(row: Dict, weight: float) -> Dict:
    return {
        "moves": list(row["moves"]),
        "item": row["item_id"] or "",
        "ability": row["ability_id"] or "",
        "nature": row["nature"] or "Hardy",
        "evs": _parse_evs(row.get("evs")),
        "tera": row.get("tera_type") or None,
        "weight": float(weight),
    }


def _tier_pool(species: str, fmt: str) -> Optional[List[Dict]]:
    """Same-gen tier sets (equal weight) when the format itself has none."""
    gen = _fmt_gen(fmt)
    conn = db.connect()
    try:
        tiers = [
            r["format_id"]
            for r in conn.execute("SELECT DISTINCT format_id FROM strategy ORDER BY format_id")
        ]
    finally:
        conn.close()
    pool: List[Dict] = []
    for f in tiers:
        try:
            if _fmt_gen(f) != gen:
                continue
        except ValueError:
            continue
        pool.extend(lookup.get_common_sets(species, f))
    return pool or None


def _fallback_candidates(species: str, fmt: str) -> List[Dict]:
    legal = lookup.legal_moves(species, fmt)
    if not legal:
        return []
    sp = lookup.get_species(species, _fmt_gen(fmt))
    abilities = [a["id"] for a in sp.get("abilities") or []] or [""]
    pool = legal[:_FALLBACK_MOVE_POOL]
    combos = [list(c) for c in itertools.combinations(pool, min(_FALLBACK_SLOTS, len(pool)))]
    n = len(combos) * len(abilities)
    w = 1.0 / n
    out: List[Dict] = []
    for moves in combos:
        for ability in abilities:
            out.append(
                {
                    "moves": moves,
                    "item": "",
                    "ability": ability,
                    "nature": "Hardy",
                    "evs": {},
                    "tera": None,
                    "weight": w,
                }
            )
    return out


def build_prior(species: str, fmt: str) -> List[Dict]:
    """Candidate list for one opponent slot (weights sum to 1, all > 0).

    Raises ``ValueError`` when the species has no usable prior at all in the
    format (no sets AND no legal moves) — a loud failure, never an empty
    belief (PLAN.md Step 3 architecture: the posterior can never be empty).
    """
    rows = lookup.get_common_sets(species, fmt)
    if rows:
        return [_candidate_from_set(r, r["probability"]) for r in rows]

    tier = _tier_pool(species, fmt)
    if tier:
        w = 1.0 / len(tier)
        return [_candidate_from_set(r, w) for r in tier]

    fb = _fallback_candidates(species, fmt)
    if fb:
        return fb

    raise ValueError(f"no prior for {species!r} in {fmt!r}: no sets and no legal moves")


class SlotBelief:
    """Posterior over candidates for ONE opponent slot."""

    def __init__(self, slot: str, species: str, candidates: List[Dict]):
        if not slot.startswith("p2"):
            raise ValueError(f"belief slots are opponent-side (p2*): got {slot!r}")
        if not candidates:
            raise ValueError(f"empty candidate list for {species!r} in slot {slot!r}")
        if any(c["weight"] <= 0 for c in candidates):
            raise ValueError(f"non-positive weight in {slot!r} candidates")
        self.slot = slot
        self.species = species
        self.candidates = list(candidates)
        self.normalize()

    @property
    def total(self) -> float:
        return sum(c["weight"] for c in self.candidates)

    def normalize(self) -> None:
        t = self.total
        if t <= 0:
            raise ValueError(f"{self.slot!r}: cannot normalize — total weight {t}")
        for c in self.candidates:
            c["weight"] /= t

    @property
    def posterior(self) -> List[float]:
        return [c["weight"] for c in self.candidates]

    def mass(self, predicate) -> float:
        """Total posterior mass over candidates satisfying ``predicate``."""
        return sum(c["weight"] for c in self.candidates if predicate(c))


class BeliefState:
    """The bot's belief about the opponent side (p2 slots) in one battle."""

    def __init__(self, fmt: str):
        self.fmt = fmt
        self._slots: Dict[str, SlotBelief] = {}

    def add_opponent(self, slot: str, species: str) -> SlotBelief:
        """Register an opponent mon (on switch-in) and give it its prior."""
        if slot in self._slots:
            return self._slots[slot]
        sb = SlotBelief(slot, species, build_prior(species, self.fmt))
        self._slots[slot] = sb
        return sb

    def slot(self, slot: str) -> SlotBelief:
        try:
            return self._slots[slot]
        except KeyError:
            raise KeyError(f"no belief for slot {slot!r} (add_opponent first)") from None

    @property
    def slots(self) -> List[str]:
        return list(self._slots)
