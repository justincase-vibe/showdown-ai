"""Global move-identity action space (single source of truth).

Why this exists
---------------
The old action space was *slot-based*: action ``6+k`` meant "use move slot ``k``" of the
active mon, where the slot index meant *reveal-order* during imitation training but
*moveset-order* during live play. Those two orders differ, so the *same* integer meant a
*different move* in training vs. live -- a silent train/live mismatch.

This module deletes that class of bug by construction: an action is **a move by identity**
("use Outrage"), not a slot. The move's position in the team's moveset no longer matters;
only which move it is.

Layout (one fixed, global space spanning gens 6-9, so a single policy plays all tiers)
----------------------------------------------------------------------------------------
    MOVE       0 .. N_MOVES-1         every move in the union of the gen6-9 move files
                                      (base moves AND Z-moves AND G-Max moves -- they are
                                      distinct moves with distinct effects, so distinct IDs)
    SWITCH     N_MOVES .. N_MOVES+5   6 team slots (switch to slot 0..5)
    TRANSFORM  N_MOVES+6 .. N_MOVES+8 mega / dynamax / tera ("special actions" beside the
                                      battlefield; a mon-level transformation, not a move)

    N_ACTIONS  = N_MOVES + 6 + 3      (= 963 for the current 954-move universe)

This is the standard "fixed action vector + per-situation mask" pattern (e.g. AlphaGo's
legal-move masking): the policy always emits a fixed-width logit vector; a boolean mask
selects the legal subset each turn. The legal set is small (~50-80 moves + a few switches
+ <=1 transform) even though the space is 963.

Phase-aware masking (the agreed design)
---------------------------------------
* **Self-play**: the exact moveset is known -> the MOVE band is masked to *the mon's real
  4 moves*. Tight.
* **Imitation** (replays): the full moveset is NOT recoverable from logs -> the MOVE band
  is masked to the *species-legal prior* (every move the species can hold in that gen).
  Looser, but a superset of the tight mask, so a net trained under the loose mask
  warm-starts cleanly into the tight one.

poke-env step-time note (lives in env.py, not here)
---------------------------------------------------
poke-env couples a gimmick to a base move in a single order. This module stores the *data*
needed to translate a move-identity action back to a concrete order (``z_base`` /
``move_gimmick``); the battle-state logic (which base move, which slot) is in ``env.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence

import numpy as np

from reveal import ZMOVE_TO_BASE

# ---------------------------------------------------------------------------
# Paths (poke-env ships the move tables + the learnset; we read them read-only)
# ---------------------------------------------------------------------------
_PE_STATIC = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Local", "Programs", "Python", "Python312", "Lib",
    "site-packages", "poke_env", "data", "static",
)
GENS: Sequence[int] = (6, 7, 8, 9)
N_SWITCH = 6
TRANSFORMS: Sequence[str] = ("mega", "dynamax", "tera")
N_TRANSFORM = len(TRANSFORMS)

# --- moveset-shape summary (gen-agnostic "what could this species do" prior) ---
_MOVE_TYPES: Sequence[str] = (
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "steel", "fairy",
)
_N_SHAPE_TYPES = len(_MOVE_TYPES)
_N_SHAPE_CAPS = 5  # has-priority, has-recovery, has-status, has-hazard, has-setup
SHAPE_DIM = _N_SHAPE_TYPES + 2 + _N_SHAPE_CAPS + 1  # 18 + phys/spec + caps + count
_HAZARD_MOVES: FrozenSet[str] = frozenset(
    {"stealthrock", "spikes", "toxicspikes"}
)
_TYPE_IDX = {t: i for i, t in enumerate(_MOVE_TYPES)}


def canon_move(name: Optional[str]) -> Optional[str]:
    """Canonical move id: lowercase, alnum only. ``"Bolt Strike"`` -> ``"boltstrike"``."""
    if name is None:
        return None
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return s or None


def canon_species(name: Optional[str]) -> Optional[str]:
    """Canonical species key matching learnset.json keys.
    ``"Garchomp"`` -> ``"garchomp"``; ``"Rotom-Mow"`` -> ``"rotommow"``;
    ``"Slowbro-Galar"`` -> ``"slowbrogalar"``."""
    if name is None:
        return None
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return s or None


class ActionSpace:
    """Fixed global move-identity action space + masking + legality lookups."""

    def __init__(self, static_dir: str = _PE_STATIC):
        # ---- move universe: union of the gen6-9 move files (keeps first props seen) ----
        self._move_names: List[str] = []
        self._move_props: Dict[str, dict] = {}
        for g in GENS:
            path = os.path.join(static_dir, "moves", f"gen{g}moves.json")
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for name, props in data.items():
                c = canon_move(name)
                if c is None:
                    continue
                if c not in self._move_props:
                    self._move_names.append(c)
                    self._move_props[c] = props if isinstance(props, dict) else {}
        self.N_MOVES = len(self._move_names)
        self.move_to_id: Dict[str, int] = {n: i for i, n in enumerate(self._move_names)}
        self.id_to_move: List[str] = list(self._move_names)

        # ---- band offsets ----
        self.SWITCH_BASE = self.N_MOVES
        self.TRANSFORM_BASE = self.SWITCH_BASE + N_SWITCH
        self.transform_id: Dict[str, int] = {
            t: self.TRANSFORM_BASE + i for i, t in enumerate(TRANSFORMS)
        }
        self.N_ACTIONS = self.TRANSFORM_BASE + N_TRANSFORM

        # ---- learnset: species -> {move: [genMethod tags]} ----
        with open(os.path.join(static_dir, "learnset.json"), encoding="utf-8") as fh:
            self._learnset_raw = json.load(fh)
        self._species_key = {canon_species(k): k for k in self._learnset_raw.keys()}
        # Longest-first canonical keys for the forme->base prefix fallback
        # (see _species_key_candidates): 'Blastoise-Mega' -> 'Blastoise', etc.
        self._canon_keys_by_len_desc = sorted(
            self._species_key.keys(), key=len, reverse=True
        )

        # ---- Z-move -> base move. reveal.py's curated map is the single source of
        #      truth (the move files do NOT carry a usable Z->base link). Canonicalize
        #      both sides to our id form. Covers standard + hidden Z-moves. ----
        self._z_to_base: Dict[str, str] = {}
        for zname, base in ZMOVE_TO_BASE.items():
            z, b = canon_move(zname), canon_move(base)
            if z is not None and b in self.move_to_id:
                self._z_to_base[z] = b
        # Canonical set of "this is a Z-move" names (curated map + any isZ in the data).
        self._z_moves: set = set(self._z_to_base.keys())
        for m, p in self._move_props.items():
            if isinstance(p, dict) and p.get("isZ"):
                self._z_moves.add(m)

    # ------------------------------------------------------------------ ids
    @property
    def n_actions(self) -> int:
        return self.N_ACTIONS

    def move_id(self, name: Optional[str]) -> Optional[int]:
        c = canon_move(name)
        return self.move_to_id.get(c) if c else None

    def switch_id(self, slot: int) -> Optional[int]:
        return self.SWITCH_BASE + slot if 0 <= slot < N_SWITCH else None

    def transform(self, kind: str) -> Optional[int]:
        return self.transform_id.get(kind)

    def band(self, action: int) -> str:
        if action < self.SWITCH_BASE:
            return "move"
        if action < self.TRANSFORM_BASE:
            return "switch"
        return "transform"

    # ------------------------------------------------------------- legality
    def species_key(self, name: Optional[str]) -> Optional[str]:
        keys = self._species_key_candidates(name)
        return keys[0] if keys else None

    def _species_key_candidates(self, name: Optional[str]) -> List[str]:
        """learnset.json keys that apply to ``name``: its exact key (if the forme/variant
        is its own entry -- e.g. ``Rotom-Heat``, ``Arceus-Fire``, ``Landorus-Therian``)
        AND/OR its base key (the longest proper-prefix key), e.g. ``Blastoise-Mega`` ->
        ``Blastoise``, ``Charizard-Mega-X`` -> ``Charizard``, ``Silvally-Ghost`` ->
        ``Silvally``. poke-env's learnset is incomplete in both directions -- some forme
        entries exist but list no moves (``rotomheat``) while others are absent entirely
        (``blastoisemega``) -- so callers union every candidate at lookup time.
        """
        c = canon_species(name)
        if not c:
            return []
        out: List[str] = []
        if c in self._species_key:
            out.append(self._species_key[c])
        for kc in self._canon_keys_by_len_desc:
            if kc != c and c.startswith(kc):
                base = self._species_key[kc]
                if base not in out:
                    out.append(base)
                break
        return out

    @lru_cache(maxsize=None)
    def legal_moves(self, species: str, gen: int,
                    methods: Optional[FrozenSet[str]] = None) -> FrozenSet[str]:
        """All moves ``species`` can hold in ``gen`` (canonical names).

        A move is legal iff it appears in the species' learnset with at least one tag of
        the form ``<gen><method>`` where method is not ``-`` (illegal marker). If
        ``methods`` is given (e.g. ``frozenset({'L','E'})``), only those acquisition
        methods count (L=level-up, E=egg, T=tutor, M=machine, S=special).
        """
        # NOTE: forme/variant species resolve to every applicable key (own + base) and
        # the learnsets are unioned, so incomplete poke-env entries do not drop legal
        # moves (see _species_key_candidates).
        keys = self._species_key_candidates(species)
        if not keys or gen not in GENS:
            return frozenset()
        out: set = set()
        prefix = str(gen)
        for key in keys:
            entry = self._learnset_raw.get(key, {}).get("learnset", {})
            for mv, tags in entry.items():
                c = canon_move(mv)
                if c is None or c not in self.move_to_id:
                    continue
                if not isinstance(tags, list):
                    continue
                for t in tags:
                    if isinstance(t, str) and t.startswith(prefix) and len(t) > 1 and t[1] != "-":
                        if methods is None or t[1] in methods:
                            out.add(c)
                            break
        return frozenset(out)

    @lru_cache(maxsize=None)
    def natural_moves(self, species: str) -> FrozenSet[str]:
        """Level-up + egg moves across gens -- the species' *natural* profile, excluding
        artificial tutors/machines that would make every species look like it can do
        everything. Used for the moveset-shape prior."""
        out: set = set()
        for g in GENS:
            out |= self.legal_moves(species, g, methods=frozenset({"L", "E"}))
        return frozenset(out)

    @lru_cache(maxsize=None)
    def moveset_shape(self, species: str) -> tuple:
        """Gen-agnostic capability summary of a species' full move potential (SHAPE_DIM).

        This is the "what could this species do" prior the observation needs so the net
        can reason about hidden/unknown moves (e.g. a Garchomp lead implies a physical
        dragon + hazard/choice profile). Computed over the union of the species' legal
        moves across gens 6-9, so it never needs the live generation.
        """
        legal = set(self.natural_moves(species))
        if not legal:
            return tuple(0.0 for _ in range(SHAPE_DIM))
        type_cov = [0.0] * _N_SHAPE_TYPES
        phys = spec = other = 0
        cap_priority = cap_recover = cap_status = cap_hazard = cap_setup = 0
        for mv in legal:
            p = self._move_props.get(mv, {})
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            ti = _TYPE_IDX.get((t or "").lower())
            if ti is not None:
                type_cov[ti] += 1.0
            cat = (p.get("category") or "").lower()
            if cat == "physical":
                phys += 1
            elif cat == "special":
                spec += 1
            else:
                other += 1
            if (p.get("priority") or 0) > 0:
                cap_priority = 1
            if p.get("drain") or p.get("heal"):
                cap_recover = 1
            if p.get("status") or p.get("volatileStatus"):
                cap_status = 1
            if mv in _HAZARD_MOVES:
                cap_hazard = 1
            if p.get("selfBoost") or p.get("boosts"):
                cap_setup = 1
        dmg = max(phys + spec, 1)
        vec: List[float] = [1.0 if x > 0 else 0.0 for x in type_cov]
        vec += [phys / dmg, spec / dmg]
        vec += [cap_priority, cap_recover, cap_status, cap_hazard, cap_setup]
        vec += [min(len(legal), 32) / 32.0]
        return tuple(vec[:SHAPE_DIM])

    # ----------------------------------------------------------------- props
    def move_props(self, name: Optional[str]) -> dict:
        c = canon_move(name)
        return self._move_props.get(c, {}) if c else {}

    def move_gimmick(self, name: Optional[str]) -> str:
        """``""`` for a plain move, ``"z"`` for a Z-move, ``"dynamax"`` for a G-Max move."""
        c = canon_move(name)
        if c is None:
            return ""
        if c in self._z_moves:
            return "z"
        p = self._move_props.get(c, {})
        if isinstance(p, dict) and p.get("isMax"):
            return "dynamax"
        return ""

    def z_base(self, name: Optional[str]) -> Optional[str]:
        c = canon_move(name)
        return self._z_to_base.get(c) if c else None

    # ----------------------------------------------------------------- mask
    def build_mask(
        self,
        *,
        gen: int,
        allowed_moves: Optional[Iterable[str]] = None,
        alive_slots: Optional[Iterable[int]] = None,
        trapped: bool = False,
        can_mega: bool = False,
        can_dynamax: bool = False,
        can_tera: bool = False,
    ) -> np.ndarray:
        """Boolean legal-action mask of shape ``(N_ACTIONS,)`` for one situation.

        ``allowed_moves`` is the set of MOVE-band actions to unmask. The caller supplies
        the tight set (self-play: the mon's real 4) or the loose prior (imitation: the
        species-legal set); this method just applies whatever it is handed.
        """
        mask = np.zeros(self.N_ACTIONS, dtype=bool)
        for nm in (allowed_moves or ()):
            i = self.move_id(nm)
            if i is not None:
                mask[i] = True
        if not trapped:
            for s in (alive_slots or ()):
                sid = self.switch_id(int(s))
                if sid is not None:
                    mask[sid] = True
        if can_mega:
            t = self.transform("mega")
            if t is not None:
                mask[t] = True
        if can_dynamax:
            t = self.transform("dynamax")
            if t is not None:
                mask[t] = True
        if can_tera:
            t = self.transform("tera")
            if t is not None:
                mask[t] = True
        return mask

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ActionSpace(N_MOVES={self.N_MOVES}, SWITCH={self.SWITCH_BASE}-"
            f"{self.TRANSFORM_BASE - 1}, TRANSFORM={self.TRANSFORM_BASE}-"
            f"{self.N_ACTIONS - 1}, N_ACTIONS={self.N_ACTIONS})"
        )


# ---------------------------------------------------------------------------
# Lazy singleton (loading the 3.9 MB learnset is not free)
# ---------------------------------------------------------------------------
_INSTANCE: Optional[ActionSpace] = None


def get_action_space() -> ActionSpace:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ActionSpace()
    return _INSTANCE
