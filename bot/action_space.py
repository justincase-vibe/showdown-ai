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
        c = canon_species(name)
        return self._species_key.get(c) if c else None

    @lru_cache(maxsize=None)
    def legal_moves(self, species: str, gen: int) -> FrozenSet[str]:
        """All moves ``species`` can hold in ``gen`` (canonical names).

        A move is legal iff it appears in the species' learnset with at least one tag of
        the form ``<gen><method>`` where method is not ``-`` (illegal marker).
        """
        key = self._species_key.get(canon_species(species))
        out: set = set()
        if key is None or gen not in GENS:
            return frozenset()
        entry = self._learnset_raw.get(key, {}).get("learnset", {})
        prefix = str(gen)
        for mv, tags in entry.items():
            c = canon_move(mv)
            if c is None or c not in self.move_to_id:
                continue
            if not isinstance(tags, list):
                continue
            for t in tags:
                if isinstance(t, str) and t.startswith(prefix) and len(t) > 1 and t[1] != "-":
                    out.add(c)
                    break
        return frozenset(out)

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
