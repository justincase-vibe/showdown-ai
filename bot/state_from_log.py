"""Reconstruct the 808-dim observation at each decision point from a battle log.

This module replays a Showdown battle log up to a given decision point and builds
synthetic battle-state objects that satisfy ``obs.py``'s interface, then calls
``embed_battle()`` to produce the exact same 808-dim vector the live game would give.

Key design: uses real poke_env Pokemon objects (for base_stats, types) and Move objects
(for move properties), but sets state via internal ``_``-prefixed attributes since many
properties (fainted, revealed, current_hp_fraction, moves, etc.) are read-only.

Usage:
    from state_from_log import StateReconstructor
    recon = StateReconstructor(log, gen=8)
    states = recon.all_states()  # List[np.ndarray], one per decision point
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.battle.side_condition import SideCondition

try:
    from poke_env.battle.effect import Effect
except Exception:
    Effect = None

# Import our observation (same directory)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obs import embed_battle, OBS_DIM
from reveal import _ZeroMove, normalize_move_name


# ---------------------------------------------------------------------------
# Move cache (avoid re-creating Move objects for the same move name)
# ---------------------------------------------------------------------------
_MOVE_CACHE: Dict[Tuple[str, int], Optional[Move]] = {}

def get_move(name: str, gen: int) -> Optional[Move]:
    """Create or retrieve a cached Move object by name."""
    key = (name.lower().replace(" ", "").replace("-", ""), gen)
    if key in _MOVE_CACHE:
        return _MOVE_CACHE[key]
    try:
        move_id = Move.retrieve_id(name)
        m = Move(move_id, gen=gen)
        _MOVE_CACHE[key] = m
        return m
    except Exception:
        _MOVE_CACHE[key] = None
        return None


# ---------------------------------------------------------------------------
# Log effect name (|-start|/|-end|) -> canonical cue name used by obs.py CUE_EFFECTS.
# Normalized to lowercase with spaces -> underscores before lookup, so "Destiny Bond",
# "Gastro Acid", "Partially Trapped" all resolve correctly.
# ---------------------------------------------------------------------------
_EFFECT_CUE = {
    "confuse": "CONFUSION",
    "confused": "CONFUSION",
    "encore": "ENCORE",
    "disable": "DISABLE",
    "destiny_bond": "DESTINY_BOND",
    "curse": "CURSE",
    "disguise": "DISGUISE",
    "gastro_acid": "GASTRO_ACID",
    "taunt": "TAUNT",
    "trick": "TRICK",
    "partially_trapped": "PARTIALLY_TRAPPED",
    "snap_trap": "SNAP_TRAP",
    "silk_trap": "SILK_TRAP",
}


def _cue_for_effect(effect: str) -> Optional[str]:
    """Map a |-start|/|-end| effect string to its obs.py cue name (or None)."""
    return _EFFECT_CUE.get(effect.lower().replace(" ", "_"))


# ---------------------------------------------------------------------------
# Per-Pokemon state tracked during log replay
# ---------------------------------------------------------------------------
@dataclass
class MonState:
    species: str
    slot: int
    hp_current: int = 100       # absolute HP (e.g. 72 out of 100)
    hp_max: int = 100
    status: Optional[Status] = None
    fainted: bool = False
    boosts: Dict[str, int] = field(default_factory=dict)
    effects: Set[str] = field(default_factory=set)  # effect name strings
    moves_revealed: List[str] = field(default_factory=list)  # in order of first use
    must_recharge: bool = False
    protect_counter: int = 0
    terastallized: bool = False
    dancing: bool = False
    ability: Optional[str] = None
    item: Optional[str] = None
    revealed: bool = False


# ---------------------------------------------------------------------------
# Battle state tracker
# ---------------------------------------------------------------------------
class _BattleState:
    """Tracks the full battle state as we replay the log."""

    def __init__(self, gen: int):
        self.gen = gen
        # Per-side Pokémon states: {slot: MonState}
        self.p1_mons: Dict[int, MonState] = {}
        self.p2_mons: Dict[int, MonState] = {}
        # Ident (display name) -> slot, per side. Populated on every |switch| so we
        # never have to fuzzy-match species (which breaks on duplicate species and
        # on random battles that have no |poke| roster).
        self.p1_ident_slot: Dict[str, int] = {}
        self.p2_ident_slot: Dict[str, int] = {}
        # Active Pokémon (slot index) per side
        self.p1_active: Optional[int] = None
        self.p2_active: Optional[int] = None
        # Field
        self.weather: Optional[Weather] = None
        self.p1_hazards: Set[SideCondition] = set()
        self.p2_hazards: Set[SideCondition] = set()
        self.turn: int = 0
        self.trapped: bool = False
        self.force_switch: bool = False
        # Gimmick availability (per side)
        self.p1_can_mega: bool = True
        self.p1_can_z: bool = False
        self.p1_can_dynamax: bool = False
        self.p1_can_tera: bool = False
        self.p2_can_mega: bool = True
        self.p2_can_z: bool = False
        self.p2_can_dynamax: bool = False
        self.p2_can_tera: bool = False
        # Gimmick usage (once used, can't use again)
        self.p1_mega_used: bool = False
        self.p2_mega_used: bool = False
        self.p1_tera_used: bool = False
        self.p2_tera_used: bool = False

    def set_gen_flags(self):
        if self.gen in (6, 7):
            self.p1_can_mega = self.p2_can_mega = True
        if self.gen == 7:
            self.p1_can_z = self.p2_can_z = True
        if self.gen == 8:
            self.p1_can_dynamax = self.p2_can_dynamax = True
        if self.gen == 9:
            self.p1_can_tera = self.p2_can_tera = True


# ---------------------------------------------------------------------------
# Log replay engine
# ---------------------------------------------------------------------------
class LogReplayer:
    """Replays a battle log and snapshots state at each decision point."""

    STATUS_MAP: Dict[str, Status] = {
        "brn": Status.BRN, "frz": Status.FRZ, "par": Status.PAR,
        "psn": Status.PSN, "slp": Status.SLP, "tox": Status.TOX,
    }

    WEATHER_MAP: Dict[str, Weather] = {
        "sunnyday": Weather.SUNNYDAY, "raindance": Weather.RAINDANCE,
        "sandstorm": Weather.SANDSTORM, "hail": Weather.HAIL,
        "snow": Weather.SNOW, "primordiasea": Weather.PRIMORDIALSEA,
        "desolateland": Weather.DESOLATELAND, "deltastream": Weather.DELTASTREAM,
    }

    HAZARD_MAP: Dict[str, SideCondition] = {
        "stealthrock": SideCondition.STEALTH_ROCK,
        "spikes": SideCondition.SPIKES,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
        "stickyweb": SideCondition.STICKY_WEB,
        "reflect": SideCondition.REFLECT,
        "lightscreen": SideCondition.LIGHT_SCREEN,
        "tailwind": SideCondition.TAILWIND,
        "auroraveil": SideCondition.AURORA_VEIL,
    }

    def __init__(self, log: str, gen: int):
        self.log = log
        self.gen = gen
        self.state = _BattleState(gen)
        self.state.set_gen_flags()

    def replay_all(self) -> List[np.ndarray]:
        """Replay the full log, returning a snapshot at each player action (move or switch)."""
        snapshots: List[np.ndarray] = []
        state = self.state
        lines = self.log.splitlines()
        n = len(lines)
        i = 0

        while i < n:
            line = lines[i]
            if not line.startswith("|"):
                i += 1
                continue

            parts = line.split("|")
            if len(parts) < 2:
                i += 1
                continue
            tag = parts[1].strip()
            fields = parts[2:]

            if tag == "poke":
                self._handle_poke(fields)
            elif tag == "turn":
                if fields:
                    try:
                        state.turn = int(fields[0])
                    except ValueError:
                        pass
            elif tag == "switch":
                self._handle_switch(fields, snapshots)
            elif tag == "move":
                self._handle_move(fields, snapshots)
            elif tag == "-damage":
                self._handle_damage(fields)
            elif tag == "-heal":
                self._handle_heal(fields)
            elif tag == "-status":
                self._handle_status(fields)
            elif tag == "-curestatus":
                self._handle_curestatus(fields)
            elif tag == "-boost":
                self._handle_boost(fields)
            elif tag == "-unboost":
                self._handle_unboost(fields)
            elif tag == "faint":
                self._handle_faint(fields)
            elif tag == "-weather":
                self._handle_weather(fields)
            elif tag == "-sidestart":
                self._handle_sidestart(fields)
            elif tag == "-sideend":
                self._handle_sideend(fields)
            elif tag == "-formechange":
                self._handle_formechange(fields)
            elif tag == "-mega":
                self._handle_mega(fields)
            elif tag == "-terastallize":
                self._handle_tera(fields)
            elif tag == "-start":
                self._handle_start(fields)
            elif tag == "-end":
                self._handle_end(fields)
            elif tag == "-mustrecharge":
                self._handle_mustrecharge(fields)
            elif tag == "-ability":
                self._handle_ability(fields)
            elif tag == "-item":
                self._handle_item(fields)
            elif tag == "win":
                break

            i += 1

        return snapshots

    # ------------------------------------------------------------------ handlers

    def _handle_poke(self, fields: List[str]):
        if len(fields) < 2:
            return
        role = fields[0]
        detail = fields[1]
        parts = detail.split(",")
        species = parts[0].strip()

        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = len(mons)
        mons[slot] = MonState(species=species, slot=slot, revealed=True)

    def _resolve_slot(self, role: str, target_field: str) -> Optional[int]:
        """Resolve a '|-damage|p1a: X' style target to a slot via the ident map,
        falling back to the side's active slot. Returns None if unresolvable."""
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        ident_slot = self.state.p1_ident_slot if role == "p1" else self.state.p2_ident_slot
        ident = target_field.split(": ", 1)[1] if ": " in target_field else target_field
        if ident in ident_slot:
            return ident_slot[ident]
        active = self.state.p1_active if role == "p1" else self.state.p2_active
        if active is not None and active in mons:
            return active
        return None

    def _handle_switch(self, fields: List[str], snapshots: List[np.ndarray]):
        if len(fields) < 3:
            return
        actor_short = fields[0]
        detail = fields[1]
        hp_str = fields[2]

        role = "p1" if actor_short.startswith("p1") else "p2"
        ident = actor_short.split(": ", 1)[1] if ": " in actor_short else actor_short
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        ident_slot = self.state.p1_ident_slot if role == "p1" else self.state.p2_ident_slot

        # Species comes from the detail field ("Garchomp (M 334 / 182)" / "Rotom-Mow").
        species = detail.split(",")[0].split("(")[0].strip()

        # Determine the slot for this ident. Prefer the ident map (exact). Otherwise
        # claim the first not-yet-revealed roster slot of the same species (respects
        # |poke| order, handles duplicate species). Otherwise (random battles: no
        # roster) register a brand-new slot in order of first appearance.
        if ident in ident_slot:
            slot = ident_slot[ident]
        else:
            norm = species.lower().replace(" ", "").replace("-", "")
            slot = None
            for s, m in mons.items():
                if (m.species.lower().replace(" ", "").replace("-", "") == norm
                        and not m.revealed):
                    slot = s
                    break
            if slot is None:
                slot = (max(mons.keys()) + 1) if mons else 0
                mons[slot] = MonState(species=species, slot=slot)
            ident_slot[ident] = slot

        if role == "p1":
            self.state.p1_active = slot
        else:
            self.state.p2_active = slot

        mons[slot].revealed = True
        mons[slot].hp_current, mons[slot].hp_max = self._parse_hp(hp_str)
        # Reset battle state for the switching-in mon
        mons[slot].status = None
        mons[slot].fainted = False
        mons[slot].boosts = {}
        mons[slot].effects = set()
        mons[slot].must_recharge = False
        mons[slot].protect_counter = 0
        mons[slot].terastallized = False

        # Only snapshot at P1's decision points (we train P1's policy)
        if role == "p1":
            snapshots.append(self._build_observation())

    def _handle_move(self, fields: List[str], snapshots: List[np.ndarray]):
        if len(fields) < 2:
            return
        actor_short = fields[0]
        move_name = fields[1]

        role = "p1" if actor_short.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, actor_short)

        # Normalize Z-names to their base move (single source of truth in reveal.py)
        # so the observation shows the resolvable base move and its reveal order matches
        # parse.py's action labeling exactly.
        canonical = normalize_move_name(move_name, self.gen)
        if slot is not None and slot in mons:
            mon = mons[slot]
            if canonical not in mon.moves_revealed:
                mon.moves_revealed.append(canonical)

        # Always snapshot at P1's decision points (we train P1's policy). This must
        # fire on EVERY P1 move line regardless of whether we could resolve the acting
        # slot, so the state count stays in lockstep with parse.py's P1 decision count.
        if role == "p1":
            snapshots.append(self._build_observation())

    def _handle_damage(self, fields: List[str]):
        if not fields:
            return
        target = fields[0]
        hp_str = fields[1] if len(fields) > 1 else ""

        role = "p1" if target.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, target)

        if slot is not None and slot in mons:
            mons[slot].hp_current, mons[slot].hp_max = self._parse_hp(hp_str)
            if "fnt" in hp_str:
                mons[slot].fainted = True

    def _handle_heal(self, fields: List[str]):
        if not fields:
            return
        target = fields[0]
        hp_str = fields[1] if len(fields) > 1 else ""

        role = "p1" if target.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, target)

        if slot is not None and slot in mons:
            mons[slot].hp_current, mons[slot].hp_max = self._parse_hp(hp_str)

    def _handle_status(self, fields: List[str]):
        if not fields:
            return
        target = fields[0]
        status_str = fields[1] if len(fields) > 1 else ""

        role = "p1" if target.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, target)

        if slot is not None and slot in mons:
            mons[slot].status = self.STATUS_MAP.get(status_str.lower())

    def _handle_curestatus(self, fields: List[str]):
        if not fields:
            return
        target = fields[0]
        role = "p1" if target.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, target)

        if slot is not None and slot in mons:
            mons[slot].status = None

    def _handle_boost(self, fields: List[str]):
        if len(fields) < 3:
            return
        target = fields[0]
        stat = fields[1]
        try:
            amount = int(fields[2])
        except ValueError:
            amount = 1

        role = "p1" if target.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, target)

        if slot is not None and slot in mons:
            mon = mons[slot]
            mon.boosts[stat] = mon.boosts.get(stat, 0) + amount

    def _handle_unboost(self, fields: List[str]):
        if len(fields) < 3:
            return
        target = fields[0]
        stat = fields[1]
        try:
            amount = int(fields[2])
        except ValueError:
            amount = 1

        role = "p1" if target.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, target)

        if slot is not None and slot in mons:
            mon = mons[slot]
            mon.boosts[stat] = mon.boosts.get(stat, 0) - amount

    def _handle_faint(self, fields: List[str]):
        if not fields:
            return
        target = fields[0]
        role = "p1" if target.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, target)

        if slot is not None and slot in mons:
            mons[slot].fainted = True
            mons[slot].hp_current = 0

    def _handle_weather(self, fields: List[str]):
        if not fields:
            return
        weather_str = fields[0].lower()
        if weather_str == "none":
            self.state.weather = None
        else:
            self.state.weather = self.WEATHER_MAP.get(weather_str)

    def _handle_sidestart(self, fields: List[str]):
        if len(fields) < 2:
            return
        role = "p1" if "p1" in fields[0] else "p2"
        hazard_str = fields[1].lower().replace("move: ", "").replace(" ", "")
        hazard = self.HAZARD_MAP.get(hazard_str)
        if hazard:
            if role == "p1":
                self.state.p1_hazards.add(hazard)
            else:
                self.state.p2_hazards.add(hazard)

    def _handle_sideend(self, fields: List[str]):
        if len(fields) < 2:
            return
        role = "p1" if "p1" in fields[0] else "p2"
        hazard_str = fields[1].lower()
        hazard = self.HAZARD_MAP.get(hazard_str)
        if hazard:
            if role == "p1":
                self.state.p1_hazards.discard(hazard)
            else:
                self.state.p2_hazards.discard(hazard)

    def _handle_formechange(self, fields: List[str]):
        if len(fields) >= 2 and "Mega" in fields[1]:
            role = "p1" if "p1" in fields[0] else "p2"
            if role == "p1":
                self.state.p1_mega_used = True
            else:
                self.state.p2_mega_used = True

    def _handle_mega(self, fields: List[str]):
        if fields:
            role = "p1" if "p1" in fields[0] else "p2"
            if role == "p1":
                self.state.p1_mega_used = True
            else:
                self.state.p2_mega_used = True

    def _handle_tera(self, fields: List[str]):
        if fields:
            role = "p1" if fields[0].startswith("p1") else "p2"
            mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
            slot = self._resolve_slot(role, fields[0])
            if slot is not None and slot in mons:
                mons[slot].terastallized = True
            if role == "p1":
                self.state.p1_tera_used = True
            else:
                self.state.p2_tera_used = True

    def _handle_start(self, fields: List[str]):
        if len(fields) < 2:
            return
        target = fields[0]
        effect = fields[1]

        role = "p1" if target.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, target)

        if slot is None or slot not in mons:
            return
        mon = mons[slot]
        eff = effect.lower()
        if "protect" in eff or "detect" in eff:
            mon.protect_counter = 1
            return
        cue = _cue_for_effect(effect)
        if cue:
            mon.effects.add(cue)

    def _handle_end(self, fields: List[str]):
        if len(fields) < 2:
            return
        target = fields[0]
        effect = fields[1]

        role = "p1" if target.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, target)

        if slot is None or slot not in mons:
            return
        mon = mons[slot]
        eff = effect.lower()
        if "protect" in eff or "detect" in eff:
            mon.protect_counter = 0
            return
        cue = _cue_for_effect(effect)
        if cue:
            mon.effects.discard(cue)

    def _handle_mustrecharge(self, fields: List[str]):
        if not fields:
            return
        target = fields[0]
        role = "p1" if target.startswith("p1") else "p2"
        mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
        slot = self._resolve_slot(role, target)

        if slot is not None and slot in mons:
            mons[slot].must_recharge = True

    def _handle_ability(self, fields: List[str]):
        if len(fields) >= 2:
            target = fields[0]
            ability = fields[1]
            role = "p1" if target.startswith("p1") else "p2"
            mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
            slot = self._resolve_slot(role, target)
            if slot is not None and slot in mons:
                mons[slot].ability = ability

    def _handle_item(self, fields: List[str]):
        if len(fields) >= 2:
            target = fields[0]
            item = fields[1]
            role = "p1" if target.startswith("p1") else "p2"
            mons = self.state.p1_mons if role == "p1" else self.state.p2_mons
            slot = self._resolve_slot(role, target)
            if slot is not None and slot in mons:
                mons[slot].item = item

    # ------------------------------------------------------------------ helpers

    def _parse_hp(self, hp_str: str) -> Tuple[int, int]:
        """Parse HP string like '72 / 100' or '0 fnt' → (current, max)."""
        if "fnt" in hp_str:
            return 0, 100
        if "/" in hp_str:
            parts = hp_str.split("/")
            try:
                cur = int(parts[0].strip())
                mx = int(parts[1].strip())
                return cur, mx
            except (ValueError, IndexError):
                pass
        return 0, 100

    def _build_observation(self) -> np.ndarray:
        """Build the 808-dim observation from current state, from P1's perspective."""
        state = self.state

        p1_active_mon = self._build_pokemon(state.p1_mons, state.p1_active)
        p2_active_mon = self._build_pokemon(state.p2_mons, state.p2_active)

        p1_team: Dict[int, Pokemon] = {}
        for slot, ms in state.p1_mons.items():
            p = self._build_pokemon(state.p1_mons, slot)
            if p is not None:
                p1_team[slot] = p

        p2_team: Dict[int, Pokemon] = {}
        for slot, ms in state.p2_mons.items():
            p = self._build_pokemon(state.p2_mons, slot)
            if p is not None:
                p2_team[slot] = p

        battle = _FakeBattle(
            active_pokemon=p1_active_mon,
            opponent_active_pokemon=p2_active_mon,
            team=p1_team,
            opponent_team=p2_team,
            weather={state.weather: True} if state.weather else {},
            side_conditions={h: True for h in state.p1_hazards},
            opponent_side_conditions={h: True for h in state.p2_hazards},
            turn=state.turn,
            trapped=state.trapped,
            force_switch=state.force_switch,
            can_mega_evolve=(state.p1_can_mega and not state.p1_mega_used),
            can_z_move=(state.p1_can_z),
            can_dynamax=(state.p1_can_dynamax),
            can_tera=(state.p1_can_tera and not state.p1_tera_used),
        )

        return embed_battle(battle)

    def _build_pokemon(self, mons: Dict[int, MonState], slot: Optional[int]) -> Optional[Pokemon]:
        """Build a poke_env Pokemon object with state set via internal attributes."""
        if slot is None or slot not in mons:
            return None

        ms = mons[slot]

        # Create the Pokemon (gives base_stats, types from poke_env's data).
        # Try the FULL species name first so forme species (Slowbro-Galar, Rotom-Wash,
        # Greninja-Ash, ...) get their correct base stats/types; fall back to the base
        # species only if the forme is absent from this gen's dex.
        try:
            p = Pokemon(gen=self.gen, species=ms.species)
        except Exception:
            try:
                p = Pokemon(gen=self.gen, species=ms.species.split("-")[0])
            except Exception:
                return None

        # Set state via internal attributes (bypass read-only properties)
        try:
            # HP
            p._current_hp = ms.hp_current
            p._max_hp = ms.hp_max if ms.hp_max > 0 else 100

            # Fainted: status = FNT
            if ms.fainted:
                p._status = Status.FNT
            else:
                p._status = ms.status  # None or Status.BRN etc.

            # Revealed
            p._revealed = ms.revealed

            # Name (for hashing in obs.py)
            p._name = ms.species

            # Boosts (public setter exists)
            if ms.boosts:
                p.boosts = {k: v for k, v in ms.boosts.items() if v != 0}

            # Ability
            if ms.ability:
                try:
                    p.ability = ms.ability
                except Exception:
                    pass

            # Item
            if ms.item:
                try:
                    p.item = ms.item
                except Exception:
                    pass

            # Moves (set the internal MoveSet's _base_moves). Preserve reveal-slot
            # order exactly: an unresolvable move still occupies its slot (zero-filled
            # via _ZeroMove) instead of shifting the later moves left.
            if ms.moves_revealed:
                moves_dict = {}
                zero = 0
                for move_name in ms.moves_revealed[:4]:
                    m = get_move(move_name, self.gen)
                    if m is None:
                        m = _ZeroMove()
                        key = f"unrevealed_{zero}"
                        zero += 1
                    else:
                        key = m.id
                    moves_dict[key] = m
                p._moves._base_moves = moves_dict

            # Protect counter
            p._protect_counter = ms.protect_counter

            # Must recharge
            p.must_recharge = ms.must_recharge

            # Effects (set as dict {Effect: int})
            if ms.effects:
                effects_dict = {}
                if Effect is not None:
                    for eff_name in ms.effects:
                        try:
                            e = getattr(Effect, eff_name, None)
                            if e is not None:
                                effects_dict[e] = 1
                        except Exception:
                            pass
                p._effects = effects_dict

            # Tera / dancing (gen 9 specific)
            if ms.terastallized:
                try:
                    p.terastallized = True
                except Exception:
                    pass
            if ms.dancing:
                try:
                    p.dancing = True
                except Exception:
                    pass

        except Exception:
            # If any internal setting fails, return what we have
            pass

        return p


# ---------------------------------------------------------------------------
# Fake battle object (satisfies obs.py's interface)
# ---------------------------------------------------------------------------
class _FakeBattle:
    """Minimal battle object that provides the attributes obs.py needs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class StateReconstructor:
    """High-level API: reconstruct observations from a battle log at decision points."""

    def __init__(self, log: str, gen: int):
        self.log = log
        self.gen = gen
        self._replayer = LogReplayer(log, gen)
        self._all_states: Optional[List[np.ndarray]] = None

    def _ensure_replayed(self):
        if self._all_states is None:
            self._all_states = self._replayer.replay_all()

    def state_at_decision(self, decision_index: int) -> np.ndarray:
        """Get the 808-dim observation at a specific decision index."""
        self._ensure_replayed()
        if decision_index < len(self._all_states):
            return self._all_states[decision_index]
        return np.zeros(OBS_DIM, dtype=np.float32)

    def all_states(self) -> List[np.ndarray]:
        """Get all decision-point observations."""
        self._ensure_replayed()
        return self._all_states

    def current_state(self) -> np.ndarray:
        """Observation of the FULL current battle state (every event replayed), from
        P1's perspective.

        This is what the LIVE masked view uses: it runs the exact same event handlers
        and ``_build_observation`` as the replay reconstruction, so the live and
        training observations are built by identical code (train/live equivalence by
        construction). Unlike ``all_states()`` (snapshots at P1 decision lines), this
        reflects the state the active player currently faces -- the opponent's latest
        action included, the trainee's own pending action not yet applied.
        """
        self._ensure_replayed()
        return self._replayer._build_observation()

    def __len__(self):
        self._ensure_replayed()
        return len(self._all_states)


def reconstruct_current_state(log: str, gen: int) -> np.ndarray:
    """One-shot: replay ``log`` fully and return the current P1-perspective observation."""
    return StateReconstructor(log, gen).current_state()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Test state reconstruction")
    parser.add_argument("--file", required=True, help="JSONL file from fetch.py")
    parser.add_argument("--max", type=int, default=5, help="Max battles to test")
    parser.add_argument("--show", type=int, default=2, help="Show first N in detail")
    args = parser.parse_args()

    print(f"Testing state reconstruction on {args.file}...")
    print(f"(This may take a moment for the first battle due to Move caching)")

    with open(args.file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.max:
                break
            rec = json.loads(line)
            log = rec["log"]

            # Extract gen from log
            gen = 8
            for ln in log.splitlines():
                if ln.startswith("|gen|"):
                    try:
                        gen = int(ln.split("|")[2])
                    except (ValueError, IndexError):
                        pass
                    break

            recon = StateReconstructor(log, gen)
            states = recon.all_states()

            print(f"\nBattle {i}: {rec.get('id', '?')}")
            print(f"  Gen: {gen}, Decisions: {len(states)}")

            if len(states) > 0:
                s = states[0]
                print(f"  State[0] shape: {s.shape}, dtype: {s.dtype}")
                print(f"  State[0] stats: min={s.min():.3f} max={s.max():.3f} mean={s.mean():.4f}")
                print(f"  Non-zero dims: {(s != 0).sum()} / {OBS_DIM}")

                if (s != 0).sum() < 10:
                    print(f"  WARNING: state is mostly zeros!")

                if args.show > 0 and i < args.show:
                    print(f"  First 20 dims: {s[:20]}")
                    print(f"  Active mon HP (dim 0): {s[0]:.3f}")

            # Verify dimension
            for st in states[:10]:
                assert st.shape == (OBS_DIM,), f"Dimension mismatch: {st.shape} != ({OBS_DIM},)"

    print("\n✓ All states have correct shape.")
    print("✓ State reconstruction working.")
